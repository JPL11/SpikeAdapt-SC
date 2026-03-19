"""SpikeAdapt-SC v4 — Architecture improvements from ablation + literature.

Combines best SNN-native techniques from ablation study:
  A) Learnable surrogate slope
  B) LIF with learnable leak β
  C) BNTT OR MPBN (configurable)
  E) Spike-rate regularization
  F) STDP temporal loss
  (D skipped — membrane init showed no improvement)

Plus literature-inspired additions:
  KD) Knowledge distillation (ANN teacher → SNN student)
  CG) Channel gating (SNR-adaptive bottleneck width)

Usage:
  python train/train_aid_v4.py --exp v4A        # SNN combo only
  python train/train_aid_v4.py --exp v4B        # + KD
  python train/train_aid_v4.py --exp v4C        # + KD + channel gating
  python train/train_aid_v4.py --exp v4D        # + KD + MPBN (replaces BNTT)
"""

import os, sys, random, json, math, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from train_aid_v2 import (
    AIDDataset, ResNet50Front, ResNet50Back,
    BSC_Channel, ChannelConditionedScorer, LearnedBlockMask,
    sample_noise
)

# ############################################################################
# ARGS
# ############################################################################
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--exp', type=str, default='v4B',
                   help='v4A (SNN combo), v4B (+KD), v4C (+KD+CG), v4D (+KD+MPBN)')
    p.add_argument('--epochs_s3', type=int, default=40)
    p.add_argument('--target_rate', type=float, default=0.75)
    p.add_argument('--C2', type=int, default=36)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--lambda_spike', type=float, default=0.1)
    p.add_argument('--lambda_stdp', type=float, default=0.05)
    p.add_argument('--lambda_kd', type=float, default=0.5)
    p.add_argument('--kd_temp', type=float, default=4.0)
    p.add_argument('--target_fr', type=float, default=0.25)
    return p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_STEPS = 8

# ############################################################################
# ENHANCED SNN MODULES (from snn_native, with additions)
# ############################################################################

class SpikeFunction_Learnable(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane, threshold, slope):
        if isinstance(threshold, (int, float)):
            threshold = torch.tensor(float(threshold), device=membrane.device)
        ctx.save_for_backward(membrane, threshold, slope)
        ctx.th_needs_grad = isinstance(threshold, torch.Tensor) and threshold.requires_grad
        return (membrane > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        membrane, threshold, slope = ctx.saved_tensors
        s = slope.clamp(min=1.0, max=100.0)
        sig = torch.sigmoid(s * (membrane - threshold))
        sg = sig * (1 - sig) * s
        grad_mem = grad_output * sg
        grad_th = -(grad_output * sg).sum() if ctx.th_needs_grad else None
        grad_slope = (grad_output * sig * (1 - sig) * (membrane - threshold)).sum()
        return grad_mem, grad_th, grad_slope


class LIFNeuron(nn.Module):
    """LIF with learnable leak and slope (Techniques A+B)."""
    def __init__(self, C, th=1.0):
        super().__init__()
        self.threshold = th
        self.beta_raw = nn.Parameter(torch.ones(1, C, 1, 1) * 2.2)  # sigmoid(2.2) ≈ 0.9
        self.slope = nn.Parameter(torch.tensor(10.0))

    def forward(self, x, mem=None):
        if mem is None:
            mem = torch.zeros_like(x)
        beta = torch.sigmoid(self.beta_raw)
        mem = beta * mem + x
        sp = SpikeFunction_Learnable.apply(mem, self.threshold, self.slope)
        return sp, mem - sp * self.threshold


class LIFNeuron_LTh(nn.Module):
    """LIF with learnable threshold + leak + slope."""
    def __init__(self, C, th=1.0):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(float(th)))
        self.beta_raw = nn.Parameter(torch.ones(1, C, 1, 1) * 2.2)
        self.slope = nn.Parameter(torch.tensor(10.0))

    def forward(self, x, mem=None):
        if mem is None:
            mem = torch.zeros_like(x)
        beta = torch.sigmoid(self.beta_raw)
        mem = beta * mem + x
        sp = SpikeFunction_Learnable.apply(mem, self.threshold, self.slope)
        return sp, mem - sp * self.threshold


class BNTT(nn.Module):
    """Batch Normalization Through Time (Technique C)."""
    def __init__(self, C, T):
        super().__init__()
        self.bns = nn.ModuleList([nn.BatchNorm2d(C) for _ in range(T)])
        self.T = T

    def forward(self, x, t):
        return self.bns[min(t, self.T - 1)](x)


class MPBN(nn.Module):
    """Membrane Potential Batch Normalization (ICCV 2023).

    Normalizes membrane potential BEFORE the spike function,
    giving adaptive per-channel effective thresholds.
    """
    def __init__(self, C, T):
        super().__init__()
        self.bns = nn.ModuleList([nn.BatchNorm2d(C, affine=True) for _ in range(T)])
        self.T = T

    def forward(self, x, t):
        return self.bns[min(t, self.T - 1)](x)


class ChannelGate(nn.Module):
    """SNR-adaptive channel gating (GLOBECOM 2024 inspired).

    Learns to mute uninformative bottleneck channels under good SNR.
    """
    def __init__(self, C2=36, hidden=32):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(True),
            nn.Linear(hidden, C2), nn.Sigmoid()
        )

    def forward(self, spikes, channel_estimate):
        """spikes: (B, C2, H, W), channel_estimate: scalar or (B,1)"""
        B = spikes.size(0)
        if isinstance(channel_estimate, (int, float)):
            ch_in = torch.full((B, 1), channel_estimate, device=spikes.device)
        else:
            ch_in = channel_estimate.view(B, 1)
        g = self.gate(ch_in)  # (B, C2)
        return spikes * g.unsqueeze(-1).unsqueeze(-1), g.mean()


# ############################################################################
# V4 ENCODER & DECODER
# ############################################################################

class EncoderV4(nn.Module):
    def __init__(self, C_in=1024, C1=256, C2=36, T=8, use_mpbn=False):
        super().__init__()
        self.T = T
        self.use_mpbn = use_mpbn

        self.conv1 = nn.Conv2d(C_in, C1, 3, 1, 1)
        self.conv2 = nn.Conv2d(C1, C2, 3, 1, 1)

        # Pre-spike BN: standard or BNTT
        if not use_mpbn:
            self.bn1 = BNTT(C1, T)
            self.bn2 = BNTT(C2, T)
        else:
            self.bn1 = nn.BatchNorm2d(C1)
            self.bn2 = nn.BatchNorm2d(C2)

        # Post-conv, pre-spike MPBN (only if use_mpbn)
        if use_mpbn:
            self.mpbn1 = MPBN(C1, T)
            self.mpbn2 = MPBN(C2, T)

        self.if1 = LIFNeuron(C1)
        self.if2 = LIFNeuron(C2)

    def forward(self, F, m1=None, m2=None, t=0):
        if self.use_mpbn:
            h1 = self.bn1(self.conv1(F))
        else:
            h1 = self.bn1(self.conv1(F), t)

        if self.use_mpbn:
            # MPBN: normalize membrane potential before spiking
            if m1 is None:
                m1 = torch.zeros_like(h1)
            beta1 = torch.sigmoid(self.if1.beta_raw)
            m1 = beta1 * m1 + h1
            m1 = self.mpbn1(m1, t)
            s1 = SpikeFunction_Learnable.apply(m1, self.if1.threshold, self.if1.slope)
            m1 = m1 - s1 * self.if1.threshold
        else:
            s1, m1 = self.if1(h1, m1)

        if self.use_mpbn:
            h2 = self.bn2(self.conv2(s1))
        else:
            h2 = self.bn2(self.conv2(s1), t)

        if self.use_mpbn:
            if m2 is None:
                m2 = torch.zeros_like(h2)
            beta2 = torch.sigmoid(self.if2.beta_raw)
            m2 = beta2 * m2 + h2
            m2 = self.mpbn2(m2, t)
            s2 = SpikeFunction_Learnable.apply(m2, self.if2.threshold, self.if2.slope)
            m2 = m2 - s2 * self.if2.threshold
        else:
            s2, m2 = self.if2(h2, m2)

        return s1, s2, m1, m2


class DecoderV4(nn.Module):
    def __init__(self, C_out=1024, C1=256, C2=36, T=8, use_mpbn=False):
        super().__init__()
        self.T = T
        self.use_mpbn = use_mpbn

        self.conv3 = nn.Conv2d(C2, C1, 3, 1, 1)
        self.conv4 = nn.Conv2d(C1, C_out, 3, 1, 1)

        if not use_mpbn:
            self.bn3 = BNTT(C1, T)
            self.bn4 = BNTT(C_out, T)
        else:
            self.bn3 = nn.BatchNorm2d(C1)
            self.bn4 = nn.BatchNorm2d(C_out)

        if use_mpbn:
            self.mpbn3 = MPBN(C1, T)
            self.mpbn4 = MPBN(C_out, T)

        self.if3 = LIFNeuron(C1)
        self.ihf = LIFNeuron_LTh(C_out)

        self.converter_fc = nn.Linear(2 * T, 2 * T)

    def forward(self, recv_all, mask):
        T_use = len(recv_all)
        m3, m4 = None, None
        Fs, Fm = [], []
        for t in range(T_use):
            inp = recv_all[t] * mask

            if self.use_mpbn:
                h3 = self.bn3(self.conv3(inp))
                if m3 is None:
                    m3 = torch.zeros_like(h3)
                beta3 = torch.sigmoid(self.if3.beta_raw)
                m3 = beta3 * m3 + h3
                m3 = self.mpbn3(m3, t)
                s3 = SpikeFunction_Learnable.apply(m3, self.if3.threshold, self.if3.slope)
                m3 = m3 - s3 * self.if3.threshold
            else:
                h3 = self.bn3(self.conv3(inp), t)
                s3, m3 = self.if3(h3, m3)

            if self.use_mpbn:
                h4 = self.bn4(self.conv4(s3))
                if m4 is None:
                    m4 = torch.zeros_like(h4)
                beta4 = torch.sigmoid(self.ihf.beta_raw)
                m4 = beta4 * m4 + h4
                m4 = self.mpbn4(m4, t)
                sp = SpikeFunction_Learnable.apply(m4, self.ihf.threshold, self.ihf.slope)
                m4 = m4 - sp * self.ihf.threshold
            else:
                h4 = self.bn4(self.conv4(s3), t)
                sp, m4 = self.ihf(h4, m4)

            Fs.append(sp); Fm.append(m4.clone())

        while len(Fs) < self.T:
            Fs.append(torch.zeros_like(Fs[0]))
            Fm.append(torch.zeros_like(Fm[0]))

        il = []
        for t in range(self.T):
            il.append(Fs[t]); il.append(Fm[t])
        x = torch.stack(il, 1).permute(0, 2, 3, 4, 1)
        return (x * torch.sigmoid(self.converter_fc(x))).sum(-1)


class SpikeAdaptSC_v4(nn.Module):
    def __init__(self, C_in=1024, C1=256, C2=36, T=8,
                 target_rate=0.75, grid_size=14,
                 use_mpbn=False, use_channel_gate=False):
        super().__init__()
        self.T = T; self.C2 = C2; self.grid_size = grid_size

        self.encoder = EncoderV4(C_in, C1, C2, T, use_mpbn)
        self.scorer = ChannelConditionedScorer(C_spike=C2, hidden=32)
        self.block_mask = LearnedBlockMask(target_rate, 0.5)
        self.decoder = DecoderV4(C_in, C1, C2, T, use_mpbn)
        self.channel = BSC_Channel()

        self.use_channel_gate = use_channel_gate
        if use_channel_gate:
            self.ch_gate = ChannelGate(C2, hidden=32)

    def forward(self, feat, noise_param=0.0, target_rate_override=None):
        all_S2, m1, m2 = [], None, None

        for t in range(self.T):
            _, s2, m1, m2 = self.encoder(feat, m1, m2, t=t)
            all_S2.append(s2)

        importance = self.scorer(all_S2, noise_param)

        if target_rate_override is not None:
            old = self.block_mask.target_rate
            self.block_mask.target_rate = target_rate_override
            mask, tx = self.block_mask(importance, training=False)
            self.block_mask.target_rate = old
        else:
            mask, tx = self.block_mask(importance, training=self.training)

        # Channel gating (optional)
        gate_rate = 1.0
        if self.use_channel_gate:
            gated_S2 = []
            for t in range(self.T):
                gs, gr = self.ch_gate(self.block_mask.apply_mask(all_S2[t], mask), noise_param)
                gated_S2.append(gs)
                gate_rate = gr.item()
            recv = [self.channel(gated_S2[t], noise_param) for t in range(self.T)]
        else:
            recv = [self.channel(self.block_mask.apply_mask(all_S2[t], mask), noise_param)
                    for t in range(self.T)]

        Fp = self.decoder(recv, mask)

        return Fp, {
            'tx_rate': tx.item(),
            'mask': mask,
            'importance': importance,
            'all_S2': all_S2,
            'gate_rate': gate_rate,
        }


# ############################################################################
# LOSS HELPERS
# ############################################################################

def spike_rate_loss(all_S2, target_fr=0.25):
    fr = torch.stack(all_S2).float().mean()
    return (fr - target_fr) ** 2

def stdp_temporal_loss(all_S2):
    if len(all_S2) < 2:
        return torch.tensor(0.0, device=all_S2[0].device)
    corr = 0
    for t in range(len(all_S2) - 1):
        corr = corr + (all_S2[t].float() * all_S2[t+1].float()).mean()
    return -corr / (len(all_S2) - 1)

def knowledge_distillation_loss(logits_snn, logits_ann, temperature=4.0):
    """KL divergence between teacher (ANN) and student (SNN) soft logits."""
    p_ann = F.log_softmax(logits_ann / temperature, dim=1)
    q_snn = F.softmax(logits_snn / temperature, dim=1)
    return F.kl_div(p_ann, q_snn, reduction='batchmean') * (temperature ** 2)


# ############################################################################
# WEIGHT TRANSFER
# ############################################################################

def transfer_weights(model_new, snap_dir, device):
    s2f = sorted([f for f in os.listdir(snap_dir) if f.startswith("v2_s2_")])
    if not s2f:
        print("  ⚠ No S2 checkpoint found")
        return

    old_state = torch.load(os.path.join(snap_dir, s2f[-1]), map_location=device)
    new_state = model_new.state_dict()
    transferred = 0

    for old_key, old_val in old_state.items():
        # Direct conv/fc/scorer/mask transfer
        for prefix in ['encoder.conv1.', 'encoder.conv2.', 'decoder.conv3.',
                       'decoder.conv4.', 'decoder.converter_fc.', 'scorer.', 'block_mask.']:
            if old_key.startswith(prefix):
                if old_key in new_state and new_state[old_key].shape == old_val.shape:
                    new_state[old_key] = old_val
                    transferred += 1
                break

        # Handle BN → BNTT transfer
        for bn_name in ['encoder.bn1.', 'encoder.bn2.', 'decoder.bn3.', 'decoder.bn4.']:
            if old_key.startswith(bn_name):
                suffix = old_key[len(bn_name):]
                if old_key in new_state and new_state[old_key].shape == old_val.shape:
                    new_state[old_key] = old_val
                    transferred += 1
                else:
                    for t in range(T_STEPS):
                        bntt_key = f"{bn_name}bns.{t}.{suffix}"
                        if bntt_key in new_state and new_state[bntt_key].shape == old_val.shape:
                            new_state[bntt_key] = old_val
                            transferred += 1
                break

    model_new.load_state_dict(new_state, strict=False)
    print(f"  ✓ Transferred {transferred} params from S2")


# ############################################################################
# MAIN
# ############################################################################

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    print(f"Device: {device}")

    exp = args.exp.lower()
    use_kd = exp in ('v4b', 'v4c', 'v4d')
    use_cg = exp == 'v4c'
    use_mpbn = exp == 'v4d'

    exp_label = args.exp.upper()
    print(f"Experiment: {exp_label}")
    print(f"  KD={use_kd}, ChannelGate={use_cg}, MPBN={use_mpbn}")
    print(f"  SNN: LIF+LearnableSlope+BNTT/MPBN+SpikeReg+STDP")

    SNAP_V1 = "./snapshots_aid/"
    SNAP_V2 = f"./snapshots_aid_v2_seed{args.seed}/"
    SNAP_OUT = f"./snapshots_v4_{exp_label}_seed{args.seed}/"
    os.makedirs(SNAP_OUT, exist_ok=True)

    # Load backbone
    front = ResNet50Front(grid_size=14).to(device)
    back = ResNet50Back(30).to(device)
    bb = torch.load(os.path.join(SNAP_V1, "backbone_best.pth"), map_location=device)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))}, strict=False)
    front.eval()

    # Load back from v2 S3
    s3f = sorted([f for f in os.listdir(SNAP_V2) if f.startswith("v2_s3_")])
    if s3f:
        ck = torch.load(os.path.join(SNAP_V2, s3f[-1]), map_location=device)
        back.load_state_dict(ck['back'])
        print(f"Loaded back: {s3f[-1]}")

    # Create v4 model
    model = SpikeAdaptSC_v4(
        C_in=1024, C1=256, C2=args.C2, T=T_STEPS,
        target_rate=args.target_rate, grid_size=14,
        use_mpbn=use_mpbn, use_channel_gate=use_cg
    ).to(device)

    transfer_weights(model, SNAP_V2, device)

    # Data
    tf_train = T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
                           T.RandomRotation(15), T.ColorJitter(0.2, 0.2, 0.2, 0.1),
                           T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))])
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])
    train_ds = AIDDataset("./data", tf_train, split='train', seed=args.seed)
    test_ds = AIDDataset("./data", tf_test, split='test', seed=args.seed)
    train_loader = DataLoader(train_ds, 32, True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, 64, False, num_workers=4)

    criterion = nn.CrossEntropyLoss()

    def evaluate_v4(front, model, back, loader, noise_param=0.0):
        model.eval(); back.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                feat = front(imgs)
                Fp, _ = model(feat, noise_param=noise_param)
                preds = back(Fp).argmax(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
        return 100. * correct / total

    acc_init = evaluate_v4(front, model, back, test_loader)
    print(f"\n  Initial accuracy: {acc_init:.2f}%")

    # --- S3: Joint fine-tuning ---
    print(f"\n{'='*60}")
    print(f"S3 TRAINING — V4 [{exp_label}]")
    print(f"{'='*60}")

    for p in back.parameters():
        p.requires_grad = True

    # Separate param groups: higher lr for slope params
    slope_params = [p for n, p in model.named_parameters() if 'slope' in n]
    other_params = [p for n, p in model.named_parameters() if 'slope' not in n]
    params = [
        {'params': other_params, 'lr': args.lr},
        {'params': slope_params, 'lr': args.lr * 10},  # 10x lr for slopes
        {'params': back.parameters(), 'lr': args.lr},
    ]
    opt = optim.Adam(params, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs_s3, eta_min=1e-7)

    best_acc = 0.0
    history = []

    for ep in range(args.epochs_s3):
        model.train(); back.train()
        pbar = tqdm(train_loader, desc=f"S3 E{ep+1}/{args.epochs_s3}")
        opt.zero_grad()
        ep_loss, ep_fr, n_steps = 0, 0, 0

        for step, (img, lab) in enumerate(pbar):
            img, lab = img.to(device), lab.to(device)
            noise = sample_noise('bsc')
            with torch.no_grad():
                feat = front(img)

            Fp, stats = model(feat, noise_param=noise)
            logits_snn = back(Fp)

            # Main loss
            loss = criterion(logits_snn, lab) + 2.0 * (stats['tx_rate'] - args.target_rate) ** 2

            # Knowledge Distillation
            if use_kd:
                with torch.no_grad():
                    logits_ann = back(feat)  # ANN direct path (teacher)
                loss = loss + args.lambda_kd * knowledge_distillation_loss(
                    logits_snn, logits_ann, args.kd_temp)

            # Spike-rate regularization (E)
            loss = loss + args.lambda_spike * spike_rate_loss(stats['all_S2'], args.target_fr)

            # STDP temporal loss (F)
            loss = loss + args.lambda_stdp * stdp_temporal_loss(stats['all_S2'])

            loss = loss / 2  # gradient accumulation
            loss.backward()

            if (step + 1) % 2 == 0:
                all_params = [p for g in opt.param_groups for p in g['params']]
                nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                opt.step(); opt.zero_grad()

            with torch.no_grad():
                fr = torch.stack(stats['all_S2']).float().mean().item()
                ep_fr += fr; ep_loss += loss.item(); n_steps += 1

            pbar.set_postfix({
                'L': f'{loss.item():.3f}',
                'tx': f'{stats["tx_rate"]:.2f}',
                'fr': f'{fr:.3f}'
            })

        sched.step()

        # Evaluate every 5 epochs
        if (ep + 1) % 5 == 0 or ep == 0:
            acc = evaluate_v4(front, model, back, test_loader)
            avg_fr = ep_fr / max(n_steps, 1)

            # Get slope and beta info
            info = ""
            for name, p in model.named_parameters():
                if 'beta_raw' in name:
                    info += f", β={torch.sigmoid(p).mean().item():.3f}"
                    break
            for name, p in model.named_parameters():
                if 'slope' in name and 'beta' not in name:
                    info += f", slope={p.item():.1f}"
                    break

            print(f"  S3 E{ep+1}: {acc:.2f}%, FR={avg_fr:.3f}{info}")

            history.append({
                'epoch': ep + 1, 'accuracy': acc,
                'firing_rate': avg_fr, 'loss': ep_loss / max(n_steps, 1)
            })

            if acc > best_acc:
                best_acc = acc
                torch.save({'model': model.state_dict(), 'back': back.state_dict()},
                           os.path.join(SNAP_OUT, f"v4_{exp_label}_{acc:.2f}.pth"))
                print(f"  ✓ Best: {best_acc:.2f}%")

    # Final eval
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION — V4 [{exp_label}]")
    print(f"{'='*60}")

    best_f = sorted([f for f in os.listdir(SNAP_OUT) if f.startswith(f"v4_{exp_label}_")])
    if best_f:
        ck = torch.load(os.path.join(SNAP_OUT, best_f[-1]), map_location=device)
        model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])

    acc_clean = evaluate_v4(front, model, back, test_loader, noise_param=0.0)
    print(f"  Clean: {acc_clean:.2f}%")

    ber_results = {'clean': acc_clean}
    for ber in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        acc_n = evaluate_v4(front, model, back, test_loader, noise_param=ber)
        ber_results[f'ber_{ber:.2f}'] = acc_n
        print(f"  BER={ber:.2f}: {acc_n:.2f}%")

    results = {
        'experiment': exp_label,
        'version': 'v4',
        'seed': args.seed,
        'best_accuracy': best_acc,
        'original_v2': 96.35,
        'delta_v2': best_acc - 96.35,
        'ber_results': ber_results,
        'history': history,
        'config': {
            'kd': use_kd, 'channel_gate': use_cg, 'mpbn': use_mpbn,
            'snn_native': 'A+B+C+E+F (skip D)',
            'lr': args.lr, 'epochs': args.epochs_s3,
            'lambda_kd': args.lambda_kd if use_kd else None,
            'lambda_spike': args.lambda_spike,
            'lambda_stdp': args.lambda_stdp,
        }
    }
    with open(os.path.join(SNAP_OUT, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ V4 [{exp_label}]: {best_acc:.2f}% (Δ = {best_acc - 96.35:+.2f}pp from original)")
