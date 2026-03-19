"""SpikeAdapt-SC v2 — SNN-Native Enhancements for S3.

Implements 6 SNN-native techniques as ablatable flags:
  A) Learnable surrogate gradient slope (per-layer)
  B) LIF neurons with learnable leak factor β
  C) Batch Normalization Through Time (BNTT)
  D) Membrane potential initialization from features
  E) Spike-rate regularization
  F) STDP-inspired temporal correlation loss

Usage:
  python train/train_aid_v2_snn_native.py --exp A      # single technique
  python train/train_aid_v2_snn_native.py --exp ALL    # all combined
  python train/train_aid_v2_snn_native.py --exp ABCD   # any combo
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
    evaluate, sample_noise
)

# ############################################################################
# ARGS
# ############################################################################
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--exp', type=str, default='ALL',
                   help='Experiment: A, B, C, D, E, F, or any combo (e.g. ABCD, ALL)')
    p.add_argument('--epochs_s3', type=int, default=30)
    p.add_argument('--target_rate', type=float, default=0.75)
    p.add_argument('--C2', type=int, default=36)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--lambda_spike', type=float, default=0.1,
                   help='Spike-rate regularization weight')
    p.add_argument('--lambda_stdp', type=float, default=0.05,
                   help='STDP temporal correlation weight')
    p.add_argument('--target_fr', type=float, default=0.25,
                   help='Target firing rate for spike reg')
    return p.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_STEPS = 8

# ############################################################################
# ENHANCED SNN MODULES
# ############################################################################

class SpikeFunction_Learnable(torch.autograd.Function):
    """Spike function with learnable surrogate slope passed as context."""
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
        s = slope.clamp(min=1.0, max=100.0)  # safety clamp
        sig = torch.sigmoid(s * (membrane - threshold))
        sg = sig * (1 - sig) * s
        grad_mem = grad_output * sg
        grad_th = -(grad_output * sg).sum() if ctx.th_needs_grad else None
        grad_slope = (grad_output * sig * (1 - sig) * (membrane - threshold)).sum()
        return grad_mem, grad_th, grad_slope


class SpikeFunction_Fixed(torch.autograd.Function):
    """Original fixed-slope spike function."""
    @staticmethod
    def forward(ctx, membrane, threshold):
        if isinstance(threshold, (int, float)):
            threshold = torch.tensor(float(threshold), device=membrane.device)
        ctx.save_for_backward(membrane, threshold)
        ctx.th_needs_grad = isinstance(threshold, torch.Tensor) and threshold.requires_grad
        return (membrane > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        membrane, threshold = ctx.saved_tensors
        scale = 10.0
        sig = torch.sigmoid(scale * (membrane - threshold))
        sg = sig * (1 - sig) * scale
        return grad_output * sg, -(grad_output * sg).sum() if ctx.th_needs_grad else None


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron with learnable leak β and optional learnable slope.

    Technique B: Learnable leak β (per-channel)
    Technique A: Learnable surrogate slope
    """
    def __init__(self, C, th=1.0, learnable_leak=True, learnable_slope=True):
        super().__init__()
        self.threshold = th
        self.learnable_leak = learnable_leak
        self.learnable_slope = learnable_slope

        if learnable_leak:
            # Initialize β so sigmoid(β_raw) ≈ 0.9 (mild leak)
            self.beta_raw = nn.Parameter(torch.ones(1, C, 1, 1) * 2.2)
        else:
            self.register_buffer('beta_raw', torch.ones(1, C, 1, 1) * 100.)  # effectively 1.0

        if learnable_slope:
            self.slope = nn.Parameter(torch.tensor(10.0))

    def forward(self, x, mem=None):
        if mem is None:
            mem = torch.zeros_like(x)

        beta = torch.sigmoid(self.beta_raw)  # ∈ (0,1)
        mem = beta * mem + x

        if self.learnable_slope:
            sp = SpikeFunction_Learnable.apply(mem, self.threshold, self.slope)
        else:
            sp = SpikeFunction_Fixed.apply(mem, self.threshold)

        return sp, mem - sp * self.threshold


class LIFNeuron_Learnable_Th(nn.Module):
    """LIF neuron with learnable threshold (for decoder IHF replacement)."""
    def __init__(self, C, th=1.0, learnable_leak=True, learnable_slope=True):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(float(th)))
        self.learnable_leak = learnable_leak
        self.learnable_slope = learnable_slope

        if learnable_leak:
            self.beta_raw = nn.Parameter(torch.ones(1, C, 1, 1) * 2.2)
        else:
            self.register_buffer('beta_raw', torch.ones(1, C, 1, 1) * 100.)

        if learnable_slope:
            self.slope = nn.Parameter(torch.tensor(10.0))

    def forward(self, x, mem=None):
        if mem is None:
            mem = torch.zeros_like(x)

        beta = torch.sigmoid(self.beta_raw)
        mem = beta * mem + x

        if self.learnable_slope:
            sp = SpikeFunction_Learnable.apply(mem, self.threshold, self.slope)
        else:
            sp = SpikeFunction_Fixed.apply(mem, self.threshold)

        return sp, mem - sp * self.threshold


class BNTT(nn.Module):
    """Batch Normalization Through Time — separate BN per timestep (Zheng et al., 2021)."""
    def __init__(self, C, T):
        super().__init__()
        self.bns = nn.ModuleList([nn.BatchNorm2d(C) for _ in range(T)])
        self.T = T

    def forward(self, x, t):
        return self.bns[min(t, self.T - 1)](x)


class MembraneInitializer(nn.Module):
    """Learns an initial membrane potential from the input feature map."""
    def __init__(self, C_in, C_out):
        super().__init__()
        self.proj = nn.Conv2d(C_in, C_out, 1, bias=False)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)  # small init

    def forward(self, feat):
        return self.proj(feat)


# ############################################################################
# ENHANCED ENCODER & DECODER
# ############################################################################

class EncoderSNN(nn.Module):
    """Enhanced SNN encoder with BNTT, LIF, and membrane init."""
    def __init__(self, C_in=1024, C1=256, C2=36, T=8, cfg=None):
        super().__init__()
        self.T = T
        cfg = cfg or {}
        use_leak = cfg.get('B', False)
        use_slope = cfg.get('A', False)
        use_bntt = cfg.get('C', False)
        use_mem_init = cfg.get('D', False)

        self.conv1 = nn.Conv2d(C_in, C1, 3, 1, 1)
        self.conv2 = nn.Conv2d(C1, C2, 3, 1, 1)

        # Batch norm: BNTT or standard
        if use_bntt:
            self.bn1 = BNTT(C1, T)
            self.bn2 = BNTT(C2, T)
        else:
            self.bn1 = nn.BatchNorm2d(C1)
            self.bn2 = nn.BatchNorm2d(C2)
        self.use_bntt = use_bntt

        # Neurons: LIF or standard IF
        self.if1 = LIFNeuron(C1, th=1.0, learnable_leak=use_leak, learnable_slope=use_slope)
        self.if2 = LIFNeuron(C2, th=1.0, learnable_leak=use_leak, learnable_slope=use_slope)

        # Membrane initialization
        self.use_mem_init = use_mem_init
        if use_mem_init:
            self.mem_init1 = MembraneInitializer(C_in, C1)
            self.mem_init2 = MembraneInitializer(C1, C2)

    def forward(self, F, m1=None, m2=None, t=0):
        """Forward for a single timestep. Caller loops over T."""
        if self.use_mem_init and t == 0:
            if m1 is None:
                m1 = self.mem_init1(F)
            if m2 is None:
                # Need s1 shape — compute conv1 output for shape
                h1 = self.conv1(F)
                m2 = self.mem_init2(h1.detach())  # detach to avoid double backward

        if self.use_bntt:
            h1 = self.bn1(self.conv1(F), t)
        else:
            h1 = self.bn1(self.conv1(F))
        s1, m1 = self.if1(h1, m1)

        if self.use_bntt:
            h2 = self.bn2(self.conv2(s1), t)
        else:
            h2 = self.bn2(self.conv2(s1))
        s2, m2 = self.if2(h2, m2)

        return s1, s2, m1, m2


class DecoderSNN(nn.Module):
    """Enhanced SNN decoder with BNTT, LIF, and temporal combiner."""
    def __init__(self, C_out=1024, C1=256, C2=36, T=8, cfg=None):
        super().__init__()
        self.T = T
        cfg = cfg or {}
        use_leak = cfg.get('B', False)
        use_slope = cfg.get('A', False)
        use_bntt = cfg.get('C', False)

        self.conv3 = nn.Conv2d(C2, C1, 3, 1, 1)
        self.conv4 = nn.Conv2d(C1, C_out, 3, 1, 1)

        if use_bntt:
            self.bn3 = BNTT(C1, T)
            self.bn4 = BNTT(C_out, T)
        else:
            self.bn3 = nn.BatchNorm2d(C1)
            self.bn4 = nn.BatchNorm2d(C_out)
        self.use_bntt = use_bntt

        self.if3 = LIFNeuron(C1, th=1.0, learnable_leak=use_leak, learnable_slope=use_slope)
        self.ihf = LIFNeuron_Learnable_Th(C_out, th=1.0, learnable_leak=use_leak,
                                           learnable_slope=use_slope)

        self.converter_fc = nn.Linear(2 * T, 2 * T)

    def forward(self, recv_all, mask):
        T_use = len(recv_all)
        m3, m4 = None, None
        Fs, Fm = [], []
        for t in range(T_use):
            if self.use_bntt:
                h3 = self.bn3(self.conv3(recv_all[t] * mask), t)
            else:
                h3 = self.bn3(self.conv3(recv_all[t] * mask))
            s3, m3 = self.if3(h3, m3)

            if self.use_bntt:
                h4 = self.bn4(self.conv4(s3), t)
            else:
                h4 = self.bn4(self.conv4(s3))
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


class SpikeAdaptSC_v2_SNN(nn.Module):
    """SpikeAdapt-SC v2 with SNN-native enhancements."""
    def __init__(self, C_in=1024, C1=256, C2=36, T=8,
                 target_rate=0.75, channel_type='bsc', grid_size=14, cfg=None):
        super().__init__()
        self.T = T; self.C2 = C2; self.grid_size = grid_size
        self.cfg = cfg or {}

        self.encoder = EncoderSNN(C_in, C1, C2, T, cfg)
        self.scorer = ChannelConditionedScorer(C_spike=C2, hidden=32)
        self.block_mask = LearnedBlockMask(target_rate, 0.5)
        self.decoder = DecoderSNN(C_in, C1, C2, T, cfg)
        self.channel = BSC_Channel()

    def forward(self, feat, noise_param=0.0, target_rate_override=None):
        all_S2, m1, m2 = [], None, None

        # Encode all timesteps
        for t in range(self.T):
            _, s2, m1, m2 = self.encoder(feat, m1, m2, t=t)
            all_S2.append(s2)

        # Channel-conditioned scoring
        importance = self.scorer(all_S2, noise_param)

        if target_rate_override is not None:
            old = self.block_mask.target_rate
            self.block_mask.target_rate = target_rate_override
            mask, tx = self.block_mask(importance, training=False)
            self.block_mask.target_rate = old
        else:
            mask, tx = self.block_mask(importance, training=self.training)

        # Transmit through channel
        recv = [self.channel(self.block_mask.apply_mask(all_S2[t], mask), noise_param)
                for t in range(self.T)]

        Fp = self.decoder(recv, mask)

        return Fp, {
            'tx_rate': tx.item(),
            'mask': mask,
            'importance': importance,
            'all_S2': all_S2,  # needed for spike/STDP losses
        }


# ############################################################################
# LOSS HELPERS
# ############################################################################

def spike_rate_loss(all_S2, target_fr=0.25):
    """Penalize firing rates far from target (Technique E)."""
    fr = torch.stack(all_S2).float().mean()
    return (fr - target_fr) ** 2


def stdp_temporal_loss(all_S2):
    """STDP-inspired: encourage temporal correlation between consecutive spikes (Technique F).

    Loss = -mean(S[t] * S[t+1]) → encourages consistent firing across timesteps.
    """
    if len(all_S2) < 2:
        return torch.tensor(0.0, device=all_S2[0].device)
    corr = 0
    for t in range(len(all_S2) - 1):
        corr = corr + (all_S2[t].float() * all_S2[t+1].float()).mean()
    return -corr / (len(all_S2) - 1)


# ############################################################################
# WEIGHT TRANSFER FROM v2 → SNN-native
# ############################################################################

def transfer_weights(model_new, snap_dir, device):
    """Load weights from best v2 S2 checkpoint into new SNN-native model.

    Maps old encoder/decoder weights to new LIF-based architecture.
    """
    s2f = sorted([f for f in os.listdir(snap_dir) if f.startswith("v2_s2_")])
    if not s2f:
        print("  ⚠ No S2 checkpoint found, training from scratch")
        return

    old_state = torch.load(os.path.join(snap_dir, s2f[-1]), map_location=device)
    new_state = model_new.state_dict()

    transferred = 0
    for old_key, old_val in old_state.items():
        # Direct mappings
        mappings = {
            'encoder.conv1.': 'encoder.conv1.',
            'encoder.conv2.': 'encoder.conv2.',
            'decoder.conv3.': 'decoder.conv3.',
            'decoder.conv4.': 'decoder.conv4.',
            'decoder.converter_fc.': 'decoder.converter_fc.',
            'scorer.': 'scorer.',
            'block_mask.': 'block_mask.',
        }

        matched = False
        for old_prefix, new_prefix in mappings.items():
            if old_key.startswith(old_prefix):
                new_key = old_key.replace(old_prefix, new_prefix, 1)
                if new_key in new_state and new_state[new_key].shape == old_val.shape:
                    new_state[new_key] = old_val
                    transferred += 1
                    matched = True
                break

        if not matched:
            # Handle BN weights: old bn1 → new bn1 (if not BNTT)
            # or old bn1 → new bn1.bns.0 through .bns.T-1 (if BNTT)
            for bn_name in ['encoder.bn1.', 'encoder.bn2.', 'decoder.bn3.', 'decoder.bn4.']:
                if old_key.startswith(bn_name):
                    suffix = old_key[len(bn_name):]
                    # Try direct key first
                    if old_key in new_state and new_state[old_key].shape == old_val.shape:
                        new_state[old_key] = old_val
                        transferred += 1
                    else:
                        # BNTT: copy same weights to all timestep BNs
                        for t in range(T_STEPS):
                            bntt_key = f"{bn_name}bns.{t}.{suffix}"
                            if bntt_key in new_state and new_state[bntt_key].shape == old_val.shape:
                                new_state[bntt_key] = old_val
                                transferred += 1
                    break

    model_new.load_state_dict(new_state, strict=False)
    print(f"  ✓ Transferred {transferred} weight tensors from S2 checkpoint")


# ############################################################################
# MAIN
# ############################################################################

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    print(f"Device: {device}")

    # Parse experiment config
    exp = args.exp.upper()
    if exp == 'ALL':
        cfg = {k: True for k in 'ABCDEF'}
    elif exp == 'BASELINE':
        cfg = {}
    else:
        cfg = {c: True for c in exp if c in 'ABCDEF'}

    exp_name = exp if exp in ('ALL', 'BASELINE') else ''.join(sorted(cfg.keys()))
    if not exp_name:
        exp_name = 'BASELINE'
    print(f"Experiment: {exp_name} → {cfg}")

    SNAP_V1 = "./snapshots_aid/"
    SNAP_V2 = f"./snapshots_aid_v2_seed{args.seed}/"
    SNAP_OUT = f"./snapshots_snn_native_{exp_name}_seed{args.seed}/"
    os.makedirs(SNAP_OUT, exist_ok=True)

    # Load backbone
    front = ResNet50Front(grid_size=14).to(device)
    back = ResNet50Back(30).to(device)
    bb = torch.load(os.path.join(SNAP_V1, "backbone_best.pth"), map_location=device)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))}, strict=False)
    front.eval()

    # Load back from v2 S3 checkpoint
    s3f = sorted([f for f in os.listdir(SNAP_V2) if f.startswith("v2_s3_")])
    if s3f:
        ck = torch.load(os.path.join(SNAP_V2, s3f[-1]), map_location=device)
        back.load_state_dict(ck['back'])
        print(f"Loaded back from: {s3f[-1]}")

    # Create enhanced model
    model = SpikeAdaptSC_v2_SNN(
        C_in=1024, C1=256, C2=args.C2, T=T_STEPS,
        target_rate=args.target_rate, channel_type='bsc',
        grid_size=14, cfg=cfg
    ).to(device)

    # Transfer weights from v2
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

    # Evaluate baseline first
    def evaluate_snn(front, model, back, loader, noise_param=0.0):
        model.eval(); back.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                feat = front(imgs)
                Fp, stats = model(feat, noise_param=noise_param)
                preds = back(Fp).argmax(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
        return 100. * correct / total

    acc_init = evaluate_snn(front, model, back, test_loader)
    print(f"\n  Initial accuracy (after weight transfer): {acc_init:.2f}%")

    # --- S3: Joint fine-tuning with SNN-native enhancements ---
    print(f"\n{'='*60}")
    print(f"S3 TRAINING — SNN-Native [{exp_name}]")
    print(f"{'='*60}")

    for p in back.parameters():
        p.requires_grad = True
    params = list(back.parameters()) + list(model.parameters())
    opt = optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs_s3, eta_min=1e-7)

    use_spike_reg = cfg.get('E', False)
    use_stdp = cfg.get('F', False)

    best_acc = 0.0
    history = []

    for ep in range(args.epochs_s3):
        model.train(); back.train()
        pbar = tqdm(train_loader, desc=f"S3 E{ep+1}/{args.epochs_s3}")
        opt.zero_grad()
        ep_loss, ep_fr = 0, 0
        n_steps = 0

        for step, (img, lab) in enumerate(pbar):
            img, lab = img.to(device), lab.to(device)
            noise = sample_noise('bsc')
            with torch.no_grad():
                feat = front(img)

            Fp, stats = model(feat, noise_param=noise)
            logits = back(Fp)

            # Main loss
            loss = criterion(logits, lab) + 2.0 * (stats['tx_rate'] - args.target_rate) ** 2

            # Technique E: Spike-rate regularization
            if use_spike_reg:
                loss = loss + args.lambda_spike * spike_rate_loss(stats['all_S2'], args.target_fr)

            # Technique F: STDP temporal correlation
            if use_stdp:
                loss = loss + args.lambda_stdp * stdp_temporal_loss(stats['all_S2'])

            loss = loss / 2  # gradient accumulation
            loss.backward()

            if (step + 1) % 2 == 0:
                nn.utils.clip_grad_norm_(params, max_norm=1.0)
                opt.step(); opt.zero_grad()

            # Track stats
            with torch.no_grad():
                fr = torch.stack(stats['all_S2']).float().mean().item()
                ep_fr += fr; ep_loss += loss.item(); n_steps += 1

            pbar.set_postfix({
                'L': f'{loss.item():.3f}',
                'tx': f'{stats["tx_rate"]:.2f}',
                'fr': f'{fr:.3f}'
            })

        sched.step()

        # Evaluate every 5 epochs (more frequent for debugging)
        if (ep + 1) % 5 == 0 or ep == 0:
            acc = evaluate_snn(front, model, back, test_loader)
            avg_fr = ep_fr / max(n_steps, 1)

            # Also get firing rate stats
            leak_info = ""
            for name, p in model.named_parameters():
                if 'beta_raw' in name:
                    beta_val = torch.sigmoid(p).mean().item()
                    leak_info = f", β={beta_val:.3f}"
                    break

            slope_info = ""
            for name, p in model.named_parameters():
                if 'slope' in name and 'beta' not in name:
                    slope_info = f", slope={p.item():.1f}"
                    break

            print(f"  S3 E{ep+1}: {acc:.2f}%, FR={avg_fr:.3f}{leak_info}{slope_info}")

            history.append({
                'epoch': ep + 1, 'accuracy': acc,
                'firing_rate': avg_fr, 'loss': ep_loss / max(n_steps, 1)
            })

            if acc > best_acc:
                best_acc = acc
                torch.save({'model': model.state_dict(), 'back': back.state_dict()},
                           os.path.join(SNAP_OUT, f"snn_{exp_name}_{acc:.2f}.pth"))
                print(f"  ✓ Best: {best_acc:.2f}%")

    # Final evaluation
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION — SNN-Native [{exp_name}]")
    print(f"{'='*60}")

    # Load best
    best_f = sorted([f for f in os.listdir(SNAP_OUT) if f.startswith(f"snn_{exp_name}_")])
    if best_f:
        ck = torch.load(os.path.join(SNAP_OUT, best_f[-1]), map_location=device)
        model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])

    acc_clean = evaluate_snn(front, model, back, test_loader, noise_param=0.0)
    print(f"  Clean: {acc_clean:.2f}%")

    ber_results = {'clean': acc_clean}
    for ber in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        acc_n = evaluate_snn(front, model, back, test_loader, noise_param=ber)
        ber_results[f'ber_{ber:.2f}'] = acc_n
        print(f"  BER={ber:.2f}: {acc_n:.2f}%")

    # Save results
    results = {
        'experiment': exp_name,
        'config': cfg,
        'seed': args.seed,
        'best_accuracy': best_acc,
        'baseline': 96.35,
        'delta': best_acc - 96.35,
        'ber_results': ber_results,
        'history': history,
        'hyperparams': {
            'lr': args.lr, 'epochs': args.epochs_s3,
            'lambda_spike': args.lambda_spike if use_spike_reg else None,
            'lambda_stdp': args.lambda_stdp if use_stdp else None,
            'target_fr': args.target_fr if use_spike_reg else None,
        }
    }
    with open(os.path.join(SNAP_OUT, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {SNAP_OUT}/results.json")
    print(f"   Experiment {exp_name}: {best_acc:.2f}% (Δ = {best_acc - 96.35:+.2f}pp from baseline)")
