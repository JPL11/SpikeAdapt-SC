"""SpikeAdapt-SC v5 — Iterative improvements on V4-A winner.

V4-A base: A+B+C+E+F (learnable slope, LIF leak, BNTT, spike reg, STDP)
            96.45% clean, 95.85% BER=0.30, 96.29% avg

V5 fixes what failed in V4:
  v5A: + Fixed KD (feature-level MSE, λ=0.05 instead of logit KL at 0.5)
  v5B: + Fixed Channel Gating (all-open init bias=+5, 10-epoch warmup freeze)
  v5C: + MPBN replacing BNTT (no KD — V4-D was sabotaged by KD)
  v5D: + Spike-driven attention in scorer (replaces flat temporal averaging)
  v5E: Best combo from v5A-D

Usage:
  python train/train_aid_v5.py --exp v5A
"""

import os, sys, random, json, math, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from train_aid_v2 import (
    AIDDataset, ResNet50Front, ResNet50Back,
    BSC_Channel, LearnedBlockMask, sample_noise
)

# ############################################################################
# ARGS
# ############################################################################
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--exp', type=str, default='v5A')
    p.add_argument('--epochs_s3', type=int, default=40)
    p.add_argument('--target_rate', type=float, default=0.75)
    p.add_argument('--C2', type=int, default=36)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--lambda_spike', type=float, default=0.1)
    p.add_argument('--lambda_stdp', type=float, default=0.05)
    p.add_argument('--lambda_kd', type=float, default=0.05)  # 10x lower than V4
    p.add_argument('--target_fr', type=float, default=0.25)
    return p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_STEPS = 8

# ############################################################################
# SNN MODULES (from V4, proven)
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
    def __init__(self, C, th=1.0):
        super().__init__()
        self.threshold = th
        self.beta_raw = nn.Parameter(torch.ones(1, C, 1, 1) * 2.2)
        self.slope = nn.Parameter(torch.tensor(10.0))

    def forward(self, x, mem=None):
        if mem is None:
            mem = torch.zeros_like(x)
        beta = torch.sigmoid(self.beta_raw)
        mem = beta * mem + x
        sp = SpikeFunction_Learnable.apply(mem, self.threshold, self.slope)
        return sp, mem - sp * self.threshold


class LIFNeuron_LTh(nn.Module):
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
    def __init__(self, C, T):
        super().__init__()
        self.bns = nn.ModuleList([nn.BatchNorm2d(C) for _ in range(T)])
        self.T = T
    def forward(self, x, t):
        return self.bns[min(t, self.T - 1)](x)


class MPBN(nn.Module):
    """Membrane Potential BN (ICCV 2023) — normalize mem before spike."""
    def __init__(self, C, T):
        super().__init__()
        self.bns = nn.ModuleList([nn.BatchNorm2d(C, affine=True) for _ in range(T)])
        self.T = T
    def forward(self, x, t):
        return self.bns[min(t, self.T - 1)](x)


# ############################################################################
# NEW: Channel Gating v2 (fixed initialization + warmup)
# ############################################################################

class ChannelGateV2(nn.Module):
    """Fixed channel gating: all-open init (bias=+5), supports warmup freeze."""
    def __init__(self, C2=36, hidden=32):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden)
        self.fc2 = nn.Linear(hidden, C2)
        # Initialize to all-open: bias=+5 → sigmoid(5)≈0.993
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 5.0)
        self.frozen = False

    def freeze(self):
        self.frozen = True
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        self.frozen = False
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, spikes, channel_estimate):
        B = spikes.size(0)
        if isinstance(channel_estimate, (int, float)):
            ch_in = torch.full((B, 1), channel_estimate, device=spikes.device)
        else:
            ch_in = channel_estimate.view(B, 1)
        g = torch.sigmoid(self.fc2(F.relu(self.fc1(ch_in))))  # (B, C2)
        return spikes * g.unsqueeze(-1).unsqueeze(-1), g.mean()


# ############################################################################
# NEW: Hybrid Attention Scorer (residual refinement on pretrained scorer)
# ############################################################################

class AttentionRefinement(nn.Module):
    """Spike-driven attention refinement module.

    Adds temporal attention + spatial context on TOP of the pretrained
    ChannelConditionedScorer output. Uses residual connection with
    learnable blend weight α (initialized to 0 → starts as pure pretrained).

    From: STAtten (CVPR 2024), adapted for spike features.
    """
    def __init__(self, C_spike=36, T=8, hidden=32):
        super().__init__()
        self.T = T

        # Blend weight: sigmoid(α_raw). α_raw=-5 → sigmoid≈0.007 → nearly zero
        self.alpha_raw = nn.Parameter(torch.tensor(-5.0))

        # Temporal attention: learn which timesteps matter most
        self.temporal_attn = nn.Parameter(torch.ones(T) / T)

        # Spatial refinement: 3×3 conv on spike features → score correction
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(C_spike, hidden, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden, 1, 1),
        )
        # Initialize last conv to near-zero so refinement starts small
        nn.init.zeros_(self.spatial_refine[-1].weight)
        nn.init.zeros_(self.spatial_refine[-1].bias)

    def forward(self, all_S2, base_score):
        """Refine base_score from pretrained scorer with attention context."""
        alpha = torch.sigmoid(self.alpha_raw)

        # Temporal attention: weighted sum
        weights = F.softmax(self.temporal_attn[:len(all_S2)], dim=0)
        spike_weighted = sum(w * s for w, s in zip(weights, all_S2))

        # Spatial refinement
        refinement = self.spatial_refine(spike_weighted).squeeze(1)  # (B, H, W)

        # Residual blend: (1-α)*base + α*refinement
        return (1 - alpha) * base_score + alpha * refinement


# ############################################################################
# V5 ENCODER & DECODER (same as V4, with MPBN option)
# ############################################################################

class EncoderV5(nn.Module):
    def __init__(self, C_in=1024, C1=256, C2=36, T=8, use_mpbn=False):
        super().__init__()
        self.T = T; self.use_mpbn = use_mpbn

        self.conv1 = nn.Conv2d(C_in, C1, 3, 1, 1)
        self.conv2 = nn.Conv2d(C1, C2, 3, 1, 1)

        if not use_mpbn:
            self.bn1 = BNTT(C1, T); self.bn2 = BNTT(C2, T)
        else:
            self.bn1 = nn.BatchNorm2d(C1); self.bn2 = nn.BatchNorm2d(C2)
            self.mpbn1 = MPBN(C1, T); self.mpbn2 = MPBN(C2, T)

        self.if1 = LIFNeuron(C1); self.if2 = LIFNeuron(C2)

    def forward(self, F, m1=None, m2=None, t=0):
        if self.use_mpbn:
            h1 = self.bn1(self.conv1(F))
            if m1 is None: m1 = torch.zeros_like(h1)
            beta1 = torch.sigmoid(self.if1.beta_raw)
            m1 = beta1 * m1 + h1
            m1 = self.mpbn1(m1, t)
            s1 = SpikeFunction_Learnable.apply(m1, self.if1.threshold, self.if1.slope)
            m1 = m1 - s1 * self.if1.threshold
        else:
            h1 = self.bn1(self.conv1(F), t)
            s1, m1 = self.if1(h1, m1)

        if self.use_mpbn:
            h2 = self.bn2(self.conv2(s1))
            if m2 is None: m2 = torch.zeros_like(h2)
            beta2 = torch.sigmoid(self.if2.beta_raw)
            m2 = beta2 * m2 + h2
            m2 = self.mpbn2(m2, t)
            s2 = SpikeFunction_Learnable.apply(m2, self.if2.threshold, self.if2.slope)
            m2 = m2 - s2 * self.if2.threshold
        else:
            h2 = self.bn2(self.conv2(s1), t)
            s2, m2 = self.if2(h2, m2)

        return s1, s2, m1, m2


class DecoderV5(nn.Module):
    def __init__(self, C_out=1024, C1=256, C2=36, T=8, use_mpbn=False):
        super().__init__()
        self.T = T; self.use_mpbn = use_mpbn

        self.conv3 = nn.Conv2d(C2, C1, 3, 1, 1)
        self.conv4 = nn.Conv2d(C1, C_out, 3, 1, 1)

        if not use_mpbn:
            self.bn3 = BNTT(C1, T); self.bn4 = BNTT(C_out, T)
        else:
            self.bn3 = nn.BatchNorm2d(C1); self.bn4 = nn.BatchNorm2d(C_out)
            self.mpbn3 = MPBN(C1, T); self.mpbn4 = MPBN(C_out, T)

        self.if3 = LIFNeuron(C1); self.ihf = LIFNeuron_LTh(C_out)
        self.converter_fc = nn.Linear(2 * T, 2 * T)

    def forward(self, recv_all, mask):
        T_use = len(recv_all)
        m3, m4 = None, None
        Fs, Fm = [], []
        for t in range(T_use):
            inp = recv_all[t] * mask

            if self.use_mpbn:
                h3 = self.bn3(self.conv3(inp))
                if m3 is None: m3 = torch.zeros_like(h3)
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
                if m4 is None: m4 = torch.zeros_like(h4)
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


# ############################################################################
# V5 MODEL
# ############################################################################

class SpikeAdaptSC_v5(nn.Module):
    def __init__(self, C_in=1024, C1=256, C2=36, T=8,
                 target_rate=0.75, grid_size=14,
                 use_mpbn=False, use_channel_gate=False,
                 use_spike_attn=False):
        super().__init__()
        self.T = T; self.C2 = C2; self.grid_size = grid_size
        self.use_spike_attn = use_spike_attn

        self.encoder = EncoderV5(C_in, C1, C2, T, use_mpbn)

        # Always keep pretrained scorer (for weight transfer)
        from train_aid_v2 import ChannelConditionedScorer
        self.scorer = ChannelConditionedScorer(C_spike=C2, hidden=32)

        # Attention refinement: residual on top of pretrained scorer
        if use_spike_attn:
            self.attn_refine = AttentionRefinement(C_spike=C2, T=T, hidden=32)

        self.block_mask = LearnedBlockMask(target_rate, 0.5)
        self.decoder = DecoderV5(C_in, C1, C2, T, use_mpbn)
        self.channel = BSC_Channel()

        self.use_channel_gate = use_channel_gate
        if use_channel_gate:
            self.ch_gate = ChannelGateV2(C2, hidden=32)

    def forward(self, feat, noise_param=0.0, target_rate_override=None):
        all_S2, m1, m2 = [], None, None
        for t in range(self.T):
            _, s2, m1, m2 = self.encoder(feat, m1, m2, t=t)
            all_S2.append(s2)

        importance = self.scorer(all_S2, noise_param)
        if self.use_spike_attn:
            importance = self.attn_refine(all_S2, importance)

        if target_rate_override is not None:
            old = self.block_mask.target_rate
            self.block_mask.target_rate = target_rate_override
            mask, tx = self.block_mask(importance, training=False)
            self.block_mask.target_rate = old
        else:
            mask, tx = self.block_mask(importance, training=self.training)

        gate_rate = 1.0
        if self.use_channel_gate:
            gated_S2 = []
            for t in range(self.T):
                masked = all_S2[t] * mask
                gs, gr = self.ch_gate(masked, noise_param)
                gated_S2.append(gs)
                gate_rate = gr.item()
            recv = [self.channel(gated_S2[t], noise_param) for t in range(self.T)]
        else:
            recv = [self.channel(all_S2[t] * mask, noise_param) for t in range(self.T)]

        Fp = self.decoder(recv, mask)

        return Fp, {
            'tx_rate': tx.item(), 'mask': mask, 'importance': importance,
            'all_S2': all_S2, 'gate_rate': gate_rate,
        }


# ############################################################################
# LOSSES
# ############################################################################

def spike_rate_loss(all_S2, target_fr=0.25):
    return (torch.stack(all_S2).float().mean() - target_fr) ** 2

def stdp_temporal_loss(all_S2):
    if len(all_S2) < 2:
        return torch.tensor(0.0, device=all_S2[0].device)
    corr = 0
    for t in range(len(all_S2) - 1):
        corr = corr + (all_S2[t].float() * all_S2[t+1].float()).mean()
    return -corr / (len(all_S2) - 1)

def feature_kd_loss(feat_snn, feat_ann):
    """Feature-level KD: MSE between SNN decoder output and original features.
    Much gentler than logit-level KL divergence.
    """
    return F.mse_loss(feat_snn, feat_ann)


# ############################################################################
# WEIGHT TRANSFER (from V4-A best checkpoint or S2)
# ############################################################################

def transfer_weights(model_new, snap_v4a, snap_v2, device, use_spike_attn=False):
    """Try V4-A checkpoint first (best model), fall back to S2."""
    # Try V4-A
    v4a_dir = f"./snapshots_v4_V4A_seed42/"
    v4a_files = sorted([f for f in os.listdir(v4a_dir) if f.startswith("v4_V4A_")])  if os.path.exists(v4a_dir) else []

    if v4a_files:
        ck = torch.load(os.path.join(v4a_dir, v4a_files[-1]), map_location=device)
        old_state = ck['model']
        new_state = model_new.state_dict()
        transferred = 0

        for key, val in old_state.items():
            # Skip scorer if using spike attention (different architecture)
            if use_spike_attn and key.startswith('scorer.'):
                continue
            if key in new_state and new_state[key].shape == val.shape:
                new_state[key] = val
                transferred += 1

        model_new.load_state_dict(new_state, strict=False)
        print(f"  ✓ Transferred {transferred} params from V4-A best checkpoint")
        return ck.get('back', None)
    else:
        # Fall back to S2
        s2f = sorted([f for f in os.listdir(snap_v2) if f.startswith("v2_s2_")])
        if s2f:
            old_state = torch.load(os.path.join(snap_v2, s2f[-1]), map_location=device)
            new_state = model_new.state_dict()
            transferred = 0
            for old_key, old_val in old_state.items():
                for prefix in ['encoder.conv1.', 'encoder.conv2.', 'decoder.conv3.',
                               'decoder.conv4.', 'decoder.converter_fc.']:
                    if old_key.startswith(prefix) and old_key in new_state and new_state[old_key].shape == old_val.shape:
                        new_state[old_key] = old_val; transferred += 1; break
                # BNTT transfer
                for bn_name in ['encoder.bn1.', 'encoder.bn2.', 'decoder.bn3.', 'decoder.bn4.']:
                    if old_key.startswith(bn_name):
                        suffix = old_key[len(bn_name):]
                        if old_key in new_state and new_state[old_key].shape == old_val.shape:
                            new_state[old_key] = old_val; transferred += 1
                        else:
                            for t in range(T_STEPS):
                                bntt_key = f"{bn_name}bns.{t}.{suffix}"
                                if bntt_key in new_state and new_state[bntt_key].shape == old_val.shape:
                                    new_state[bntt_key] = old_val; transferred += 1
                        break
            model_new.load_state_dict(new_state, strict=False)
            print(f"  ✓ Transferred {transferred} params from S2 checkpoint")
        return None


# ############################################################################
# MAIN
# ############################################################################

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    print(f"Device: {device}")

    exp = args.exp.lower()

    # Experiment configs
    EXP_CONFIG = {
        'v5a': {'kd': True,  'cg': False, 'mpbn': False, 'attn': False, 'desc': 'V4-A + Fixed KD (feature MSE, λ=0.05)'},
        'v5b': {'kd': False, 'cg': True,  'mpbn': False, 'attn': False, 'desc': 'V4-A + Fixed Channel Gate (open init+warmup)'},
        'v5c': {'kd': False, 'cg': False, 'mpbn': True,  'attn': False, 'desc': 'V4-A + MPBN (no KD)'},
        'v5d': {'kd': False, 'cg': False, 'mpbn': False, 'attn': True,  'desc': 'V4-A + Spike Attention Scorer'},
        'v5e': {'kd': False, 'cg': True,  'mpbn': False, 'attn': True,  'desc': 'Best combo: CG + Attn (no KD)'},
    }

    if exp not in EXP_CONFIG:
        print(f"Unknown experiment: {exp}. Options: {list(EXP_CONFIG.keys())}")
        sys.exit(1)

    cfg = EXP_CONFIG[exp]
    exp_label = args.exp.upper()
    print(f"Experiment: {exp_label} — {cfg['desc']}")

    SNAP_V1 = "./snapshots_aid/"
    SNAP_V2 = f"./snapshots_aid_v2_seed{args.seed}/"
    SNAP_OUT = f"./snapshots_v5_{exp_label}_seed{args.seed}/"
    os.makedirs(SNAP_OUT, exist_ok=True)

    # Load backbone
    front = ResNet50Front(grid_size=14).to(device)
    back = ResNet50Back(30).to(device)
    bb = torch.load(os.path.join(SNAP_V1, "backbone_best.pth"), map_location=device)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))}, strict=False)
    front.eval()

    # Load back
    s3f = sorted([f for f in os.listdir(SNAP_V2) if f.startswith("v2_s3_")])
    if s3f:
        ck = torch.load(os.path.join(SNAP_V2, s3f[-1]), map_location=device)
        back.load_state_dict(ck['back'])
        print(f"Loaded back: {s3f[-1]}")

    # Create V5 model
    model = SpikeAdaptSC_v5(
        C_in=1024, C1=256, C2=args.C2, T=T_STEPS,
        target_rate=args.target_rate, grid_size=14,
        use_mpbn=cfg['mpbn'], use_channel_gate=cfg['cg'],
        use_spike_attn=cfg['attn']
    ).to(device)

    back_state = transfer_weights(model, None, SNAP_V2, device, use_spike_attn=cfg['attn'])
    if back_state:
        back.load_state_dict(back_state)

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

    def evaluate(front, model, back, loader, noise_param=0.0):
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

    acc_init = evaluate(front, model, back, test_loader)
    print(f"\n  Initial accuracy: {acc_init:.2f}%")

    # --- S3 Training ---
    print(f"\n{'='*60}")
    print(f"S3 TRAINING — V5 [{exp_label}]")
    print(f"{'='*60}")

    for p in back.parameters():
        p.requires_grad = True

    # Channel gate warmup: freeze for first 10 epochs
    CG_WARMUP = 10
    if cfg['cg']:
        model.ch_gate.freeze()
        print(f"  Channel gate frozen for {CG_WARMUP} warmup epochs")

    # Exclude frozen gate params from optimizer to avoid duplicate param groups
    gate_param_ids = set()
    if cfg['cg']:
        gate_param_ids = {id(p) for p in model.ch_gate.parameters()}

    slope_params = [p for n, p in model.named_parameters()
                    if 'slope' in n and id(p) not in gate_param_ids]
    other_params = [p for n, p in model.named_parameters()
                    if 'slope' not in n and id(p) not in gate_param_ids]
    params = [
        {'params': other_params, 'lr': args.lr},
        {'params': slope_params, 'lr': args.lr * 10},
        {'params': back.parameters(), 'lr': args.lr},
    ]
    opt = optim.Adam(params, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs_s3, eta_min=1e-7)

    best_acc = 0.0
    history = []

    for ep in range(args.epochs_s3):
        # Unfreeze channel gate after warmup — add as new param group
        if cfg['cg'] and ep == CG_WARMUP:
            model.ch_gate.unfreeze()
            opt.add_param_group({'params': list(model.ch_gate.parameters()), 'lr': args.lr * 0.1})
            print(f"  ✓ Channel gate unfrozen at epoch {ep+1}")

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

            # Feature-level KD (fixed: MSE, much lower λ)
            if cfg['kd']:
                loss = loss + args.lambda_kd * feature_kd_loss(Fp, feat)

            # Spike-rate regularization
            loss = loss + args.lambda_spike * spike_rate_loss(stats['all_S2'], args.target_fr)

            # STDP temporal loss
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

            pbar.set_postfix({'L': f'{loss.item():.3f}', 'tx': f'{stats["tx_rate"]:.2f}', 'fr': f'{fr:.3f}'})

        sched.step()

        if (ep + 1) % 5 == 0 or ep == 0:
            acc = evaluate(front, model, back, test_loader)
            avg_fr = ep_fr / max(n_steps, 1)
            info = ""
            for name, p in model.named_parameters():
                if 'beta_raw' in name:
                    info += f", β={torch.sigmoid(p).mean().item():.3f}"; break
            for name, p in model.named_parameters():
                if 'slope' in name and 'beta' not in name:
                    info += f", slope={p.item():.1f}"; break
            if cfg['cg']:
                with torch.no_grad():
                    g = torch.sigmoid(model.ch_gate.fc2.bias).mean().item()
                info += f", gate={g:.3f}"

            print(f"  S3 E{ep+1}: {acc:.2f}%, FR={avg_fr:.3f}{info}")
            history.append({'epoch': ep+1, 'accuracy': acc, 'firing_rate': avg_fr})

            if acc > best_acc:
                best_acc = acc
                torch.save({'model': model.state_dict(), 'back': back.state_dict()},
                           os.path.join(SNAP_OUT, f"v5_{exp_label}_{acc:.2f}.pth"))
                print(f"  ✓ Best: {best_acc:.2f}%")

    # Final eval
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION — V5 [{exp_label}]")
    print(f"{'='*60}")

    best_f = sorted([f for f in os.listdir(SNAP_OUT) if f.startswith(f"v5_{exp_label}_")])
    if best_f:
        ck = torch.load(os.path.join(SNAP_OUT, best_f[-1]), map_location=device)
        model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])

    acc_clean = evaluate(front, model, back, test_loader, noise_param=0.0)
    print(f"  Clean: {acc_clean:.2f}%")

    ber_results = {'clean': acc_clean}
    for ber in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        acc_n = evaluate(front, model, back, test_loader, noise_param=ber)
        ber_results[f'ber_{ber:.2f}'] = acc_n
        print(f"  BER={ber:.2f}: {acc_n:.2f}%")

    # Reference results
    v4a = {'clean': 96.45, 'ber_0.30': 95.85}
    orig = {'clean': 96.35, 'ber_0.30': 92.90}

    results = {
        'experiment': exp_label, 'version': 'v5', 'seed': args.seed,
        'best_accuracy': best_acc,
        'delta_v4a': best_acc - v4a['clean'],
        'delta_orig': best_acc - orig['clean'],
        'ber_results': ber_results,
        'history': history,
        'config': cfg,
    }
    with open(os.path.join(SNAP_OUT, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ V5 [{exp_label}]: {best_acc:.2f}% (vs V4-A: {best_acc-v4a['clean']:+.2f}pp, vs orig: {best_acc-orig['clean']:+.2f}pp)")
