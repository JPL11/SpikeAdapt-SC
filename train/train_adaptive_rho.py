#!/usr/bin/env python3
"""Train adaptive rho(BER) policies — 3 approaches from literature.

Option 1: Learned MLP Policy (inspired by SNR-EQ-JSCC, IEEE 2025)
    Single MLP maps BER → optimal rho. Supervised by oracle rho* from
    per_rho_scorer_results.json. Simplest, fastest, minimal overhead.

Option 2: MoE Gated Scorer (inspired by MoE-JSCC, arXiv 2026)
    3 pre-trained scorers (rho=0.5, 0.625, 0.875) + gating network
    that selects which scorer to use based on BER + content features.
    More overhead but captures per-image adaptation.

Option 3: Entropy-Adaptive Rate Control (inspired by arXiv Jan 2025)
    Two-policy approach: P1 estimates image complexity (entropy of spike
    features), P2 combines (entropy, BER) to select rho. Content-aware
    and channel-aware rate selection.

All options start from existing V5C-NA checkpoints (no backbone retraining).

Usage:
    python train/train_adaptive_rho.py --option 1 --dataset both
    python train/train_adaptive_rho.py --option 2 --dataset aid
    python train/train_adaptive_rho.py --option 3 --dataset resisc45
    python train/train_adaptive_rho.py --eval-only --option 1
"""

import os, sys, random, json, math, glob, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from train_aid_v2 import ResNet50Front, ResNet50Back, BSC_Channel, LearnedBlockMask, sample_noise
from train_aid_v5 import EncoderV5, DecoderV5
from noise_aware_scorer import NoiseAwareScorer
from run_final_pipeline import AIDDataset5050, RESISC45Dataset, SpikeAdaptSC_v5c_NA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_STEPS = 8
BER_SWEEP = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
RHO_VALUES = [0.10, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875]

# Oracle rho* from per_rho_scorer_results.json analysis
ORACLE_RHO = {
    'aid': {
        0.0: 0.625, 0.05: 0.625, 0.10: 0.625, 0.15: 0.625,
        0.20: 0.750, 0.25: 0.750, 0.30: 0.625,
        0.35: 0.500, 0.40: 0.375, 0.45: 0.375, 0.50: 0.625,
    },
    'resisc45': {
        0.0: 0.875, 0.05: 0.875, 0.10: 0.875, 0.15: 0.875,
        0.20: 0.875, 0.25: 0.625, 0.30: 0.625,
        0.35: 0.500, 0.40: 0.375, 0.45: 0.375, 0.50: 0.625,
    },
}

DATASET_CONFIGS = {
    'aid': {
        'n_classes': 30,
        'ds_cls': AIDDataset5050,
        'ds_kwargs': dict(seed=42),
        'bb_path': './snapshots_aid_5050_seed42/backbone_best.pth',
        'v5_ck_path': './snapshots_aid_v5cna_seed42/v5cna_best_95.42.pth',
        'snap_base': './snapshots_aid_adaptive_rho',
    },
    'resisc45': {
        'n_classes': 45,
        'ds_cls': RESISC45Dataset,
        'ds_kwargs': dict(train_ratio=0.20, seed=42),
        'bb_path': './snapshots_resisc45_5050_seed42/backbone_best.pth',
        'v5_ck_path': './snapshots_resisc45_v5cna_seed42/v5cna_best_92.01.pth',
        'snap_base': './snapshots_resisc45_adaptive_rho',
    },
}


# ======================================================================
# OPTION 1: Learned MLP Policy (SNR-EQ-JSCC inspired)
# ======================================================================

class AdaptiveRhoMLP(nn.Module):
    """Simple MLP: BER → optimal rho.

    Supervised by oracle rho* from per-rho scorer experiments.
    At inference: given estimated BER, outputs the best rho.
    """
    def __init__(self, hidden=64, rho_min=0.10, rho_max=0.875):
        super().__init__()
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        # Init to output ~0.75 at BER=0
        target_sig = (0.75 - rho_min) / (rho_max - rho_min)
        init_bias = math.log(target_sig / (1 - target_sig))
        nn.init.zeros_(self.mlp[-2].weight)
        nn.init.constant_(self.mlp[-2].bias, init_bias)

    def forward(self, ber):
        if isinstance(ber, (int, float)):
            ber_t = torch.tensor([[ber]], dtype=torch.float32,
                                 device=next(self.parameters()).device)
        else:
            ber_t = ber.view(-1, 1)
        raw = self.mlp(ber_t)
        rho = self.rho_min + (self.rho_max - self.rho_min) * raw
        return rho.squeeze()


class SpikeAdaptSC_Option1(nn.Module):
    """V5C-NA + learned rho(BER) MLP policy."""

    def __init__(self, base_model, rho_policy):
        super().__init__()
        self.base = base_model
        self.rho_policy = rho_policy

    def forward(self, feat, noise_param=0.0, target_rate_override=None):
        if target_rate_override is not None:
            return self.base(feat, noise_param, target_rate_override)

        # Policy selects rho based on BER
        rho = self.rho_policy(noise_param)
        rho_val = rho.item() if isinstance(rho, torch.Tensor) else rho
        Fp, stats = self.base(feat, noise_param, target_rate_override=rho_val)
        stats['adaptive_rho'] = rho_val
        return Fp, stats


def train_option1(ds_name, cfg, train_loader, test_loader, front, epochs=30):
    """Train Option 1: MLP rho policy with oracle supervision + end-to-end FT."""
    print(f"\n{'='*60}")
    print(f"  OPTION 1: MLP rho(BER) Policy — {ds_name.upper()}")
    print(f"{'='*60}")

    snap_dir = cfg['snap_base']
    os.makedirs(snap_dir, exist_ok=True)

    # Load base model
    back = ResNet50Back(cfg['n_classes']).to(device)
    base_model = SpikeAdaptSC_v5c_NA(C_in=1024, C1=256, C2=36, T=T_STEPS,
                                      target_rate=0.75, grid_size=14).to(device)
    ck = torch.load(cfg['v5_ck_path'], map_location=device, weights_only=False)
    base_model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])

    # Create policy
    rho_policy = AdaptiveRhoMLP(hidden=64).to(device)
    model = SpikeAdaptSC_Option1(base_model, rho_policy).to(device)

    oracle = ORACLE_RHO[ds_name]

    # Phase 1: Supervised policy pre-training (fast, ~1 min)
    print("\n  Phase 1: Supervised policy pre-training...")
    opt_policy = optim.Adam(rho_policy.parameters(), lr=1e-3)
    for ep in range(200):
        loss_total = 0
        for ber, target_rho in oracle.items():
            pred_rho = rho_policy(ber)
            target = torch.tensor(target_rho, device=device)
            loss = (pred_rho - target) ** 2
            opt_policy.zero_grad(); loss.backward(); opt_policy.step()
            loss_total += loss.item()
        if (ep + 1) % 50 == 0:
            print(f"    E{ep+1}: MSE={loss_total/len(oracle):.6f}")
            for ber in [0.0, 0.15, 0.30, 0.35]:
                pred = rho_policy(ber).item()
                print(f"      BER={ber:.2f} → rho={pred:.3f} (oracle={oracle[ber]:.3f})")

    # Phase 2: End-to-end fine-tuning with adaptive rho
    print(f"\n  Phase 2: End-to-end fine-tuning ({epochs} epochs)...")
    criterion = nn.CrossEntropyLoss()

    # Only fine-tune scorer + policy + decoder (keep encoder frozen for stability)
    ft_params = (list(base_model.scorer.parameters()) +
                 list(base_model.decoder.parameters()) +
                 list(rho_policy.parameters()) +
                 list(back.parameters()))
    opt = optim.Adam(ft_params, lr=5e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-7)
    best_acc = 0

    for ep in range(1, epochs + 1):
        model.train(); back.train()
        base_model.encoder.eval()  # Keep encoder stable
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            ber = sample_noise('bsc')
            Fp, stats = model(feat, noise_param=ber)
            loss = criterion(back(Fp), labels)

            # Rate penalty: encourage policy to match oracle
            if ber in oracle:
                target_rho = oracle[ber]
                rho_pred = rho_policy(ber)
                loss = loss + 5.0 * (rho_pred - target_rho) ** 2

            # Diversity loss
            div = base_model.scorer.compute_diversity_loss(stats['all_S2'], 0.0, 0.30)
            loss = loss + 0.05 * div

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(ft_params, 1.0)
            opt.step()
        scheduler.step()

        if ep % 5 == 0 or ep == epochs:
            acc = evaluate_adaptive(model, back, front, test_loader, rho_policy)
            print(f"    E{ep:02d}: Adaptive acc={acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                torch.save({
                    'model': base_model.state_dict(),
                    'back': back.state_dict(),
                    'rho_policy': rho_policy.state_dict(),
                }, os.path.join(snap_dir, f'opt1_best_{best_acc:.2f}.pth'))

    print(f"  Option 1 best: {best_acc:.2f}%")
    return model, back, rho_policy


# ======================================================================
# OPTION 2: MoE Gated Scorer (MoE-JSCC inspired)
# ======================================================================

class MoEGatingNetwork(nn.Module):
    """Gating network: (BER, content_stats) → expert selection weights.

    3 experts correspond to 3 pre-trained scorer rhos:
        Expert 0: rho=0.875 (high rate, for clean/low noise)
        Expert 1: rho=0.625 (medium rate, for moderate noise)
        Expert 2: rho=0.500 (low rate, for high noise)
    """
    def __init__(self, C_spike=36, hidden=32, n_experts=3):
        super().__init__()
        self.n_experts = n_experts

        # BER branch
        self.ber_embed = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(inplace=True),
        )

        # Content branch: global average pool of spike features → compact stats
        self.content_embed = nn.Sequential(
            nn.Linear(C_spike, hidden),
            nn.ReLU(inplace=True),
        )

        # Gating: combine BER + content → expert weights
        self.gate = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_experts),
        )

    def forward(self, avg_spikes, ber):
        """
        Args:
            avg_spikes: [B, C, H, W] time-averaged spike features
            ber: float
        Returns:
            weights: [B, n_experts] softmax gating weights
        """
        B = avg_spikes.size(0)
        device = avg_spikes.device

        # Content statistics: global average pool
        content_stats = avg_spikes.mean(dim=(2, 3))  # [B, C]
        content_emb = self.content_embed(content_stats)  # [B, hidden]

        # BER embedding
        ber_input = torch.full((B, 1), ber, device=device)
        ber_emb = self.ber_embed(ber_input)  # [B, hidden]

        # Gate
        combined = torch.cat([ber_emb, content_emb], dim=1)  # [B, 2*hidden]
        logits = self.gate(combined)  # [B, n_experts]

        # Gumbel-softmax for differentiable discrete selection during training
        if self.training:
            weights = F.gumbel_softmax(logits, tau=0.5, hard=False)
        else:
            weights = F.softmax(logits, dim=1)

        return weights


class SpikeAdaptSC_Option2(nn.Module):
    """V5C-NA with MoE scorer: 3 expert scorers + gating network."""

    EXPERT_RHOS = [0.875, 0.625, 0.500]

    def __init__(self, C_in=1024, C1=256, C2=36, T=8, grid_size=14):
        super().__init__()
        self.T = T
        self.C2 = C2

        # Shared encoder/decoder (from base model)
        self.encoder = EncoderV5(C_in, C1, C2, T, use_mpbn=True)
        self.decoder = DecoderV5(C_in, C1, C2, T, use_mpbn=True)
        self.channel = BSC_Channel()

        # 3 expert scorers (one per rho regime)
        self.expert_scorers = nn.ModuleList([
            NoiseAwareScorer(C_spike=C2, hidden=32) for _ in range(3)
        ])

        # Block masks for each expert rho
        self.block_masks = nn.ModuleList([
            LearnedBlockMask(rho, 0.5) for rho in self.EXPERT_RHOS
        ])

        # Gating network
        self.gating = MoEGatingNetwork(C_spike=C2, hidden=32, n_experts=3)

    def forward(self, feat, noise_param=0.0, target_rate_override=None):
        # Encode
        all_S2, m1, m2 = [], None, None
        for t in range(self.T):
            _, s2, m1, m2 = self.encoder(feat, m1, m2, t=t)
            all_S2.append(s2)

        avg_spikes = torch.stack(all_S2).mean(0)  # [B, C, H, W]

        if target_rate_override is not None:
            # Fixed rho override for evaluation sweeps
            importance = self.expert_scorers[1](all_S2, noise_param).squeeze(1)
            old = self.block_masks[1].target_rate
            self.block_masks[1].target_rate = target_rate_override
            mask, tx = self.block_masks[1](importance, training=False)
            self.block_masks[1].target_rate = old
            recv = [self.channel(all_S2[t] * mask, noise_param) for t in range(self.T)]
            Fp = self.decoder(recv, mask)
            with torch.no_grad():
                fr = torch.stack(all_S2).mean().item()
            return Fp, {'tx_rate': tx.item(), 'mask': mask, 'importance': importance,
                        'firing_rate': fr, 'all_S2': all_S2, 'adaptive_rho': target_rate_override}

        # Get gating weights
        weights = self.gating(avg_spikes, noise_param)  # [B, 3]

        # Compute each expert's output
        expert_outputs = []
        expert_masks = []
        expert_rhos = []
        for i in range(3):
            imp = self.expert_scorers[i](all_S2, noise_param).squeeze(1)
            mask_i, tx_i = self.block_masks[i](imp, training=self.training)
            recv_i = [self.channel(all_S2[t] * mask_i, noise_param) for t in range(self.T)]
            Fp_i = self.decoder(recv_i, mask_i)
            expert_outputs.append(Fp_i)
            expert_masks.append(mask_i)
            expert_rhos.append(self.EXPERT_RHOS[i])

        # Weighted combination (soft MoE)
        Fp = sum(weights[:, i].view(-1, 1, 1, 1) * expert_outputs[i] for i in range(3))

        # Effective rho (weighted average)
        eff_rho = sum(weights[:, i].mean().item() * expert_rhos[i] for i in range(3))

        # Use the mask from the dominant expert for stats
        dominant = weights.mean(0).argmax().item()
        mask = expert_masks[dominant]

        with torch.no_grad():
            fr = torch.stack(all_S2).mean().item()

        return Fp, {
            'tx_rate': eff_rho, 'mask': mask, 'importance': None,
            'firing_rate': fr, 'all_S2': all_S2,
            'adaptive_rho': eff_rho,
            'expert_weights': weights.detach().mean(0).cpu().tolist(),
        }


def train_option2(ds_name, cfg, train_loader, test_loader, front, epochs=25):
    """Train Option 2: MoE gated scorer."""
    print(f"\n{'='*60}")
    print(f"  OPTION 2: MoE Gated Scorer — {ds_name.upper()}")
    print(f"{'='*60}")

    snap_dir = cfg['snap_base']
    os.makedirs(snap_dir, exist_ok=True)

    # Load base model weights
    back = ResNet50Back(cfg['n_classes']).to(device)
    ck = torch.load(cfg['v5_ck_path'], map_location=device, weights_only=False)

    model = SpikeAdaptSC_Option2(C_in=1024, C1=256, C2=36, T=T_STEPS, grid_size=14).to(device)

    # Load shared encoder/decoder from base checkpoint
    base_state = ck['model']
    enc_state = {k.replace('encoder.', ''): v for k, v in base_state.items() if k.startswith('encoder.')}
    dec_state = {k.replace('decoder.', ''): v for k, v in base_state.items() if k.startswith('decoder.')}
    model.encoder.load_state_dict(enc_state, strict=False)
    model.decoder.load_state_dict(dec_state, strict=False)

    # Initialize all expert scorers from base scorer
    scorer_state = {k.replace('scorer.', ''): v for k, v in base_state.items() if k.startswith('scorer.')}
    for expert in model.expert_scorers:
        expert.load_state_dict(scorer_state)

    back.load_state_dict(ck['back'])
    print(f"  Loaded base weights from {cfg['v5_ck_path']}")

    # Load per-rho scorer weights if available
    per_rho_dir = cfg['snap_base'].replace('_adaptive_rho', '_per_rho')
    for i, rho in enumerate(SpikeAdaptSC_Option2.EXPERT_RHOS):
        rho_tag = f'rho{rho:.3f}'
        cks = sorted(glob.glob(os.path.join(per_rho_dir, f's5_{rho_tag}_best_*.pth')),
                     key=lambda x: float(x.split('_')[-1].replace('.pth', '')))
        if cks:
            per_ck = torch.load(cks[-1], map_location=device, weights_only=False)
            per_scorer_state = {k.replace('scorer.', ''): v
                                for k, v in per_ck['model'].items() if k.startswith('scorer.')}
            model.expert_scorers[i].load_state_dict(per_scorer_state)
            print(f"  Expert {i} (rho={rho}): loaded from {cks[-1]}")

    # Freeze encoder, only train gating + expert scorers + decoder
    for p in model.encoder.parameters():
        p.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    ft_params = (list(model.gating.parameters()) +
                 list(model.expert_scorers.parameters()) +
                 list(model.decoder.parameters()) +
                 list(back.parameters()))
    opt = optim.Adam(ft_params, lr=5e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-7)
    best_acc = 0

    for ep in range(1, epochs + 1):
        model.train(); back.train()
        model.encoder.eval()
        ep_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            ber = sample_noise('bsc')
            Fp, stats = model(feat, noise_param=ber)
            loss = criterion(back(Fp), labels)

            # Load-balancing loss: prevent gate collapse to single expert
            if 'expert_weights' in stats:
                w = torch.tensor(stats['expert_weights'], device=device)
                balance_loss = -(w * torch.log(w + 1e-8)).sum()  # maximize entropy
                loss = loss - 0.1 * balance_loss

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(ft_params, 1.0)
            opt.step()
            ep_loss += loss.item()
        scheduler.step()

        if ep % 5 == 0 or ep == epochs:
            acc = evaluate_adaptive_moe(model, back, front, test_loader)
            print(f"    E{ep:02d}: acc={acc:.2f}%, loss={ep_loss/len(train_loader):.4f}")
            if acc > best_acc:
                best_acc = acc
                torch.save({
                    'model': model.state_dict(), 'back': back.state_dict(),
                }, os.path.join(snap_dir, f'opt2_best_{best_acc:.2f}.pth'))

    print(f"  Option 2 best: {best_acc:.2f}%")
    return model, back


# ======================================================================
# OPTION 3: Entropy-Adaptive Rate Control
# ======================================================================

class EntropyRhoPolicy(nn.Module):
    """Two-policy entropy + BER → rho selection.

    P1: Estimates image complexity from spike feature entropy
    P2: Combines (complexity, BER) → optimal rho

    Inspired by "Semantic Communication with Entropy-and-Channel-Adaptive
    Rate Control" (arXiv 2501.15414, Jan 2025).
    """

    def __init__(self, C_spike=36, hidden=32, rho_min=0.10, rho_max=0.875):
        super().__init__()
        self.rho_min = rho_min
        self.rho_max = rho_max

        # P1: Spike features → complexity score (scalar per image)
        self.complexity_net = nn.Sequential(
            nn.Linear(C_spike + 2, hidden),  # C channel stats + spatial stats
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),  # complexity in [0, 1]
        )

        # P2: (complexity, BER) → optimal rho
        self.rho_net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

        # Init rho_net to output ~0.75
        target_sig = (0.75 - rho_min) / (rho_max - rho_min)
        init_bias = math.log(target_sig / (1 - target_sig))
        nn.init.zeros_(self.rho_net[-2].weight)
        nn.init.constant_(self.rho_net[-2].bias, init_bias)

    def compute_complexity(self, avg_spikes):
        """Compute per-image complexity from spike statistics.

        Args:
            avg_spikes: [B, C, H, W] time-averaged spike features
        Returns:
            complexity: [B, 1] in [0, 1]
        """
        B = avg_spikes.size(0)

        # Channel-wise mean firing rate
        channel_stats = avg_spikes.mean(dim=(2, 3))  # [B, C]

        # Spatial entropy proxy: variance across spatial locations
        spatial_var = avg_spikes.var(dim=(2, 3)).mean(dim=1, keepdim=True)  # [B, 1]

        # Mean firing rate
        mean_fr = avg_spikes.mean(dim=(1, 2, 3), keepdim=True).squeeze(-1).squeeze(-1)  # [B, 1]

        features = torch.cat([channel_stats, spatial_var, mean_fr], dim=1)  # [B, C+2]
        return self.complexity_net(features)  # [B, 1]

    def forward(self, avg_spikes, ber):
        """
        Args:
            avg_spikes: [B, C, H, W]
            ber: float
        Returns:
            rho: [B] optimal rho per image
        """
        B = avg_spikes.size(0)
        device = avg_spikes.device

        complexity = self.compute_complexity(avg_spikes)  # [B, 1]
        ber_input = torch.full((B, 1), ber, device=device)

        combined = torch.cat([complexity, ber_input], dim=1)  # [B, 2]
        raw = self.rho_net(combined)  # [B, 1]
        rho = self.rho_min + (self.rho_max - self.rho_min) * raw

        return rho.squeeze(1)  # [B]


class SpikeAdaptSC_Option3(nn.Module):
    """V5C-NA with entropy-adaptive rho selection."""

    def __init__(self, C_in=1024, C1=256, C2=36, T=8,
                 target_rate=0.75, grid_size=14):
        super().__init__()
        self.T = T
        self.C2 = C2

        self.encoder = EncoderV5(C_in, C1, C2, T, use_mpbn=True)
        self.scorer = NoiseAwareScorer(C_spike=C2, hidden=32)
        self.block_mask = LearnedBlockMask(target_rate, 0.5)
        self.decoder = DecoderV5(C_in, C1, C2, T, use_mpbn=True)
        self.channel = BSC_Channel()

        # Entropy-adaptive rho policy
        self.entropy_policy = EntropyRhoPolicy(C_spike=C2, hidden=32)

    def forward(self, feat, noise_param=0.0, target_rate_override=None,
                use_adaptive=True):
        # Encode
        all_S2, m1, m2 = [], None, None
        for t in range(self.T):
            _, s2, m1, m2 = self.encoder(feat, m1, m2, t=t)
            all_S2.append(s2)

        avg_spikes = torch.stack(all_S2).mean(0)

        # Score importance
        importance = self.scorer(all_S2, noise_param).squeeze(1)

        # Determine rho
        if target_rate_override is not None:
            rho_val = target_rate_override
        elif use_adaptive:
            # Per-image adaptive rho from entropy policy
            rho_per_image = self.entropy_policy(avg_spikes, noise_param)  # [B]
            rho_val = rho_per_image.mean().item()  # Use batch mean for mask
        else:
            rho_val = self.block_mask.target_rate

        # Apply mask at selected rho
        old = self.block_mask.target_rate
        self.block_mask.target_rate = rho_val
        mask, tx = self.block_mask(importance, training=self.training)
        self.block_mask.target_rate = old

        # Channel + decode
        recv = [self.channel(all_S2[t] * mask, noise_param) for t in range(self.T)]
        Fp = self.decoder(recv, mask)

        with torch.no_grad():
            fr = torch.stack(all_S2).mean().item()

        return Fp, {
            'tx_rate': tx.item(), 'mask': mask, 'importance': importance,
            'firing_rate': fr, 'all_S2': all_S2,
            'adaptive_rho': rho_val,
            'complexity': self.entropy_policy.compute_complexity(avg_spikes).detach().mean().item()
                if use_adaptive else None,
        }


def train_option3(ds_name, cfg, train_loader, test_loader, front, epochs=30):
    """Train Option 3: Entropy-adaptive rate control."""
    print(f"\n{'='*60}")
    print(f"  OPTION 3: Entropy-Adaptive Rate Control — {ds_name.upper()}")
    print(f"{'='*60}")

    snap_dir = cfg['snap_base']
    os.makedirs(snap_dir, exist_ok=True)

    # Load base model
    back = ResNet50Back(cfg['n_classes']).to(device)
    model = SpikeAdaptSC_Option3(C_in=1024, C1=256, C2=36, T=T_STEPS,
                                  target_rate=0.75, grid_size=14).to(device)
    ck = torch.load(cfg['v5_ck_path'], map_location=device, weights_only=False)

    # Load base weights
    base_state = ck['model']
    model_state = model.state_dict()
    for k, v in base_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
    model.load_state_dict(model_state, strict=False)
    back.load_state_dict(ck['back'])
    print(f"  Loaded base weights from {cfg['v5_ck_path']}")

    oracle = ORACLE_RHO[ds_name]

    # Freeze encoder for stability
    for p in model.encoder.parameters():
        p.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    ft_params = (list(model.scorer.parameters()) +
                 list(model.decoder.parameters()) +
                 list(model.entropy_policy.parameters()) +
                 list(back.parameters()))
    opt = optim.Adam(ft_params, lr=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-7)
    best_acc = 0

    for ep in range(1, epochs + 1):
        model.train(); back.train()
        model.encoder.eval()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            ber = sample_noise('bsc')
            Fp, stats = model(feat, noise_param=ber, use_adaptive=True)
            loss = criterion(back(Fp), labels)

            # Oracle supervision for rho policy
            if ber in oracle:
                target_rho = oracle[ber]
                pred_rho = stats['adaptive_rho']
                loss = loss + 3.0 * (pred_rho - target_rho) ** 2

            # Diversity loss
            div = model.scorer.compute_diversity_loss(stats['all_S2'], 0.0, 0.30)
            loss = loss + 0.05 * div

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(ft_params, 1.0)
            opt.step()
        scheduler.step()

        if ep % 5 == 0 or ep == epochs:
            acc = evaluate_adaptive_opt3(model, back, front, test_loader)
            print(f"    E{ep:02d}: acc={acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                torch.save({
                    'model': model.state_dict(), 'back': back.state_dict(),
                }, os.path.join(snap_dir, f'opt3_best_{best_acc:.2f}.pth'))

    print(f"  Option 3 best: {best_acc:.2f}%")
    return model, back


# ======================================================================
# EVALUATION HELPERS
# ======================================================================

def evaluate_adaptive(model, back, front, loader, rho_policy):
    """Evaluate Option 1 across BER sweep with adaptive rho."""
    model.eval(); back.eval()
    results = {}
    for ber in BER_SWEEP:
        correct, total = 0, 0
        rho_used = rho_policy(ber).item()
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                Fp, _ = model(feat=front(imgs), noise_param=ber)
                correct += back(Fp).argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        acc = 100. * correct / total
        results[str(ber)] = {'acc': round(acc, 2), 'rho': round(rho_used, 3)}
    # Print summary
    for ber in [0.0, 0.15, 0.30, 0.35]:
        r = results.get(str(ber), {})
        print(f"    BER={ber:.2f}: acc={r.get('acc',0):.2f}% rho={r.get('rho',0):.3f}")
    # Return average accuracy over [0, 0.30]
    return np.mean([results[str(b)]['acc'] for b in BER_SWEEP if b <= 0.30])


def evaluate_adaptive_moe(model, back, front, loader):
    """Evaluate Option 2 with MoE gating."""
    model.eval(); back.eval()
    total_acc = 0
    for ber in [0.0, 0.15, 0.30]:
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                Fp, stats = model(front(imgs), noise_param=ber)
                correct += back(Fp).argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        acc = 100. * correct / total
        total_acc += acc
    return total_acc / 3


def evaluate_adaptive_opt3(model, back, front, loader):
    """Evaluate Option 3 with entropy-adaptive rho."""
    model.eval(); back.eval()
    total_acc = 0
    for ber in [0.0, 0.15, 0.30]:
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                Fp, _ = model(front(imgs), noise_param=ber, use_adaptive=True)
                correct += back(Fp).argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        acc = 100. * correct / total
        total_acc += acc
    return total_acc / 3


def full_eval(model, back, front, loader, option, ds_name):
    """Full BER sweep evaluation for any option."""
    model.eval(); back.eval()
    results = {}
    print(f"\n  Full BER sweep — {ds_name.upper()} (Option {option}):")
    print(f"  {'BER':<8} {'Acc':>8} {'Rho':>8}")
    print(f"  {'-'*26}")

    for ber in BER_SWEEP:
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                Fp, stats = model(front(imgs), noise_param=ber)
                correct += back(Fp).argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        acc = 100. * correct / total
        rho = stats.get('adaptive_rho', 0.75)
        results[str(ber)] = {'acc': round(acc, 2), 'rho': round(rho, 3)}
        label = 'Clean' if ber == 0 else f'{ber:.2f}'
        print(f"  {label:<8} {acc:>7.2f}% {rho:>7.3f}")

    return results


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=int, required=True, choices=[1, 2, 3])
    parser.add_argument('--dataset', default='both', choices=['aid', 'resisc45', 'both'])
    parser.add_argument('--epochs', type=int, default=None,
                        help='Training epochs (default: 30 for opt1/3, 25 for opt2)')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    ds_list = ['aid', 'resisc45'] if args.dataset == 'both' else [args.dataset]

    default_epochs = {1: 30, 2: 25, 3: 30}
    epochs = args.epochs or default_epochs[args.option]

    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])
    tf_train = T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
                           T.ColorJitter(0.2, 0.2, 0.2),
                           T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))])

    all_results = {}

    for ds_name in ds_list:
        cfg = DATASET_CONFIGS[ds_name]
        print(f"\n{'#'*60}\n  {ds_name.upper()} — Option {args.option}\n{'#'*60}")

        # Load datasets
        train_ds = cfg['ds_cls']("./data", tf_train, 'train', **cfg['ds_kwargs'])
        test_ds = cfg['ds_cls']("./data", tf_test, 'test', **cfg['ds_kwargs'])
        train_loader = DataLoader(train_ds, 32, True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)

        # Load backbone
        front = ResNet50Front(grid_size=14).to(device)
        bb = torch.load(cfg['bb_path'], map_location=device, weights_only=False)
        front.load_state_dict({k: v for k, v in bb.items()
                               if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))},
                              strict=False)
        front.eval()
        for p in front.parameters(): p.requires_grad = False

        # Train or eval
        if args.option == 1:
            model, back, rho_policy = train_option1(
                ds_name, cfg, train_loader, test_loader, front, epochs)
            all_results[ds_name] = full_eval(model, back, front, test_loader, 1, ds_name)

        elif args.option == 2:
            model, back = train_option2(
                ds_name, cfg, train_loader, test_loader, front, epochs)
            all_results[ds_name] = full_eval(model, back, front, test_loader, 2, ds_name)

        elif args.option == 3:
            model, back = train_option3(
                ds_name, cfg, train_loader, test_loader, front, epochs)
            all_results[ds_name] = full_eval(model, back, front, test_loader, 3, ds_name)

        del front
        torch.cuda.empty_cache()

    # Save results
    os.makedirs('eval/seed_results', exist_ok=True)
    out = f'eval/seed_results/adaptive_rho_opt{args.option}_results.json'
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {out}")


if __name__ == '__main__':
    main()
