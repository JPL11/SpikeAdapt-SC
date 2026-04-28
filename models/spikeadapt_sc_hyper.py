"""SpikeAdaptSC with Hyperprior + Adaptive rho(BER) — enhanced V5C-NA.

Features:
  1. Hyperprior side-channel for noise robustness
  2. Learned adaptive rho policy: MLP(BER → optimal rho)
     At low BER: rho ≈ 0.75 (use full bandwidth)
     At high BER: rho drops (transmit fewer, better-selected blocks)

Compatible with existing V5C-NA checkpoints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))

from train_aid_v2 import BSC_Channel, LearnedBlockMask
from train_aid_v5 import EncoderV5, DecoderV5
from models.noise_aware_scorer import NoiseAwareScorer
from models.classification_hyperprior import ClassificationHyperprior


class AdaptiveRhoPolicy(nn.Module):
    """Learned mapping BER → optimal transmission rate rho.

    At BER=0: rho ≈ 0.75 (default, use most bandwidth)
    At BER=0.35: rho ≈ 0.375 (transmit fewer but important blocks)
    At BER=0.50: rho ≈ 0.1 (minimal transmission, everything gets corrupted)

    Output is clamped to [rho_min, rho_max] for stability.
    """
    def __init__(self, hidden=32, rho_min=0.05, rho_max=0.90, default_rho=0.75):
        super().__init__()
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.default_rho = default_rho

        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),  # Output in [0, 1], scaled to [rho_min, rho_max]
        )

        # Initialize so that MLP(0) ≈ default_rho
        # Sigmoid output for default_rho: (default_rho - rho_min) / (rho_max - rho_min)
        nn.init.zeros_(self.mlp[-2].weight)
        target_sigmoid = (default_rho - rho_min) / (rho_max - rho_min)
        init_bias = torch.log(torch.tensor(target_sigmoid / (1 - target_sigmoid)))
        nn.init.constant_(self.mlp[-2].bias, init_bias.item())

    def forward(self, ber):
        """
        Args:
            ber: float or tensor, bit error rate
        Returns:
            rho: float, optimal transmission rate
        """
        if isinstance(ber, (int, float)):
            ber_t = torch.tensor([[ber]], dtype=torch.float32,
                                 device=next(self.parameters()).device)
        else:
            ber_t = ber.view(-1, 1)
        raw = self.mlp(ber_t)  # [0, 1]
        rho = self.rho_min + (self.rho_max - self.rho_min) * raw
        return rho.squeeze()


class SpikeAdaptSC_Hyper(nn.Module):
    """V5C-NA + Hyperprior + Adaptive rho(BER) policy."""

    def __init__(self, C_in=1024, C1=256, C2=36, T=8,
                 target_rate=0.75, grid_size=14, C_hyper=8,
                 adaptive_rho=False):
        super().__init__()
        self.T = T
        self.C2 = C2
        self.grid_size = grid_size
        self.adaptive_rho = adaptive_rho

        # Core V5C-NA components
        self.encoder = EncoderV5(C_in, C1, C2, T, use_mpbn=True)
        self.scorer = NoiseAwareScorer(C_spike=C2, hidden=32)
        self.block_mask = LearnedBlockMask(target_rate, 0.5)
        self.decoder = DecoderV5(C_in, C1, C2, T, use_mpbn=True)
        self.channel = BSC_Channel()

        # Hyperprior side-channel
        self.hyperprior = ClassificationHyperprior(C_spike=C2, C_hyper=C_hyper)

        # Adaptive rho policy
        self.rho_policy = AdaptiveRhoPolicy(
            hidden=32, rho_min=0.05, rho_max=0.90, default_rho=target_rate
        )

    def forward(self, feat, noise_param=0.0, target_rate_override=None,
                use_hyperprior=True, use_adaptive_rho=None):
        if use_adaptive_rho is None:
            use_adaptive_rho = self.adaptive_rho

        # SNN Encoding
        all_S2, m1, m2 = [], None, None
        for t in range(self.T):
            _, s2, m1, m2 = self.encoder(feat, m1, m2, t=t)
            all_S2.append(s2)

        # Scoring
        importance = self.scorer(all_S2, noise_param).squeeze(1)

        # Determine rho
        if target_rate_override is not None:
            # Explicit override (for eval sweeps)
            old = self.block_mask.target_rate
            self.block_mask.target_rate = target_rate_override
            mask, tx = self.block_mask(importance, training=False)
            self.block_mask.target_rate = old
            adaptive_rho_val = target_rate_override
        elif use_adaptive_rho:
            # Learned policy: BER → optimal rho
            adaptive_rho_val = self.rho_policy(noise_param)
            rho_float = adaptive_rho_val.item() if isinstance(adaptive_rho_val, torch.Tensor) else adaptive_rho_val
            old = self.block_mask.target_rate
            self.block_mask.target_rate = rho_float
            mask, tx = self.block_mask(importance, training=self.training)
            self.block_mask.target_rate = old
        else:
            # Fixed rho (original behavior)
            mask, tx = self.block_mask(importance, training=self.training)
            adaptive_rho_val = self.block_mask.target_rate

        # BSC channel
        recv = [self.channel(all_S2[t] * mask, noise_param) for t in range(self.T)]

        # Hyperprior
        if use_hyperprior:
            recv = self.hyperprior(recv, ber=noise_param)

        # Decode
        Fp = self.decoder(recv, mask)

        with torch.no_grad():
            fr = torch.stack(all_S2).mean().item()

        return Fp, {
            'tx_rate': tx.item(),
            'mask': mask,
            'importance': importance,
            'firing_rate': fr,
            'all_S2': all_S2,
            'adaptive_rho': adaptive_rho_val.item() if isinstance(adaptive_rho_val, torch.Tensor) else adaptive_rho_val,
        }
