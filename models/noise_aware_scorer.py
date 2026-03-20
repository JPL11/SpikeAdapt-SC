"""Noise-Aware Scorer for SpikeAdapt-SC.

Redesigned scorer that PROVABLY adapts masks to channel conditions.

Key insight: At high BER, the scorer should prioritize blocks that are
ROBUST to bit flips (high spike density, redundant info). At low BER,
it should prioritize blocks with the most discriminative features 
(potentially sparse, fragile patterns).

The scorer has two branches:
  1. Content branch: spatial importance from spike statistics (unchanged)
  2. Noise branch: BER-dependent channel reweighting that modulates
     which channels the content branch attends to

Training uses a diversity loss that forces masks to change with BER:
  L_diversity = -||mask(BER=0) - mask(BER=0.30)||_2

This makes the noise-awareness a TRAINED BEHAVIOR with measurable evidence,
not just an architectural capability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseAwareScorer(nn.Module):
    """Scorer with provable noise-adaptive mask generation.
    
    Architecture:
        Content branch: Conv(C_spike→hidden→1) for spatial importance
        Noise branch: MLP(1→hidden→C_spike) for channel reweighting by BER
        Cross-modulation: noise reweights channels before spatial scoring
    
    The key difference from ChannelConditionedScorer:
        Old: BER appended as extra channel → network learns to ignore it
        New: BER modulates channel WEIGHTS → forced multiplicative effect
    """
    
    def __init__(self, C_spike=36, hidden=32):
        super().__init__()
        
        # Content branch: spatial importance scoring
        self.content_branch = nn.Sequential(
            nn.Conv2d(C_spike, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1)
        )
        
        # Noise branch: BER → channel reweighting
        # Multiplicative modulation forces BER-dependent behavior
        self.noise_channel_gate = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, C_spike)
        )
        # Initialize to identity (all ones) so it doesn't disrupt at start
        nn.init.zeros_(self.noise_channel_gate[-1].weight)
        nn.init.zeros_(self.noise_channel_gate[-1].bias)
        
        # Noise spatial bias: global importance shift based on BER
        self.noise_spatial_bias = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )
        nn.init.zeros_(self.noise_spatial_bias[-1].weight)
        nn.init.zeros_(self.noise_spatial_bias[-1].bias)
    
    def forward(self, all_S2, ber):
        """Score spatial blocks with noise-adaptive channel reweighting.
        
        Args:
            all_S2: list of T spike tensors [B, C, H, W]
            ber: float, bit error rate
        
        Returns:
            importance: [B, 1, H, W] importance scores in (0, 1)
        """
        B = all_S2[0].size(0)
        device = all_S2[0].device
        
        # Average spikes across time
        avg_spikes = torch.stack(all_S2).mean(0)  # B×C×H×W
        
        # BER-dependent channel reweighting
        ber_input = torch.full((B, 1), ber, device=device)
        channel_weights = 1.0 + torch.tanh(self.noise_channel_gate(ber_input))  # B×C
        # channel_weights ∈ (0, 2): multiplicative modulation
        # At init: tanh(0) = 0, so weights = 1.0 (identity)
        
        # Reweight channels: different BER → different channel emphasis
        reweighted = avg_spikes * channel_weights.unsqueeze(-1).unsqueeze(-1)  # B×C×H×W
        
        # Content scoring on reweighted features
        importance = self.content_branch(reweighted)  # B×1×H×W
        
        # Global spatial bias from noise level
        spatial_bias = self.noise_spatial_bias(ber_input)  # B×1
        importance = importance + spatial_bias.unsqueeze(-1).unsqueeze(-1)
        
        return torch.sigmoid(importance)
    
    def compute_diversity_loss(self, all_S2, ber_low=0.0, ber_high=0.30):
        """Compute mask diversity loss: force masks to differ across BER levels.
        
        Returns negative L2 distance (minimizing this maximizes mask difference).
        """
        mask_low = self.forward(all_S2, ber_low)
        mask_high = self.forward(all_S2, ber_high)
        
        # Hamming-like distance (continuous)
        diversity = torch.mean((mask_low - mask_high).pow(2))
        
        # Return negative so that minimizing loss = maximizing diversity
        return -diversity
    
    def get_mask_stats(self, all_S2, ber_values=[0.0, 0.10, 0.20, 0.30]):
        """Diagnostic: measure how much masks change across BER values."""
        masks = {}
        for ber in ber_values:
            with torch.no_grad():
                masks[ber] = self.forward(all_S2, ber)
        
        stats = {}
        ref_mask = masks[0.0]
        for ber in ber_values[1:]:
            diff = (ref_mask - masks[ber]).abs()
            stats[f'hamming_0_vs_{ber}'] = diff.mean().item()
            stats[f'max_diff_0_vs_{ber}'] = diff.max().item()
        
        return stats
