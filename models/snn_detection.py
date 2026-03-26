"""SpikeAdapt-SC Detection: Faithful adaptation for YOLO26-OBB.

Wraps per-level SpikeAdaptSC instances around YOLO26 FPN features,
reusing the exact same components as the classification pipeline:
  - Encoder with LIF neurons + MPBN
  - NoiseAwareScorer (BER-conditioned channel gating)
  - LearnedBlockMask (Gumbel-sigmoid / top-k)
  - BSC channel
  - Decoder with IHF neurons + spike-to-feature converter
"""

import torch
import torch.nn as nn

from .spikeadapt_sc import SpikeAdaptSC


class SpikeAdaptSC_Detection(nn.Module):
    """Multi-scale SpikeAdapt-SC for object detection FPN features.
    
    Creates one SpikeAdaptSC instance per FPN level (P3, P4, P5),
    each with its own encoder/scorer/mask/decoder.
    """
    def __init__(self, channel_sizes, C1=128, C2=36, T=8, 
                 target_rate=0.75, channel_type='bsc'):
        super().__init__()
        self.levels = nn.ModuleList([
            SpikeAdaptSC(
                C_in=c, C1=min(C1, c), C2=C2, T=T,
                target_rate=target_rate, channel_type=channel_type
            )
            for c in channel_sizes
        ])
        self.channel_sizes = channel_sizes
    
    def forward(self, features, ber=0.0, target_rate_override=None):
        """Process all FPN levels through SpikeAdapt-SC.
        
        Args:
            features: list of tensors [P3, P4, P5]
            ber: bit error rate for BSC channel
            target_rate_override: override transmission rate
        
        Returns:
            reconstructed: list of reconstructed feature tensors
            all_info: list of info dicts per level
        """
        reconstructed = []
        all_info = []
        for feat, level in zip(features, self.levels):
            recon, info = level(feat, noise_param=ber,
                              target_rate_override=target_rate_override)
            reconstructed.append(recon)
            all_info.append(info)
        return reconstructed, all_info


class SNN_Detection_Hook(nn.Module):
    """Hook layer to inject SpikeAdaptSC into a YOLO backbone layer.
    
    Args:
        alpha: Residual mixing weight. output = alpha*original + (1-alpha)*SNN_recon.
               alpha=0 means pure SNN, alpha=1 means bypass SNN entirely.
    """
    def __init__(self, original_layer, spikeadapt_level, ber=0.0, alpha=0.0):
        super().__init__()
        self.original_layer = original_layer
        self.spikeadapt = spikeadapt_level
        self.ber = ber
        self.alpha = alpha  # 0=pure SNN, 1=pure original
        # Copy YOLO routing attributes
        self.f = getattr(original_layer, 'f', -1)
        self.i = getattr(original_layer, 'i', 0)
        self.type = getattr(original_layer, 'type', type(original_layer).__name__)
        self.np = getattr(original_layer, 'np', 0)
        # Last forward info
        self.last_info = {}
    
    def forward(self, x):
        out = self.original_layer(x)
        recon, info = self.spikeadapt(out, noise_param=self.ber)
        self.last_info = info
        if self.alpha > 0:
            return self.alpha * out + (1 - self.alpha) * recon
        return recon
