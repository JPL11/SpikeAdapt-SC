"""SpikeAdapt-SC: Content-Adaptive Semantic Communication Model.

Main model class combining:
- SNN encoder with LIF neurons and MPBN (paper V5C architecture)
- Noise-aware importance scorer with BER-conditioned channel gates
- Differentiable block masking (Gumbel-sigmoid / top-k)
- BSC/AWGN/Rayleigh/BEC channel simulation
- SNN decoder with IHF neurons and spike-to-feature converter

This module matches the actual training code in train/train_aid_v5.py
and train/run_final_pipeline.py.
"""

import torch
import torch.nn as nn

from .snn_modules import (
    LIFNeuron, IFNeuron, IHFNeuron, MPBN,
    get_channel
)
from .noise_aware_scorer import NoiseAwareScorer


# ============================================================================
# Block Masking
# ============================================================================

class LearnedBlockMask(nn.Module):
    """Differentiable block masking with Gumbel-sigmoid (train) / top-k (eval).
    
    During training: soft masks via Gumbel-sigmoid with straight-through.
    During evaluation: deterministic top-k selection by importance score.
    """
    def __init__(self, target_rate=0.75, temperature=0.5):
        super().__init__()
        self.target_rate = target_rate
        self.temperature = temperature

    def forward(self, importance, training=True):
        B, _, H, W = importance.shape
        importance = importance.squeeze(1)  # B×H×W
        if training:
            logits = torch.log(importance / (1 - importance + 1e-7) + 1e-7)
            u = torch.rand_like(logits).clamp(1e-7, 1 - 1e-7)
            soft = torch.sigmoid((logits - torch.log(-torch.log(u))) / self.temperature)
            hard = (soft > 0.5).float()
            mask = hard + (soft - soft.detach())  # STE
        else:
            k = max(1, int(self.target_rate * H * W))
            flat = importance.view(B, -1)
            _, idx = flat.topk(k, dim=1)
            mask = torch.zeros_like(flat)
            mask.scatter_(1, idx, 1.0)
            mask = mask.view(B, H, W)
        return mask.unsqueeze(1), mask.mean()

    def apply_mask(self, x, mask):
        return x * mask


# ============================================================================
# Encoder (V5C: LIF + MPBN)
# ============================================================================

class Encoder(nn.Module):
    """SNN Encoder with LIF neurons and MPBN (paper architecture).
    
    Two 3×3 conv layers with:
    - LIF neurons (leaky IF with learnable β and slope)
    - MPBN (membrane potential batch normalization, per-timestep BN)
    
    This is the V5C architecture that achieves firing rate 0.167.
    """
    def __init__(self, C_in=1024, C1=256, C2=36, T=8):
        super().__init__()
        self.conv1 = nn.Conv2d(C_in, C1, 3, 1, 1)
        self.mpbn1 = MPBN(C1, T)
        self.lif1 = LIFNeuron(C1)
        self.conv2 = nn.Conv2d(C1, C2, 3, 1, 1)
        self.mpbn2 = MPBN(C2, T)
        self.lif2 = LIFNeuron(C2)

    def forward(self, F, m1=None, m2=None, t=0):
        x1 = self.mpbn1(self.conv1(F), t)
        s1, m1 = self.lif1(x1, m1)
        x2 = self.mpbn2(self.conv2(s1), t)
        s2, m2 = self.lif2(x2, m2)
        return s1, s2, m1, m2


# ============================================================================
# Decoder
# ============================================================================

class Decoder(nn.Module):
    """SNN Decoder: reconstructs features from received spikes.
    
    Uses IF/IHF neurons and a learned spike-to-feature converter
    (FC layer with sigmoid gating over 2T timestep channels).
    """
    def __init__(self, C_out=1024, C1=256, C2=36, T=8):
        super().__init__()
        self.T, self.C1, self.C_out = T, C1, C_out
        self.conv3 = nn.Conv2d(C2, C1, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(C1)
        self.if3 = IFNeuron()
        self.conv4 = nn.Conv2d(C1, C_out, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(C_out)
        self.ihf = IHFNeuron()
        self.converter_fc = nn.Linear(2 * T, 2 * T)

    def forward(self, recv_all, mask):
        m3, m4 = None, None
        Fs, Fm, s3_all = [], [], []
        for t in range(self.T):
            s3, m3 = self.if3(self.bn3(self.conv3(recv_all[t] * mask)), m3)
            sp, m4 = self.ihf(self.bn4(self.conv4(s3)), m4)
            Fs.append(sp)
            Fm.append(m4.clone())
            s3_all.append(s3)
        # Spike-to-feature converter
        il = []
        for t in range(self.T):
            il.append(Fs[t])
            il.append(Fm[t])
        stk = torch.stack(il, dim=1)  # (B, 2T, C, H, W)
        x = stk.permute(0, 2, 3, 4, 1)
        return (x * torch.sigmoid(self.converter_fc(x))).sum(dim=-1), s3_all, Fs


# ============================================================================
# Full Model
# ============================================================================

class SpikeAdaptSC(nn.Module):
    """SpikeAdapt-SC: Content-Adaptive SNN Semantic Communication.
    
    Args:
        C_in: Input feature channels (from ResNet-50 backbone, typically 1024)
        C1: Intermediate encoder channels (256)
        C2: Bottleneck channels transmitted over channel (36)
        T: Number of SNN timesteps (8)
        target_rate: Target fraction of spatial blocks to transmit (0.75)
        channel_type: 'bsc', 'awgn', 'rayleigh', or 'bec'
    """
    def __init__(self, C_in=1024, C1=256, C2=36, T=8,
                 target_rate=0.75, channel_type='bsc'):
        super().__init__()
        self.T = T
        self.encoder = Encoder(C_in, C1, C2, T)
        self.scorer = NoiseAwareScorer(C_spike=C2, hidden=32)
        self.block_mask = LearnedBlockMask(target_rate, 0.5)
        self.decoder = Decoder(C_in, C1, C2, T)
        self.channel = get_channel(channel_type)
        self.channel_type = channel_type

    def forward(self, feat, noise_param=0.0, target_rate_override=None):
        """Forward pass: encode → score → mask → channel → decode.
        
        Args:
            feat: Input features from backbone [B, C_in, H, W]
            noise_param: Channel noise parameter (BER for BSC/BEC, SNR for AWGN/Rayleigh)
            target_rate_override: Override target transmission rate for eval
        
        Returns:
            Fp: Reconstructed features [B, C_in, H, W]
            info: Dict with mask, tx_rate, spike tensors, etc.
        """
        # Encode over T timesteps
        all_S1, all_S2, m1, m2 = [], [], None, None
        for t in range(self.T):
            s1, s2, m1, m2 = self.encoder(feat, m1, m2, t=t)
            all_S1.append(s1)
            all_S2.append(s2)

        # Score importance (noise-aware)
        importance = self.scorer(all_S2, ber=noise_param if isinstance(noise_param, float) else 0.0)
        
        if target_rate_override is not None:
            old = self.block_mask.target_rate
            self.block_mask.target_rate = target_rate_override
            mask, tx = self.block_mask(importance, training=False)
            self.block_mask.target_rate = old
        else:
            mask, tx = self.block_mask(importance, training=self.training)

        # Apply mask and transmit through channel
        recv = [self.channel(self.block_mask.apply_mask(all_S2[t], mask), noise_param)
                for t in range(self.T)]

        # Decode
        Fp, s3_all, s4_all = self.decoder(recv, mask)

        return Fp, {
            'tx_rate': tx.item() if isinstance(tx, torch.Tensor) else tx,
            'mask': mask,
            'importance': importance,
            'all_S1': all_S1,
            'all_S2': all_S2,
            's3_all': s3_all,
            's4_all': s4_all,
        }
