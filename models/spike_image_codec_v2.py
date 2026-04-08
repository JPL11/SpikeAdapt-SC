#!/usr/bin/env python3
"""SpikeAdapt-SC Image Codec v2: Improved architecture for higher PSNR.

Improvements over v1:
  1. Residual blocks in encoder/decoder
  2. Wider channels (96/192/384 vs 64/128/256)
  3. T=8 timesteps for better spike-rate resolution
  4. Stage 4 joint fine-tuning support

NOTE: No encoder→decoder skip connections.
  In a JSCC system ALL information must traverse the channel.
  Skip connections would bypass the SNN bottleneck + channel,
  making masking and noise irrelevant.

Architecture:
  Image (3×H×W)
   → ResNet Encoder (strided convs + residual blocks, 16× downscale)
   → SNN Bottleneck (LIF + MPBN, binary spikes, T=8)
   → Content Scorer (spike-rate variance, noise-conditioned)
   → Top-K Block Mask (binary, exact CBR control)
   → Channel (AWGN or BSC)
   → SNN Decoder (IHF + spike converter)
   → ResNet Decoder (transposed convs + residual blocks)
   → Reconstructed Image
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# Building Blocks
# =============================================================================

class ResBlock(nn.Module):
    """Residual block with two convolutions."""
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        return self.act(self.net(x) + x)


class DownBlock(nn.Module):
    """Strided convolution + residual blocks for encoder."""
    def __init__(self, in_ch, out_ch, n_res=2):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, 2, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res = nn.Sequential(*[ResBlock(out_ch) for _ in range(n_res)])
    
    def forward(self, x):
        x = self.down(x)
        x = self.res(x)
        return x


class UpBlock(nn.Module):
    """Transposed convolution + residual blocks for decoder, with optional skip."""
    def __init__(self, in_ch, out_ch, skip_ch=0, n_res=2):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Fuse skip connection
        if skip_ch > 0:
            self.skip_conv = nn.Sequential(
                nn.Conv2d(out_ch + skip_ch, out_ch, 1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.skip_conv = None
        self.res = nn.Sequential(*[ResBlock(out_ch) for _ in range(n_res)])
    
    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None and self.skip_conv is not None:
            # Handle size mismatches (pad if needed)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = self.skip_conv(torch.cat([x, skip], dim=1))
        x = self.res(x)
        return x


# =============================================================================
# SNN Primitives
# =============================================================================

class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire with surrogate gradient."""
    def __init__(self, channels=None):
        super().__init__()
        self.beta = nn.Parameter(torch.full((channels,), 0.5)) if channels else nn.Parameter(torch.tensor(0.5))
        self.slope = nn.Parameter(torch.tensor(5.0))

    def forward(self, x, mem=None):
        if mem is None:
            mem = torch.zeros_like(x)
        beta = torch.sigmoid(self.beta)
        if beta.dim() == 1:
            beta = beta.view(1, -1, 1, 1)
        mem = beta * mem + x
        spike = self._surrogate(mem - 1.0)
        mem = mem * (1 - spike)
        return spike, mem

    def _surrogate(self, x):
        slope = torch.clamp(self.slope, 1.0, 20.0)
        return (x >= 0).float() + (torch.sigmoid(slope * x) - (x >= 0).float()).detach() + \
               torch.sigmoid(slope * x) - torch.sigmoid(slope * x).detach()


class MPBN(nn.Module):
    """Membrane Potential Batch Normalization (per-timestep BN)."""
    def __init__(self, channels, T):
        super().__init__()
        self.bns = nn.ModuleList([nn.BatchNorm2d(channels) for _ in range(T)])

    def forward(self, x, t):
        return self.bns[t](x)


class IHFNeuron(nn.Module):
    """Integrate-and-Hold-Fire neuron for decoder."""
    def forward(self, x, mem=None):
        if mem is None:
            mem = torch.zeros_like(x)
        mem = mem + x
        spike = (mem >= 1.0).float()
        return spike, mem


# =============================================================================
# Improved Encoder
# =============================================================================

class ImageEncoderV2(nn.Module):
    """ResNet encoder: 3×H×W → C_feat×H/16×W/16"""
    def __init__(self, C_feat=384):
        super().__init__()
        self.net = nn.Sequential(
            # Input stem
            nn.Conv2d(3, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 64×H×W → 96×H/2×W/2
        self.down1 = DownBlock(64, 96, n_res=2)
        # 96×H/2 → 192×H/4
        self.down2 = DownBlock(96, 192, n_res=2)
        # 192×H/4 → 384×H/8
        self.down3 = DownBlock(192, 384, n_res=2)
        # 384×H/8 → C_feat×H/16
        self.down4 = DownBlock(384, C_feat, n_res=2)
    
    def forward(self, img):
        x = self.net(img)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        return x


# =============================================================================
# Improved Decoder  
# =============================================================================

class ImageDecoderV2(nn.Module):
    """ResNet decoder (no skip connections): C_feat×H/16 → 3×H×W"""
    def __init__(self, C_feat=384):
        super().__init__()
        # C_feat×H/16 → 384×H/8
        self.up1 = UpBlock(C_feat, 384, skip_ch=0, n_res=2)
        # 384×H/8 → 192×H/4
        self.up2 = UpBlock(384, 192, skip_ch=0, n_res=2)
        # 192×H/4 → 96×H/2
        self.up3 = UpBlock(192, 96, skip_ch=0, n_res=2)
        # 96×H/2 → 64×H
        self.up4 = UpBlock(96, 64, skip_ch=0, n_res=1)
        # Final: 64×H → 3×H
        self.final = nn.Sequential(
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, feat):
        x = self.up1(feat)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.final(x)


# =============================================================================
# SNN Bottleneck Encoder (same as v1 but with configurable T and wider)
# =============================================================================

class SNNBottleneckEncoderV2(nn.Module):
    """Feature → Binary spikes via LIF neurons."""
    def __init__(self, C_feat=384, C_tx=64, T=8):
        super().__init__()
        self.T = T
        self.conv1 = nn.Conv2d(C_feat, 192, 3, 1, 1)
        self.mpbn1 = MPBN(192, T)
        self.lif1 = LIFNeuron(192)
        self.conv2 = nn.Conv2d(192, C_tx, 3, 1, 1)
        self.mpbn2 = MPBN(C_tx, T)
        self.lif2 = LIFNeuron(C_tx)

    def forward(self, feat):
        m1, m2 = None, None
        spikes = []
        for t in range(self.T):
            x1 = self.mpbn1(self.conv1(feat), t)
            s1, m1 = self.lif1(x1, m1)
            x2 = self.mpbn2(self.conv2(s1), t)
            s2, m2 = self.lif2(x2, m2)
            spikes.append(s2)
        return spikes


# =============================================================================
# SNN Bottleneck Decoder
# =============================================================================

class TemporalAttention(nn.Module):
    """Attention over timesteps for SNN decoder aggregation.
    
    Source: Temporal Model Calibration (NeurIPS 2023), hybrid coding.
    Instead of fixed linear mixing, learns to weight each timestep
    (spike + membrane) adaptively per spatial location.
    """
    def __init__(self, C_feat, T):
        super().__init__()
        self.slots = 2 * T  # T spike frames + T membrane frames
        self.attn_net = nn.Sequential(
            nn.Conv2d(C_feat, C_feat // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C_feat // 4, 1, 1),
        )
    
    def forward(self, spike_features, mem_features):
        # spike_features: list of T tensors, each B×C×H×W
        # mem_features: list of T tensors, each B×C×H×W
        all_feats = spike_features + mem_features  # 2T items
        stacked = torch.stack(all_feats, dim=1)  # B×2T×C×H×W
        scores = []
        for t in range(self.slots):
            s = self.attn_net(stacked[:, t])  # B×1×H×W
            scores.append(s)
        weights = torch.cat(scores, dim=1).softmax(dim=1)  # B×2T×H×W
        weights = weights.unsqueeze(2)  # B×2T×1×H×W
        return (stacked * weights).sum(dim=1)  # B×C×H×W


class SNNBottleneckDecoderV2(nn.Module):
    """Received spikes → Feature map reconstruction with temporal attention."""
    def __init__(self, C_feat=384, C_tx=64, T=8):
        super().__init__()
        self.T = T
        self.conv3 = nn.Conv2d(C_tx, 192, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(192)
        self.conv4 = nn.Conv2d(192, C_feat, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(C_feat)
        self.ihf = IHFNeuron()
        self.temporal_attn = TemporalAttention(C_feat, T)

    def forward(self, recv_spikes):
        m3, m4 = None, None
        Fs, Fm = [], []
        for t in range(self.T):
            s3 = F.relu(self.bn3(self.conv3(recv_spikes[t])))
            sp, m4 = self.ihf(self.bn4(self.conv4(s3)), m4)
            Fs.append(sp)
            Fm.append(m4.clone())
        return self.temporal_attn(Fs, Fm)


# =============================================================================
# Content Scorer (spike-rate variance — best performing)
# =============================================================================

class ContentScorer(nn.Module):
    """Content-aware importance scorer using spike-rate statistics.
    
    Uses spike rate + variance as importance signal with noise conditioning.
    Empirically outperforms CBAM attention (+2.0 dB peak gap vs -0.97 dB).
    """
    def __init__(self, C_tx=64, hidden=64):
        super().__init__()
        self.C_tx = C_tx
        self.content_net = nn.Sequential(
            nn.Conv2d(C_tx + 1, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )
        self.noise_gate = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.noise_gate[-1].weight)
        nn.init.zeros_(self.noise_gate[-1].bias)
    
    def forward(self, spikes, noise_param):
        B = spikes[0].size(0)
        spike_stack = torch.stack(spikes, dim=0)
        rate = spike_stack.float().mean(dim=0)
        rate_var = rate.var(dim=1, keepdim=True)
        content_input = torch.cat([rate, rate_var], dim=1)
        logits = self.content_net(content_input)
        noise_input = torch.full((B, 1), float(noise_param), device=rate.device)
        noise_bias = self.noise_gate(noise_input)
        logits = logits + noise_bias.unsqueeze(-1).unsqueeze(-1)
        importance = torch.sigmoid(logits)
        return importance
    
    def compute_diversity_loss(self, spikes, noise_high=0.0, noise_low=0.3):
        imp_h = self.forward(spikes, noise_high)
        imp_l = self.forward(spikes, noise_low)
        return -torch.mean((imp_h - imp_l).pow(2))


class TopKBlockMask(nn.Module):
    """Binary spatial block masking with top-k selection."""
    def __init__(self, target_rate=0.5, temperature=0.5):
        super().__init__()
        self.target_rate = target_rate
        self.temperature = temperature
    
    def forward(self, spikes, importance, training=True, target_rate_override=None):
        B, _, H, W = importance.shape
        target = target_rate_override if target_rate_override is not None else self.target_rate
        
        if training:
            logits = torch.log(importance / (1 - importance + 1e-7) + 1e-7)
            u = torch.rand_like(logits).clamp(1e-7, 1 - 1e-7)
            soft = torch.sigmoid((logits - torch.log(-torch.log(u))) / self.temperature)
            hard = (soft > 0.5).float()
            mask = hard + (soft - soft.detach())
            actual_rate = mask.mean().item()
        else:
            k = max(1, int(target * H * W))
            flat = importance.view(B, -1)
            _, idx = flat.topk(k, dim=1)
            mask = torch.zeros_like(flat)
            mask.scatter_(1, idx, 1.0)
            mask = mask.view(B, 1, H, W)
            actual_rate = k / (H * W)
        
        masked = [s * mask for s in spikes]
        return masked, actual_rate


# =============================================================================
# Channel Models
# =============================================================================

class AWGNChannel(nn.Module):
    def forward(self, x, snr_db):
        if snr_db >= 100:
            return x
        snr_lin = 10 ** (snr_db / 10.0)
        noise_std = 1.0 / math.sqrt(2 * snr_lin)
        return x + torch.randn_like(x) * noise_std

class BSCChannel(nn.Module):
    def forward(self, x, ber):
        if ber <= 0:
            return x
        flip = (torch.rand_like(x) < ber).float()
        noisy = x * (1 - flip) + (1 - x) * flip
        if self.training:
            return x + (noisy - x).detach()  # STE
        return noisy


# =============================================================================
# Stereo Cross-Attention (Improvement #1)
# Source: ECSIC (WACV 2024), Li et al. Hyperprior JSCC (TCCN 2024)
# =============================================================================

class StereoCrossAttention(nn.Module):
    """Epipolar cross-attention for stereo feature fusion.
    
    Fuses information from the other-view decoded features into the
    current view. Attention is restricted to epipolar lines (same row)
    for rectified stereo pairs, keeping complexity O(W²) per row.
    """
    def __init__(self, channels=384, n_heads=4):
        super().__init__()
        self.channels = channels
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1),
            nn.Sigmoid(),
        )
        self.norm = nn.BatchNorm2d(channels)
    
    def forward(self, feat_self, feat_other):
        B, C, H, W = feat_self.shape
        nh, hd = self.n_heads, self.head_dim
        
        # Project queries from self, keys/values from other view
        q = self.q(feat_self).view(B, nh, hd, H, W)    # B×nh×hd×H×W
        k = self.k(feat_other).view(B, nh, hd, H, W)
        v = self.v(feat_other).view(B, nh, hd, H, W)
        
        # Row-wise attention (epipolar constraint: same row in stereo pair)
        # q/k/v: work along W dimension for each row h
        # Reshape: B×nh×H × hd×W then attn along W
        q = q.permute(0, 1, 3, 2, 4).reshape(B * nh * H, hd, W)
        k = k.permute(0, 1, 3, 2, 4).reshape(B * nh * H, hd, W)
        v = v.permute(0, 1, 3, 2, 4).reshape(B * nh * H, hd, W)
        
        attn = torch.bmm(q.transpose(1, 2), k) / math.sqrt(hd)  # (B*nh*H)×W×W
        attn = attn.softmax(dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2))  # (B*nh*H)×hd×W
        
        out = out.reshape(B, nh, H, hd, W).permute(0, 1, 3, 2, 4).reshape(B, C, H, W)
        out = self.out_proj(out)
        
        # Gated fusion: let model learn how much stereo info to blend
        gate = self.gate(torch.cat([feat_self, out], dim=1))
        fused = feat_self + gate * out
        return self.norm(fused)


# =============================================================================
# Decoder Refinement Network (Improvement #2)
# Source: Post-processing quality enhancement (JSCC literature 2024)
# =============================================================================

class DecoderRefinementNet(nn.Module):
    """Lightweight post-processing to recover high-frequency details.
    
    Uses residual learning: predicts a residual correction to the
    initial reconstruction. Runs at receiver only, ~0.5M params.
    """
    def __init__(self, n_blocks=4, channels=64):
        super().__init__()
        self.head = nn.Conv2d(3, channels, 3, 1, 1)
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(n_blocks)])
        self.tail = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, 3, 3, 1, 1),
        )
    
    def forward(self, x_coarse):
        feat = self.head(x_coarse)
        feat = self.body(feat)
        residual = self.tail(feat)
        return torch.sigmoid(x_coarse + residual)


# =============================================================================
# Full Image Codec V2 (with stereo, refinement, temporal attention)
# =============================================================================

class SpikeImageCodecV2(nn.Module):
    """SpikeAdapt-SC Image Codec V2 with three enhancements:
    
    1. Stereo cross-attention (exploit left-right correlation)
    2. Decoder refinement net (residual post-processing)
    3. Temporal attention (adaptive timestep aggregation in SNN decoder)
    
    Architecture:
      Image → Encoder → SNN → Scorer/Mask → Channel → SNN Decoder
        → [StereoCrossAttention] → Image Decoder → [RefinementNet] → Output
    """
    def __init__(self, C_feat=384, C_tx=64, T=8, target_cbr=0.5, stereo=False):
        super().__init__()
        self.T = T
        self.C_tx = C_tx
        self.target_cbr = target_cbr
        self.stereo = stereo
        
        # Encoder path (transmitter) — shared for left/right
        self.img_encoder = ImageEncoderV2(C_feat)
        self.snn_encoder = SNNBottleneckEncoderV2(C_feat, C_tx, T)
        
        # Scoring and masking
        self.scorer = ContentScorer(C_tx, hidden=64)
        self.masker = TopKBlockMask(target_cbr, 0.5)
        
        # Channel
        self.awgn = AWGNChannel()
        self.bsc = BSCChannel()
        
        # Decoder path (receiver) — shared for left/right
        self.snn_decoder = SNNBottleneckDecoderV2(C_feat, C_tx, T)
        self.img_decoder = ImageDecoderV2(C_feat)
        
        # Stereo cross-attention (receiver-side fusion)
        if stereo:
            self.stereo_attn = StereoCrossAttention(C_feat, n_heads=4)
        
        # Decoder refinement net (receiver-side post-processing)
        self.refinement = DecoderRefinementNet(n_blocks=4, channels=64)
    
    def _encode_and_transmit(self, img, noise_param, channel,
                              use_masking, target_cbr_override):
        """Shared encode + mask + channel for one view."""
        feat = self.img_encoder(img)
        spikes = self.snn_encoder(feat)
        
        if use_masking:
            importance = self.scorer(spikes, noise_param)
            masked_spikes, actual_cbr = self.masker(
                spikes, importance, training=self.training,
                target_rate_override=target_cbr_override
            )
        else:
            importance = None
            masked_spikes = spikes
            actual_cbr = 1.0
        
        if channel == 'awgn':
            recv = [self.awgn(s, noise_param) for s in masked_spikes]
        elif channel == 'bsc':
            recv = [self.bsc(s, noise_param) for s in masked_spikes]
        else:
            recv = masked_spikes
        
        feat_recon = self.snn_decoder(recv)
        return feat_recon, importance, actual_cbr, spikes
    
    def forward(self, img, noise_param=0.0, channel='awgn',
                use_masking=True, target_cbr_override=None,
                img_right=None, use_refinement=True):
        # Encode and transmit left view
        feat_left, importance, actual_cbr, spikes = self._encode_and_transmit(
            img, noise_param, channel, use_masking, target_cbr_override
        )
        
        # Stereo mode: encode right view and fuse
        if self.stereo and img_right is not None:
            feat_right, _, _, _ = self._encode_and_transmit(
                img_right, noise_param, channel, use_masking, target_cbr_override
            )
            # Cross-attention fusion (receiver side)
            feat_left = self.stereo_attn(feat_left, feat_right)
        
        # Image decoder
        img_recon = self.img_decoder(feat_left)
        
        # Refinement (receiver-side post-processing)
        if use_refinement:
            img_recon = self.refinement(img_recon)
        
        return img_recon, {
            'importance': importance,
            'actual_cbr': actual_cbr,
            'spikes': spikes,
        }
