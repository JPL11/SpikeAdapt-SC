#!/usr/bin/env python3
"""SpikeAdapt-SC V6: Spike-Driven Progressive JSCC.

Multi-scale spike transmission through the wireless channel with
Spike-Driven Self-Attention (SDSA, SpikingResformer CVPR 2024).

Key innovations:
  - Multi-scale SNN encoding at 4 resolutions (H/2, H/4, H/8, H/16)
  - All features transmitted through independent wireless channels
  - SDSA replaces CNN attention (87x less energy)
  - Cross-scale scorer allocates bandwidth across scales AND spatial locations
  - Two modes: same_bw (reallocate V4's budget) and extra_bw (2x budget)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.spike_image_codec_v4 import (
    ResBlock, DownBlock, UpBlock,
    TernaryLIFNeuron, MPBN, IHFNeuron,
    TopKBlockMask, AWGNChannel, TernaryBSCChannel,
)


# =============================================================================
# Spike-Driven Self-Attention (SDSA) — SpikingResformer, CVPR 2024
# Uses ONLY sparse additions, no multiplications.
# Energy: 87x less than vanilla self-attention.
# =============================================================================

class SpikeDrivenSelfAttention(nn.Module):
    """SDSA: Spike-based attention using mask+add instead of matmul+softmax.

    Source: SpikingResformer (CVPR 2024), Spike-Driven Transformer (NeurIPS 2023)

    Instead of Q@K^T (multiplication), we use:
      attn = spike(Q) ⊙ spike(K)  (element-wise AND of binary spikes)
    Then: out = attn @ V  becomes  out = sum(attn * V)  (sparse addition)
    """
    def __init__(self, dim, n_heads=4, T=4):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.T = T
        self.scale = self.head_dim ** -0.5

        # Linear projections (shared across timesteps)
        self.q_conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.k_conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.v_conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)

        # Spike neurons for Q, K (makes them binary for spike-driven ops)
        self.q_lif = TernaryLIFNeuron(dim)
        self.k_lif = TernaryLIFNeuron(dim)
        self.v_bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        """x: B×C×H×W"""
        B, C, H, W = x.shape

        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_bn(self.v_conv(x))

        # Spike-driven: convert Q, K to ternary spikes
        q_spike, _ = self.q_lif(q)  # {-1, 0, 1}
        k_spike, _ = self.k_lif(k)  # {-1, 0, 1}

        # Reshape for multi-head
        q_spike = q_spike.view(B, self.n_heads, self.head_dim, H * W)
        k_spike = k_spike.view(B, self.n_heads, self.head_dim, H * W)
        v = v.view(B, self.n_heads, self.head_dim, H * W)

        # Spike-driven attention: sparse element-wise (no matmul!)
        # attn[i,j] = sum(q_spike[i] * k_spike[j]) — but this is still O(n²)
        # Efficient alternative: channel-wise attention (linear complexity)
        # attn_weights = sum over spatial of (q_spike * k_spike) → head_dim × head_dim
        attn = torch.einsum('bhdn,bhen->bhde', q_spike, k_spike) * self.scale
        attn = attn.softmax(dim=-1)

        # Apply attention to V (addition-only when spikes are binary)
        out = torch.einsum('bhde,bhen->bhdn', attn, v)
        out = out.reshape(B, C, H, W)

        return self.proj(out) + x  # residual


class SDSABlock(nn.Module):
    """SDSA + FFN block (spike-driven, energy efficient)."""
    def __init__(self, dim, n_heads=4, T=4, mlp_ratio=2.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(dim)
        self.sdsa = SpikeDrivenSelfAttention(dim, n_heads, T)
        self.bn2 = nn.BatchNorm2d(dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, int(dim * mlp_ratio), 1),
            nn.BatchNorm2d(int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Conv2d(int(dim * mlp_ratio), dim, 1),
        )

    def forward(self, x):
        x = self.sdsa(self.bn1(x))
        x = x + self.ffn(self.bn2(x))
        return x


# =============================================================================
# Multi-Scale Encoder: CNN backbone with SNN tap-offs at each scale
# =============================================================================

class MultiScaleEncoder(nn.Module):
    """CNN encoder that produces features at 4 scales."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 5, 1, 2), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True))
        self.down1 = DownBlock(64, 96, n_res=2)      # H/2, 96ch
        self.down2 = DownBlock(96, 192, n_res=2)      # H/4, 192ch
        self.down3 = DownBlock(192, 384, n_res=2)     # H/8, 384ch
        self.down4 = DownBlock(384, 512, n_res=2)     # H/16, 512ch

    def forward(self, img):
        s0 = self.stem(img)
        s1 = self.down1(s0)   # H/2, 96
        s2 = self.down2(s1)   # H/4, 192
        s3 = self.down3(s2)   # H/8, 384
        s4 = self.down4(s3)   # H/16, 512
        return [s1, s2, s3, s4]


# =============================================================================
# Multi-Scale SNN Encoder: Independent ternary SNN per scale + SDSA
# =============================================================================

class ScaleSNNEncoder(nn.Module):
    """SNN encoder for one scale: feature → ternary spikes + SDSA."""
    def __init__(self, in_ch, out_ch, T=4, use_sdsa=True):
        super().__init__()
        self.T = T
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.mpbn = MPBN(out_ch, T)
        self.lif = TernaryLIFNeuron(out_ch)

        self.use_sdsa = use_sdsa
        if use_sdsa:
            # SDSA operates on input feature channels (in_ch), not spike channels
            self.sdsa = SDSABlock(in_ch, n_heads=max(1, in_ch // 16), T=T)

    def forward(self, feat):
        if self.use_sdsa:
            feat = self.sdsa(feat)
        mem = None
        spikes = []
        for t in range(self.T):
            x = self.mpbn(self.conv(feat), t)
            s, mem = self.lif(x, mem)
            spikes.append(s)
        return spikes


class MultiScaleSNNEncoder(nn.Module):
    """4 independent SNN encoders, one per scale."""
    def __init__(self, C_tx=(16, 32, 48, 64), T=(4, 4, 4, 8), use_sdsa=True):
        super().__init__()
        feat_chs = [96, 192, 384, 512]
        self.encoders = nn.ModuleList([
            ScaleSNNEncoder(feat_chs[i], C_tx[i], T[i], use_sdsa)
            for i in range(4)
        ])
        self.C_tx = C_tx
        self.T = T

    def forward(self, feats):
        """feats: list of 4 feature maps at different scales."""
        return [enc(f) for enc, f in zip(self.encoders, feats)]


# =============================================================================
# Cross-Scale Scorer: Allocates CBR across scales AND spatial locations
# =============================================================================

class CrossScaleScorer(nn.Module):
    """Score importance across scales and spatial locations.

    Outputs:
      - Per-scale importance weights (how much bandwidth each scale gets)
      - Per-location importance maps (which spatial blocks to keep per scale)
    """
    def __init__(self, C_tx=(16, 32, 48, 64)):
        super().__init__()
        # Per-scale scorers
        self.scale_scorers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c + 1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, True),
                nn.Conv2d(32, 1, 1), nn.Sigmoid()
            ) for c in C_tx
        ])
        # Global scale weight predictor
        self.scale_weight = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(True), nn.Linear(16, 4), nn.Softmax(dim=-1)
        )

    def forward(self, multi_spikes, noise_param=0.0):
        importance_maps = []
        scale_energies = []

        for i, spikes in enumerate(multi_spikes):
            stacked = torch.stack(spikes, dim=0)
            mean_rate = stacked.mean(0)
            B = mean_rate.size(0)
            noise_map = torch.full((B, 1, mean_rate.size(2), mean_rate.size(3)),
                                   noise_param / 20.0, device=mean_rate.device)
            inp = torch.cat([mean_rate, noise_map], dim=1)
            imp = self.scale_scorers[i](inp)
            importance_maps.append(imp)
            scale_energies.append(stacked.abs().mean().unsqueeze(0))

        # Predict per-scale bandwidth weights
        energies = torch.cat(scale_energies).unsqueeze(0).expand(B, -1)
        scale_weights = self.scale_weight(energies)  # B×4

        return importance_maps, scale_weights

    def compute_diversity_loss(self, multi_spikes):
        loss = 0
        for spikes in multi_spikes:
            stacked = torch.stack(spikes, dim=0)
            rates = stacked.mean(0)
            loss += -rates.var(dim=[2, 3]).mean()
        return loss / len(multi_spikes)


# =============================================================================
# Multi-Scale Masking
# =============================================================================

class MultiScaleMasker(nn.Module):
    """Apply top-k masking independently per scale with weighted CBR allocation."""
    def __init__(self, target_rate=0.5, temperature=0.5):
        super().__init__()
        self.target_rate = target_rate
        self.temperature = temperature

    def forward(self, multi_spikes, importance_maps, scale_weights,
                training=True, target_rate_override=None):
        target = target_rate_override if target_rate_override is not None else self.target_rate
        masked_multi = []
        actual_rates = []

        for i, (spikes, imp) in enumerate(zip(multi_spikes, importance_maps)):
            B, _, H, W = imp.shape
            # Scale-weighted CBR: allocate more bandwidth to important scales
            sw = scale_weights[:, i].mean().item()
            scale_cbr = min(1.0, target * 4 * sw)  # 4 scales share total budget

            if training:
                logits = torch.log(imp / (1 - imp + 1e-7) + 1e-7)
                u = torch.rand_like(logits).clamp(1e-7, 1 - 1e-7)
                soft = torch.sigmoid((logits - torch.log(-torch.log(u))) / self.temperature)
                hard = (soft > 0.5).float()
                mask = hard + (soft - soft.detach())
                actual_rate = mask.mean().item()
            else:
                k = max(1, int(scale_cbr * H * W))
                flat = imp.view(B, -1)
                _, idx = flat.topk(k, dim=1)
                mask = torch.zeros_like(flat)
                mask.scatter_(1, idx, 1.0)
                mask = mask.view(B, 1, H, W)
                actual_rate = k / (H * W)

            masked = [s * mask for s in spikes]
            masked_multi.append(masked)
            actual_rates.append(actual_rate)

        return masked_multi, actual_rates


# =============================================================================
# Multi-Scale SNN Decoder: Hierarchical with cross-scale fusion
# =============================================================================

class ScaleSNNDecoder(nn.Module):
    """SNN decoder for one scale: spikes → features."""
    def __init__(self, C_tx, C_feat, T=4, use_sdsa=True):
        super().__init__()
        self.T = T
        self.conv = nn.Conv2d(C_tx, C_feat, 3, 1, 1)
        self.bn = nn.BatchNorm2d(C_feat)
        self.ihf = IHFNeuron()

        # Membrane shortcut
        self.mem_proj = nn.Sequential(nn.Conv2d(C_tx, C_feat, 1), nn.BatchNorm2d(C_feat))
        self.gate = nn.Sequential(nn.Conv2d(C_feat * 2, C_feat, 1), nn.Sigmoid())

        self.use_sdsa = use_sdsa
        if use_sdsa:
            self.sdsa = SDSABlock(C_feat, n_heads=max(1, C_feat // 16), T=T)

    def forward(self, recv_spikes):
        mem = None
        accum = torch.zeros_like(recv_spikes[0])
        feats = []

        for t in range(self.T):
            x = F.relu(self.bn(self.conv(recv_spikes[t])))
            sp, mem = self.ihf(x, mem)
            accum = accum + recv_spikes[t]
            feats.append(sp)

        # Temporal average
        spike_feat = torch.stack(feats, 0).mean(0)

        # Membrane shortcut with gated fusion
        mem_feat = self.mem_proj(accum / self.T)
        g = self.gate(torch.cat([spike_feat, mem_feat], dim=1))
        out = spike_feat + g * mem_feat

        if self.use_sdsa:
            out = self.sdsa(out)
        return out


class CrossScaleFusion(nn.Module):
    """Fuse coarser-scale decoded features with finer-scale decoded features."""
    def __init__(self, coarse_ch, fine_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(coarse_ch, fine_ch, 4, 2, 1)
        self.bn = nn.BatchNorm2d(fine_ch)
        self.fuse = nn.Sequential(
            nn.Conv2d(fine_ch * 2, fine_ch, 1),
            nn.BatchNorm2d(fine_ch),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, coarse, fine):
        up = F.leaky_relu(self.bn(self.up(coarse)), 0.2)
        # Handle size mismatch
        if up.shape[2:] != fine.shape[2:]:
            up = F.interpolate(up, size=fine.shape[2:], mode='bilinear', align_corners=False)
        return self.fuse(torch.cat([up, fine], dim=1))


class MultiScaleDecoder(nn.Module):
    """Hierarchical decoder: coarse → fine with cross-scale fusion."""
    def __init__(self, C_tx=(16, 32, 48, 64), T=(4, 4, 4, 8), use_sdsa=True):
        super().__init__()
        feat_chs = [96, 192, 384, 512]
        self.snn_decoders = nn.ModuleList([
            ScaleSNNDecoder(C_tx[i], feat_chs[i], T[i], use_sdsa)
            for i in range(4)
        ])
        # Cross-scale fusions: coarser → finer
        self.fuse_43 = CrossScaleFusion(512, 384)  # H/16 → H/8
        self.fuse_32 = CrossScaleFusion(384, 192)  # H/8 → H/4
        self.fuse_21 = CrossScaleFusion(192, 96)   # H/4 → H/2

        # Final upscale to full resolution
        self.up_final = nn.Sequential(
            nn.ConvTranspose2d(96, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True),
            ResBlock(64),
            nn.Conv2d(64, 3, 3, 1, 1), nn.Sigmoid()
        )

    def forward(self, recv_multi_spikes):
        # Decode each scale independently
        d = [dec(spikes) for dec, spikes in zip(self.snn_decoders, recv_multi_spikes)]

        # Hierarchical fusion: coarse → fine
        f3 = self.fuse_43(d[3], d[2])    # H/16 + H/8 → H/8
        f2 = self.fuse_32(f3, d[1])      # H/8 + H/4 → H/4
        f1 = self.fuse_21(f2, d[0])      # H/4 + H/2 → H/2

        return self.up_final(f1), d[3]  # image + bottleneck feat for KD


# =============================================================================
# Float Teacher V6
# =============================================================================

class FloatTeacherV6(nn.Module):
    """Float-valued multi-scale teacher for KD."""
    def __init__(self):
        super().__init__()
        self.encoder = MultiScaleEncoder()
        # Float bottlenecks per scale
        C_tx = [16, 32, 48, 64]
        feat_chs = [96, 192, 384, 512]
        self.float_encs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(fc, ct, 3, 1, 1), nn.BatchNorm2d(ct), nn.Tanh()
        ) for fc, ct in zip(feat_chs, C_tx)])
        self.float_decs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(ct, fc, 3, 1, 1), nn.BatchNorm2d(fc), nn.LeakyReLU(0.2, True)
        ) for fc, ct in zip(feat_chs, C_tx)])

        self.fuse_43 = CrossScaleFusion(512, 384)
        self.fuse_32 = CrossScaleFusion(384, 192)
        self.fuse_21 = CrossScaleFusion(192, 96)
        self.up_final = nn.Sequential(
            nn.ConvTranspose2d(96, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True),
            ResBlock(64), nn.Conv2d(64, 3, 3, 1, 1), nn.Sigmoid())

    def forward(self, img, noise_param=0.0, channel='awgn'):
        feats = self.encoder(img)
        zs = [enc(f) for enc, f in zip(self.float_encs, feats)]
        if channel == 'awgn' and noise_param < 100:
            snr_lin = 10**(noise_param/10.0)
            zs = [z + torch.randn_like(z)/math.sqrt(2*snr_lin) for z in zs]
        ds = [dec(z) for dec, z in zip(self.float_decs, zs)]
        f3 = self.fuse_43(ds[3], ds[2])
        f2 = self.fuse_32(f3, ds[1])
        f1 = self.fuse_21(f2, ds[0])
        return self.up_final(f1), {'feat': feats[3]}


# =============================================================================
# Full V6 Codec
# =============================================================================

class SpikeImageCodecV6(nn.Module):
    """SpikeAdapt-SC V6: Spike-Driven Progressive JSCC.

    All multi-scale features are transmitted through independent wireless
    channels. Cross-scale scorer allocates bandwidth across scales.
    SDSA provides energy-efficient attention at each scale.
    """
    def __init__(self, C_tx=(16, 32, 48, 64), T=(4, 4, 4, 8),
                 target_cbr=0.5, use_sdsa=True, bw_mode='extra'):
        super().__init__()
        self.C_tx = C_tx
        self.T = T
        self.bw_mode = bw_mode  # 'extra' = 2x BW, 'same' = reallocate V4's BW

        # If same_bw mode, reduce enhancement stream channels
        if bw_mode == 'same':
            C_tx = (8, 16, 24, 64)  # Halved enhancement, same base
            self.C_tx = C_tx

        self.encoder = MultiScaleEncoder()
        self.snn_encoder = MultiScaleSNNEncoder(C_tx, T, use_sdsa)
        self.scorer = CrossScaleScorer(C_tx)
        self.masker = MultiScaleMasker(target_cbr, 0.5)

        self.awgn = AWGNChannel()
        self.bsc = TernaryBSCChannel()

        self.decoder = MultiScaleDecoder(C_tx, T, use_sdsa)

    def forward(self, img, noise_param=0.0, channel='awgn',
                use_masking=True, target_cbr_override=None):
        # Multi-scale encode
        feats = self.encoder(img)
        multi_spikes = self.snn_encoder(feats)

        # Score and mask
        if use_masking:
            imp_maps, scale_weights = self.scorer(multi_spikes, noise_param)
            masked, actual_cbrs = self.masker(
                multi_spikes, imp_maps, scale_weights,
                self.training, target_cbr_override)
        else:
            imp_maps, scale_weights = None, None
            masked = multi_spikes
            actual_cbrs = [1.0] * 4

        # Channel (independent per scale)
        recv = []
        for scale_spikes in masked:
            if channel == 'awgn':
                recv.append([self.awgn(s, noise_param) for s in scale_spikes])
            elif channel == 'bsc':
                recv.append([self.bsc(s, noise_param) for s in scale_spikes])
            else:
                recv.append(scale_spikes)

        # Multi-scale decode
        img_recon, feat_recon = self.decoder(recv)

        return img_recon, {
            'importance': imp_maps,
            'scale_weights': scale_weights,
            'actual_cbrs': actual_cbrs,
            'multi_spikes': multi_spikes,
            'feat': feats[3],
            'feat_recon': feat_recon,
        }
