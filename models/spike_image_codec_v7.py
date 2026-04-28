#!/usr/bin/env python3
"""SpikeAdapt-SC V7: V6 + Lightweight Hyperprior + Multi-bit Encoding.

Stage B: Hyperprior side-channel for reconstruction hints
Stage C: Multi-bit membrane potential encoding (replaces ternary)

V7 = V6 architecture + HyperEncoder/HyperDecoder + optional MultibitLIF
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.spike_image_codec_v6 import (
    SpikeImageCodecV6, MultiScaleEncoder, MultiScaleSNNEncoder,
    CrossScaleScorer, MultiScaleMasker, MultiScaleDecoder,
    CrossScaleFusion, SpikeDrivenSelfAttention, SDSABlock, ScaleSNNEncoder,
    ScaleSNNDecoder,
)
from models.spike_image_codec_v4 import (
    ResBlock, TernaryLIFNeuron, MPBN, IHFNeuron,
    TopKBlockMask, AWGNChannel, TernaryBSCChannel,
)


# =============================================================================
# Stage B: Lightweight Hyperprior
# =============================================================================

class HyperEncoder(nn.Module):
    """Compress bottleneck spike statistics into compact side information.

    Per scale: mean spike rate + spatial stats → small latent (further downsampled).
    The hyperprior captures image-level structure that helps the decoder.
    """
    def __init__(self, C_tx=(16, 32, 48, 64), C_hyper=(4, 8, 12, 16)):
        super().__init__()
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ct, ch, 3, 1, 1),
                nn.BatchNorm2d(ch),
                nn.LeakyReLU(0.2, True),
                nn.AvgPool2d(2),  # Downsample 2× → half spatial resolution
                nn.Conv2d(ch, ch, 3, 1, 1),
                nn.BatchNorm2d(ch),
                nn.Tanh(),  # Bound to [-1, 1] for channel compatibility
            )
            for ct, ch in zip(C_tx, C_hyper)
        ])

    def forward(self, multi_spikes):
        """Encode spike statistics into compact hyperprior latents.

        Args:
            multi_spikes: list of 4 lists of spike tensors (T timesteps each)
        Returns:
            hyper_latents: list of 4 tensors at half spatial resolution
        """
        latents = []
        for i, spikes in enumerate(multi_spikes):
            # Average spikes across timesteps → B×C_tx×H×W
            mean_rate = torch.stack(spikes, dim=0).mean(0)
            latents.append(self.encoders[i](mean_rate))
        return latents


class HyperDecoder(nn.Module):
    """Decode hyperprior latents into reconstruction hints for main decoder.

    Predicts per-location (mean, scale) that the main decoder uses to
    better reconstruct from the sparse received spikes.
    """
    def __init__(self, C_tx=(16, 32, 48, 64), C_hyper=(4, 8, 12, 16)):
        super().__init__()
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(ch, ct, 4, 2, 1),  # Upsample 2×
                nn.BatchNorm2d(ct),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ct, ct, 3, 1, 1),
                nn.BatchNorm2d(ct),
            )
            for ct, ch in zip(C_tx, C_hyper)
        ])

    def forward(self, hyper_latents):
        """Decode hyperprior latents into per-scale reconstruction hints.

        Returns:
            hints: list of 4 tensors at original spatial resolution
        """
        return [dec(z) for dec, z in zip(self.decoders, hyper_latents)]


class HyperpriorBSC(nn.Module):
    """Quantize hyperprior latent to ternary and send through BSC."""
    def __init__(self):
        super().__init__()
        self.bsc = TernaryBSCChannel()

    def forward(self, z, ber):
        """Ternary quantize + BSC.

        z is Tanh-bounded [-1, 1]. Quantize to {-1, 0, 1} then BSC.
        """
        # Ternary quantize via double-threshold with STE
        z_ternary = torch.zeros_like(z)
        z_ternary[z > 0.33] = 1.0
        z_ternary[z < -0.33] = -1.0
        z_ternary = z + (z_ternary - z).detach()  # STE

        if ber > 0:
            z_ternary = self.bsc(z_ternary, ber)
        return z_ternary


class HyperpriorAWGN(nn.Module):
    """Send hyperprior latent through AWGN (continuous)."""
    def __init__(self):
        super().__init__()
        self.awgn = AWGNChannel()

    def forward(self, z, snr_db):
        return self.awgn(z, snr_db)


# =============================================================================
# Stage C: Multi-bit Membrane Encoding
# =============================================================================

class MultibitLIFNeuron(nn.Module):
    """LIF neuron with multi-level quantized output.

    Instead of ternary {-1, 0, 1}, outputs one of n_levels learnable values.
    Provides log2(n_levels) bits/symbol vs 1.58 bits for ternary.

    n_levels=4 → 2 bits/symbol
    n_levels=8 → 3 bits/symbol
    n_levels=16 → 4 bits/symbol
    """
    def __init__(self, channels, n_levels=8):
        super().__init__()
        self.n_levels = n_levels
        self.beta_raw = nn.Parameter(torch.zeros(channels))
        self.slope = nn.Parameter(torch.tensor(5.0))
        # Learnable quantization levels, initialized uniformly in [-1, 1]
        self.levels = nn.Parameter(torch.linspace(-1, 1, n_levels))

    def forward(self, x, mem=None):
        if mem is None:
            mem = torch.zeros_like(x)
        beta = torch.sigmoid(self.beta_raw)
        if beta.dim() == 1:
            beta = beta.view(1, -1, 1, 1)
        mem = beta * mem + x

        # Multi-level quantization via nearest-neighbor + STE
        # levels: (n_levels,), mem: (B, C, H, W)
        levels = self.levels.view(1, 1, 1, 1, -1)  # broadcast
        mem_exp = mem.unsqueeze(-1)  # B×C×H×W×1
        dists = (mem_exp - levels).abs()  # B×C×H×W×n_levels
        idx = dists.argmin(dim=-1)  # B×C×H×W
        quantized = self.levels[idx]  # B×C×H×W

        # STE: forward uses quantized, backward uses continuous
        output = mem + (quantized - mem).detach()

        # Soft reset: subtract quantized value from membrane
        mem = mem - quantized.detach()

        return output, mem


class MultilevelBSC(nn.Module):
    """BSC for multi-level symbols.

    Each symbol independently has probability `ser` (symbol error rate) of being
    corrupted to a random level. Analogous to BSC but for multi-level signals.
    """
    def forward(self, x, ser, levels):
        """
        Args:
            x: quantized symbols (values from levels)
            ser: symbol error rate (0-1)
            levels: the quantization levels tensor
        """
        if ser <= 0:
            return x
        n_levels = levels.shape[0]
        flip_mask = (torch.rand_like(x) < ser).float()
        random_idx = torch.randint(0, n_levels, x.shape, device=x.device)
        random_vals = levels[random_idx]
        noisy = x * (1 - flip_mask) + random_vals * flip_mask
        if self.training:
            return x + (noisy - x).detach()  # STE
        return noisy


# =============================================================================
# V7 Codec: V6 + Hyperprior + Multi-bit
# =============================================================================

class SpikeImageCodecV7(nn.Module):
    """SpikeAdapt-SC V7: V6 + Hyperprior + optional Multi-bit encoding.

    Modes:
      - 'ternary': V6 ternary spikes + hyperprior (Stage B)
      - 'multibit': Multi-bit LIF + hyperprior (Stage C)
    """
    def __init__(self, C_tx=(16, 32, 48, 64), T=(4, 4, 4, 8),
                 C_hyper=(4, 8, 12, 16),
                 target_cbr=0.065, use_sdsa=True, bw_mode='extra',
                 encoding_mode='ternary', n_levels=8):
        super().__init__()
        self.C_tx = C_tx
        self.C_hyper = C_hyper
        self.T = T
        self.encoding_mode = encoding_mode
        self.n_levels = n_levels

        # Core V6 components
        self.encoder = MultiScaleEncoder()
        self.snn_encoder = MultiScaleSNNEncoder(C_tx, T, use_sdsa)
        self.scorer = CrossScaleScorer(C_tx)
        self.masker = MultiScaleMasker(target_cbr, 0.5)

        # Channels
        self.awgn = AWGNChannel()
        self.bsc = TernaryBSCChannel()

        # V6 decoder
        self.decoder = MultiScaleDecoder(C_tx, T, use_sdsa)

        # Stage B: Hyperprior
        self.hyper_encoder = HyperEncoder(C_tx, C_hyper)
        self.hyper_decoder = HyperDecoder(C_tx, C_hyper)
        self.hyper_bsc = HyperpriorBSC()
        self.hyper_awgn = HyperpriorAWGN()

        # Per-scale fusion: combine received spikes + hyperprior hints
        feat_chs = [96, 192, 384, 512]
        self.hint_fuse = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ct * 2, ct, 1),  # Concatenate received + hint
                nn.BatchNorm2d(ct),
                nn.LeakyReLU(0.2, True),
            )
            for ct in C_tx
        ])

        # Stage C: Multi-bit neurons (optional, replaces SNN encoder)
        if encoding_mode == 'multibit':
            self.multibit_encoders = nn.ModuleList([
                MultibitScaleEncoder(feat_ch, ct, T_i, n_levels, use_sdsa)
                for feat_ch, ct, T_i in zip([96, 192, 384, 512], C_tx, T)
            ])
            self.multilevel_bsc = MultilevelBSC()

    def forward(self, img, noise_param=0.0, channel='awgn',
                use_masking=True, target_cbr_override=None,
                use_hyperprior=True):
        # Encode features
        feats = self.encoder(img)

        # SNN or Multi-bit encoding
        if self.encoding_mode == 'multibit':
            multi_spikes = [enc(f) for enc, f in zip(self.multibit_encoders, feats)]
        else:
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

        # Main channel
        recv = []
        for scale_spikes in masked:
            if channel == 'awgn':
                recv.append([self.awgn(s, noise_param) for s in scale_spikes])
            elif channel == 'bsc':
                if self.encoding_mode == 'multibit':
                    recv.append([self.multilevel_bsc(s, noise_param,
                                self.multibit_encoders[0].lif.levels) for s in scale_spikes])
                else:
                    recv.append([self.bsc(s, noise_param) for s in scale_spikes])
            else:
                recv.append(scale_spikes)

        # Hyperprior side channel
        hyper_hints = None
        if use_hyperprior:
            hyper_latents = self.hyper_encoder(multi_spikes)

            # Send hyperprior through channel
            if channel == 'awgn':
                hyper_recv = [self.hyper_awgn(z, noise_param) for z in hyper_latents]
            elif channel == 'bsc':
                hyper_recv = [self.hyper_bsc(z, noise_param) for z in hyper_latents]
            else:
                hyper_recv = hyper_latents

            # Decode hyperprior hints
            hyper_hints = self.hyper_decoder(hyper_recv)

            # Fuse hints with received spikes (per timestep, per scale)
            fused_recv = []
            for i, (scale_spikes, hint) in enumerate(zip(recv, hyper_hints)):
                # Handle size mismatch from downsampling/upsampling
                fused_scale = []
                for t, s in enumerate(scale_spikes):
                    if hint.shape[2:] != s.shape[2:]:
                        hint_resized = F.interpolate(hint, size=s.shape[2:],
                                                    mode='bilinear', align_corners=False)
                    else:
                        hint_resized = hint
                    fused = self.hint_fuse[i](torch.cat([s, hint_resized], dim=1))
                    fused_scale.append(fused)
                fused_recv.append(fused_scale)
            recv = fused_recv

        # Decode
        img_recon, feat_recon = self.decoder(recv)

        # Compute hyperprior bandwidth overhead
        hyper_bw = 0
        if use_hyperprior and hyper_hints is not None:
            for i, lat in enumerate(hyper_latents if use_hyperprior else []):
                hyper_bw += lat.numel() / lat.size(0)  # per-sample

        return img_recon, {
            'importance': imp_maps,
            'scale_weights': scale_weights,
            'actual_cbrs': actual_cbrs,
            'multi_spikes': multi_spikes,
            'feat': feats[3],
            'feat_recon': feat_recon,
            'hyper_bw': hyper_bw,
        }


class MultibitScaleEncoder(nn.Module):
    """Multi-bit SNN encoder for one scale."""
    def __init__(self, in_ch, out_ch, T=4, n_levels=8, use_sdsa=True):
        super().__init__()
        self.T = T
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.mpbn = MPBN(out_ch, T)
        self.lif = MultibitLIFNeuron(out_ch, n_levels)
        self.use_sdsa = use_sdsa
        if use_sdsa:
            self.sdsa = SDSABlock(in_ch, n_heads=max(1, in_ch // 16), T=T)

    def forward(self, feat):
        if self.use_sdsa:
            feat = self.sdsa(feat)
        mem = None
        outputs = []
        for t in range(self.T):
            x = self.mpbn(self.conv(feat), t)
            out, mem = self.lif(x, mem)
            outputs.append(out)
        return outputs
