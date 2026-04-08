#!/usr/bin/env python3
"""SpikeAdapt-SC Image Codec V5: V4 + U-Net Skips + Wider Bottleneck.

Key changes over V4:
  - U-Net skip connections from encoder to decoder (bypass bottleneck for
    high-frequency details — the #1 missing ingredient for PSNR)
  - Wider C_feat=512 (more capacity at bottleneck)
  - C_tx=96 (50% wider channel, still ternary)
  - All V4 innovations retained: ternary spikes, membrane shortcuts,
    T=16, Swin blocks, diffusion refiner, KD support

Why U-Net skips work:
  In V2/V4, the encoder extracts features at 16× downscale, ALL info must
  pass through 64-ch ternary spikes, and the decoder rebuilds from scratch.
  High-frequency details (edges, textures) are destroyed. U-Net skips let
  the decoder directly access multi-scale encoder features, so only the
  semantic essence needs to pass through the spike bottleneck.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Import shared components from V4
from models.spike_image_codec_v4 import (
    ResBlock, DownBlock, UpBlock, SwinBlock,
    TernaryLIFNeuron, MPBN, IHFNeuron,
    TemporalAttention, ContentScorer, TopKBlockMask,
    AWGNChannel, TernaryBSCChannel, DiffusionRefiner,
)


# =============================================================================
# U-Net Encoder: returns multi-scale features for skip connections
# =============================================================================

class UNetEncoderV5(nn.Module):
    """Encoder that returns intermediate features for skip connections."""
    def __init__(self, C_feat=512, use_swin=True):
        super().__init__()
        self.use_swin = use_swin
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.down1 = DownBlock(64, 96, n_res=2)      # H/2,  96ch
        self.down2 = DownBlock(96, 192, n_res=2)      # H/4,  192ch
        self.down3 = DownBlock(192, 384, n_res=2)     # H/8,  384ch
        self.down4 = DownBlock(384, C_feat, n_res=2)  # H/16, 512ch

        if use_swin:
            self.swin = nn.Sequential(
                SwinBlock(C_feat, window_size=8, n_heads=8),
                SwinBlock(C_feat, window_size=8, n_heads=8),
            )

    def forward(self, img):
        s0 = self.stem(img)      # H,    64ch
        s1 = self.down1(s0)      # H/2,  96ch
        s2 = self.down2(s1)      # H/4,  192ch
        s3 = self.down3(s2)      # H/8,  384ch
        s4 = self.down4(s3)      # H/16, 512ch
        if self.use_swin:
            s4 = self.swin(s4)
        return s4, [s0, s1, s2, s3]  # bottleneck + skip features


# =============================================================================
# U-Net Decoder: receives skip connections from encoder
# =============================================================================

class SkipFusion(nn.Module):
    """Fuse decoder feature with encoder skip via 1x1 conv + residual."""
    def __init__(self, dec_ch, skip_ch):
        super().__init__()
        self.proj = nn.Conv2d(dec_ch + skip_ch, dec_ch, 1)
        self.bn = nn.BatchNorm2d(dec_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, dec_feat, skip_feat):
        x = torch.cat([dec_feat, skip_feat], dim=1)
        return self.act(self.bn(self.proj(x)))


class UNetDecoderV5(nn.Module):
    """Decoder with skip connections from encoder.

    Skip connections bypass the spike bottleneck, letting high-frequency
    details reach the decoder directly. Only the semantic essence needs
    to pass through the noisy channel.
    """
    def __init__(self, C_feat=512, use_swin=True, use_skips=True):
        super().__init__()
        self.use_swin = use_swin
        self.use_skips = use_skips

        if use_swin:
            self.swin = nn.Sequential(
                SwinBlock(C_feat, window_size=8, n_heads=8),
                SwinBlock(C_feat, window_size=8, n_heads=8),
            )

        self.up1 = UpBlock(C_feat, 384, n_res=2)   # H/8
        self.up2 = UpBlock(384, 192, n_res=2)       # H/4
        self.up3 = UpBlock(192, 96, n_res=2)        # H/2
        self.up4 = UpBlock(96, 64, n_res=1)         # H

        if use_skips:
            # Skip fusions: match decoder channel dims with encoder skips
            self.skip3 = SkipFusion(384, 384)   # s3: 384ch at H/8
            self.skip2 = SkipFusion(192, 192)   # s2: 192ch at H/4
            self.skip1 = SkipFusion(96, 96)     # s1: 96ch  at H/2
            self.skip0 = SkipFusion(64, 64)     # s0: 64ch  at H

        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, feat, skips=None):
        x = feat
        if self.use_swin:
            x = self.swin(x)

        x = self.up1(x)  # H/8, 384ch
        if self.use_skips and skips is not None:
            x = self.skip3(x, skips[3])

        x = self.up2(x)  # H/4, 192ch
        if self.use_skips and skips is not None:
            x = self.skip2(x, skips[2])

        x = self.up3(x)  # H/2, 96ch
        if self.use_skips and skips is not None:
            x = self.skip1(x, skips[1])

        x = self.up4(x)  # H, 64ch
        if self.use_skips and skips is not None:
            x = self.skip0(x, skips[0])

        return self.final(x)


# =============================================================================
# SNN Bottleneck V5 (wider C_tx=96, ternary, T=16)
# =============================================================================

class SNNBottleneckEncoderV5(nn.Module):
    """Feature → Ternary spikes, wider bottleneck."""
    def __init__(self, C_feat=512, C_tx=96, T=16):
        super().__init__()
        self.T = T
        self.conv1 = nn.Conv2d(C_feat, 256, 3, 1, 1)
        self.mpbn1 = MPBN(256, T)
        self.lif1 = TernaryLIFNeuron(256)
        self.conv2 = nn.Conv2d(256, C_tx, 3, 1, 1)
        self.mpbn2 = MPBN(C_tx, T)
        self.lif2 = TernaryLIFNeuron(C_tx)

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


class SNNBottleneckDecoderV5(nn.Module):
    """Received spikes → Features with membrane shortcuts."""
    def __init__(self, C_feat=512, C_tx=96, T=16):
        super().__init__()
        self.T = T
        self.conv3 = nn.Conv2d(C_tx, 256, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, C_feat, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(C_feat)
        self.ihf = IHFNeuron()

        # Membrane shortcut
        self.mem_shortcut = nn.Sequential(
            nn.Conv2d(C_tx, C_feat, 1),
            nn.BatchNorm2d(C_feat),
        )
        self.mem_gate = nn.Sequential(
            nn.Conv2d(C_feat * 2, C_feat, 1),
            nn.Sigmoid(),
        )
        self.temporal_attn = TemporalAttention(C_feat, T)

    def forward(self, recv_spikes):
        m4 = None
        Fs, Fm = [], []
        mem_accum = torch.zeros(
            recv_spikes[0].shape[0], recv_spikes[0].shape[1],
            recv_spikes[0].shape[2], recv_spikes[0].shape[3],
            device=recv_spikes[0].device
        )

        for t in range(self.T):
            s3 = F.relu(self.bn3(self.conv3(recv_spikes[t])))
            sp, m4 = self.ihf(self.bn4(self.conv4(s3)), m4)
            mem_accum = mem_accum + recv_spikes[t]
            Fs.append(sp)
            Fm.append(m4.clone())

        spike_feat = self.temporal_attn(Fs, Fm)
        mem_feat = self.mem_shortcut(mem_accum / self.T)
        gate = self.mem_gate(torch.cat([spike_feat, mem_feat], dim=1))
        return spike_feat + gate * mem_feat


# =============================================================================
# Float Teacher V5 (for KD — matches U-Net encoder)
# =============================================================================

class FloatTeacherV5(nn.Module):
    """Float-valued teacher with U-Net skips (KD upper bound)."""
    def __init__(self, C_feat=512, C_tx=96, use_swin=True):
        super().__init__()
        self.img_encoder = UNetEncoderV5(C_feat, use_swin)
        self.bottleneck_enc = nn.Sequential(
            nn.Conv2d(C_feat, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, C_tx, 3, 1, 1),
            nn.BatchNorm2d(C_tx),
            nn.Tanh(),
        )
        self.bottleneck_dec = nn.Sequential(
            nn.Conv2d(C_tx, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, C_feat, 3, 1, 1),
            nn.BatchNorm2d(C_feat),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.img_decoder = UNetDecoderV5(C_feat, use_swin, use_skips=True)

    def forward(self, img, noise_param=0.0, channel='awgn'):
        feat, skips = self.img_encoder(img)
        z = self.bottleneck_enc(feat)
        if channel == 'awgn' and noise_param < 100:
            snr_lin = 10 ** (noise_param / 10.0)
            z = z + torch.randn_like(z) / math.sqrt(2 * snr_lin)
        feat_recon = self.bottleneck_dec(z)
        img_recon = self.img_decoder(feat_recon, skips)
        return img_recon, {'z': z, 'feat': feat, 'feat_recon': feat_recon}


# =============================================================================
# Full SpikeImageCodecV5
# =============================================================================

class SpikeImageCodecV5(nn.Module):
    """SpikeAdapt-SC V5: V4 + U-Net skips + wider bottleneck.

    Key insight: skip connections let encoder features bypass the spike
    bottleneck, so high-frequency details reach the decoder directly.
    The spike channel only needs to carry semantic/structural info.
    """
    def __init__(self, C_feat=512, C_tx=96, T=16, target_cbr=0.5,
                 use_swin=True, use_diffusion=False, use_skips=True):
        super().__init__()
        self.T = T
        self.C_tx = C_tx
        self.target_cbr = target_cbr
        self.use_skips = use_skips

        # U-Net encoder (returns skips)
        self.img_encoder = UNetEncoderV5(C_feat, use_swin)

        # SNN bottleneck (ternary, T=16, wider)
        self.snn_encoder = SNNBottleneckEncoderV5(C_feat, C_tx, T)

        # Scorer + masking
        self.scorer = ContentScorer(C_tx, hidden=64)
        self.masker = TopKBlockMask(target_cbr, 0.5)

        # Channels
        self.awgn = AWGNChannel()
        self.bsc = TernaryBSCChannel()

        # SNN decoder (membrane shortcuts)
        self.snn_decoder = SNNBottleneckDecoderV5(C_feat, C_tx, T)

        # U-Net decoder (with skip connections)
        self.img_decoder = UNetDecoderV5(C_feat, use_swin, use_skips)

        # Diffusion refiner
        self.use_diffusion = use_diffusion
        if use_diffusion:
            self.diffusion = DiffusionRefiner(channels=64, n_steps=8)

    def forward(self, img, noise_param=0.0, channel='awgn',
                use_masking=True, target_cbr_override=None,
                use_diffusion=False):
        # Encode (with skip features)
        feat, skips = self.img_encoder(img)
        spikes = self.snn_encoder(feat)

        # Score and mask
        if use_masking:
            importance = self.scorer(spikes, noise_param)
            masked_spikes, actual_cbr = self.masker(
                spikes, importance, self.training, target_cbr_override)
        else:
            importance = None
            masked_spikes = spikes
            actual_cbr = 1.0

        # Channel
        if channel == 'awgn':
            recv = [self.awgn(s, noise_param) for s in masked_spikes]
        elif channel == 'bsc':
            recv = [self.bsc(s, noise_param) for s in masked_spikes]
        else:
            recv = masked_spikes

        # Decode (with skip connections from encoder)
        feat_recon = self.snn_decoder(recv)
        # Skip connections bypass the channel — they carry high-freq info
        # directly from encoder to decoder (receiver-side: skips are local)
        img_recon = self.img_decoder(
            feat_recon,
            skips if self.use_skips else None
        )

        # Diffusion refinement
        if use_diffusion and self.use_diffusion and not self.training:
            img_recon = self.diffusion(img_recon, training=False)

        return img_recon, {
            'importance': importance,
            'actual_cbr': actual_cbr,
            'spikes': spikes,
            'feat': feat,
            'feat_recon': feat_recon,
        }
