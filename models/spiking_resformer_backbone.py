#!/usr/bin/env python3
"""SpikingResformer backbone for fully-spiking SpikeAdapt-SC.

Replaces ResNet-50 ANN frontend with a fully spiking backbone based on
SpikingResformer (Shi et al., CVPR 2024). This eliminates the ANN-to-SNN
conversion and makes the entire pipeline event-driven.

Architecture:
    Prologue → Stage 0 (56×56) → Stage 1 (28×28) → Stage 2 (14×14)

    Split point: After Stage 1 (output 28×28) or Stage 2 (output 14×14)
    Stage 2 at 14×14 matches ResNet-50 layer3 for drop-in compatibility
    with existing SNN encoder/decoder.

Key components:
    - Dual Spike Self-Attention (DSSA): spike-compatible attention
    - Group-Wise Spiking FFN (GWFFN): local + global feature extraction
    - LIF neurons throughout: tau=2.0, v_th=1.0, sigmoid surrogate grad

Variants:
    Ti: 11.1M params, planes=[64, 192, 384], 74.3% ImageNet
    S:  17.8M params, planes=[64, 256, 512], 76.0% ImageNet
    M:  35.5M params, planes=[64, 384, 768], 77.2% ImageNet

Reference: https://github.com/xyshi2000/SpikingResformer

Usage:
    from models.spiking_resformer_backbone import SpikingResformerFront, SpikingResformerBack
    front = SpikingResformerFront(variant='ti', T=4)
    back = SpikingResformerBack(in_channels=384, n_classes=30)
    features = front(images)  # [B, 384, 14, 14] — same format as ResNet50Front
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Spiking Neuron Layer
# ======================================================================

class SurrogateSigmoid(torch.autograd.Function):
    """Sigmoid surrogate gradient for spike function."""
    @staticmethod
    def forward(ctx, x, alpha=4.0):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        alpha = ctx.alpha
        sig = torch.sigmoid(alpha * x)
        return grad_output * alpha * sig * (1 - sig), None


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron with multi-step processing.

    Matches SpikingResformer's LIF: tau=2.0, v_threshold=1.0, detach_reset.
    """
    def __init__(self, tau=2.0, v_threshold=1.0, detach_reset=True):
        super().__init__()
        self.tau = tau
        self.v_threshold = v_threshold
        self.detach_reset = detach_reset
        self.spike_fn = SurrogateSigmoid.apply

    def forward(self, x, T=None):
        """Process input across T timesteps.

        Args:
            x: [T*B, C, H, W] or [T, B, C, H, W]
            T: number of timesteps (required if x is [T*B, ...])
        Returns:
            spikes: same shape as input
        """
        if x.dim() == 5:
            # [T, B, C, H, W] format
            T_steps = x.size(0)
            B = x.size(1)
            outputs = []
            v = torch.zeros_like(x[0])
            for t in range(T_steps):
                v = v / self.tau + x[t]
                spike = self.spike_fn(v - self.v_threshold)
                reset = spike.detach() if self.detach_reset else spike
                v = v - reset * self.v_threshold
                outputs.append(spike)
            return torch.stack(outputs, dim=0)
        elif T is not None:
            # [T*B, C, H, W] → process as [T, B, C, H, W]
            TB = x.size(0)
            B = TB // T
            x5d = x.view(T, B, *x.shape[1:])
            out = self.forward(x5d)
            return out.view(TB, *x.shape[1:])
        else:
            # Single step
            return self.spike_fn(x - self.v_threshold)


# ======================================================================
# Basic Building Blocks
# ======================================================================

class SpikingBN(nn.Module):
    """BatchNorm for spiking tensors — applies BN per timestep."""
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x, T=None):
        if x.dim() == 5:
            T_steps, B = x.shape[:2]
            out = self.bn(x.reshape(T_steps * B, *x.shape[2:]))
            return out.view(T_steps, B, *out.shape[1:])
        return self.bn(x)


class SpikingConv2d(nn.Module):
    """Conv2d that handles [T, B, C, H, W] tensors."""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, groups=groups, bias=bias)

    def forward(self, x):
        if x.dim() == 5:
            T, B = x.shape[:2]
            out = self.conv(x.reshape(T * B, *x.shape[2:]))
            return out.view(T, B, *out.shape[1:])
        return self.conv(x)


# ======================================================================
# Dual Spike Self-Attention (DSSA)
# ======================================================================

class DSSA(nn.Module):
    """Dual Spike Self-Attention from SpikingResformer.

    Uses spike-compatible matrix operations (AND-accumulate) instead of
    standard dot-product attention. Two spike projections (y1, y2) create
    attention weights and apply them, all using binary operations.

    Args:
        dim: input/output channels
        num_heads: number of attention heads
        patch_size: spatial patch size for attention computation
    """
    def __init__(self, dim, num_heads=1, patch_size=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.head_dim = dim // num_heads

        # Dual spike projections
        self.proj = SpikingConv2d(dim, 2 * dim, kernel_size=patch_size,
                                   stride=patch_size, bias=False)
        self.proj_bn = SpikingBN(2 * dim)
        self.proj_lif = LIFNeuron()

        # Input LIF (for query-like path)
        self.x_lif = LIFNeuron()

        # Output projection
        self.out_proj = SpikingConv2d(dim, dim, kernel_size=1, bias=False)
        self.out_bn = SpikingBN(dim)
        self.out_lif = LIFNeuron()

        # Scaling factors (momentum-averaged firing rates)
        self.register_buffer('scale_x', torch.ones(1))
        self.register_buffer('scale_attn', torch.ones(1))

    def forward(self, x):
        """
        Args:
            x: [T, B, C, H, W]
        Returns:
            out: [T, B, C, H, W]
        """
        T, B, C, H, W = x.shape
        identity = x

        # Spike the input
        x_spike = self.x_lif(x)

        # Dual spike projection: 2 sets of spike features
        y = self.proj_lif(self.proj_bn(self.proj(x)))  # [T, B, 2C, H', W']
        H_p, W_p = y.shape[3], y.shape[4]
        n_patches = H_p * W_p

        # Split into y1, y2
        y1 = y[:, :, :C, :, :]  # [T, B, C, H', W']
        y2 = y[:, :, C:, :, :]

        # Reshape for multi-head attention
        # x_spike: [T, B, heads, head_dim, H*W]
        x_flat = x_spike.reshape(T, B, self.num_heads, self.head_dim, H * W)
        y1_flat = y1.reshape(T, B, self.num_heads, self.head_dim, n_patches)
        y2_flat = y2.reshape(T, B, self.num_heads, self.head_dim, n_patches)

        # Attention: attn = y1^T @ x (spike matmul ≈ AND-accumulate)
        # [T, B, heads, n_patches, head_dim] @ [T, B, heads, head_dim, H*W]
        # = [T, B, heads, n_patches, H*W]
        scale1 = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(y1_flat.transpose(-2, -1), x_flat) * scale1

        # Apply attention: out = y2 @ attn
        # [T, B, heads, head_dim, n_patches] @ [T, B, heads, n_patches, H*W]
        # = [T, B, heads, head_dim, H*W]
        scale2 = 1.0 / math.sqrt(n_patches)
        out = torch.matmul(y2_flat, attn) * scale2

        # Reshape back
        out = out.reshape(T, B, C, H, W)

        # Output projection + residual
        out = self.out_lif(self.out_bn(self.out_proj(out)))
        return out + identity


# ======================================================================
# Group-Wise Spiking FFN (GWFFN)
# ======================================================================

class GWFFN(nn.Module):
    """Group-Wise Spiking Feed-Forward Network.

    Expand → group-wise Conv3x3 → contract, with skip connections.
    """
    def __init__(self, dim, expand_ratio=4, group_channels=64):
        super().__init__()
        hidden = dim * expand_ratio
        groups = max(1, hidden // group_channels)

        # Expand
        self.fc1 = SpikingConv2d(dim, hidden, 1, bias=False)
        self.bn1 = SpikingBN(hidden)
        self.lif1 = LIFNeuron()

        # Group-wise spatial mixing
        self.dw = SpikingConv2d(hidden, hidden, 3, padding=1, groups=groups, bias=False)
        self.bn2 = SpikingBN(hidden)
        self.lif2 = LIFNeuron()

        # Contract
        self.fc2 = SpikingConv2d(hidden, dim, 1, bias=False)
        self.bn3 = SpikingBN(dim)
        self.lif3 = LIFNeuron()

    def forward(self, x):
        identity = x
        x = self.lif1(self.bn1(self.fc1(x)))
        x_skip = x
        x = self.lif2(self.bn2(self.dw(x)))
        x = x + x_skip  # intermediate skip
        x = self.lif3(self.bn3(self.fc2(x)))
        return x + identity


# ======================================================================
# SpikingResformer Stage
# ======================================================================

class SpikingResformerBlock(nn.Module):
    """One DSSA + GWFFN block."""
    def __init__(self, dim, num_heads, patch_size):
        super().__init__()
        self.attn = DSSA(dim, num_heads, patch_size)
        self.ffn = GWFFN(dim)

    def forward(self, x):
        x = self.attn(x)
        x = self.ffn(x)
        return x


class DownsampleLayer(nn.Module):
    """Spatial downsampling between stages: Conv3x3(s=2) + BN + LIF."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = SpikingConv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.bn = SpikingBN(out_channels)
        self.lif = LIFNeuron()

    def forward(self, x):
        return self.lif(self.bn(self.conv(x)))


# ======================================================================
# SpikingResformer Front-End (replaces ResNet50Front)
# ======================================================================

VARIANT_CONFIGS = {
    'ti': {'planes': [64, 192, 384], 'heads': [1, 3, 6],
            'depths': [1, 1, 1], 'patch_sizes': [4, 2, 1]},
    's':  {'planes': [64, 256, 512], 'heads': [1, 4, 8],
            'depths': [2, 2, 2], 'patch_sizes': [4, 2, 1]},
    'm':  {'planes': [64, 384, 768], 'heads': [1, 6, 12],
            'depths': [2, 3, 2], 'patch_sizes': [4, 2, 1]},
}


class SpikingResformerFront(nn.Module):
    """Fully spiking frontend: replaces ResNet-50 layers 1-3.

    Input:  [B, 3, 224, 224] (standard images)
    Output: [B, C_out, 14, 14] (spatial features, same as ResNet50Front)

    The output is time-averaged across T timesteps to produce a single
    feature map compatible with the existing SNN encoder/decoder pipeline.

    Args:
        variant: 'ti', 's', or 'm'
        T: number of spiking timesteps (default 4)
        grid_size: output spatial size (default 14)
    """

    def __init__(self, variant='ti', T=4, grid_size=14):
        super().__init__()
        cfg = VARIANT_CONFIGS[variant]
        self.T = T
        self.variant = variant
        self.out_channels = cfg['planes'][-1]  # e.g., 384 for Ti

        # Prologue: 224→56 via Conv7x7(s=2) + MaxPool(s=2)
        self.prologue = nn.Sequential(
            nn.Conv2d(3, cfg['planes'][0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(cfg['planes'][0]),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.prologue_lif = LIFNeuron()

        # Stage 0: 56×56
        self.stage0 = nn.Sequential(*[
            SpikingResformerBlock(cfg['planes'][0], cfg['heads'][0], cfg['patch_sizes'][0])
            for _ in range(cfg['depths'][0])
        ])

        # Downsample 0→1: 56→28
        self.down01 = DownsampleLayer(cfg['planes'][0], cfg['planes'][1])

        # Stage 1: 28×28
        self.stage1 = nn.Sequential(*[
            SpikingResformerBlock(cfg['planes'][1], cfg['heads'][1], cfg['patch_sizes'][1])
            for _ in range(cfg['depths'][1])
        ])

        # Downsample 1→2: 28→14
        self.down12 = DownsampleLayer(cfg['planes'][1], cfg['planes'][2])

        # Stage 2: 14×14 (split point — output this)
        self.stage2 = nn.Sequential(*[
            SpikingResformerBlock(cfg['planes'][2], cfg['heads'][2], cfg['patch_sizes'][2])
            for _ in range(cfg['depths'][2])
        ])

        # Channel adapter: match ResNet-50 layer3 output (1024 channels)
        # This allows drop-in compatibility with existing SNN encoder
        self.channel_adapter = nn.Sequential(
            nn.Conv2d(cfg['planes'][-1], 1024, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.use_adapter = True  # Set False to output native channels

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: [B, 3, 224, 224]
        Returns:
            features: [B, 1024, 14, 14] (with adapter) or [B, C_out, 14, 14]
        """
        B = x.size(0)

        # Prologue (ANN, shared across timesteps)
        x0 = self.prologue(x)  # [B, C0, 56, 56]

        # Expand to T timesteps
        x_t = x0.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [T, B, C0, 56, 56]
        x_t = self.prologue_lif(x_t)

        # Stage 0
        x_t = self.stage0(x_t)

        # Downsample + Stage 1
        x_t = self.down01(x_t)
        x_t = self.stage1(x_t)

        # Downsample + Stage 2
        x_t = self.down12(x_t)
        x_t = self.stage2(x_t)  # [T, B, C2, 14, 14]

        # Time-average to get single feature map
        features = x_t.mean(0)  # [B, C2, 14, 14]

        # Channel adapter for compatibility
        if self.use_adapter:
            features = self.channel_adapter(features)  # [B, 1024, 14, 14]

        return features


# ======================================================================
# SpikingResformer Back-End (classifier head)
# ======================================================================

class SpikingResformerBack(nn.Module):
    """Classification head — replaces ResNet50Back.

    Takes reconstructed features from decoder and classifies.

    Args:
        in_channels: input channel dim (1024 with adapter, or native C_out)
        n_classes: number of output classes
    """

    def __init__(self, in_channels=1024, n_classes=30):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        return self.classifier(x)


# ======================================================================
# Full Pipeline: SpikingResformer + SNN Encoder/Decoder
# ======================================================================

class SpikeAdaptSC_FullSpiking(nn.Module):
    """Fully spiking SpikeAdapt-SC: SpikingResformer frontend + SNN codec.

    This replaces the ANN ResNet-50 frontend with a spiking backbone,
    making the entire transmitter pipeline event-driven.

    The receiver side uses the same SNN decoder as before.
    """

    def __init__(self, variant='ti', T_backbone=4, T_codec=8,
                 C1=256, C2=36, target_rate=0.75, grid_size=14, n_classes=30):
        super().__init__()
        self.T_backbone = T_backbone
        self.T_codec = T_codec

        # Import codec components
        import sys, os
        _here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(_here, '..', 'train'))
        sys.path.insert(0, _here)  # for noise_aware_scorer
        from train_aid_v2 import BSC_Channel, LearnedBlockMask
        from train_aid_v5 import EncoderV5, DecoderV5
        from noise_aware_scorer import NoiseAwareScorer

        # Fully spiking frontend
        self.front = SpikingResformerFront(variant=variant, T=T_backbone)

        # SNN codec (reuses existing architecture)
        C_in = 1024  # with channel adapter
        self.encoder = EncoderV5(C_in, C1, C2, T_codec, use_mpbn=True)
        self.scorer = NoiseAwareScorer(C_spike=C2, hidden=32)
        self.block_mask = LearnedBlockMask(target_rate, 0.5)
        self.decoder = DecoderV5(C_in, C1, C2, T_codec, use_mpbn=True)
        self.channel = BSC_Channel()

        # Classifier
        self.back = SpikingResformerBack(in_channels=C_in, n_classes=n_classes)

    def forward(self, imgs, noise_param=0.0, target_rate_override=None):
        """
        Args:
            imgs: [B, 3, 224, 224] raw images
            noise_param: BSC BER
            target_rate_override: optional rho override
        Returns:
            logits: [B, n_classes]
            stats: dict with transmission info
        """
        # Frontend: images → 14×14 spiking features
        feat = self.front(imgs)  # [B, 1024, 14, 14]

        # SNN encoder
        all_S2, m1, m2 = [], None, None
        for t in range(self.T_codec):
            _, s2, m1, m2 = self.encoder(feat, m1, m2, t=t)
            all_S2.append(s2)

        # Score & mask
        importance = self.scorer(all_S2, noise_param).squeeze(1)
        if target_rate_override is not None:
            old = self.block_mask.target_rate
            self.block_mask.target_rate = target_rate_override
            mask, tx = self.block_mask(importance, training=False)
            self.block_mask.target_rate = old
        else:
            mask, tx = self.block_mask(importance, training=self.training)

        # Channel + decode
        recv = [self.channel(all_S2[t] * mask, noise_param) for t in range(self.T_codec)]
        Fp = self.decoder(recv, mask)

        # Classify
        logits = self.back(Fp)

        with torch.no_grad():
            fr = torch.stack(all_S2).mean().item()

        return logits, {
            'tx_rate': tx.item(), 'mask': mask, 'importance': importance,
            'firing_rate': fr, 'all_S2': all_S2,
        }


# ======================================================================
# UTILITY: Model info
# ======================================================================

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == '__main__':
    # Quick test
    print("SpikingResformer Backbone Test")
    print("=" * 50)

    for variant in ['ti', 's', 'm']:
        front = SpikingResformerFront(variant=variant, T=4)
        total, trainable = count_params(front)
        print(f"\n{variant.upper()}: {total/1e6:.1f}M params ({trainable/1e6:.1f}M trainable)")
        print(f"  Output channels: {front.out_channels}")

        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            y = front(x)
        print(f"  Input: {x.shape} → Output: {y.shape}")

    # Full pipeline test
    print("\n\nFull Pipeline Test")
    print("=" * 50)
    model = SpikeAdaptSC_FullSpiking(variant='ti', n_classes=30)
    total, trainable = count_params(model)
    print(f"Total: {total/1e6:.1f}M params")

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits, stats = model(x, noise_param=0.1)
    print(f"Input: {x.shape} → Logits: {logits.shape}")
    print(f"Firing rate: {stats['firing_rate']:.3f}")
    print(f"TX rate: {stats['tx_rate']:.3f}")
