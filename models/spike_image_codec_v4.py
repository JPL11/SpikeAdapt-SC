#!/usr/bin/env python3
"""SpikeAdapt-SC Image Codec V4: All 6 improvements combined.

Improvements over V2:
  #1 Ternary Spikes {-1, 0, 1} with trainable amplitude α (AAAI 2024)
  #2 Membrane Potential Shortcuts in decoder (residual continuous path)
  #3 Knowledge Distillation support (float teacher → SNN student)
  #4 T=16 timesteps (2× capacity over V2's T=8)
  #5 Diffusion refinement support (lightweight DDPM post-processor)
  #6 Swin Transformer hybrid encoder/decoder blocks

Architecture:
  Image (3×H×W)
   → Hybrid Encoder (CNN + Swin blocks, 16× downscale)
   → SNN Bottleneck (Ternary LIF + MPBN, T=16)
   → Content Scorer (spike-rate variance, noise-conditioned)
   → Top-K Block Mask (binary, exact CBR control)
   → Channel (AWGN or BSC for ternary)
   → SNN Decoder (membrane shortcuts + temporal attention)
   → Hybrid Decoder (CNN + Swin blocks)
   → [Optional: DiffusionRefiner]
   → Reconstructed Image
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# Building Blocks (same as V2 + Swin additions)
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
        return self.res(self.down(x))


class UpBlock(nn.Module):
    """Transposed convolution + residual blocks for decoder."""
    def __init__(self, in_ch, out_ch, n_res=2):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.res = nn.Sequential(*[ResBlock(out_ch) for _ in range(n_res)])
    
    def forward(self, x):
        return self.res(self.up(x))


# =============================================================================
# Improvement #6: Swin Transformer Block (Lightweight)
# Source: SwinJSCC (IEEE 2024)
# =============================================================================

class WindowAttention(nn.Module):
    """Window-based multi-head self-attention (Swin style)."""
    def __init__(self, dim, window_size=8, n_heads=4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), n_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        coords = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords, coords, indexing='ij')).flatten(1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        self.register_buffer('relative_position_index', 
                            relative_coords.sum(-1).view(-1))
    
    def forward(self, x):
        # x: (num_windows*B, window_size*window_size, C)
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        bias = self.relative_position_bias_table[self.relative_position_index]
        bias = bias.view(N, N, -1).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)
        
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


class SwinBlock(nn.Module):
    """Simplified Swin Transformer block for JSCC."""
    def __init__(self, dim, window_size=8, n_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, n_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
    
    def _window_partition(self, x, H, W):
        B, _, C = x.shape
        x = x.view(B, H, W, C)
        ws = self.window_size
        # Pad if needed
        pH = (ws - H % ws) % ws
        pW = (ws - W % ws) % ws
        if pH > 0 or pW > 0:
            x = F.pad(x, (0, 0, 0, pW, 0, pH))
        Hp, Wp = H + pH, W + pW
        x = x.view(B, Hp // ws, ws, Wp // ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)
        return x, Hp, Wp
    
    def _window_reverse(self, x, Hp, Wp, H, W, B):
        ws = self.window_size
        x = x.view(B, Hp // ws, Wp // ws, ws, ws, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)
        if Hp > H or Wp > W:
            x = x[:, :H, :W, :]
        return x.reshape(B, H * W, -1)
    
    def forward(self, x):
        # x: B×C×H×W (conv format) → reshape for attention
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # B×(H*W)×C
        
        # Window attention
        shortcut = x_flat
        x_norm = self.norm1(x_flat)
        x_win, Hp, Wp = self._window_partition(x_norm, H, W)
        x_win = self.attn(x_win)
        x_flat = self._window_reverse(x_win, Hp, Wp, H, W, B)
        x_flat = shortcut + x_flat
        
        # MLP
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        
        return x_flat.transpose(1, 2).reshape(B, C, H, W)


# =============================================================================
# Improvement #1: Ternary LIF Neuron {-1, 0, 1}
# Source: "Ternary Spike" (AAAI 2024)
# =============================================================================

class TernaryLIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire with ternary output {-1, 0, 1}.
    
    Key innovation: trainable amplitude α per layer, folded into weights
    at inference via re-parameterization.
    
    Information capacity: log₂(3) = 1.585 bits/spike vs 1.0 for binary.
    """
    def __init__(self, channels=None):
        super().__init__()
        self.beta = nn.Parameter(torch.full((channels,), 0.5)) if channels else nn.Parameter(torch.tensor(0.5))
        self.slope = nn.Parameter(torch.tensor(5.0))
        # Trainable amplitude (AAAI 2024 innovation)
        self.alpha = nn.Parameter(torch.ones(channels if channels else 1))
        # Dual thresholds: positive and negative
        self.threshold_pos = nn.Parameter(torch.ones(channels if channels else 1))
        self.threshold_neg = nn.Parameter(torch.ones(channels if channels else 1) * -1.0)
    
    def forward(self, x, mem=None):
        if mem is None:
            mem = torch.zeros_like(x)
        beta = torch.sigmoid(self.beta)
        if beta.dim() == 1:
            beta = beta.view(1, -1, 1, 1)
        mem = beta * mem + x
        
        # Ternary spiking: positive spike, negative spike, or no spike
        thr_pos = self.threshold_pos.abs().view(1, -1, 1, 1)
        thr_neg = -self.threshold_neg.abs().view(1, -1, 1, 1)
        alpha = self.alpha.abs().view(1, -1, 1, 1)
        
        # Surrogate gradients for both thresholds
        spike_pos = self._surrogate(mem - thr_pos)        # fires +1
        spike_neg = self._surrogate(-(mem - thr_neg))     # fires -1 (when mem < neg_threshold)
        
        # Ternary output: +α, 0, or -α
        spike = alpha * (spike_pos - spike_neg)
        
        # Reset membrane for both positive and negative spikes
        fired = (spike_pos + spike_neg).clamp(0, 1)
        mem = mem * (1 - fired)
        
        return spike, mem
    
    def _surrogate(self, x):
        slope = torch.clamp(self.slope, 1.0, 20.0)
        return (x >= 0).float() + (torch.sigmoid(slope * x) - (x >= 0).float()).detach() + \
               torch.sigmoid(slope * x) - torch.sigmoid(slope * x).detach()


# =============================================================================
# SNN Primitives (kept from V2)
# =============================================================================

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
# Hybrid Encoder (CNN + Swin) — Improvement #6
# =============================================================================

class HybridEncoderV4(nn.Module):
    """ResNet encoder with Swin Transformer blocks at bottleneck."""
    def __init__(self, C_feat=384, use_swin=True):
        super().__init__()
        self.use_swin = use_swin
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.down1 = DownBlock(64, 96, n_res=2)     # H/2
        self.down2 = DownBlock(96, 192, n_res=2)     # H/4
        self.down3 = DownBlock(192, 384, n_res=2)    # H/8
        self.down4 = DownBlock(384, C_feat, n_res=2) # H/16
        
        # Swin blocks at the bottleneck (lowest resolution = cheapest)
        if use_swin:
            self.swin = nn.Sequential(
                SwinBlock(C_feat, window_size=8, n_heads=6),
                SwinBlock(C_feat, window_size=8, n_heads=6),
            )
    
    def forward(self, img):
        x = self.net(img)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        if self.use_swin:
            x = self.swin(x)
        return x


class HybridDecoderV4(nn.Module):
    """ResNet decoder with Swin Transformer blocks."""
    def __init__(self, C_feat=384, use_swin=True):
        super().__init__()
        self.use_swin = use_swin
        
        if use_swin:
            self.swin = nn.Sequential(
                SwinBlock(C_feat, window_size=8, n_heads=6),
                SwinBlock(C_feat, window_size=8, n_heads=6),
            )
        
        self.up1 = UpBlock(C_feat, 384, n_res=2)
        self.up2 = UpBlock(384, 192, n_res=2)
        self.up3 = UpBlock(192, 96, n_res=2)
        self.up4 = UpBlock(96, 64, n_res=1)
        self.final = nn.Sequential(
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, feat):
        x = feat
        if self.use_swin:
            x = self.swin(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.final(x)


# =============================================================================
# SNN Bottleneck Encoder V4 (Ternary + T=16)
# =============================================================================

class SNNBottleneckEncoderV4(nn.Module):
    """Feature → Ternary spikes {-1, 0, 1} via Ternary LIF neurons."""
    def __init__(self, C_feat=384, C_tx=64, T=16):
        super().__init__()
        self.T = T
        self.conv1 = nn.Conv2d(C_feat, 192, 3, 1, 1)
        self.mpbn1 = MPBN(192, T)
        self.lif1 = TernaryLIFNeuron(192)
        self.conv2 = nn.Conv2d(192, C_tx, 3, 1, 1)
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


# =============================================================================
# SNN Bottleneck Decoder V4 (Membrane Shortcuts + Temporal Attention)
# Improvement #2: Membrane potential shortcuts
# =============================================================================

class TemporalAttention(nn.Module):
    """Attention over timesteps for aggregation."""
    def __init__(self, C_feat, T):
        super().__init__()
        self.slots = 2 * T
        self.attn_net = nn.Sequential(
            nn.Conv2d(C_feat, C_feat // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C_feat // 4, 1, 1),
        )
    
    def forward(self, spike_features, mem_features):
        all_feats = spike_features + mem_features
        stacked = torch.stack(all_feats, dim=1)
        scores = []
        for t in range(self.slots):
            s = self.attn_net(stacked[:, t])
            scores.append(s)
        weights = torch.cat(scores, dim=1).softmax(dim=1)
        weights = weights.unsqueeze(2)
        return (stacked * weights).sum(dim=1)


class SNNBottleneckDecoderV4(nn.Module):
    """Received spikes → Features with membrane potential shortcuts.
    
    Improvement #2: Instead of discarding membrane potential between
    conv layers, we add a continuous residual path that preserves
    fine-grained information through the decoder.
    """
    def __init__(self, C_feat=384, C_tx=64, T=16):
        super().__init__()
        self.T = T
        self.conv3 = nn.Conv2d(C_tx, 192, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(192)
        self.conv4 = nn.Conv2d(192, C_feat, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(C_feat)
        self.ihf = IHFNeuron()
        
        # Membrane shortcut: continuous residual path (Improvement #2)
        self.mem_shortcut = nn.Sequential(
            nn.Conv2d(C_tx, C_feat, 1),  # project C_tx → C_feat
            nn.BatchNorm2d(C_feat),
        )
        self.mem_gate = nn.Sequential(
            nn.Conv2d(C_feat * 2, C_feat, 1),
            nn.Sigmoid(),
        )
        
        self.temporal_attn = TemporalAttention(C_feat, T)
    
    def forward(self, recv_spikes):
        m3, m4 = None, None
        Fs, Fm = [], []
        
        # Accumulate membrane potential from input spikes (continuous path)
        mem_accum = torch.zeros_like(self.mem_shortcut[0].weight.new_zeros(
            recv_spikes[0].shape[0], recv_spikes[0].shape[1],
            recv_spikes[0].shape[2], recv_spikes[0].shape[3]
        ))
        
        for t in range(self.T):
            # Standard spike path
            s3 = F.relu(self.bn3(self.conv3(recv_spikes[t])))
            sp, m4 = self.ihf(self.bn4(self.conv4(s3)), m4)
            
            # Membrane shortcut: accumulate input membrane potential
            mem_accum = mem_accum + recv_spikes[t]
            
            Fs.append(sp)
            Fm.append(m4.clone())
        
        # Combine spike-based features with membrane shortcut
        spike_feat = self.temporal_attn(Fs, Fm)
        
        # Membrane shortcut projection
        mem_feat = self.mem_shortcut(mem_accum / self.T)  # normalize by T
        
        # Gated fusion of spike path and membrane path
        gate = self.mem_gate(torch.cat([spike_feat, mem_feat], dim=1))
        return spike_feat + gate * mem_feat


# =============================================================================
# Improvement #5: Lightweight Diffusion Refinement
# Source: DiffJSCC (IEEE 2024) — simplified 8-step version
# =============================================================================

class DiffusionRefiner(nn.Module):
    """Lightweight DDPM post-processor for quality enhancement.
    
    Instead of full Stable Diffusion (4GB, 100 steps), this is a custom
    lightweight 8-step denoiser (~2M params) conditioned on the initial
    reconstruction.
    """
    def __init__(self, channels=64, n_steps=8):
        super().__init__()
        self.n_steps = n_steps
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        
        # UNet-lite: encoder-decoder with skip connections
        self.enc1 = nn.Sequential(nn.Conv2d(3, channels, 3, 1, 1), nn.BatchNorm2d(channels), nn.SiLU())
        self.enc2 = nn.Sequential(nn.Conv2d(channels, channels*2, 4, 2, 1), nn.BatchNorm2d(channels*2), nn.SiLU())
        self.mid = nn.Sequential(ResBlock(channels*2), ResBlock(channels*2))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(channels*4, channels, 4, 2, 1), nn.BatchNorm2d(channels), nn.SiLU())
        self.dec1 = nn.Conv2d(channels*2, 3, 3, 1, 1)
        
        # Condition injection (from initial reconstruction)
        self.cond_proj = nn.Conv2d(3, channels, 1)
        
        # Noise schedule (fixed linear)
        betas = torch.linspace(1e-4, 0.02, n_steps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
    
    def _predict_noise(self, x_noisy, t_embed, cond):
        """Predict noise given noisy input, time embedding, and condition."""
        e1 = self.enc1(x_noisy)
        # Inject condition and time
        cond_feat = self.cond_proj(cond)
        t_feat = t_embed.unsqueeze(-1).unsqueeze(-1).expand_as(e1)
        e1 = e1 + cond_feat + t_feat
        
        e2 = self.enc2(e1)
        mid = self.mid(e2)
        d2 = self.dec2(torch.cat([mid, e2], dim=1))
        out = self.dec1(torch.cat([d2, e1], dim=1))
        return out
    
    def forward(self, x_coarse, training=True):
        """During training: add noise to target, predict noise.
           During inference: iteratively denoise from x_coarse.
        """
        if training:
            return x_coarse  # Diffusion training handled separately
        
        # Inference: start from coarse reconstruction, refine
        x = x_coarse
        for step in reversed(range(self.n_steps)):
            t = torch.tensor([step / self.n_steps], device=x.device).float()
            t_embed = self.time_embed(t.unsqueeze(0)).squeeze(0)
            t_embed = t_embed.unsqueeze(0).expand(x.size(0), -1)
            
            noise_pred = self._predict_noise(x, t_embed, x_coarse)
            
            # DDPM reverse step
            alpha = self.alphas[step]
            alpha_bar = self.alphas_cumprod[step]
            beta = self.betas[step]
            
            x = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_bar)) * noise_pred)
            
            if step > 0:
                noise = torch.randn_like(x) * torch.sqrt(beta) * 0.3  # reduced noise
                x = x + noise
        
        return x.clamp(0, 1)
    
    def training_loss(self, x_clean, x_coarse):
        """Compute diffusion training loss."""
        B = x_clean.size(0)
        t = torch.randint(0, self.n_steps, (B,), device=x_clean.device)
        
        noise = torch.randn_like(x_clean)
        sqrt_ab = self.sqrt_alphas_cumprod[t].view(B, 1, 1, 1)
        sqrt_omab = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1, 1)
        
        x_noisy = sqrt_ab * x_clean + sqrt_omab * noise
        
        t_float = t.float() / self.n_steps
        t_embed = self.time_embed(t_float.unsqueeze(1))
        
        noise_pred = self._predict_noise(x_noisy, t_embed, x_coarse)
        
        return F.mse_loss(noise_pred, noise)


# =============================================================================
# Improvement #3: Float Teacher Codec (for Knowledge Distillation)
# =============================================================================

class FloatTeacherCodec(nn.Module):
    """Float-valued (non-spiking) version of the codec — used as KD teacher.
    
    Same architecture but replaces SNN bottleneck with continuous-valued
    encoder/decoder. Provides the upper bound on achievable PSNR.
    """
    def __init__(self, C_feat=384, C_tx=64, use_swin=True):
        super().__init__()
        self.img_encoder = HybridEncoderV4(C_feat, use_swin)
        
        # Float bottleneck (no spikes — continuous valued)
        self.bottleneck_enc = nn.Sequential(
            nn.Conv2d(C_feat, 192, 3, 1, 1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(192, C_tx, 3, 1, 1),
            nn.BatchNorm2d(C_tx),
            nn.Tanh(),  # bounded [-1, 1] like our ternary spikes
        )
        self.bottleneck_dec = nn.Sequential(
            nn.Conv2d(C_tx, 192, 3, 1, 1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(192, C_feat, 3, 1, 1),
            nn.BatchNorm2d(C_feat),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.img_decoder = HybridDecoderV4(C_feat, use_swin)
    
    def forward(self, img, noise_param=0.0, channel='awgn'):
        feat = self.img_encoder(img)
        z = self.bottleneck_enc(feat)
        
        # Add channel noise
        if channel == 'awgn' and noise_param < 100:
            snr_lin = 10 ** (noise_param / 10.0)
            noise_std = 1.0 / math.sqrt(2 * snr_lin)
            z = z + torch.randn_like(z) * noise_std
        
        feat_recon = self.bottleneck_dec(z)
        img_recon = self.img_decoder(feat_recon)
        return img_recon, {'z': z, 'feat': feat, 'feat_recon': feat_recon}


# =============================================================================
# Content Scorer (same as V2 — spike-rate variance)
# =============================================================================

class ContentScorer(nn.Module):
    """Score spatial blocks by spike-rate variance (noise-conditioned)."""
    def __init__(self, C_tx=64, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C_tx + 1, hidden, 3, 1, 1),
            nn.BatchNorm2d(hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.BatchNorm2d(hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, spikes, noise_param=0.0):
        stacked = torch.stack(spikes, dim=0)
        mean_rate = stacked.mean(0)  # B×C_tx×H×W
        
        noise_map = torch.full(
            (mean_rate.size(0), 1, mean_rate.size(2), mean_rate.size(3)),
            noise_param / 20.0, device=mean_rate.device
        )
        features = torch.cat([mean_rate, noise_map], dim=1)  # B×(C_tx+1)×H×W
        
        return self.net(features)
    
    def compute_diversity_loss(self, spikes):
        stacked = torch.stack(spikes, dim=0)
        rates = stacked.mean(0)
        spatial_var = rates.var(dim=[2, 3]).mean()
        return -spatial_var


# =============================================================================
# Masking & Channels
# =============================================================================

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


class AWGNChannel(nn.Module):
    def forward(self, x, snr_db):
        if snr_db >= 100:
            return x
        snr_lin = 10 ** (snr_db / 10.0)
        noise_std = 1.0 / math.sqrt(2 * snr_lin)
        return x + torch.randn_like(x) * noise_std


class TernaryBSCChannel(nn.Module):
    """BSC for ternary signals: independently flips each trit."""
    def forward(self, x, ber):
        if ber <= 0:
            return x
        # For ternary: flip each element with probability ber
        # When flipped, randomly assign to one of the other 2 values
        flip_mask = (torch.rand_like(x) < ber).float()
        # Random ternary values: -α, 0, +α
        random_trit = (torch.randint(0, 3, x.shape, device=x.device).float() - 1)
        # Scale random_trit to match x's scale
        x_scale = x.abs().clamp(min=0.1).mean()
        random_trit = random_trit * x_scale
        
        noisy = x * (1 - flip_mask) + random_trit * flip_mask
        if self.training:
            return x + (noisy - x).detach()  # STE
        return noisy


# =============================================================================
# Full Image Codec V4
# =============================================================================

class SpikeImageCodecV4(nn.Module):
    """SpikeAdapt-SC V4: All 6 improvements.
    
    #1 Ternary spikes {-1,0,1} — 58% more info capacity
    #2 Membrane potential shortcuts — decoder residual path
    #3 KD support — teacher/student training mode
    #4 T=16 — double timesteps
    #5 Diffusion refiner — lightweight DDPM post-processor
    #6 Swin Transformer — hybrid CNN+Swin encoder/decoder
    """
    def __init__(self, C_feat=384, C_tx=64, T=16, target_cbr=0.5,
                 use_swin=True, use_diffusion=True):
        super().__init__()
        self.T = T
        self.C_tx = C_tx
        self.target_cbr = target_cbr
        
        # Encoder (#6: hybrid CNN+Swin)
        self.img_encoder = HybridEncoderV4(C_feat, use_swin)
        
        # SNN bottleneck (#1: ternary, #4: T=16)
        self.snn_encoder = SNNBottleneckEncoderV4(C_feat, C_tx, T)
        
        # Scoring and masking
        self.scorer = ContentScorer(C_tx, hidden=64)
        self.masker = TopKBlockMask(target_cbr, 0.5)
        
        # Channels (ternary-compatible)
        self.awgn = AWGNChannel()
        self.bsc = TernaryBSCChannel()
        
        # SNN decoder (#2: membrane shortcuts)
        self.snn_decoder = SNNBottleneckDecoderV4(C_feat, C_tx, T)
        
        # Image decoder (#6: hybrid CNN+Swin)
        self.img_decoder = HybridDecoderV4(C_feat, use_swin)
        
        # Diffusion refiner (#5)
        self.use_diffusion = use_diffusion
        if use_diffusion:
            self.diffusion = DiffusionRefiner(channels=64, n_steps=8)
    
    def forward(self, img, noise_param=0.0, channel='awgn',
                use_masking=True, target_cbr_override=None,
                use_diffusion=False):
        # Encode
        feat = self.img_encoder(img)
        spikes = self.snn_encoder(feat)
        
        # Score and mask
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
        
        # Channel
        if channel == 'awgn':
            recv = [self.awgn(s, noise_param) for s in masked_spikes]
        elif channel == 'bsc':
            recv = [self.bsc(s, noise_param) for s in masked_spikes]
        else:
            recv = masked_spikes
        
        # Decode
        feat_recon = self.snn_decoder(recv)
        img_recon = self.img_decoder(feat_recon)
        
        # Diffusion refinement (inference only by default)
        if use_diffusion and self.use_diffusion and not self.training:
            img_recon = self.diffusion(img_recon, training=False)
        
        return img_recon, {
            'importance': importance,
            'actual_cbr': actual_cbr,
            'spikes': spikes,
            'feat': feat,
            'feat_recon': feat_recon,
        }
