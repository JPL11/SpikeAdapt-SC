#!/usr/bin/env python3
"""SpikeAdapt-SC Image Codec: Spiking image transmission with variable-rate masking.

Architecture:
  Image (3×H×W)
   → CNN Encoder (strided convs, 16× downscale → C×H/16×W/16)
   → SNN Bottleneck (LIF + MPBN, binary spikes)
   → Variable-Rate Scorer (entropy-guided, noise-conditioned)
   → Rate Mask (variable channels per block)
   → Channel (AWGN or BSC)
   → SNN Decoder (IHF + spike converter)
   → CNN Decoder (transposed convs, upscale → 3×H×W)
   → Reconstructed Image

Key innovations:
  1. Binary spike bottleneck for BSC robustness
  2. Variable-rate masking: allocates more channels to important blocks
  3. Entropy-guided scoring from spike-rate variance
  4. Reconstruction loss (MSE + MS-SSIM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================================
# SNN Primitives (reused from existing codebase)
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
# CNN Encoder (Image → Feature Map, 16× downscale)
# =============================================================================

class ImageEncoder(nn.Module):
    """CNN encoder: 3×H×W → C_feat×H/16×W/16"""
    def __init__(self, C_feat=256):
        super().__init__()
        self.net = nn.Sequential(
            # 3 → 64, stride 2 (H/2)
            nn.Conv2d(3, 64, 5, 2, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 → 128, stride 2 (H/4)
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 → 256, stride 2 (H/8)
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 → C_feat, stride 2 (H/16)
            nn.Conv2d(256, C_feat, 3, 2, 1),
            nn.BatchNorm2d(C_feat),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, img):
        return self.net(img)


# =============================================================================
# CNN Decoder (Feature Map → Reconstructed Image)
# =============================================================================

class ImageDecoder(nn.Module):
    """CNN decoder: C_feat×H/16×W/16 → 3×H×W"""
    def __init__(self, C_feat=256):
        super().__init__()
        self.net = nn.Sequential(
            # C_feat → 256, upsample 2×
            nn.ConvTranspose2d(C_feat, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 → 128, upsample 2×
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 → 64, upsample 2×
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 → 3, upsample 2×
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Sigmoid(),  # Output in [0, 1]
        )
    
    def forward(self, feat):
        return self.net(feat)


# =============================================================================
# SNN Bottleneck Encoder
# =============================================================================

class SNNBottleneckEncoder(nn.Module):
    """Feature → Binary spikes via LIF neurons."""
    def __init__(self, C_feat=256, C_tx=48, T=4):
        super().__init__()
        self.T = T
        self.conv1 = nn.Conv2d(C_feat, 128, 3, 1, 1)
        self.mpbn1 = MPBN(128, T)
        self.lif1 = LIFNeuron(128)
        self.conv2 = nn.Conv2d(128, C_tx, 3, 1, 1)
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
        return spikes  # list of T tensors, each B×C_tx×H×W


# =============================================================================
# SNN Bottleneck Decoder
# =============================================================================

class SNNBottleneckDecoder(nn.Module):
    """Received spikes → Feature map reconstruction."""
    def __init__(self, C_feat=256, C_tx=48, T=4):
        super().__init__()
        self.T = T
        self.conv3 = nn.Conv2d(C_tx, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, C_feat, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(C_feat)
        self.ihf = IHFNeuron()
        # Spike-to-feature converter
        self.converter = nn.Linear(2 * T, 2 * T)

    def forward(self, recv_spikes):
        m3, m4 = None, None
        Fs, Fm = [], []
        for t in range(self.T):
            s3 = F.relu(self.bn3(self.conv3(recv_spikes[t])))
            sp, m4 = self.ihf(self.bn4(self.conv4(s3)), m4)
            Fs.append(sp)
            Fm.append(m4.clone())
        # Converter
        il = []
        for t in range(self.T):
            il.extend([Fs[t], Fm[t]])
        stk = torch.stack(il, dim=1)  # B × 2T × C × H × W
        x = stk.permute(0, 2, 3, 4, 1)  # B × C × H × W × 2T
        return (x * torch.sigmoid(self.converter(x))).sum(dim=-1)


# =============================================================================
# Variable-Rate Entropy-Guided Scorer
# =============================================================================

class ContentScorer(nn.Module):
    """Content-aware importance scorer for spatial blocks.
    
    Outputs a single importance score per spatial block.
    Uses spike-rate statistics as content complexity proxy.
    Noise-conditioned to adapt mask to channel conditions.
    """
    def __init__(self, C_tx=48, hidden=64):
        super().__init__()
        self.C_tx = C_tx
        
        # Content branch: spike statistics → importance
        self.content_net = nn.Sequential(
            nn.Conv2d(C_tx + 1, hidden, 1),  # +1 for variance map
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),  # Single importance score per block
        )
        
        # Noise conditioning
        self.noise_gate = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.noise_gate[-1].weight)
        nn.init.zeros_(self.noise_gate[-1].bias)
    
    def forward(self, spikes, noise_param):
        """Returns importance map B×1×H×W in [0,1]."""
        B = spikes[0].size(0)
        
        # Spike rate: average over time → B×C×H×W
        spike_stack = torch.stack(spikes, dim=0)  # T×B×C×H×W
        rate = spike_stack.float().mean(dim=0)  # B×C×H×W
        
        # Spike-rate variance across channels → complexity proxy
        rate_var = rate.var(dim=1, keepdim=True)  # B×1×H×W
        
        # Content features
        content_input = torch.cat([rate, rate_var], dim=1)  # B×(C+1)×H×W
        logits = self.content_net(content_input)  # B×1×H×W
        
        # Noise conditioning
        noise_input = torch.full((B, 1), float(noise_param), device=rate.device)
        noise_bias = self.noise_gate(noise_input)  # B×1
        logits = logits + noise_bias.unsqueeze(-1).unsqueeze(-1)
        
        importance = torch.sigmoid(logits)  # B×1×H×W in [0,1]
        return importance
    
    def compute_diversity_loss(self, spikes, noise_high=0.0, noise_low=0.3):
        imp_h = self.forward(spikes, noise_high)
        imp_l = self.forward(spikes, noise_low)
        return -torch.mean((imp_h - imp_l).pow(2))


class TopKBlockMask(nn.Module):
    """Binary spatial block masking with top-k selection.
    
    Training: Gumbel-sigmoid + STE for differentiable sampling.
    Eval: Deterministic top-k by importance score.
    Gives exact CBR control.
    """
    def __init__(self, target_rate=0.5, temperature=0.5):
        super().__init__()
        self.target_rate = target_rate
        self.temperature = temperature
    
    def forward(self, spikes, importance, training=True, target_rate_override=None):
        """Apply binary spatial mask to spike tensor.
        
        Args:
            spikes: list of T tensors, each B×C×H×W
            importance: B×1×H×W importance scores
            training: use Gumbel-sigmoid if True, top-k if False
            target_rate_override: override target rate
        Returns:
            masked_spikes: list of T masked tensors
            actual_rate: float
        """
        B, _, H, W = importance.shape
        target = target_rate_override if target_rate_override is not None else self.target_rate
        
        if training:
            # Gumbel-sigmoid with STE
            logits = torch.log(importance / (1 - importance + 1e-7) + 1e-7)
            u = torch.rand_like(logits).clamp(1e-7, 1 - 1e-7)
            soft = torch.sigmoid((logits - torch.log(-torch.log(u))) / self.temperature)
            hard = (soft > 0.5).float()
            mask = hard + (soft - soft.detach())  # STE
            actual_rate = mask.mean().item()
        else:
            # Deterministic top-k
            k = max(1, int(target * H * W))
            flat = importance.view(B, -1)  # B × (H*W)
            _, idx = flat.topk(k, dim=1)
            mask = torch.zeros_like(flat)
            mask.scatter_(1, idx, 1.0)
            mask = mask.view(B, 1, H, W)
            actual_rate = k / (H * W)
        
        # Apply mask: zero out entire block (all channels) for dropped blocks
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
        noisy = x * (1 - flip) + (1 - x) * flip  # flip bits
        if self.training:
            return x + (noisy - x).detach()  # STE
        return noisy


# =============================================================================
# Full Image Codec
# =============================================================================

class SpikeImageCodec(nn.Module):
    """Complete SpikeAdapt-SC Image Codec.
    
    End-to-end: Image → CNN Encoder → SNN Bottleneck → Variable-Rate Mask
    → Channel → SNN Decoder → CNN Decoder → Reconstructed Image
    """
    def __init__(self, C_feat=256, C_tx=48, T=4, target_cbr=0.5):
        super().__init__()
        self.T = T
        self.C_tx = C_tx
        self.target_cbr = target_cbr
        
        # Encoder path
        self.img_encoder = ImageEncoder(C_feat)
        self.snn_encoder = SNNBottleneckEncoder(C_feat, C_tx, T)
        
        # Scoring and masking
        self.scorer = ContentScorer(C_tx, hidden=64)
        self.masker = TopKBlockMask(target_cbr, 0.5)
        
        # Channel
        self.awgn = AWGNChannel()
        self.bsc = BSCChannel()
        
        # Decoder path
        self.snn_decoder = SNNBottleneckDecoder(C_feat, C_tx, T)
        self.img_decoder = ImageDecoder(C_feat)
    
    def forward(self, img, noise_param=0.0, channel='awgn', 
                use_masking=True, target_cbr_override=None):
        # Encode image to features
        feat = self.img_encoder(img)
        
        # SNN encode to spikes
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
        
        # Transmit through channel
        if channel == 'awgn':
            recv = [self.awgn(s, noise_param) for s in masked_spikes]
        elif channel == 'bsc':
            recv = [self.bsc(s, noise_param) for s in masked_spikes]
        else:
            recv = masked_spikes
        
        # Decode
        feat_recon = self.snn_decoder(recv)
        img_recon = self.img_decoder(feat_recon)
        
        return img_recon, {
            'importance': importance,
            'actual_cbr': actual_cbr,
            'spikes': spikes,
        }
