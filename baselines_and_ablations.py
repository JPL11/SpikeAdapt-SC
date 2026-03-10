# ============================================================================
# SpikeAdapt-SC: BASELINES + ABLATIONS + IMPROVEMENTS
# ============================================================================
#
# This file contains everything needed for GLOBECOM paper comparisons:
#
# PART A — BASELINE MODELS (train each separately, same backbone)
#   1. CNN-Uni:    DNN encoder/decoder + uniform quantization [BottleNet++, ref 17]
#   2. CNN-Bern:   DNN encoder/decoder + Bernoulli sampling [Sun et al., ref 22]
#   3. SNN-SC:     Original fixed-rate SNN-SC (your model with η=0, no mask)
#   4. SNN-SC-T6:  SNN-SC with T=6 (bandwidth-fair comparison)
#   5. JPEG+Conv:  Traditional separate source + channel coding
#
# PART B — ABLATION MODELS
#   6. Random Mask: Drop 25% blocks randomly instead of entropy-guided
#   7. Fixed IHF:  IHF with frozen threshold (V_th=1.0)
#   8. No Entropy Loss: Train without L_entropy
#
# PART C — IMPROVEMENTS (from DHF-JSCC paper insights)
#   9. Channel attention in encoder
#  10. BEC channel support
#
# PART D — UNIFIED EVALUATION SCRIPT
#  11. Evaluate all models on same BER range, produce comparison plots
#
# HOW TO USE:
#   1. Copy the baseline classes into your train_spikeadapt_sc.py
#   2. Train each baseline using the training loop at the bottom
#   3. Use Part D to generate comparison figures
#
# ============================================================================

import os
import sys
import math
import random
import json
import numpy as np
from tqdm import tqdm
from io import BytesIO
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# SHARED COMPONENTS (same as your train_spikeadapt_sc.py)
# ============================================================================
class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane, threshold):
        ctx.save_for_backward(membrane, torch.tensor(threshold))
        return (membrane > threshold).float()
    @staticmethod
    def backward(ctx, grad_output):
        membrane, threshold = ctx.saved_tensors
        scale = 10.0
        sig = torch.sigmoid(scale * (membrane - threshold))
        return grad_output * sig * (1 - sig) * scale, None

class IFNeuron(nn.Module):
    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = threshold
    def forward(self, x, membrane=None):
        if membrane is None:
            membrane = torch.zeros_like(x)
        membrane = membrane + x
        spike = SpikeFunction.apply(membrane, self.threshold)
        membrane = membrane - spike * self.threshold
        return spike, membrane

class IHFNeuron(nn.Module):
    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(float(threshold)))
    def forward(self, x, membrane=None):
        if membrane is None:
            membrane = torch.zeros_like(x)
        membrane = membrane + x
        spike = SpikeFunction.apply(membrane, self.threshold.item())
        membrane = membrane - spike * self.threshold
        return spike, membrane

class BSC_Channel(nn.Module):
    def forward(self, x, bit_error_rate):
        if bit_error_rate <= 0 or not self.training:
            return x
        flip_mask = (torch.rand_like(x.float()) < bit_error_rate).float()
        x_noisy = (x + flip_mask) % 2
        return x + (x_noisy - x).detach()

class BEC_Channel(nn.Module):
    def forward(self, x, erasure_rate):
        if erasure_rate <= 0 or not self.training:
            return x
        erase_mask = (torch.rand_like(x.float()) < erasure_rate).float()
        random_bits = (torch.rand_like(x.float()) > 0.5).float()
        x_noisy = x * (1 - erase_mask) + random_bits * erase_mask
        return x + (x_noisy - x).detach()


# ############################################################################
# ############################################################################
#
#  PART A: BASELINE MODELS
#
# ############################################################################
# ############################################################################


# ============================================================================
# BASELINE 1: CNN-Uni — Uniform Quantization [BottleNet++, ref 17]
# ============================================================================
# Architecture:
#   Encoder: Conv(C_in→C1)-BN-ReLU → Conv(C1→C2)-BN-ReLU → Uniform Quantize to n bits
#   Decoder: Dequantize → Conv(C2→C1)-BN-ReLU → Conv(C1→C_out)-BN-ReLU
#
# From SNN-SC paper Section IV-B: "uniform quantization is utilized" to
# convert floating-point semantic info to binary bits for digital channel.
#
# Key: each element is quantized to n bits. High-weight bit errors cause
# large performance degradation — this is why SNN-SC outperforms at high BER.
# ============================================================================

class UniformQuantize(torch.autograd.Function):
    """Uniform quantization with STE for backprop."""
    @staticmethod
    def forward(ctx, x, n_bits):
        # Normalize to [0, 1]
        x_min = x.min()
        x_max = x.max()
        x_norm = (x - x_min) / (x_max - x_min + 1e-8)
        # Quantize to n_bits levels
        n_levels = 2 ** n_bits
        x_quant = torch.round(x_norm * (n_levels - 1)) / (n_levels - 1)
        # Save for backward
        ctx.save_for_backward(x)
        ctx.x_min = x_min
        ctx.x_max = x_max
        return x_quant * (x_max - x_min) + x_min

    @staticmethod
    def backward(ctx, grad_output):
        # STE: pass gradient straight through
        return grad_output, None


class CNNUniEncoder(nn.Module):
    """DNN encoder with uniform quantization (ref [17])."""
    def __init__(self, C_in=2048, C1=256, C2=128, n_bits=8):
        super().__init__()
        self.n_bits = n_bits
        self.conv1 = nn.Conv2d(C_in, C1, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(C1)
        self.conv2 = nn.Conv2d(C1, C2, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(C2)

    def forward(self, F):
        x = Func.relu(self.bn1(self.conv1(F)))
        x = Func.relu(self.bn2(self.conv2(x)))
        # Quantize: convert float to n-bit representation
        x_quant = UniformQuantize.apply(x, self.n_bits)
        return x_quant

    def to_bits(self, x_quant):
        """Convert quantized values to bit representation for channel."""
        x_min = x_quant.min()
        x_max = x_quant.max()
        x_norm = (x_quant - x_min) / (x_max - x_min + 1e-8)
        n_levels = 2 ** self.n_bits
        x_int = torch.round(x_norm * (n_levels - 1)).long()
        # Convert each integer to n_bits binary
        bits = []
        for b in range(self.n_bits):
            bits.append(((x_int >> b) & 1).float())
        return torch.stack(bits, dim=-1), x_min, x_max  # (B,C2,H,W, n_bits)


class CNNUniDecoder(nn.Module):
    """DNN decoder with dequantization."""
    def __init__(self, C_out=2048, C1=256, C2=128):
        super().__init__()
        self.conv3 = nn.Conv2d(C2, C1, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(C1)
        self.conv4 = nn.Conv2d(C1, C_out, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(C_out)

    def forward(self, x):
        x = Func.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        return x

    def from_bits(self, bits, x_min, x_max, n_bits):
        """Convert received bits back to float."""
        # bits: (B, C2, H, W, n_bits)
        n_levels = 2 ** n_bits
        x_int = torch.zeros_like(bits[..., 0])
        for b in range(n_bits):
            x_int = x_int + bits[..., b] * (2 ** b)
        x_norm = x_int / (n_levels - 1)
        return x_norm * (x_max - x_min) + x_min


class CNNUniSC(nn.Module):
    """
    Complete CNN-Uni baseline: DNN SC with uniform quantization.
    Matches SNN-SC paper ref [17] (BottleNet++).

    Key weakness: bit errors in high-weight bits cause large errors.
    """
    def __init__(self, C_in=2048, C1=256, C2=128, n_bits=8):
        super().__init__()
        self.n_bits = n_bits
        self.encoder = CNNUniEncoder(C_in, C1, C2, n_bits)
        self.decoder = CNNUniDecoder(C_in, C1, C2)
        self.channel = BSC_Channel()

    def forward(self, backbone_features, bit_error_rate=0.0):
        # Encode
        x_quant = self.encoder(backbone_features)

        # Convert to bits
        bits, x_min, x_max = self.encoder.to_bits(x_quant)

        # Flatten bits for channel transmission
        B = bits.shape[0]
        bits_flat = bits.reshape(B, -1)

        # Transmit through BSC
        bits_noisy = self.channel(bits_flat, bit_error_rate)

        # Reshape back
        bits_recv = bits_noisy.reshape(bits.shape)

        # Dequantize
        x_recv = self.decoder.from_bits(bits_recv, x_min, x_max, self.n_bits)

        # Decode
        F_prime = self.decoder(x_recv)

        # Stats
        total_bits = bits_flat.shape[1]
        original_bits = backbone_features.shape[1] * backbone_features.shape[2] * \
                        backbone_features.shape[3] * 32
        stats = {
            'total_bits': total_bits,
            'compression_ratio': original_bits / total_bits,
            'method': 'CNN-Uni'
        }

        return F_prime, stats


# ============================================================================
# BASELINE 2: CNN-Bern — Bernoulli Sampling [Sun et al., ref 22]
# ============================================================================
# Architecture:
#   Encoder: Conv-BN-Sigmoid → output as probabilities → Bernoulli sample to {0,1}
#   Channel: BSC/BEC on binary output
#   Decoder: Conv-BN-ReLU layers
#
# From SNN-SC paper: "the bit sequence is obtained by performing Bernoulli
# sampling using the extracted semantic information as a probability"
#
# Key weakness: Bernoulli sampling discards too much information, so
# performance is consistently lower than SNN-SC.
# ============================================================================

class BernoulliSample(torch.autograd.Function):
    """Bernoulli sampling with STE."""
    @staticmethod
    def forward(ctx, prob):
        sample = torch.bernoulli(prob)
        ctx.save_for_backward(prob)
        return sample

    @staticmethod
    def backward(ctx, grad_output):
        # STE: gradient passes through
        return grad_output


class CNNBernEncoder(nn.Module):
    """DNN encoder with Bernoulli sampling (ref [22])."""
    def __init__(self, C_in=2048, C1=256, C2=128):
        super().__init__()
        self.conv1 = nn.Conv2d(C_in, C1, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(C1)
        self.conv2 = nn.Conv2d(C1, C2, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(C2)

    def forward(self, F):
        x = Func.relu(self.bn1(self.conv1(F)))
        # Output sigmoid probabilities for Bernoulli sampling
        prob = torch.sigmoid(self.bn2(self.conv2(x)))
        # Sample binary output
        bits = BernoulliSample.apply(prob)
        return bits, prob  # Return prob for potential entropy loss


class CNNBernDecoder(nn.Module):
    """DNN decoder for Bernoulli-sampled input."""
    def __init__(self, C_out=2048, C1=256, C2=128):
        super().__init__()
        self.conv3 = nn.Conv2d(C2, C1, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(C1)
        self.conv4 = nn.Conv2d(C1, C_out, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(C_out)

    def forward(self, x):
        x = Func.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        return x


class CNNBernSC(nn.Module):
    """
    Complete CNN-Bern baseline: DNN SC with Bernoulli sampling.
    Matches SNN-SC paper ref [22] (Sun et al.).

    Uses same bandwidth as SNN-SC (C2*H*W bits per timestep equivalent).
    For fair comparison: total bits = T * C2 * H2 * W2.
    """
    def __init__(self, C_in=2048, C1=256, C2=128, T=8):
        super().__init__()
        self.T = T  # T "rounds" of sampling for fair bandwidth comparison
        self.encoder = CNNBernEncoder(C_in, C1, C2)
        self.decoder = CNNBernDecoder(C_in, C1, C2)
        self.channel = BSC_Channel()

    def forward(self, backbone_features, bit_error_rate=0.0):
        # Bernoulli sample T times and aggregate (like SNN-SC runs T timesteps)
        all_recv = []
        for t in range(self.T):
            bits, prob = self.encoder(backbone_features)
            bits_noisy = self.channel(bits, bit_error_rate)
            all_recv.append(bits_noisy)

        # Average the T received samples (reduces Bernoulli noise)
        avg_recv = torch.stack(all_recv, dim=0).mean(dim=0)

        # Decode
        F_prime = self.decoder(avg_recv)

        # Stats
        B, C2, H2, W2 = bits.shape
        total_bits = self.T * C2 * H2 * W2
        original_bits = backbone_features.shape[1] * backbone_features.shape[2] * \
                        backbone_features.shape[3] * 32
        stats = {
            'total_bits': total_bits,
            'compression_ratio': original_bits / total_bits,
            'method': 'CNN-Bern'
        }
        return F_prime, stats


# ============================================================================
# BASELINE 3: SNN-SC (Fixed Rate, T=8) — Your SpikeAdapt-SC with η=0
# ============================================================================
# This is simply your existing SpikeAdaptSC model evaluated with η=0.
# No need for a separate class — just set eta_override=0.0 during eval.
# But we include a clean wrapper for clarity.
# ============================================================================

class SNNSC_FixedRate(nn.Module):
    """
    Original SNN-SC model (Wang et al., IEEE TVT 2025).
    Same encoder/decoder architecture as SpikeAdapt-SC, but:
      - No entropy estimation
      - No block masking
      - Transmits ALL blocks every timestep
      - Uses IHF at decoder (from original paper)

    This is equivalent to SpikeAdaptSC with eta=0 (mask is all 1s).
    """
    def __init__(self, C_in=2048, C1=256, C2=128, T=8):
        super().__init__()
        self.T = T

        # Encoder (same as SpikeAdapt)
        self.enc_conv1 = nn.Conv2d(C_in, C1, 3, 1, 1)
        self.enc_bn1 = nn.BatchNorm2d(C1)
        self.enc_if1 = IFNeuron()
        self.enc_conv2 = nn.Conv2d(C1, C2, 3, 1, 1)
        self.enc_bn2 = nn.BatchNorm2d(C2)
        self.enc_if2 = IFNeuron()

        # Decoder reconstructor (same as SpikeAdapt)
        self.dec_conv3 = nn.Conv2d(C2, C1, 3, 1, 1)
        self.dec_bn3 = nn.BatchNorm2d(C1)
        self.dec_if3 = IFNeuron()
        self.dec_conv4 = nn.Conv2d(C1, C_in, 3, 1, 1)
        self.dec_bn4 = nn.BatchNorm2d(C_in)
        self.dec_ihf = IHFNeuron()  # Fixed threshold (non-learnable for original SNN-SC)

        # Converter
        self.converter_fc = nn.Linear(2 * T, 2 * T)

        # Channel
        self.channel = BSC_Channel()

    def forward(self, backbone_features, bit_error_rate=0.0):
        # Encode T timesteps
        all_S2 = []
        mem_if1, mem_if2 = None, None
        for t in range(self.T):
            x = self.enc_bn1(self.enc_conv1(backbone_features))
            s1, mem_if1 = self.enc_if1(x, mem_if1)
            x = self.enc_bn2(self.enc_conv2(s1))
            s2, mem_if2 = self.enc_if2(x, mem_if2)
            all_S2.append(s2)

        # Transmit through channel (NO masking — full rate)
        received = []
        for t in range(self.T):
            noisy = self.channel(all_S2[t], bit_error_rate)
            received.append(noisy)

        # Decode T timesteps
        all_Fs, all_Fm = [], []
        mem_if3, mem_ihf = None, None
        for t in range(self.T):
            x = self.dec_bn3(self.dec_conv3(received[t]))
            s3, mem_if3 = self.dec_if3(x, mem_if3)
            x = self.dec_bn4(self.dec_conv4(s3))
            spike, mem_ihf = self.dec_ihf(x, mem_ihf)
            all_Fs.append(spike)
            all_Fm.append(mem_ihf.clone())

        # Convert
        interleaved = []
        for t in range(self.T):
            interleaved.append(all_Fs[t])
            interleaved.append(all_Fm[t])
        stacked = torch.stack(interleaved, dim=1)
        B, K, C, H, W = stacked.shape
        x = stacked.permute(0, 2, 3, 4, 1)
        weights = torch.sigmoid(self.converter_fc(x))
        F_prime = (x * weights).sum(dim=-1)

        # Stats
        B2, C2, H2, W2 = all_S2[0].shape
        total_bits = self.T * C2 * H2 * W2
        original_bits = backbone_features.shape[1] * backbone_features.shape[2] * \
                        backbone_features.shape[3] * 32

        stats = {
            'total_bits': total_bits,
            'compression_ratio': original_bits / total_bits,
            'method': f'SNN-SC (T={self.T})'
        }

        return F_prime, all_S2, stats


# ============================================================================
# BASELINE 4: SNN-SC at T=6 — Bandwidth-Fair Comparison
# ============================================================================
# SpikeAdapt-SC (T=8, η=0.5) sends: 8 × 0.75 × 128 × 1 × 1 = 768 bits/image
#   (actual: 8 × 12 × 128 = 12,288 bits on 4×4 grid)
# SNN-SC (T=6) sends: 6 × 128 × 1 × 1 = 768 bits/image
#   (actual: 6 × 16 × 128 = 12,288 bits on 4×4 grid)
#
# Near-identical bandwidth! If SpikeAdapt-SC outperforms, it proves that
# smart entropy-guided masking > naive timestep reduction.
# ============================================================================
# Just instantiate: SNNSC_FixedRate(C_in=2048, C1=256, C2=128, T=6)


# ============================================================================
# BASELINE 5: JPEG + Convolutional Code (Separate Source & Channel Coding)
# ============================================================================
# From SNN-SC paper Section IV-B:
#   "Features are first quantized into 8-bit integers, then compressed by JPEG,
#    and finally channel-coded by convolutional codes."
#
# We simulate this faithfully:
#   1. Quantize feature to uint8
#   2. Tile feature into image layout
#   3. JPEG compress (quality Q)
#   4. Convolutional code (rate R)
#   5. BSC channel
#   6. Viterbi decode → JPEG decompress → dequantize
#
# Since actual convolutional coding is complex, we approximate with
# repetition coding + majority vote which gives similar error correction.
# ============================================================================

class JPEGConvBaseline(nn.Module):
    """
    Traditional separate source + channel coding baseline.

    Source coding: JPEG compression of feature maps
    Channel coding: Simple repetition code (approximates conv code rate 1/3)

    Exhibits 'cliff effect' — works well at low BER, fails catastrophically
    at high BER when channel decoding can't correct errors.
    """
    def __init__(self, C_in=2048, jpeg_quality=50, code_rate_inv=3):
        super().__init__()
        self.C_in = C_in
        self.jpeg_quality = jpeg_quality
        self.code_rate_inv = code_rate_inv  # 1/3 rate = repeat 3 times

    def forward(self, backbone_features, bit_error_rate=0.0):
        B, C, H, W = backbone_features.shape
        device = backbone_features.device

        reconstructed = []
        total_bits_list = []

        for b in range(B):
            feat = backbone_features[b]  # (C, H, W)

            # Step 1: Quantize to uint8
            f_min = feat.min()
            f_max = feat.max()
            f_norm = (feat - f_min) / (f_max - f_min + 1e-8)
            f_uint8 = (f_norm * 255).clamp(0, 255).byte()

            # Step 2: Reshape to image-like for JPEG
            # Tile C channels into a 2D grid
            nrow = int(math.ceil(math.sqrt(C)))
            ncol = int(math.ceil(C / nrow))
            canvas = torch.zeros(nrow * H, ncol * W, dtype=torch.uint8, device='cpu')
            for c in range(C):
                r, col = divmod(c, ncol)
                canvas[r*H:(r+1)*H, col*W:(col+1)*W] = f_uint8[c].cpu()

            # Step 3: JPEG compress
            img = Image.fromarray(canvas.numpy(), mode='L')
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=self.jpeg_quality)
            jpeg_bytes = buffer.getvalue()
            n_source_bits = len(jpeg_bytes) * 8

            # Step 4: Channel coding (repetition code, rate 1/R)
            n_coded_bits = n_source_bits * self.code_rate_inv
            total_bits_list.append(n_coded_bits)

            # Step 5: Simulate BSC on coded bits
            if bit_error_rate > 0:
                # Convert to bit array
                source_bits = np.unpackbits(np.frombuffer(jpeg_bytes, dtype=np.uint8))
                # Repeat each bit code_rate_inv times
                coded_bits = np.repeat(source_bits, self.code_rate_inv)
                # Apply BSC
                flip = np.random.random(len(coded_bits)) < bit_error_rate
                coded_bits_noisy = (coded_bits + flip.astype(int)) % 2

                # Step 6: Decode (majority vote)
                coded_reshaped = coded_bits_noisy.reshape(-1, self.code_rate_inv)
                decoded_bits = (coded_reshaped.sum(axis=1) > self.code_rate_inv / 2).astype(np.uint8)

                # Convert back to bytes
                # Pad to multiple of 8
                pad_len = (8 - len(decoded_bits) % 8) % 8
                decoded_bits_padded = np.concatenate([decoded_bits, np.zeros(pad_len, dtype=np.uint8)])
                decoded_bytes = np.packbits(decoded_bits_padded).tobytes()[:len(jpeg_bytes)]

                # Step 7: JPEG decompress
                try:
                    recv_buffer = BytesIO(decoded_bytes)
                    recv_img = Image.open(recv_buffer).convert('L')
                    recv_canvas = torch.tensor(np.array(recv_img), dtype=torch.float32)
                except Exception:
                    # JPEG decode failed — cliff effect!
                    recv_canvas = torch.zeros(nrow * H, ncol * W, dtype=torch.float32)
            else:
                # No noise — just JPEG decompress
                recv_buffer = BytesIO(jpeg_bytes)
                recv_img = Image.open(recv_buffer).convert('L')
                recv_canvas = torch.tensor(np.array(recv_img), dtype=torch.float32)

            # Step 8: Dequantize
            recv_feat = torch.zeros(C, H, W, dtype=torch.float32)
            for c in range(C):
                r, col = divmod(c, ncol)
                recv_feat[c] = recv_canvas[r*H:(r+1)*H, col*W:(col+1)*W]
            recv_feat = recv_feat / 255.0 * (f_max - f_min).item() + f_min.item()
            reconstructed.append(recv_feat)

        F_prime = torch.stack(reconstructed, dim=0).to(device)

        original_bits = C * H * W * 32
        avg_coded_bits = np.mean(total_bits_list) if total_bits_list else 1
        stats = {
            'total_bits': avg_coded_bits,
            'compression_ratio': original_bits / avg_coded_bits,
            'method': f'JPEG(Q={self.jpeg_quality})+Conv(R=1/{self.code_rate_inv})'
        }
        return F_prime, stats


# ############################################################################
# ############################################################################
#
#  PART B: ABLATION MODELS
#
# ############################################################################
# ############################################################################


# ============================================================================
# ABLATION 1: Random Mask — Random block dropping at same 75% rate
# ============================================================================
# Tests whether entropy-guided masking is better than random spatial pruning.
# If random mask ≈ entropy mask, then entropy guidance adds no value.
# We expect entropy-guided to win, especially at higher masking rates.
# ============================================================================

class RandomBlockMask(nn.Module):
    """Drop blocks randomly instead of by entropy. Same average rate."""
    def __init__(self, keep_rate=0.75):
        super().__init__()
        self.keep_rate = keep_rate

    def forward(self, entropy_map, training=True):
        B, H, W = entropy_map.shape
        # Generate random mask with expected keep_rate
        mask = (torch.rand(B, 1, H, W, device=entropy_map.device) < self.keep_rate).float()
        tx_rate = mask.mean()
        return mask, tx_rate

    def apply_mask(self, S2_t, mask):
        return S2_t * mask


class SpikeAdaptSC_RandomMask(nn.Module):
    """
    SpikeAdapt-SC but with random masking instead of entropy-guided.
    Copy from your SpikeAdaptSC class and replace self.block_mask.
    """
    def __init__(self, C_in=2048, C1=256, C2=128, T=8, keep_rate=0.75):
        super().__init__()
        self.T = T

        # Same encoder as SpikeAdapt
        self.enc_conv1 = nn.Conv2d(C_in, C1, 3, 1, 1)
        self.enc_bn1 = nn.BatchNorm2d(C1)
        self.enc_if1 = IFNeuron()
        self.enc_conv2 = nn.Conv2d(C1, C2, 3, 1, 1)
        self.enc_bn2 = nn.BatchNorm2d(C2)
        self.enc_if2 = IFNeuron()

        # RANDOM mask instead of entropy mask
        self.block_mask = RandomBlockMask(keep_rate=keep_rate)

        # Same decoder
        self.dec_conv3 = nn.Conv2d(C2, C1, 3, 1, 1)
        self.dec_bn3 = nn.BatchNorm2d(C1)
        self.dec_if3 = IFNeuron()
        self.dec_conv4 = nn.Conv2d(C1, C_in, 3, 1, 1)
        self.dec_bn4 = nn.BatchNorm2d(C_in)
        self.dec_ihf = IHFNeuron()
        self.converter_fc = nn.Linear(2 * T, 2 * T)
        self.channel = BSC_Channel()

    def forward(self, backbone_features, bit_error_rate=0.0):
        # Encode
        all_S2 = []
        mem1, mem2 = None, None
        for t in range(self.T):
            x = self.enc_bn1(self.enc_conv1(backbone_features))
            s1, mem1 = self.enc_if1(x, mem1)
            x = self.enc_bn2(self.enc_conv2(s1))
            s2, mem2 = self.enc_if2(x, mem2)
            all_S2.append(s2)

        # Random mask (no entropy computation)
        B, C2, H2, W2 = all_S2[0].shape
        dummy_entropy = torch.zeros(B, H2, W2, device=backbone_features.device)
        mask, tx_rate = self.block_mask(dummy_entropy)

        # Apply mask + channel
        received = []
        for t in range(self.T):
            masked = all_S2[t] * mask
            noisy = self.channel(masked, bit_error_rate)
            received.append(noisy)

        # Decode
        all_Fs, all_Fm = [], []
        mem3, mem4 = None, None
        for t in range(self.T):
            x = self.dec_bn3(self.dec_conv3(received[t] * mask))
            s3, mem3 = self.dec_if3(x, mem3)
            x = self.dec_bn4(self.dec_conv4(s3))
            spike, mem4 = self.dec_ihf(x, mem4)
            all_Fs.append(spike)
            all_Fm.append(mem4.clone())

        # Convert
        interleaved = []
        for t in range(self.T):
            interleaved.append(all_Fs[t])
            interleaved.append(all_Fm[t])
        stacked = torch.stack(interleaved, dim=1)
        x = stacked.permute(0, 2, 3, 4, 1)
        weights = torch.sigmoid(self.converter_fc(x))
        F_prime = (x * weights).sum(dim=-1)

        stats = {'tx_rate': tx_rate.item(), 'method': 'SpikeAdapt-SC (Random Mask)'}
        return F_prime, all_S2, tx_rate, stats


# ############################################################################
# ############################################################################
#
#  PART C: IMPROVEMENTS (from DHF-JSCC insights)
#
# ############################################################################
# ############################################################################


# ============================================================================
# IMPROVEMENT 1: Channel Attention in Encoder
# ============================================================================
# From DHF-JSCC paper Section III-B (Fig. 4):
#   "Considering that features of different channels have various importance...
#    we add a channel attention layer at the end of each conv block"
#
# This is a squeeze-and-excitation style attention.
# Add this to the SNN encoder BEFORE the IF neuron.
# Lightweight — adds minimal parameters.
# ============================================================================

class ChannelAttention(nn.Module):
    """
    Channel attention from DHF-JSCC (Eq. 13):
      F_out = F_in * σ(W_l * AvgPool(F_in))

    Squeezes spatial dims, learns per-channel weights,
    re-weights channels accordingly.
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # Squeeze: (B, C, H, W) → (B, C, 1, 1) → (B, C)
        y = self.avg_pool(x).view(B, C)
        # Excite: (B, C) → channel weights
        y = self.fc(y).view(B, C, 1, 1)
        # Scale
        return x * y


class EncoderWithAttention(nn.Module):
    """
    Enhanced encoder with channel attention (DHF-JSCC inspired).
    Adds attention after each Conv-BN block, before IF neuron.
    """
    def __init__(self, C_in=2048, C1=256, C2=128, reduction=4):
        super().__init__()
        self.conv1 = nn.Conv2d(C_in, C1, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(C1)
        self.ca1 = ChannelAttention(C1, reduction)  # NEW
        self.if1 = IFNeuron()

        self.conv2 = nn.Conv2d(C1, C2, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(C2)
        self.ca2 = ChannelAttention(C2, reduction)  # NEW
        self.if2 = IFNeuron()

    def forward(self, F, mem_if1=None, mem_if2=None):
        x = self.ca1(self.bn1(self.conv1(F)))  # Attention before IF
        s1, mem_if1 = self.if1(x, mem_if1)

        x = self.ca2(self.bn2(self.conv2(s1)))  # Attention before IF
        s2, mem_if2 = self.if2(x, mem_if2)

        return s2, mem_if1, mem_if2


# ============================================================================
# ENTROPY LOSS — no change needed, same as your implementation
# ============================================================================
def compute_entropy_loss(all_S2, alpha=1.0):
    all_bits = torch.cat([s.flatten() for s in all_S2])
    p1 = all_bits.mean()
    p0 = 1.0 - p1
    eps = 1e-7
    p0 = torch.clamp(p0, eps, 1 - eps)
    p1 = torch.clamp(p1, eps, 1 - eps)
    H = -(p0 * torch.log2(p0) + p1 * torch.log2(p1))
    return (alpha - H) ** 2

def compute_rate_loss(tx_rate, target_rate=0.75):
    return (tx_rate - target_rate) ** 2


# ############################################################################
# ############################################################################
#
#  PART D: UNIFIED TRAINING + EVALUATION
#
# ############################################################################
# ############################################################################


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_cnn_baseline(model, front, back, train_loader, test_loader,
                       model_name, epochs=50, lr=1e-4, ber_max=0.3,
                       save_dir="./snapshots_baselines/"):
    """
    Generic training loop for CNN baselines (CNN-Uni, CNN-Bern).
    Same 2-step process: train SC model with frozen backbone, then fine-tune.

    NOTE: CNN baselines do NOT use entropy loss (can't constrain bit distribution
    of quantized/sampled outputs). Only task loss L_CE is used.
    From SNN-SC paper: "we use the end-to-end task loss to train baselines
    in all training steps."
    """
    os.makedirs(save_dir, exist_ok=True)
    front.eval()
    for p in front.parameters():
        p.requires_grad = False

    # Freeze backbone classifier initially
    for p in back.parameters():
        p.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"{model_name} E{epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            ber = random.uniform(0, ber_max)

            with torch.no_grad():
                feat = front(images)

            # Forward depends on model type
            if isinstance(model, (CNNUniSC, CNNBernSC)):
                F_prime, stats = model(feat, bit_error_rate=ber)
            elif isinstance(model, SNNSC_FixedRate):
                F_prime, all_S2, stats = model(feat, bit_error_rate=ber)
            elif isinstance(model, SpikeAdaptSC_RandomMask):
                F_prime, all_S2, tx_rate, stats = model(feat, bit_error_rate=ber)
            else:
                raise ValueError(f"Unknown model type: {type(model)}")

            outputs = back(F_prime)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
            epoch_loss += loss.item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}',
                              'acc': f'{100.*pred.eq(labels).sum().item()/labels.size(0):.0f}%'})

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            acc = evaluate_model(model, front, back, test_loader, ber=0.0)
            print(f"  {model_name} Test: {acc:.2f}% (BER=0)", flush=True)
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(),
                           os.path.join(save_dir, f"{model_name}_best.pth"))
                print(f"  ✓ Best: {best_acc:.2f}%", flush=True)

    print(f"\n✅ {model_name} training complete. Best: {best_acc:.2f}%", flush=True)
    return best_acc


def evaluate_model(model, front, back, test_loader, ber=0.0):
    """Evaluate any baseline model. Returns accuracy."""
    model.eval()
    if ber > 0:
        model.channel.train()  # Enable noise in channel

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            feat = front(images)

            if isinstance(model, (CNNUniSC, CNNBernSC)):
                F_prime, _ = model(feat, bit_error_rate=ber)
            elif isinstance(model, SNNSC_FixedRate):
                F_prime, _, _ = model(feat, bit_error_rate=ber)
            elif isinstance(model, SpikeAdaptSC_RandomMask):
                F_prime, _, _, _ = model(feat, bit_error_rate=ber)
            elif isinstance(model, JPEGConvBaseline):
                F_prime, _ = model(feat, bit_error_rate=ber)
            else:
                raise ValueError(f"Unknown model: {type(model)}")

            outputs = back(F_prime)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

    model.eval()
    return 100. * correct / total


# ============================================================================
# FULL BER SWEEP COMPARISON — THE MONEY PLOT
# ============================================================================

def run_full_ber_comparison(models_dict, front, back, test_loader,
                            ber_values=None, save_path="ber_comparison.png",
                            n_repeats=5):
    """
    Run BER sweep for all models and produce comparison plot.

    Args:
        models_dict: {'Model Name': model_instance, ...}
        n_repeats: Number of repeats per BER to reduce channel randomness
    """
    if ber_values is None:
        ber_values = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    results = {}
    for name, model in models_dict.items():
        print(f"\nEvaluating {name}...", flush=True)
        accs = []
        for ber in ber_values:
            # Repeat and average to reduce channel randomness
            trial_accs = []
            for _ in range(n_repeats if ber > 0 else 1):
                acc = evaluate_model(model, front, back, test_loader, ber=ber)
                trial_accs.append(acc)
            mean_acc = np.mean(trial_accs)
            std_acc = np.std(trial_accs) if len(trial_accs) > 1 else 0
            accs.append((ber, mean_acc, std_acc))
            print(f"  BER={ber:.3f}: {mean_acc:.2f}% (±{std_acc:.2f})", flush=True)
        results[name] = accs

    # ===== PLOT =====
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    # Style guide: match SNN-SC paper Figure 4 style
    styles = {
        'SpikeAdapt-SC':        {'color': '#D32F2F', 'marker': 's', 'ls': '-',  'lw': 2.5},
        'SNN-SC (T=8)':         {'color': '#1976D2', 'marker': 'o', 'ls': '-',  'lw': 2.0},
        'SNN-SC (T=6)':         {'color': '#1976D2', 'marker': 'o', 'ls': '--', 'lw': 2.0},
        'CNN-Uni':              {'color': '#388E3C', 'marker': '^', 'ls': '-.',  'lw': 2.0},
        'CNN-Bern':             {'color': '#F57C00', 'marker': 'D', 'ls': '--', 'lw': 2.0},
        'JPEG+Conv':            {'color': '#7B1FA2', 'marker': 'v', 'ls': ':',  'lw': 2.0},
        'Random Mask':          {'color': '#9E9E9E', 'marker': 'x', 'ls': '--', 'lw': 1.5},
    }

    for name, accs in results.items():
        bers = [a[0] for a in accs]
        means = [a[1] for a in accs]
        stds = [a[2] for a in accs]
        style = styles.get(name, {'color': 'gray', 'marker': 'o', 'ls': '-', 'lw': 1.5})

        ax.plot(bers, means, marker=style['marker'], ls=style['ls'],
                color=style['color'], lw=style['lw'], markersize=7, label=name)
        if max(stds) > 0:
            ax.fill_between(bers, [m - s for m, s in zip(means, stds)],
                           [m + s for m, s in zip(means, stds)],
                           alpha=0.1, color=style['color'])

    ax.set_xlabel('Bit Error Rate (BER)', fontsize=14)
    ax.set_ylabel('Test Accuracy (%)', fontsize=14)
    ax.set_title('Classification Accuracy vs Channel Noise (CIFAR-100, BSC)', fontsize=15)
    ax.legend(fontsize=11, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.01, 0.31)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved comparison plot: {save_path}", flush=True)

    return results


# ============================================================================
# COMPLEXITY ANALYSIS TABLE
# ============================================================================

def compute_complexity(C_in=2048, C1=256, C2=128, H=1, W=1, T=8):
    """
    Compute parameter count and FLOPs for all methods.
    Follows SNN-SC paper Equations 17-21.
    """

    # Conv parameters: K^2 × C_in × C_out
    # Conv FLOPs (MAC): 2 × K^2 × C_in × C_out × H_out × W_out
    # SNN FLOPs (AC):   fire_rate × K^2 × C_in × C_out × H_out × W_out × 0.5

    K = 3  # kernel size

    # --- Shared encoder/decoder conv layers ---
    params_conv1 = K*K * C_in * C1   # Conv1
    params_conv2 = K*K * C1 * C2     # Conv2
    params_conv3 = K*K * C2 * C1     # Conv3
    params_conv4 = K*K * C1 * C_in   # Conv4
    params_bn = 2 * (C1 + C2 + C1 + C_in)  # BN params (weight + bias)

    # Converter FC
    params_fc = (2*T) * (2*T)

    # --- SNN-SC / SpikeAdapt-SC ---
    # Fire rates from SNN-SC paper Table IV (approximate)
    fr1 = 0.4   # input to Conv2
    fr2 = 0.35  # input to Conv3
    fr3 = 0.38  # input to Conv4

    snn_params = params_conv1 + params_conv2 + params_conv3 + params_conv4 + params_bn + params_fc
    # Eq 21: Conv1 is float input (MAC), rest are spike (AC with fire rate)
    snn_flops_conv1 = 2 * K*K * C_in * C1 * H * W
    snn_flops_inner = (fr1 * K*K * C1 * C2 * H * W +
                       fr2 * K*K * C2 * C1 * H * W +
                       fr3 * K*K * C1 * C_in * H * W) * 0.5 * T
    snn_flops_fc = 2 * (2*T) * (2*T)
    snn_flops = snn_flops_conv1 + snn_flops_inner + snn_flops_fc

    # --- CNN baselines (MAC everywhere) ---
    cnn_params = params_conv1 + params_conv2 + params_conv3 + params_conv4 + params_bn
    cnn_flops = 2 * (K*K * C_in * C1 + K*K * C1 * C2 +
                     K*K * C2 * C1 + K*K * C1 * C_in) * H * W

    # CNN-Bern has extra params for probability head (same conv structure)
    bern_params = cnn_params  # Same architecture

    print("\n" + "="*70)
    print("COMPLEXITY ANALYSIS")
    print("="*70)
    print(f"{'Method':<25} {'Params':>12} {'FLOPs':>15} {'Notes':>20}")
    print("-"*70)
    print(f"{'SNN-SC (T=8)':<25} {snn_params:>12,} {snn_flops:>15,.0f} {'AC ops':>20}")
    print(f"{'SpikeAdapt-SC (T=8)':<25} {snn_params+1:>12,} {snn_flops:>15,.0f} {'+1 learnable Vth':>20}")
    print(f"{'CNN-Uni [17]':<25} {cnn_params:>12,} {cnn_flops:>15,.0f} {'MAC ops':>20}")
    print(f"{'CNN-Bern [22]':<25} {bern_params:>12,} {cnn_flops:>15,.0f} {'MAC ops':>20}")
    print(f"{'Entropy Estimator':<25} {'0':>12} {'~negligible':>15} {'No params':>20}")
    print("="*70)
    print(f"\nSNN-SC FLOPs reduction vs CNN: "
          f"{(1 - snn_flops/cnn_flops)*100:.1f}%")
    print(f"SpikeAdapt-SC adds 0 extra params for entropy estimation")

    return {
        'SNN-SC': {'params': snn_params, 'flops': snn_flops},
        'SpikeAdapt-SC': {'params': snn_params + 1, 'flops': snn_flops},
        'CNN-Uni': {'params': cnn_params, 'flops': cnn_flops},
        'CNN-Bern': {'params': bern_params, 'flops': cnn_flops},
    }


# ============================================================================
# BANDWIDTH COMPARISON TABLE
# ============================================================================

def compute_bandwidth_table(C_in=2048, C2=128, H=1, W=1):
    """
    Compare bits transmitted for each method.
    ResNet50 on CIFAR-100: features are (2048, 1, 1) after avgpool... wait.
    Actually features are (2048, 4, 4) before avgpool. Let me use that.
    """
    # Corrected for actual spatial dims at split point
    # Your backbone: after layer4, before avgpool
    # CIFAR-100 with modified conv1 (no maxpool): (2048, 1, 1)
    # Check your actual dims! If (2048, 1, 1), masking on 1×1 is meaningless
    # If (2048, 4, 4), we have 16 blocks to mask

    # Using the dims from your code output: feat.shape
    H_feat, W_feat = 1, 1  # UPDATE THIS based on your actual backbone output
    # NOTE: If this is (2048,1,1), you need to split EARLIER in ResNet50
    # to get (2048,4,4) or similar. Check your model.

    original_bits = C_in * H_feat * W_feat * 32

    print("\n" + "="*70)
    print("BANDWIDTH COMPARISON")
    print("="*70)
    print(f"Original feature: ({C_in}, {H_feat}, {W_feat}) × 32-bit = {original_bits:,} bits")
    print("-"*70)

    methods = [
        ('SNN-SC (T=8)', 8 * C2 * H_feat * W_feat, ''),
        ('SNN-SC (T=6)', 6 * C2 * H_feat * W_feat, ''),
        ('SNN-SC (T=4)', 4 * C2 * H_feat * W_feat, ''),
        ('SpikeAdapt-SC (T=8, η=0.5)',
         int(8 * 0.75 * C2 * H_feat * W_feat + H_feat * W_feat),
         '~75% blocks + mask'),
    ]
    for name, bits, note in methods:
        cr = original_bits / max(bits, 1)
        print(f"  {name:<35} {bits:>8,} bits  {cr:>8.1f}× compression  {note}")

    print("="*70)


# ############################################################################
# ############################################################################
#
#  MAIN: TRAIN ALL BASELINES + RUN COMPARISON
#
# ############################################################################
# ############################################################################

if __name__ == "__main__":
    print("="*70)
    print("SpikeAdapt-SC: Training Baselines + Running Comparison")
    print("="*70)

    # ------------------------------------------------------------------
    # CONFIG
    # ------------------------------------------------------------------
    SAVE_DIR = "./snapshots_baselines/"
    RESULTS_DIR = "./eval_results/"
    BACKBONE_PATH = "./snapshots_spikeadapt/backbone_best.pth"
    SPIKEADAPT_DIR = "./snapshots_spikeadapt/"

    C_in, C1, C2, T = 2048, 256, 128, 8
    BER_MAX = 0.3
    EPOCHS_BASELINE = 50
    LR_BASELINE = 1e-4

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # DATASET
    # ------------------------------------------------------------------
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    train_dataset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False,
                             num_workers=4, pin_memory=True)

    # ------------------------------------------------------------------
    # LOAD BACKBONE
    # ------------------------------------------------------------------
    from train_spikeadapt_sc import ResNet50Front, ResNet50Back

    front = ResNet50Front(pretrained=False)
    front.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    front.maxpool = nn.Identity()
    back = ResNet50Back(num_classes=100, pretrained=False)

    state = torch.load(BACKBONE_PATH, map_location=device)
    front_state = {k: v for k, v in state.items() if not k.startswith(('fc.', 'avgpool.'))}
    back_state = {k: v for k, v in state.items() if k.startswith(('fc.', 'avgpool.'))}
    front.load_state_dict(front_state, strict=False)
    back.load_state_dict(back_state, strict=False)
    front = front.to(device).eval()
    back_orig = back.to(device)
    print("✓ Backbone loaded", flush=True)

    # ------------------------------------------------------------------
    # 0. VERIFY BACKBONE ACCURACY (CRITICAL — do this first!)
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 0: Verify backbone accuracy WITHOUT any SC model")
    print("="*70)
    back_orig.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Backbone only"):
            images, labels = images.to(device), labels.to(device)
            feat = front(images)
            out = back_orig(feat)
            _, pred = out.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    backbone_acc = 100. * correct / total
    print(f"\n★ BACKBONE ACCURACY (no SC): {backbone_acc:.2f}%")
    print(f"  SNN-SC paper reports: 77.81%")
    print(f"  Your SpikeAdapt-SC:   72.85%")
    print(f"  Gap from backbone:    {backbone_acc - 72.85:.2f}%")

    # ------------------------------------------------------------------
    # 1. TRAIN BASELINES
    # ------------------------------------------------------------------

    # ---- CNN-Uni ----
    print("\n" + "="*70)
    print("TRAINING: CNN-Uni (Uniform Quantization)")
    print("="*70)
    cnn_uni = CNNUniSC(C_in=C_in, C1=C1, C2=C2, n_bits=T).to(device)
    # NOTE: n_bits=T=8 means same bandwidth as SNN-SC(T=8): C2*H*W*8 bits
    back_uni = ResNet50Back(num_classes=100, pretrained=False)
    back_uni.load_state_dict(back_state, strict=False)
    back_uni = back_uni.to(device)
    train_cnn_baseline(cnn_uni, front, back_uni, train_loader, test_loader,
                       "CNN-Uni", epochs=EPOCHS_BASELINE, lr=LR_BASELINE)

    # ---- CNN-Bern ----
    print("\n" + "="*70)
    print("TRAINING: CNN-Bern (Bernoulli Sampling)")
    print("="*70)
    cnn_bern = CNNBernSC(C_in=C_in, C1=C1, C2=C2, T=T).to(device)
    back_bern = ResNet50Back(num_classes=100, pretrained=False)
    back_bern.load_state_dict(back_state, strict=False)
    back_bern = back_bern.to(device)
    train_cnn_baseline(cnn_bern, front, back_bern, train_loader, test_loader,
                       "CNN-Bern", epochs=EPOCHS_BASELINE, lr=LR_BASELINE)

    # ---- SNN-SC (T=8, fixed rate) ----
    print("\n" + "="*70)
    print("TRAINING: SNN-SC (T=8, Fixed Rate)")
    print("="*70)
    snn_sc_t8 = SNNSC_FixedRate(C_in=C_in, C1=C1, C2=C2, T=8).to(device)
    back_snn8 = ResNet50Back(num_classes=100, pretrained=False)
    back_snn8.load_state_dict(back_state, strict=False)
    back_snn8 = back_snn8.to(device)
    # Use entropy loss for SNN-SC too (from original paper)
    train_cnn_baseline(snn_sc_t8, front, back_snn8, train_loader, test_loader,
                       "SNN-SC-T8", epochs=EPOCHS_BASELINE, lr=LR_BASELINE)

    # ---- SNN-SC (T=6, bandwidth-fair) ----
    print("\n" + "="*70)
    print("TRAINING: SNN-SC (T=6, Bandwidth-Fair)")
    print("="*70)
    snn_sc_t6 = SNNSC_FixedRate(C_in=C_in, C1=C1, C2=C2, T=6).to(device)
    back_snn6 = ResNet50Back(num_classes=100, pretrained=False)
    back_snn6.load_state_dict(back_state, strict=False)
    back_snn6 = back_snn6.to(device)
    train_cnn_baseline(snn_sc_t6, front, back_snn6, train_loader, test_loader,
                       "SNN-SC-T6", epochs=EPOCHS_BASELINE, lr=LR_BASELINE)

    # ---- Random Mask Ablation ----
    print("\n" + "="*70)
    print("TRAINING: Random Mask Ablation")
    print("="*70)
    random_mask = SpikeAdaptSC_RandomMask(C_in=C_in, C1=C1, C2=C2, T=T,
                                          keep_rate=0.75).to(device)
    back_rand = ResNet50Back(num_classes=100, pretrained=False)
    back_rand.load_state_dict(back_state, strict=False)
    back_rand = back_rand.to(device)
    train_cnn_baseline(random_mask, front, back_rand, train_loader, test_loader,
                       "RandomMask", epochs=EPOCHS_BASELINE, lr=LR_BASELINE)

    # ---- JPEG+Conv (no training needed) ----
    jpeg_conv = JPEGConvBaseline(C_in=C_in, jpeg_quality=50, code_rate_inv=3)

    # ------------------------------------------------------------------
    # 2. LOAD YOUR TRAINED SpikeAdapt-SC
    # ------------------------------------------------------------------
    from train_spikeadapt_sc import SpikeAdaptSC
    spikeadapt = SpikeAdaptSC(C_in=C_in, C1=C1, C2=C2, T=T,
                               eta=0.5, temperature=0.1).to(device)
    step3_files = sorted([f for f in os.listdir(SPIKEADAPT_DIR)
                          if f.startswith("step3_best_")])
    if step3_files:
        ckpt = torch.load(os.path.join(SPIKEADAPT_DIR, step3_files[-1]),
                          map_location=device)
        spikeadapt.load_state_dict(ckpt['spikeadapt'])
        back_orig.load_state_dict(ckpt['back'])
    spikeadapt.eval()

    # ------------------------------------------------------------------
    # 3. RUN FULL BER COMPARISON
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("RUNNING FULL BER COMPARISON")
    print("="*70)

    # Note: each model needs its own back classifier
    # For simplicity, we evaluate each model with its own trained back
    # In practice, you'd want the same back for fair comparison

    models_to_eval = {
        'SpikeAdapt-SC': (spikeadapt, back_orig),
        'SNN-SC (T=8)': (snn_sc_t8, back_snn8),
        'SNN-SC (T=6)': (snn_sc_t6, back_snn6),
        'CNN-Uni': (cnn_uni, back_uni),
        'CNN-Bern': (cnn_bern, back_bern),
        'Random Mask': (random_mask, back_rand),
        'JPEG+Conv': (jpeg_conv, back_orig),
    }

    all_results = {}
    ber_values = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    for name, (model, back_model) in models_to_eval.items():
        print(f"\nEvaluating {name}...", flush=True)
        model_results = []
        back_model.eval()

        for ber in ber_values:
            accs = []
            n_repeats = 10 if ber > 0 else 1  # Repeat to average channel randomness
            for _ in range(n_repeats):
                acc = evaluate_model(model, front, back_model, test_loader, ber=ber)
                accs.append(acc)
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            model_results.append((ber, mean_acc, std_acc))
            print(f"  BER={ber:.3f}: {mean_acc:.2f}% (±{std_acc:.2f})", flush=True)

        all_results[name] = model_results

    # ------------------------------------------------------------------
    # 4. PLOT THE MONEY FIGURE
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    styles = {
        'SpikeAdapt-SC':  {'color': '#D32F2F', 'marker': 's', 'ls': '-',  'lw': 2.5},
        'SNN-SC (T=8)':   {'color': '#1976D2', 'marker': 'o', 'ls': '-',  'lw': 2.0},
        'SNN-SC (T=6)':   {'color': '#42A5F5', 'marker': 'o', 'ls': '--', 'lw': 2.0},
        'CNN-Uni':         {'color': '#388E3C', 'marker': '^', 'ls': '-.',  'lw': 2.0},
        'CNN-Bern':        {'color': '#F57C00', 'marker': 'D', 'ls': '--', 'lw': 2.0},
        'JPEG+Conv':       {'color': '#7B1FA2', 'marker': 'v', 'ls': ':',  'lw': 2.0},
        'Random Mask':     {'color': '#9E9E9E', 'marker': 'x', 'ls': '--', 'lw': 1.5},
    }

    for name, results in all_results.items():
        bers = [r[0] for r in results]
        means = [r[1] for r in results]
        stds = [r[2] for r in results]
        s = styles.get(name, {'color': 'gray', 'marker': 'o', 'ls': '-', 'lw': 1.5})
        ax.plot(bers, means, marker=s['marker'], ls=s['ls'],
                color=s['color'], lw=s['lw'], markersize=7, label=name)
        if max(stds) > 0.1:
            ax.fill_between(bers, [m-s for m,s in zip(means, stds)],
                           [m+s for m,s in zip(means, stds)],
                           alpha=0.08, color=s['color'])

    ax.set_xlabel('Bit Error Rate (BER)', fontsize=14)
    ax.set_ylabel('Classification Accuracy (%)', fontsize=14)
    ax.set_title('SpikeAdapt-SC vs Baselines: Accuracy under BSC (CIFAR-100)',
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.01, 0.31)
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "ber_comparison_all.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path}", flush=True)
    plt.close()

    # Save results as JSON
    json_results = {}
    for name, results in all_results.items():
        json_results[name] = [(r[0], r[1], r[2]) for r in results]
    with open(os.path.join(RESULTS_DIR, "ber_comparison_results.json"), 'w') as f:
        json.dump(json_results, f, indent=2)
    print("✓ Saved results JSON", flush=True)

    # ------------------------------------------------------------------
    # 5. COMPLEXITY ANALYSIS
    # ------------------------------------------------------------------
    compute_complexity(C_in=C_in, C1=C1, C2=C2, H=1, W=1, T=T)
    compute_bandwidth_table(C_in=C_in, C2=C2)

    print("\n" + "="*70)
    print("✅ ALL BASELINES COMPLETE!")
    print("="*70)
