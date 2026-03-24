"""Core SNN building blocks: spike functions, neuron models, and channel models.

This module provides the actual components used in the SpikeAdapt-SC paper:
- SpikeFunction_Learnable: surrogate gradient with learnable slope
- LIFNeuron: Leaky Integrate-and-Fire with learnable leak (β) and MPBN
- MPBN: Membrane Potential Batch Normalization (ICCV 2023)
- Channel models: BSC, AWGN, Rayleigh, BEC
"""

import math
import torch
import torch.nn as nn


# ============================================================================
# Spike Functions
# ============================================================================

class SpikeFunction(torch.autograd.Function):
    """Heaviside spike with sigmoid surrogate gradient for backprop.
    
    Forward: spike = 1 if membrane > threshold else 0
    Backward: surrogate gradient via scaled sigmoid derivative
    """
    @staticmethod
    def forward(ctx, membrane, threshold):
        th_needs_grad = isinstance(threshold, torch.Tensor) and threshold.requires_grad
        if isinstance(threshold, (int, float)):
            threshold = torch.tensor(float(threshold), device=membrane.device)
        ctx.save_for_backward(membrane, threshold)
        ctx.th_needs_grad = th_needs_grad
        return (membrane > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        membrane, threshold = ctx.saved_tensors
        scale = 10.0
        sig = torch.sigmoid(scale * (membrane - threshold))
        sg = sig * (1 - sig) * scale
        grad_membrane = grad_output * sg
        grad_threshold = -(grad_output * sg).sum() if ctx.th_needs_grad else None
        return grad_membrane, grad_threshold


class SpikeFunction_Learnable(torch.autograd.Function):
    """Spike function with learnable surrogate gradient slope.
    
    Same as SpikeFunction but the sigmoid scale is a trainable parameter,
    allowing the network to adapt gradient sharpness during training.
    """
    @staticmethod
    def forward(ctx, membrane, threshold, slope):
        if isinstance(threshold, (int, float)):
            threshold = torch.tensor(float(threshold), device=membrane.device)
        ctx.save_for_backward(membrane, threshold, slope)
        ctx.th_needs_grad = isinstance(threshold, torch.Tensor) and threshold.requires_grad
        return (membrane > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        membrane, threshold, slope = ctx.saved_tensors
        s = slope.clamp(min=1.0, max=100.0)
        sig = torch.sigmoid(s * (membrane - threshold))
        sg = sig * (1 - sig) * s
        grad_mem = grad_output * sg
        grad_th = -(grad_output * sg).sum() if ctx.th_needs_grad else None
        grad_slope = (grad_output * sig * (1 - sig) * (membrane - threshold)).sum()
        return grad_mem, grad_th, grad_slope


# ============================================================================
# Neuron Models
# ============================================================================

class IFNeuron(nn.Module):
    """Integrate-and-Fire neuron with fixed threshold.
    
    Basic IF: accumulates input, fires when exceeding threshold,
    resets by subtracting threshold. Used in early versions (V1-V2).
    """
    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, x, mem=None):
        if mem is None:
            mem = torch.zeros_like(x)
        mem = mem + x
        sp = SpikeFunction.apply(mem, self.threshold)
        return sp, mem - sp * self.threshold


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron with learnable leak and slope.
    
    This is the neuron used in the paper's V5C architecture:
    - Learnable leak (β via sigmoid of raw parameter)
    - Learnable surrogate gradient slope
    - Per-channel leak parameters
    
    membrane(t) = β * membrane(t-1) + input(t)
    spike = 1 if membrane > threshold else 0
    membrane_reset = membrane - spike * threshold
    """
    def __init__(self, C, th=1.0):
        super().__init__()
        self.threshold = th
        self.beta_raw = nn.Parameter(torch.ones(1, C, 1, 1) * 2.2)  # sigmoid(2.2) ≈ 0.9
        self.slope = nn.Parameter(torch.tensor(10.0))

    def forward(self, x, mem=None):
        if mem is None:
            mem = torch.zeros_like(x)
        beta = torch.sigmoid(self.beta_raw)
        mem = beta * mem + x
        sp = SpikeFunction_Learnable.apply(mem, self.threshold, self.slope)
        return sp, mem - sp * self.threshold


class IHFNeuron(nn.Module):
    """Integrate-and-Fire neuron with Heterogeneous (learnable) threshold.
    
    Same as IFNeuron but threshold is a trainable parameter, allowing
    the decoder to learn optimal firing thresholds for reconstruction.
    """
    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(float(threshold)))

    def forward(self, x, mem=None):
        if mem is None:
            mem = torch.zeros_like(x)
        mem = mem + x
        sp = SpikeFunction.apply(mem, self.threshold)
        return sp, mem - sp * self.threshold


class MPBN(nn.Module):
    """Membrane Potential Batch Normalization (ICCV 2023).
    
    Normalizes membrane potential before the spike function at each timestep.
    Uses separate BN statistics per timestep to preserve temporal structure.
    This is the key component that reduces firing rate from 0.266 to 0.167.
    
    Reference: Guo et al., "Membrane Potential Batch Normalization for 
    Spiking Neural Networks," ICCV 2023.
    """
    def __init__(self, C, T):
        super().__init__()
        self.bns = nn.ModuleList([nn.BatchNorm2d(C, affine=True) for _ in range(T)])
        self.T = T

    def forward(self, x, t):
        return self.bns[min(t, self.T - 1)](x)


# ============================================================================
# Channel Models
# ============================================================================

class BSC_Channel(nn.Module):
    """Binary Symmetric Channel.
    
    Flips each bit independently with probability = BER.
    Uses straight-through estimator during training.
    """
    def forward(self, x, ber):
        if ber <= 0:
            return x
        flip = (torch.rand_like(x.float()) < ber).float()
        x_noisy = (x + flip) % 2
        if self.training:
            return x + (x_noisy - x).detach()  # STE
        return x_noisy


class AWGN_Channel(nn.Module):
    """AWGN Channel with BPSK modulation.
    
    Maps binary {0,1} → BPSK {-1,+1}, adds Gaussian noise,
    then hard-decision decodes. Parameterized by SNR in dB.
    """
    def forward(self, x, snr_db):
        if snr_db >= 100:
            return x  # no noise
        bpsk = 2.0 * x - 1.0
        snr_linear = 10 ** (snr_db / 10.0)
        noise_std = 1.0 / math.sqrt(2 * snr_linear)
        noise = torch.randn_like(bpsk) * noise_std
        received = bpsk + noise
        decoded = (received > 0).float()
        if self.training:
            return x + (decoded - x).detach()
        return decoded


class Rayleigh_Channel(nn.Module):
    """Rayleigh fading channel with AWGN.
    
    Models wireless multipath: y = h·x + n, where |h| ~ Rayleigh.
    Assumes perfect CSI at receiver (coherent detection).
    """
    def forward(self, x, snr_db):
        if snr_db >= 100:
            return x
        bpsk = 2.0 * x - 1.0
        snr_linear = 10 ** (snr_db / 10.0)
        # Rayleigh fading coefficient
        h_real = torch.randn_like(bpsk) / math.sqrt(2)
        h_imag = torch.randn_like(bpsk) / math.sqrt(2)
        h_mag = torch.sqrt(h_real**2 + h_imag**2)
        # Faded signal + noise
        noise_std = 1.0 / math.sqrt(2 * snr_linear)
        noise = torch.randn_like(bpsk) * noise_std
        received = h_mag * bpsk + noise
        # Coherent detection (perfect CSI)
        equalized = received / (h_mag + 1e-8)
        decoded = (equalized > 0).float()
        if self.training:
            return x + (decoded - x).detach()
        return decoded


class BEC_Channel(nn.Module):
    """Binary Erasure Channel.
    
    Each bit is independently erased (set to 0) with probability p.
    With SNN firing rate ~0.167, ~83% of bits are already 0,
    making SNNs inherently immune to BEC.
    """
    def forward(self, x, erasure_prob):
        if erasure_prob <= 0:
            return x
        erasure_mask = (torch.rand_like(x.float()) < erasure_prob).float()
        x_erased = x * (1.0 - erasure_mask)
        if self.training:
            return x + (x_erased - x).detach()  # STE
        return x_erased


def get_channel(channel_type):
    """Factory function to create channel by name."""
    channels = {
        'bsc': BSC_Channel,
        'awgn': AWGN_Channel,
        'rayleigh': Rayleigh_Channel,
        'bec': BEC_Channel,
    }
    if channel_type not in channels:
        raise ValueError(f"Unknown channel: {channel_type}. Choose from {list(channels.keys())}")
    return channels[channel_type]()
