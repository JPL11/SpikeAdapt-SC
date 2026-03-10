"""Core SNN building blocks: spike functions, neuron models, and channel models."""

import math
import torch
import torch.nn as nn


# ============================================================================
# Spike Function with Surrogate Gradient
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


# ============================================================================
# Neuron Models
# ============================================================================

class IFNeuron(nn.Module):
    """Integrate-and-Fire neuron with fixed threshold.
    
    Accumulates input into membrane potential and fires when exceeding
    threshold. Membrane resets by subtracting threshold after spike.
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


def get_channel(channel_type):
    """Factory function to create channel by name."""
    channels = {
        'bsc': BSC_Channel,
        'awgn': AWGN_Channel,
        'rayleigh': Rayleigh_Channel,
    }
    if channel_type not in channels:
        raise ValueError(f"Unknown channel: {channel_type}. Choose from {list(channels.keys())}")
    return channels[channel_type]()
