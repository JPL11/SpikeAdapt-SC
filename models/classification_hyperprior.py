"""Hyperprior for Classification SpikeAdapt-SC (V5C-NA).

Adds a lightweight side-channel that transmits compact image statistics
alongside the main spike stream. The decoder uses these hints to better
reconstruct the 1024-dim features, improving classification under noise.

Architecture:
  Main path (existing V5C-NA):
    ResNet50 Front → [1024×14×14] → SNN Encoder → [36×14×14×T] spikes
    → Scorer + Mask → BSC → SNN Decoder → [1024×14×14] → ResNet50 Back → logits

  Hyperprior side-channel (NEW):
    [36×14×14] mean spikes → HyperEncoder → [C_hyper×7×7] compact latent
    → BSC channel → HyperDecoder → [36×14×14] reconstruction hints
    → Fused with received spikes before main decoder

Transferable to detection: same pattern but at P3/P4/P5 scales.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHyperEncoder(nn.Module):
    """Compress spike statistics into compact side information.

    Input: mean spike rate [B, C_spike, H, W] (e.g., [B, 36, 14, 14])
    Output: compact latent [B, C_hyper, H/2, W/2] (e.g., [B, 8, 7, 7])
    """
    def __init__(self, C_spike=36, C_hyper=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(C_spike, C_hyper * 2, 3, 1, 1),
            nn.BatchNorm2d(C_hyper * 2),
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2),  # 14×14 → 7×7
            nn.Conv2d(C_hyper * 2, C_hyper, 3, 1, 1),
            nn.BatchNorm2d(C_hyper),
            nn.Tanh(),  # Bound to [-1, 1]
        )

    def forward(self, mean_spikes):
        return self.encoder(mean_spikes)


class ClassificationHyperDecoder(nn.Module):
    """Decode hyperprior into reconstruction hints.

    Input: received latent [B, C_hyper, H/2, W/2]
    Output: hints [B, C_spike, H, W] — added to received spikes
    """
    def __init__(self, C_spike=36, C_hyper=8):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(C_hyper, C_hyper * 2, 4, 2, 1),  # 7×7 → 14×14
            nn.BatchNorm2d(C_hyper * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(C_hyper * 2, C_spike, 3, 1, 1),
            nn.BatchNorm2d(C_spike),
        )

    def forward(self, hyper_latent):
        return self.decoder(hyper_latent)


class HyperpriorBSC(nn.Module):
    """Ternary quantize + BSC for hyperprior latent."""
    def forward(self, z, ber):
        # Ternary quantize: [-1, 1] → {-1, 0, 1}
        z_ternary = torch.zeros_like(z)
        z_ternary[z > 0.33] = 1.0
        z_ternary[z < -0.33] = -1.0
        z_ternary = z + (z_ternary - z).detach()  # STE

        if ber > 0:
            flip = (torch.rand_like(z_ternary) < ber).float()
            random_trit = (torch.randint(0, 3, z.shape, device=z.device).float() - 1)
            z_noisy = z_ternary * (1 - flip) + random_trit * flip
            z_ternary = z_ternary + (z_noisy - z_ternary).detach()  # STE
        return z_ternary


class ClassificationHyperprior(nn.Module):
    """Wraps hyper-encoder + channel + hyper-decoder + fusion.

    Drop-in addition to SpikeAdaptSC_v5c_NA — call after SNN encoding,
    before SNN decoding.
    """
    def __init__(self, C_spike=36, C_hyper=8):
        super().__init__()
        self.hyper_enc = ClassificationHyperEncoder(C_spike, C_hyper)
        self.hyper_dec = ClassificationHyperDecoder(C_spike, C_hyper)
        self.hyper_bsc = HyperpriorBSC()

        # Fusion: combine received spikes + hyperprior hints
        self.fuse = nn.Sequential(
            nn.Conv2d(C_spike * 2, C_spike, 1),
            nn.BatchNorm2d(C_spike),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, all_S2_received, ber=0.0):
        """
        Args:
            all_S2_received: list of T received spike tensors [B, C, H, W]
                             (already passed through BSC)
            ber: current BER for hyperprior channel

        Returns:
            all_S2_fused: list of T fused spike tensors (same shape as input)
        """
        # 1. Compute mean spike rate from received spikes
        mean_spikes = torch.stack(all_S2_received).mean(0)  # B×C×H×W

        # 2. Encode to compact latent
        hyper_latent = self.hyper_enc(mean_spikes)  # B×C_hyper×7×7

        # 3. Send through BSC
        hyper_recv = self.hyper_bsc(hyper_latent, ber)

        # 4. Decode to hints
        hints = self.hyper_dec(hyper_recv)  # B×C_spike×14×14

        # Handle size mismatch
        if hints.shape[2:] != mean_spikes.shape[2:]:
            hints = F.interpolate(hints, size=mean_spikes.shape[2:],
                                  mode='bilinear', align_corners=False)

        # 5. Fuse hints with each timestep's received spikes
        fused = []
        for t_spikes in all_S2_received:
            combined = torch.cat([t_spikes, hints], dim=1)  # B×(2C)×H×W
            fused.append(self.fuse(combined))  # B×C×H×W

        return fused

    def get_overhead_ratio(self, C_spike=36, H=14, W=14, C_hyper=8):
        """Compute bandwidth overhead of hyperprior as fraction of main stream."""
        main_bits = C_spike * H * W  # 36 × 14 × 14 = 7056
        hyper_bits = C_hyper * (H // 2) * (W // 2)  # 8 × 7 × 7 = 392
        return hyper_bits / main_bits  # ~5.6%
