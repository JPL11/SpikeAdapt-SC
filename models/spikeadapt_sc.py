"""SpikeAdapt-SC: Content-Adaptive Semantic Communication Model.

Main model class combining SNN encoder, importance scorer, block masking,
channel simulation, and SNN decoder.
"""

import torch
import torch.nn as nn

from .snn_modules import IFNeuron, IHFNeuron, get_channel


# ============================================================================
# Learned Importance Scoring
# ============================================================================

class LearnedImportanceScorer(nn.Module):
    """Lightweight 1×1 conv network that scores spatial block importance.
    
    Takes time-averaged spike maps and produces per-block importance
    scores in (0, 1) for adaptive masking decisions.
    """
    def __init__(self, C_in=128, hidden=32):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Conv2d(C_in, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, all_S2):
        return self.scorer(torch.stack(all_S2, dim=0).mean(dim=0)).squeeze(1)


class LearnedBlockMask(nn.Module):
    """Differentiable block masking with Gumbel-sigmoid (train) / top-k (eval).
    
    During training: soft masks via Gumbel-sigmoid with straight-through.
    During evaluation: deterministic top-k selection by importance score.
    """
    def __init__(self, target_rate=0.75, temperature=0.5):
        super().__init__()
        self.target_rate = target_rate
        self.temperature = temperature

    def forward(self, importance, training=True):
        B, H, W = importance.shape
        if training:
            logits = torch.log(importance / (1 - importance + 1e-7) + 1e-7)
            u = torch.rand_like(logits).clamp(1e-7, 1 - 1e-7)
            soft = torch.sigmoid((logits - torch.log(-torch.log(u))) / self.temperature)
            hard = (soft > 0.5).float()
            mask = hard + (soft - soft.detach())  # STE
        else:
            k = max(1, int(self.target_rate * H * W))
            flat = importance.view(B, -1)
            _, idx = flat.topk(k, dim=1)
            mask = torch.zeros_like(flat)
            mask.scatter_(1, idx, 1.0)
            mask = mask.view(B, H, W)
        return mask.unsqueeze(1), mask.mean()

    def apply_mask(self, x, mask):
        return x * mask


# ============================================================================
# Encoder / Decoder
# ============================================================================

class Encoder(nn.Module):
    """SNN Encoder: two conv+BN+IF layers converting features to spikes."""
    def __init__(self, C_in=1024, C1=256, C2=128):
        super().__init__()
        self.C1, self.C2 = C1, C2
        self.conv1 = nn.Conv2d(C_in, C1, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(C1)
        self.if1 = IFNeuron()
        self.conv2 = nn.Conv2d(C1, C2, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(C2)
        self.if2 = IFNeuron()

    def forward(self, F, m1=None, m2=None):
        s1, m1 = self.if1(self.bn1(self.conv1(F)), m1)
        s2, m2 = self.if2(self.bn2(self.conv2(s1)), m2)
        return s1, s2, m1, m2


class Decoder(nn.Module):
    """SNN Decoder: reconstructs features from received spikes using IHF neurons."""
    def __init__(self, C_out=1024, C1=256, C2=128, T=8):
        super().__init__()
        self.T, self.C1, self.C_out = T, C1, C_out
        self.conv3 = nn.Conv2d(C2, C1, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(C1)
        self.if3 = IFNeuron()
        self.conv4 = nn.Conv2d(C1, C_out, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(C_out)
        self.ihf = IHFNeuron()
        self.converter_fc = nn.Linear(2 * T, 2 * T)

    def forward(self, recv_all, mask):
        m3, m4 = None, None
        Fs, Fm, s3_all = [], [], []
        for t in range(self.T):
            s3, m3 = self.if3(self.bn3(self.conv3(recv_all[t] * mask)), m3)
            sp, m4 = self.ihf(self.bn4(self.conv4(s3)), m4)
            Fs.append(sp)
            Fm.append(m4.clone())
            s3_all.append(s3)
        # Spike-to-feature converter
        il = []
        for t in range(self.T):
            il.append(Fs[t])
            il.append(Fm[t])
        stk = torch.stack(il, dim=1)  # (B, 2T, C, H, W)
        x = stk.permute(0, 2, 3, 4, 1)
        return (x * torch.sigmoid(self.converter_fc(x))).sum(dim=-1), s3_all, Fs


# ============================================================================
# Full Model
# ============================================================================

class SpikeAdaptSC(nn.Module):
    """SpikeAdapt-SC: Content-Adaptive SNN Semantic Communication.
    
    Args:
        C_in: Input feature channels (from backbone)
        C1: Intermediate encoder channels
        C2: Bottleneck channels (transmitted)
        T: Number of SNN timesteps
        target_rate: Target fraction of blocks to transmit
        channel_type: 'bsc', 'awgn', or 'rayleigh'
    """
    def __init__(self, C_in=1024, C1=256, C2=128, T=8,
                 target_rate=0.75, channel_type='bsc'):
        super().__init__()
        self.T = T
        self.encoder = Encoder(C_in, C1, C2)
        self.importance_scorer = LearnedImportanceScorer(C_in=C2, hidden=32)
        self.block_mask = LearnedBlockMask(target_rate, 0.5)
        self.decoder = Decoder(C_in, C1, C2, T)
        self.channel = get_channel(channel_type)
        self.channel_type = channel_type

    def forward(self, feat, noise_param=0.0, target_rate_override=None):
        # Encode over T timesteps
        all_S1, all_S2, m1, m2 = [], [], None, None
        for t in range(self.T):
            s1, s2, m1, m2 = self.encoder(feat, m1, m2)
            all_S1.append(s1)
            all_S2.append(s2)

        # Score importance and create mask
        importance = self.importance_scorer(all_S2)
        if target_rate_override is not None:
            old = self.block_mask.target_rate
            self.block_mask.target_rate = target_rate_override
            mask, tx = self.block_mask(importance, training=False)
            self.block_mask.target_rate = old
        else:
            mask, tx = self.block_mask(importance, training=self.training)

        # Apply mask and transmit through channel
        recv = [self.channel(self.block_mask.apply_mask(all_S2[t], mask), noise_param)
                for t in range(self.T)]

        # Decode
        Fp, s3_all, s4_all = self.decoder(recv, mask)

        return Fp, {
            'tx_rate': tx.item(),
            'mask': mask,
            'all_S1': all_S1,
            'all_S2': all_S2,
            's3_all': s3_all,
            's4_all': s4_all,
        }
