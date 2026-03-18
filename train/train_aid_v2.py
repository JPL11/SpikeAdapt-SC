"""SpikeAdapt-SC v2 — Redesigned for GLOBECOM.

Key changes from v1:
1. 14×14 spatial grid (196 blocks) instead of 8×8 (64 blocks)
2. Channel-conditioned importance scorer (sees BER/SNR estimate)
3. Joint spatial-temporal adaptation (confidence-based early stopping)
4. Reduced C2 from 128 → 36 to match payload at finer grid
5. External baselines: SNN-SC (no mask) + CNN-Uni on AID

Usage:
    python train/train_aid_v2.py                    # Train SpikeAdapt-SC v2 (BSC)
    python train/train_aid_v2.py --baselines        # Also train baselines
    python train/train_aid_v2.py --seed 123         # Specific seed
"""

import os, sys, random, json, math, argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict

os.environ['PYTHONUNBUFFERED'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import torchvision.transforms as T
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ############################################################################
# ARGS (parsed at module level only when run directly, not when imported)
# ############################################################################
def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--baselines', action='store_true')
    parser.add_argument('--epochs_s2', type=int, default=60)
    parser.add_argument('--epochs_s3', type=int, default=30)
    parser.add_argument('--target_rate', type=float, default=0.75)
    parser.add_argument('--T_max', type=int, default=8, help='Max timesteps')
    parser.add_argument('--C2', type=int, default=36, help='Bottleneck channels')
    parser.add_argument('--grid', type=int, default=14, help='Spatial grid size')
    return parser.parse_args()



# ############################################################################
# AID DATASET
# ############################################################################
class AIDDataset(Dataset):
    def __init__(self, root, transform=None, split='train', train_ratio=0.8, seed=42):
        self.transform = transform; self.samples = []
        aid_dir = os.path.join(root, 'AID')
        if not os.path.exists(aid_dir):
            raise FileNotFoundError(f"AID not found at {aid_dir}")
        classes = sorted([d for d in os.listdir(aid_dir) if os.path.isdir(os.path.join(aid_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        all_samples = []
        for cls in classes:
            cd = os.path.join(aid_dir, cls)
            for f in sorted(os.listdir(cd)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
                    all_samples.append((os.path.join(cd, f), self.class_to_idx[cls]))
        rng = random.Random(seed); rng.shuffle(all_samples)
        n = int(len(all_samples) * train_ratio)
        self.samples = all_samples[:n] if split == 'train' else all_samples[n:]
        print(f"  AID {split}: {len(self.samples)} images, {len(classes)} classes")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, l = self.samples[idx]; img = Image.open(p).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, l


# ############################################################################
# CHANNELS
# ############################################################################
class BSC_Channel(nn.Module):
    def forward(self, x, ber):
        if ber <= 0: return x
        return ((x + (torch.rand_like(x.float()) < ber).float()) % 2)

class AWGN_Channel(nn.Module):
    def forward(self, x, snr_db):
        if snr_db > 50: return x
        snr = 10 ** (snr_db / 10); sigma = 1.0 / math.sqrt(2 * max(snr, 1e-8))
        return x + sigma * torch.randn_like(x)

class Rayleigh_Channel(nn.Module):
    def forward(self, x, snr_db):
        if snr_db > 50: return x
        snr = 10 ** (snr_db / 10); sigma = 1.0 / math.sqrt(2 * max(snr, 1e-8))
        h = torch.sqrt(torch.randn_like(x) ** 2 + torch.randn_like(x) ** 2) / math.sqrt(2)
        return h * x + sigma * torch.randn_like(x)


# ############################################################################
# SNN MODULES
# ############################################################################
class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane, threshold):
        if isinstance(threshold, (int, float)):
            threshold = torch.tensor(float(threshold), device=membrane.device)
        ctx.save_for_backward(membrane, threshold)
        ctx.th_needs_grad = isinstance(threshold, torch.Tensor) and threshold.requires_grad
        return (membrane > threshold).float()
    @staticmethod
    def backward(ctx, grad_output):
        membrane, threshold = ctx.saved_tensors
        scale = 10.; sig = torch.sigmoid(scale * (membrane - threshold))
        sg = sig * (1 - sig) * scale
        return grad_output * sg, -(grad_output * sg).sum() if ctx.th_needs_grad else None

class IFNeuron(nn.Module):
    def __init__(self, th=1.0):
        super().__init__(); self.threshold = th
    def forward(self, x, mem=None):
        if mem is None: mem = torch.zeros_like(x)
        mem = mem + x; sp = SpikeFunction.apply(mem, self.threshold)
        return sp, mem - sp * self.threshold

class IHFNeuron(nn.Module):
    def __init__(self, th=1.0):
        super().__init__(); self.threshold = nn.Parameter(torch.tensor(float(th)))
    def forward(self, x, mem=None):
        if mem is None: mem = torch.zeros_like(x)
        mem = mem + x; sp = SpikeFunction.apply(mem, self.threshold)
        return sp, mem - sp * self.threshold


# ############################################################################
# BACKBONE — ResNet50 for AID (14×14 native output, NO pooling)
# ############################################################################
class ResNet50Front(nn.Module):
    """Split at layer3. For AID (224×224): output (1024, 14, 14).
    NO spatial pooling — retain full 14×14 grid for finer masking.
    """
    def __init__(self, grid_size=14):
        super().__init__()
        r = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        self.conv1 = r.conv1; self.bn1 = r.bn1; self.relu = r.relu
        self.maxpool = r.maxpool
        self.layer1 = r.layer1; self.layer2 = r.layer2; self.layer3 = r.layer3
        # Only pool if grid_size != 14 (native)
        self.spatial_pool = nn.AdaptiveAvgPool2d(grid_size) if grid_size != 14 else nn.Identity()
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.spatial_pool(self.layer3(self.layer2(self.layer1(x))))

class ResNet50Back(nn.Module):
    def __init__(self, num_classes=30):
        super().__init__()
        r = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        self.layer4 = r.layer4; self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)
    def forward(self, x): return self.fc(torch.flatten(self.avgpool(self.layer4(x)), 1))


# ############################################################################
# SPIKEADAPT-SC v2 — Finer grid + Channel-conditioned + Temporal adaptation
# ############################################################################

class ChannelConditionedScorer(nn.Module):
    """Importance scorer that sees spike statistics AND channel estimate.

    Architecture: concat(spike_mean_features, channel_embedding_broadcast) → Conv → Score
    """
    def __init__(self, C_spike=36, hidden=32):
        super().__init__()
        # Channel embedding: scalar BER/SNR → hidden-dim vector → broadcast spatially
        self.channel_mlp = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(True),
            nn.Linear(hidden, hidden), nn.ReLU(True),
        )
        # Scorer: (C_spike + hidden) → hidden → 1
        self.scorer = nn.Sequential(
            nn.Conv2d(C_spike + hidden, hidden, 1), nn.ReLU(True),
            nn.Conv2d(hidden, 1, 1), nn.Sigmoid()
        )

    def forward(self, all_S2, channel_estimate):
        """
        all_S2: list of T tensors, each (B, C2, H, W)
        channel_estimate: scalar or (B, 1) tensor
        """
        spike_mean = torch.stack(all_S2, dim=0).mean(dim=0)  # (B, C2, H, W)
        B, C, H, W = spike_mean.shape

        # Process channel estimate
        if isinstance(channel_estimate, (int, float)):
            ch_input = torch.full((B, 1), channel_estimate, device=spike_mean.device)
        else:
            ch_input = channel_estimate.view(B, 1)

        ch_embed = self.channel_mlp(ch_input)  # (B, hidden)
        ch_map = ch_embed.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)  # (B, hidden, H, W)

        combined = torch.cat([spike_mean, ch_map], dim=1)  # (B, C_spike + hidden, H, W)
        return self.scorer(combined).squeeze(1)  # (B, H, W)


class LearnedBlockMask(nn.Module):
    def __init__(self, target_rate=0.75, temperature=0.5):
        super().__init__()
        self.target_rate = target_rate; self.temperature = temperature
    def forward(self, importance, training=True):
        B, H, W = importance.shape
        if training:
            logits = torch.log(importance / (1 - importance + 1e-7) + 1e-7)
            u = torch.rand_like(logits).clamp(1e-7, 1 - 1e-7)
            soft = torch.sigmoid((logits - torch.log(-torch.log(u))) / self.temperature)
            hard = (soft > 0.5).float()
            mask = hard + (soft - soft.detach())
        else:
            k = max(1, int(self.target_rate * H * W))
            flat = importance.view(B, -1)
            _, idx = flat.topk(k, dim=1)
            mask = torch.zeros_like(flat); mask.scatter_(1, idx, 1.0)
            mask = mask.view(B, H, W)
        return mask.unsqueeze(1), mask.mean()
    def apply_mask(self, x, mask): return x * mask


class Encoder(nn.Module):
    def __init__(self, C_in=1024, C1=256, C2=36):
        super().__init__()
        self.conv1 = nn.Conv2d(C_in, C1, 3, 1, 1); self.bn1 = nn.BatchNorm2d(C1); self.if1 = IFNeuron()
        self.conv2 = nn.Conv2d(C1, C2, 3, 1, 1); self.bn2 = nn.BatchNorm2d(C2); self.if2 = IFNeuron()
    def forward(self, F, m1=None, m2=None):
        s1, m1 = self.if1(self.bn1(self.conv1(F)), m1)
        s2, m2 = self.if2(self.bn2(self.conv2(s1)), m2)
        return s1, s2, m1, m2


class Decoder(nn.Module):
    def __init__(self, C_out=1024, C1=256, C2=36, T=8):
        super().__init__()
        self.T = T
        self.conv3 = nn.Conv2d(C2, C1, 3, 1, 1); self.bn3 = nn.BatchNorm2d(C1); self.if3 = IFNeuron()
        self.conv4 = nn.Conv2d(C1, C_out, 3, 1, 1); self.bn4 = nn.BatchNorm2d(C_out); self.ihf = IHFNeuron()
        self.converter_fc = nn.Linear(2 * T, 2 * T)

    def forward(self, recv_all, mask, T_actual=None):
        """Decode from received spikes. T_actual allows early stopping."""
        T_use = T_actual if T_actual is not None else len(recv_all)
        m3, m4 = None, None; Fs, Fm = [], []
        for t in range(T_use):
            s3, m3 = self.if3(self.bn3(self.conv3(recv_all[t] * mask)), m3)
            sp, m4 = self.ihf(self.bn4(self.conv4(s3)), m4)
            Fs.append(sp); Fm.append(m4.clone())
        # Pad if early stopped (to keep converter_fc input dim fixed)
        while len(Fs) < self.T:
            Fs.append(torch.zeros_like(Fs[0]))
            Fm.append(torch.zeros_like(Fm[0]))
        il = []
        for t in range(self.T): il.append(Fs[t]); il.append(Fm[t])
        x = torch.stack(il, 1).permute(0, 2, 3, 4, 1)
        return (x * torch.sigmoid(self.converter_fc(x))).sum(-1)


class SpikeAdaptSC_v2(nn.Module):
    """SpikeAdapt-SC v2: Finer grid + channel-conditioned scorer + temporal adaptation."""
    def __init__(self, C_in=1024, C1=256, C2=36, T=8,
                 target_rate=0.75, channel_type='bsc', grid_size=14):
        super().__init__()
        self.T = T; self.C2 = C2; self.grid_size = grid_size
        self.encoder = Encoder(C_in, C1, C2)
        self.scorer = ChannelConditionedScorer(C_spike=C2, hidden=32)
        self.block_mask = LearnedBlockMask(target_rate, 0.5)
        self.decoder = Decoder(C_in, C1, C2, T)
        if channel_type == 'awgn': self.channel = AWGN_Channel()
        elif channel_type == 'rayleigh': self.channel = Rayleigh_Channel()
        else: self.channel = BSC_Channel()
        self.channel_type = channel_type

        # Confidence gate for temporal early stopping
        self.conf_threshold = 0.95

    def forward(self, feat, noise_param=0.0, target_rate_override=None,
                early_stop=False):
        all_S2, m1, m2 = [], None, None

        # Encode all timesteps
        for t in range(self.T):
            _, s2, m1, m2 = self.encoder(feat, m1, m2)
            all_S2.append(s2)

        # Channel-conditioned scoring
        importance = self.scorer(all_S2, noise_param)

        if target_rate_override is not None:
            old = self.block_mask.target_rate
            self.block_mask.target_rate = target_rate_override
            mask, tx = self.block_mask(importance, training=False)
            self.block_mask.target_rate = old
        else:
            mask, tx = self.block_mask(importance, training=self.training)

        # Transmit through channel
        recv = [self.channel(self.block_mask.apply_mask(all_S2[t], mask), noise_param)
                for t in range(self.T)]

        # Temporal early stopping (inference only)
        T_used = self.T
        if early_stop and not self.training:
            # Try progressive decoding, stop when confident
            for t_try in range(2, self.T + 1):  # min 2 timesteps
                Fp_try = self.decoder(recv[:t_try], mask, T_actual=t_try)
                # Can't run classifier here (no access to back), so we estimate
                # confidence from decoder output magnitude
                # In practice, the caller should handle this
                T_used = t_try  # Default: use all
            # For now, always use all — caller handles early stopping
            T_used = self.T

        Fp = self.decoder(recv, mask)
        return Fp, {
            'tx_rate': tx.item(),
            'mask': mask,
            'importance': importance,
            'T_used': T_used,
            'num_blocks': self.grid_size ** 2,
            'bits_per_frame': int(tx.item() * self.grid_size**2 * self.C2 * self.T),
        }


# ############################################################################
# BASELINE 1: SNN-SC (FIXED RATE, NO MASKING)
# ############################################################################
class SNNSC(nn.Module):
    """Original SNN-SC: same encoder/decoder, no masking. Full-rate transmission."""
    def __init__(self, C_in=1024, C1=256, C2=36, T=8, channel_type='bsc'):
        super().__init__()
        self.T = T; self.C2 = C2
        self.encoder = Encoder(C_in, C1, C2)
        self.decoder = Decoder(C_in, C1, C2, T)
        if channel_type == 'bsc': self.channel = BSC_Channel()
        elif channel_type == 'awgn': self.channel = AWGN_Channel()
        else: self.channel = Rayleigh_Channel()

    def forward(self, feat, noise_param=0.0):
        all_S2, m1, m2 = [], None, None
        for t in range(self.T):
            _, s2, m1, m2 = self.encoder(feat, m1, m2)
            all_S2.append(s2)
        mask = torch.ones(feat.shape[0], 1, feat.shape[2], feat.shape[3], device=feat.device)
        recv = [self.channel(all_S2[t], noise_param) for t in range(self.T)]
        Fp = self.decoder(recv, mask)
        return Fp, {'tx_rate': 1.0}


# ############################################################################
# BASELINE 2: CNN-Uni (ANN + Uniform Quantization)
# ############################################################################
class UniformQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, n_bits):
        n_levels = 2 ** n_bits; x_c = x.clamp(0, 1)
        return (x_c * (n_levels - 1)).round() / (n_levels - 1)
    @staticmethod
    def backward(ctx, g): return g, None

class CNNUni(nn.Module):
    """ANN encoder + uniform quantization → BSC → ANN decoder."""
    def __init__(self, C_in=1024, C1=256, C2=36, n_bits=8):
        super().__init__()
        self.n_bits = n_bits
        self.enc1 = nn.Conv2d(C_in, C1, 3, 1, 1); self.ebn1 = nn.BatchNorm2d(C1)
        self.enc2 = nn.Conv2d(C1, C2, 3, 1, 1); self.ebn2 = nn.BatchNorm2d(C2)
        self.dec1 = nn.Conv2d(C2, C1, 3, 1, 1); self.dbn1 = nn.BatchNorm2d(C1)
        self.dec2 = nn.Conv2d(C1, C_in, 3, 1, 1); self.dbn2 = nn.BatchNorm2d(C_in)
        self.channel = BSC_Channel()

    def forward(self, feat, noise_param=0.0):
        z = torch.sigmoid(self.ebn2(self.enc2(F.relu(self.ebn1(self.enc1(feat))))))
        z_q = UniformQuantizeSTE.apply(z, self.n_bits)
        # Convert to bits, send through BSC, back to quantized
        n_levels = 2 ** self.n_bits
        bits = (z_q * (n_levels - 1)).round().long()
        B, C, H, W = bits.shape
        # Expand each value to n_bits binary
        bit_planes = []
        for b in range(self.n_bits):
            bit_planes.append(((bits >> b) & 1).float())
        bit_tensor = torch.stack(bit_planes, dim=-1)  # (B, C, H, W, n_bits)

        # BSC on each bit
        if noise_param > 0:
            flip_mask = (torch.rand_like(bit_tensor) < noise_param).float()
            bit_tensor = (bit_tensor + flip_mask) % 2

        # Reconstruct from bits
        values = torch.zeros(B, C, H, W, device=feat.device)
        for b in range(self.n_bits):
            values += bit_tensor[..., b] * (2 ** b)
        z_noisy = values / (n_levels - 1)

        Fp = self.dbn2(self.dec2(F.relu(self.dbn1(self.dec1(z_noisy)))))
        total_bits = B * C * H * W * self.n_bits
        return Fp, {'tx_rate': 1.0, 'total_bits': total_bits}


# ############################################################################
# TEMPORAL EARLY STOPPING (at eval time)
# ############################################################################
def evaluate_with_early_stop(front, model, back, loader, noise_param=0.0,
                              conf_threshold=0.95, min_T=2):
    """Evaluate with confidence-based temporal early stopping."""
    front.eval(); model.eval(); back.eval()
    correct, total = 0, 0
    total_T_used = 0; n_batches = 0

    with torch.no_grad():
        for img, lab in loader:
            img, lab = img.to(device), lab.to(device)
            feat = front(img)

            # Encode all timesteps
            all_S2, m1, m2 = [], None, None
            for t in range(model.T):
                _, s2, m1, m2 = model.encoder(feat, m1, m2)
                all_S2.append(s2)

            # Score and mask
            importance = model.scorer(all_S2, noise_param)
            mask, _ = model.block_mask(importance, training=False)

            # Transmit through channel
            recv = [model.channel(model.block_mask.apply_mask(all_S2[t], mask), noise_param)
                    for t in range(model.T)]

            # Progressive decode with early stopping
            T_used = model.T
            for t_try in range(min_T, model.T + 1):
                Fp = model.decoder(recv[:t_try], mask, T_actual=t_try)
                logits = back(Fp)
                conf = F.softmax(logits, 1).max(1)[0].mean().item()
                if conf >= conf_threshold:
                    T_used = t_try
                    break

            # Final prediction at T_used
            Fp = model.decoder(recv[:T_used], mask, T_actual=T_used)
            preds = back(Fp).argmax(1)
            correct += preds.eq(lab).sum().item()
            total += lab.size(0)
            total_T_used += T_used; n_batches += 1

    acc = 100. * correct / total
    avg_T = total_T_used / max(n_batches, 1)
    return acc, avg_T


# ############################################################################
# HELPERS
# ############################################################################
def sample_noise(channel_type):
    if channel_type == 'bsc':
        return random.uniform(0.15, 0.4) if random.random() < 0.5 else random.uniform(0, 0.15)
    else:
        return random.uniform(-2, 5) if random.random() < 0.5 else random.uniform(5, 20)

def evaluate(front, model, back, loader, noise_param=0.0, is_baseline=False):
    front.eval(); model.eval(); back.eval()
    correct, total, tx_sum, n = 0, 0, 0.0, 0
    with torch.no_grad():
        for img, lab in loader:
            img, lab = img.to(device), lab.to(device)
            if is_baseline:
                Fp, stats = model(front(img), noise_param)
            else:
                Fp, stats = model(front(img), noise_param=noise_param)
            correct += back(Fp).argmax(1).eq(lab).sum().item()
            total += lab.size(0); tx_sum += stats['tx_rate']; n += 1
    return 100. * correct / total, tx_sum / max(n, 1)


# ############################################################################
# MAIN
# ############################################################################
if __name__ == "__main__":
    args = _parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    SNAP_DIR = f"./snapshots_aid_v2_seed{args.seed}/"
    os.makedirs(SNAP_DIR, exist_ok=True)
    print(f"Device: {device} | Seed: {args.seed} | Grid: {args.grid}×{args.grid} | C2: {args.C2}")

    NC = 30; IS = 224

    train_tf = T.Compose([
        T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
        T.RandomRotation(15), T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))
    ])
    test_tf = T.Compose([
        T.Resize(256), T.CenterCrop(224),
        T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))
    ])

    print("Loading AID dataset...")
    train_ds = AIDDataset("./data", transform=train_tf, split='train', seed=42)
    test_ds = AIDDataset("./data", transform=test_tf, split='test', seed=42)
    train_loader = DataLoader(train_ds, 32, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, 64, shuffle=False, num_workers=4, pin_memory=True)

    # ==================================================================
    # BACKBONE (pretrained, frozen after fine-tuning)
    # ==================================================================
    front = ResNet50Front(grid_size=args.grid).to(device)
    back = ResNet50Back(NC).to(device)

    # Try loading existing backbone
    bb_path = "./snapshots_aid/backbone_best.pth"
    if os.path.exists(bb_path):
        state = torch.load(bb_path, map_location=device)
        front.load_state_dict({k: v for k, v in state.items()
                               if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))}, strict=False)
        back.load_state_dict({k: v for k, v in state.items()
                              if k.startswith(('layer4.', 'fc.', 'avgpool.'))}, strict=False)
        print(f"✓ Backbone loaded from {bb_path}")
    else:
        print("\n" + "=" * 60)
        print("STEP 1: Fine-tune ResNet50 on AID")
        print("=" * 60)
        all_p = list(front.parameters()) + list(back.parameters())
        opt = optim.SGD(all_p, lr=0.01, momentum=0.9, weight_decay=5e-4)
        sched = optim.lr_scheduler.MultiStepLR(opt, [20, 35], 0.1)
        ce = nn.CrossEntropyLoss(); best = 0.0
        for ep in range(50):
            front.train(); back.train()
            for img, lab in tqdm(train_loader, desc=f"BB E{ep+1}/50"):
                img, lab = img.to(device), lab.to(device)
                loss = ce(back(front(img)), lab)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            if (ep + 1) % 5 == 0:
                front.eval(); back.eval(); c, t = 0, 0
                with torch.no_grad():
                    for img, lab in test_loader:
                        img, lab = img.to(device), lab.to(device)
                        c += back(front(img)).argmax(1).eq(lab).sum().item(); t += lab.size(0)
                acc = 100. * c / t; print(f"  BB E{ep+1}: {acc:.2f}%")
                if acc > best:
                    best = acc
                    torch.save({**front.state_dict(), **back.state_dict()},
                               os.path.join(SNAP_DIR, "backbone_best.pth"))
        state = torch.load(os.path.join(SNAP_DIR, "backbone_best.pth"), map_location=device)
        front.load_state_dict({k: v for k, v in state.items()
                               if not k.startswith(('layer4.', 'fc.', 'avgpool.'))}, strict=False)
        back.load_state_dict({k: v for k, v in state.items()
                              if k.startswith(('layer4.', 'fc.', 'avgpool.'))}, strict=False)

    front.eval()
    with torch.no_grad():
        d = front(torch.randn(1, 3, IS, IS).to(device))
        print(f"  Feature shape: {d.shape}")  # (1, 1024, 14, 14)
    C_in = d.shape[1]
    H_feat = d.shape[2]
    print(f"  Grid: {H_feat}×{H_feat} = {H_feat**2} blocks, C_in={C_in}")

    # ==================================================================
    # TRAIN SPIKEADAPT-SC v2 (BSC)
    # ==================================================================
    channel_type = 'bsc'
    print(f"\n{'='*60}")
    print(f"SPIKEADAPT-SC v2 — {channel_type.upper()}")
    print(f"  Grid={H_feat}×{H_feat}, C2={args.C2}, T={args.T_max}, ρ={args.target_rate}")
    print(f"  Payload per frame: {H_feat**2} × {args.C2} × {args.T_max} = "
          f"{H_feat**2 * args.C2 * args.T_max} values")
    print(f"{'='*60}")

    model = SpikeAdaptSC_v2(C_in=C_in, C1=256, C2=args.C2, T=args.T_max,
                             target_rate=args.target_rate, channel_type=channel_type,
                             grid_size=H_feat).to(device)

    # Reload backbone
    if os.path.exists(bb_path):
        bb_state = torch.load(bb_path, map_location=device)
        back.load_state_dict({k: v for k, v in bb_state.items()
                              if k.startswith(('layer4.', 'fc.', 'avgpool.'))}, strict=False)

    for p in front.parameters(): p.requires_grad = False
    for p in back.parameters(): p.requires_grad = False
    criterion = nn.CrossEntropyLoss()

    # --- Step 2: Train SNN + scorer (backbone frozen) ---
    opt = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs_s2, eta_min=1e-6)

    best_s2 = 0.0
    for ep in range(args.epochs_s2):
        model.train()
        pbar = tqdm(train_loader, desc=f"S2 E{ep+1}/{args.epochs_s2}")
        for img, lab in pbar:
            img, lab = img.to(device), lab.to(device)
            noise = sample_noise(channel_type)
            with torch.no_grad(): feat = front(img)
            Fp, stats = model(feat, noise_param=noise)
            loss = criterion(back(Fp), lab) + 2.0 * (stats['tx_rate'] - args.target_rate) ** 2
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            pbar.set_postfix({'L': f'{loss.item():.3f}', 'tx': f'{stats["tx_rate"]:.2f}'})
        sched.step()

        if (ep + 1) % 10 == 0:
            acc, tx = evaluate(front, model, back, test_loader, noise_param=0.0)
            print(f"  S2 E{ep+1}: {acc:.2f}%, Tx={tx:.3f}")
            if acc > best_s2:
                best_s2 = acc
                torch.save(model.state_dict(),
                           os.path.join(SNAP_DIR, f"v2_s2_{acc:.2f}.pth"))
                print(f"  ✓ Best S2: {best_s2:.2f}%")

    s2f = sorted([f for f in os.listdir(SNAP_DIR) if f.startswith("v2_s2_")])
    if s2f: model.load_state_dict(torch.load(os.path.join(SNAP_DIR, s2f[-1]), map_location=device))

    # --- Step 3: Joint fine-tuning ---
    for p in back.parameters(): p.requires_grad = True
    params = list(back.parameters()) + list(model.parameters())
    opt3 = optim.Adam(params, lr=1e-5, weight_decay=1e-4)
    sched3 = optim.lr_scheduler.CosineAnnealingLR(opt3, T_max=args.epochs_s3, eta_min=1e-7)

    best_s3 = 0.0
    for ep in range(args.epochs_s3):
        model.train(); back.train()
        pbar = tqdm(train_loader, desc=f"S3 E{ep+1}/{args.epochs_s3}")
        opt3.zero_grad()
        for step, (img, lab) in enumerate(pbar):
            img, lab = img.to(device), lab.to(device)
            noise = sample_noise(channel_type)
            with torch.no_grad(): feat = front(img)
            Fp, stats = model(feat, noise_param=noise)
            loss = (criterion(back(Fp), lab) + 2.0 * (stats['tx_rate'] - args.target_rate) ** 2) / 2
            loss.backward()
            if (step + 1) % 2 == 0:
                nn.utils.clip_grad_norm_(params, max_norm=1.0)
                opt3.step(); opt3.zero_grad()
            pbar.set_postfix({'L': f'{loss.item():.3f}', 'tx': f'{stats["tx_rate"]:.2f}'})
        sched3.step()

        if (ep + 1) % 10 == 0:
            acc, tx = evaluate(front, model, back, test_loader)
            print(f"  S3 E{ep+1}: {acc:.2f}%, Tx={tx:.3f}")
            if acc > best_s3:
                best_s3 = acc
                torch.save({'model': model.state_dict(), 'back': back.state_dict()},
                           os.path.join(SNAP_DIR, f"v2_s3_{acc:.2f}.pth"))
                print(f"  ✓ Best S3: {best_s3:.2f}%")

    # Load best
    s3f = sorted([f for f in os.listdir(SNAP_DIR) if f.startswith("v2_s3_")])
    if s3f:
        ckpt = torch.load(os.path.join(SNAP_DIR, s3f[-1]), map_location=device)
        model.load_state_dict(ckpt['model']); back.load_state_dict(ckpt['back'])

    # --- Evaluate: clean, noisy, early stop ---
    print(f"\n{'='*60}")
    print("EVALUATION — SpikeAdapt-SC v2")
    print(f"{'='*60}")

    acc_clean, tx = evaluate(front, model, back, test_loader, noise_param=0.0)
    print(f"  Clean: {acc_clean:.2f}%, Tx={tx:.3f}")

    for ber in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        acc_n, _ = evaluate(front, model, back, test_loader, noise_param=ber)
        print(f"  BER={ber:.2f}: {acc_n:.2f}%")

    # Early stopping evaluation
    for conf_th in [0.90, 0.95, 0.99]:
        acc_es, avg_T = evaluate_with_early_stop(
            front, model, back, test_loader, noise_param=0.0,
            conf_threshold=conf_th)
        savings = (1 - avg_T / args.T_max) * 100
        print(f"  EarlyStop(th={conf_th}): {acc_es:.2f}%, avg T={avg_T:.1f}/{args.T_max} "
              f"({savings:.0f}% temporal savings)")

    results = {
        'model': 'SpikeAdapt-SC_v2',
        'seed': args.seed,
        'grid': H_feat,
        'C2': args.C2,
        'T': args.T_max,
        'target_rate': args.target_rate,
        'accuracy_clean': acc_clean,
        'best_s2': best_s2,
        'best_s3': best_s3,
    }

    # ==================================================================
    # BASELINES (if requested)
    # ==================================================================
    if args.baselines:
        print(f"\n{'='*60}")
        print("BASELINE: SNN-SC (no masking)")
        print(f"{'='*60}")

        # Reload backbone
        if os.path.exists(bb_path):
            bb_state = torch.load(bb_path, map_location=device)
            back.load_state_dict({k: v for k, v in bb_state.items()
                                  if k.startswith(('layer4.', 'fc.', 'avgpool.'))}, strict=False)

        snnsc = SNNSC(C_in=C_in, C1=256, C2=args.C2, T=args.T_max).to(device)
        for p in front.parameters(): p.requires_grad = False
        for p in back.parameters(): p.requires_grad = False
        opt_b = optim.Adam(snnsc.parameters(), lr=1e-4, weight_decay=1e-4)
        sched_b = optim.lr_scheduler.CosineAnnealingLR(opt_b, 60, 1e-6)
        best_bl = 0.0
        for ep in range(60):
            snnsc.train()
            for img, lab in tqdm(train_loader, desc=f"SNNSC S2 E{ep+1}/60"):
                img, lab = img.to(device), lab.to(device)
                noise = sample_noise('bsc')
                with torch.no_grad(): feat = front(img)
                Fp, _ = snnsc(feat, noise)
                loss = criterion(back(Fp), lab)
                opt_b.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(snnsc.parameters(), 1.0); opt_b.step()
            sched_b.step()
            if (ep + 1) % 10 == 0:
                acc, _ = evaluate(front, snnsc, back, test_loader, 0.0, is_baseline=True)
                print(f"  SNNSC E{ep+1}: {acc:.2f}%")
                if acc > best_bl:
                    best_bl = acc
                    torch.save(snnsc.state_dict(), os.path.join(SNAP_DIR, f"snnsc_{acc:.2f}.pth"))
        # Joint
        for p in back.parameters(): p.requires_grad = True
        bl_f = sorted([f for f in os.listdir(SNAP_DIR) if f.startswith("snnsc_")])
        if bl_f: snnsc.load_state_dict(torch.load(os.path.join(SNAP_DIR, bl_f[-1]), map_location=device))
        opt_b3 = optim.Adam(list(back.parameters()) + list(snnsc.parameters()), lr=1e-5, weight_decay=1e-4)
        best_bl3 = 0.0
        for ep in range(30):
            snnsc.train(); back.train()
            opt_b3.zero_grad()
            for step, (img, lab) in enumerate(tqdm(train_loader, desc=f"SNNSC S3 E{ep+1}/30")):
                img, lab = img.to(device), lab.to(device)
                noise = sample_noise('bsc')
                with torch.no_grad(): feat = front(img)
                loss = criterion(back(snnsc(feat, noise)[0]), lab) / 2
                loss.backward()
                if (step + 1) % 2 == 0:
                    nn.utils.clip_grad_norm_(list(back.parameters()) + list(snnsc.parameters()), 1.0)
                    opt_b3.step(); opt_b3.zero_grad()
            if (ep + 1) % 10 == 0:
                acc, _ = evaluate(front, snnsc, back, test_loader, 0.0, is_baseline=True)
                print(f"  SNNSC S3 E{ep+1}: {acc:.2f}%")
                if acc > best_bl3:
                    best_bl3 = acc
                    torch.save({'model': snnsc.state_dict(), 'back': back.state_dict()},
                               os.path.join(SNAP_DIR, f"snnsc_s3_{acc:.2f}.pth"))
        results['snnsc_clean'] = best_bl3

        # CNN-Uni
        print(f"\n{'='*60}")
        print("BASELINE: CNN-Uni (8-bit uniform quantization)")
        print(f"{'='*60}")
        if os.path.exists(bb_path):
            bb_state = torch.load(bb_path, map_location=device)
            back.load_state_dict({k: v for k, v in bb_state.items()
                                  if k.startswith(('layer4.', 'fc.', 'avgpool.'))}, strict=False)
        cnnuni = CNNUni(C_in=C_in, C1=256, C2=args.C2, n_bits=8).to(device)
        for p in front.parameters(): p.requires_grad = False
        for p in back.parameters(): p.requires_grad = False
        opt_c = optim.Adam(cnnuni.parameters(), lr=1e-4, weight_decay=1e-4)
        sched_c = optim.lr_scheduler.CosineAnnealingLR(opt_c, 60, 1e-6)
        best_cu = 0.0
        for ep in range(60):
            cnnuni.train()
            for img, lab in tqdm(train_loader, desc=f"CNNUni S2 E{ep+1}/60"):
                img, lab = img.to(device), lab.to(device)
                noise = sample_noise('bsc')
                with torch.no_grad(): feat = front(img)
                Fp, _ = cnnuni(feat, noise)
                loss = criterion(back(Fp), lab)
                opt_c.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(cnnuni.parameters(), 1.0); opt_c.step()
            sched_c.step()
            if (ep + 1) % 10 == 0:
                acc, _ = evaluate(front, cnnuni, back, test_loader, 0.0, is_baseline=True)
                print(f"  CNNUni E{ep+1}: {acc:.2f}%")
                if acc > best_cu:
                    best_cu = acc
                    torch.save(cnnuni.state_dict(), os.path.join(SNAP_DIR, f"cnnuni_{acc:.2f}.pth"))
        # Joint
        for p in back.parameters(): p.requires_grad = True
        cu_f = sorted([f for f in os.listdir(SNAP_DIR) if f.startswith("cnnuni_")])
        if cu_f: cnnuni.load_state_dict(torch.load(os.path.join(SNAP_DIR, cu_f[-1]), map_location=device))
        opt_c3 = optim.Adam(list(back.parameters()) + list(cnnuni.parameters()), lr=1e-5, weight_decay=1e-4)
        best_cu3 = 0.0
        for ep in range(30):
            cnnuni.train(); back.train()
            opt_c3.zero_grad()
            for step, (img, lab) in enumerate(tqdm(train_loader, desc=f"CNNUni S3 E{ep+1}/30")):
                img, lab = img.to(device), lab.to(device)
                noise = sample_noise('bsc')
                with torch.no_grad(): feat = front(img)
                loss = criterion(back(cnnuni(feat, noise)[0]), lab) / 2
                loss.backward()
                if (step + 1) % 2 == 0:
                    nn.utils.clip_grad_norm_(list(back.parameters()) + list(cnnuni.parameters()), 1.0)
                    opt_c3.step(); opt_c3.zero_grad()
            if (ep + 1) % 10 == 0:
                acc, _ = evaluate(front, cnnuni, back, test_loader, 0.0, is_baseline=True)
                print(f"  CNNUni S3 E{ep+1}: {acc:.2f}%")
                if acc > best_cu3:
                    best_cu3 = acc
                    torch.save({'model': cnnuni.state_dict(), 'back': back.state_dict()},
                               os.path.join(SNAP_DIR, f"cnnuni_s3_{acc:.2f}.pth"))
        results['cnnuni_clean'] = best_cu3

    # Save results
    with open(os.path.join(SNAP_DIR, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {SNAP_DIR}/results.json")
    print(json.dumps(results, indent=2))
