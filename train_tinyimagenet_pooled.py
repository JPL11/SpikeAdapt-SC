# ============================================================================
# SPIKEADAPT-SC: TINY-IMAGENET + AWGN/RAYLEIGH + ENERGY METRICS
# ============================================================================
# GlobeCom enhancements:
#   1. Second dataset: Tiny-ImageNet-200 (200 classes, 64×64)
#   2. AWGN channel: realistic soft-decision channel
#   3. Rayleigh fading: models wireless multipath
#   4. Spike-op energy: count SynOps for energy comparison
#   5. Adaptive bandwidth framing: Tx rate as bandwidth control
# ============================================================================

import os, random, json, math
import numpy as np
from tqdm import tqdm
from collections import defaultdict

os.environ['PYTHONUNBUFFERED'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

SNAP_DIR = "./snapshots_tinyimagenet_pooled/"
os.makedirs(SNAP_DIR, exist_ok=True)
BB_DIR = "./snapshots_tinyimagenet/"  # reuse existing backbone


# ############################################################################
# TINY-IMAGENET DATASET
# ############################################################################

class TinyImageNet(Dataset):
    """Tiny-ImageNet-200: 200 classes, 64×64 RGB images."""
    def __init__(self, root, split='train', transform=None):
        self.root = os.path.join(root, 'tiny-imagenet-200')
        self.split = split
        self.transform = transform
        self.samples = []

        # Build class-to-index mapping
        wnids = sorted(os.listdir(os.path.join(self.root, 'train')))
        self.class_to_idx = {w: i for i, w in enumerate(wnids)}

        if split == 'train':
            for wnid in wnids:
                img_dir = os.path.join(self.root, 'train', wnid, 'images')
                if os.path.isdir(img_dir):
                    for fname in os.listdir(img_dir):
                        if fname.endswith('.JPEG'):
                            self.samples.append((os.path.join(img_dir, fname),
                                                self.class_to_idx[wnid]))
        else:  # val
            val_dir = os.path.join(self.root, 'val')
            ann_file = os.path.join(val_dir, 'val_annotations.txt')
            val_map = {}
            with open(ann_file) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    val_map[parts[0]] = parts[1]
            img_dir = os.path.join(val_dir, 'images')
            for fname in os.listdir(img_dir):
                if fname.endswith('.JPEG') and fname in val_map:
                    self.samples.append((os.path.join(img_dir, fname),
                                        self.class_to_idx[val_map[fname]]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# ############################################################################
# CHANNEL MODELS
# ############################################################################

class BSC_Channel(nn.Module):
    """Binary Symmetric Channel."""
    def forward(self, x, snr_or_ber):
        if snr_or_ber <= 0: return x
        flip = (torch.rand_like(x.float()) < snr_or_ber).float()
        x_n = (x + flip) % 2
        return x + (x_n - x).detach() if self.training else x_n

class AWGN_Channel(nn.Module):
    """AWGN Channel for binary signals.
    Maps binary {0,1} → BPSK {-1,+1}, adds Gaussian noise, hard-decision decode.
    SNR in dB.
    """
    def forward(self, x, snr_db):
        if snr_db >= 100: return x  # no noise
        # BPSK modulation: 0 → -1, 1 → +1
        bpsk = 2.0 * x - 1.0
        # Add noise
        snr_linear = 10 ** (snr_db / 10.0)
        noise_std = 1.0 / math.sqrt(2 * snr_linear)
        noise = torch.randn_like(bpsk) * noise_std
        received = bpsk + noise
        # Hard decision: received > 0 → 1, else → 0
        decoded = (received > 0).float()
        if self.training:
            return x + (decoded - x).detach()  # STE
        return decoded

class Rayleigh_Channel(nn.Module):
    """Rayleigh fading + AWGN.
    Models wireless multipath: y = h*x + n, where h ~ Rayleigh.
    Assumes perfect CSI at receiver (coherent detection).
    """
    def forward(self, x, snr_db):
        if snr_db >= 100: return x
        bpsk = 2.0 * x - 1.0
        snr_linear = 10 ** (snr_db / 10.0)
        # Rayleigh fading coefficient (per-bit independent)
        h_real = torch.randn_like(bpsk) / math.sqrt(2)
        h_imag = torch.randn_like(bpsk) / math.sqrt(2)
        h_mag = torch.sqrt(h_real**2 + h_imag**2)
        # Faded signal + noise
        noise_std = 1.0 / math.sqrt(2 * snr_linear)
        noise = torch.randn_like(bpsk) * noise_std
        received = h_mag * bpsk + noise
        # Coherent detection (perfect CSI): divide by h, then hard decision
        equalized = received / (h_mag + 1e-8)
        decoded = (equalized > 0).float()
        if self.training:
            return x + (decoded - x).detach()
        return decoded


def snr_to_ber_awgn(snr_db):
    """Theoretical BER for BPSK over AWGN."""
    from scipy.special import erfc
    snr_lin = 10 ** (snr_db / 10.0)
    return 0.5 * erfc(math.sqrt(snr_lin))

def snr_to_ber_rayleigh(snr_db):
    """Theoretical BER for BPSK over Rayleigh fading."""
    snr_lin = 10 ** (snr_db / 10.0)
    return 0.5 * (1 - math.sqrt(snr_lin / (1 + snr_lin)))


# ############################################################################
# SPIKE OPERATION COUNTER
# ############################################################################

class SpikeOpCounter:
    """Counts synaptic operations (SynOps) for energy estimation.
    
    For SNN layers: SynOps = spike_count × fan_out
    For ANN layers: MACs = input_elements × fan_out
    
    Energy ratio: SNN SynOp ≈ 0.9 pJ on 45nm (Horowitz 2014)
                  ANN MAC   ≈ 4.6 pJ on 45nm
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.snn_synops = 0
        self.ann_macs = 0
        self.spike_counts = defaultdict(int)
        self.total_elements = defaultdict(int)
    
    def count_snn_layer(self, spikes, layer_name, out_channels):
        """Count SynOps for an SNN conv layer."""
        n_spikes = spikes.sum().item()
        # Each spike triggers fan_out operations
        # For conv3x3: fan_out = out_channels * 3 * 3
        fan_out = out_channels * 9  # 3×3 kernel
        self.snn_synops += n_spikes * fan_out
        self.spike_counts[layer_name] += n_spikes
        self.total_elements[layer_name] += spikes.numel()
    
    def count_ann_layer(self, input_tensor, out_channels, kernel_size=3):
        """Count MACs for equivalent ANN layer."""
        B, C, H, W = input_tensor.shape
        macs_per_element = C * kernel_size * kernel_size * out_channels
        total_macs = B * H * W * macs_per_element
        self.ann_macs += total_macs
    
    def get_energy_ratio(self):
        """Return energy ratio: SNN/ANN."""
        E_SNN = self.snn_synops * 0.9   # pJ per SynOp
        E_ANN = self.ann_macs * 4.6      # pJ per MAC
        return E_SNN / max(E_ANN, 1)
    
    def get_summary(self):
        firing_rates = {}
        for name in self.spike_counts:
            if self.total_elements[name] > 0:
                firing_rates[name] = self.spike_counts[name] / self.total_elements[name]
        return {
            'snn_synops': self.snn_synops,
            'ann_macs': self.ann_macs,
            'energy_ratio': self.get_energy_ratio(),
            'firing_rates': firing_rates,
            'energy_savings_pct': (1 - self.get_energy_ratio()) * 100,
        }


# ############################################################################
# CORE SNN
# ############################################################################

class SpikeFunction(torch.autograd.Function):
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
        return grad_output * sg, -(grad_output * sg).sum() if ctx.th_needs_grad else None

class IFNeuron(nn.Module):
    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = threshold
    def forward(self, x, mem=None):
        if mem is None: mem = torch.zeros_like(x)
        mem = mem + x
        sp = SpikeFunction.apply(mem, self.threshold)
        return sp, mem - sp * self.threshold

class IHFNeuron(nn.Module):
    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(float(threshold)))
    def forward(self, x, mem=None):
        if mem is None: mem = torch.zeros_like(x)
        mem = mem + x
        sp = SpikeFunction.apply(mem, self.threshold)
        return sp, mem - sp * self.threshold


# ############################################################################
# BACKBONE — Using ResNet50 for both CIFAR-100 and Tiny-ImageNet
# ############################################################################

class ResNet50Front(nn.Module):
    """Split at layer3 + pool to 8×8. Output: (1024, 8, 8)."""
    def __init__(self, input_size=64, pool_size=8):
        super().__init__()
        r = torchvision.models.resnet50(weights=None)
        if input_size <= 64:
            self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            self.maxpool = nn.Identity()
        else:
            self.conv1 = r.conv1
            self.maxpool = r.maxpool
        self.bn1 = r.bn1; self.relu = r.relu
        self.layer1 = r.layer1; self.layer2 = r.layer2; self.layer3 = r.layer3
        self.spatial_pool = nn.AdaptiveAvgPool2d(pool_size)  # 16×16 → 8×8
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(self.maxpool(x))))
        return self.spatial_pool(x)  # Pool to 8×8

class ResNet50Back(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()
        r = torchvision.models.resnet50(weights=None)
        self.layer4 = r.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)
    def forward(self, x):
        return self.fc(torch.flatten(self.avgpool(self.layer4(x)), 1))


# ############################################################################
# SPIKEADAPT-SC with multi-channel support + spike counting
# ############################################################################

class LearnedImportanceScorer(nn.Module):
    def __init__(self, C_in=128, hidden=32):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Conv2d(C_in, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1), nn.Sigmoid()
        )
    def forward(self, all_S2):
        return self.scorer(torch.stack(all_S2, dim=0).mean(dim=0)).squeeze(1)

class LearnedBlockMask(nn.Module):
    def __init__(self, target_rate=0.75, temperature=0.5):
        super().__init__()
        self.target_rate = target_rate; self.temperature = temperature
    def forward(self, importance, training=True):
        B, H, W = importance.shape
        if training:
            logits = torch.log(importance / (1 - importance + 1e-7) + 1e-7)
            u = torch.rand_like(logits).clamp(1e-7, 1-1e-7)
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
    def __init__(self, C_in=1024, C1=256, C2=128):
        super().__init__()
        self.C1, self.C2 = C1, C2
        self.conv1 = nn.Conv2d(C_in, C1, 3, 1, 1); self.bn1 = nn.BatchNorm2d(C1); self.if1 = IFNeuron()
        self.conv2 = nn.Conv2d(C1, C2, 3, 1, 1); self.bn2 = nn.BatchNorm2d(C2); self.if2 = IFNeuron()
    def forward(self, F, m1=None, m2=None):
        s1, m1 = self.if1(self.bn1(self.conv1(F)), m1)
        s2, m2 = self.if2(self.bn2(self.conv2(s1)), m2)
        return s1, s2, m1, m2  # Return s1 too for spike counting

class Decoder(nn.Module):
    def __init__(self, C_out=1024, C1=256, C2=128, T=8):
        super().__init__()
        self.T = T; self.C1 = C1; self.C_out = C_out
        self.conv3 = nn.Conv2d(C2, C1, 3, 1, 1); self.bn3 = nn.BatchNorm2d(C1); self.if3 = IFNeuron()
        self.conv4 = nn.Conv2d(C1, C_out, 3, 1, 1); self.bn4 = nn.BatchNorm2d(C_out); self.ihf = IHFNeuron()
        self.converter_fc = nn.Linear(2*T, 2*T)
    def forward(self, recv_all, mask):
        m3, m4 = None, None; Fs, Fm = [], []; s3_all = []
        for t in range(self.T):
            s3, m3 = self.if3(self.bn3(self.conv3(recv_all[t] * mask)), m3)
            sp, m4 = self.ihf(self.bn4(self.conv4(s3)), m4)
            Fs.append(sp); Fm.append(m4.clone()); s3_all.append(s3)
        il = []
        for t in range(self.T): il.append(Fs[t]); il.append(Fm[t])
        stk = torch.stack(il, dim=1)
        x = stk.permute(0, 2, 3, 4, 1)
        return (x * torch.sigmoid(self.converter_fc(x))).sum(dim=-1), s3_all, Fs

class SpikeAdaptSC(nn.Module):
    def __init__(self, C_in=1024, C1=256, C2=128, T=8,
                 target_rate=0.75, channel_type='bsc'):
        super().__init__()
        self.T = T
        self.encoder = Encoder(C_in, C1, C2)
        self.importance_scorer = LearnedImportanceScorer(C_in=C2, hidden=32)
        self.block_mask = LearnedBlockMask(target_rate, 0.5)
        self.decoder = Decoder(C_in, C1, C2, T)
        if channel_type == 'awgn':
            self.channel = AWGN_Channel()
        elif channel_type == 'rayleigh':
            self.channel = Rayleigh_Channel()
        else:
            self.channel = BSC_Channel()
        self.channel_type = channel_type
    
    def forward(self, feat, noise_param=0.0, target_rate_override=None):
        all_S1, all_S2, m1, m2 = [], [], None, None
        for t in range(self.T):
            s1, s2, m1, m2 = self.encoder(feat, m1, m2)
            all_S1.append(s1); all_S2.append(s2)
        importance = self.importance_scorer(all_S2)
        if target_rate_override is not None:
            old = self.block_mask.target_rate; self.block_mask.target_rate = target_rate_override
            mask, tx = self.block_mask(importance, training=False)
            self.block_mask.target_rate = old
        else:
            mask, tx = self.block_mask(importance, training=self.training)
        recv = [self.channel(self.block_mask.apply_mask(all_S2[t], mask), noise_param)
                for t in range(self.T)]
        Fp, s3_all, s4_all = self.decoder(recv, mask)
        return Fp, {'tx_rate': tx.item(), 'mask': mask,
                    'all_S1': all_S1, 'all_S2': all_S2,
                    's3_all': s3_all, 's4_all': s4_all}


def count_spike_ops(stats, encoder, decoder):
    """Count SynOps from a forward pass."""
    counter = SpikeOpCounter()
    for t, (s1, s2) in enumerate(zip(stats['all_S1'], stats['all_S2'])):
        counter.count_snn_layer(s1, f'enc_if1_t{t}', encoder.C2)
        counter.count_snn_layer(s2, f'enc_if2_t{t}', encoder.C2)
    for t, (s3, s4) in enumerate(zip(stats['s3_all'], stats['s4_all'])):
        counter.count_snn_layer(s3, f'dec_if3_t{t}', decoder.C1)
        counter.count_snn_layer(s4, f'dec_if4_t{t}', decoder.C_out)
    # ANN equivalent: same conv layers but with dense activations
    s1_ex = stats['all_S1'][0]
    counter.count_ann_layer(s1_ex, encoder.C2)  # enc conv2
    counter.count_ann_layer(s1_ex, encoder.C2)  # repeated T times would be T×
    counter.ann_macs *= len(stats['all_S1'])  # Scale by T
    return counter


def evaluate(front, model, back, loader, noise_param=0.0, count_energy=False):
    front.eval(); model.eval(); back.eval()
    correct, total, tx_sum, n = 0, 0, 0.0, 0
    energy_data = None
    with torch.no_grad():
        for img, lab in loader:
            img, lab = img.to(device), lab.to(device)
            Fp, stats = model(front(img), noise_param=noise_param)
            correct += back(Fp).argmax(1).eq(lab).sum().item()
            total += lab.size(0); tx_sum += stats['tx_rate']; n += 1
            if count_energy and energy_data is None:
                energy_data = count_spike_ops(stats, model.encoder, model.decoder).get_summary()
    return 100.*correct/total, tx_sum/max(n,1), energy_data


def sample_noise(channel_type):
    """Sample noise parameter based on channel type."""
    if channel_type == 'bsc':
        if random.random() < 0.5:
            return random.uniform(0.15, 0.4)
        return random.uniform(0, 0.15)
    else:  # awgn/rayleigh — sample SNR in dB
        # Low SNR (noisy) to high SNR (clean)
        if random.random() < 0.5:
            return random.uniform(-2, 5)   # Noisy
        return random.uniform(5, 20)        # Clean


# ############################################################################
# MAIN
# ############################################################################

if __name__ == "__main__":
    NUM_CLASSES = 200
    INPUT_SIZE = 64

    # Tiny-ImageNet transforms
    train_tf = T.Compose([
        T.RandomCrop(64, padding=8),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
    ])

    print("Loading Tiny-ImageNet...")
    train_ds = TinyImageNet("./data", 'train', transform=train_tf)
    test_ds = TinyImageNet("./data", 'val', transform=test_tf)
    print(f"  Train: {len(train_ds)}, Val: {len(test_ds)}")

    train_loader = DataLoader(train_ds, 64, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, 128, shuffle=False, num_workers=4, pin_memory=True)

    # ==================================================================
    # STEP 1: Train backbone on Tiny-ImageNet
    # ==================================================================
    front = ResNet50Front(INPUT_SIZE).to(device)
    back = ResNet50Back(NUM_CLASSES).to(device)

    # Reuse backbone from the original (non-pooled) training
    bb_path = os.path.join(BB_DIR, "backbone_best.pth")
    if os.path.exists(bb_path):
        state = torch.load(bb_path, map_location=device)
        front.load_state_dict({k: v for k, v in state.items()
                               if not k.startswith(('layer4.', 'fc.', 'avgpool.'))}, strict=False)
        back.load_state_dict({k: v for k, v in state.items()
                              if k.startswith(('layer4.', 'fc.', 'avgpool.'))}, strict=False)
        print(f"✓ Backbone loaded from {bb_path} (pooling {front.spatial_pool})")
    else:
        print(f"ERROR: No backbone found at {bb_path}")
        print("Run train_tinyimagenet.py first to train the backbone.")
        sys.exit(1)

    front.eval()
    with torch.no_grad():
        d = front(torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(device))
        print(f"  Feature shape: {d.shape}")  # (1, 1024, H, W)
    C_in = d.shape[1]
    H_feat, W_feat = d.shape[2], d.shape[3]

    # ==================================================================
    # STEP 2+3: Train SpikeAdapt-SC for each channel type
    # ==================================================================

    all_results = {}

    for channel_type in ['bsc', 'awgn', 'rayleigh']:
        print(f"\n{'='*60}")
        print(f"CHANNEL: {channel_type.upper()}")
        print(f"{'='*60}")

        model = SpikeAdaptSC(C_in=C_in, C1=256, C2=128, T=8,
                              target_rate=0.75, channel_type=channel_type).to(device)

        # Reload backbone back
        bb_state = torch.load(bb_path, map_location=device)
        back.load_state_dict({k: v for k, v in bb_state.items()
                              if k.startswith(('layer4.', 'fc.', 'avgpool.'))}, strict=False)

        # Step 2
        for p in front.parameters(): p.requires_grad = False
        for p in back.parameters(): p.requires_grad = False

        criterion = nn.CrossEntropyLoss()
        opt = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60, eta_min=1e-6)

        best_s2 = 0.0
        for ep in range(60):
            model.train()
            pbar = tqdm(train_loader, desc=f"{channel_type} S2 E{ep+1}/60")
            for img, lab in pbar:
                img, lab = img.to(device), lab.to(device)
                noise = sample_noise(channel_type)
                with torch.no_grad(): feat = front(img)
                Fp, stats = model(feat, noise_param=noise)
                loss = criterion(back(Fp), lab) + 2.0 * (stats['tx_rate'] - 0.75) ** 2
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                pbar.set_postfix({'L': f'{loss.item():.3f}', 'tx': f'{stats["tx_rate"]:.2f}'})
            sched.step()
            if (ep + 1) % 10 == 0:
                acc, tx, _ = evaluate(front, model, back, test_loader, noise_param=0.0)
                print(f"  S2 E{ep+1}: {acc:.2f}%, Tx={tx:.3f}")
                if acc > best_s2:
                    best_s2 = acc
                    torch.save(model.state_dict(),
                               os.path.join(SNAP_DIR, f"{channel_type}_s2_{acc:.2f}.pth"))

        s2f = sorted([f for f in os.listdir(SNAP_DIR) if f.startswith(f"{channel_type}_s2_")])
        if s2f: model.load_state_dict(torch.load(os.path.join(SNAP_DIR, s2f[-1]), map_location=device))

        # Step 3
        for p in back.parameters(): p.requires_grad = True
        params = list(back.parameters()) + list(model.parameters())
        opt3 = optim.Adam(params, lr=1e-5, weight_decay=1e-4)
        sched3 = optim.lr_scheduler.CosineAnnealingLR(opt3, T_max=30, eta_min=1e-7)
        loader_s3 = DataLoader(train_ds, 32, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

        best_s3 = 0.0
        for ep in range(30):
            model.train(); back.train()
            pbar = tqdm(loader_s3, desc=f"{channel_type} S3 E{ep+1}/30")
            opt3.zero_grad()
            for step, (img, lab) in enumerate(pbar):
                img, lab = img.to(device), lab.to(device)
                noise = sample_noise(channel_type)
                with torch.no_grad(): feat = front(img)
                Fp, stats = model(feat, noise_param=noise)
                loss = (criterion(back(Fp), lab) + 2.0*(stats['tx_rate']-0.75)**2) / 2
                loss.backward()
                if (step+1) % 2 == 0:
                    nn.utils.clip_grad_norm_(params, max_norm=1.0)
                    opt3.step(); opt3.zero_grad()
            sched3.step()
            if (ep + 1) % 10 == 0:
                acc, tx, _ = evaluate(front, model, back, test_loader, noise_param=0.0)
                print(f"  S3 E{ep+1}: {acc:.2f}%, Tx={tx:.3f}")
                if acc > best_s3:
                    best_s3 = acc
                    torch.save({'model': model.state_dict(), 'back': back.state_dict()},
                               os.path.join(SNAP_DIR, f"{channel_type}_s3_{acc:.2f}.pth"))

        # Load best
        s3f = sorted([f for f in os.listdir(SNAP_DIR) if f.startswith(f"{channel_type}_s3_")])
        if s3f:
            ckpt = torch.load(os.path.join(SNAP_DIR, s3f[-1]), map_location=device)
            model.load_state_dict(ckpt['model']); back.load_state_dict(ckpt['back'])

        # Evaluate
        print(f"\n  {channel_type.upper()} Final Evaluation:")
        chan_results = []
        if channel_type == 'bsc':
            noise_params = [0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
            noise_label = 'BER'
        else:
            noise_params = [20, 15, 10, 7, 5, 3, 1, 0, -2]
            noise_label = 'SNR(dB)'

        for np_val in noise_params:
            accs = [evaluate(front, model, back, test_loader, noise_param=np_val)[0]
                    for _ in range(5)]
            m, s = np.mean(accs), np.std(accs)
            chan_results.append({'noise': np_val, 'mean': m, 'std': s})
            print(f"    {noise_label}={np_val}: {m:.2f}% ±{s:.2f}")

        # Energy
        acc, tx, energy = evaluate(front, model, back, test_loader, noise_param=0.0, count_energy=True)
        print(f"\n  Energy: SynOps={energy['snn_synops']:,.0f}, "
              f"ANN MACs={energy['ann_macs']:,.0f}")
        print(f"  Energy ratio: {energy['energy_ratio']:.4f} "
              f"({energy['energy_savings_pct']:.1f}% savings)")

        # Rate sweep
        rate_results = []
        for rate in [1.0, 0.9, 0.75, 0.5, 0.25]:
            a, t, _ = evaluate(front, model, back, test_loader, noise_param=0.0)
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for img, lab in test_loader:
                    img, lab = img.to(device), lab.to(device)
                    Fp, _ = model(front(img), noise_param=0.0, target_rate_override=rate)
                    correct += back(Fp).argmax(1).eq(lab).sum().item()
                    total += lab.size(0)
            rate_results.append({'rate': rate, 'acc': 100.*correct/total})
            print(f"    Rate={rate:.2f}: {100.*correct/total:.2f}%")

        all_results[channel_type] = {
            'noise_sweep': chan_results,
            'energy': energy,
            'rate_sweep': rate_results,
            'best_s2': best_s2,
            'best_s3': best_s3,
        }

    # Save all results
    with open(os.path.join(SNAP_DIR, "all_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✅ TINY-IMAGENET TRAINING COMPLETE!")
    print(f"Results saved to {SNAP_DIR}all_results.json")
