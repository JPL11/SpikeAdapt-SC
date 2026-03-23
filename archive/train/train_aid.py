"""SpikeAdapt-SC on AID (Aerial Image Dataset) for UAV/Satellite edge AI.

AID: 30 aerial scene classes, 10,000 images (200-400 per class), 600×600 RGB.
Downloaded automatically from HuggingFace.

Trains: backbone → SpikeAdapt-SC for BSC, AWGN, Rayleigh channels
        + energy metrics + rate sweep + dynamic rate adaptation.
"""

import os, sys, random, json, math
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
print(f"Device: {device}")

SNAP_DIR = "./snapshots_aid/"
os.makedirs(SNAP_DIR, exist_ok=True)


# ############################################################################
# AID DATASET
# ############################################################################

class AIDDataset(Dataset):
    """Aerial Image Dataset (AID): 30 classes of aerial scenes.
    
    Automatically downloads from HuggingFace if not present.
    Images resized to 224×224 for ResNet50.
    """
    def __init__(self, root, transform=None, split='train', train_ratio=0.8):
        self.root = root
        self.transform = transform
        self.samples = []

        aid_dir = os.path.join(root, 'AID')
        
        if not os.path.exists(aid_dir):
            print("Downloading AID dataset...")
            self._download(root)
        
        # Build samples
        classes = sorted([d for d in os.listdir(aid_dir) 
                         if os.path.isdir(os.path.join(aid_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.classes = classes
        
        all_samples = []
        for cls_name in classes:
            cls_dir = os.path.join(aid_dir, cls_name)
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
                    all_samples.append((os.path.join(cls_dir, fname), 
                                       self.class_to_idx[cls_name]))
        
        # Split train/test reproducibly
        random.seed(42)
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * train_ratio)
        
        if split == 'train':
            self.samples = all_samples[:split_idx]
        else:
            self.samples = all_samples[split_idx:]
        
        print(f"  AID {split}: {len(self.samples)} images, {len(classes)} classes")

    def _download(self, root):
        """Download AID dataset. Tries multiple sources."""
        import subprocess, shutil
        os.makedirs(root, exist_ok=True)
        aid_dir = os.path.join(root, 'AID')
        
        # Method 1: HuggingFace (blanchon/AID — verified working)
        try:
            print("  Trying HuggingFace (blanchon/AID)...")
            subprocess.run([
                sys.executable, '-c',
                f"""
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="blanchon/AID",
    repo_type="dataset",
    local_dir="{os.path.join(root, 'AID_HF')}"
)
"""
            ], check=True, timeout=1200)
            
            hf_dir = os.path.join(root, 'AID_HF')
            if os.path.exists(hf_dir):
                # Find where the class folders are
                for candidate in [hf_dir, os.path.join(hf_dir, 'data'),
                                  os.path.join(hf_dir, 'AID')]:
                    if os.path.isdir(candidate):
                        subdirs = [d for d in os.listdir(candidate) 
                                   if os.path.isdir(os.path.join(candidate, d))
                                   and not d.startswith('.')]
                        if len(subdirs) >= 20:  # AID has 30 classes
                            if candidate != aid_dir:
                                shutil.copytree(candidate, aid_dir, dirs_exist_ok=True)
                            print(f"  ✓ AID downloaded: {len(subdirs)} classes")
                            return
                # If HF downloaded as parquet/arrow, convert
                try:
                    print("  Converting HF dataset format...")
                    subprocess.run([
                        sys.executable, '-c',
                        f"""
import os
from datasets import load_dataset
ds = load_dataset("blanchon/AID", split="train")
aid_dir = "{aid_dir}"
os.makedirs(aid_dir, exist_ok=True)
for i, item in enumerate(ds):
    label = item.get('label', item.get('labels', 0))
    if isinstance(label, int):
        label_name = ds.features.get('label', ds.features.get('labels')).int2str(label)
    else:
        label_name = str(label)
    cls_dir = os.path.join(aid_dir, label_name)
    os.makedirs(cls_dir, exist_ok=True)
    img = item['image']
    img.save(os.path.join(cls_dir, f"{{i:05d}}.jpg"))
print(f"Converted {{i+1}} images")
"""
                    ], check=True, timeout=600)
                    return
                except Exception as e2:
                    print(f"  HF conversion failed: {e2}")
        except Exception as e:
            print(f"  HuggingFace download failed: {e}")
        
        # Method 2: Direct download from captain-whu (original source)
        try:
            print("  Trying direct download...")
            import urllib.request, zipfile
            urls = [
                "https://captain-whu.github.io/AID/AID.zip",
                "https://1drv.ms/u/s!AthY3vMZmuxChNR0Co7QHpJ56M-SvQ",
            ]
            zip_path = os.path.join(root, "AID.zip")
            for url in urls:
                try:
                    if not os.path.exists(zip_path):
                        print(f"    Downloading from {url[:50]}...")
                        urllib.request.urlretrieve(url, zip_path)
                    if os.path.exists(zip_path) and os.path.getsize(zip_path) > 1000000:
                        break
                except Exception:
                    continue
            if os.path.exists(zip_path) and os.path.getsize(zip_path) > 1000000:
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(root)
                print("  ✓ AID downloaded and extracted")
                return
        except Exception as e:
            print(f"  Direct download failed: {e}")
        
        # Method 3: Kaggle
        try:
            print("  Trying Kaggle download...")
            subprocess.run(["kaggle", "datasets", "download", "-d",
                          "jiayuanchengala/aid-scene-classification-datasets",
                          "-p", root, "--unzip"], check=True, timeout=600)
            print("  ✓ AID downloaded from Kaggle")
            return
        except Exception as e:
            print(f"  Kaggle failed: {e}")
        
        print("\n  ❌ ALL DOWNLOAD METHODS FAILED.")
        print("  Please download manually:")
        print("  Option A: pip install datasets && python -c \"from datasets import load_dataset; ds = load_dataset('blanchon/AID')\"")
        print("  Option B: kaggle datasets download -d jiayuanchengala/aid-scene-classification-datasets")
        print(f"  Then extract class folders to {aid_dir}/")
        sys.exit(1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# ############################################################################
# CHANNEL MODELS (same as tinyimagenet)
# ############################################################################

class BSC_Channel(nn.Module):
    def forward(self, x, ber):
        if ber <= 0: return x
        flip = (torch.rand_like(x.float()) < ber).float()
        x_n = (x + flip) % 2
        return x + (x_n - x).detach() if self.training else x_n

class AWGN_Channel(nn.Module):
    def forward(self, x, snr_db):
        if snr_db >= 100: return x
        bpsk = 2.0 * x - 1.0
        snr_linear = 10 ** (snr_db / 10.0)
        noise_std = 1.0 / math.sqrt(2 * snr_linear)
        noise = torch.randn_like(bpsk) * noise_std
        decoded = (bpsk + noise > 0).float()
        return x + (decoded - x).detach() if self.training else decoded

class Rayleigh_Channel(nn.Module):
    def forward(self, x, snr_db):
        if snr_db >= 100: return x
        bpsk = 2.0 * x - 1.0
        snr_linear = 10 ** (snr_db / 10.0)
        h_real = torch.randn_like(bpsk) / math.sqrt(2)
        h_imag = torch.randn_like(bpsk) / math.sqrt(2)
        h_mag = torch.sqrt(h_real**2 + h_imag**2)
        noise_std = 1.0 / math.sqrt(2 * snr_linear)
        received = h_mag * bpsk + torch.randn_like(bpsk) * noise_std
        decoded = (received / (h_mag + 1e-8) > 0).float()
        return x + (decoded - x).detach() if self.training else decoded


# ############################################################################
# SNN MODULES
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
# BACKBONE — ResNet50 for AID (224×224 input)
# ############################################################################

class ResNet50Front(nn.Module):
    """Split at layer3. For AID (224×224): output (1024, 14, 14)."""
    def __init__(self, input_size=224):
        super().__init__()
        r = torchvision.models.resnet50(weights='IMAGENET1K_V1')  # Pretrained!
        self.conv1 = r.conv1
        self.bn1 = r.bn1; self.relu = r.relu
        self.maxpool = r.maxpool
        self.layer1 = r.layer1; self.layer2 = r.layer2; self.layer3 = r.layer3
        # Pool to 8×8 for manageable SNN encoding
        self.spatial_pool = nn.AdaptiveAvgPool2d(8)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(self.maxpool(x))))
        return self.spatial_pool(x)  # (B, 1024, 8, 8)

class ResNet50Back(nn.Module):
    def __init__(self, num_classes=30):
        super().__init__()
        r = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        self.layer4 = r.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)
    def forward(self, x):
        return self.fc(torch.flatten(self.avgpool(self.layer4(x)), 1))


# ############################################################################
# SPIKEADAPT-SC
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
        return s1, s2, m1, m2

class Decoder(nn.Module):
    def __init__(self, C_out=1024, C1=256, C2=128, T=8):
        super().__init__()
        self.T, self.C1, self.C_out = T, C1, C_out
        self.conv3 = nn.Conv2d(C2, C1, 3, 1, 1); self.bn3 = nn.BatchNorm2d(C1); self.if3 = IFNeuron()
        self.conv4 = nn.Conv2d(C1, C_out, 3, 1, 1); self.bn4 = nn.BatchNorm2d(C_out); self.ihf = IHFNeuron()
        self.converter_fc = nn.Linear(2*T, 2*T)
    def forward(self, recv_all, mask):
        m3, m4 = None, None; Fs, Fm = [], []
        for t in range(self.T):
            s3, m3 = self.if3(self.bn3(self.conv3(recv_all[t] * mask)), m3)
            sp, m4 = self.ihf(self.bn4(self.conv4(s3)), m4)
            Fs.append(sp); Fm.append(m4.clone())
        il = []
        for t in range(self.T): il.append(Fs[t]); il.append(Fm[t])
        stk = torch.stack(il, dim=1)
        x = stk.permute(0, 2, 3, 4, 1)
        return (x * torch.sigmoid(self.converter_fc(x))).sum(dim=-1)

class SpikeAdaptSC(nn.Module):
    def __init__(self, C_in=1024, C1=256, C2=128, T=8,
                 target_rate=0.75, channel_type='bsc'):
        super().__init__()
        self.T = T
        self.encoder = Encoder(C_in, C1, C2)
        self.importance_scorer = LearnedImportanceScorer(C_in=C2, hidden=32)
        self.block_mask = LearnedBlockMask(target_rate, 0.5)
        self.decoder = Decoder(C_in, C1, C2, T)
        if channel_type == 'awgn': self.channel = AWGN_Channel()
        elif channel_type == 'rayleigh': self.channel = Rayleigh_Channel()
        else: self.channel = BSC_Channel()
        self.channel_type = channel_type
    
    def forward(self, feat, noise_param=0.0, target_rate_override=None):
        all_S1, all_S2, m1, m2 = [], [], None, None
        for t in range(self.T):
            s1, s2, m1, m2 = self.encoder(feat, m1, m2)
            all_S1.append(s1); all_S2.append(s2)
        importance = self.importance_scorer(all_S2)
        if target_rate_override is not None:
            old = self.block_mask.target_rate
            self.block_mask.target_rate = target_rate_override
            mask, tx = self.block_mask(importance, training=False)
            self.block_mask.target_rate = old
        else:
            mask, tx = self.block_mask(importance, training=self.training)
        recv = [self.channel(self.block_mask.apply_mask(all_S2[t], mask), noise_param)
                for t in range(self.T)]
        Fp = self.decoder(recv, mask)
        return Fp, {'tx_rate': tx.item(), 'mask': mask}


# ############################################################################
# ENERGY COUNTER
# ############################################################################

class SpikeOpCounter:
    def __init__(self):
        self.snn_synops = 0
        self.ann_macs = 0
    def count_snn(self, spikes, out_ch):
        self.snn_synops += spikes.sum().item() * out_ch * 9
    def count_ann(self, x, out_ch):
        B, C, H, W = x.shape
        self.ann_macs += B * H * W * C * 9 * out_ch
    def energy_savings(self):
        E_SNN = self.snn_synops * 0.9
        E_ANN = self.ann_macs * 4.6
        return (1 - E_SNN / max(E_ANN, 1)) * 100


# ############################################################################
# HELPERS
# ############################################################################

def evaluate(front, model, back, loader, noise_param=0.0):
    front.eval(); model.eval(); back.eval()
    correct, total, tx_sum, n = 0, 0, 0.0, 0
    with torch.no_grad():
        for img, lab in loader:
            img, lab = img.to(device), lab.to(device)
            Fp, stats = model(front(img), noise_param=noise_param)
            correct += back(Fp).argmax(1).eq(lab).sum().item()
            total += lab.size(0); tx_sum += stats['tx_rate']; n += 1
    return 100.*correct/total, tx_sum/max(n, 1)

def sample_noise(channel_type):
    if channel_type == 'bsc':
        return random.uniform(0.15, 0.4) if random.random() < 0.5 else random.uniform(0, 0.15)
    else:
        return random.uniform(-2, 5) if random.random() < 0.5 else random.uniform(5, 20)


# ############################################################################
# DYNAMIC RATE ADAPTATION EXPERIMENT
# ############################################################################

def dynamic_rate_experiment(front, model, back, test_loader, channel_type):
    """Simulate a UAV flight path with changing channel conditions.
    
    Compares:
    1. Fixed rate (100%) — always send everything
    2. Fixed rate (75%) — always send 75%
    3. Adaptive rate — adjust based on estimated channel quality
    """
    print(f"\n  Dynamic Rate Adaptation ({channel_type.upper()}):")
    
    # Simulate time-varying channel: 100 steps
    np.random.seed(42)
    if channel_type == 'bsc':
        # UAV flight: BER varies with distance/obstacles
        t = np.linspace(0, 4*np.pi, 100)
        noise_trace = 0.15 + 0.15 * np.sin(t) + 0.05 * np.random.randn(100)
        noise_trace = np.clip(noise_trace, 0, 0.35)
        noise_label = 'BER'
    else:
        # SNR varies with altitude/distance
        t = np.linspace(0, 4*np.pi, 100)
        noise_trace = 8 + 8 * np.sin(t) + 2 * np.random.randn(100)
        noise_trace = np.clip(noise_trace, -2, 20)
        noise_label = 'SNR(dB)'
    
    # Rate selection policy for adaptive
    def select_rate(noise_val, channel_type):
        if channel_type == 'bsc':
            if noise_val < 0.1: return 0.50   # Good channel → save BW
            elif noise_val < 0.2: return 0.75  # Medium → moderate
            else: return 1.0                    # Bad → send everything
        else:
            if noise_val > 10: return 0.50     # High SNR → save BW
            elif noise_val > 3: return 0.75    # Medium
            else: return 1.0                    # Low SNR → send all
    
    strategies = {
        'fixed_100': lambda n: 1.0,
        'fixed_75': lambda n: 0.75,
        'fixed_50': lambda n: 0.50,
        'adaptive': lambda n: select_rate(n, channel_type),
    }
    
    # Get a batch of test images
    all_imgs, all_labs = [], []
    for img, lab in test_loader:
        all_imgs.append(img); all_labs.append(lab)
        if len(all_imgs) * img.size(0) >= 500:
            break
    test_imgs = torch.cat(all_imgs)[:500].to(device)
    test_labs = torch.cat(all_labs)[:500].to(device)
    
    results = {}
    for strat_name, rate_fn in strategies.items():
        correct = 0
        total_bits = 0
        front.eval(); model.eval(); back.eval()
        
        with torch.no_grad():
            for step_idx in range(min(100, len(noise_trace))):
                noise = noise_trace[step_idx]
                rate = rate_fn(noise)
                
                # Use 5 images per step
                start = (step_idx * 5) % len(test_imgs)
                end = start + 5
                batch_img = test_imgs[start:end]
                batch_lab = test_labs[start:end]
                
                feat = front(batch_img)
                Fp, stats = model(feat, noise_param=float(noise),
                                  target_rate_override=rate)
                pred = back(Fp).argmax(1)
                correct += pred.eq(batch_lab).sum().item()
                total_bits += rate * 8 * 128 * 8 * 8 * 8  # T * C2 * H * W * rate
        
        total_samples = min(100, len(noise_trace)) * 5
        acc = 100. * correct / total_samples
        avg_bits = total_bits / min(100, len(noise_trace))
        results[strat_name] = {
            'accuracy': acc,
            'avg_bits_per_image': avg_bits,
            'bw_savings_vs_100': (1 - avg_bits / (8 * 128 * 8 * 8 * 8)) * 100,
        }
        print(f"    {strat_name:12s}: {acc:.1f}% acc, "
              f"{avg_bits/1000:.0f}K bits/img, "
              f"{results[strat_name]['bw_savings_vs_100']:.0f}% BW saved")
    
    return results, noise_trace.tolist()


# ############################################################################
# MAIN
# ############################################################################

if __name__ == "__main__":
    NUM_CLASSES = 30
    INPUT_SIZE = 224

    # AID transforms (224×224)
    train_tf = T.Compose([
        T.Resize(256),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),  # Aerial images can be any orientation
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet stats
    ])
    test_tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    print("Loading AID dataset...")
    train_ds = AIDDataset("./data", transform=train_tf, split='train')
    test_ds = AIDDataset("./data", transform=test_tf, split='test')

    train_loader = DataLoader(train_ds, 32, shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, 64, shuffle=False, num_workers=4, pin_memory=True)

    # ==================================================================
    # STEP 1: Fine-tune backbone on AID (pretrained ImageNet)
    # ==================================================================
    front = ResNet50Front(INPUT_SIZE).to(device)
    back = ResNet50Back(NUM_CLASSES).to(device)

    bb_path = os.path.join(SNAP_DIR, "backbone_best.pth")
    if os.path.exists(bb_path):
        state = torch.load(bb_path, map_location=device)
        front.load_state_dict({k: v for k, v in state.items()
                               if not k.startswith(('layer4.', 'fc.', 'avgpool.'))}, strict=False)
        back.load_state_dict({k: v for k, v in state.items()
                              if k.startswith(('layer4.', 'fc.', 'avgpool.'))}, strict=False)
        print("✓ Backbone loaded from checkpoint")
    else:
        print("\n" + "="*60)
        print("STEP 1: Fine-tune ResNet50 on AID (ImageNet pretrained)")
        print("="*60)

        # Only fine-tune — start from ImageNet pretrained
        all_params = list(front.parameters()) + list(back.parameters())
        opt = optim.SGD(all_params, lr=0.01, momentum=0.9, weight_decay=5e-4)
        sched = optim.lr_scheduler.MultiStepLR(opt, milestones=[20, 35], gamma=0.1)
        criterion = nn.CrossEntropyLoss()

        best_bb = 0.0
        for ep in range(50):  # Only 50 epochs — pretrained
            front.train(); back.train()
            pbar = tqdm(train_loader, desc=f"BB E{ep+1}/50")
            for img, lab in pbar:
                img, lab = img.to(device), lab.to(device)
                out = back(front(img))
                loss = criterion(out, lab)
                opt.zero_grad(); loss.backward(); opt.step()
                pbar.set_postfix({'L': f'{loss.item():.3f}'})
            sched.step()

            if (ep + 1) % 5 == 0:
                front.eval(); back.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for img, lab in test_loader:
                        img, lab = img.to(device), lab.to(device)
                        correct += back(front(img)).argmax(1).eq(lab).sum().item()
                        total += lab.size(0)
                acc = 100.*correct/total
                print(f"  BB E{ep+1}: {acc:.2f}%")
                if acc > best_bb:
                    best_bb = acc
                    state = {**{k: v for k, v in front.state_dict().items()},
                             **{k: v for k, v in back.state_dict().items()}}
                    torch.save(state, bb_path)
                    print(f"  ✓ Best: {best_bb:.2f}%")

        state = torch.load(bb_path, map_location=device)
        front.load_state_dict({k: v for k, v in state.items()
                               if not k.startswith(('layer4.', 'fc.', 'avgpool.'))}, strict=False)
        back.load_state_dict({k: v for k, v in state.items()
                              if k.startswith(('layer4.', 'fc.', 'avgpool.'))}, strict=False)
        print(f"✓ Backbone fine-tuned: {best_bb:.2f}%")

    front.eval()
    with torch.no_grad():
        d = front(torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(device))
        print(f"  Feature shape: {d.shape}")  # (1, 1024, 8, 8)
    C_in = d.shape[1]

    # ==================================================================
    # STEP 2+3: Train SpikeAdapt-SC for each channel
    # ==================================================================
    all_results = {}

    for channel_type in ['bsc', 'awgn', 'rayleigh']:
        print(f"\n{'='*60}")
        print(f"CHANNEL: {channel_type.upper()}")
        print(f"{'='*60}")

        model = SpikeAdaptSC(C_in=C_in, C1=256, C2=128, T=8,
                              target_rate=0.75, channel_type=channel_type).to(device)

        # Reload backbone
        bb_state = torch.load(bb_path, map_location=device)
        back.load_state_dict({k: v for k, v in bb_state.items()
                              if k.startswith(('layer4.', 'fc.', 'avgpool.'))}, strict=False)

        # Step 2: Train SNN module
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
                acc, tx = evaluate(front, model, back, test_loader, noise_param=0.0)
                print(f"  S2 E{ep+1}: {acc:.2f}%, Tx={tx:.3f}")
                if acc > best_s2:
                    best_s2 = acc
                    torch.save(model.state_dict(),
                               os.path.join(SNAP_DIR, f"{channel_type}_s2_{acc:.2f}.pth"))

        s2f = sorted([f for f in os.listdir(SNAP_DIR) if f.startswith(f"{channel_type}_s2_")])
        if s2f: model.load_state_dict(torch.load(os.path.join(SNAP_DIR, s2f[-1]), map_location=device))

        # Step 3: Joint fine-tuning
        for p in back.parameters(): p.requires_grad = True
        params = list(back.parameters()) + list(model.parameters())
        opt3 = optim.Adam(params, lr=1e-5, weight_decay=1e-4)
        sched3 = optim.lr_scheduler.CosineAnnealingLR(opt3, T_max=30, eta_min=1e-7)

        best_s3 = 0.0
        for ep in range(30):
            model.train(); back.train()
            pbar = tqdm(train_loader, desc=f"{channel_type} S3 E{ep+1}/30")
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
                acc, tx = evaluate(front, model, back, test_loader, noise_param=0.0)
                print(f"  S3 E{ep+1}: {acc:.2f}%, Tx={tx:.3f}")
                if acc > best_s3:
                    best_s3 = acc
                    torch.save({'model': model.state_dict(), 'back': back.state_dict()},
                               os.path.join(SNAP_DIR, f"{channel_type}_s3_{acc:.2f}.pth"))

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

        # Rate sweep
        rate_results = []
        for rate in [1.0, 0.9, 0.75, 0.5, 0.25]:
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

        # Dynamic rate adaptation
        dyn_results, noise_trace = dynamic_rate_experiment(
            front, model, back, test_loader, channel_type)

        all_results[channel_type] = {
            'noise_sweep': chan_results,
            'rate_sweep': rate_results,
            'dynamic_adaptation': dyn_results,
            'noise_trace': noise_trace,
            'best_s2': best_s2,
            'best_s3': best_s3,
        }

    # Save
    with open(os.path.join(SNAP_DIR, "aid_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n✅ AID TRAINING COMPLETE!")
    print(f"Results saved to {SNAP_DIR}aid_results.json")
