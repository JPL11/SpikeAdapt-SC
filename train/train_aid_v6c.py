#!/usr/bin/env python3
"""SpikeAdapt-SC V6c — V6 Progressive JSCC for Classification (AID/RESISC45).

Adapts V6's multi-scale spike-driven progressive architecture for task-oriented
semantic communication (classification instead of image reconstruction).

Key differences from V6 (image reconstruction):
  - Backbone: ResNet50 front-end (frozen) produces 1024×14×14 features
  - Multi-scale SNN encoder: 3 scales from 14×14 features (14, 7, 4)
  - SDSA at each scale (spike-driven attention)
  - Cross-scale scorer allocates bandwidth
  - Back-end: ResNet50 layer4+FC for classification
  - BSC channel (task-oriented = classification accuracy, not PSNR)

Usage:
  python train/train_aid_v6c.py --dataset aid
  python train/train_aid_v6c.py --dataset resisc45
"""

import os, sys, random, json, math, argparse, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ############################################################################
# DATASETS
# ############################################################################

class AIDDataset(Dataset):
    def __init__(self, root, transform=None, split='train', train_ratio=0.8, seed=42):
        self.transform = transform; self.samples = []
        aid_dir = os.path.join(root, 'AID')
        if not os.path.exists(aid_dir):
            raise FileNotFoundError(f"AID not found at {aid_dir}")
        classes = sorted([d for d in os.listdir(aid_dir) if os.path.isdir(os.path.join(aid_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        all_samples = []
        for cls in classes:
            cd = os.path.join(aid_dir, cls)
            for f in sorted(os.listdir(cd)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
                    all_samples.append((os.path.join(cd, f), self.class_to_idx[cls]))
        rng = random.Random(seed); rng.shuffle(all_samples)
        n = int(len(all_samples) * train_ratio)
        self.samples = all_samples[:n] if split == 'train' else all_samples[n:]
        self.num_classes = len(classes)
        print(f"  {split}: {len(self.samples)} images, {self.num_classes} classes")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, l = self.samples[idx]; img = Image.open(p).convert('RGB')
        return self.transform(img) if self.transform else img, l


class RESISC45Dataset(Dataset):
    def __init__(self, root, transform=None, split='train', train_ratio=0.8, seed=42):
        self.transform = transform; self.samples = []
        rdir = os.path.join(root, 'NWPU-RESISC45')
        if not os.path.exists(rdir):
            raise FileNotFoundError(f"RESISC45 not found at {rdir}")
        classes = sorted([d for d in os.listdir(rdir) if os.path.isdir(os.path.join(rdir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        all_samples = []
        for cls in classes:
            cd = os.path.join(rdir, cls)
            for f in sorted(os.listdir(cd)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_samples.append((os.path.join(cd, f), self.class_to_idx[cls]))
        rng = random.Random(seed); rng.shuffle(all_samples)
        n = int(len(all_samples) * train_ratio)
        self.samples = all_samples[:n] if split == 'train' else all_samples[n:]
        self.num_classes = len(classes)
        print(f"  {split}: {len(self.samples)} images, {self.num_classes} classes")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, l = self.samples[idx]; img = Image.open(p).convert('RGB')
        return self.transform(img) if self.transform else img, l


# ############################################################################
# BACKBONE
# ############################################################################

class ResNet50Front(nn.Module):
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        self.conv1 = r.conv1; self.bn1 = r.bn1; self.relu = r.relu
        self.maxpool = r.maxpool
        self.layer1 = r.layer1; self.layer2 = r.layer2; self.layer3 = r.layer3
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.layer3(self.layer2(self.layer1(x)))  # (B, 1024, 14, 14)

class ResNet50Back(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        r = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        self.layer4 = r.layer4; self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)
    def forward(self, x): return self.fc(torch.flatten(self.avgpool(self.layer4(x)), 1))


# ############################################################################
# SNN MODULES (from V4/V5, proven)
# ############################################################################

class SpikeFunction_Learnable(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane, threshold, slope):
        if isinstance(threshold, (int, float)):
            threshold = torch.tensor(float(threshold), device=membrane.device)
        ctx.save_for_backward(membrane, threshold, slope)
        ctx.th_needs_grad = threshold.requires_grad
        ctx.slope_needs_grad = slope.requires_grad
        return (membrane > threshold).float()
    @staticmethod
    def backward(ctx, grad_output):
        membrane, threshold, slope = ctx.saved_tensors
        s = slope.clamp(min=1.0, max=100.0)
        sig = torch.sigmoid(s * (membrane - threshold))
        sg = sig * (1 - sig) * s
        grad_mem = grad_output * sg
        grad_th = -(grad_output * sg).sum() if ctx.th_needs_grad else None
        grad_slope = (grad_output * sig * (1 - sig) * (membrane - threshold)).sum() if ctx.slope_needs_grad else None
        return grad_mem, grad_th, grad_slope

class LIFNeuron(nn.Module):
    def __init__(self, C, th=1.0):
        super().__init__()
        self.threshold = th
        self.beta_raw = nn.Parameter(torch.ones(1, C, 1, 1) * 2.2)
        self.slope = nn.Parameter(torch.tensor(10.0))
    def forward(self, x, mem=None):
        if mem is None: mem = torch.zeros_like(x)
        mem = torch.sigmoid(self.beta_raw) * mem + x
        sp = SpikeFunction_Learnable.apply(mem, self.threshold, self.slope)
        return sp, mem - sp * self.threshold

class MPBN(nn.Module):
    def __init__(self, C, T):
        super().__init__()
        self.bns = nn.ModuleList([nn.BatchNorm2d(C, affine=True) for _ in range(T)])
        self.T = T
    def forward(self, x, t):
        return self.bns[min(t, self.T - 1)](x)


# ############################################################################
# SDSA for Classification (adapted from V6)
# ############################################################################

class SpikeDrivenSelfAttentionCls(nn.Module):
    """SDSA adapted for classification features (14×14 spatial)."""
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.dim = dim; self.n_heads = n_heads; self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.k_conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.v_conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.q_bn = nn.BatchNorm2d(dim)
        self.k_bn = nn.BatchNorm2d(dim)
        self.v_bn = nn.BatchNorm2d(dim)

    @staticmethod
    def _ternary_ste(x):
        """Ternary quantization with STE: {-1, 0, 1}, gradients pass through."""
        return x + (torch.sign(x) - x).detach()

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q_bn(self.q_conv(x))
        k = self.k_bn(self.k_conv(x))
        v = self.v_bn(self.v_conv(x))
        # Spike-driven: ternary quantize Q, K with STE
        q_spike = self._ternary_ste(q)
        k_spike = self._ternary_ste(k)
        q_spike = q_spike.view(B, self.n_heads, self.head_dim, H*W)
        k_spike = k_spike.view(B, self.n_heads, self.head_dim, H*W)
        v = v.view(B, self.n_heads, self.head_dim, H*W)
        attn = torch.einsum('bhdn,bhen->bhde', q_spike, k_spike) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhde,bhen->bhdn', attn, v).reshape(B, C, H, W)
        return self.proj(out) + x


# ############################################################################
# MULTI-SCALE SNN ENCODER FOR CLASSIFICATION
# ############################################################################

class MultiScaleSNNEncoderCls(nn.Module):
    """Multi-scale SNN encoder for 1024×14×14 features.
    
    Scale 1: 14×14 → C1 channels (fine detail)
    Scale 2: 7×7  → C2 channels (mid-level)
    Scale 3: 4×4  → C3 channels (semantic)
    """
    def __init__(self, C_in=1024, C_tx=(24, 36, 48), T=8):
        super().__init__()
        self.T = T
        self.n_scales = len(C_tx)
        
        # Scale 1: 14×14
        self.conv1 = nn.Conv2d(C_in, C_tx[0], 3, 1, 1)
        self.mpbn1 = MPBN(C_tx[0], T)
        self.lif1 = LIFNeuron(C_tx[0])
        self.sdsa1 = SpikeDrivenSelfAttentionCls(C_in, n_heads=8)
        
        # Scale 2: 7×7 (pool first)
        self.pool2 = nn.AdaptiveAvgPool2d(7)
        self.conv2 = nn.Conv2d(C_in, C_tx[1], 3, 1, 1)
        self.mpbn2 = MPBN(C_tx[1], T)
        self.lif2 = LIFNeuron(C_tx[1])
        self.sdsa2 = SpikeDrivenSelfAttentionCls(C_in, n_heads=8)
        
        # Scale 3: 4×4 (pool)
        self.pool3 = nn.AdaptiveAvgPool2d(4)
        self.conv3 = nn.Conv2d(C_in, C_tx[2], 3, 1, 1)
        self.mpbn3 = MPBN(C_tx[2], T)
        self.lif3 = LIFNeuron(C_tx[2])
        self.sdsa3 = SpikeDrivenSelfAttentionCls(C_in, n_heads=8)

    def forward(self, feat):
        """feat: (B, 1024, 14, 14) → list of 3 spike streams"""
        f1 = self.sdsa1(feat)
        f2 = self.sdsa2(self.pool2(feat))
        f3 = self.sdsa3(self.pool3(feat))
        
        spikes = [[], [], []]
        m1, m2, m3 = None, None, None
        for t in range(self.T):
            x1 = self.mpbn1(self.conv1(f1), t); s1, m1 = self.lif1(x1, m1); spikes[0].append(s1)
            x2 = self.mpbn2(self.conv2(f2), t); s2, m2 = self.lif2(x2, m2); spikes[1].append(s2)
            x3 = self.mpbn3(self.conv3(f3), t); s3, m3 = self.lif3(x3, m3); spikes[2].append(s3)
        return spikes


# ############################################################################
# CHANNEL + SCORER + MASKER
# ############################################################################

class BSC_Channel(nn.Module):
    def forward(self, x, ber):
        if ber <= 0: return x
        return ((x + (torch.rand_like(x.float()) < ber).float()) % 2)

class CrossScaleScorerCls(nn.Module):
    """Score importance across scales for classification."""
    def __init__(self, C_tx=(24, 36, 48)):
        super().__init__()
        self.scale_scorers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c + 1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, True),
                nn.Conv2d(32, 1, 1), nn.Sigmoid()
            ) for c in C_tx
        ])
    def forward(self, multi_spikes, noise_param=0.0):
        importance_maps = []
        for i, spikes in enumerate(multi_spikes):
            mean_rate = torch.stack(spikes, dim=0).mean(0)
            B = mean_rate.size(0)
            noise_map = torch.full((B, 1, mean_rate.size(2), mean_rate.size(3)),
                                   noise_param, device=mean_rate.device)
            imp = self.scale_scorers[i](torch.cat([mean_rate, noise_map], dim=1))
            importance_maps.append(imp)
        return importance_maps

class MultiScaleMaskerCls(nn.Module):
    def __init__(self, target_rate=0.75, temperature=0.5, min_coarse_rate=0.90):
        super().__init__()
        self.target_rate = target_rate; self.temperature = temperature
        self.min_coarse_rate = min_coarse_rate  # protect coarsest scale
    def forward(self, multi_spikes, importance_maps, training=True, target_override=None):
        target = target_override if target_override is not None else self.target_rate
        n_scales = len(multi_spikes)
        masked = []; rates = []
        for si, (spikes, imp) in enumerate(zip(multi_spikes, importance_maps)):
            B, _, H, W = imp.shape
            # Coarse-scale protection: last scale (most semantic) gets high min rate
            is_coarse = (si == n_scales - 1)
            if training:
                logits = torch.log(imp / (1 - imp + 1e-7) + 1e-7)
                if is_coarse:
                    # Bias coarse scale logits to stay high
                    logits = logits + 2.0  # ~0.88 baseline sigmoid
                u = torch.rand_like(logits).clamp(1e-7, 1-1e-7)
                soft = torch.sigmoid((logits - torch.log(-torch.log(u))) / self.temperature)
                hard = (soft > 0.5).float(); mask = hard + (soft - soft.detach())
            else:
                eff_rate = max(target, self.min_coarse_rate) if is_coarse else target
                k = max(1, int(eff_rate * H * W))
                flat = imp.view(B, -1); _, idx = flat.topk(k, dim=1)
                mask = torch.zeros_like(flat); mask.scatter_(1, idx, 1.0); mask = mask.view(B, 1, H, W)
            masked.append([s * mask for s in spikes])
            rates.append(mask.mean().item())
        return masked, rates


# ############################################################################
# MULTI-SCALE SNN DECODER FOR CLASSIFICATION
# ############################################################################

class MultiScaleSNNDecoderCls(nn.Module):
    """Decode multi-scale spikes back to 1024×14×14 features."""
    def __init__(self, C_out=1024, C_tx=(24, 36, 48), T=8):
        super().__init__()
        self.T = T
        # Scale decoders
        self.dec1 = nn.Sequential(nn.Conv2d(C_tx[0], 256, 3, 1, 1), nn.BatchNorm2d(256))
        self.dec2 = nn.Sequential(nn.Conv2d(C_tx[1], 256, 3, 1, 1), nn.BatchNorm2d(256))
        self.dec3 = nn.Sequential(nn.Conv2d(C_tx[2], 256, 3, 1, 1), nn.BatchNorm2d(256))
        
        self.lif_d1 = LIFNeuron(256); self.lif_d2 = LIFNeuron(256); self.lif_d3 = LIFNeuron(256)
        
        # Fusion: 768 → 1024
        self.fuse = nn.Sequential(
            nn.Conv2d(768, C_out, 1), nn.BatchNorm2d(C_out), nn.LeakyReLU(0.2, True))
        
        self.converter = nn.Linear(2*T, 2*T)

    def _decode_scale(self, spikes, dec, lif, target_size):
        mem = None; feats = []; mems = []
        for t in range(min(len(spikes), self.T)):
            x = F.relu(dec(spikes[t]))
            sp, mem = lif(x, mem)
            feats.append(sp); mems.append(mem.clone())
        while len(feats) < self.T:
            feats.append(torch.zeros_like(feats[0]))
            mems.append(torch.zeros_like(mems[0]))
        # Temporal integration
        il = []
        for t in range(self.T): il.append(feats[t]); il.append(mems[t])
        x = torch.stack(il, 1).permute(0, 2, 3, 4, 1)
        out = (x * torch.sigmoid(self.converter(x))).sum(-1)
        if out.shape[2:] != target_size:
            out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
        return out

    def forward(self, recv_multi):
        target = (14, 14)
        d1 = self._decode_scale(recv_multi[0], self.dec1, self.lif_d1, target)
        d2 = self._decode_scale(recv_multi[1], self.dec2, self.lif_d2, target)
        d3 = self._decode_scale(recv_multi[2], self.dec3, self.lif_d3, target)
        return self.fuse(torch.cat([d1, d2, d3], dim=1))


# ############################################################################
# FULL V6c MODEL
# ############################################################################

class SpikeAdaptSC_V6c(nn.Module):
    """V6c: Multi-scale progressive JSCC for classification."""
    def __init__(self, C_in=1024, C_tx=(24, 36, 48), T=8, target_rate=0.75):
        super().__init__()
        self.T = T; self.C_tx = C_tx
        self.snn_encoder = MultiScaleSNNEncoderCls(C_in, C_tx, T)
        self.scorer = CrossScaleScorerCls(C_tx)
        self.masker = MultiScaleMaskerCls(target_rate, 0.5)
        self.channel = BSC_Channel()
        self.snn_decoder = MultiScaleSNNDecoderCls(C_in, C_tx, T)

    def forward(self, feat, noise_param=0.0, target_rate_override=None):
        multi_spikes = self.snn_encoder(feat)
        imp_maps = self.scorer(multi_spikes, noise_param)
        masked, rates = self.masker(multi_spikes, imp_maps, self.training, target_rate_override)
        recv = []
        for scale_spikes in masked:
            recv.append([self.channel(s, noise_param) for s in scale_spikes])
        Fp = self.snn_decoder(recv)
        return Fp, {
            'tx_rate': sum(rates) / len(rates),
            'importance': imp_maps,
            'multi_spikes': multi_spikes,
        }


# ############################################################################
# TRAINING
# ############################################################################

def sample_noise():
    return random.uniform(0.15, 0.4) if random.random() < 0.5 else random.uniform(0, 0.15)

def sample_noise_heavy():
    """Aggressive noise for S2.5 hardening stage — always high BER."""
    return random.uniform(0.15, 0.40)

def evaluate(front, model, back, loader, noise_param=0.0):
    front.eval(); model.eval(); back.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            Fp, _ = model(front(imgs), noise_param)
            correct += back(Fp).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='aid', choices=['aid', 'resisc45'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs-s2', type=int, default=60)
    parser.add_argument('--epochs-s25', type=int, default=30)
    parser.add_argument('--epochs-s3', type=int, default=40)
    parser.add_argument('--target-rate', type=float, default=0.75)
    parser.add_argument('--T', type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    print(f"Device: {device} | Dataset: {args.dataset} | Seed: {args.seed}")

    # Data
    tf_train = T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
        T.RandomRotation(15), T.ColorJitter(0.2,0.2,0.2,0.1),
        T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))])
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224),
        T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))])

    if args.dataset == 'aid':
        train_ds = AIDDataset('./data', tf_train, 'train', seed=args.seed)
        test_ds = AIDDataset('./data', tf_test, 'test', seed=args.seed)
        NC = train_ds.num_classes
    else:
        train_ds = RESISC45Dataset('./data', tf_train, 'train', seed=args.seed)
        test_ds = RESISC45Dataset('./data', tf_test, 'test', seed=args.seed)
        NC = train_ds.num_classes

    train_loader = DataLoader(train_ds, 32, True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, 64, False, num_workers=4)

    # Backbone
    front = ResNet50Front().to(device)
    back = ResNet50Back(NC).to(device)

    # Dataset-specific backbone paths
    if args.dataset == 'aid':
        bb_path = "./snapshots_aid/backbone_best.pth"
    else:
        bb_path = f"./snapshots_resisc45/backbone_best_seed{args.seed}.pth"

    if os.path.exists(bb_path):
        state = torch.load(bb_path, map_location=device, weights_only=False)
        front.load_state_dict({k:v for k,v in state.items()
            if not k.startswith(('layer4.','fc.','avgpool.','spatial_pool.'))}, strict=False)
        back_state = back.state_dict()
        loaded = {k:v for k,v in state.items()
                  if k in back_state and back_state[k].shape == v.shape}
        back.load_state_dict(loaded, strict=False)
        print(f"✓ Backbone loaded from {bb_path}")
    else:
        # Train backbone from scratch
        print(f"\n{'='*60}")
        print(f"S0: Fine-tune ResNet50 on {args.dataset.upper()}")
        print(f"{'='*60}")
        os.makedirs(os.path.dirname(bb_path) if os.path.dirname(bb_path) else '.', exist_ok=True)
        all_p = list(front.parameters()) + list(back.parameters())
        bb_opt = optim.SGD(all_p, lr=0.01, momentum=0.9, weight_decay=5e-4)
        bb_sched = optim.lr_scheduler.MultiStepLR(bb_opt, [20, 35], 0.1)
        ce = nn.CrossEntropyLoss(); best_bb = 0.0
        for ep in range(50):
            front.train(); back.train()
            for img, lab in tqdm(train_loader, desc=f"BB E{ep+1}/50"):
                img, lab = img.to(device), lab.to(device)
                loss = ce(back(front(img)), lab)
                bb_opt.zero_grad(); loss.backward(); bb_opt.step()
            bb_sched.step()
            if (ep+1) % 5 == 0:
                front.eval(); back.eval(); c, t = 0, 0
                with torch.no_grad():
                    for img, lab in test_loader:
                        img, lab = img.to(device), lab.to(device)
                        c += back(front(img)).argmax(1).eq(lab).sum().item(); t += lab.size(0)
                acc = 100.*c/t; print(f"  BB E{ep+1}: {acc:.2f}%")
                if acc > best_bb:
                    best_bb = acc
                    torch.save({**front.state_dict(), **back.state_dict()}, bb_path)
                    print(f"  ✓ Best backbone: {best_bb:.2f}%")
        # Reload best
        state = torch.load(bb_path, map_location=device, weights_only=False)
        front.load_state_dict({k:v for k,v in state.items()
            if not k.startswith(('layer4.','fc.','avgpool.','spatial_pool.'))}, strict=False)
        back_state = back.state_dict()
        loaded = {k:v for k,v in state.items()
                  if k in back_state and back_state[k].shape == v.shape}
        back.load_state_dict(loaded, strict=False)
        print(f"✓ Backbone trained: {best_bb:.2f}%")

    front.eval()

    # V6c model
    model = SpikeAdaptSC_V6c(C_in=1024, C_tx=(24, 36, 48), T=args.T, target_rate=args.target_rate).to(device)
    print(f"V6c params: {sum(p.numel() for p in model.parameters()):,}")

    SNAP_DIR = f"./snapshots_v6c_{args.dataset}_seed{args.seed}/"
    os.makedirs(SNAP_DIR, exist_ok=True)
    criterion = nn.CrossEntropyLoss()

    # ---- S2: Train SNN (backbone frozen) ----
    print(f"\n{'='*60}\nS2: Train V6c SNN ({args.epochs_s2} epochs)\n{'='*60}")
    for p in front.parameters(): p.requires_grad = False
    for p in back.parameters(): p.requires_grad = False
    opt = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs_s2, eta_min=1e-6)
    best_s2 = 0.0
    for ep in range(args.epochs_s2):
        model.train()
        pbar = tqdm(train_loader, desc=f"S2 E{ep+1}/{args.epochs_s2}")
        for img, lab in pbar:
            img, lab = img.to(device), lab.to(device)
            noise = sample_noise()
            with torch.no_grad(): feat = front(img)
            Fp, stats = model(feat, noise)
            loss = criterion(back(Fp), lab) + 2.0 * (stats['tx_rate'] - args.target_rate)**2
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            pbar.set_postfix({'L': f'{loss.item():.3f}', 'tx': f"{stats['tx_rate']:.2f}"})
        sched.step()
        if (ep+1) % 10 == 0:
            acc = evaluate(front, model, back, test_loader)
            print(f"  S2 E{ep+1}: {acc:.2f}%")
            if acc > best_s2:
                best_s2 = acc
                torch.save(model.state_dict(), os.path.join(SNAP_DIR, f"v6c_s2_{acc:.2f}.pth"))
                print(f"  ✓ Best S2: {best_s2:.2f}%")

    s2f = sorted(glob.glob(os.path.join(SNAP_DIR, "v6c_s2_*.pth")))
    if s2f: model.load_state_dict(torch.load(s2f[-1], map_location=device, weights_only=False))

    # ---- S2.5: Heavy noise hardening (coarse-scale protection) ----
    print(f"\n{'='*60}\nS2.5: Heavy Noise Hardening ({args.epochs_s25} epochs)\n{'='*60}")
    opt25 = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    sched25 = optim.lr_scheduler.CosineAnnealingLR(opt25, T_max=args.epochs_s25, eta_min=1e-6)
    best_s25 = 0.0
    for ep in range(args.epochs_s25):
        model.train()
        pbar = tqdm(train_loader, desc=f"S2.5 E{ep+1}/{args.epochs_s25}")
        for img, lab in pbar:
            img, lab = img.to(device), lab.to(device)
            noise = sample_noise_heavy()  # always high BER
            with torch.no_grad(): feat = front(img)
            Fp, stats = model(feat, noise)
            loss = criterion(back(Fp), lab) + 2.0 * (stats['tx_rate'] - args.target_rate)**2
            opt25.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt25.step()
            pbar.set_postfix({'L': f'{loss.item():.3f}', 'tx': f"{stats['tx_rate']:.2f}", 'ber': f'{noise:.2f}'})
        sched25.step()
        if (ep+1) % 10 == 0:
            acc_c = evaluate(front, model, back, test_loader, 0.0)
            acc_n = evaluate(front, model, back, test_loader, 0.30)
            print(f"  S2.5 E{ep+1}: clean={acc_c:.2f}% ber30={acc_n:.2f}%")
            if acc_n > best_s25:
                best_s25 = acc_n
                torch.save(model.state_dict(), os.path.join(SNAP_DIR, f"v6c_s25_{acc_n:.2f}.pth"))
                print(f"  ✓ Best S2.5 (BER=0.30): {best_s25:.2f}%")

    s25f = sorted(glob.glob(os.path.join(SNAP_DIR, "v6c_s25_*.pth")))
    if s25f: model.load_state_dict(torch.load(s25f[-1], map_location=device, weights_only=False))

    # ---- S3: Joint fine-tuning ----
    print(f"\n{'='*60}\nS3: Joint FT ({args.epochs_s3} epochs)\n{'='*60}")
    for p in back.parameters(): p.requires_grad = True
    params = list(model.parameters()) + list(back.parameters())
    opt3 = optim.Adam(params, lr=1e-5, weight_decay=1e-4)
    sched3 = optim.lr_scheduler.CosineAnnealingLR(opt3, T_max=args.epochs_s3, eta_min=1e-7)
    best_s3 = 0.0
    for ep in range(args.epochs_s3):
        model.train(); back.train()
        pbar = tqdm(train_loader, desc=f"S3 E{ep+1}/{args.epochs_s3}")
        opt3.zero_grad()
        for step, (img, lab) in enumerate(pbar):
            img, lab = img.to(device), lab.to(device)
            noise = sample_noise()
            with torch.no_grad(): feat = front(img)
            Fp, stats = model(feat, noise)
            loss = (criterion(back(Fp), lab) + 2.0 * (stats['tx_rate'] - args.target_rate)**2) / 2
            loss.backward()
            if (step+1) % 2 == 0:
                nn.utils.clip_grad_norm_(params, 1.0); opt3.step(); opt3.zero_grad()
            pbar.set_postfix({'L': f'{loss.item():.3f}', 'tx': f"{stats['tx_rate']:.2f}"})
        sched3.step()
        if (ep+1) % 5 == 0:
            acc = evaluate(front, model, back, test_loader)
            print(f"  S3 E{ep+1}: {acc:.2f}%")
            if acc > best_s3:
                best_s3 = acc
                torch.save({'model': model.state_dict(), 'back': back.state_dict()},
                           os.path.join(SNAP_DIR, f"v6c_s3_{acc:.2f}.pth"))
                print(f"  ✓ Best S3: {best_s3:.2f}%")

    # Load best
    s3f = sorted(glob.glob(os.path.join(SNAP_DIR, "v6c_s3_*.pth")))
    if s3f:
        ck = torch.load(s3f[-1], map_location=device, weights_only=False)
        model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])

    # ---- Evaluation ----
    print(f"\n{'='*60}\nFINAL EVALUATION — V6c [{args.dataset}]\n{'='*60}")
    acc_clean = evaluate(front, model, back, test_loader, 0.0)
    print(f"  Clean: {acc_clean:.2f}%")
    ber_results = {'clean': acc_clean}
    for ber in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        acc_n = evaluate(front, model, back, test_loader, ber)
        ber_results[f'ber_{ber:.2f}'] = acc_n
        print(f"  BER={ber:.2f}: {acc_n:.2f}%")

    # Reference V4-A and original
    v4a = {'clean': 96.45, 'ber_0.30': 95.85}
    results = {
        'model': 'SpikeAdapt-SC_V6c', 'dataset': args.dataset, 'seed': args.seed,
        'best_s2': best_s2, 'best_s3': best_s3,
        'ber_results': ber_results,
        'delta_v4a': best_s3 - v4a['clean'],
    }
    with open(os.path.join(SNAP_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ V6c [{args.dataset}]: best={best_s3:.2f}% (vs V4-A: {best_s3-v4a['clean']:+.2f}pp)")

if __name__ == '__main__':
    main()
