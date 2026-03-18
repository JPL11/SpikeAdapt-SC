"""Retrain SpikeAdapt-SC scorer with entropy regularizer.

Key changes from original train_aid.py:
1. Score entropy loss: -H(scores) pushes scores away from uniform → more discriminative
2. Margin loss: encourages gap between kept and dropped block scores
3. Gini tracking: monitors score concentration during training

Only retrains BSC channel (BSC only) for speed.
Loads existing backbone, only trains SNN module + scorer.
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
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

SNAP_DIR = "./snapshots_aid/"
os.makedirs(SNAP_DIR, exist_ok=True)

# ############################################################################
# AID DATASET (simplified — assumes data exists)
# ############################################################################

class AIDDataset(Dataset):
    def __init__(self, root, transform=None, split='train', train_ratio=0.8):
        self.root = root; self.transform = transform; self.samples = []
        aid_dir = os.path.join(root, 'AID')
        classes = sorted([d for d in os.listdir(aid_dir) if os.path.isdir(os.path.join(aid_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        all_samples = []
        for cls in classes:
            cls_dir = os.path.join(aid_dir, cls)
            for f in sorted(os.listdir(cls_dir)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
                    all_samples.append((os.path.join(cls_dir, f), self.class_to_idx[cls]))
        random.seed(42); random.shuffle(all_samples)
        n = int(len(all_samples) * train_ratio)
        self.samples = all_samples[:n] if split == 'train' else all_samples[n:]
        print(f"  AID {split}: {len(self.samples)} images, {len(classes)} classes")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, label


# ############################################################################
# CHANNEL
# ############################################################################

class BSC_Channel(nn.Module):
    def forward(self, x, ber):
        if ber <= 0: return x
        return ((x + (torch.rand_like(x.float()) < ber).float()) % 2)


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
        scale = 10.0; sig = torch.sigmoid(scale * (membrane - threshold))
        sg = sig * (1 - sig) * scale
        return grad_output * sg, -(grad_output * sg).sum() if ctx.th_needs_grad else None

class IFNeuron(nn.Module):
    def __init__(self, threshold=1.0):
        super().__init__(); self.threshold = threshold
    def forward(self, x, mem=None):
        if mem is None: mem = torch.zeros_like(x)
        mem = mem + x; sp = SpikeFunction.apply(mem, self.threshold)
        return sp, mem - sp * self.threshold

class IHFNeuron(nn.Module):
    def __init__(self, threshold=1.0):
        super().__init__(); self.threshold = nn.Parameter(torch.tensor(float(threshold)))
    def forward(self, x, mem=None):
        if mem is None: mem = torch.zeros_like(x)
        mem = mem + x; sp = SpikeFunction.apply(mem, self.threshold)
        return sp, mem - sp * self.threshold


# ############################################################################
# BACKBONE
# ############################################################################

class ResNet50Front(nn.Module):
    def __init__(self, input_size=224):
        super().__init__()
        r = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        self.conv1 = r.conv1; self.bn1 = r.bn1; self.relu = r.relu
        self.maxpool = r.maxpool
        self.layer1 = r.layer1; self.layer2 = r.layer2; self.layer3 = r.layer3
        self.spatial_pool = nn.AdaptiveAvgPool2d(8)
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
# SPIKEADAPT-SC WITH ENHANCED SCORER
# ############################################################################

class EnhancedImportanceScorer(nn.Module):
    """Scorer that also sees the raw feature statistics, not just spike averages.
    
    Architecture: concat(spike_mean, feat_var) → Conv1x1 → ReLU → Conv1x1 → Sigmoid
    This gives the scorer MORE information to differentiate blocks.
    """
    def __init__(self, C_spike=128, C_feat=1024, hidden=64):
        super().__init__()
        # Feature-based branch: conv to reduce feat channels
        self.feat_reduce = nn.Sequential(
            nn.Conv2d(C_feat, hidden, 1), nn.ReLU(True)
        )
        # Spike-based branch
        self.spike_reduce = nn.Sequential(
            nn.Conv2d(C_spike, hidden, 1), nn.ReLU(True)
        )
        # Combined scorer
        self.scorer = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, 1), nn.ReLU(True),
            nn.Conv2d(hidden, 1, 1), nn.Sigmoid()
        )
    
    def forward(self, all_S2, feat=None):
        spike_avg = torch.stack(all_S2, dim=0).mean(dim=0)  # (B, 128, 8, 8)
        s_branch = self.spike_reduce(spike_avg)
        if feat is not None:
            f_branch = self.feat_reduce(feat)
            combined = torch.cat([s_branch, f_branch], dim=1)
        else:
            combined = torch.cat([s_branch, s_branch], dim=1)  # fallback
        return self.scorer(combined).squeeze(1)

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
    def __init__(self, C_in=1024, C1=256, C2=128):
        super().__init__()
        self.conv1 = nn.Conv2d(C_in, C1, 3, 1, 1); self.bn1 = nn.BatchNorm2d(C1); self.if1 = IFNeuron()
        self.conv2 = nn.Conv2d(C1, C2, 3, 1, 1); self.bn2 = nn.BatchNorm2d(C2); self.if2 = IFNeuron()
    def forward(self, F, m1=None, m2=None):
        s1, m1 = self.if1(self.bn1(self.conv1(F)), m1)
        s2, m2 = self.if2(self.bn2(self.conv2(s1)), m2)
        return s1, s2, m1, m2

class Decoder(nn.Module):
    def __init__(self, C_out=1024, C1=256, C2=128, T=8):
        super().__init__()
        self.T = T
        self.conv3 = nn.Conv2d(C2, C1, 3, 1, 1); self.bn3 = nn.BatchNorm2d(C1); self.if3 = IFNeuron()
        self.conv4 = nn.Conv2d(C1, C_out, 3, 1, 1); self.bn4 = nn.BatchNorm2d(C_out); self.ihf = IHFNeuron()
        self.converter_fc = nn.Linear(2 * T, 2 * T)
    def forward(self, recv_all, mask):
        m3, m4 = None, None; Fs, Fm = [], []
        for t in range(self.T):
            s3, m3 = self.if3(self.bn3(self.conv3(recv_all[t] * mask)), m3)
            sp, m4 = self.ihf(self.bn4(self.conv4(s3)), m4)
            Fs.append(sp); Fm.append(m4.clone())
        il = []
        for t in range(self.T): il.append(Fs[t]); il.append(Fm[t])
        x = torch.stack(il, 1).permute(0, 2, 3, 4, 1)
        return (x * torch.sigmoid(self.converter_fc(x))).sum(-1)

class SpikeAdaptSC_v2(nn.Module):
    """SpikeAdapt-SC with enhanced scorer that sees raw features."""
    def __init__(self, C_in=1024, C1=256, C2=128, T=8, target_rate=0.75):
        super().__init__()
        self.T = T
        self.encoder = Encoder(C_in, C1, C2)
        self.importance_scorer = EnhancedImportanceScorer(C_spike=C2, C_feat=C_in, hidden=64)
        self.block_mask = LearnedBlockMask(target_rate, 0.5)
        self.decoder = Decoder(C_in, C1, C2, T)
        self.channel = BSC_Channel()

    def forward(self, feat, noise_param=0.0, target_rate_override=None):
        all_S1, all_S2, m1, m2 = [], [], None, None
        for t in range(self.T):
            s1, s2, m1, m2 = self.encoder(feat, m1, m2)
            all_S1.append(s1); all_S2.append(s2)

        # Enhanced scorer sees both spikes AND raw features
        importance = self.importance_scorer(all_S2, feat)

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
        return Fp, {'tx_rate': tx.item(), 'mask': mask, 'importance': importance}


# ############################################################################
# NEW LOSS FUNCTIONS
# ############################################################################

def score_entropy_loss(importance):
    """Negative entropy of importance scores — pushes toward bimodal (0/1)."""
    B, H, W = importance.shape
    flat = importance.view(B, -1)  # (B, 64)
    # Treat scores as probabilities after normalization
    p = flat / (flat.sum(1, keepdim=True) + 1e-12)
    entropy = -(p * torch.log(p + 1e-12)).sum(1).mean()
    # We want LOW entropy (scores far from uniform) → minimize entropy
    return entropy

def margin_loss(importance, target_rate=0.75):
    """Encourage gap between kept and dropped blocks."""
    B, H, W = importance.shape
    flat = importance.view(B, -1)
    k = max(1, int(target_rate * H * W))
    sorted_scores, _ = flat.sort(dim=1, descending=True)
    kept = sorted_scores[:, :k].mean(1)       # mean of top-k
    dropped = sorted_scores[:, k:].mean(1)    # mean of bottom-(64-k)
    # We want kept >> dropped → maximize gap → minimize negative gap
    margin = kept - dropped
    # Hinge: margin should be at least 0.1
    loss = F.relu(0.1 - margin).mean()
    return loss

def gini_coefficient_batch(importance):
    """Compute mean Gini coefficient for a batch — for monitoring only."""
    B = importance.shape[0]
    flat = importance.view(B, -1)
    n = flat.shape[1]
    ginis = []
    for b in range(B):
        x = flat[b].sort()[0]
        idx = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
        gini = (2 * (idx * x).sum() - (n + 1) * x.sum()) / (n * x.sum() + 1e-12)
        ginis.append(gini.item())
    return np.mean(ginis)


# ############################################################################
# NOISE SAMPLING
# ############################################################################

def sample_noise():
    """BSC BER sampling with emphasis on mid-range."""
    if random.random() < 0.5:
        return random.uniform(0.15, 0.4)  # Hard
    else:
        return random.uniform(0.0, 0.15)  # Easy


# ############################################################################
# MAIN TRAINING
# ############################################################################

if __name__ == "__main__":
    NUM_CLASSES = 30
    INPUT_SIZE = 224

    train_tf = T.Compose([
        T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
        T.RandomRotation(15), T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_tf = T.Compose([
        T.Resize(256), T.CenterCrop(224),
        T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    print("Loading AID dataset...")
    train_ds = AIDDataset("./data", transform=train_tf, split='train')
    test_ds = AIDDataset("./data", transform=test_tf, split='test')
    train_loader = DataLoader(train_ds, 32, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, 64, shuffle=False, num_workers=4, pin_memory=True)

    # Load backbone (frozen)
    front = ResNet50Front(INPUT_SIZE).to(device)
    back = ResNet50Back(NUM_CLASSES).to(device)

    bb_path = os.path.join(SNAP_DIR, "backbone_best.pth")
    assert os.path.exists(bb_path), f"Backbone not found at {bb_path}!"
    state = torch.load(bb_path, map_location=device)
    front.load_state_dict({k: v for k, v in state.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.'))}, strict=False)
    back.load_state_dict({k: v for k, v in state.items()
                          if k.startswith(('layer4.', 'fc.', 'avgpool.'))}, strict=False)
    front.eval()
    print("✓ Backbone loaded")

    # Create enhanced model
    model = SpikeAdaptSC_v2(C_in=1024, C1=256, C2=128, T=8, target_rate=0.75).to(device)

    # Initialize encoder/decoder from existing best model if available
    existing = sorted([f for f in os.listdir(SNAP_DIR) if f.startswith("bsc_s3_")])
    if existing:
        ck = torch.load(os.path.join(SNAP_DIR, existing[-1]), map_location=device)
        # Load encoder and decoder (scorer is new, won't match)
        enc_state = {k.replace('encoder.', ''): v for k, v in ck['model'].items() if 'encoder.' in k}
        dec_state = {k.replace('decoder.', ''): v for k, v in ck['model'].items() if 'decoder.' in k}
        model.encoder.load_state_dict(enc_state)
        model.decoder.load_state_dict(dec_state)
        back.load_state_dict(ck['back'])
        print(f"✓ Loaded encoder/decoder from {existing[-1]} (scorer is fresh)")

    for p in front.parameters(): p.requires_grad = False
    criterion = nn.CrossEntropyLoss()

    # Hyperparameters for the new loss terms
    LAMBDA_ENTROPY = 0.05   # Weight for entropy reduction (gentle push)
    LAMBDA_MARGIN = 0.2     # Weight for margin loss (gentle push)
    LAMBDA_RATE = 2.0       # Weight for rate regularization (same as original)

    # ==================================================================
    # STEP 2: Train SNN + enhanced scorer (backbone frozen)
    # ==================================================================
    print(f"\n{'='*60}")
    print("STEP 2: Train SNN + Enhanced Scorer (BSC, backbone frozen)")
    print(f"  λ_entropy={LAMBDA_ENTROPY}, λ_margin={LAMBDA_MARGIN}, λ_rate={LAMBDA_RATE}")
    print(f"{'='*60}")

    for p in back.parameters(): p.requires_grad = False
    opt = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60, eta_min=1e-6)

    best_s2 = 0.0
    for ep in range(60):
        model.train()
        ep_ce, ep_ent, ep_mar, ep_rate = 0, 0, 0, 0
        ep_gini = []
        pbar = tqdm(train_loader, desc=f"S2 E{ep+1}/60")
        for img, lab in pbar:
            img, lab = img.to(device), lab.to(device)
            noise = sample_noise()
            with torch.no_grad(): feat = front(img)
            Fp, stats = model(feat, noise_param=noise)

            importance = stats['importance']

            # Losses
            l_ce = criterion(back(Fp), lab)
            l_ent = score_entropy_loss(importance)
            l_mar = margin_loss(importance, 0.75)
            l_rate = (stats['tx_rate'] - 0.75) ** 2

            loss = l_ce + LAMBDA_ENTROPY * l_ent + LAMBDA_MARGIN * l_mar + LAMBDA_RATE * l_rate

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            ep_ce += l_ce.item()
            ep_ent += l_ent.item()
            ep_mar += l_mar.item()
            ep_rate += l_rate

            with torch.no_grad():
                g = gini_coefficient_batch(importance)
                ep_gini.append(g)

            pbar.set_postfix({
                'CE': f'{l_ce.item():.3f}',
                'Ent': f'{l_ent.item():.3f}',
                'Mar': f'{l_mar.item():.3f}',
                'Gini': f'{g:.4f}',
                'tx': f'{stats["tx_rate"]:.2f}'
            })
        sched.step()

        n_batches = len(train_loader)
        if (ep + 1) % 5 == 0:
            model.eval(); back.eval()
            correct, total, ginis = 0, 0, []
            with torch.no_grad():
                for img, lab in test_loader:
                    img, lab = img.to(device), lab.to(device)
                    feat = front(img)
                    Fp, stats = model(feat, noise_param=0.0)
                    correct += back(Fp).argmax(1).eq(lab).sum().item()
                    total += lab.size(0)
                    ginis.append(gini_coefficient_batch(stats['importance']))
            acc = 100. * correct / total
            mean_gini = np.mean(ginis)
            print(f"  S2 E{ep+1}: Acc={acc:.2f}%, Gini={mean_gini:.4f}, "
                  f"CE={ep_ce/n_batches:.3f}, Ent={ep_ent/n_batches:.3f}, "
                  f"Mar={ep_mar/n_batches:.3f}")
            if acc > best_s2:
                best_s2 = acc
                torch.save(model.state_dict(),
                           os.path.join(SNAP_DIR, f"bsc_v3_s2_{acc:.2f}.pth"))
                print(f"  ✓ Best S2: {best_s2:.2f}%")

    s2f = sorted([f for f in os.listdir(SNAP_DIR) if f.startswith("bsc_v3_s2_")])
    if s2f: model.load_state_dict(torch.load(os.path.join(SNAP_DIR, s2f[-1]), map_location=device))

    # ==================================================================
    # STEP 3: Joint fine-tuning (back + model)
    # ==================================================================
    print(f"\n{'='*60}")
    print("STEP 3: Joint fine-tuning (BSC)")
    print(f"{'='*60}")

    for p in back.parameters(): p.requires_grad = True
    params = list(back.parameters()) + list(model.parameters())
    opt3 = optim.Adam(params, lr=1e-5, weight_decay=1e-4)
    sched3 = optim.lr_scheduler.CosineAnnealingLR(opt3, T_max=30, eta_min=1e-7)

    best_s3 = 0.0
    for ep in range(30):
        model.train(); back.train()
        pbar = tqdm(train_loader, desc=f"S3 E{ep+1}/30")
        opt3.zero_grad()
        for step, (img, lab) in enumerate(pbar):
            img, lab = img.to(device), lab.to(device)
            noise = sample_noise()
            with torch.no_grad(): feat = front(img)
            Fp, stats = model(feat, noise_param=noise)

            importance = stats['importance']
            l_ce = criterion(back(Fp), lab)
            l_ent = score_entropy_loss(importance)
            l_mar = margin_loss(importance, 0.75)
            l_rate = (stats['tx_rate'] - 0.75) ** 2

            loss = (l_ce + LAMBDA_ENTROPY * l_ent + LAMBDA_MARGIN * l_mar + LAMBDA_RATE * l_rate) / 2
            loss.backward()
            if (step + 1) % 2 == 0:
                nn.utils.clip_grad_norm_(params, max_norm=1.0)
                opt3.step(); opt3.zero_grad()

            with torch.no_grad():
                g = gini_coefficient_batch(importance)
            pbar.set_postfix({
                'CE': f'{l_ce.item():.3f}', 'Gini': f'{g:.4f}',
                'tx': f'{stats["tx_rate"]:.2f}'
            })
        sched3.step()

        if (ep + 1) % 5 == 0:
            model.eval(); back.eval()
            correct, total, ginis = 0, 0, []
            with torch.no_grad():
                for img, lab in test_loader:
                    img, lab = img.to(device), lab.to(device)
                    feat = front(img)
                    Fp, stats = model(feat, noise_param=0.0)
                    correct += back(Fp).argmax(1).eq(lab).sum().item()
                    total += lab.size(0)
                    ginis.append(gini_coefficient_batch(stats['importance']))
            acc = 100. * correct / total
            mean_gini = np.mean(ginis)
            print(f"  S3 E{ep+1}: Acc={acc:.2f}%, Gini={mean_gini:.4f}")
            if acc > best_s3:
                best_s3 = acc
                torch.save({'model': model.state_dict(), 'back': back.state_dict()},
                           os.path.join(SNAP_DIR, f"bsc_v3_s3_{acc:.2f}.pth"))
                print(f"  ✓ Best S3: {best_s3:.2f}%")

    # Final evaluation
    s3f = sorted([f for f in os.listdir(SNAP_DIR) if f.startswith("bsc_v3_s3_")])
    if s3f:
        ckpt = torch.load(os.path.join(SNAP_DIR, s3f[-1]), map_location=device)
        model.load_state_dict(ckpt['model']); back.load_state_dict(ckpt['back'])

    model.eval(); back.eval()
    correct, total, all_gini = 0, 0, []
    with torch.no_grad():
        for img, lab in test_loader:
            img, lab = img.to(device), lab.to(device)
            feat = front(img)
            Fp, stats = model(feat, noise_param=0.0)
            correct += back(Fp).argmax(1).eq(lab).sum().item()
            total += lab.size(0)
            all_gini.append(gini_coefficient_batch(stats['importance']))

    final_acc = 100. * correct / total
    final_gini = np.mean(all_gini)
    print(f"\n{'='*60}")
    print(f"FINAL: Acc={final_acc:.2f}%, Gini={final_gini:.4f}")
    print(f"{'='*60}")
    print(f"\nCompare: Original model Gini ~0.019, new model Gini = {final_gini:.4f}")
    if final_gini > 0.05:
        print("✅ Score discrimination significantly improved!")
    elif final_gini > 0.03:
        print("📈 Some improvement in score discrimination")
    else:
        print("⚠️ Scores still relatively flat — may need architecture changes")
