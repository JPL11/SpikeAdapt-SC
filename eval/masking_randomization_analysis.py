"""Comprehensive masking randomization & sanity-check analysis.

Addresses all 5 reviewer / professor checks:
  1. Random-mask distribution (200 draws) — mean/std/percentile
  2. Per-block score stats — variance, CoV, entropy, Gini
  3. Kept-vs-dropped margin — mean score gap per image
  4. Per-block ablation — remove one block, measure damage, correlate with score
  5. Stability — pairwise mask overlap, class/noise dependence
"""

import os, sys, random, json, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy import stats as scipy_stats

# IEEE style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03, 'axes.linewidth': 0.6,
    'lines.linewidth': 1.0, 'lines.markersize': 3.5,
    'grid.linewidth': 0.4, 'grid.alpha': 0.3,
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SNAP_DIR = "./snapshots_aid/"
OUT_DIR  = "./paper/figures/"
os.makedirs(OUT_DIR, exist_ok=True)

# ========================= MODEL CLASSES =========================
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
        sg = sig * (1-sig) * scale
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

class BSC_Channel(nn.Module):
    def forward(self, x, ber):
        if ber <= 0: return x
        return ((x + (torch.rand_like(x.float()) < ber).float()) % 2)

class LearnedImportanceScorer(nn.Module):
    def __init__(self, C_in=128, h=32):
        super().__init__()
        self.scorer = nn.Sequential(nn.Conv2d(C_in,h,1), nn.ReLU(True), nn.Conv2d(h,1,1), nn.Sigmoid())
    def forward(self, S): return self.scorer(torch.stack(S,0).mean(0)).squeeze(1)

class LearnedBlockMask(nn.Module):
    def __init__(self, rate=0.75):
        super().__init__(); self.target_rate = rate
    def forward(self, imp):
        B,H,W = imp.shape; k = max(1,int(self.target_rate*H*W))
        flat = imp.view(B,-1); _,idx = flat.topk(k,1)
        m = torch.zeros_like(flat); m.scatter_(1,idx,1.0)
        return m.view(B,H,W).unsqueeze(1), m.view(B,H,W).mean()

class Encoder(nn.Module):
    def __init__(self, Ci=1024, C1=256, C2=128):
        super().__init__()
        self.conv1=nn.Conv2d(Ci,C1,3,1,1); self.bn1=nn.BatchNorm2d(C1); self.if1=IFNeuron()
        self.conv2=nn.Conv2d(C1,C2,3,1,1); self.bn2=nn.BatchNorm2d(C2); self.if2=IFNeuron()
    def forward(self,F,m1=None,m2=None):
        s1,m1=self.if1(self.bn1(self.conv1(F)),m1); s2,m2=self.if2(self.bn2(self.conv2(s1)),m2)
        return s1,s2,m1,m2

class Decoder(nn.Module):
    def __init__(self, Co=1024, C1=256, C2=128, T=8):
        super().__init__(); self.T=T
        self.conv3=nn.Conv2d(C2,C1,3,1,1); self.bn3=nn.BatchNorm2d(C1); self.if3=IFNeuron()
        self.conv4=nn.Conv2d(C1,Co,3,1,1); self.bn4=nn.BatchNorm2d(Co); self.ihf=IHFNeuron()
        self.converter_fc=nn.Linear(2*T,2*T)
    def forward(self,recv,mask):
        m3,m4=None,None; Fs,Fm=[],[]
        for t in range(self.T):
            s3,m3=self.if3(self.bn3(self.conv3(recv[t]*mask)),m3)
            sp,m4=self.ihf(self.bn4(self.conv4(s3)),m4)
            Fs.append(sp); Fm.append(m4.clone())
        il=[]
        for t in range(self.T): il.append(Fs[t]); il.append(Fm[t])
        x=torch.stack(il,1).permute(0,2,3,4,1)
        return (x*torch.sigmoid(self.converter_fc(x))).sum(-1)

class ResNet50Front(nn.Module):
    def __init__(self):
        super().__init__()
        r=torchvision.models.resnet50(weights='IMAGENET1K_V1')
        self.conv1=r.conv1;self.bn1=r.bn1;self.relu=r.relu;self.maxpool=r.maxpool
        self.layer1=r.layer1;self.layer2=r.layer2;self.layer3=r.layer3
        self.pool=nn.AdaptiveAvgPool2d(8)
    def forward(self,x):
        return self.pool(self.layer3(self.layer2(self.layer1(self.maxpool(self.relu(self.bn1(self.conv1(x))))))))

class ResNet50Back(nn.Module):
    def __init__(self, nc=30):
        super().__init__()
        r=torchvision.models.resnet50(weights='IMAGENET1K_V1')
        self.layer4=r.layer4; self.avgpool=nn.AdaptiveAvgPool2d(1); self.fc=nn.Linear(2048,nc)
    def forward(self,x): return self.fc(torch.flatten(self.avgpool(self.layer4(x)),1))

class AIDDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform=transform; aid=os.path.join(root,'AID')
        classes=sorted([d for d in os.listdir(aid) if os.path.isdir(os.path.join(aid,d))])
        self.c2i={c:i for i,c in enumerate(classes)}; self.i2c={i:c for c,i in self.c2i.items()}
        all_s=[]
        for c in classes:
            cd=os.path.join(aid,c)
            for f in sorted(os.listdir(cd)):
                if f.lower().endswith(('.jpg','.jpeg','.png','.tif')): all_s.append((os.path.join(cd,f),self.c2i[c]))
        random.seed(42); random.shuffle(all_s)
        self.samples=all_s[int(len(all_s)*0.8):]
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        p,l=self.samples[i]; img=Image.open(p).convert('RGB')
        if self.transform: img=self.transform(img)
        return img,l

def load_all():
    front=ResNet50Front().to(device); back=ResNet50Back(30).to(device)
    bb=torch.load(os.path.join(SNAP_DIR,"backbone_best.pth"),map_location=device)
    front.load_state_dict({k:v for k,v in bb.items() if not k.startswith(('layer4.','fc.','avgpool.'))},strict=False)
    back.load_state_dict({k:v for k,v in bb.items() if k.startswith(('layer4.','fc.','avgpool.'))},strict=False)
    front.eval()
    enc=Encoder(1024,256,128).to(device); imp=LearnedImportanceScorer(128,32).to(device)
    dec=Decoder(1024,256,128,8).to(device); ch=BSC_Channel().to(device)
    s3=sorted([f for f in os.listdir(SNAP_DIR) if f.startswith("bsc_s3_")])
    if s3:
        ck=torch.load(os.path.join(SNAP_DIR,s3[-1]),map_location=device)
        enc.load_state_dict({k.replace('encoder.',''):v for k,v in ck['model'].items() if 'encoder.' in k})
        imp.load_state_dict({k.replace('importance_scorer.',''):v for k,v in ck['model'].items() if 'importance_scorer.' in k})
        dec.load_state_dict({k.replace('decoder.',''):v for k,v in ck['model'].items() if 'decoder.' in k})
        back.load_state_dict(ck['back']); print(f"Loaded: {s3[-1]}")
    enc.eval();imp.eval();dec.eval();back.eval()
    return front,back,enc,imp,ch,dec

T_STEPS = 8

def encode_batch(front, enc, imgs):
    """Return features and spike trains."""
    feat = front(imgs)
    S2, m1, m2 = [], None, None
    for t in range(T_STEPS):
        _, s2, m1, m2 = enc(feat, m1, m2)
        S2.append(s2)
    return feat, S2

def decode_classify(dec, back, S2, mask):
    """Decode with mask and classify."""
    recv = [S2[t] * mask for t in range(T_STEPS)]
    Fp = dec(recv, mask)
    logits = back(Fp)
    return logits, Fp

def gini_coefficient(x):
    """Gini coefficient of array x."""
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n+1)
    return (2*np.sum(index*x) - (n+1)*np.sum(x)) / (n*np.sum(x)+1e-12)

def score_entropy(x):
    """Entropy of normalized scores."""
    p = x / (x.sum() + 1e-12)
    p = p[p > 0]
    return -np.sum(p * np.log(p + 1e-12))


if __name__ == "__main__":
    print(f"Device: {device}")
    front, back, enc, imp_s, ch, dec = load_all()
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                     T.Normalize((.485,.456,.406),(.229,.224,.225))])
    ds = AIDDataset("./data", transform=tf)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)
    print(f"Test set: {len(ds)} images")

    results = {}
    N_BLOCKS = 64  # 8x8

    # ================================================================
    # PHASE 1: Collect features, scores, and learned-mask predictions
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 1: Collecting features and importance scores")
    print("="*60)

    all_feats = []      # (N, 1024, 8, 8)
    all_S2 = []         # list of T lists of (N, 128, 8, 8) — stored as list of batches
    all_scores = []     # (N, 64) importance scores
    all_labels = []
    all_correct_learned = []  # per-image correct flag with learned mask

    with torch.no_grad():
        for bi, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device)
            feat, S2 = encode_batch(front, enc, imgs)
            importance = imp_s(S2)  # (B, 8, 8)
            B = imgs.shape[0]

            all_feats.append(feat.cpu())
            # Store S2 as list of T tensors per batch
            all_S2.append([s.cpu() for s in S2])
            all_scores.append(importance.cpu().numpy().reshape(B, -1))
            all_labels.extend(labels.numpy())

            # Learned mask at rho=0.75
            k = int(0.75 * 64)
            flat = importance.view(B, -1)
            _, idx = flat.topk(k, 1)
            mask_l = torch.zeros_like(flat); mask_l.scatter_(1, idx, 1.0)
            mask_l = mask_l.view(B, 8, 8).unsqueeze(1)

            logits_l, _ = decode_classify(dec, back, S2, mask_l)
            preds_l = logits_l.argmax(1).cpu()
            all_correct_learned.extend((preds_l == labels).numpy())

            if (bi+1) % 20 == 0:
                print(f"  Batch {bi+1}/{len(loader)}")

    all_scores = np.concatenate(all_scores, axis=0)  # (N, 64)
    all_labels = np.array(all_labels)
    all_correct_learned = np.array(all_correct_learned)
    learned_acc = all_correct_learned.mean() * 100
    N = len(all_labels)
    print(f"  Collected {N} images. Learned mask accuracy: {learned_acc:.2f}%")

    # ================================================================
    # PHASE 2: Random-mask distribution (200 draws)
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 2: Random-mask distribution (200 draws)")
    print("="*60)

    N_DRAWS = 200
    random_accs = []

    for draw in range(N_DRAWS):
        correct = 0
        total = 0
        with torch.no_grad():
            for bi in range(len(all_S2)):
                S2_batch = [s.to(device) for s in all_S2[bi]]
                B_cur = S2_batch[0].shape[0]
                labels_batch = all_labels[total:total+B_cur]

                # Random mask at rho=0.75
                k = int(0.75 * 64)
                rand_flat = torch.rand(B_cur, 64, device=device)
                _, ridx = rand_flat.topk(k, 1)
                mask_r = torch.zeros(B_cur, 64, device=device)
                mask_r.scatter_(1, ridx, 1.0)
                mask_r = mask_r.view(B_cur, 8, 8).unsqueeze(1)

                logits_r, _ = decode_classify(dec, back, S2_batch, mask_r)
                preds_r = logits_r.argmax(1).cpu().numpy()
                correct += (preds_r == labels_batch).sum()
                total += B_cur

        acc = correct / total * 100
        random_accs.append(acc)
        if (draw+1) % 20 == 0:
            print(f"  Draw {draw+1}/{N_DRAWS}: acc={acc:.2f}%")

    random_accs = np.array(random_accs)
    r_mean = random_accs.mean()
    r_std = random_accs.std()
    r_min = random_accs.min()
    r_max = random_accs.max()
    r_p05 = np.percentile(random_accs, 5)
    r_p95 = np.percentile(random_accs, 95)
    # Percentile rank of learned result
    pct_rank = (random_accs < learned_acc).sum() / len(random_accs) * 100

    print(f"\n  Random mask distribution (N={N_DRAWS} draws, ρ=0.75):")
    print(f"    Mean:  {r_mean:.2f}%")
    print(f"    Std:   {r_std:.2f}%")
    print(f"    Range: [{r_min:.2f}%, {r_max:.2f}%]")
    print(f"    95% CI: [{r_p05:.2f}%, {r_p95:.2f}%]")
    print(f"    Learned: {learned_acc:.2f}%")
    print(f"    Learned exceeds {pct_rank:.1f}% of random draws")

    results['random_dist'] = {
        'mean': float(r_mean), 'std': float(r_std),
        'min': float(r_min), 'max': float(r_max),
        'p05': float(r_p05), 'p95': float(r_p95),
        'learned_acc': float(learned_acc),
        'percentile_rank': float(pct_rank)
    }

    # ================================================================
    # PHASE 2b: Same analysis at ρ=0.50
    # ================================================================
    print("\n  --- Also checking ρ=0.50 ---")

    # Learned at ρ=0.50
    learned_correct_50 = 0
    random_accs_50_draws = []

    with torch.no_grad():
        total = 0
        for bi in range(len(all_S2)):
            S2_batch = [s.to(device) for s in all_S2[bi]]
            B_cur = S2_batch[0].shape[0]
            labels_batch = all_labels[total:total+B_cur]

            # Importance from stored scores
            imp_batch = torch.tensor(all_scores[total:total+B_cur], device=device)
            k = int(0.50 * 64)
            _, idx = imp_batch.topk(k, 1)
            mask_l = torch.zeros_like(imp_batch); mask_l.scatter_(1, idx, 1.0)
            mask_l = mask_l.view(B_cur, 8, 8).unsqueeze(1)

            logits_l, _ = decode_classify(dec, back, S2_batch, mask_l)
            preds_l = logits_l.argmax(1).cpu().numpy()
            learned_correct_50 += (preds_l == labels_batch).sum()
            total += B_cur

    learned_acc_50 = learned_correct_50 / N * 100
    print(f"    Learned mask ρ=0.50: {learned_acc_50:.2f}%")

    # 50 random draws at ρ=0.50 (fewer — just need the distribution)
    for draw in range(50):
        correct = 0; total = 0
        with torch.no_grad():
            for bi in range(len(all_S2)):
                S2_batch = [s.to(device) for s in all_S2[bi]]
                B_cur = S2_batch[0].shape[0]
                labels_batch = all_labels[total:total+B_cur]
                k = int(0.50 * 64)
                rand_flat = torch.rand(B_cur, 64, device=device)
                _, ridx = rand_flat.topk(k, 1)
                mask_r = torch.zeros(B_cur, 64, device=device)
                mask_r.scatter_(1, ridx, 1.0)
                mask_r = mask_r.view(B_cur, 8, 8).unsqueeze(1)
                logits_r, _ = decode_classify(dec, back, S2_batch, mask_r)
                preds_r = logits_r.argmax(1).cpu().numpy()
                correct += (preds_r == labels_batch).sum()
                total += B_cur
        random_accs_50_draws.append(correct / total * 100)
        if (draw+1) % 10 == 0:
            print(f"    Draw {draw+1}/50: acc={random_accs_50_draws[-1]:.2f}%")

    ra50 = np.array(random_accs_50_draws)
    pct_50 = (ra50 < learned_acc_50).sum() / len(ra50) * 100
    print(f"    Random ρ=0.50: {ra50.mean():.2f} ± {ra50.std():.2f}%  [{ra50.min():.2f}, {ra50.max():.2f}]")
    print(f"    Learned exceeds {pct_50:.1f}% of random draws at ρ=0.50")

    results['rho_050'] = {
        'learned': float(learned_acc_50),
        'random_mean': float(ra50.mean()), 'random_std': float(ra50.std()),
        'percentile_rank': float(pct_50)
    }

    # ================================================================
    # PHASE 3: Per-block score statistics
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 3: Per-block score statistics")
    print("="*60)

    # Per-image stats
    per_image_var = np.var(all_scores, axis=1)
    per_image_cov = np.std(all_scores, axis=1) / (np.mean(all_scores, axis=1) + 1e-12)
    per_image_entropy = np.array([score_entropy(s) for s in all_scores])
    per_image_gini = np.array([gini_coefficient(s) for s in all_scores])

    print(f"  Per-image score variance:  {per_image_var.mean():.6f} ± {per_image_var.std():.6f}")
    print(f"  Per-image CoV:             {per_image_cov.mean():.4f} ± {per_image_cov.std():.4f}")
    print(f"  Per-image entropy:         {per_image_entropy.mean():.4f} ± {per_image_entropy.std():.4f}")
    print(f"  Max possible entropy (uniform 64): {np.log(64):.4f}")
    print(f"  Per-image Gini:            {per_image_gini.mean():.4f} ± {per_image_gini.std():.4f}")
    print(f"  (Gini 0=equal, 1=maximally concentrated)")

    # Per-block stats (across images)
    per_block_mean = all_scores.mean(axis=0)
    per_block_var = all_scores.var(axis=0)
    print(f"\n  Per-block mean range: [{per_block_mean.min():.4f}, {per_block_mean.max():.4f}]")
    print(f"  Per-block var range:  [{per_block_var.min():.6f}, {per_block_var.max():.6f}]")
    print(f"  Blocks with mean>0.95: {(per_block_mean > 0.95).sum()}/64")
    print(f"  Blocks with mean<0.85: {(per_block_mean < 0.85).sum()}/64")

    results['score_stats'] = {
        'mean_var': float(per_image_var.mean()),
        'mean_cov': float(per_image_cov.mean()),
        'mean_entropy': float(per_image_entropy.mean()),
        'max_entropy': float(np.log(64)),
        'mean_gini': float(per_image_gini.mean()),
    }

    # ================================================================
    # PHASE 4: Kept-vs-dropped margin
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 4: Kept-vs-dropped block score margin")
    print("="*60)

    k = int(0.75 * 64)  # 48 kept
    margins = []
    for i in range(N):
        scores_i = all_scores[i]
        sorted_idx = np.argsort(scores_i)[::-1]
        kept_scores = scores_i[sorted_idx[:k]]
        dropped_scores = scores_i[sorted_idx[k:]]
        margin = kept_scores.mean() - dropped_scores.mean()
        margins.append(margin)
    margins = np.array(margins)
    print(f"  Mean margin (kept - dropped): {margins.mean():.4f} ± {margins.std():.4f}")
    print(f"  Min margin: {margins.min():.4f}, Max: {margins.max():.4f}")
    print(f"  Fraction with margin > 0.01: {(margins > 0.01).mean():.2%}")
    print(f"  Fraction with margin > 0.05: {(margins > 0.05).mean():.2%}")

    results['margin'] = {
        'mean': float(margins.mean()), 'std': float(margins.std()),
        'min': float(margins.min()), 'max': float(margins.max()),
    }

    # ================================================================
    # PHASE 5: Per-block ablation (leave-one-out)
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 5: Per-block ablation (leave-one-out, 100 sample images)")
    print("="*60)

    N_SAMPLE = 100
    np.random.seed(42)
    sample_idx = np.random.choice(N, N_SAMPLE, replace=False)

    # For each sample image, remove one block at a time and measure confidence drop
    block_damage = np.zeros((N_SAMPLE, 64))  # damage when block removed
    block_scores_sample = np.zeros((N_SAMPLE, 64))

    with torch.no_grad():
        for si, img_idx in enumerate(sample_idx):
            # Find which batch this image belongs to
            bi = img_idx // 16
            local_idx = img_idx % 16

            S2_batch = [s.to(device) for s in all_S2[bi]]
            B_cur = S2_batch[0].shape[0]
            if local_idx >= B_cur:
                continue

            # Get single image spikes
            S2_single = [s[local_idx:local_idx+1] for s in S2_batch]

            # Full mask — no blocks removed
            full_mask = torch.ones(1, 1, 8, 8, device=device)
            logits_full, _ = decode_classify(dec, back, S2_single, full_mask)
            conf_full = F.softmax(logits_full, 1).max(1)[0].item()

            block_scores_sample[si] = all_scores[img_idx]

            # Remove each block one at a time
            for b in range(64):
                row, col = b // 8, b % 8
                mask_ablated = torch.ones(1, 1, 8, 8, device=device)
                mask_ablated[0, 0, row, col] = 0.0
                logits_ab, _ = decode_classify(dec, back, S2_single, mask_ablated)
                conf_ab = F.softmax(logits_ab, 1).max(1)[0].item()
                block_damage[si, b] = conf_full - conf_ab  # positive = block was important

            if (si+1) % 20 == 0:
                print(f"  Ablated {si+1}/{N_SAMPLE} images")

    # Compute correlation between importance score and ablation damage
    all_scores_flat = block_scores_sample.flatten()
    all_damage_flat = block_damage.flatten()
    corr, p_value = scipy_stats.pearsonr(all_scores_flat, all_damage_flat)
    spearman_corr, sp_p = scipy_stats.spearmanr(all_scores_flat, all_damage_flat)

    print(f"\n  Score-Damage correlation:")
    print(f"    Pearson r = {corr:.4f} (p={p_value:.2e})")
    print(f"    Spearman ρ = {spearman_corr:.4f} (p={sp_p:.2e})")

    # Bin blocks by score quartile, show mean damage
    quartiles = np.percentile(all_scores_flat, [25, 50, 75])
    q_labels = ['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']
    q_bins = np.digitize(all_scores_flat, quartiles)
    for qi in range(4):
        mask_q = q_bins == qi
        if mask_q.sum() > 0:
            print(f"    {q_labels[qi]}: mean damage = {all_damage_flat[mask_q].mean():.6f} "
                  f"(n={mask_q.sum()}, mean score={all_scores_flat[mask_q].mean():.4f})")

    results['ablation'] = {
        'pearson_r': float(corr), 'pearson_p': float(p_value),
        'spearman_rho': float(spearman_corr), 'spearman_p': float(sp_p),
    }

    # ================================================================
    # PHASE 6: Stability — mask overlap and class dependence
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 6: Mask stability and class dependence")
    print("="*60)

    # Generate binary masks for all images at rho=0.75
    k = int(0.75 * 64)
    all_masks = np.zeros((N, 64), dtype=np.float32)
    for i in range(N):
        sorted_idx = np.argsort(all_scores[i])[::-1]
        all_masks[i, sorted_idx[:k]] = 1.0

    # Unique masks
    unique_masks = len(set([tuple(m) for m in all_masks]))
    print(f"  Unique masks: {unique_masks}/{N}")

    # Pairwise overlap (sample 500 pairs)
    np.random.seed(42)
    pairs = [(np.random.randint(N), np.random.randint(N)) for _ in range(500)]
    overlaps = [np.sum(all_masks[i] * all_masks[j]) / k for i, j in pairs if i != j]
    print(f"  Mean pairwise overlap: {np.mean(overlaps):.4f} (1.0 = identical masks)")
    print(f"  Overlap std: {np.std(overlaps):.4f}")

    # Per-class mask analysis
    class_names = sorted(os.listdir(os.path.join("./data", "AID")))
    class_names = [c for c in class_names if os.path.isdir(os.path.join("./data", "AID", c))]
    print(f"\n  Per-class mask diversity:")
    class_stats = {}
    for ci, cls in enumerate(class_names):
        cls_mask_idx = all_labels == ci
        if cls_mask_idx.sum() < 5:
            continue
        cls_masks = all_masks[cls_mask_idx]
        cls_unique = len(set([tuple(m) for m in cls_masks]))
        # Intra-class overlap
        if len(cls_masks) >= 2:
            pairs_c = [(0, j) for j in range(1, min(len(cls_masks), 20))]
            ovlp = [np.sum(cls_masks[i] * cls_masks[j]) / k for i, j in pairs_c]
            mean_ovlp = np.mean(ovlp)
        else:
            mean_ovlp = 1.0
        class_stats[cls] = {'unique': cls_unique, 'total': int(cls_mask_idx.sum()),
                           'intra_overlap': float(mean_ovlp)}
        if ci < 10 or cls in ['Airport', 'Desert', 'Harbor', 'Farmland']:
            print(f"    {cls:15s}: {cls_unique:4d}/{cls_mask_idx.sum()} unique, "
                  f"intra-overlap={mean_ovlp:.4f}")

    results['stability'] = {
        'unique_masks': unique_masks,
        'mean_overlap': float(np.mean(overlaps)),
        'class_stats': class_stats,
    }

    # ================================================================
    # GENERATE FIGURES
    # ================================================================
    print("\n" + "="*60)
    print("Generating publication figures")
    print("="*60)

    # ------- FIGURE: Random-mask distribution histogram -------
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5))

    ax = axes[0]
    ax.hist(random_accs, bins=30, color='#90CAF9', edgecolor='white', linewidth=0.3, alpha=0.9)
    ax.axvline(learned_acc, color='#E53935', linestyle='-', linewidth=1.5,
               label=f'Learned mask: {learned_acc:.2f}%')
    ax.axvline(r_mean, color='#666', linestyle='--', linewidth=1,
               label=f'Random mean: {r_mean:.2f}%')
    ax.axvspan(r_p05, r_p95, alpha=0.15, color='#1E88E5', label='90% CI')
    ax.set_xlabel('Accuracy (%)', fontsize=8)
    ax.set_ylabel('Count', fontsize=8)
    ax.set_title(r'(a) Random mask distribution ($\rho=0.75$, 200 draws)', fontsize=8)
    ax.legend(fontsize=6)

    ax = axes[1]
    ax.hist(random_accs_50_draws, bins=20, color='#A5D6A7', edgecolor='white', linewidth=0.3, alpha=0.9)
    ax.axvline(learned_acc_50, color='#E53935', linestyle='-', linewidth=1.5,
               label=f'Learned: {learned_acc_50:.2f}%')
    ax.axvline(ra50.mean(), color='#666', linestyle='--', linewidth=1,
               label=f'Random mean: {ra50.mean():.2f}%')
    ax.set_xlabel('Accuracy (%)', fontsize=8)
    ax.set_ylabel('Count', fontsize=8)
    ax.set_title(r'(b) Random mask distribution ($\rho=0.50$, 50 draws)', fontsize=8)
    ax.legend(fontsize=6)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig_random_dist.png'), facecolor='white')
    print("  Saved fig_random_dist.png")
    plt.close()

    # ------- FIGURE: Ablation correlation -------
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5))

    ax = axes[0]
    # Scatter with alpha
    ax.scatter(all_scores_flat, all_damage_flat, s=0.5, alpha=0.1, color='#1E88E5')
    # Binned means
    bins = np.linspace(all_scores_flat.min(), all_scores_flat.max(), 20)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = []
    for i in range(len(bins)-1):
        in_bin = (all_scores_flat >= bins[i]) & (all_scores_flat < bins[i+1])
        if in_bin.sum() > 0:
            bin_means.append(all_damage_flat[in_bin].mean())
        else:
            bin_means.append(np.nan)
    ax.plot(bin_centers, bin_means, 'o-', color='#E53935', linewidth=1.2, markersize=4,
            label=f'Binned mean (r={corr:.3f})')
    ax.set_xlabel('Learned Importance Score', fontsize=8)
    ax.set_ylabel('Confidence Drop (block ablated)', fontsize=8)
    ax.set_title('(a) Score vs. Ablation Damage', fontsize=8)
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # Panel 2: Quartile bar chart
    ax = axes[1]
    q_means = []
    q_errs = []
    for qi in range(4):
        mask_q = q_bins == qi
        if mask_q.sum() > 0:
            q_means.append(all_damage_flat[mask_q].mean())
            q_errs.append(all_damage_flat[mask_q].std() / np.sqrt(mask_q.sum()))
        else:
            q_means.append(0); q_errs.append(0)
    colors_q = ['#BBDEFB', '#64B5F6', '#1E88E5', '#0D47A1']
    ax.bar(range(4), q_means, yerr=q_errs, color=colors_q, edgecolor='white',
           capsize=3, linewidth=0.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(q_labels, fontsize=7)
    ax.set_ylabel('Mean Confidence Drop', fontsize=8)
    ax.set_title('(b) Damage by Score Quartile', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig_ablation_corr.png'), facecolor='white')
    print("  Saved fig_ablation_corr.png")
    plt.close()

    # ------- FIGURE: Score distribution and margin -------
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2))

    # Panel 1: Score distribution for 5 example images
    ax = axes[0]
    np.random.seed(42)
    ex_idxs = np.random.choice(N, 5, replace=False)
    colors_ex = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA']
    for ei, idx in enumerate(ex_idxs):
        scores_sorted = np.sort(all_scores[idx])[::-1]
        ax.plot(range(64), scores_sorted, '-', color=colors_ex[ei], linewidth=0.8, alpha=0.7)
    ax.axhline(1/64, color='gray', linestyle=':', linewidth=0.6, label='Uniform')
    ax.set_xlabel('Block rank', fontsize=7)
    ax.set_ylabel('Importance score', fontsize=7)
    ax.set_title('(a) Score Profiles', fontsize=8)
    ax.grid(True, alpha=0.2)

    # Panel 2: Histogram of per-image Gini
    ax = axes[1]
    ax.hist(per_image_gini, bins=40, color='#FF7043', edgecolor='white', linewidth=0.3, alpha=0.85)
    ax.axvline(per_image_gini.mean(), color='#333', linestyle='--', linewidth=1,
               label=f'Mean Gini={per_image_gini.mean():.4f}')
    ax.set_xlabel('Gini Coefficient', fontsize=7)
    ax.set_ylabel('Count', fontsize=7)
    ax.set_title('(b) Score Concentration', fontsize=8)
    ax.legend(fontsize=6)

    # Panel 3: Margin histogram
    ax = axes[2]
    ax.hist(margins, bins=40, color='#66BB6A', edgecolor='white', linewidth=0.3, alpha=0.85)
    ax.axvline(margins.mean(), color='#333', linestyle='--', linewidth=1,
               label=f'Mean margin={margins.mean():.4f}')
    ax.set_xlabel('Score Gap (kept − dropped)', fontsize=7)
    ax.set_ylabel('Count', fontsize=7)
    ax.set_title('(c) Kept vs. Dropped Margin', fontsize=8)
    ax.legend(fontsize=6)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig_score_analysis.png'), facecolor='white')
    print("  Saved fig_score_analysis.png")
    plt.close()

    # Save all results
    with open(os.path.join(OUT_DIR, 'masking_analysis_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("\n  Saved masking_analysis_results.json")

    # ================================================================
    # FINAL VERDICT
    # ================================================================
    print("\n" + "="*60)
    print("FINAL VERDICT")
    print("="*60)
    print(f"\n  1. RANDOM DISTRIBUTION: Learned {learned_acc:.2f}% exceeds {pct_rank:.0f}% "
          f"of random draws (mean={r_mean:.2f}±{r_std:.2f})")
    if pct_rank >= 95:
        print("     → STRONG: Learned mask is statistically better than random")
    elif pct_rank >= 75:
        print("     → MODERATE: Learned mask is above average but not dominant")
    else:
        print("     → WEAK: Learned mask is not clearly better than random")

    print(f"\n  2. SCORE DISCRIMINATION: Gini={per_image_gini.mean():.4f}, "
          f"CoV={per_image_cov.mean():.4f}")
    if per_image_gini.mean() > 0.05:
        print("     → Scores show meaningful concentration")
    else:
        print("     → Scores are relatively flat (low selectivity)")

    print(f"\n  3. KEPT-DROPPED MARGIN: {margins.mean():.4f} ± {margins.std():.4f}")

    print(f"\n  4. ABLATION CORRELATION: Pearson r={corr:.4f} (p={p_value:.2e}), "
          f"Spearman ρ={spearman_corr:.4f}")
    if abs(corr) > 0.1 and p_value < 0.01:
        print("     → Scores correlate with block importance (learned scorer is meaningful)")
    else:
        print("     → Weak correlation (scorer may not be strongly informative)")

    print(f"\n  5. MASK DIVERSITY: {unique_masks}/{N} unique masks, "
          f"pairwise overlap={np.mean(overlaps):.4f}")

    print(f"\n  BONUS: ρ=0.50 — Learned: {learned_acc_50:.2f}%, Random: {ra50.mean():.2f}±{ra50.std():.2f}%, "
          f"exceeds {pct_50:.0f}% of draws")
