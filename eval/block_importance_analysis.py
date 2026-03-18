"""Per-block importance analysis — Professor's sanity check.

Analyzes:
1. Per-block importance score distribution across images
2. Variance of importance per spatial position (high = content-adaptive, low = static)
3. Mask area vs MSE (how much feature distortion at different rates?)
4. Mask area vs Confidence (does confidence degrade smoothly?)
5. Learned vs Random: side-by-side comparison at per-block level
"""

import os, sys, random, math, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# IEEE style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
    'axes.linewidth': 0.6,
    'lines.linewidth': 1.0,
    'lines.markersize': 3.5,
    'grid.linewidth': 0.4,
    'grid.alpha': 0.3,
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SNAP_DIR = "./snapshots_aid/"
OUT_DIR = "./paper/figures/"
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
        scale = 10.0; sig = torch.sigmoid(scale * (membrane - threshold))
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
        return img,l,p

def load_all():
    front=ResNet50Front().to(device); back=ResNet50Back(30).to(device)
    bb=torch.load(os.path.join(SNAP_DIR,"backbone_best.pth"),map_location=device)
    front.load_state_dict({k:v for k,v in bb.items() if not k.startswith(('layer4.','fc.','avgpool.'))},strict=False)
    back.load_state_dict({k:v for k,v in bb.items() if k.startswith(('layer4.','fc.','avgpool.'))},strict=False)
    front.eval()
    enc=Encoder(1024,256,128).to(device); imp=LearnedImportanceScorer(128,32).to(device)
    bm=LearnedBlockMask(0.75); dec=Decoder(1024,256,128,8).to(device); ch=BSC_Channel().to(device)
    s3=sorted([f for f in os.listdir(SNAP_DIR) if f.startswith("bsc_s3_")])
    if s3:
        ck=torch.load(os.path.join(SNAP_DIR,s3[-1]),map_location=device)
        enc.load_state_dict({k.replace('encoder.',''):v for k,v in ck['model'].items() if 'encoder.' in k})
        imp.load_state_dict({k.replace('importance_scorer.',''):v for k,v in ck['model'].items() if 'importance_scorer.' in k})
        dec.load_state_dict({k.replace('decoder.',''):v for k,v in ck['model'].items() if 'decoder.' in k})
        back.load_state_dict(ck['back']); print(f"Loaded: {s3[-1]}")
    enc.eval();imp.eval();dec.eval();back.eval()
    return front,back,enc,imp,bm,ch,dec


if __name__ == "__main__":
    print(f"Device: {device}")
    front,back,enc,imp_s,bm,ch,dec = load_all()
    tf=T.Compose([T.Resize(256),T.CenterCrop(224),T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])
    ds=AIDDataset("./data",transform=tf)
    print(f"Test set: {len(ds)} images")

    # ================================================================
    # ANALYSIS 1: Collect per-block importance scores for ALL test imgs
    # ================================================================
    print("\n=== Collecting per-block importance scores ===")
    all_importance = []  # (N, 8, 8)
    all_labels = []
    all_confs_learned = []
    all_confs_random = []
    all_mse_rates = {r: [] for r in [0.25, 0.5, 0.75, 1.0]}
    all_conf_rates = {r: [] for r in [0.25, 0.5, 0.75, 1.0]}

    T_steps = 8
    batch_count = 0
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)

    for imgs, labels, paths in loader:
        imgs = imgs.to(device)
        with torch.no_grad():
            feat = front(imgs)
            feat_clean = feat.clone()

            # Encode
            S2, m1, m2 = [], None, None
            for t in range(T_steps):
                _, s2, m1, m2 = enc(feat, m1, m2)
                S2.append(s2)

            # Importance scores
            importance = imp_s(S2)  # (B, 8, 8)
            all_importance.append(importance.cpu().numpy())
            all_labels.extend(labels.numpy())

            # For each rate: compute MSE and confidence
            for rate in [0.25, 0.5, 0.75, 1.0]:
                B,H,W = importance.shape; k = max(1,int(rate*H*W))

                # Learned mask
                flat = importance.view(B,-1); _,idx = flat.topk(k,1)
                mask_l = torch.zeros_like(flat); mask_l.scatter_(1,idx,1.0)
                mask_l = mask_l.view(B,H,W).unsqueeze(1)

                # Decode with learned mask (BSC clean)
                recv_l = [S2[t]*mask_l for t in range(T_steps)]
                Fp_l = dec(recv_l, mask_l)
                mse_l = F.mse_loss(Fp_l, feat_clean, reduction='none').mean(dim=[1,2,3])
                logits_l = back(Fp_l)
                conf_l = F.softmax(logits_l, 1).max(1)[0]

                all_mse_rates[rate].extend(mse_l.cpu().numpy())
                all_conf_rates[rate].extend(conf_l.cpu().numpy())

                if rate == 0.75:
                    all_confs_learned.extend(conf_l.cpu().numpy())

                    # Random mask at same rate
                    rand_flat = torch.rand(B, H*W, device=device)
                    _,ridx = rand_flat.topk(k,1)
                    mask_r = torch.zeros_like(rand_flat); mask_r.scatter_(1,ridx,1.0)
                    mask_r = mask_r.view(B,H,W).unsqueeze(1)
                    recv_r = [S2[t]*mask_r for t in range(T_steps)]
                    Fp_r = dec(recv_r, mask_r)
                    conf_r = F.softmax(back(Fp_r), 1).max(1)[0]
                    all_confs_random.extend(conf_r.cpu().numpy())

        batch_count += 1
        if batch_count % 20 == 0:
            print(f"  Processed {batch_count * 16}/{len(ds)} images")

    all_importance = np.concatenate(all_importance, axis=0)  # (N, 8, 8)
    all_labels = np.array(all_labels)
    print(f"  Total: {all_importance.shape[0]} images, importance shape: {all_importance.shape}")

    # ================================================================
    # ANALYSIS 2: Per-block variance (Professor's key question)
    # ================================================================
    print("\n=== Per-block importance statistics ===")

    # Flatten to (N, 64) — each row is one image's importance vector
    imp_flat = all_importance.reshape(all_importance.shape[0], -1)  # (N, 64)

    # Per-block variance across images
    per_block_var = np.var(imp_flat, axis=0)  # (64,)
    per_block_mean = np.mean(imp_flat, axis=0)  # (64,)

    # Per-image variance (does each image have different block importances?)
    per_image_var = np.var(imp_flat, axis=1)  # (N,)

    print(f"  Per-block variance: min={per_block_var.min():.6f}, max={per_block_var.max():.6f}, mean={per_block_var.mean():.6f}")
    print(f"  Per-image variance: min={per_image_var.min():.6f}, max={per_image_var.max():.6f}, mean={per_image_var.mean():.6f}")

    # Check if some blocks are always selected / always dropped
    always_high = np.sum(per_block_mean > 0.9)
    always_low = np.sum(per_block_mean < 0.1)
    print(f"  Blocks with mean>0.9 (always selected): {always_high}/64")
    print(f"  Blocks with mean<0.1 (always dropped): {always_low}/64")
    print(f"  Blocks with high variance (>0.01): {np.sum(per_block_var > 0.01)}/64")

    # ================================================================
    # FIGURE A: Per-block importance variance heatmap (8x8)
    # ================================================================
    print("\n=== Generating figures ===")

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2))

    # Panel 1: Mean importance per block
    ax = axes[0]
    im = ax.imshow(per_block_mean.reshape(8,8), cmap='RdYlGn', vmin=0, vmax=1, aspect='equal')
    ax.set_title('Mean Importance', fontsize=8)
    ax.set_xlabel('Block column', fontsize=7)
    ax.set_ylabel('Block row', fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(8):
        for j in range(8):
            ax.text(j, i, f'{per_block_mean.reshape(8,8)[i,j]:.2f}',
                   ha='center', va='center', fontsize=4.5, color='black')

    # Panel 2: Variance per block
    ax = axes[1]
    im = ax.imshow(per_block_var.reshape(8,8), cmap='hot', aspect='equal')
    ax.set_title('Importance Variance', fontsize=8)
    ax.set_xlabel('Block column', fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(8):
        for j in range(8):
            ax.text(j, i, f'{per_block_var.reshape(8,8)[i,j]:.3f}',
                   ha='center', va='center', fontsize=4, color='white')

    # Panel 3: Distribution of per-image importance variance
    ax = axes[2]
    ax.hist(per_image_var, bins=50, color='#2196F3', edgecolor='white', linewidth=0.3, alpha=0.85)
    ax.axvline(per_image_var.mean(), color='red', linestyle='--', linewidth=1, label=f'Mean={per_image_var.mean():.4f}')
    ax.set_xlabel('Within-image importance variance', fontsize=7)
    ax.set_ylabel('Count', fontsize=7)
    ax.set_title('Per-Image Score Spread', fontsize=8)
    ax.legend(fontsize=6)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig5_block_importance.png'), facecolor='white')
    print("  Saved fig5_block_importance.png")
    plt.close()

    # ================================================================
    # FIGURE B: Mask Area vs MSE and Confidence (2-panel)
    # ================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.5))

    rates = [0.25, 0.5, 0.75, 1.0]
    mean_mse = [np.mean(all_mse_rates[r]) for r in rates]
    std_mse = [np.std(all_mse_rates[r]) for r in rates]
    mean_conf = [np.mean(all_conf_rates[r]) for r in rates]
    std_conf = [np.std(all_conf_rates[r]) for r in rates]

    # Panel 1: Mask area vs MSE
    ax1.errorbar([r*100 for r in rates], mean_mse, yerr=std_mse,
                 fmt='o-', color='#E53935', linewidth=1.2, markersize=5, capsize=3, label='Feature MSE')
    ax1.set_xlabel('Transmission Rate (%)', fontsize=8)
    ax1.set_ylabel('Feature MSE', fontsize=8)
    ax1.set_title('Feature Distortion vs. Rate', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([25, 50, 75, 100])

    # Panel 2: Mask area vs Confidence
    ax2.errorbar([r*100 for r in rates], mean_conf, yerr=std_conf,
                 fmt='s-', color='#43A047', linewidth=1.2, markersize=5, capsize=3, label='Confidence')
    ax2.set_xlabel('Transmission Rate (%)', fontsize=8)
    ax2.set_ylabel('Classification Confidence', fontsize=8)
    ax2.set_title('Confidence vs. Rate', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks([25, 50, 75, 100])
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig6_rate_vs_metrics.png'), facecolor='white')
    print("  Saved fig6_rate_vs_metrics.png")
    plt.close()

    # ================================================================
    # FIGURE C: Learned vs Random — sorted importance & confidence
    # ================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.5))

    # Panel 1: Sort importance scores for 5 example images
    np.random.seed(42)
    example_idxs = np.random.choice(len(imp_flat), 5, replace=False)
    colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA']

    for i, idx in enumerate(example_idxs):
        scores = np.sort(imp_flat[idx])[::-1]  # descending
        ax1.plot(range(64), scores, '-', color=colors[i], linewidth=0.8,
                alpha=0.7, label=f'Image {idx}')

    # Also plot: random "importance" (flat line)
    ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Random (uniform)')
    ax1.set_xlabel('Block rank (sorted by importance)', fontsize=8)
    ax1.set_ylabel('Importance score', fontsize=8)
    ax1.set_title('Learned Importance Ranking', fontsize=9)
    ax1.legend(fontsize=5.5, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Confidence histogram — learned vs random
    ax2.hist(all_confs_learned, bins=50, alpha=0.7, color='#1E88E5',
             edgecolor='white', linewidth=0.3, label=f'Learned (mean={np.mean(all_confs_learned):.4f})')
    ax2.hist(all_confs_random, bins=50, alpha=0.5, color='#E53935',
             edgecolor='white', linewidth=0.3, label=f'Random (mean={np.mean(all_confs_random):.4f})')
    ax2.set_xlabel('Classification Confidence', fontsize=8)
    ax2.set_ylabel('Count', fontsize=8)
    ax2.set_title(r'Learned vs Random Mask ($\rho=0.75$)', fontsize=9)
    ax2.legend(fontsize=6)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig7_learned_vs_random.png'), facecolor='white')
    print("  Saved fig7_learned_vs_random.png")
    plt.close()

    # ================================================================
    # FIGURE D: Per-class importance heatmaps (4 contrasting classes)
    # ================================================================
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.0))
    target_classes = ['Airport', 'Desert', 'Harbor', 'Farmland']
    # Map class names to indices
    class_names = sorted(os.listdir(os.path.join("./data", "AID")))
    class_names = [c for c in class_names if os.path.isdir(os.path.join("./data", "AID", c))]
    c2i = {c:i for i,c in enumerate(class_names)}

    for ci, cls in enumerate(target_classes):
        if cls not in c2i:
            continue
        cls_idx = c2i[cls]
        cls_mask = all_labels == cls_idx
        cls_imp = all_importance[cls_mask]  # (N_cls, 8, 8)

        if len(cls_imp) == 0:
            continue

        # Top: mean importance for this class
        ax = axes[0, ci]
        mean_map = cls_imp.mean(axis=0)
        im = ax.imshow(mean_map, cmap='RdYlGn', vmin=0, vmax=1, aspect='equal')
        ax.set_title(cls, fontsize=8, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        if ci == 0:
            ax.set_ylabel('Mean', fontsize=7)

        # Bottom: variance for this class
        ax = axes[1, ci]
        var_map = cls_imp.var(axis=0)
        im = ax.imshow(var_map, cmap='hot', aspect='equal')
        ax.set_xticks([]); ax.set_yticks([])
        if ci == 0:
            ax.set_ylabel('Variance', fontsize=7)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig8_class_importance.png'), facecolor='white')
    print("  Saved fig8_class_importance.png")
    plt.close()

    # ================================================================
    # Print summary stats for paper
    # ================================================================
    print("\n" + "="*60)
    print("SUMMARY FOR PAPER")
    print("="*60)
    print(f"\nPer-block importance statistics (over {len(imp_flat)} images):")
    print(f"  Mean of block means: {per_block_mean.mean():.4f}")
    print(f"  Std of block means:  {per_block_mean.std():.4f}")
    print(f"  Mean per-block variance: {per_block_var.mean():.6f}")
    print(f"  Mean per-image variance: {per_image_var.mean():.6f}")
    print(f"  Blocks always selected (mean>0.9): {always_high}/64")
    print(f"  Blocks always dropped (mean<0.1): {always_low}/64")
    print(f"  Content-adaptive blocks (0.1<mean<0.9): {64 - always_high - always_low}/64")

    print(f"\nRate vs Feature MSE:")
    for r in rates:
        print(f"  ρ={r:.2f}: MSE={np.mean(all_mse_rates[r]):.4f} ± {np.std(all_mse_rates[r]):.4f}")

    print(f"\nRate vs Confidence:")
    for r in rates:
        print(f"  ρ={r:.2f}: Conf={np.mean(all_conf_rates[r]):.4f} ± {np.std(all_conf_rates[r]):.4f}")

    print(f"\nLearned vs Random (ρ=0.75):")
    print(f"  Learned confidence: {np.mean(all_confs_learned):.4f} ± {np.std(all_confs_learned):.4f}")
    print(f"  Random confidence:  {np.mean(all_confs_random):.4f} ± {np.std(all_confs_random):.4f}")
    print(f"  Gap: {np.mean(all_confs_learned) - np.mean(all_confs_random):.4f}")

    # Per-class analysis
    print(f"\nPer-class importance spread:")
    for cls in target_classes:
        if cls not in c2i: continue
        cls_idx = c2i[cls]
        cls_imp = all_importance[all_labels == cls_idx]
        if len(cls_imp) == 0: continue
        cls_var = cls_imp.reshape(len(cls_imp), -1).var(axis=1).mean()
        print(f"  {cls:12s}: mean within-image var = {cls_var:.6f}, n={len(cls_imp)}")

    print(f"\n✅ All figures saved to {OUT_DIR}")
