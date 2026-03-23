"""Generate ALL professor-requested metrics and graphs.

Output: figures in paper/figures/ and updated analysis .tex
Metrics:
  1. Mask area vs MSE
  2. Confidence vs Mask area  
  3. Random vs Learned (energy savings + confidence)
  4. Per-block entropy distribution
  5. Sorted importance rankings (5 example images)
  6. Per-block Gini coefficient distribution
  7. Kept-vs-dropped margin histogram
"""

import os, sys, random, math, json
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
OUT_DIR = "./paper/figures/"
os.makedirs(OUT_DIR, exist_ok=True)

# ========= Model classes (same as before) =========
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

T_STEPS = 8

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


if __name__ == "__main__":
    print(f"Device: {device}")
    front,back,enc,imp_s,ch,dec = load_all()
    tf=T.Compose([T.Resize(256),T.CenterCrop(224),T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])
    ds=AIDDataset("./data",transform=tf)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)
    print(f"Test set: {len(ds)} images")

    # ============================================================
    # Collect everything
    # ============================================================
    print("\n=== Collecting data ===")
    all_scores = []      # (N, 64)
    all_labels = []
    all_S2_batches = []
    all_feats = []

    # Per-rate data (learned and random)
    rates = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    learned_mse_by_rate = {r: [] for r in rates}
    learned_conf_by_rate = {r: [] for r in rates}
    random_mse_by_rate = {r: [] for r in rates}
    random_conf_by_rate = {r: [] for r in rates}

    # Firing rate data for energy
    learned_firing_by_rate = {r: [] for r in rates}
    random_firing_by_rate = {r: [] for r in rates}

    with torch.no_grad():
        for bi, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device)
            feat = front(imgs)
            B = imgs.shape[0]

            # Encode
            S2, m1, m2 = [], None, None
            for t in range(T_STEPS):
                _, s2, m1, m2 = enc(feat, m1, m2)
                S2.append(s2)

            importance = imp_s(S2)  # (B, 8, 8)
            all_scores.append(importance.cpu().numpy().reshape(B, -1))
            all_labels.extend(labels.numpy())
            all_S2_batches.append([s.cpu() for s in S2])
            all_feats.append(feat.cpu())

            # For each rate: compute MSE, confidence, firing rate (learned + random)
            for rate in rates:
                k = max(1, int(rate * 64))
                flat = importance.view(B, -1)

                # LEARNED mask
                _, idx = flat.topk(k, 1)
                mask_l = torch.zeros_like(flat); mask_l.scatter_(1, idx, 1.0)
                mask_l = mask_l.view(B, 8, 8).unsqueeze(1)
                recv_l = [S2[t] * mask_l for t in range(T_STEPS)]
                Fp_l = dec(recv_l, mask_l)
                mse_l = F.mse_loss(Fp_l, feat, reduction='none').mean(dim=[1,2,3])
                logits_l = back(Fp_l)
                conf_l = F.softmax(logits_l, 1).max(1)[0]
                # Firing rate of transmitted spikes
                tx_spikes = torch.stack([S2[t] * mask_l for t in range(T_STEPS)], 0)
                fire_l = tx_spikes.mean().item()

                learned_mse_by_rate[rate].extend(mse_l.cpu().numpy())
                learned_conf_by_rate[rate].extend(conf_l.cpu().numpy())
                learned_firing_by_rate[rate].append(fire_l)

                # RANDOM mask
                rand_flat = torch.rand(B, 64, device=device)
                _, ridx = rand_flat.topk(k, 1)
                mask_r = torch.zeros(B, 64, device=device); mask_r.scatter_(1, ridx, 1.0)
                mask_r = mask_r.view(B, 8, 8).unsqueeze(1)
                recv_r = [S2[t] * mask_r for t in range(T_STEPS)]
                Fp_r = dec(recv_r, mask_r)
                mse_r = F.mse_loss(Fp_r, feat, reduction='none').mean(dim=[1,2,3])
                conf_r = F.softmax(back(Fp_r), 1).max(1)[0]
                tx_spikes_r = torch.stack([S2[t] * mask_r for t in range(T_STEPS)], 0)
                fire_r = tx_spikes_r.mean().item()

                random_mse_by_rate[rate].extend(mse_r.cpu().numpy())
                random_conf_by_rate[rate].extend(conf_r.cpu().numpy())
                random_firing_by_rate[rate].append(fire_r)

            if (bi+1) % 20 == 0:
                print(f"  Batch {bi+1}/{len(loader)}")

    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.array(all_labels)
    N = len(all_labels)
    print(f"  Total: {N} images")

    # ============================================================
    # Compute per-image statistics
    # ============================================================
    per_image_var = np.var(all_scores, axis=1)
    per_image_entropy = np.array([
        -np.sum((s/s.sum()) * np.log(s/s.sum() + 1e-12)) for s in all_scores
    ])
    per_image_gini = np.array([
        (2*np.sum(np.arange(1,65)*np.sort(s)) - 65*np.sum(s)) / (64*np.sum(s)+1e-12)
        for s in all_scores
    ])
    # Kept-dropped margin at each rate
    margins_by_rate = {}
    for rate in rates:
        k = max(1, int(rate * 64))
        ms = []
        for i in range(N):
            si = np.argsort(all_scores[i])[::-1]
            ms.append(all_scores[i][si[:k]].mean() - all_scores[i][si[k:]].mean() if k < 64 else 0)
        margins_by_rate[rate] = np.array(ms)

    # ============================================================
    # FIGURE 1: Mask Area vs MSE (learned vs random)
    # ============================================================
    print("\n=== Generating figures ===")

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))

    # Panel 1: Mask area vs MSE
    ax = axes[0]
    lm = [np.mean(learned_mse_by_rate[r]) for r in rates]
    ls = [np.std(learned_mse_by_rate[r]) for r in rates]
    rm = [np.mean(random_mse_by_rate[r]) for r in rates]
    rs = [np.std(random_mse_by_rate[r]) for r in rates]
    pcts = [r*100 for r in rates]
    ax.plot(pcts, lm, 'o-', color='#1E88E5', linewidth=1.2, markersize=4, label='Learned mask')
    ax.fill_between(pcts, np.array(lm)-np.array(ls), np.array(lm)+np.array(ls), alpha=0.15, color='#1E88E5')
    ax.plot(pcts, rm, 's--', color='#E53935', linewidth=1.0, markersize=3.5, label='Random mask')
    ax.fill_between(pcts, np.array(rm)-np.array(rs), np.array(rm)+np.array(rs), alpha=0.12, color='#E53935')
    ax.set_xlabel('Transmission Rate (%)')
    ax.set_ylabel('Feature MSE')
    ax.set_title('(a) Mask Area vs. Feature MSE')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # Panel 2: Mask area vs Confidence
    ax = axes[1]
    lc = [np.mean(learned_conf_by_rate[r]) for r in rates]
    lcs = [np.std(learned_conf_by_rate[r]) for r in rates]
    rc = [np.mean(random_conf_by_rate[r]) for r in rates]
    rcs = [np.std(random_conf_by_rate[r]) for r in rates]
    ax.plot(pcts, lc, 'o-', color='#1E88E5', linewidth=1.2, markersize=4, label='Learned mask')
    ax.fill_between(pcts, np.array(lc)-np.array(lcs), np.array(lc)+np.array(lcs), alpha=0.15, color='#1E88E5')
    ax.plot(pcts, rc, 's--', color='#E53935', linewidth=1.0, markersize=3.5, label='Random mask')
    ax.fill_between(pcts, np.array(rc)-np.array(rcs), np.array(rc)+np.array(rcs), alpha=0.12, color='#E53935')
    ax.set_xlabel('Transmission Rate (%)')
    ax.set_ylabel('Classification Confidence')
    ax.set_title('(b) Mask Area vs. Confidence')
    ax.legend(fontsize=6, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.02])

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'prof_mse_conf_vs_rate.png'), facecolor='white')
    print("  Saved prof_mse_conf_vs_rate.png")
    plt.close()

    # ============================================================
    # FIGURE 2: Random vs Learned — Energy savings + Confidence
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))

    # Energy savings estimation from firing rates
    MAC_ENERGY = 4.6  # pJ
    SYNOP_ENERGY = 0.9  # pJ
    ax = axes[0]
    l_fire = [np.mean(learned_firing_by_rate[r]) for r in rates]
    r_fire = [np.mean(random_firing_by_rate[r]) for r in rates]
    # Energy savings = 1 - (synop_cost / mac_cost) = 1 - (firing_rate * SYNOP / MAC)
    l_energy = [(1 - f * SYNOP_ENERGY / MAC_ENERGY) * 100 for f in l_fire]
    r_energy = [(1 - f * SYNOP_ENERGY / MAC_ENERGY) * 100 for f in r_fire]
    ax.plot(pcts, l_energy, 'o-', color='#43A047', linewidth=1.2, markersize=4, label='Learned mask')
    ax.plot(pcts, r_energy, 's--', color='#FB8C00', linewidth=1.0, markersize=3.5, label='Random mask')
    ax.set_xlabel('Transmission Rate (%)')
    ax.set_ylabel('Estimated Energy Savings (%)')
    ax.set_title('(a) Energy Savings vs. Rate')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # Confidence gap (learned - random)
    ax = axes[1]
    gap = [np.mean(learned_conf_by_rate[r]) - np.mean(random_conf_by_rate[r]) for r in rates]
    colors = ['#E53935' if g < 0 else '#43A047' for g in gap]
    ax.bar(pcts, gap, width=10, color=colors, edgecolor='white', linewidth=0.5, alpha=0.85)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('Transmission Rate (%)')
    ax.set_ylabel('Confidence Gap (Learned − Random)')
    ax.set_title('(b) Learned vs. Random Confidence Gap')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'prof_energy_and_gap.png'), facecolor='white')
    print("  Saved prof_energy_and_gap.png")
    plt.close()

    # ============================================================
    # FIGURE 3: Per-block score distributions
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.4))

    # Panel 1: Sorted importance rankings for 5 example images
    ax = axes[0]
    np.random.seed(42)
    ex_idx = np.random.choice(N, 5, replace=False)
    colors_ex = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA']
    class_names = sorted(os.listdir(os.path.join("./data", "AID")))
    class_names = [c for c in class_names if os.path.isdir(os.path.join("./data", "AID", c))]
    for ei, idx in enumerate(ex_idx):
        scores_sorted = np.sort(all_scores[idx])[::-1]
        cls_name = class_names[all_labels[idx]] if all_labels[idx] < len(class_names) else '?'
        ax.plot(range(64), scores_sorted, '-', color=colors_ex[ei], linewidth=1.0,
                alpha=0.8, label=f'{cls_name}')
    # Reference: uniform
    ax.axhline(1/64, color='gray', linestyle=':', linewidth=0.6, alpha=0.7, label='Uniform')
    ax.set_xlabel('Block rank (sorted)')
    ax.set_ylabel('Importance score')
    ax.set_title('(a) Sorted Score Profiles')
    ax.legend(fontsize=5, ncol=2)
    ax.grid(True, alpha=0.2)

    # Panel 2: Entropy histogram
    ax = axes[1]
    ax.hist(per_image_entropy, bins=40, color='#7E57C2', edgecolor='white', linewidth=0.3, alpha=0.85)
    ax.axvline(per_image_entropy.mean(), color='#333', linestyle='--', linewidth=1,
               label=f'Mean={per_image_entropy.mean():.3f}')
    ax.axvline(np.log(64), color='#E53935', linestyle=':', linewidth=1,
               label=f'Max (uniform)={np.log(64):.3f}')
    ax.set_xlabel('Score Entropy (nats)')
    ax.set_ylabel('Count')
    ax.set_title('(b) Score Entropy')
    ax.legend(fontsize=5.5)

    # Panel 3: Gini histogram
    ax = axes[2]
    ax.hist(per_image_gini, bins=40, color='#FF7043', edgecolor='white', linewidth=0.3, alpha=0.85)
    ax.axvline(per_image_gini.mean(), color='#333', linestyle='--', linewidth=1,
               label=f'Mean={per_image_gini.mean():.4f}')
    ax.set_xlabel('Gini Coefficient')
    ax.set_ylabel('Count')
    ax.set_title('(c) Score Concentration (Gini)')
    ax.legend(fontsize=6)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'prof_score_distributions.png'), facecolor='white')
    print("  Saved prof_score_distributions.png")
    plt.close()

    # ============================================================
    # FIGURE 4: Kept-vs-dropped margin at each rate
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5))

    # Panel 1: Mean margin vs rate
    ax = axes[0]
    margin_means = [margins_by_rate[r].mean() for r in rates[:-1]]  # skip 1.0 (no margin)
    margin_stds = [margins_by_rate[r].std() for r in rates[:-1]]
    ax.errorbar([r*100 for r in rates[:-1]], margin_means, yerr=margin_stds,
                fmt='o-', color='#1E88E5', linewidth=1.2, markersize=4, capsize=3)
    ax.set_xlabel('Transmission Rate (%)')
    ax.set_ylabel('Score Gap (kept − dropped)')
    ax.set_title('(a) Kept-Dropped Margin vs. Rate')
    ax.grid(True, alpha=0.3)

    # Panel 2: Margin histogram at ρ=0.75
    ax = axes[1]
    ax.hist(margins_by_rate[0.75], bins=40, color='#66BB6A', edgecolor='white', linewidth=0.3, alpha=0.85)
    ax.axvline(margins_by_rate[0.75].mean(), color='#333', linestyle='--', linewidth=1,
               label=f'Mean={margins_by_rate[0.75].mean():.4f}')
    ax.set_xlabel(r'Score Gap at $\rho=0.75$')
    ax.set_ylabel('Count')
    ax.set_title(r'(b) Margin Distribution ($\rho=0.75$)')
    ax.legend(fontsize=6)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'prof_margin_analysis.png'), facecolor='white')
    print("  Saved prof_margin_analysis.png")
    plt.close()

    # ============================================================
    # Print summary for .tex
    # ============================================================
    print("\n" + "="*60)
    print("SUMMARY FOR ANALYSIS TEX")
    print("="*60)
    print(f"\nScore Statistics:")
    print(f"  Per-image variance: {per_image_var.mean():.6f} ± {per_image_var.std():.6f}")
    print(f"  Per-image entropy:  {per_image_entropy.mean():.4f} ± {per_image_entropy.std():.4f}")
    print(f"  Max entropy (uniform): {np.log(64):.4f}")
    print(f"  Entropy ratio: {per_image_entropy.mean()/np.log(64):.4f}")
    print(f"  Per-image Gini:     {per_image_gini.mean():.4f} ± {per_image_gini.std():.4f}")

    print(f"\nMask Area vs MSE (learned/random):")
    for r in rates:
        lm_v = np.mean(learned_mse_by_rate[r])
        rm_v = np.mean(random_mse_by_rate[r])
        print(f"  ρ={r:.3f}: Learned MSE={lm_v:.2f}, Random MSE={rm_v:.2f}, Gap={rm_v-lm_v:.2f}")

    print(f"\nMask Area vs Confidence (learned/random):")
    for r in rates:
        lc_v = np.mean(learned_conf_by_rate[r])
        rc_v = np.mean(random_conf_by_rate[r])
        print(f"  ρ={r:.3f}: Learned={lc_v:.4f}, Random={rc_v:.4f}, Gap={lc_v-rc_v:.4f}")

    print(f"\nEnergy Savings (learned vs random):")
    for r in [0.25, 0.50, 0.75, 1.0]:
        lf_v = np.mean(learned_firing_by_rate[r])
        rf_v = np.mean(random_firing_by_rate[r])
        le = (1 - lf_v * SYNOP_ENERGY / MAC_ENERGY) * 100
        re = (1 - rf_v * SYNOP_ENERGY / MAC_ENERGY) * 100
        print(f"  ρ={r:.2f}: Learned firing={lf_v:.4f} ({le:.1f}%), Random firing={rf_v:.4f} ({re:.1f}%)")

    print(f"\nKept-dropped margin at ρ=0.75:")
    print(f"  {margins_by_rate[0.75].mean():.4f} ± {margins_by_rate[0.75].std():.4f}")
    print(f"  All positive: {(margins_by_rate[0.75] > 0).all()}")

    print(f"\n✅ All professor figures saved to {OUT_DIR}")
