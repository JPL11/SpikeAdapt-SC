# ============================================================================
# ENTROPY DIAGNOSTIC — LAYER3 SPLIT (8×8 = 64 blocks)
# ============================================================================
# Adapted from diagnose_entropy.py for the Layer3 model
# Tests same hypotheses on (1024, 8, 8) features
# ============================================================================

import os, json, math
import numpy as np
from tqdm import tqdm

os.environ['PYTHONUNBUFFERED'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

L3_SNAP = "./snapshots_layer3/"
BB_SNAP = "./snapshots_spikeadapt/"
EVAL_DIR = "./eval_results/"
os.makedirs(EVAL_DIR, exist_ok=True)


# ############################################################################
# MODEL DEFINITIONS (Layer3 versions)
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
        grad_th = -(grad_output * sg).sum() if ctx.th_needs_grad else None
        return grad_output * sg, grad_th

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

class BSC_Channel(nn.Module):
    def forward(self, x, ber):
        if ber <= 0: return x
        flip = (torch.rand_like(x.float()) < ber).float()
        x_n = (x + flip) % 2
        return x + (x_n - x).detach() if self.training else x_n

class ResNet50Front_L3(nn.Module):
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet50(weights=None)
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = r.bn1; self.relu = r.relu; self.maxpool = nn.Identity()
        self.layer1 = r.layer1; self.layer2 = r.layer2; self.layer3 = r.layer3
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        return self.layer3(self.layer2(self.layer1(self.maxpool(x))))

class ResNet50Back_L3(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        r = torchvision.models.resnet50(weights=None)
        self.layer4 = r.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)
    def forward(self, x):
        return self.fc(torch.flatten(self.avgpool(self.layer4(x)), 1))

class SpikeRateEntropyEstimator(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
    def forward(self, all_S2):
        stacked = torch.stack(all_S2, dim=0)
        fr = stacked.mean(dim=0).mean(dim=1)
        f = fr.clamp(self.eps, 1.0 - self.eps)
        return -f * torch.log2(f) - (1 - f) * torch.log2(1 - f), fr

class BlockMask(nn.Module):
    def __init__(self, eta=0.5, temperature=0.1):
        super().__init__()
        self.eta = eta; self.temperature = temperature
    def forward(self, ent, training=True):
        if training:
            soft = torch.sigmoid((ent - self.eta) / self.temperature)
            hard = (ent >= self.eta).float()
            mask = hard + (soft - soft.detach())
        else:
            mask = (ent >= self.eta).float()
        mask = mask.unsqueeze(1)
        return mask, mask.mean()
    def apply_mask(self, x, mask): return x * mask

class Encoder(nn.Module):
    def __init__(self, C_in=1024, C1=256, C2=128):
        super().__init__()
        self.conv1 = nn.Conv2d(C_in, C1, 3, 1, 1); self.bn1 = nn.BatchNorm2d(C1); self.if1 = IFNeuron()
        self.conv2 = nn.Conv2d(C1, C2, 3, 1, 1); self.bn2 = nn.BatchNorm2d(C2); self.if2 = IFNeuron()
    def forward(self, F, m1=None, m2=None):
        s1, m1 = self.if1(self.bn1(self.conv1(F)), m1)
        s2, m2 = self.if2(self.bn2(self.conv2(s1)), m2)
        return s2, m1, m2

class Decoder(nn.Module):
    def __init__(self, C_out=1024, C1=256, C2=128, T=8):
        super().__init__()
        self.T = T
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
    def __init__(self, C_in=1024, C1=256, C2=128, T=8, eta=0.5, temperature=0.1):
        super().__init__()
        self.T = T
        self.encoder = Encoder(C_in, C1, C2)
        self.entropy_est = SpikeRateEntropyEstimator()
        self.block_mask = BlockMask(eta, temperature)
        self.decoder = Decoder(C_in, C1, C2, T)
        self.channel = BSC_Channel()
    def forward(self, feat, bit_error_rate=0.0, eta_override=None):
        all_S2, m1, m2 = [], None, None
        for t in range(self.T):
            s2, m1, m2 = self.encoder(feat, m1, m2)
            all_S2.append(s2)
        ent, fr = self.entropy_est(all_S2)
        if eta_override is not None:
            old = self.block_mask.eta; self.block_mask.eta = eta_override
            mask, tx = self.block_mask(ent, training=False)
            self.block_mask.eta = old
        else:
            mask, tx = self.block_mask(ent, training=self.training)
        recv = [self.channel(self.block_mask.apply_mask(all_S2[t], mask), bit_error_rate)
                for t in range(self.T)]
        Fp = self.decoder(recv, mask)
        return Fp, all_S2, tx, {'tx_rate': tx.item(), 'entropy_map': ent,
                                 'firing_rate': fr, 'mask': mask}


# ############################################################################
# LOAD MODELS
# ############################################################################

print("Loading Layer3 models...")
test_tf = T.Compose([T.ToTensor(), T.Normalize((0.5071,.4867,.4408),(.2675,.2565,.2761))])
test_ds = torchvision.datasets.CIFAR100("./data", False, download=True, transform=test_tf)
test_loader = DataLoader(test_ds, 64, shuffle=False, num_workers=4, pin_memory=True)

front = ResNet50Front_L3().to(device)
back = ResNet50Back_L3(100).to(device)

bb_state = torch.load(os.path.join(BB_SNAP, "backbone_best.pth"), map_location=device)
front_keys = ['conv1.', 'bn1.', 'layer1.', 'layer2.', 'layer3.']
back_keys = ['layer4.', 'fc.']
front.load_state_dict({k: v for k, v in bb_state.items() if any(k.startswith(p) for p in front_keys)}, strict=False)
back.load_state_dict({k: v for k, v in bb_state.items() if any(k.startswith(p) for p in back_keys)}, strict=False)
front.eval()

spikeadapt = SpikeAdaptSC(C_in=1024, C1=256, C2=128, T=8, eta=0.5, temperature=0.1).to(device)

s3f = sorted([f for f in os.listdir(L3_SNAP) if f.startswith("s3_best_")])
if s3f:
    ckpt = torch.load(os.path.join(L3_SNAP, s3f[-1]), map_location=device)
    spikeadapt.load_state_dict(ckpt['spikeadapt'])
    back.load_state_dict(ckpt['back'])
    print(f"  Loaded: {s3f[-1]}")
spikeadapt.eval()
print("Models loaded.\n")

H2, W2 = 8, 8  # Layer3 spatial dims


# ############################################################################
# DIAGNOSTIC 1: ENTROPY DISTRIBUTION & MASK UNIQUENESS
# ############################################################################
print("=" * 70)
print("DIAGNOSTIC 1: Entropy Distribution (Layer3, 8×8)")
print("=" * 70)

all_entropies = []
all_firing_rates = []
per_image_tx = []
mask_hashes = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Collecting stats"):
        images = images.to(device)
        feat = front(images)
        _, all_S2, _, stats = spikeadapt(feat, bit_error_rate=0.0)
        ent = stats['entropy_map']
        mask = stats['mask']
        all_entropies.append(ent.cpu().numpy().flatten())
        all_firing_rates.append(stats['firing_rate'].cpu().numpy().flatten())
        for i in range(ent.shape[0]):
            per_image_tx.append((ent[i] >= 0.5).float().mean().item())
            mask_hashes.append(mask[i, 0].cpu().numpy().tobytes())

all_entropies = np.concatenate(all_entropies)
all_firing_rates = np.concatenate(all_firing_rates)
per_image_tx = np.array(per_image_tx)
unique_masks = len(set(mask_hashes))

print(f"\n  Entropy: mean={all_entropies.mean():.4f}, std={all_entropies.std():.4f}")
print(f"    >= 0.5: {(all_entropies >= 0.5).mean()*100:.1f}%")
print(f"    >= 0.9: {(all_entropies >= 0.9).mean()*100:.1f}%")
print(f"  Firing rate: mean={all_firing_rates.mean():.4f}, std={all_firing_rates.std():.4f}")
print(f"\n  Per-image tx_rate: mean={per_image_tx.mean():.4f}, std={per_image_tx.std():.4f}")
print(f"    Min={per_image_tx.min():.4f}, Max={per_image_tx.max():.4f}")
print(f"    Unique rates: {len(np.unique(np.round(per_image_tx, 4)))}")
print(f"\n  Mask uniqueness: {unique_masks} / {len(mask_hashes)} images")
if unique_masks < 50:
    print(f"    ⚠️ Still largely static ({unique_masks} masks)")
else:
    print(f"    ✓ Content-adaptive ({unique_masks} distinct masks)")


# ############################################################################
# DIAGNOSTIC 2: GRADIENT-BASED IMPORTANCE (ORACLE)
# ############################################################################
print("\n" + "=" * 70)
print("DIAGNOSTIC 2: Gradient-Based Importance (Oracle) — 8×8")
print("=" * 70)

criterion = nn.CrossEntropyLoss()
grad_importance_all = []
entropy_values_all = []
N_GRAD = 300

spikeadapt.eval()
count = 0
for images, labels in tqdm(test_loader, desc="Computing importance"):
    images, labels = images.to(device), labels.to(device)
    for i in range(images.shape[0]):
        if count >= N_GRAD: break
        feat = front(images[i:i+1])
        all_S2_i, m1, m2 = [], None, None
        for t in range(spikeadapt.T):
            s2, m1, m2 = spikeadapt.encoder(feat, m1, m2)
            all_S2_i.append(s2.detach())
        ent_i, _ = spikeadapt.entropy_est(all_S2_i)
        entropy_values_all.append(ent_i[0].cpu().numpy())

        # Base prediction
        ones_mask = torch.ones(1, 1, H2, W2, device=device)
        base_fp = spikeadapt.decoder([s.clone() for s in all_S2_i], ones_mask)
        base_loss = criterion(back(base_fp), labels[i:i+1]).item()

        imp = np.zeros((H2, W2))
        for h in range(H2):
            for w in range(W2):
                mod_recv = [s.clone() for s in all_S2_i]
                for s in mod_recv: s[:, :, h, w] = 0
                mod_fp = spikeadapt.decoder(mod_recv, ones_mask)
                imp[h, w] = criterion(back(mod_fp), labels[i:i+1]).item() - base_loss
        grad_importance_all.append(imp)
        count += 1
    if count >= N_GRAD: break

grad_importance_all = np.array(grad_importance_all)
entropy_values_all = np.array(entropy_values_all)

flat_ent = entropy_values_all.flatten()
flat_imp = grad_importance_all.flatten()
correlation = np.corrcoef(flat_ent, flat_imp)[0, 1]

print(f"\n  Entropy-Importance correlation: {correlation:.4f}")
if abs(correlation) < 0.1:
    print(f"    ⚠️ Near-zero: entropy unrelated to task importance")
elif correlation < -0.1:
    print(f"    ⚠️ Negative: entropy drops important blocks")
elif correlation > 0.1:
    print(f"    ✓ Positive: entropy captures some importance (r={correlation:.3f})")

kept = (entropy_values_all >= 0.5)
dropped = ~kept
kept_imp = flat_imp[kept.flatten()]
dropped_imp = flat_imp[dropped.flatten()]
print(f"\n  KEPT (entropy >= 0.5): {kept.sum()} blocks, avg importance = {kept_imp.mean():.4f}")
print(f"  DROPPED (< 0.5):      {dropped.sum()} blocks, avg importance = {dropped_imp.mean():.4f}")
if dropped_imp.mean() > kept_imp.mean():
    print(f"    ⚠️ Dropped blocks more important!")
else:
    print(f"    ✓ Keeping the right blocks")


# ############################################################################
# DIAGNOSTIC 3: MASKING STRATEGY COMPARISON
# ############################################################################
print("\n" + "=" * 70)
print("DIAGNOSTIC 3: Masking Strategy Comparison (8×8)")
print("=" * 70)

def eval_strategy(strategy, keep_rate=0.75, n_batches=80):
    correct, total = 0, 0
    batch_count = 0
    with torch.no_grad():
        for images, labels in test_loader:
            if batch_count >= n_batches: break
            images, labels = images.to(device), labels.to(device)
            feat = front(images)
            all_S2, m1, m2 = [], None, None
            for t in range(spikeadapt.T):
                s2, m1, m2 = spikeadapt.encoder(feat, m1, m2)
                all_S2.append(s2)
            B, C2, H, W = all_S2[0].shape

            if strategy == 'all':
                mask = torch.ones(B, 1, H, W, device=device)
            elif strategy == 'entropy':
                ent, _ = spikeadapt.entropy_est(all_S2)
                k = max(1, int(keep_rate * H * W))
                _, idx = ent.view(B, -1).topk(k, dim=1)
                mask = torch.zeros(B, H*W, device=device)
                mask.scatter_(1, idx, 1.0)
                mask = mask.view(B, H, W).unsqueeze(1)
            elif strategy == 'anti_entropy':
                ent, _ = spikeadapt.entropy_est(all_S2)
                k = max(1, int(keep_rate * H * W))
                _, idx = (-ent.view(B, -1)).topk(k, dim=1)
                mask = torch.zeros(B, H*W, device=device)
                mask.scatter_(1, idx, 1.0)
                mask = mask.view(B, H, W).unsqueeze(1)
            elif strategy == 'oracle':
                # Use precomputed importance (only for first N_GRAD images)
                # For fair comparison, just use gradient to pick top-k
                ent, _ = spikeadapt.entropy_est(all_S2)
                mask = torch.ones(B, 1, H, W, device=device)
                # Fall back to entropy for oracle since we can't easily do per-image grad here
            elif strategy == 'random':
                mask = (torch.rand(B, 1, H, W, device=device) < keep_rate).float()
            else:
                mask = torch.ones(B, 1, H, W, device=device)

            recv = [all_S2[t] * mask for t in range(spikeadapt.T)]
            Fp = spikeadapt.decoder(recv, mask)
            correct += back(Fp).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
            batch_count += 1
    return 100.*correct/total if total > 0 else 0.0

for rate in [0.75, 0.50]:
    print(f"\n  Keep rate = {rate*100:.0f}%:")
    for name, strat in [('All blocks', 'all'), ('Entropy top-k', 'entropy'),
                         ('Anti-entropy top-k', 'anti_entropy'), ('Random', 'random')]:
        accs = [eval_strategy(strat, rate) for _ in (range(3) if strat == 'random' else range(1))]
        print(f"    {name:<22s}: {np.mean(accs):.2f}%")


# ############################################################################
# DIAGNOSTIC 4: SPATIAL PATTERNS
# ############################################################################
print("\n" + "=" * 70)
print("DIAGNOSTIC 4: Spatial Patterns (8×8)")
print("=" * 70)

avg_entropy = entropy_values_all.mean(axis=0)
avg_importance = grad_importance_all.mean(axis=0)

print(f"\n  Average entropy map (8×8):")
for h in range(H2):
    print("    " + "  ".join(f"{avg_entropy[h,w]:.2f}" for w in range(W2)))

print(f"\n  Average importance map (8×8):")
for h in range(H2):
    print("    " + "  ".join(f"{avg_importance[h,w]:+.3f}" for w in range(W2)))

# Rank correlation between avg patterns
from scipy import stats as sp_stats
rank_corr, rank_p = sp_stats.spearmanr(avg_entropy.flatten(), avg_importance.flatten())
print(f"\n  Spatial pattern Spearman correlation: ρ={rank_corr:.4f} (p={rank_p:.4f})")


# ############################################################################
# CREATE FIGURE
# ############################################################################
print("\n" + "=" * 70)
print("Creating diagnostic figure...")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Entropy histogram
axes[0, 0].hist(all_entropies, bins=50, color='steelblue', alpha=0.7, edgecolor='navy')
axes[0, 0].axvline(0.5, color='red', ls='--', lw=2, label='η=0.5')
axes[0, 0].set_xlabel('Block Entropy'); axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title(f'Entropy Distribution (8×8)\n{(all_entropies>=0.5).mean()*100:.0f}% above η=0.5')
axes[0, 0].legend()

# Per-image tx histogram
axes[0, 1].hist(per_image_tx, bins=40, color='mediumpurple', alpha=0.7, edgecolor='indigo')
axes[0, 1].set_xlabel('Per-Image Tx Rate'); axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title(f'Tx Rate Distribution\nstd={per_image_tx.std():.4f}, {unique_masks} unique masks')

# Entropy vs importance scatter
n_pts = min(5000, len(flat_ent))
idx = np.random.choice(len(flat_ent), n_pts, replace=False)
axes[0, 2].scatter(flat_ent[idx], flat_imp[idx], alpha=0.15, s=8, color='teal')
axes[0, 2].axvline(0.5, color='red', ls='--', alpha=0.5)
axes[0, 2].set_xlabel('Block Entropy'); axes[0, 2].set_ylabel('Importance (Δloss)')
axes[0, 2].set_title(f'Entropy vs Importance\nr={correlation:.3f}')

# Average entropy map
im1 = axes[1, 0].imshow(avg_entropy, cmap='YlOrRd', vmin=0, vmax=1)
axes[1, 0].set_title('Avg Entropy (8×8)')
plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)

# Average importance map
im2 = axes[1, 1].imshow(avg_importance, cmap='YlOrRd')
axes[1, 1].set_title('Avg Importance (8×8)')
plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)

# Comparison: L4 vs L3
comp_data = {
    '4×4 (L4)': {'unique_masks': 2, 'correlation': -0.142, 'tx_std': 0.0065},
    '8×8 (L3)': {'unique_masks': unique_masks, 'correlation': correlation, 'tx_std': per_image_tx.std()},
}
labels = list(comp_data.keys())
metrics = ['unique_masks', 'correlation', 'tx_std']
x = np.arange(len(metrics)); w = 0.35
for i, label in enumerate(labels):
    vals = [comp_data[label][m] for m in metrics]
    # Normalize for visualization
    norm_vals = [min(v / 100, 1) if m == 'unique_masks' else abs(v) for v, m in zip(vals, metrics)]
    axes[1, 2].bar(x + i*w - w/2, vals, w, label=label, alpha=0.8)
axes[1, 2].set_xticks(x)
axes[1, 2].set_xticklabels(['Unique Masks', 'Ent-Imp Corr', 'Tx Rate Std'])
axes[1, 2].legend()
axes[1, 2].set_title('L4 (4×4) vs L3 (8×8) Comparison')

fig.suptitle('Layer3 (8×8) Entropy Diagnostic', fontsize=16, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(EVAL_DIR, "entropy_diagnostic_L3.png"), dpi=150)
plt.close()
print(f"  Saved: {EVAL_DIR}entropy_diagnostic_L3.png")


# ############################################################################
# SAVE & VERDICT
# ############################################################################
results = {
    'entropy': {'mean': float(all_entropies.mean()), 'std': float(all_entropies.std())},
    'unique_masks': unique_masks,
    'total_images': len(mask_hashes),
    'correlation': float(correlation),
    'kept_importance': float(kept_imp.mean()),
    'dropped_importance': float(dropped_imp.mean()),
    'tx_rate_std': float(per_image_tx.std()),
    'spatial_rank_corr': float(rank_corr),
}
with open(os.path.join(EVAL_DIR, "entropy_diagnostic_L3.json"), 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("FINAL VERDICT: Layer3 (8×8) vs Layer4 (4×4)")
print("=" * 70)
print(f"\n  {'Metric':<30} {'L4 (4×4)':<15} {'L3 (8×8)':<15} {'Better?'}")
print(f"  {'-'*75}")
print(f"  {'Unique masks':<30} {'2':<15} {unique_masks:<15} {'✓ L3' if unique_masks > 2 else '✗ same'}")
print(f"  {'Ent-Imp correlation':<30} {'-0.142':<15} {correlation:<15.4f} {'✓ L3' if correlation > -0.142 else '✗ L4'}")
print(f"  {'Tx rate std':<30} {'0.0065':<15} {per_image_tx.std():<15.4f} {'✓ L3' if per_image_tx.std() > 0.0065 else '✗ L4'}")
print(f"  {'Kept block importance':<30} {'-0.372':<15} {kept_imp.mean():<15.4f}")
print(f"  {'Dropped block importance':<30} {'-0.008':<15} {dropped_imp.mean():<15.4f}")

print(f"\n{'='*70}")
print("DIAGNOSTIC COMPLETE")
print(f"{'='*70}")
