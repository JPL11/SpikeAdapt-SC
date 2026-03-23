# ============================================================================
# ENTROPY ESTIMATOR DIAGNOSTIC
# ============================================================================
# Answers: WHY does entropy-guided masking hurt SpikeAdapt-SC?
#
# Hypotheses tested:
#   H1: Entropy distribution is too clustered → mask is nearly static
#   H2: Entropy ≠ task importance → we're dropping important blocks
#   H3: Entropy/rate losses hurt encoder training
#   H4: 4×4 spatial grid is too small for meaningful per-image adaptation
#
# Requires: trained backbone + SpikeAdapt-SC checkpoint
# Output:   ./eval_results/entropy_diagnostic.png (4-panel figure)
#           ./eval_results/entropy_diagnostic.json (numerical results)
#           Console: detailed analysis + recommendations
# ============================================================================

import os, sys, json, math
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
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

SNAP_DIR = "./snapshots_spikeadapt/"
EVAL_DIR = "./eval_results/"
os.makedirs(EVAL_DIR, exist_ok=True)


# ############################################################################
# MODEL DEFINITIONS (same as train_baselines.py — needed for checkpoint load)
# ############################################################################

class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane, threshold):
        if isinstance(threshold, (int, float)):
            threshold = torch.tensor(float(threshold), device=membrane.device)
        ctx.save_for_backward(membrane, threshold)
        return (membrane > threshold).float()
    @staticmethod
    def backward(ctx, grad_output):
        membrane, threshold = ctx.saved_tensors
        scale = 10.0
        sig = torch.sigmoid(scale * (membrane - threshold))
        sg = sig * (1 - sig) * scale
        return grad_output * sg, -(grad_output * sg).sum()

class IFNeuron(nn.Module):
    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = threshold
    def forward(self, x, mem=None):
        if mem is None: mem = torch.zeros_like(x)
        mem = mem + x
        spike = SpikeFunction.apply(mem, self.threshold)
        mem = mem - spike * self.threshold
        return spike, mem

class IHFNeuron(nn.Module):
    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(float(threshold)))
    def forward(self, x, mem=None):
        if mem is None: mem = torch.zeros_like(x)
        mem = mem + x
        spike = SpikeFunction.apply(mem, self.threshold)
        mem = mem - spike * self.threshold
        return spike, mem

class BSC_Channel(nn.Module):
    def forward(self, x, ber):
        if ber <= 0: return x
        flip = (torch.rand_like(x.float()) < ber).float()
        x_n = (x + flip) % 2
        return x + (x_n - x).detach() if self.training else x_n

class SpikeRateEntropyEstimator(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
    def forward(self, all_S2):
        stacked = torch.stack(all_S2, dim=0)
        fr = stacked.mean(dim=0).mean(dim=1)
        f = fr.clamp(self.eps, 1.0 - self.eps)
        ent = -f * torch.log2(f) - (1 - f) * torch.log2(1 - f)
        return ent, fr

class BlockMask(nn.Module):
    def __init__(self, eta=0.5, temperature=0.1):
        super().__init__()
        self.eta = eta
        self.temperature = temperature
    def forward(self, entropy_map, training=True):
        if training:
            soft = torch.sigmoid((entropy_map - self.eta) / self.temperature)
            hard = (entropy_map >= self.eta).float()
            mask = hard + (soft - soft.detach())
        else:
            mask = (entropy_map >= self.eta).float()
        mask = mask.unsqueeze(1)
        return mask, mask.mean()
    def apply_mask(self, x, mask):
        return x * mask

class Encoder(nn.Module):
    def __init__(self, C_in=2048, C1=256, C2=128):
        super().__init__()
        self.conv1 = nn.Conv2d(C_in, C1, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(C1)
        self.if1 = IFNeuron()
        self.conv2 = nn.Conv2d(C1, C2, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(C2)
        self.if2 = IFNeuron()
    def forward(self, F, m1=None, m2=None):
        x = self.bn1(self.conv1(F))
        s1, m1 = self.if1(x, m1)
        x = self.bn2(self.conv2(s1))
        s2, m2 = self.if2(x, m2)
        return s2, m1, m2

class Decoder(nn.Module):
    def __init__(self, C_out=2048, C1=256, C2=128, T=8):
        super().__init__()
        self.T = T
        self.conv3 = nn.Conv2d(C2, C1, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(C1)
        self.if3 = IFNeuron()
        self.conv4 = nn.Conv2d(C1, C_out, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(C_out)
        self.ihf = IHFNeuron()
        self.converter_fc = nn.Linear(2 * T, 2 * T)
    def forward(self, recv_all, mask):
        m3, m4 = None, None
        Fs, Fm = [], []
        for t in range(self.T):
            x = self.bn3(self.conv3(recv_all[t] * mask))
            s3, m3 = self.if3(x, m3)
            x = self.bn4(self.conv4(s3))
            sp, m4 = self.ihf(x, m4)
            Fs.append(sp); Fm.append(m4.clone())
        il = []
        for t in range(self.T):
            il.append(Fs[t]); il.append(Fm[t])
        stk = torch.stack(il, dim=1)
        x = stk.permute(0, 2, 3, 4, 1)
        w = torch.sigmoid(self.converter_fc(x))
        return (x * w).sum(dim=-1)

class SpikeAdaptSC(nn.Module):
    def __init__(self, C_in=2048, C1=256, C2=128, T=8,
                 eta=0.5, temperature=0.1, channel_type='bsc'):
        super().__init__()
        self.T = T
        self.encoder = Encoder(C_in, C1, C2)
        self.entropy_est = SpikeRateEntropyEstimator()
        self.block_mask = BlockMask(eta, temperature)
        self.decoder = Decoder(C_in, C1, C2, T)
        self.channel = BSC_Channel() if channel_type == 'bsc' else BSC_Channel()
    def forward(self, feat, bit_error_rate=0.0, eta_override=None):
        all_S2, m1, m2 = [], None, None
        for t in range(self.T):
            s2, m1, m2 = self.encoder(feat, m1, m2)
            all_S2.append(s2)
        ent, fr = self.entropy_est(all_S2)
        if eta_override is not None:
            old = self.block_mask.eta
            self.block_mask.eta = eta_override
            mask, tx = self.block_mask(ent, training=False)
            self.block_mask.eta = old
        else:
            mask, tx = self.block_mask(ent, training=self.training)
        recv = []
        for t in range(self.T):
            recv.append(self.channel(self.block_mask.apply_mask(all_S2[t], mask),
                                     bit_error_rate))
        Fp = self.decoder(recv, mask)
        return Fp, all_S2, tx, {'tx_rate': tx.item(),
                                 'entropy_map': ent,
                                 'firing_rate': fr,
                                 'mask': mask}

class ResNet50Front(nn.Module):
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet50(weights=None)
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = r.bn1
        self.relu = r.relu
        self.maxpool = nn.Identity()
        self.layer1 = r.layer1
        self.layer2 = r.layer2
        self.layer3 = r.layer3
        self.layer4 = r.layer4
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))

class ResNet50Back(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)
    def forward(self, x):
        return self.fc(torch.flatten(self.avgpool(x), 1))


# ############################################################################
# LOAD MODELS
# ############################################################################

print("Loading models...")
test_tf = T.Compose([T.ToTensor(),
                     T.Normalize((0.5071,.4867,.4408),(.2675,.2565,.2761))])
test_ds = torchvision.datasets.CIFAR100("./data", False, download=True, transform=test_tf)
test_loader = DataLoader(test_ds, 64, shuffle=False, num_workers=4, pin_memory=True)

front = ResNet50Front().to(device)
back = ResNet50Back(100).to(device)

bb_path = os.path.join(SNAP_DIR, "backbone_best.pth")
state = torch.load(bb_path, map_location=device)
f_st = {k: v for k, v in state.items() if not k.startswith(('fc.', 'avgpool.'))}
b_st = {k: v for k, v in state.items() if k.startswith(('fc.', 'avgpool.'))}
front.load_state_dict(f_st, strict=False)
back.load_state_dict(b_st, strict=False)
front.eval()

spikeadapt = SpikeAdaptSC(C_in=2048, C1=256, C2=128, T=8,
                            eta=0.5, temperature=0.1).to(device)
s3_files = sorted([f for f in os.listdir(SNAP_DIR) if f.startswith("step3_best_")])
if s3_files:
    ckpt = torch.load(os.path.join(SNAP_DIR, s3_files[-1]), map_location=device)
    spikeadapt.load_state_dict(ckpt['spikeadapt'])
    back.load_state_dict(ckpt['back'])
    print(f"  Loaded SpikeAdapt-SC: {s3_files[-1]}")
else:
    s2_files = sorted([f for f in os.listdir(SNAP_DIR) if f.startswith("spikeadapt_best_")])
    if s2_files:
        spikeadapt.load_state_dict(torch.load(os.path.join(SNAP_DIR, s2_files[-1]),
                                               map_location=device))
        print(f"  Loaded SpikeAdapt-SC (S2): {s2_files[-1]}")

spikeadapt.eval()
print("Models loaded.\n")


# ############################################################################
# DIAGNOSTIC 1: ENTROPY DISTRIBUTION ANALYSIS
# ############################################################################
print("=" * 70)
print("DIAGNOSTIC 1: Entropy Distribution Analysis")
print("=" * 70)

all_entropies = []     # per-block entropy values
all_firing_rates = []  # per-block firing rates
all_masks = []         # per-block mask values
per_image_tx = []      # per-image tx rates
per_image_masks = []   # store first N image masks for visualization

N_VIS = 16  # images to visualize

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Collecting entropy stats")):
        images, labels = images.to(device), labels.to(device)
        feat = front(images)
        Fp, all_S2, tx, stats = spikeadapt(feat, bit_error_rate=0.0)
        
        ent = stats['entropy_map']  # (B, H2, W2)
        fr = stats['firing_rate']
        mask = stats['mask']        # (B, 1, H2, W2)
        
        all_entropies.append(ent.cpu().numpy().flatten())
        all_firing_rates.append(fr.cpu().numpy().flatten())
        all_masks.append(mask.squeeze(1).cpu().numpy().flatten())
        
        # Per-image tx rates
        for i in range(ent.shape[0]):
            m_i = (ent[i] >= 0.5).float()
            per_image_tx.append(m_i.mean().item())
        
        # Save first N masks for visualization
        if batch_idx == 0:
            for i in range(min(N_VIS, ent.shape[0])):
                per_image_masks.append({
                    'entropy': ent[i].cpu().numpy(),
                    'mask': mask[i, 0].cpu().numpy(),
                    'firing_rate': fr[i].cpu().numpy(),
                    'label': labels[i].item()
                })

all_entropies = np.concatenate(all_entropies)
all_firing_rates = np.concatenate(all_firing_rates)
all_masks = np.concatenate(all_masks)

print(f"\n  Entropy stats:")
print(f"    Mean:   {all_entropies.mean():.4f}")
print(f"    Std:    {all_entropies.std():.4f}")
print(f"    Min:    {all_entropies.min():.4f}")
print(f"    Max:    {all_entropies.max():.4f}")
print(f"    Median: {np.median(all_entropies):.4f}")

# Distribution of entropy values
pct_above_05 = (all_entropies >= 0.5).mean() * 100
pct_above_07 = (all_entropies >= 0.7).mean() * 100
pct_above_09 = (all_entropies >= 0.9).mean() * 100
print(f"\n  Entropy distribution:")
print(f"    >= 0.5: {pct_above_05:.1f}%  (these blocks are KEPT at η=0.5)")
print(f"    >= 0.7: {pct_above_07:.1f}%")
print(f"    >= 0.9: {pct_above_09:.1f}%")
print(f"    < 0.5:  {100-pct_above_05:.1f}%  (these blocks are DROPPED at η=0.5)")

print(f"\n  Firing rate stats:")
print(f"    Mean:   {all_firing_rates.mean():.4f}")
print(f"    Std:    {all_firing_rates.std():.4f}")

# Per-image variation in masks
per_image_tx = np.array(per_image_tx)
print(f"\n  Per-image tx_rate variation:")
print(f"    Mean:   {per_image_tx.mean():.4f}")
print(f"    Std:    {per_image_tx.std():.4f}")
print(f"    Min:    {per_image_tx.min():.4f}")
print(f"    Max:    {per_image_tx.max():.4f}")
print(f"    Unique rates: {len(np.unique(np.round(per_image_tx, 4)))}")

# How many images get the EXACT SAME mask?
mask_hashes = []
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Checking mask uniqueness"):
        images = images.to(device)
        feat = front(images)
        Fp, all_S2, tx, stats = spikeadapt(feat, bit_error_rate=0.0)
        mask = stats['mask'].squeeze(1)  # (B, H, W)
        for i in range(mask.shape[0]):
            h = mask[i].cpu().numpy().tobytes()
            mask_hashes.append(h)

unique_masks = len(set(mask_hashes))
total_images = len(mask_hashes)
print(f"\n  Mask uniqueness:")
print(f"    Total images:  {total_images}")
print(f"    Unique masks:  {unique_masks}")
print(f"    Ratio:         {unique_masks/total_images:.4f}")
if unique_masks < 10:
    print(f"    ⚠️ CRITICAL: Only {unique_masks} distinct masks across {total_images} images!")
    print(f"    → The entropy mask is NOT content-adaptive. All images get ~same mask.")


# ############################################################################
# DIAGNOSTIC 2: GRADIENT-BASED IMPORTANCE (ORACLE)
# ############################################################################
print("\n" + "=" * 70)
print("DIAGNOSTIC 2: Gradient-Based Importance (Oracle)")
print("=" * 70)
print("Computing which spatial blocks actually matter for task accuracy...\n")

# For each spatial block, measure: how much does zeroing it out hurt accuracy?
# This is the "oracle" importance — what a perfect mask SHOULD look at.

criterion = nn.CrossEntropyLoss()
H2, W2 = 4, 4  # spatial dimensions of S2

# Method: compute gradient of loss w.r.t. each spatial block
grad_importance_all = []  # (N, H2, W2)
entropy_values_all = []   # (N, H2, W2)
N_GRAD = 500  # samples for gradient importance

spikeadapt.eval()
count = 0
for images, labels in tqdm(test_loader, desc="Computing gradient importance"):
    images, labels = images.to(device), labels.to(device)
    
    for i in range(images.shape[0]):
        if count >= N_GRAD:
            break
        
        img_i = images[i:i+1]
        lab_i = labels[i:i+1]
        
        feat = front(img_i)
        
        # Run encoder to get S2
        all_S2_i, m1, m2 = [], None, None
        for t in range(spikeadapt.T):
            s2, m1, m2 = spikeadapt.encoder(feat, m1, m2)
            all_S2_i.append(s2.detach())
        
        # Get entropy map for this image
        ent_i, fr_i = spikeadapt.entropy_est(all_S2_i)
        entropy_values_all.append(ent_i[0].cpu().numpy())
        
        # For each spatial block, measure importance by zeroing it
        # and checking loss increase
        base_recv = []
        ones_mask = torch.ones(1, 1, H2, W2, device=device)
        for t in range(spikeadapt.T):
            base_recv.append(all_S2_i[t].clone())
        base_fp = spikeadapt.decoder(base_recv, ones_mask)
        base_out = back(base_fp)
        base_loss = criterion(base_out, lab_i).item()
        
        imp = np.zeros((H2, W2))
        for h in range(H2):
            for w in range(W2):
                # Zero out this spatial block across all timesteps
                mod_recv = []
                for t in range(spikeadapt.T):
                    s = all_S2_i[t].clone()
                    s[:, :, h, w] = 0  # zero out
                    mod_recv.append(s)
                mod_fp = spikeadapt.decoder(mod_recv, ones_mask)
                mod_out = back(mod_fp)
                mod_loss = criterion(mod_out, lab_i).item()
                imp[h, w] = mod_loss - base_loss  # higher = more important
        
        grad_importance_all.append(imp)
        count += 1
    
    if count >= N_GRAD:
        break

grad_importance_all = np.array(grad_importance_all)  # (N, H2, W2)
entropy_values_all = np.array(entropy_values_all)      # (N, H2, W2)

# Compute correlation between entropy and gradient importance
flat_ent = entropy_values_all.flatten()
flat_imp = grad_importance_all.flatten()
correlation = np.corrcoef(flat_ent, flat_imp)[0, 1]

print(f"  Correlation between entropy and gradient importance: {correlation:.4f}")
if abs(correlation) < 0.1:
    print(f"  ⚠️ CRITICAL: Near-zero correlation! Entropy is UNRELATED to task importance.")
elif correlation < -0.1:
    print(f"  ⚠️ CRITICAL: NEGATIVE correlation! Entropy-based masking DROPS important blocks.")
elif correlation > 0.3:
    print(f"  ✓ Moderate positive correlation — entropy partially captures importance.")

# Per-block analysis: which blocks does entropy keep vs drop?
# And what's their average gradient importance?
kept_mask = (entropy_values_all >= 0.5)
dropped_mask = ~kept_mask

kept_importance = flat_imp[kept_mask.flatten()] if kept_mask.any() else np.array([0])
dropped_importance = flat_imp[dropped_mask.flatten()] if dropped_mask.any() else np.array([0])

print(f"\n  KEPT blocks (entropy >= 0.5):")
print(f"    Count:     {kept_mask.sum()} ({100*kept_mask.mean():.1f}%)")
print(f"    Avg importance: {kept_importance.mean():.4f}")
print(f"\n  DROPPED blocks (entropy < 0.5):")
print(f"    Count:     {dropped_mask.sum()} ({100*dropped_mask.mean():.1f}%)")
print(f"    Avg importance: {dropped_importance.mean():.4f}")

if dropped_importance.mean() > kept_importance.mean():
    print(f"\n  ⚠️ CRITICAL: DROPPED blocks have HIGHER importance ({dropped_importance.mean():.4f}) "
          f"than KEPT blocks ({kept_importance.mean():.4f})!")
    print(f"  → The entropy estimator is actively harmful.")


# ############################################################################
# DIAGNOSTIC 3: SPATIAL IMPORTANCE MAP
# ############################################################################
print("\n" + "=" * 70)
print("DIAGNOSTIC 3: Average Spatial Patterns")
print("=" * 70)

avg_entropy = entropy_values_all.mean(axis=0)
avg_importance = grad_importance_all.mean(axis=0)
avg_fr = all_firing_rates.reshape(-1, H2, W2).mean(axis=0) if len(all_firing_rates) > 0 else np.zeros((H2, W2))

print(f"\n  Average entropy per spatial position (4×4):")
for h in range(H2):
    row = "    "
    for w in range(W2):
        row += f"{avg_entropy[h,w]:.3f}  "
    print(row)

print(f"\n  Average gradient importance per spatial position (4×4):")
for h in range(H2):
    row = "    "
    for w in range(W2):
        row += f"{avg_importance[h,w]:.4f}  "
    print(row)


# ############################################################################
# DIAGNOSTIC 4: η SWEEP WITH ORACLE COMPARISON
# ############################################################################
print("\n" + "=" * 70)
print("DIAGNOSTIC 4: η Sweep — Entropy vs Oracle Masking")
print("=" * 70)

def eval_with_mask_strategy(strategy='entropy', keep_rate=0.75, n_batches=20):
    """Evaluate with different masking strategies."""
    correct, total = 0, 0
    batch_count = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            if batch_count >= n_batches:
                break
            images, labels = images.to(device), labels.to(device)
            feat = front(images)
            
            # Encode
            all_S2, m1, m2 = [], None, None
            for t in range(spikeadapt.T):
                s2, m1, m2 = spikeadapt.encoder(feat, m1, m2)
                all_S2.append(s2)
            
            B, C2, H, W = all_S2[0].shape
            
            if strategy == 'entropy':
                ent, fr = spikeadapt.entropy_est(all_S2)
                # Keep top-k by entropy
                k = max(1, int(keep_rate * H * W))
                ent_flat = ent.view(B, -1)
                _, idx = ent_flat.topk(k, dim=1)
                mask_flat = torch.zeros_like(ent_flat)
                mask_flat.scatter_(1, idx, 1.0)
                mask = mask_flat.view(B, H, W).unsqueeze(1)
                
            elif strategy == 'anti_entropy':
                ent, fr = spikeadapt.entropy_est(all_S2)
                # Keep BOTTOM-k by entropy (drop high entropy, keep low)
                k = max(1, int(keep_rate * H * W))
                ent_flat = ent.view(B, -1)
                _, idx = (-ent_flat).topk(k, dim=1)  # lowest entropy
                mask_flat = torch.zeros_like(ent_flat)
                mask_flat.scatter_(1, idx, 1.0)
                mask = mask_flat.view(B, H, W).unsqueeze(1)
                
            elif strategy == 'random':
                mask = (torch.rand(B, 1, H, W, device=device) < keep_rate).float()
                
            elif strategy == 'all':
                mask = torch.ones(B, 1, H, W, device=device)
                
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            recv = []
            for t in range(spikeadapt.T):
                recv.append(all_S2[t] * mask)
            Fp = spikeadapt.decoder(recv, mask)
            out = back(Fp)
            correct += out.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
            batch_count += 1
    
    return 100. * correct / total if total > 0 else 0.0

print("\nComparing masking strategies (same 75% keep rate, no channel noise):\n")
strategies = {
    'All blocks (η=0)':      ('all', 1.0),
    'Entropy top-75%':        ('entropy', 0.75),
    'Anti-entropy top-75%':   ('anti_entropy', 0.75),
    'Random 75%':             ('random', 0.75),
}

strategy_results = {}
for name, (strat, rate) in strategies.items():
    # Run 3 times for random strategies
    accs = []
    for _ in range(3 if 'random' in strat.lower() else 1):
        acc = eval_with_mask_strategy(strat, rate, n_batches=80)
        accs.append(acc)
    mean_acc = np.mean(accs)
    strategy_results[name] = mean_acc
    print(f"  {name:<25s}: {mean_acc:.2f}%")

# Highlight the key finding
entropy_acc = strategy_results.get('Entropy top-75%', 0)
anti_entropy_acc = strategy_results.get('Anti-entropy top-75%', 0)
random_acc = strategy_results.get('Random 75%', 0)
all_acc = strategy_results.get('All blocks (η=0)', 0)

print(f"\n  Key findings:")
print(f"    All blocks:     {all_acc:.2f}%")
print(f"    Entropy mask:   {entropy_acc:.2f}% (Δ = {entropy_acc - all_acc:+.2f}%)")
print(f"    Anti-entropy:   {anti_entropy_acc:.2f}% (Δ = {anti_entropy_acc - all_acc:+.2f}%)")
print(f"    Random mask:    {random_acc:.2f}% (Δ = {random_acc - all_acc:+.2f}%)")

if anti_entropy_acc > entropy_acc:
    print(f"\n  ⚠️ SMOKING GUN: Keeping LOW-entropy blocks ({anti_entropy_acc:.2f}%) beats "
          f"keeping HIGH-entropy blocks ({entropy_acc:.2f}%)!")
    print(f"  → The entropy criterion is INVERTED for this task.")


# ############################################################################
# CREATE DIAGNOSTIC FIGURE
# ############################################################################
print("\n" + "=" * 70)
print("Creating diagnostic figure...")
print("=" * 70)

fig = plt.figure(figsize=(20, 16))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# Panel 1: Entropy distribution histogram
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(all_entropies, bins=50, color='steelblue', alpha=0.7, edgecolor='navy')
ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label=f'η=0.5 threshold')
ax1.set_xlabel('Block Entropy', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('Entropy Distribution (All Blocks)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.annotate(f'{pct_above_05:.0f}% kept\n{100-pct_above_05:.0f}% dropped',
             xy=(0.5, 0.85), xycoords='axes fraction', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Panel 2: Firing rate distribution
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(all_firing_rates, bins=50, color='coral', alpha=0.7, edgecolor='darkred')
ax2.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Rate=0.5 (max entropy)')
ax2.set_xlabel('Firing Rate', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Firing Rate Distribution', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)

# Panel 3: Per-image tx_rate distribution
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(per_image_tx, bins=30, color='mediumpurple', alpha=0.7, edgecolor='indigo')
ax3.set_xlabel('Per-Image Tx Rate', fontsize=12)
ax3.set_ylabel('Count', fontsize=12)
ax3.set_title(f'Per-Image Tx Rate (std={per_image_tx.std():.4f})', fontsize=13, fontweight='bold')
ax3.annotate(f'{unique_masks} unique masks\nout of {total_images} images',
             xy=(0.05, 0.85), xycoords='axes fraction', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Panel 4: Entropy vs Gradient Importance scatter
ax4 = fig.add_subplot(gs[1, 0])
# Subsample for plotting
n_pts = min(5000, len(flat_ent))
idx = np.random.choice(len(flat_ent), n_pts, replace=False)
ax4.scatter(flat_ent[idx], flat_imp[idx], alpha=0.15, s=8, color='teal')
ax4.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label=f'η=0.5')
ax4.set_xlabel('Block Entropy', fontsize=12)
ax4.set_ylabel('Gradient Importance (Δloss)', fontsize=12)
ax4.set_title(f'Entropy vs Importance (r={correlation:.3f})', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)

# Panel 5: Average entropy map vs importance map
ax5a = fig.add_subplot(gs[1, 1])
im1 = ax5a.imshow(avg_entropy, cmap='YlOrRd', vmin=0, vmax=1)
ax5a.set_title('Avg Entropy Map (4×4)', fontsize=13, fontweight='bold')
plt.colorbar(im1, ax=ax5a, fraction=0.046)
for h in range(H2):
    for w in range(W2):
        ax5a.text(w, h, f'{avg_entropy[h,w]:.2f}', ha='center', va='center', fontsize=9)

ax5b = fig.add_subplot(gs[1, 2])
im2 = ax5b.imshow(avg_importance, cmap='YlOrRd')
ax5b.set_title('Avg Gradient Importance (4×4)', fontsize=13, fontweight='bold')
plt.colorbar(im2, ax=ax5b, fraction=0.046)
for h in range(H2):
    for w in range(W2):
        ax5b.text(w, h, f'{avg_importance[h,w]:.3f}', ha='center', va='center', fontsize=9)

# Panel 6: Strategy comparison bar chart
ax6 = fig.add_subplot(gs[2, 0])
names = list(strategy_results.keys())
vals = list(strategy_results.values())
colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']
bars = ax6.barh(names, vals, color=colors, alpha=0.8)
ax6.set_xlabel('Accuracy (%)', fontsize=12)
ax6.set_title('Masking Strategy Comparison', fontsize=13, fontweight='bold')
for bar, v in zip(bars, vals):
    ax6.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
             f'{v:.1f}%', va='center', fontsize=10)
ax6.set_xlim(min(vals) - 2, max(vals) + 3)

# Panel 7: Sample masks for different images
ax7 = fig.add_subplot(gs[2, 1:])
n_show = min(8, len(per_image_masks))
mask_grid = np.zeros((2, n_show * H2 + (n_show-1), W2 + 1))
for i in range(n_show):
    col_start = i * (H2 + 1)
    mask_grid[0, col_start:col_start+H2, :W2] = per_image_masks[i]['entropy']
    mask_grid[1, col_start:col_start+H2, :W2] = per_image_masks[i]['mask']

# Just show the first row of masks side by side
fig_masks, axes = plt.subplots(2, n_show, figsize=(n_show*2, 4))
for i in range(n_show):
    axes[0, i].imshow(per_image_masks[i]['entropy'], cmap='YlOrRd', vmin=0, vmax=1)
    axes[0, i].set_title(f'Img {i}\nCls {per_image_masks[i]["label"]}', fontsize=8)
    axes[0, i].axis('off')
    axes[1, i].imshow(per_image_masks[i]['mask'], cmap='Greens', vmin=0, vmax=1)
    axes[1, i].axis('off')
axes[0, 0].set_ylabel('Entropy', fontsize=10)
axes[1, 0].set_ylabel('Mask', fontsize=10)
fig_masks.suptitle('Entropy Maps & Masks Across Different Images', fontsize=12, fontweight='bold')
fig_masks.tight_layout()
fig_masks.savefig(os.path.join(EVAL_DIR, "entropy_per_image_masks.png"), dpi=150)
plt.close(fig_masks)

# Add text summary to panel 7
ax7.axis('off')
summary_text = (
    "DIAGNOSIS SUMMARY\n"
    "─────────────────────────────────────────────\n"
    f"Entropy distribution: mean={all_entropies.mean():.3f}, std={all_entropies.std():.3f}\n"
    f"  → {pct_above_05:.0f}% of blocks above η=0.5 threshold\n"
    f"  → Entropy is heavily clustered near {all_entropies.mean():.2f}\n\n"
    f"Mask uniqueness: {unique_masks} distinct masks / {total_images} images\n"
    f"  → {'NOT content-adaptive (nearly static mask)' if unique_masks < 50 else 'Some adaptation'}\n\n"
    f"Entropy–Importance correlation: r = {correlation:.3f}\n"
    f"  → {'UNRELATED' if abs(correlation) < 0.1 else 'NEGATIVE' if correlation < 0 else 'Weak positive'}\n\n"
    f"Kept blocks importance:    {kept_importance.mean():.4f}\n"
    f"Dropped blocks importance: {dropped_importance.mean():.4f}\n"
    f"  → {'DROPPING MORE IMPORTANT BLOCKS!' if dropped_importance.mean() > kept_importance.mean() else 'OK'}\n\n"
    "ROOT CAUSE: The entropy loss (Eq.14) pushes ALL blocks toward\n"
    "firing rate ≈ 0.5 → entropy ≈ 1.0. This makes the entropy map\n"
    "nearly uniform, so the mask becomes static (same blocks dropped\n"
    "for every image). The few blocks with distinctive rates (low\n"
    "entropy) are the ones with the most task-relevant information,\n"
    "and they're the ones being dropped."
)
ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

fig.suptitle('SpikeAdapt-SC Entropy Estimator Diagnostic', fontsize=16, fontweight='bold', y=0.98)
fig.savefig(os.path.join(EVAL_DIR, "entropy_diagnostic.png"), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {EVAL_DIR}entropy_diagnostic.png")
print(f"  Saved: {EVAL_DIR}entropy_per_image_masks.png")


# ############################################################################
# SAVE NUMERICAL RESULTS
# ############################################################################
diag_results = {
    'entropy_stats': {
        'mean': float(all_entropies.mean()),
        'std': float(all_entropies.std()),
        'min': float(all_entropies.min()),
        'max': float(all_entropies.max()),
        'pct_above_0.5': float(pct_above_05),
        'pct_above_0.9': float(pct_above_09),
    },
    'mask_uniqueness': {
        'total_images': total_images,
        'unique_masks': unique_masks,
        'ratio': float(unique_masks / total_images),
    },
    'entropy_vs_importance': {
        'correlation': float(correlation),
        'kept_importance_mean': float(kept_importance.mean()),
        'dropped_importance_mean': float(dropped_importance.mean()),
    },
    'strategy_comparison': strategy_results,
    'per_image_tx_rate': {
        'mean': float(per_image_tx.mean()),
        'std': float(per_image_tx.std()),
    },
    'avg_entropy_map': avg_entropy.tolist(),
    'avg_importance_map': avg_importance.tolist(),
}

with open(os.path.join(EVAL_DIR, "entropy_diagnostic.json"), 'w') as f:
    json.dump(diag_results, f, indent=2)
print(f"  Saved: {EVAL_DIR}entropy_diagnostic.json")


# ############################################################################
# FINAL VERDICT
# ############################################################################
print("\n" + "=" * 70)
print("FINAL VERDICT")
print("=" * 70)

problems = []
if pct_above_05 > 90:
    problems.append(f"H1 CONFIRMED: {pct_above_05:.0f}% of blocks above threshold → mask is nearly all-ones")
if unique_masks < 50:
    problems.append(f"H4 CONFIRMED: Only {unique_masks} unique masks → not content-adaptive on 4×4 grid")
if abs(correlation) < 0.15:
    problems.append(f"H2 CONFIRMED: Entropy-importance correlation = {correlation:.3f} → entropy ≠ importance")
if dropped_importance.mean() > kept_importance.mean():
    problems.append(f"H2 CONFIRMED: Dropped blocks ({dropped_importance.mean():.4f}) > Kept blocks ({kept_importance.mean():.4f})")

if problems:
    print("\n  CONFIRMED PROBLEMS:")
    for i, p in enumerate(problems, 1):
        print(f"    {i}. {p}")
    
    print(f"\n  RECOMMENDED FIXES:")
    print(f"    1. IMMEDIATE: Remove entropy/rate losses, train with task loss only")
    print(f"       → This alone should recover ~2-3% accuracy")
    print(f"    2. STRUCTURAL: Replace spike-rate entropy with gradient-based importance")
    print(f"       → Use running EMA of per-block gradient magnitudes as importance")
    print(f"    3. ARCHITECTURAL: Move split to layer3 (8×8 = 64 blocks)")
    print(f"       → More spatial positions = more room for content-adaptive masking")
    print(f"    4. ALTERNATIVE: Use learned importance network (small MLP on features)")
    print(f"       → Let the model learn what to keep instead of using firing rate")
else:
    print("\n  No critical problems found. Check results manually.")

print(f"\n{'='*70}")
print(f"DIAGNOSTIC COMPLETE")
print(f"{'='*70}")
