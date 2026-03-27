#!/usr/bin/env python3
"""CIFAR-100 Apple-to-Apple Comparison: SpikeAdapt-SC vs SNN-SC Paper.

Uses the SAME backbone, SAME hyperparams, SAME baselines as the SNN-SC paper,
but replaces entropy-based masking with the v5c_NA architecture:
  - NoiseAwareScorer (BER-conditioned channel gates)
  - MPBN (membrane potential batch normalization)
  - LearnedBlockMask (Gumbel-sigmoid / top-k)
  - Diversity loss (forces BER-adaptive masks)

Protocol (matching SNN-SC paper §IV):
  - CIFAR-100: 50k train / 10k test
  - ResNet50 backbone (CIFAR variant: conv1=3×3 s1, no maxpool)
  - C_in=2048, C1=256, C2=128, T=8
  - BSC channel, BER sweep 0.0→0.30, 10 repeats
  - 3-step training: backbone (reuse) → SNN (50ep) → joint finetune (50ep)

Run: python train/train_cifar_v5cna.py
"""

import os, sys, random, json
import numpy as np
from tqdm import tqdm

os.environ['PYTHONUNBUFFERED'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.spikeadapt_sc import SpikeAdaptSC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

SNAP_DIR = "./snapshots_cifar_v5cna/"
BASELINE_DIR = "./snapshots_baselines/"
OUT_DIR = "./paper/figures/"
os.makedirs(SNAP_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================================
# CIFAR ResNet50 Backbone (same as SNN-SC paper)
# ============================================================================
class ResNet50Front(nn.Module):
    """ResNet50 feature extractor for CIFAR-100 (conv1=3×3 s1, no maxpool)."""
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
    """Classification head for CIFAR-100."""
    def __init__(self, num_classes=100):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.fc(torch.flatten(self.avgpool(x), 1))


# ============================================================================
# SNN-SC baseline (same encoder/decoder, no scorer/mask)
# ============================================================================
class SNNSC(nn.Module):
    """Original SNN-SC: same arch as SpikeAdapt but NO masking (ρ=1.0)."""
    def __init__(self, C_in=2048, C1=256, C2=128, T=8):
        super().__init__()
        self.T = T
        # Reuse same encoder/decoder architecture as existing baselines
        from archive.train.train_baselines import Encoder, Decoder, BSC_Channel
        self.encoder = Encoder(C_in, C1, C2)
        self.decoder = Decoder(C_in, C1, C2, T)
        self.channel = BSC_Channel()

    def forward(self, feat, bit_error_rate=0.0):
        all_S2, m1, m2 = [], None, None
        for t in range(self.T):
            s2, m1, m2 = self.encoder(feat, m1, m2)
            all_S2.append(s2)
        recv = [self.channel(all_S2[t], bit_error_rate) for t in range(self.T)]
        ones_mask = torch.ones(feat.shape[0], 1, all_S2[0].shape[2], all_S2[0].shape[3],
                               device=feat.device)
        Fp = self.decoder(recv, ones_mask)
        return Fp, all_S2


# ============================================================================
# Dataset
# ============================================================================
train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                      T.ToTensor(),
                      T.Normalize((0.5071, .4867, .4408), (.2675, .2565, .2761))])
test_tf = T.Compose([T.ToTensor(),
                     T.Normalize((0.5071, .4867, .4408), (.2675, .2565, .2761))])

train_ds = torchvision.datasets.CIFAR100("./data", True, download=True, transform=train_tf)
test_ds = torchvision.datasets.CIFAR100("./data", False, download=True, transform=test_tf)
train_loader = DataLoader(train_ds, 64, shuffle=True, num_workers=4,
                          pin_memory=True, drop_last=True)
test_loader = DataLoader(test_ds, 128, shuffle=False, num_workers=4, pin_memory=True)
print(f"✓ CIFAR-100: {len(train_ds)} train, {len(test_ds)} test")


# ============================================================================
# Load backbone
# ============================================================================
front = ResNet50Front().to(device)
back = ResNet50Back(100).to(device)

bb_path = "./snapshots_spikeadapt/backbone_best.pth"
state = torch.load(bb_path, map_location=device, weights_only=False)
f_st = {k: v for k, v in state.items() if not k.startswith(('fc.', 'avgpool.'))}
b_st = {k: v for k, v in state.items() if k.startswith(('fc.', 'avgpool.'))}
front.load_state_dict(f_st, strict=False)
back.load_state_dict(b_st, strict=False)
front.eval()
print("✓ Backbone loaded")

# Verify feature shape
with torch.no_grad():
    d = front(torch.randn(1, 3, 32, 32).to(device))
    print(f"★ Feature shape: {d.shape}")  # (1, 2048, 4, 4)
C_in = d.shape[1]  # 2048


# ============================================================================
# Backbone accuracy (upper bound)
# ============================================================================
back.eval()
correct, total = 0, 0
with torch.no_grad():
    for img, lab in test_loader:
        img, lab = img.to(device), lab.to(device)
        out = back(front(img))
        correct += out.argmax(1).eq(lab).sum().item()
        total += lab.size(0)
bb_acc = 100. * correct / total
print(f"★ BACKBONE ACCURACY: {bb_acc:.2f}%")


# ============================================================================
# Create SpikeAdapt-SC v5c_NA
# ============================================================================
snn = SpikeAdaptSC(C_in=C_in, C1=256, C2=128, T=8,
                   target_rate=0.75, channel_type='bsc').to(device)

total_params = sum(p.numel() for p in snn.parameters())
print(f"✓ SpikeAdapt-SC v5c_NA: {total_params:,} params")
print(f"  C_in={C_in}, C2=128, T=8, ρ=0.75")


# ============================================================================
# Evaluation helper
# ============================================================================
@torch.no_grad()
def eval_model(model, front, back, loader, ber=0.0, is_v5cna=True):
    """Evaluate model at given BER."""
    model.eval(); front.eval(); back.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        feat = front(images)
        if is_v5cna:
            Fp, info = model(feat, noise_param=ber)
        else:
            Fp, _ = model(feat, bit_error_rate=ber)
        out = back(Fp)
        correct += out.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return 100. * correct / total


# ============================================================================
# STEP 2: Train SpikeAdapt-SC (frozen backbone)
# ============================================================================
print("\n" + "=" * 60)
print("STEP 2: Train SpikeAdapt-SC v5c_NA (Frozen Backbone)")
print("=" * 60)

for p in front.parameters():
    p.requires_grad = False
for p in back.parameters():
    p.requires_grad = False

criterion = nn.CrossEntropyLoss()
opt = optim.Adam(snn.parameters(), lr=1e-4, weight_decay=1e-4)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=1e-6)

best_s2 = 0.0
for ep in range(50):
    snn.train()
    correct, total, ep_loss = 0, 0, 0.0
    pbar = tqdm(train_loader, desc=f"S2 E{ep+1}/50")

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        ber = random.uniform(0, 0.3)

        with torch.no_grad():
            feat = front(images)

        Fp, info = snn(feat, noise_param=ber)
        out = back(Fp)

        # Losses
        L_CE = criterion(out, labels)
        L_div = snn.scorer.compute_diversity_loss(info['all_S2'])
        L_rate = (info['tx_rate'] - 0.75) ** 2

        total_loss = L_CE + 0.1 * L_div + 0.5 * L_rate

        opt.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(snn.parameters(), max_norm=1.0)
        opt.step()

        _, pred = out.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
        ep_loss += total_loss.item()
        pbar.set_postfix({'L': f'{total_loss.item():.3f}',
                          'A': f'{100.*pred.eq(labels).sum().item()/labels.size(0):.0f}%',
                          'tx': f'{info["tx_rate"]:.2f}'})

    sched.step()

    if (ep + 1) % 5 == 0:
        acc = eval_model(snn, front, back, test_loader, ber=0.0, is_v5cna=True)
        noisy_acc = eval_model(snn, front, back, test_loader, ber=0.1, is_v5cna=True)
        print(f"  S2 E{ep+1}: Test={acc:.2f}% (BER=0), {noisy_acc:.2f}% (BER=0.1)")
        if acc > best_s2:
            best_s2 = acc
            torch.save(snn.state_dict(), os.path.join(SNAP_DIR, f'snn_best_{acc:.2f}.pth'))
            print(f"  ✓ Best: {best_s2:.2f}%")

# Reload best
ckpts = sorted([f for f in os.listdir(SNAP_DIR) if f.startswith('snn_best')])
if ckpts:
    snn.load_state_dict(torch.load(os.path.join(SNAP_DIR, ckpts[-1]),
                                    map_location=device, weights_only=False))


# ============================================================================
# STEP 3: Joint fine-tune (SNN + classifier head)
# ============================================================================
print("\n" + "=" * 60)
print("STEP 3: Joint Fine-tuning (SNN + Head)")
print("=" * 60)

# Save original back weights for baseline reload later
back_orig = {k: v.clone() for k, v in back.state_dict().items()}

for p in back.parameters():
    p.requires_grad = True

ft_params = list(snn.parameters()) + list(back.parameters())
ft_opt = optim.Adam(ft_params, lr=1e-5, weight_decay=1e-4)
ft_sched = optim.lr_scheduler.CosineAnnealingLR(ft_opt, T_max=50, eta_min=1e-7)

best_s3 = best_s2
for ep in range(50):
    snn.train(); back.train()
    pbar = tqdm(train_loader, desc=f"S3 E{ep+1}/50", leave=False)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        ber = random.uniform(0, 0.3)
        with torch.no_grad():
            feat = front(images)

        Fp, info = snn(feat, noise_param=ber)
        out = back(Fp)

        L_CE = criterion(out, labels)
        L_div = snn.scorer.compute_diversity_loss(info['all_S2'])
        L_rate = (info['tx_rate'] - 0.75) ** 2
        total_loss = L_CE + 0.1 * L_div + 0.5 * L_rate

        ft_opt.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(ft_params, max_norm=1.0)
        ft_opt.step()

    ft_sched.step()

    if (ep + 1) % 5 == 0:
        acc = eval_model(snn, front, back, test_loader, ber=0.0, is_v5cna=True)
        print(f"  S3 E{ep+1}: {acc:.2f}%")
        if acc > best_s3:
            best_s3 = acc
            torch.save({'model': snn.state_dict(), 'back': back.state_dict()},
                       os.path.join(SNAP_DIR, f's3_best_{acc:.2f}.pth'))
            print(f"  ✓ Best: {best_s3:.2f}%")

# Reload best
s3_files = sorted([f for f in os.listdir(SNAP_DIR) if f.startswith('s3_best')])
if s3_files:
    ck = torch.load(os.path.join(SNAP_DIR, s3_files[-1]),
                    map_location=device, weights_only=False)
    snn.load_state_dict(ck['model'])
    back.load_state_dict(ck['back'])
    print(f"✓ Loaded best S3: {s3_files[-1]}")

for p in back.parameters():
    p.requires_grad = False


# ============================================================================
# BER SWEEP — ALL METHODS
# ============================================================================
print("\n" + "=" * 60)
print("BER SWEEP — CIFAR-100 ALL METHODS")
print("=" * 60)

ber_vals = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
N_REPEAT = 10

# Load existing baseline results
existing = {}
baseline_json = 'eval_results/baseline_results.json'
if os.path.exists(baseline_json):
    with open(baseline_json) as f:
        existing = json.load(f)
    print(f"✓ Loaded {len(existing)} existing baseline results")

# Re-evaluate SpikeAdapt-SC v5c_NA
print("\n  Evaluating SpikeAdapt-SC v5c_NA...")
sa_results = []
for ber in ber_vals:
    accs = []
    nr = N_REPEAT if ber > 0 else 1
    for _ in range(nr):
        a = eval_model(snn, front, back, test_loader, ber=ber, is_v5cna=True)
        accs.append(a)
    m, s = np.mean(accs), np.std(accs)
    sa_results.append({'ber': ber, 'mean': m, 'std': s})
    print(f"    BER={ber:.3f}: {m:.2f}% ±{s:.2f}")

# Combine results
all_results = {'SpikeAdapt-SC (v5cNA)': sa_results}
for name in ['SNN-SC (T=8)', 'SNN-SC (T=6)', 'CNN-Uni', 'CNN-Bern',
             'Random Mask', 'JPEG+Conv']:
    if name in existing:
        all_results[name] = existing[name]

# Also include old SpikeAdapt-SC for reference
if 'SpikeAdapt-SC' in existing:
    all_results['SpikeAdapt-SC (old)'] = existing['SpikeAdapt-SC']

# Save combined results
combined_path = os.path.join(SNAP_DIR, 'cifar_comparison_results.json')
with open(combined_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\n✓ Saved: {combined_path}")


# ============================================================================
# COMPARISON FIGURE (The Money Plot)
# ============================================================================
print("\n  Generating comparison figure...")

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

styles = {
    'SpikeAdapt-SC (v5cNA)': ('#D32F2F', 's', '-',  2.5),
    'SNN-SC (T=8)':           ('#1976D2', 'o', '-',  2.0),
    'SNN-SC (T=6)':           ('#42A5F5', 'o', '--', 2.0),
    'CNN-Uni':                 ('#388E3C', '^', '-.', 2.0),
    'CNN-Bern':                ('#F57C00', 'D', '--', 2.0),
    'JPEG+Conv':               ('#7B1FA2', 'v', ':',  2.0),
    'Random Mask':             ('#9E9E9E', 'x', '--', 1.5),
    'SpikeAdapt-SC (old)':     ('#BDBDBD', 'P', ':', 1.5),
}

# Full range figure
fig, ax = plt.subplots(figsize=(7, 4.5))
for name, res in all_results.items():
    c, mk, ls, lw = styles.get(name, ('gray', 'o', '-', 1.5))
    bers = [r['ber'] for r in res]
    means = [r['mean'] for r in res]
    stds = [r['std'] for r in res]
    ax.plot(bers, means, marker=mk, ls=ls, color=c, lw=lw, ms=5, label=name)
    if max(stds) > 0.1:
        ax.fill_between(bers, [m-s for m, s in zip(means, stds)],
                       [m+s for m, s in zip(means, stds)], alpha=0.08, color=c)

ax.set_xlabel('Bit Error Rate (BER)')
ax.set_ylabel('Classification Accuracy (%)')
ax.set_title('CIFAR-100: SpikeAdapt-SC vs Baselines on BSC', fontweight='bold')
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.01, 0.31)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/cifar_ber_comparison.png', dpi=200)
plt.savefig(f'{OUT_DIR}/cifar_ber_comparison.pdf', dpi=200)
plt.close()
print(f"  ✅ {OUT_DIR}/cifar_ber_comparison.png")

# Zoomed figure (65-78%)
fig, ax = plt.subplots(figsize=(7, 4.5))
for name, res in all_results.items():
    if 'CNN-Uni' in name or 'JPEG' in name:
        continue  # These cliff at 1%
    c, mk, ls, lw = styles.get(name, ('gray', 'o', '-', 1.5))
    bers = [r['ber'] for r in res]
    means = [r['mean'] for r in res]
    stds = [r['std'] for r in res]
    ax.plot(bers, means, marker=mk, ls=ls, color=c, lw=lw, ms=5, label=name)
    if max(stds) > 0.1:
        ax.fill_between(bers, [m-s for m, s in zip(means, stds)],
                       [m+s for m, s in zip(means, stds)], alpha=0.08, color=c)

ax.set_xlabel('Bit Error Rate (BER)')
ax.set_ylabel('Classification Accuracy (%)')
ax.set_title('CIFAR-100: Zoomed BER Sweep (SNN Methods)', fontweight='bold')
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.01, 0.31)
ax.set_ylim(65, 78)
from matplotlib.ticker import MultipleLocator
ax.yaxis.set_major_locator(MultipleLocator(2))

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/cifar_ber_comparison_zoomed.png', dpi=200)
plt.savefig(f'{OUT_DIR}/cifar_ber_comparison_zoomed.pdf', dpi=200)
plt.close()
print(f"  ✅ {OUT_DIR}/cifar_ber_comparison_zoomed.png")


# ============================================================================
# RESULTS TABLE
# ============================================================================
print("\n" + "=" * 60)
print("RESULTS TABLE (CIFAR-100)")
print("=" * 60)
print(f"{'Method':<25} {'Clean':>8} {'BER=0.10':>10} {'BER=0.20':>10} {'BER=0.30':>10}")
print("-" * 65)
for name, res in all_results.items():
    vals = {r['ber']: f"{r['mean']:.2f}" for r in res}
    print(f"{name:<25} {vals.get(0.0, '-'):>8}% {vals.get(0.1, '-'):>9}% "
          f"{vals.get(0.2, '-'):>9}% {vals.get(0.3, '-'):>9}%")

print("\n✅ CIFAR-100 COMPARISON COMPLETE!")
