#!/usr/bin/env python3
"""CIFAR-100 Ablation Studies for SpikeAdapt-SC Paper.

1. ρ-sweep: Accuracy vs transmission rate (0.25→1.0) at BER=0 and BER=0.30
2. Component ablation: NoiseAwareScorer off, MPBN off, diversity loss off
3. Comparison with SNN-SC paper protocol

Uses trained model from snapshots_cifar_v5cna/s3_best_*.pth

Run: python eval/cifar_ablations.py
"""

import os, sys, json, random
import numpy as np
from tqdm import tqdm

os.environ['PYTHONUNBUFFERED'] = '1'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.spikeadapt_sc import SpikeAdaptSC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# IEEE styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

OUT = './paper/figures/'
os.makedirs(OUT, exist_ok=True)

# ============================================================================
# Backbone
# ============================================================================
class ResNet50Front(nn.Module):
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet50(weights=None)
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = r.bn1; self.relu = r.relu
        self.maxpool = nn.Identity()
        self.layer1 = r.layer1; self.layer2 = r.layer2
        self.layer3 = r.layer3; self.layer4 = r.layer4
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))

class ResNet50Back(nn.Module):
    def __init__(self, n=100):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, n)
    def forward(self, x):
        return self.fc(torch.flatten(self.avgpool(x), 1))


# ============================================================================
# Dataset
# ============================================================================
test_tf = T.Compose([T.ToTensor(),
                     T.Normalize((0.5071, .4867, .4408), (.2675, .2565, .2761))])
test_ds = torchvision.datasets.CIFAR100("./data", False, download=True, transform=test_tf)
test_loader = DataLoader(test_ds, 128, shuffle=False, num_workers=4, pin_memory=True)


# ============================================================================
# Load models
# ============================================================================
front = ResNet50Front().to(device)
bb = torch.load('./snapshots_spikeadapt/backbone_best.pth',
                map_location=device, weights_only=False)
f_st = {k: v for k, v in bb.items() if not k.startswith(('fc.', 'avgpool.'))}
front.load_state_dict(f_st, strict=False)
front.eval()
for p in front.parameters():
    p.requires_grad = False

# Load best v5cNA model
back = ResNet50Back(100).to(device)
snn = SpikeAdaptSC(C_in=2048, C1=256, C2=128, T=8,
                   target_rate=0.75, channel_type='bsc').to(device)

s3_files = sorted([f for f in os.listdir('./snapshots_cifar_v5cna/')
                   if f.startswith('s3_best')])
ck = torch.load(f'./snapshots_cifar_v5cna/{s3_files[-1]}',
                map_location=device, weights_only=False)
snn.load_state_dict(ck['model'])
back.load_state_dict(ck['back'])
snn.eval(); back.eval()
for p in list(snn.parameters()) + list(back.parameters()):
    p.requires_grad = False
print(f"✓ Loaded {s3_files[-1]}")


# ============================================================================
# Eval helper
# ============================================================================
@torch.no_grad()
def eval_at(model, front, back, loader, ber=0.0, rho_override=None, n_repeat=10):
    """Eval model at given BER, optionally overriding ρ."""
    nr = n_repeat if ber > 0 else 1
    accs = []
    for _ in range(nr):
        model.eval()
        correct, total = 0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            feat = front(images)
            Fp, info = model(feat, noise_param=ber,
                            target_rate_override=rho_override)
            out = back(Fp)
            correct += out.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
        accs.append(100. * correct / total)
    return np.mean(accs), np.std(accs)


# ============================================================================
# 1. ρ-SWEEP: Accuracy vs Transmission Rate
# ============================================================================
print("\n" + "=" * 60)
print("  ABLATION 1: ρ-Sweep (Transmission Rate)")
print("=" * 60)

rho_vals = [0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
rho_results = {}

for ber_lbl, ber_val in [('BER=0.0', 0.0), ('BER=0.30', 0.30)]:
    results = []
    for rho in rho_vals:
        m, s = eval_at(snn, front, back, test_loader, ber=ber_val,
                       rho_override=rho, n_repeat=5)
        results.append({'rho': rho, 'mean': m, 'std': s})
        print(f"  {ber_lbl}, ρ={rho:.3f}: {m:.2f}% ±{s:.2f}")
    rho_results[ber_lbl] = results

# Also get SNN-SC baselines for comparison lines
existing = {}
if os.path.exists('eval_results/baseline_results.json'):
    with open('eval_results/baseline_results.json') as f:
        existing = json.load(f)

snnsc_clean = next((r['mean'] for r in existing.get('SNN-SC (T=8)', [])
                    if r['ber'] == 0.0), None)
snnsc_noisy = next((r['mean'] for r in existing.get('SNN-SC (T=8)', [])
                    if r['ber'] == 0.3), None)

# Plot ρ-sweep
fig, ax = plt.subplots(figsize=(6, 4))
for ber_lbl, color, mk, ls in [('BER=0.0', '#1976D2', 'o', '-'),
                                 ('BER=0.30', '#D32F2F', 's', '--')]:
    res = rho_results[ber_lbl]
    rhos = [r['rho'] for r in res]
    means = [r['mean'] for r in res]
    stds = [r['std'] for r in res]
    ax.plot(rhos, means, marker=mk, ls=ls, color=color, lw=2, ms=6,
            label=f'SpikeAdapt-SC ({ber_lbl})')
    if max(stds) > 0.1:
        ax.fill_between(rhos, [m-s for m, s in zip(means, stds)],
                        [m+s for m, s in zip(means, stds)], alpha=0.1, color=color)

# Reference lines for SNN-SC (ρ=1.0 always)
if snnsc_clean:
    ax.axhline(y=snnsc_clean, color='#1976D2', ls=':', lw=1.2, alpha=0.6)
    ax.text(0.3, snnsc_clean + 0.15, f'SNN-SC (BER=0, {snnsc_clean:.1f}%)',
            fontsize=7, color='#1976D2', alpha=0.7)
if snnsc_noisy:
    ax.axhline(y=snnsc_noisy, color='#D32F2F', ls=':', lw=1.2, alpha=0.6)
    ax.text(0.3, snnsc_noisy + 0.15, f'SNN-SC (BER=0.30, {snnsc_noisy:.1f}%)',
            fontsize=7, color='#D32F2F', alpha=0.7)

# Mark operating point
ax.axvline(x=0.75, color='green', ls=':', lw=1, alpha=0.5)
ax.text(0.76, ax.get_ylim()[0] + 1, 'ρ=0.75\n(default)', fontsize=7, color='green')

ax.set_xlabel('Transmission Rate (ρ)')
ax.set_ylabel('Classification Accuracy (%)')
ax.set_title('CIFAR-100: Accuracy vs Transmission Rate', fontweight='bold')
ax.legend(loc='lower right', fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.2, 1.05)

plt.tight_layout()
plt.savefig(f'{OUT}/cifar_rho_sweep.png', dpi=200)
plt.savefig(f'{OUT}/cifar_rho_sweep.pdf', dpi=200)
plt.close()
print(f"\n  ✅ {OUT}cifar_rho_sweep.png")


# ============================================================================
# 2. COMPONENT ABLATION
# ============================================================================
print("\n" + "=" * 60)
print("  ABLATION 2: Component Ablation")
print("=" * 60)

# Full model (already evaluated)
full_clean, full_clean_std = eval_at(snn, front, back, test_loader, ber=0.0)
full_noisy, full_noisy_std = eval_at(snn, front, back, test_loader, ber=0.30, n_repeat=5)
print(f"  Full model: {full_clean:.2f}% (BER=0), {full_noisy:.2f}% (BER=0.3)")

# Ablation: No scorer (random importance → uniform mask)
print("  Ablating: NoiseAwareScorer → random importance...")
orig_forward = snn.scorer.forward
def random_scorer(all_S2, ber):
    B = all_S2[0].size(0)
    H, W = all_S2[0].shape[2], all_S2[0].shape[3]
    return torch.rand(B, 1, H, W, device=all_S2[0].device)
snn.scorer.forward = random_scorer

no_scorer_clean, _ = eval_at(snn, front, back, test_loader, ber=0.0)
no_scorer_noisy, no_scorer_std = eval_at(snn, front, back, test_loader, ber=0.30, n_repeat=5)
print(f"  No scorer (random mask): {no_scorer_clean:.2f}% (BER=0), {no_scorer_noisy:.2f}% (BER=0.3)")
snn.scorer.forward = orig_forward

# Ablation: No BER-awareness (fix BER=0 in scorer)
print("  Ablating: BER-awareness (always BER=0 in scorer)...")
def no_ber_scorer(all_S2, ber):
    return orig_forward(all_S2, ber=0.0)  # Always score as if BER=0
snn.scorer.forward = no_ber_scorer

no_ber_clean, _ = eval_at(snn, front, back, test_loader, ber=0.0)
no_ber_noisy, no_ber_std = eval_at(snn, front, back, test_loader, ber=0.30, n_repeat=5)
print(f"  No BER-awareness: {no_ber_clean:.2f}% (BER=0), {no_ber_noisy:.2f}% (BER=0.3)")
snn.scorer.forward = orig_forward

# Ablation: No masking (ρ=1.0)
print("  Ablating: No masking (ρ=1.0)...")
no_mask_clean, _ = eval_at(snn, front, back, test_loader, ber=0.0, rho_override=1.0)
no_mask_noisy, no_mask_std = eval_at(snn, front, back, test_loader, ber=0.30,
                                      rho_override=1.0, n_repeat=5)
print(f"  No masking (ρ=1.0): {no_mask_clean:.2f}% (BER=0), {no_mask_noisy:.2f}% (BER=0.3)")

# Collect results
ablation_results = {
    'Full SpikeAdapt-SC': {'clean': full_clean, 'noisy': full_noisy, 'noisy_std': full_noisy_std},
    'No scorer (random)': {'clean': no_scorer_clean, 'noisy': no_scorer_noisy, 'noisy_std': no_scorer_std},
    'No BER-awareness': {'clean': no_ber_clean, 'noisy': no_ber_noisy, 'noisy_std': no_ber_std},
    'No masking (ρ=1.0)': {'clean': no_mask_clean, 'noisy': no_mask_noisy, 'noisy_std': no_mask_std},
}

# Save results
combined = {'rho_sweep': rho_results, 'ablation': ablation_results}
with open(f'{OUT}/cifar_ablation_results.json', 'w') as f:
    json.dump(combined, f, indent=2)

# Plot ablation bar chart
fig, ax = plt.subplots(figsize=(7, 4))
methods = list(ablation_results.keys())
clean_vals = [ablation_results[m]['clean'] for m in methods]
noisy_vals = [ablation_results[m]['noisy'] for m in methods]
noisy_stds = [ablation_results[m]['noisy_std'] for m in methods]

x = np.arange(len(methods))
w = 0.35
bars1 = ax.bar(x - w/2, clean_vals, w, label='BER=0.0', color='#1976D2', alpha=0.85)
bars2 = ax.bar(x + w/2, noisy_vals, w, yerr=noisy_stds, label='BER=0.30',
               color='#D32F2F', alpha=0.85, capsize=3)

# Add value labels
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{bar.get_height():.1f}', ha='center', fontsize=7.5)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
            f'{bar.get_height():.1f}', ha='center', fontsize=7.5)

# SNN-SC reference
if snnsc_clean:
    ax.axhline(y=snnsc_clean, color='gray', ls=':', lw=1, alpha=0.5)
    ax.text(len(methods) - 0.5, snnsc_clean + 0.15, f'SNN-SC T=8', fontsize=7, color='gray')

ax.set_xlabel('')
ax.set_ylabel('Classification Accuracy (%)')
ax.set_title('CIFAR-100: Component Ablation', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=8.5, ha='center')
ax.legend(loc='lower right', fontsize=8)
ax.grid(True, alpha=0.2, axis='y')
ax.set_ylim(min(min(clean_vals), min(noisy_vals)) - 2, max(clean_vals) + 2)

plt.tight_layout()
plt.savefig(f'{OUT}/cifar_ablation.png', dpi=200)
plt.savefig(f'{OUT}/cifar_ablation.pdf', dpi=200)
plt.close()
print(f"\n  ✅ {OUT}cifar_ablation.png")


# ============================================================================
# SUMMARY TABLE
# ============================================================================
print("\n" + "=" * 60)
print("  ABLATION SUMMARY")
print("=" * 60)

print(f"\n  {'Method':<25} {'BER=0':>8} {'BER=0.30':>10} {'Δ noisy':>8}")
print("  " + "-" * 53)
for name, vals in ablation_results.items():
    delta = vals['noisy'] - full_noisy
    sign = '+' if delta >= 0 else ''
    print(f"  {name:<25} {vals['clean']:>7.2f}% {vals['noisy']:>9.2f}% "
          f"{sign}{delta:>6.2f}pp")

print(f"\n  ρ-Sweep Summary:")
for ber_lbl, res in rho_results.items():
    vals = [f"ρ={r['rho']:.2f}:{r['mean']:.1f}%" for r in res]
    print(f"    {ber_lbl}: " + ", ".join(vals))

print("\n✅ Ablation studies complete!")
