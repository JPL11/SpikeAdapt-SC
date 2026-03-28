#!/usr/bin/env python3
"""Train SNN-SC baseline on AID 50/50 split then evaluate both models for BER sweep.

The SNN-SC shares the SAME backbone as SpikeAdapt-SC for apple-to-apple comparison.
"""

import os, sys, random, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9, 'axes.labelsize': 10,
    'axes.titlesize': 10, 'legend.fontsize': 7.5,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

OUT = './paper/figures/'
os.makedirs(OUT, exist_ok=True)

from train_aid_v2 import ResNet50Front, ResNet50Back, SNNSC, BSC_Channel, sample_noise
from run_final_pipeline import AIDDataset5050, SpikeAdaptSC_v5c_NA

SEED = 42
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

BB_DIR = 'snapshots_aid_5050_seed42'
V5C_DIR = 'snapshots_aid_v5cna_seed42'
SNNSC_DIR = 'snapshots_aid_snnsc_5050'
os.makedirs(SNNSC_DIR, exist_ok=True)

tf_train = T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
                       T.RandomRotation(15), T.ColorJitter(0.2, 0.2),
                       T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))])
tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                      T.Normalize((.485,.456,.406),(.229,.224,.225))])


def evaluate(front, model, back, loader, ber=0.0):
    front.eval(); model.eval(); back.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            Fp, _ = model(front(imgs), ber)
            correct += back(Fp).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total


# ============================================================================
# STEP 1: Train SNN-SC on AID 50/50 (same backbone as SpikeAdapt-SC)
# ============================================================================
print("\n" + "=" * 60)
print("  Training SNN-SC baseline on AID 50/50 split")
print("=" * 60)

# Load backbone (same as SpikeAdapt-SC)
front = ResNet50Front(grid_size=14).to(device)
bb = torch.load(f"./{BB_DIR}/backbone_best.pth", map_location=device, weights_only=False)
front.load_state_dict({k: v for k, v in bb.items()
                       if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))},
                      strict=False)
front.eval()
for p in front.parameters():
    p.requires_grad = False

# Load back (start from backbone back weights)
back = ResNet50Back(30).to(device)
back_state = {k: v for k, v in bb.items()
              if k.startswith(('layer4.', 'fc.', 'avgpool.'))}
back.load_state_dict(back_state, strict=False)

# Create SNN-SC (no mask, same architecture dims as SpikeAdapt-SC)
snnsc = SNNSC(C_in=1024, C1=256, C2=36, T=8).to(device)

# Data
train_ds = AIDDataset5050(transform=tf_train, split='train', seed=SEED, root="./data")
test_ds = AIDDataset5050(transform=tf_test, split='test', seed=SEED, root="./data")
train_loader = DataLoader(train_ds, 32, True, num_workers=4, pin_memory=True, drop_last=True)
test_loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)

# Check if already trained
existing = sorted([f for f in os.listdir(SNNSC_DIR) if f.startswith('snnsc_s3_')])
if existing:
    ck = torch.load(f"./{SNNSC_DIR}/{existing[-1]}", map_location=device, weights_only=False)
    snnsc.load_state_dict(ck['model'])
    back.load_state_dict(ck['back'])
    acc = evaluate(front, snnsc, back, test_loader)
    print(f"  ✓ Loaded existing SNN-SC from {existing[-1]}: {acc:.2f}%")
else:
    criterion = nn.CrossEntropyLoss()

    # Phase 2: Train SNN encoder/decoder (back frozen)
    print("\n  Phase 2: Train SNN encoder/decoder (60 epochs)...")
    for p in back.parameters():
        p.requires_grad = False
    opt = optim.Adam(snnsc.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, 60, 1e-6)
    best_s2 = 0.0

    for ep in range(60):
        snnsc.train()
        for img, lab in tqdm(train_loader, desc=f"SNNSC S2 E{ep+1}/60", leave=False):
            img, lab = img.to(device), lab.to(device)
            ber = sample_noise('bsc')
            with torch.no_grad():
                feat = front(img)
            Fp, _ = snnsc(feat, ber)
            loss = criterion(back(Fp), lab)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(snnsc.parameters(), 1.0)
            opt.step()
        sched.step()

        if (ep + 1) % 10 == 0:
            acc = evaluate(front, snnsc, back, test_loader)
            print(f"    S2 E{ep+1}: {acc:.2f}%")
            if acc > best_s2:
                best_s2 = acc
                torch.save(snnsc.state_dict(), f"./{SNNSC_DIR}/snnsc_{acc:.2f}.pth")

    # Load best S2
    s2f = sorted([f for f in os.listdir(SNNSC_DIR) if f.startswith('snnsc_') and not f.startswith('snnsc_s3')])
    if s2f:
        snnsc.load_state_dict(torch.load(f"./{SNNSC_DIR}/{s2f[-1]}", map_location=device, weights_only=False))

    # Phase 3: Joint finetune (back unfrozen)
    print("\n  Phase 3: Joint finetuning (30 epochs)...")
    for p in back.parameters():
        p.requires_grad = True
    params = list(back.parameters()) + list(snnsc.parameters())
    opt3 = optim.Adam(params, lr=1e-5, weight_decay=1e-4)
    sched3 = optim.lr_scheduler.CosineAnnealingLR(opt3, 30, 1e-7)
    best_s3 = 0.0

    for ep in range(30):
        snnsc.train(); back.train()
        opt3.zero_grad()
        for step, (img, lab) in enumerate(tqdm(train_loader, desc=f"SNNSC S3 E{ep+1}/30", leave=False)):
            img, lab = img.to(device), lab.to(device)
            ber = sample_noise('bsc')
            with torch.no_grad():
                feat = front(img)
            loss = criterion(back(snnsc(feat, ber)[0]), lab) / 2
            loss.backward()
            if (step + 1) % 2 == 0:
                nn.utils.clip_grad_norm_(params, 1.0)
                opt3.step(); opt3.zero_grad()
        sched3.step()

        if (ep + 1) % 10 == 0:
            acc = evaluate(front, snnsc, back, test_loader)
            print(f"    S3 E{ep+1}: {acc:.2f}%")
            if acc > best_s3:
                best_s3 = acc
                torch.save({'model': snnsc.state_dict(), 'back': back.state_dict()},
                           f"./{SNNSC_DIR}/snnsc_s3_{acc:.2f}.pth")

    # Load best
    s3f = sorted([f for f in os.listdir(SNNSC_DIR) if f.startswith('snnsc_s3_')])
    if s3f:
        ck = torch.load(f"./{SNNSC_DIR}/{s3f[-1]}", map_location=device, weights_only=False)
        snnsc.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
    acc = evaluate(front, snnsc, back, test_loader)
    print(f"  ✅ SNN-SC best: {acc:.2f}% (clean)")


# ============================================================================
# STEP 2: BER sweep both models
# ============================================================================
print("\n" + "=" * 60)
print("  BSC Sweep: SpikeAdapt-SC vs SNN-SC on AID 50/50")
print("=" * 60)

ber_points = [0.0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
N_REPEAT = 5

# Reload SpikeAdapt-SC
back_sa = ResNet50Back(30).to(device)
sa_model = SpikeAdaptSC_v5c_NA(C_in=1024, C1=256, C2=36, T=8,
                                target_rate=0.75, grid_size=14).to(device)
ck_files = sorted([f for f in os.listdir(f"./{V5C_DIR}") if f.startswith('v5cna_best')])
ck = torch.load(f"./{V5C_DIR}/{ck_files[-1]}", map_location=device, weights_only=False)
sa_model.load_state_dict(ck['model']); back_sa.load_state_dict(ck['back'])
sa_model.eval(); back_sa.eval()

# SpikeAdapt-SC sweep
print("\n  SpikeAdapt-SC (ρ=0.75):")
sa_results = {}
for ber in ber_points:
    accs = []
    nr = N_REPEAT if ber > 0 else 1
    for _ in range(nr):
        front.eval(); sa_model.eval(); back_sa.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                Fp, _ = sa_model(front(imgs), noise_param=ber)
                correct += back_sa(Fp).argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        accs.append(100. * correct / total)
    sa_results[str(ber)] = {'mean': float(np.mean(accs)), 'std': float(np.std(accs))}
    print(f"    BER={ber:.2f}: {np.mean(accs):.2f}±{np.std(accs):.2f}%")

# SNN-SC sweep
print("\n  SNN-SC (no mask, ρ=1.0):")
sc_results = {}
for ber in ber_points:
    accs = []
    nr = N_REPEAT if ber > 0 else 1
    for _ in range(nr):
        acc = evaluate(front, snnsc, back, test_loader, ber=ber)
        accs.append(acc)
    sc_results[str(ber)] = {'mean': float(np.mean(accs)), 'std': float(np.std(accs))}
    print(f"    BER={ber:.2f}: {np.mean(accs):.2f}±{np.std(accs):.2f}%")

# Verify data distinctness
print(f"\n  VERIFICATION:")
for bk in ['0.0', '0.1', '0.2', '0.3']:
    if bk in sa_results and bk in sc_results:
        diff = abs(sa_results[bk]['mean'] - sc_results[bk]['mean'])
        print(f"    BER={bk}: SA={sa_results[bk]['mean']:.2f} vs SC={sc_results[bk]['mean']:.2f} "
              f"Δ={diff:.2f} {'✓ distinct' if diff > 0.05 else '⚠ OVERLAP'}")

# Save corrected data
data = {'spikeadapt_sc': sa_results, 'snn_sc': sc_results}
with open(f'{OUT}/ber_sweep_corrected.json', 'w') as f:
    json.dump(data, f, indent=2)
print(f"\n  ✅ Corrected data saved")


# ============================================================================
# STEP 3: Generate corrected figures
# ============================================================================
print("\n" + "=" * 60)
print("  Generating corrected BER sweep figures")
print("=" * 60)

cnn_mc = {}
if os.path.exists('eval/cnn_multichannel.json'):
    with open('eval/cnn_multichannel.json') as f:
        cnn_mc = json.load(f)

bers_plot = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

# Full figure
fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.2))

sa_means = [sa_results[str(b)]['mean'] for b in bers_plot]
sa_stds = [sa_results[str(b)]['std'] for b in bers_plot]
ax.plot(bers_plot, sa_means, '-o', color='#1B5E20', lw=2.5, ms=6, zorder=6,
        label=r'SpikeAdapt-SC ($\rho$=0.75)', markeredgecolor='white', markeredgewidth=0.5)
ax.fill_between(bers_plot, [m-s for m,s in zip(sa_means, sa_stds)],
                [m+s for m,s in zip(sa_means, sa_stds)], alpha=0.15, color='#1B5E20')

sc_means = [sc_results[str(b)]['mean'] for b in bers_plot]
sc_stds = [sc_results[str(b)]['std'] for b in bers_plot]
ax.plot(bers_plot, sc_means, '-s', color='#1976D2', lw=2.0, ms=5, zorder=5,
        label=r'SNN-SC (no mask, $\rho$=1.0)')
ax.fill_between(bers_plot, [m-s for m,s in zip(sc_means, sc_stds)],
                [m+s for m,s in zip(sc_means, sc_stds)], alpha=0.12, color='#1976D2')

ds_label = 'AID'
if ds_label in cnn_mc:
    cu = cnn_mc[ds_label].get('CNN-Uni', {}).get('bsc', {})
    if cu:
        ax.plot(bers_plot, [cu.get(str(b), 1.0) for b in bers_plot],
                '--^', color='#E53935', lw=1.5, ms=4, zorder=4, label='CNN-Uni (8-bit)')
    cb = cnn_mc[ds_label].get('CNN-Bern', {}).get('bsc', {})
    if cb:
        ax.plot(bers_plot, [cb.get(str(b), 1.0) for b in bers_plot],
                '--v', color='#FF9800', lw=1.5, ms=4, zorder=4, label='CNN-Bern')

ax.set_xlabel('Bit Error Rate (BER)')
ax.set_ylabel('Classification Accuracy (%)')
ax.set_title('AID — BSC Robustness', fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(-0.01, 0.31)
ax.legend(loc='lower left', fontsize=8, frameon=True, fancybox=True, framealpha=0.9)
plt.tight_layout()
plt.savefig(f'{OUT}/fig_ber_sweep_fixed.png', dpi=200)
plt.savefig(f'{OUT}/fig_ber_sweep_fixed.pdf')
plt.close()
print(f"  ✅ {OUT}fig_ber_sweep_fixed.png")

# Zoomed
fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.2))
ax.plot(bers_plot, sa_means, '-o', color='#1B5E20', lw=2.5, ms=6, zorder=6,
        label=r'SpikeAdapt-SC ($\rho$=0.75)', markeredgecolor='white', markeredgewidth=0.5)
ax.fill_between(bers_plot, [m-s for m,s in zip(sa_means, sa_stds)],
                [m+s for m,s in zip(sa_means, sa_stds)], alpha=0.15, color='#1B5E20')
ax.plot(bers_plot, sc_means, '-s', color='#1976D2', lw=2.0, ms=5, zorder=5,
        label=r'SNN-SC (no mask, $\rho$=1.0)')
ax.fill_between(bers_plot, [m-s for m,s in zip(sc_means, sc_stds)],
                [m+s for m,s in zip(sc_means, sc_stds)], alpha=0.12, color='#1976D2')
ax.set_xlabel('Bit Error Rate (BER)')
ax.set_ylabel('Classification Accuracy (%)')
ax.set_title('AID — BSC Zoomed (SNN Methods)', fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(-0.01, 0.31)
ymin = min(min(sa_means), min(sc_means)) - 1
ymax = max(max(sa_means), max(sc_means)) + 0.5
ax.set_ylim(max(ymin, 85), ymax)
ax.legend(loc='lower left', fontsize=8, frameon=True, fancybox=True, framealpha=0.9)
plt.tight_layout()
plt.savefig(f'{OUT}/fig_ber_sweep_fixed_zoomed.png', dpi=200)
plt.savefig(f'{OUT}/fig_ber_sweep_fixed_zoomed.pdf')
plt.close()
print(f"  ✅ {OUT}fig_ber_sweep_fixed_zoomed.png")


# ============================================================================
# STEP 4: CIFAR-100 BSC comparison with all baselines
# ============================================================================
print("\n" + "=" * 60)
print("  CIFAR-100 BSC comparison with all baselines")
print("=" * 60)

cifar_path = 'snapshots_cifar_v5cna/cifar_comparison_results.json'
if os.path.exists(cifar_path):
    with open(cifar_path) as f:
        cifar_bsc = json.load(f)

    styles = {
        'SpikeAdapt-SC (v5cNA)': {'c': '#1B5E20', 'm': 'o', 'ls': '-', 'lw': 2.5, 'ms': 6, 'l': r'SpikeAdapt-SC ($\rho$=0.75)'},
        'SNN-SC (T=8)': {'c': '#1976D2', 'm': 's', 'ls': '-', 'lw': 2.0, 'ms': 5, 'l': 'SNN-SC (T=8)'},
        'SNN-SC (T=6)': {'c': '#42A5F5', 'm': 's', 'ls': '--', 'lw': 1.5, 'ms': 4, 'l': 'SNN-SC (T=6)'},
        'CNN-Bern': {'c': '#FF9800', 'm': 'v', 'ls': '--', 'lw': 1.5, 'ms': 4, 'l': 'CNN-Bern'},
        'Random Mask': {'c': '#9C27B0', 'm': 'D', 'ls': '-.', 'lw': 1.5, 'ms': 4, 'l': 'Random Mask'},
        'JPEG+Conv': {'c': '#795548', 'm': 'x', 'ls': '--', 'lw': 1.5, 'ms': 5, 'l': 'JPEG+Conv'},
        'CNN-Uni': {'c': '#E53935', 'm': '^', 'ls': '--', 'lw': 1.5, 'ms': 4, 'l': 'CNN-Uni (8-bit)'},
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    for model_name in ['SpikeAdapt-SC (v5cNA)', 'SNN-SC (T=8)', 'SNN-SC (T=6)',
                        'CNN-Bern', 'Random Mask', 'JPEG+Conv', 'CNN-Uni']:
        if model_name not in cifar_bsc: continue
        d = cifar_bsc[model_name]; s = styles[model_name]
        ax1.plot([x['ber'] for x in d], [x['mean'] for x in d],
                 f"{s['ls']}", marker=s['m'], color=s['c'], lw=s['lw'], ms=s['ms'], label=s['l'])

    ax1.set_xlabel('Bit Error Rate (BER)'); ax1.set_ylabel('Classification Accuracy (%)')
    ax1.set_title('(a) CIFAR-100 BSC — Full Range', fontweight='bold')
    ax1.set_ylim(0, 82); ax1.grid(True, alpha=0.3, linestyle='--')

    for model_name in ['SpikeAdapt-SC (v5cNA)', 'SNN-SC (T=8)', 'SNN-SC (T=6)',
                        'CNN-Bern', 'Random Mask']:
        if model_name not in cifar_bsc: continue
        d = cifar_bsc[model_name]; s = styles[model_name]
        bers_d = [x['ber'] for x in d]; means_d = [x['mean'] for x in d]; stds_d = [x['std'] for x in d]
        ax2.plot(bers_d, means_d, f"{s['ls']}", marker=s['m'], color=s['c'], lw=s['lw'], ms=s['ms'], label=s['l'])
        ax2.fill_between(bers_d, [m-sd for m,sd in zip(means_d, stds_d)],
                         [m+sd for m,sd in zip(means_d, stds_d)], alpha=0.1, color=s['c'])

    ax2.set_xlabel('Bit Error Rate (BER)'); ax2.set_ylabel('Classification Accuracy (%)')
    ax2.set_title('(b) CIFAR-100 BSC — Zoomed', fontweight='bold')
    ax2.set_ylim(69, 77); ax2.grid(True, alpha=0.3, linestyle='--')

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.04), frameon=True, fancybox=True, framealpha=0.9)
    plt.tight_layout(); plt.subplots_adjust(bottom=0.18)
    plt.savefig(f'{OUT}/cifar_bsc_comparison.png', dpi=200)
    plt.savefig(f'{OUT}/cifar_bsc_comparison.pdf')
    plt.close()
    print(f"  ✅ {OUT}cifar_bsc_comparison.png")
else:
    print(f"  ⚠ No CIFAR data at {cifar_path}")

print("\n✅ All done!")
