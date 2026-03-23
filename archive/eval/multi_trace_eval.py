"""Multi-trace adaptive rate evaluation for SpikeAdapt-SC v2.

Evaluates dynamic rate adaptation across 5 diverse synthetic trajectories
with different BER profiles, instead of a single sinusoidal trajectory.
"""

import os, sys, json, math, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path.insert(0, './train')
from train_aid_v2 import (
    AIDDataset, ResNet50Front, ResNet50Back, SpikeAdaptSC_v2,
    evaluate_with_early_stop
)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 8, 'figure.dpi': 300,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

SNAP_V1 = "./snapshots_aid/"
SNAP_V2 = "./snapshots_aid_v2_seed42/"
OUT_DIR = "./paper/figures/"
T_STEPS = 8


def generate_trajectory(traj_type, n_steps=200):
    """Generate BER trajectory for BSC channel."""
    t = np.linspace(0, 1, n_steps)
    if traj_type == 'sinusoidal':
        ber = 0.15 + 0.15 * np.sin(2 * np.pi * t * 2)
    elif traj_type == 'step_changes':
        ber = np.zeros(n_steps)
        for i in range(n_steps):
            if i < 50: ber[i] = 0.05
            elif i < 100: ber[i] = 0.20
            elif i < 150: ber[i] = 0.10
            else: ber[i] = 0.30
    elif traj_type == 'ramp_up':
        ber = 0.02 + 0.28 * t
    elif traj_type == 'urban_canyon':
        ber = 0.10 + 0.15 * np.abs(np.sin(2 * np.pi * t * 5)) + 0.05 * np.random.randn(n_steps)
    elif traj_type == 'rural_stable':
        ber = 0.05 + 0.03 * np.random.randn(n_steps)
    else:
        raise ValueError(f"Unknown trajectory type: {traj_type}")
    return np.clip(ber, 0, 0.40)


def rate_policy(ber):
    """Adaptive rate policy from the paper."""
    if ber < 0.10: return 0.50
    elif ber < 0.20: return 0.75
    else: return 1.00


def evaluate_trajectory(front, model, back, test_imgs, test_labels,
                        trajectory, step_indices,
                        strategy='adaptive', fixed_rate=None):
    """Evaluate a single trajectory with given strategy.

    Args:
        step_indices: list of arrays, precomputed image indices for each step.
                      MUST be the same across strategies for fair comparison.
    """
    n_steps = len(trajectory)
    correct, total = 0, 0
    bw_used = []

    with torch.no_grad():
        for step in range(n_steps):
            ber = trajectory[step]
            idx = step_indices[step]
            imgs = test_imgs[idx].to(device)
            labels = test_labels[idx].to(device)
            feat = front(imgs)

            if strategy == 'adaptive':
                rate = rate_policy(ber)
            elif strategy == 'fixed':
                rate = fixed_rate
            else:
                raise ValueError

            Fp, stats = model(feat, noise_param=ber, target_rate_override=rate)
            preds = back(Fp).argmax(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            bw_used.append(rate)

    acc = 100. * correct / total
    avg_bw = np.mean(bw_used)
    return acc, avg_bw


if __name__ == "__main__":
    print(f"Device: {device}")

    # Load v2 model
    front = ResNet50Front(grid_size=14).to(device)
    back = ResNet50Back(30).to(device)
    bb = torch.load(os.path.join(SNAP_V1, "backbone_best.pth"), map_location=device)
    front.load_state_dict({k:v for k,v in bb.items()
                           if not k.startswith(('layer4.','fc.','avgpool.','spatial_pool.'))}, strict=False)
    front.eval()

    model = SpikeAdaptSC_v2(C_in=1024, C1=256, C2=36, T=8, target_rate=0.75,
                             channel_type='bsc', grid_size=14).to(device)
    s3f = sorted([f for f in os.listdir(SNAP_V2) if f.startswith("v2_s3_")])
    if s3f:
        ck = torch.load(os.path.join(SNAP_V2, s3f[-1]), map_location=device)
        model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
        print(f"Loaded: {s3f[-1]}")
    model.eval(); back.eval()

    # Load test data
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                     T.Normalize((.485,.456,.406),(.229,.224,.225))])
    ds = AIDDataset("./data", transform=tf, split='test', seed=42)
    # Pre-load all test images
    all_imgs, all_labels = [], []
    for img, lab in DataLoader(ds, 64, False, num_workers=4):
        all_imgs.append(img); all_labels.append(lab)
    all_imgs = torch.cat(all_imgs, 0)
    all_labels = torch.cat(all_labels, 0)
    print(f"Test set: {len(all_imgs)} images")

    # Trajectories
    traj_types = ['sinusoidal', 'step_changes', 'ramp_up', 'urban_canyon', 'rural_stable']
    traj_names = ['Sinusoidal', 'Step Changes', 'Ramp Up', 'Urban Canyon', 'Rural Stable']
    N_STEPS = 200

    results = {}
    print("\n" + "="*70)
    print("MULTI-TRACE ADAPTIVE RATE EVALUATION (5 trajectories × 200 steps)")
    print("="*70)

    for ti, (ttype, tname) in enumerate(zip(traj_types, traj_names)):
        np.random.seed(42 + ti)
        traj = generate_trajectory(ttype, N_STEPS)

        # Precompute deterministic image indices for each step
        # (shared across ALL strategies for fair comparison)
        rng = np.random.RandomState(42 + ti)
        n_per_step = min(10, len(all_imgs))
        step_indices = [rng.choice(len(all_imgs), n_per_step, replace=False)
                        for _ in range(N_STEPS)]

        # Adaptive
        acc_ad, bw_ad = evaluate_trajectory(front, model, back, all_imgs, all_labels,
                                             traj, step_indices, 'adaptive')
        # Fixed ρ=1.0
        acc_f1, bw_f1 = evaluate_trajectory(front, model, back, all_imgs, all_labels,
                                             traj, step_indices, 'fixed', 1.0)
        # Fixed ρ=0.75
        acc_f75, bw_f75 = evaluate_trajectory(front, model, back, all_imgs, all_labels,
                                               traj, step_indices, 'fixed', 0.75)
        # Fixed ρ=0.50
        acc_f50, bw_f50 = evaluate_trajectory(front, model, back, all_imgs, all_labels,
                                               traj, step_indices, 'fixed', 0.50)

        bw_saved = (1 - bw_ad) * 100
        print(f"\n  {tname} (mean BER={traj.mean():.3f}):")
        print(f"    Adaptive:     {acc_ad:.2f}%  (avg ρ={bw_ad:.2f}, {bw_saved:.0f}% BW saved)")
        print(f"    Fixed ρ=1.0:  {acc_f1:.2f}%  (0% saved)")
        print(f"    Fixed ρ=0.75: {acc_f75:.2f}%  (25% saved)")
        print(f"    Fixed ρ=0.50: {acc_f50:.2f}%  (50% saved)")

        results[ttype] = {
            'name': tname, 'mean_ber': float(traj.mean()),
            'adaptive': {'acc': acc_ad, 'bw': bw_ad},
            'fixed_1.0': {'acc': acc_f1, 'bw': 1.0},
            'fixed_0.75': {'acc': acc_f75, 'bw': 0.75},
            'fixed_0.50': {'acc': acc_f50, 'bw': 0.50},
        }

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Trajectory':<16} {'Adaptive':>10} {'Fixed 1.0':>10} {'Fixed 0.75':>10} {'Fixed 0.50':>10} {'BW Saved':>10}")
    for ttype, tname in zip(traj_types, traj_names):
        r = results[ttype]
        print(f"{tname:<16} {r['adaptive']['acc']:>9.2f}% {r['fixed_1.0']['acc']:>9.2f}% "
              f"{r['fixed_0.75']['acc']:>9.2f}% {r['fixed_0.50']['acc']:>9.2f}% "
              f"{(1-r['adaptive']['bw'])*100:>8.0f}%")

    # Mean across all trajectories
    ad_mean = np.mean([results[t]['adaptive']['acc'] for t in traj_types])
    f1_mean = np.mean([results[t]['fixed_1.0']['acc'] for t in traj_types])
    f75_mean = np.mean([results[t]['fixed_0.75']['acc'] for t in traj_types])
    f50_mean = np.mean([results[t]['fixed_0.50']['acc'] for t in traj_types])
    bw_mean = np.mean([(1-results[t]['adaptive']['bw'])*100 for t in traj_types])
    print(f"{'MEAN':<16} {ad_mean:>9.2f}% {f1_mean:>9.2f}% {f75_mean:>9.2f}% "
          f"{f50_mean:>9.2f}% {bw_mean:>8.0f}%")

    # ============================================================
    # FIGURE: Multi-trace visualization
    # ============================================================
    fig, axes = plt.subplots(2, 1, figsize=(7.16, 4.5), gridspec_kw={'height_ratios': [1.5, 1]})

    # Top: Trajectories
    ax = axes[0]
    colors = ['#1E88E5', '#E53935', '#43A047', '#FB8C00', '#8E24AA']
    for ti, (ttype, tname) in enumerate(zip(traj_types, traj_names)):
        np.random.seed(42 + ti)
        traj = generate_trajectory(ttype, N_STEPS)
        ax.plot(range(N_STEPS), traj * 100, '-', color=colors[ti], lw=0.8, alpha=0.8, label=tname)
    ax.set_ylabel('BER (%)')
    ax.set_title('(a) Channel Quality Trajectories')
    ax.legend(fontsize=6, ncol=3, loc='upper center')
    ax.grid(True, alpha=0.2)
    ax.set_xlim([0, N_STEPS])

    # Bottom: Bar chart
    ax = axes[1]
    x = np.arange(len(traj_types))
    w = 0.18
    bars_ad = [results[t]['adaptive']['acc'] for t in traj_types]
    bars_f1 = [results[t]['fixed_1.0']['acc'] for t in traj_types]
    bars_f75 = [results[t]['fixed_0.75']['acc'] for t in traj_types]
    bars_f50 = [results[t]['fixed_0.50']['acc'] for t in traj_types]
    ax.bar(x - 1.5*w, bars_ad, w, label='Adaptive', color='#1E88E5', edgecolor='white', lw=0.3)
    ax.bar(x - 0.5*w, bars_f1, w, label='Fixed ρ=1.0', color='#78909C', edgecolor='white', lw=0.3)
    ax.bar(x + 0.5*w, bars_f75, w, label='Fixed ρ=0.75', color='#FFB74D', edgecolor='white', lw=0.3)
    ax.bar(x + 1.5*w, bars_f50, w, label='Fixed ρ=0.50', color='#EF9A9A', edgecolor='white', lw=0.3)
    ax.set_xticks(x); ax.set_xticklabels([n.replace(' ', '\n') for n in traj_names], fontsize=6)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('(b) Accuracy by Strategy and Trajectory')
    ax.legend(fontsize=5.5, ncol=4, loc='lower left')
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylim([80, 100])

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'v2_multi_trace.png'), facecolor='white')
    print(f"\n  Saved v2_multi_trace.png")
    plt.close()

    # Save results
    with open(os.path.join(OUT_DIR, 'v2_multi_trace_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved v2_multi_trace_results.json")
