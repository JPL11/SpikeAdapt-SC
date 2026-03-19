"""Professor Sanity Check — Full-dataset score analysis.

Generates publication-quality figures for both 8×8 (v1) and 14×14 (v2):

Figure 1: Sorted score profiles (all 2000 images overlaid + mean ± std)
Figure 2: Per-image score variance distribution (histogram + KDE)
Figure 3: Per-image CoV / Gini / Entropy / Score Range distributions
Figure 4: Accuracy vs ρ curve (ρ = 1.0, 0.75, 0.50, 0.25, 0.10) learned vs random
Figure 5: Within-image score distribution (violin plots for 10 sample images)
Figure 6: Variance vs image index (all 2000 images, sorted)
Figure 7: Gap-vs-variance scatter (per-image learned-random gap vs score variance)

Usage:
    python eval/sanity_check_full.py
"""

import os, sys, random, json, math
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
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# Nice publication style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────
# Import models
# ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, './train')

# v1 (8×8)
from train_aid import (
    ResNet50Front as Front_v1, ResNet50Back,
    SpikeAdaptSC, LearnedImportanceScorer
)
# v2 (14×14)
from train_aid_v2 import (
    ResNet50Front as Front_v2,
    SpikeAdaptSC_v2, ChannelConditionedScorer,
    AIDDataset, sample_noise
)


def compute_score_stats(scores_np):
    """Compute per-image statistics from score array (N_blocks,)."""
    mean = scores_np.mean()
    var = scores_np.var()
    std = scores_np.std()
    cov = std / (mean + 1e-8)
    score_range = scores_np.max() - scores_np.min()

    # Gini coefficient
    sorted_s = np.sort(scores_np)
    n = len(sorted_s)
    idx = np.arange(1, n + 1)
    gini = (2 * np.sum(idx * sorted_s)) / (n * np.sum(sorted_s) + 1e-8) - (n + 1) / n

    # Entropy ratio (vs uniform)
    p = scores_np / (scores_np.sum() + 1e-8)
    entropy = -np.sum(p * np.log(p + 1e-8))
    max_entropy = np.log(n)
    entropy_ratio = entropy / max_entropy

    return {
        'mean': mean, 'var': var, 'std': std, 'cov': cov,
        'gini': gini, 'entropy_ratio': entropy_ratio,
        'score_range': score_range,
        'min': scores_np.min(), 'max': scores_np.max()
    }


def extract_all_scores(front, model, loader, grid_label, is_v2=False):
    """Extract importance scores for ALL test images."""
    front.eval(); model.eval()
    all_scores = []  # list of (N_blocks,) arrays
    all_labels = []

    with torch.no_grad():
        for img, lab in tqdm(loader, desc=f"Extracting scores ({grid_label})"):
            img = img.to(device)
            feat = front(img)
            B = feat.shape[0]

            # Encode
            all_S2, m1, m2 = [], None, None
            for t in range(model.T):
                _, s2, m1, m2 = model.encoder(feat, m1, m2)
                all_S2.append(s2)

            # Score
            if is_v2:
                importance = model.scorer(all_S2, 0.0)  # (B, H, W)
            else:
                importance = model.importance_scorer(all_S2)  # (B, H, W)

            for b in range(B):
                scores = importance[b].cpu().numpy().flatten()
                all_scores.append(scores)
                all_labels.append(lab[b].item())

    return all_scores, all_labels


def evaluate_at_rate(front, model, back, loader, rho, is_v2=False, noise=0.0):
    """Evaluate accuracy at a specific masking rate."""
    front.eval(); model.eval(); back.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for img, lab in loader:
            img, lab = img.to(device), lab.to(device)
            feat = front(img)
            if is_v2:
                Fp, stats = model(feat, noise_param=noise,
                                   target_rate_override=rho)
            else:
                Fp, stats = model(feat, noise_param=noise,
                                   target_rate_override=rho)
            preds = back(Fp).argmax(1)
            correct += preds.eq(lab).sum().item()
            total += lab.size(0)
    return 100. * correct / total


def evaluate_random_at_rate(front, model, back, loader, rho, n_draws=50,
                             is_v2=False, noise=0.0):
    """Evaluate accuracy with random block selection at rate rho."""
    front.eval(); model.eval(); back.eval()
    accs = []
    for draw in range(n_draws):
        correct, total = 0, 0
        with torch.no_grad():
            for img, lab in loader:
                img, lab = img.to(device), lab.to(device)
                feat = front(img)
                B = feat.shape[0]

                # Encode
                all_S2, m1, m2 = [], None, None
                for t in range(model.T):
                    _, s2, m1, m2 = model.encoder(feat, m1, m2)
                    all_S2.append(s2)

                H, W = all_S2[0].shape[2], all_S2[0].shape[3]
                k = max(1, int(rho * H * W))

                # Random mask
                rand_flat = torch.rand(B, H * W, device=device)
                _, idx = rand_flat.topk(k, dim=1)
                mask = torch.zeros(B, H * W, device=device)
                mask.scatter_(1, idx, 1.0)
                mask = mask.view(B, 1, H, W)

                # Channel
                if is_v2:
                    recv = [model.channel(all_S2[t] * mask, noise)
                            for t in range(model.T)]
                else:
                    recv = [model.channel(all_S2[t] * mask, noise)
                            for t in range(model.T)]

                Fp = model.decoder(recv, mask)
                preds = back(Fp).argmax(1)
                correct += preds.eq(lab).sum().item()
                total += lab.size(0)
        accs.append(100. * correct / total)
    return np.mean(accs), np.std(accs), accs


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    OUT_DIR = "./paper/figures/"
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("PROFESSOR SANITY CHECK — Full dataset analysis")
    print("  Top-1 classification accuracy on 2,000-image AID test set")
    print("=" * 70)

    # Dataset
    test_tf = T.Compose([
        T.Resize(256), T.CenterCrop(224),
        T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))
    ])
    test_ds = AIDDataset("./data", transform=test_tf, split='test', seed=42)
    test_loader = DataLoader(test_ds, 64, shuffle=False, num_workers=4,
                              pin_memory=True)

    # ─── LOAD v1 (8×8, C2=128) ───
    print("\n--- Loading v1 (8×8, C2=128) ---")
    front_v1 = Front_v1(input_size=224).to(device)
    back_v1 = ResNet50Back(30).to(device)
    model_v1 = SpikeAdaptSC(C_in=1024, C1=256, C2=128, T=8,
                             target_rate=0.75, channel_type='bsc').to(device)

    bb_path = "./snapshots_aid/backbone_best.pth"
    state = torch.load(bb_path, map_location=device)
    front_v1.load_state_dict({k: v for k, v in state.items()
                               if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))},
                              strict=False)

    v1_ck = torch.load("./snapshots_aid/bsc_s3_96.10.pth", map_location=device)
    if 'model' in v1_ck:
        model_v1.load_state_dict(v1_ck['model'])
        back_v1.load_state_dict(v1_ck['back'])
    else:
        model_v1.load_state_dict(v1_ck)
        back_v1.load_state_dict({k: v for k, v in state.items()
                                  if k.startswith(('layer4.', 'fc.', 'avgpool.'))},
                                 strict=False)
    front_v1.eval(); model_v1.eval(); back_v1.eval()
    for p in front_v1.parameters(): p.requires_grad = False
    print("  ✓ v1 loaded")

    # ─── LOAD v2 (14×14, C2=36) ───
    print("--- Loading v2 (14×14, C2=36) ---")
    front_v2 = Front_v2(grid_size=14).to(device)
    back_v2 = ResNet50Back(30).to(device)
    model_v2 = SpikeAdaptSC_v2(C_in=1024, C1=256, C2=36, T=8,
                                 target_rate=0.75, channel_type='bsc',
                                 grid_size=14).to(device)

    front_v2.load_state_dict({k: v for k, v in state.items()
                               if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))},
                              strict=False)
    v2_ck = torch.load("./snapshots_aid_v2_seed42/v2_s3_96.30.pth",
                        map_location=device)
    model_v2.load_state_dict(v2_ck['model'])
    back_v2.load_state_dict(v2_ck['back'])
    front_v2.eval(); model_v2.eval(); back_v2.eval()
    for p in front_v2.parameters(): p.requires_grad = False
    print("  ✓ v2 loaded")

    # ─────────────────────────────────────────────────────────────────
    # STEP 1: Extract all scores
    # ─────────────────────────────────────────────────────────────────
    print("\n[1/7] Extracting scores for all 2000 test images...")
    scores_v1, labels_v1 = extract_all_scores(front_v1, model_v1, test_loader,
                                               "8×8", is_v2=False)
    scores_v2, labels_v2 = extract_all_scores(front_v2, model_v2, test_loader,
                                               "14×14", is_v2=True)

    # Compute per-image statistics
    stats_v1 = [compute_score_stats(s) for s in scores_v1]
    stats_v2 = [compute_score_stats(s) for s in scores_v2]

    print(f"  v1: {len(scores_v1)} images, {len(scores_v1[0])} blocks each")
    print(f"  v2: {len(scores_v2)} images, {len(scores_v2[0])} blocks each")

    # Summary table
    for label, stats_list in [("v1 (8×8)", stats_v1), ("v2 (14×14)", stats_v2)]:
        vars_ = [s['var'] for s in stats_list]
        covs = [s['cov'] for s in stats_list]
        ginis = [s['gini'] for s in stats_list]
        entrs = [s['entropy_ratio'] for s in stats_list]
        ranges = [s['score_range'] for s in stats_list]
        print(f"\n  {label} — Per-image statistics over {len(stats_list)} images:")
        print(f"    Variance:      {np.mean(vars_):.6f} ± {np.std(vars_):.6f}")
        print(f"    CoV:           {np.mean(covs):.4f} ± {np.std(covs):.4f}")
        print(f"    Gini:          {np.mean(ginis):.4f} ± {np.std(ginis):.4f}")
        print(f"    Entropy ratio: {np.mean(entrs):.6f} ± {np.std(entrs):.6f}")
        print(f"    Score range:   {np.mean(ranges):.4f} ± {np.std(ranges):.4f}")

    # ─────────────────────────────────────────────────────────────────
    # FIGURE 1: Sorted score profiles (professor's upper-left sketch)
    # ─────────────────────────────────────────────────────────────────
    print("\n[2/7] Figure 1: Sorted score profiles...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, scores_list, label, n_blocks in [
        (axes[0], scores_v1, "v1 (8×8, 64 blocks)", 64),
        (axes[1], scores_v2, "v2 (14×14, 196 blocks)", 196)
    ]:
        # Sort each image's scores descending
        sorted_profiles = np.array([np.sort(s)[::-1] for s in scores_list])
        x = np.arange(1, n_blocks + 1)

        # Plot all 2000 profiles (very faint)
        for i in range(len(sorted_profiles)):
            ax.plot(x, sorted_profiles[i], color='steelblue', alpha=0.015,
                    linewidth=0.5)

        # Mean ± std band
        mean_profile = sorted_profiles.mean(axis=0)
        std_profile = sorted_profiles.std(axis=0)
        ax.plot(x, mean_profile, color='darkblue', linewidth=2.5,
                label='Mean', zorder=5)
        ax.fill_between(x, mean_profile - std_profile,
                        mean_profile + std_profile,
                        color='royalblue', alpha=0.3, label='±1 std', zorder=4)

        # Reference: perfectly uniform
        uniform_val = 1.0 / n_blocks * n_blocks  # normalized
        ax.axhline(y=mean_profile.mean(), color='red', linestyle='--',
                   alpha=0.7, linewidth=1.5, label='Uniform')

        ax.set_xlabel("Block rank (sorted descending)")
        ax.set_ylabel("Importance score")
        ax.set_title(f"Sorted Score Profiles – {label}\n(2,000 images overlaid)")
        ax.legend(loc='upper right')
        ax.set_xlim(1, n_blocks)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "sanity_sorted_profiles.png"))
    plt.close()
    print("  ✓ Saved sanity_sorted_profiles.png")

    # ─────────────────────────────────────────────────────────────────
    # FIGURE 2: Per-image variance distribution
    # ─────────────────────────────────────────────────────────────────
    print("[3/7] Figure 2: Per-image variance distribution...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, stats_list, label, color in [
        (axes[0], stats_v1, "v1 (8×8)", '#2196F3'),
        (axes[1], stats_v2, "v2 (14×14)", '#FF5722')
    ]:
        vars_ = [s['var'] for s in stats_list]
        ax.hist(vars_, bins=60, color=color, alpha=0.7, edgecolor='white',
                linewidth=0.5)
        ax.axvline(np.mean(vars_), color='black', linestyle='--', linewidth=2,
                   label=f'Mean = {np.mean(vars_):.6f}')
        ax.axvline(np.median(vars_), color='green', linestyle=':', linewidth=2,
                   label=f'Median = {np.median(vars_):.6f}')
        ax.set_xlabel("Per-image score variance")
        ax.set_ylabel("Count (out of 2,000 images)")
        ax.set_title(f"Score Variance Distribution – {label}")
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "sanity_variance_dist.png"))
    plt.close()
    print("  ✓ Saved sanity_variance_dist.png")

    # ─────────────────────────────────────────────────────────────────
    # FIGURE 3: Multi-metric distributions (CoV, Gini, Entropy, Range)
    # ─────────────────────────────────────────────────────────────────
    print("[4/7] Figure 3: Multi-metric distributions...")

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    metrics = ['cov', 'gini', 'entropy_ratio', 'score_range']
    metric_labels = ['CoV (std/mean)', 'Gini coefficient',
                     'Entropy ratio (vs uniform)', 'Score range (max - min)']

    for row, (stats_list, grid_label, color) in enumerate([
        (stats_v1, "v1 (8×8)", '#2196F3'),
        (stats_v2, "v2 (14×14)", '#FF5722')
    ]):
        for col, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row, col]
            vals = [s[metric] for s in stats_list]
            ax.hist(vals, bins=50, color=color, alpha=0.7, edgecolor='white',
                    linewidth=0.5)
            ax.axvline(np.mean(vals), color='black', linestyle='--',
                       linewidth=1.5, label=f'μ={np.mean(vals):.4f}')
            ax.set_xlabel(mlabel)
            if col == 0:
                ax.set_ylabel(f"{grid_label}\nCount")
            ax.legend(fontsize=8)
            ax.set_title(f"{mlabel}" if row == 0 else "")

    plt.suptitle("Per-Image Score Statistics (2,000 AID test images)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "sanity_multi_metrics.png"))
    plt.close()
    print("  ✓ Saved sanity_multi_metrics.png")

    # ─────────────────────────────────────────────────────────────────
    # FIGURE 4: Accuracy vs ρ (learned vs random)
    # ─────────────────────────────────────────────────────────────────
    print("[5/7] Figure 4: Accuracy vs ρ curve...")

    rhos = [1.0, 0.75, 0.50, 0.25, 0.10]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, front, model, back, is_v2, label, color in [
        (axes[0], front_v1, model_v1, back_v1, False, "v1 (8×8, C2=128)", '#2196F3'),
        (axes[1], front_v2, model_v2, back_v2, True, "v2 (14×14, C2=36)", '#FF5722')
    ]:
        learned_accs = []
        random_means = []
        random_stds = []

        for rho in rhos:
            print(f"    {label} ρ={rho}...")

            if rho == 1.0:
                acc_l = evaluate_at_rate(front, model, back, test_loader,
                                         rho, is_v2=is_v2)
                learned_accs.append(acc_l)
                random_means.append(acc_l)
                random_stds.append(0.0)
            else:
                acc_l = evaluate_at_rate(front, model, back, test_loader,
                                         rho, is_v2=is_v2)
                learned_accs.append(acc_l)

                r_mean, r_std, _ = evaluate_random_at_rate(
                    front, model, back, test_loader, rho, n_draws=50,
                    is_v2=is_v2)
                random_means.append(r_mean)
                random_stds.append(r_std)

            print(f"      Learned: {learned_accs[-1]:.2f}%, "
                  f"Random: {random_means[-1]:.2f} ± {random_stds[-1]:.2f}%")

        learned_accs = np.array(learned_accs)
        random_means = np.array(random_means)
        random_stds = np.array(random_stds)

        ax.plot(rhos, learned_accs, 'o-', color=color, linewidth=2.5,
                markersize=8, label='Learned mask', zorder=5)
        ax.errorbar(rhos, random_means, yerr=random_stds, fmt='s--',
                    color='gray', linewidth=2, markersize=7,
                    capsize=5, label='Random mask (50 draws)', zorder=4)

        # Annotate gaps
        for i, rho in enumerate(rhos):
            if rho < 1.0:
                gap = learned_accs[i] - random_means[i]
                if abs(gap) > 0.1:
                    ax.annotate(f'{gap:+.1f}pp',
                                xy=(rho, (learned_accs[i] + random_means[i]) / 2),
                                fontsize=9, ha='left', fontweight='bold',
                                color='darkred')

        ax.set_xlabel("Masking rate ρ (fraction of blocks kept)")
        ax.set_ylabel("Top-1 Accuracy (%)")
        ax.set_title(f"Accuracy vs Rate – {label}")
        ax.legend(loc='lower left')
        ax.set_xticks(rhos)
        ax.set_xlim(0.05, 1.05)
        ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "sanity_acc_vs_rho.png"))
    plt.close()
    print("  ✓ Saved sanity_acc_vs_rho.png")

    # ─────────────────────────────────────────────────────────────────
    # FIGURE 5: Within-image score distributions (10 sample images)
    # ─────────────────────────────────────────────────────────────────
    print("[6/7] Figure 5: Within-image score distributions (10 samples)...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Pick 10 evenly-spaced images by variance (low→high)
    for ax, scores_list, stats_list, label, color in [
        (axes[0], scores_v1, stats_v1, "v1 (8×8)", '#2196F3'),
        (axes[1], scores_v2, stats_v2, "v2 (14×14)", '#FF5722')
    ]:
        vars_sorted_idx = np.argsort([s['var'] for s in stats_list])
        sample_idx = vars_sorted_idx[np.linspace(0, len(vars_sorted_idx)-1,
                                                   10, dtype=int)]

        positions = range(10)
        violin_data = [scores_list[i] for i in sample_idx]
        parts = ax.violinplot(violin_data, positions=positions, showmeans=True,
                               showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)

        # Annotate with variance
        for j, i in enumerate(sample_idx):
            v = stats_list[i]['var']
            ax.text(j, ax.get_ylim()[1] * 0.95, f'var={v:.5f}',
                    ha='center', fontsize=7, rotation=45)

        ax.set_xlabel("Image index (sorted by increasing variance)")
        ax.set_ylabel("Block importance score")
        ax.set_title(f"Within-Image Score Distribution – {label} (10 samples)")
        ax.set_xticks(positions)
        ax.set_xticklabels([f'img {i}' for i in sample_idx], fontsize=8,
                            rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "sanity_within_image_dist.png"))
    plt.close()
    print("  ✓ Saved sanity_within_image_dist.png")

    # ─────────────────────────────────────────────────────────────────
    # FIGURE 6: Variance across all 2000 images (sorted)
    # ─────────────────────────────────────────────────────────────────
    print("[7/7] Figure 6: Variance across all 2000 images...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, stats_list, label, color in [
        (axes[0], stats_v1, "v1 (8×8)", '#2196F3'),
        (axes[1], stats_v2, "v2 (14×14)", '#FF5722')
    ]:
        vars_ = np.array(sorted([s['var'] for s in stats_list]))
        x = np.arange(len(vars_))

        ax.fill_between(x, 0, vars_, color=color, alpha=0.4)
        ax.plot(x, vars_, color=color, linewidth=1)
        ax.axhline(np.mean(vars_), color='black', linestyle='--',
                   linewidth=1.5, label=f'Mean = {np.mean(vars_):.6f}')
        ax.axhline(np.median(vars_), color='green', linestyle=':',
                   linewidth=1.5, label=f'Median = {np.median(vars_):.6f}')
        # Mark 90th and 99th percentile
        p90 = np.percentile(vars_, 90)
        p99 = np.percentile(vars_, 99)
        ax.axhline(p90, color='orange', linestyle='-.', linewidth=1,
                   label=f'P90 = {p90:.6f}')
        ax.axhline(p99, color='red', linestyle='-.', linewidth=1,
                   label=f'P99 = {p99:.6f}')

        ax.set_xlabel("Image index (sorted by variance)")
        ax.set_ylabel("Per-image score variance")
        ax.set_title(f"Score Variance – {label}\n(all 2,000 images, sorted)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "sanity_variance_sorted.png"))
    plt.close()
    print("  ✓ Saved sanity_variance_sorted.png")

    # ─────────────────────────────────────────────────────────────────
    # FIGURE 7: Combined dashboard (professor's sketch layout)
    # ─────────────────────────────────────────────────────────────────
    print("\n[BONUS] Figure 7: Combined dashboard...")

    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

    # Panel A (top-left, 2 cols): Sorted profiles for v2
    ax_a = fig.add_subplot(gs[0, 0:2])
    sorted_profiles_v2 = np.array([np.sort(s)[::-1] for s in scores_v2])
    n_blocks = 196
    x = np.arange(1, n_blocks + 1)
    for i in range(len(sorted_profiles_v2)):
        ax_a.plot(x, sorted_profiles_v2[i], color='#FF5722', alpha=0.012,
                  linewidth=0.4)
    mean_p = sorted_profiles_v2.mean(axis=0)
    std_p = sorted_profiles_v2.std(axis=0)
    ax_a.plot(x, mean_p, color='darkred', linewidth=2.5, label='Mean')
    ax_a.fill_between(x, mean_p - std_p, mean_p + std_p,
                      color='#FF5722', alpha=0.25, label='±1 std')
    ax_a.set_xlabel("Block rank (sorted desc)")
    ax_a.set_ylabel("Score")
    ax_a.set_title("(A) Sorted Score Profiles — v2 (14×14, 196 blocks)")
    ax_a.legend()

    # Panel B (top-right, 2 cols): Sorted profiles for v1
    ax_b = fig.add_subplot(gs[0, 2:4])
    sorted_profiles_v1 = np.array([np.sort(s)[::-1] for s in scores_v1])
    n_blocks_v1 = 64
    x1 = np.arange(1, n_blocks_v1 + 1)
    for i in range(len(sorted_profiles_v1)):
        ax_b.plot(x1, sorted_profiles_v1[i], color='#2196F3', alpha=0.012,
                  linewidth=0.4)
    mean_p1 = sorted_profiles_v1.mean(axis=0)
    std_p1 = sorted_profiles_v1.std(axis=0)
    ax_b.plot(x1, mean_p1, color='darkblue', linewidth=2.5, label='Mean')
    ax_b.fill_between(x1, mean_p1 - std_p1, mean_p1 + std_p1,
                      color='#2196F3', alpha=0.25, label='±1 std')
    ax_b.set_xlabel("Block rank (sorted desc)")
    ax_b.set_ylabel("Score")
    ax_b.set_title("(B) Sorted Score Profiles — v1 (8×8, 64 blocks)")
    ax_b.legend()

    # Panel C (middle-left): Variance histograms side by side
    ax_c = fig.add_subplot(gs[1, 0:2])
    vars_v1 = [s['var'] for s in stats_v1]
    vars_v2 = [s['var'] for s in stats_v2]
    ax_c.hist(vars_v1, bins=50, color='#2196F3', alpha=0.6, label='v1 (8×8)',
              edgecolor='white', linewidth=0.5)
    ax_c.hist(vars_v2, bins=50, color='#FF5722', alpha=0.6, label='v2 (14×14)',
              edgecolor='white', linewidth=0.5)
    ax_c.axvline(np.mean(vars_v1), color='#2196F3', linestyle='--', linewidth=2)
    ax_c.axvline(np.mean(vars_v2), color='#FF5722', linestyle='--', linewidth=2)
    ax_c.set_xlabel("Per-image score variance")
    ax_c.set_ylabel("Count")
    ax_c.set_title("(C) Per-Image Variance Distribution")
    ax_c.legend()

    # Panel D (middle-right): Gini + CoV box plots
    ax_d = fig.add_subplot(gs[1, 2:4])
    ginis_v1 = [s['gini'] for s in stats_v1]
    ginis_v2 = [s['gini'] for s in stats_v2]
    covs_v1 = [s['cov'] for s in stats_v1]
    covs_v2 = [s['cov'] for s in stats_v2]
    bp = ax_d.boxplot([ginis_v1, ginis_v2, covs_v1, covs_v2],
                       labels=['Gini\nv1', 'Gini\nv2', 'CoV\nv1', 'CoV\nv2'],
                       patch_artist=True)
    colors_bp = ['#2196F3', '#FF5722', '#2196F3', '#FF5722']
    for patch, c in zip(bp['boxes'], colors_bp):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax_d.set_title("(D) Gini & CoV Distributions")
    ax_d.set_ylabel("Value")

    # Panel E (bottom-left, 2 cols): Variance sorted for both
    ax_e = fig.add_subplot(gs[2, 0:2])
    v1_sorted = np.sort(vars_v1)
    v2_sorted = np.sort(vars_v2)
    x_v1 = np.linspace(0, 1, len(v1_sorted))
    x_v2 = np.linspace(0, 1, len(v2_sorted))
    ax_e.fill_between(x_v1, 0, v1_sorted, color='#2196F3', alpha=0.3)
    ax_e.plot(x_v1, v1_sorted, color='#2196F3', linewidth=1.5, label='v1 (8×8)')
    ax_e.fill_between(x_v2, 0, v2_sorted, color='#FF5722', alpha=0.3)
    ax_e.plot(x_v2, v2_sorted, color='#FF5722', linewidth=1.5, label='v2 (14×14)')
    ax_e.set_xlabel("Image percentile")
    ax_e.set_ylabel("Score variance")
    ax_e.set_title("(E) Variance CDF — All 2,000 Images")
    ax_e.legend()

    # Panel F (bottom-right, 2 cols): Summary table as text
    ax_f = fig.add_subplot(gs[2, 2:4])
    ax_f.axis('off')
    table_data = [
        ['Metric', 'v1 (8×8)', 'v2 (14×14)'],
        ['Mean var', f'{np.mean(vars_v1):.6f}', f'{np.mean(vars_v2):.6f}'],
        ['Mean CoV', f'{np.mean(covs_v1):.4f}', f'{np.mean(covs_v2):.4f}'],
        ['Mean Gini', f'{np.mean(ginis_v1):.4f}', f'{np.mean(ginis_v2):.4f}'],
        ['Mean entropy', f'{np.mean([s["entropy_ratio"] for s in stats_v1]):.6f}',
                         f'{np.mean([s["entropy_ratio"] for s in stats_v2]):.6f}'],
        ['Mean range', f'{np.mean([s["score_range"] for s in stats_v1]):.4f}',
                       f'{np.mean([s["score_range"] for s in stats_v2]):.4f}'],
        ['P99 var', f'{np.percentile(vars_v1, 99):.6f}',
                    f'{np.percentile(vars_v2, 99):.6f}'],
        ['Blocks', '64', '196'],
        ['C₂', '128', '36'],
        ['Payload/T', '8,192', '7,056'],
    ]
    table = ax_f.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    # Header styling
    for j in range(3):
        table[0, j].set_facecolor('#E0E0E0')
        table[0, j].set_text_props(fontweight='bold')
    ax_f.set_title("(F) Summary Statistics", pad=20)

    plt.suptitle("SpikeAdapt-SC — Full-Dataset Scoring Sanity Check\n"
                 "Top-1 accuracy on 2,000-image AID test set",
                 fontsize=16, y=1.01)
    plt.savefig(os.path.join(OUT_DIR, "sanity_dashboard.png"))
    plt.close()
    print("  ✓ Saved sanity_dashboard.png")

    # ─────────────────────────────────────────────────────────────────
    # Save raw data for professor
    # ─────────────────────────────────────────────────────────────────
    raw = {
        'v1_8x8': {
            'n_images': len(scores_v1),
            'n_blocks': len(scores_v1[0]),
            'mean_var': float(np.mean(vars_v1)),
            'std_var': float(np.std(vars_v1)),
            'mean_cov': float(np.mean(covs_v1)),
            'mean_gini': float(np.mean(ginis_v1)),
            'mean_entropy_ratio': float(np.mean([s['entropy_ratio'] for s in stats_v1])),
            'mean_score_range': float(np.mean([s['score_range'] for s in stats_v1])),
            'p90_var': float(np.percentile(vars_v1, 90)),
            'p99_var': float(np.percentile(vars_v1, 99)),
        },
        'v2_14x14': {
            'n_images': len(scores_v2),
            'n_blocks': len(scores_v2[0]),
            'mean_var': float(np.mean(vars_v2)),
            'std_var': float(np.std(vars_v2)),
            'mean_cov': float(np.mean(covs_v2)),
            'mean_gini': float(np.mean(ginis_v2)),
            'mean_entropy_ratio': float(np.mean([s['entropy_ratio'] for s in stats_v2])),
            'mean_score_range': float(np.mean([s['score_range'] for s in stats_v2])),
            'p90_var': float(np.percentile(vars_v2, 90)),
            'p99_var': float(np.percentile(vars_v2, 99)),
        },
        'accuracy_definition': 'Top-1 classification accuracy on the 2,000-image AID test set (80/20 split, seed=42).',
    }
    with open(os.path.join(OUT_DIR, "sanity_check_data.json"), 'w') as f:
        json.dump(raw, f, indent=2)
    print(f"\n✓ Saved sanity_check_data.json")
    print(f"\nAll figures saved to {OUT_DIR}")
    print("\nAccuracy definition: Top-1 classification accuracy on the 2,000-image "
          "AID test set (80/20 split, seed=42).")
