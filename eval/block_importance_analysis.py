"""
Block Importance Analysis — Professor's Sanity Check Plots

Generates four sets of figures per dataset (14×14 = 196 blocks):
  A. Per-image sorted score curves (mean ± std band + example images)
  B. Per-image variance/spread across all test images
  C. Spread metric distributions (variance, max-min, CoV, Gini)
  D. Accuracy vs ρ sweep (learned vs random, including ρ=0.1)

Usage:
  python eval/block_importance_analysis.py
  python eval/block_importance_analysis.py --skip_sweep   # faster, skip ρ sweep
"""

import sys, os, json, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from train.run_final_pipeline import SpikeAdaptSC_v5c_NA, RESISC45Dataset, AIDDataset5050
from train.train_aid_v2 import ResNet50Front, ResNet50Back
import torchvision.transforms as T
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_STEPS = 8


def load_model(dataset, seed=42):
    """Load trained model for a dataset."""
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                     T.Normalize((.485,.456,.406), (.229,.224,.225))])

    if dataset == 'aid':
        n_cls = 30
        test_ds = AIDDataset5050("./data", tf, 'test', seed=42)
    else:
        n_cls = 45
        test_ds = RESISC45Dataset("./data", tf, 'test', train_ratio=0.20, seed=42)

    # Find checkpoints
    ckpt_path = None
    for pat in [f"./snapshots_{dataset}_v5cna_seed{seed}/v5cna_best_*.pth"]:
        matches = glob.glob(pat)
        if matches:
            ckpt_path = sorted(matches)[-1]
            break
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found for {dataset} seed {seed}")

    bb_path = None
    for pat in [f"./snapshots_{dataset}_5050_seed{seed}/backbone_best.pth",
                f"./snapshots_{dataset}_seed{seed}/backbone_best.pth"]:
        if os.path.exists(pat):
            bb_path = pat; break
    if bb_path is None:
        for pat in [f"./snapshots_{dataset}_*_seed{seed}/backbone_best.pth"]:
            matches = glob.glob(pat)
            if matches:
                bb_path = matches[0]; break

    test_loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Backbone: {bb_path}")
    print(f"  Test set: {len(test_ds)} images, {n_cls} classes")

    front = ResNet50Front(grid_size=14).to(device)
    bb = torch.load(bb_path, map_location=device)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))},
                          strict=False)
    front.eval()

    model = SpikeAdaptSC_v5c_NA(C_in=1024, C1=256, C2=36, T=T_STEPS,
                                 target_rate=0.75, grid_size=14).to(device)
    back = ResNet50Back(n_cls).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck['model'])
    back.load_state_dict(ck['back'])
    model.eval(); back.eval(); front.eval()

    return model, back, front, test_loader, n_cls


def extract_all_scores(model, front, test_loader, ber=0.0):
    """Extract block importance scores for all test images via model forward."""
    all_scores = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            feats = front(imgs)
            _, stats = model(feats, noise_param=ber)
            # importance: (B, H, W) → flatten to (B, H*W)
            imp = stats['importance']
            all_scores.append(imp.view(imp.size(0), -1).cpu().numpy())
    return np.concatenate(all_scores, axis=0)


def compute_spread_metrics(scores):
    """Compute per-image spread metrics."""
    n_imgs = scores.shape[0]
    metrics = {
        'variance': np.var(scores, axis=1),
        'max_min': np.max(scores, axis=1) - np.min(scores, axis=1),
        'std': np.std(scores, axis=1),
        'coeff_var': np.std(scores, axis=1) / (np.mean(scores, axis=1) + 1e-8),
    }
    # Gini
    gini = np.zeros(n_imgs)
    for i in range(n_imgs):
        s = np.sort(scores[i])
        n = len(s)
        idx = np.arange(1, n + 1)
        gini[i] = (2 * np.sum(idx * s) / (n * np.sum(s) + 1e-8)) - (n + 1) / n
    metrics['gini'] = gini
    # Entropy ratio
    ent = np.zeros(n_imgs)
    for i in range(n_imgs):
        p = scores[i] / (scores[i].sum() + 1e-8)
        p = np.clip(p, 1e-10, 1.0)
        e = -np.sum(p * np.log(p))
        ent[i] = e / np.log(len(p)) if np.log(len(p)) > 0 else 0
    metrics['entropy_ratio'] = ent
    return metrics


def evaluate_at_rho(model, back, front, test_loader, ber, rho):
    """Evaluate accuracy at a specific ρ using learned mask."""
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            Fp, _ = model(front(imgs), noise_param=ber, target_rate_override=rho)
            pred = back(Fp).argmax(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total


def evaluate_random_at_rho(model, back, front, test_loader, ber, rho, n_draws=50):
    """Evaluate random mask accuracy by temporarily replacing the block_mask."""
    H, W = 14, 14
    k = max(1, int(rho * H * W))
    draw_accs = []

    # Save original block_mask forward
    orig_forward = model.block_mask.forward

    class RandomMask:
        def __init__(self, k, H, W):
            self.k, self.H, self.W = k, H, W
        def __call__(self, importance, training=False):
            B = importance.size(0)
            mask = torch.zeros(B, self.H * self.W, device=importance.device)
            for b in range(B):
                idx = torch.randperm(self.H * self.W, device=importance.device)[:self.k]
                mask[b, idx] = 1.0
            mask = mask.view(B, 1, self.H, self.W)  # (B, 1, H, W) — matches LearnedBlockMask
            tx = torch.tensor(self.k / (self.H * self.W))
            return mask, tx

    rand_mask_fn = RandomMask(k, H, W)

    for draw in range(n_draws):
        # Monkey-patch
        model.block_mask.forward = rand_mask_fn
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                Fp, _ = model(front(imgs), noise_param=ber)
                pred = back(Fp).argmax(1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)
        draw_accs.append(100. * correct / total)

    # Restore
    model.block_mask.forward = orig_forward
    return np.mean(draw_accs), np.std(draw_accs)


# ─── PLOTTING ───

def set_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
        'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'axes.grid': True, 'grid.alpha': 0.3,
    })


def plot_sorted_score_curves(scores, ds_name, save_dir):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    n_blocks = scores.shape[1]
    idx = np.arange(1, n_blocks + 1)
    sorted_s = np.sort(scores, axis=1)[:, ::-1]

    ax = axes[0]
    n_show = min(300, sorted_s.shape[0])
    for i in range(n_show):
        ax.plot(idx, sorted_s[i], color='steelblue', alpha=0.02, linewidth=0.4)
    mu = sorted_s.mean(0); sd = sorted_s.std(0)
    ax.plot(idx, mu, color='darkblue', lw=2, label='Mean')
    ax.fill_between(idx, mu - sd, mu + sd, color='steelblue', alpha=0.3, label='±1 std')
    ax.set_xlabel('Ranked Block Index'); ax.set_ylabel('Importance Score')
    ax.set_title(f'{ds_name} — Sorted Block Scores (All Images)')
    ax.legend(); ax.set_xlim(1, n_blocks)

    ax = axes[1]
    var = np.var(scores, axis=1)
    pcts = [10, 30, 50, 70, 90]
    colors = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3']
    for pct, c in zip(pcts, colors):
        thr = np.percentile(var, pct)
        j = np.argmin(np.abs(var - thr))
        ax.plot(idx, sorted_s[j], c=c, lw=1.5, label=f'P{pct} (var={var[j]:.4f})')
    ax.set_xlabel('Ranked Block Index'); ax.set_ylabel('Importance Score')
    ax.set_title(f'{ds_name} — Example Images at Different Spread Levels')
    ax.legend(fontsize=8); ax.set_xlim(1, n_blocks)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(save_dir, f'fig_block_scores_{ds_name.lower()}.{ext}'))
    plt.close(fig)
    print(f"  Saved: fig_block_scores_{ds_name.lower()}.pdf")


def plot_per_image_variance(scores, ds_name, save_dir):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    var = np.var(scores, axis=1)

    ax = axes[0]
    ax.scatter(np.arange(1, len(var)+1), var, s=1, alpha=0.4, c='steelblue')
    ax.axhline(np.mean(var), c='red', ls='--', lw=1.5, label=f'Mean={np.mean(var):.4f}')
    ax.axhline(np.median(var), c='orange', ls=':', lw=1.5, label=f'Median={np.median(var):.4f}')
    ax.set_xlabel('Image Index'); ax.set_ylabel('Score Variance')
    ax.set_title(f'{ds_name} — Per-Image Score Variance'); ax.legend()

    ax = axes[1]
    ax.hist(var, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(np.mean(var), c='red', ls='--', lw=1.5, label=f'Mean={np.mean(var):.4f}')
    ax.axvline(np.median(var), c='orange', ls=':', lw=1.5, label=f'Median={np.median(var):.4f}')
    ax.set_xlabel('Score Variance'); ax.set_ylabel('Count')
    ax.set_title(f'{ds_name} — Distribution of Per-Image Variance'); ax.legend()

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(save_dir, f'fig_per_image_variance_{ds_name.lower()}.{ext}'))
    plt.close(fig)
    print(f"  Saved: fig_per_image_variance_{ds_name.lower()}.pdf")


def plot_spread_metrics(metrics, ds_name, save_dir):
    set_style()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    names = ['variance', 'max_min', 'std', 'coeff_var', 'gini', 'entropy_ratio']
    titles = ['Variance', 'Max − Min', 'Std Dev', 'Coeff. of Variation', 'Gini Index', 'Entropy Ratio']
    for ax, n, t in zip(axes.flat, names, titles):
        v = metrics[n]
        ax.hist(v, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
        ax.axvline(np.mean(v), c='red', ls='--', lw=1.5)
        ax.set_title(f'{t}\nμ={np.mean(v):.4f}, σ={np.std(v):.4f}')
        ax.set_xlabel(t); ax.set_ylabel('Count')
    fig.suptitle(f'{ds_name} — Spread Metric Distributions (196 blocks/image)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(save_dir, f'fig_spread_metrics_{ds_name.lower()}.{ext}'))
    plt.close(fig)
    print(f"  Saved: fig_spread_metrics_{ds_name.lower()}.pdf")


def plot_rho_sweep(results, ds_name, save_dir):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rhos = sorted(results.keys())
    for i, ber in enumerate(['0.0', '0.3']):
        ax = axes[i]
        learned = [results[r]['learned'][ber] for r in rhos]
        rmean = [results[r]['random'][ber]['mean'] for r in rhos]
        rstd = [results[r]['random'][ber]['std'] for r in rhos]
        ax.plot(rhos, learned, 'o-', c='#e41a1c', lw=2, ms=6, label='Learned mask', zorder=3)
        ax.errorbar(rhos, rmean, yerr=rstd, fmt='s--', c='#377eb8', lw=1.5, ms=5, capsize=3,
                    label='Random mask (50 draws)', zorder=2)
        for r, l, rm in zip(rhos, learned, rmean):
            gap = l - rm
            if abs(gap) > 0.3:
                ax.annotate(f'{gap:+.1f}', xy=(r, max(l, rm) + 0.3), fontsize=7, ha='center')
        ber_l = 'Clean' if ber == '0.0' else f'BER={ber}'
        ax.set_xlabel('Retention Rate ρ'); ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{ds_name} — {ber_l}')
        ax.legend(loc='lower right'); ax.set_xticks(rhos)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(save_dir, f'fig_rho_sweep_learned_vs_random_{ds_name.lower()}.{ext}'))
    plt.close(fig)
    print(f"  Saved: fig_rho_sweep_learned_vs_random_{ds_name.lower()}.pdf")


# ─── MAIN ───

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='both', choices=['aid', 'resisc45', 'both'])
    parser.add_argument('--save_dir', default='paper/figures')
    parser.add_argument('--skip_sweep', action='store_true')
    parser.add_argument('--n_random', type=int, default=50)
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    datasets = ['aid', 'resisc45'] if args.dataset == 'both' else [args.dataset]
    all_results = {}

    for ds in datasets:
        ds_name = 'AID' if ds == 'aid' else 'RESISC45'
        print(f"\n{'='*60}")
        print(f"  Block Importance Analysis: {ds_name} (14×14 = 196 blocks)")
        print(f"{'='*60}")

        model, back, front, test_loader, n_cls = load_model(ds)

        # A: Sorted score curves
        print("\n  [A] Extracting block importance scores...")
        scores = extract_all_scores(model, front, test_loader, ber=0.0)
        print(f"      Shape: {scores.shape}, range: [{scores.min():.4f}, {scores.max():.4f}]")
        plot_sorted_score_curves(scores, ds_name, args.save_dir)

        # B: Per-image variance
        print("\n  [B] Per-image variance...")
        plot_per_image_variance(scores, ds_name, args.save_dir)

        # C: Spread metrics
        print("\n  [C] Spread metrics...")
        metrics = compute_spread_metrics(scores)
        plot_spread_metrics(metrics, ds_name, args.save_dir)

        print(f"\n  {'Metric':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print(f"  {'-'*60}")
        for k in ['variance', 'max_min', 'std', 'coeff_var', 'gini', 'entropy_ratio']:
            v = metrics[k]
            print(f"  {k:<20} {np.mean(v):>10.4f} {np.std(v):>10.4f} {np.min(v):>10.4f} {np.max(v):>10.4f}")

        # D: ρ sweep
        if not args.skip_sweep:
            print(f"\n  [D] ρ sweep (learned vs random)...")
            rhos = [0.10, 0.25, 0.50, 0.625, 0.75, 1.0]
            sweep = {}
            for rho in rhos:
                sweep[rho] = {'learned': {}, 'random': {}}
                for ber in [0.0, 0.30]:
                    l_acc = evaluate_at_rho(model, back, front, test_loader, ber, rho)
                    nd = min(args.n_random, 10) if rho >= 0.75 else args.n_random
                    r_mean, r_std = evaluate_random_at_rho(model, back, front, test_loader,
                                                           ber, rho, n_draws=nd)
                    sweep[rho]['learned'][str(ber)] = l_acc
                    sweep[rho]['random'][str(ber)] = {'mean': r_mean, 'std': r_std}
                    print(f"    ρ={rho:.3f} BER={ber:.2f}: Learned={l_acc:.2f}%  "
                          f"Random={r_mean:.2f}±{r_std:.2f}%  Δ={l_acc - r_mean:+.2f}")
            plot_rho_sweep(sweep, ds_name, args.save_dir)
            all_results[ds] = sweep

        # Save
        ms = {k: {'mean': float(np.mean(v)), 'std': float(np.std(v)),
                   'min': float(np.min(v)), 'max': float(np.max(v))}
              for k, v in metrics.items()}
        with open(f'eval/block_importance_{ds}.json', 'w') as f:
            json.dump(ms, f, indent=2)

        del model, back, front; torch.cuda.empty_cache()

    if all_results:
        with open('eval/rho_sweep_learned_vs_random.json', 'w') as f:
            json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("  BLOCK IMPORTANCE ANALYSIS COMPLETE")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
