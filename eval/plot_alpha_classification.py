#!/usr/bin/env python3
"""Plot α metric for classification: α(ρ, BER) = (A(ρ) - A(ρ=1)) / A(ρ=1).

Shows how noise-aware masking at different ρ affects accuracy relative to
full transmission (ρ=1). Key insight: under high BER, masking can INCREASE
accuracy by focusing on important blocks.

Supports two data sources:
  1. eval/extended_ber_rho_results.json (extended, BER up to 0.4)
  2. eval/ablation_final_results.json (existing, BER ∈ {0.0, 0.15, 0.30})

Output: eval/figures/alpha_classification_{aid,resisc45}.pdf

Usage:
    python eval/plot_alpha_classification.py [--extended]
"""

import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def load_data(use_extended=False):
    """Load ρ×BER sweep data."""
    if use_extended:
        path = 'eval/extended_ber_rho_results.json'
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f), True
        print(f"Extended results not found at {path}, falling back to ablation data.")

    path = 'eval/ablation_final_results.json'
    with open(path) as f:
        raw = json.load(f)
    # Extract rho_sweep from ablation format
    data = {}
    for ds in ['aid', 'resisc45']:
        data[ds] = raw[ds]['rho_sweep']
    return data, False


def compute_alpha(data_ds):
    """Compute α(ρ, BER) = (A(ρ) - A(ρ=1)) / A(ρ=1) for each BER level."""
    rho_vals = sorted([float(r) for r in data_ds.keys()])
    ber_vals = sorted([float(b) for b in data_ds[str(rho_vals[0])].keys()])

    # Get A(ρ=1) for each BER
    a_full = {}
    for ber in ber_vals:
        key = '1.0'
        val = data_ds[key][str(ber)]
        a_full[ber] = val if isinstance(val, (int, float)) else val['mean']

    alpha = {}  # alpha[ber] = [(rho, alpha_val), ...]
    accuracy = {}  # accuracy[ber] = [(rho, acc), ...]
    for ber in ber_vals:
        alpha[ber] = []
        accuracy[ber] = []
        for rho in rho_vals:
            val = data_ds[str(rho)][str(ber)]
            acc = val if isinstance(val, (int, float)) else val['mean']
            a = (acc - a_full[ber]) / a_full[ber] if a_full[ber] > 0 else 0
            alpha[ber].append((rho, a))
            accuracy[ber].append((rho, acc))

    return alpha, accuracy, rho_vals, ber_vals


def plot_alpha(alpha, rho_vals, ber_vals, dataset_name, outdir):
    """Plot α vs ρ for different BER levels."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Color map: BER=0 is blue, higher BER is more red
    cmap = plt.cm.RdYlBu_r
    norm = plt.Normalize(0, max(ber_vals))

    for ber in ber_vals:
        rhos = [p[0] for p in alpha[ber]]
        alphas = [p[1] for p in alpha[ber]]
        color = cmap(norm(ber))
        label = f'BER={ber:.2f}' if ber > 0 else 'BER=0 (clean)'
        ax.plot(rhos, alphas, 'o-', color=color, label=label, linewidth=2, markersize=6)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_xlabel(r'Transmission rate $\rho$', fontsize=13)
    ax.set_ylabel(r'$\alpha = \frac{A(\rho) - A(\rho\!=\!1)}{A(\rho\!=\!1)}$', fontsize=14)
    ax.set_title(f'{dataset_name}: Relative accuracy change vs transmission rate', fontsize=13)
    ax.legend(fontsize=9, loc='lower right', ncol=2)
    ax.set_xlim(0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Annotate the "masking helps" region
    # Find BER levels where α > 0 for some ρ
    for ber in ber_vals:
        positive_rhos = [(r, a) for r, a in alpha[ber] if a > 0.005 and r < 1.0]
        if positive_rhos:
            best = max(positive_rhos, key=lambda x: x[1])
            ax.annotate(f'α>0\n(masking helps)',
                       xy=(best[0], best[1]),
                       xytext=(best[0]-0.15, best[1]+0.02),
                       fontsize=8, color='darkgreen',
                       arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.2))
            break  # Just annotate the most prominent one

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    out_pdf = os.path.join(outdir, f'alpha_classification_{dataset_name.lower()}.pdf')
    out_png = os.path.join(outdir, f'alpha_classification_{dataset_name.lower()}.png')
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_pdf}")
    return out_pdf


def plot_accuracy_vs_rho(accuracy, rho_vals, ber_vals, dataset_name, outdir):
    """Plot absolute accuracy vs ρ for different BER levels."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    cmap = plt.cm.RdYlBu_r
    norm = plt.Normalize(0, max(ber_vals))

    for ber in ber_vals:
        rhos = [p[0] for p in accuracy[ber]]
        accs = [p[1] for p in accuracy[ber]]
        color = cmap(norm(ber))
        label = f'BER={ber:.2f}' if ber > 0 else 'BER=0 (clean)'
        ax.plot(rhos, accs, 'o-', color=color, label=label, linewidth=2, markersize=6)

    ax.set_xlabel(r'Transmission rate $\rho$', fontsize=13)
    ax.set_ylabel('Classification Accuracy (%)', fontsize=13)
    ax.set_title(f'{dataset_name}: Accuracy vs transmission rate at different BER', fontsize=13)
    ax.legend(fontsize=9, loc='lower right', ncol=2)
    ax.set_xlim(0.05, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    out_pdf = os.path.join(outdir, f'accuracy_vs_rho_{dataset_name.lower()}.pdf')
    out_png = os.path.join(outdir, f'accuracy_vs_rho_{dataset_name.lower()}.png')
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_pdf}")
    return out_pdf


def plot_combined(data, outdir):
    """Combined 2×2 plot: accuracy and α for both datasets."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    datasets = [('aid', 'AID'), ('resisc45', 'RESISC45')]

    for col, (ds_key, ds_name) in enumerate(datasets):
        alpha, accuracy, rho_vals, ber_vals = compute_alpha(data[ds_key])
        cmap = plt.cm.RdYlBu_r
        norm = plt.Normalize(0, max(ber_vals))

        # Top row: absolute accuracy
        ax = axes[0, col]
        for ber in ber_vals:
            rhos = [p[0] for p in accuracy[ber]]
            accs = [p[1] for p in accuracy[ber]]
            color = cmap(norm(ber))
            label = f'BER={ber:.2f}' if ber > 0 else 'Clean'
            ax.plot(rhos, accs, 'o-', color=color, label=label, linewidth=2, markersize=5)
        ax.set_xlabel(r'$\rho$', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(f'{ds_name}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, ncol=2)
        ax.set_xlim(0.05, 1.05)
        ax.grid(True, alpha=0.3)

        # Bottom row: α metric
        ax = axes[1, col]
        for ber in ber_vals:
            rhos = [p[0] for p in alpha[ber]]
            alphas = [p[1] for p in alpha[ber]]
            color = cmap(norm(ber))
            label = f'BER={ber:.2f}' if ber > 0 else 'Clean'
            ax.plot(rhos, alphas, 'o-', color=color, label=label, linewidth=2, markersize=5)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.set_xlabel(r'$\rho$', fontsize=12)
        ax.set_ylabel(r'$\alpha = \frac{A(\rho)-A(1)}{A(1)}$', fontsize=13)
        ax.set_title(f'{ds_name}: Relative accuracy change', fontsize=12)
        ax.legend(fontsize=8, ncol=2)
        ax.set_xlim(0.05, 1.05)
        ax.grid(True, alpha=0.3)

        # Shade the "masking helps" region
        for ber in reversed(ber_vals):
            rhos_pos = [r for r, a in alpha[ber] if a > 0 and r < 1.0]
            if rhos_pos:
                ax.axvspan(min(rhos_pos) - 0.03, max(rhos_pos) + 0.03,
                          alpha=0.08, color='green')

    fig.suptitle(r'SpikeAdapt-SC Classification: $\alpha(\rho, BER)$ Analysis', fontsize=14, y=1.01)
    plt.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    out_pdf = os.path.join(outdir, 'alpha_combined_both_datasets.pdf')
    out_png = os.path.join(outdir, 'alpha_combined_both_datasets.png')
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_pdf}")


def print_alpha_table(data):
    """Print α values in table format for quick inspection."""
    for ds_key, ds_name in [('aid', 'AID'), ('resisc45', 'RESISC45')]:
        alpha, _, rho_vals, ber_vals = compute_alpha(data[ds_key])
        print(f"\n{'='*70}")
        print(f"α TABLE — {ds_name}")
        print(f"{'='*70}")
        header = f"{'ρ':>8}" + "".join(f"  BER={b:.2f}" for b in ber_vals)
        print(header)
        print("-" * len(header))
        for i, rho in enumerate(rho_vals):
            row = f"{rho:>8.3f}"
            for ber in ber_vals:
                a = alpha[ber][i][1]
                marker = "+" if a > 0.001 else " "
                row += f"  {marker}{a:>7.4f}"
            print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extended', action='store_true',
                       help='Use extended BER results (up to 0.4)')
    args = parser.parse_args()

    data, is_extended = load_data(args.extended)
    source = "extended (BER ≤ 0.4)" if is_extended else "ablation (BER ∈ {0.0, 0.15, 0.30})"
    print(f"Data source: {source}")

    outdir = 'eval/figures'

    # Print α table
    print_alpha_table(data)

    # Generate plots
    for ds_key, ds_name in [('aid', 'AID'), ('resisc45', 'RESISC45')]:
        alpha, accuracy, rho_vals, ber_vals = compute_alpha(data[ds_key])
        plot_accuracy_vs_rho(accuracy, rho_vals, ber_vals, ds_name, outdir)
        plot_alpha(alpha, rho_vals, ber_vals, ds_name, outdir)

    # Combined plot
    plot_combined(data, outdir)

    print("\nDone! All figures saved to eval/figures/")


if __name__ == '__main__':
    main()
