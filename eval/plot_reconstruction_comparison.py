#!/usr/bin/env python3
"""Reconstruction PSNR comparison: SpikeAdapt-SC V6 vs DHF-JSCC analysis.

Plots:
  1. PSNR vs CBR for BSC channel at different BER levels (InStereo2K)
  2. PSNR vs CBR for AWGN channel at different CSNR levels (InStereo2K)
  3. PSNR vs BER at fixed CBR (showing noise resilience)
  4. PSNR vs CSNR with matched BER overlay (BSC≡AWGN comparison)

Output: eval/figures/reconstruction_*.pdf

Usage:
    python eval/plot_reconstruction_comparison.py
"""

import os, sys, json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def ber_from_snr_awgn(snr_db):
    """Theoretical BER for uncoded BPSK over AWGN: Q(sqrt(2*SNR))."""
    snr_lin = 10 ** (snr_db / 10.0)
    return 0.5 * math.erfc(math.sqrt(snr_lin))


def snr_from_ber_awgn(ber):
    """Inverse: approximate SNR (dB) for a target BER via bisection."""
    if ber <= 0:
        return 100.0
    if ber >= 0.5:
        return -20.0
    lo, hi = -10, 40
    for _ in range(60):
        mid = (lo + hi) / 2
        if ber_from_snr_awgn(mid) > ber:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def load_v6_results():
    """Load V6 reconstruction results for both datasets."""
    results = {}
    for ds, label in [('kitti_image_v6_instereo2k', 'InStereo2K'),
                       ('kitti_image_v6_kitti', 'KITTI')]:
        path = f'eval/seed_results/{ds}.json'
        if os.path.exists(path):
            with open(path) as f:
                results[label] = json.load(f)
    return results


def plot_bsc_rd_curves(data, dataset, outdir):
    """Plot PSNR vs CBR for different BER levels (BSC channel)."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    bsc = data[dataset]['bsc']
    ber_list = sorted(bsc.keys(), key=float)
    cbr_list = sorted([float(c) for c in bsc[ber_list[0]].keys()])

    cmap = plt.cm.RdYlBu_r
    norm = plt.Normalize(0, 0.3)

    for ber in ber_list:
        psnrs = [bsc[ber][str(c)]['psnr'] for c in cbr_list]
        color = cmap(norm(float(ber)))
        label = f'BER={float(ber):.2f}' if float(ber) > 0 else 'Clean'
        ax.plot(cbr_list, psnrs, 'o-', color=color, linewidth=2, markersize=5, label=label)

    ax.set_xlabel('Channel Bandwidth Ratio (CBR)', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title(f'{dataset}: SpikeAdapt-SC V6 Reconstruction (BSC)', fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.05, 1.05)

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f'reconstruction_bsc_{dataset.lower()}.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_awgn_rd_curves(data, dataset, outdir):
    """Plot PSNR vs CBR for different CSNR levels (AWGN channel)."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    awgn = data[dataset]['awgn']
    snr_list = sorted(awgn.keys(), key=float)
    cbr_list = sorted([float(c) for c in awgn[snr_list[0]].keys()])

    cmap = plt.cm.viridis
    norm = plt.Normalize(float(snr_list[0]), float(snr_list[-1]))

    for snr in snr_list:
        psnrs = [awgn[snr][str(c)]['psnr'] for c in cbr_list]
        color = cmap(norm(float(snr)))
        ax.plot(cbr_list, psnrs, 'o-', color=color, linewidth=2, markersize=5,
                label=f'CSNR={snr} dB')

    ax.set_xlabel('Channel Bandwidth Ratio (CBR)', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title(f'{dataset}: SpikeAdapt-SC V6 Reconstruction (AWGN)', fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.05, 1.05)

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f'reconstruction_awgn_{dataset.lower()}.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_noise_resilience(data, dataset, outdir):
    """Plot PSNR vs BER at fixed CBR values, showing noise resilience."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    bsc = data[dataset]['bsc']
    ber_list = sorted(bsc.keys(), key=float)
    cbr_list = sorted([float(c) for c in bsc[ber_list[0]].keys()])

    colors = plt.cm.tab10(np.linspace(0, 1, len(cbr_list)))
    for i, cbr in enumerate(cbr_list):
        psnrs = [bsc[ber][str(cbr)]['psnr'] for ber in ber_list]
        bers = [float(b) for b in ber_list]
        ax.plot(bers, psnrs, 'o-', color=colors[i], linewidth=2, markersize=5,
                label=f'CBR={cbr}')

    ax.set_xlabel('Bit Error Rate (BER)', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title(f'{dataset}: Reconstruction PSNR vs BER (BSC)', fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f'reconstruction_psnr_vs_ber_{dataset.lower()}.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_bsc_vs_awgn_matched(data, dataset, outdir):
    """Compare BSC vs AWGN at matched channel quality.

    Key insight: for binary SNN spikes with BPSK + hard decision,
    AWGN at SNR x dB ≡ BSC at BER = Q(√(2·SNR)).

    This plot shows they overlap, proving no quantization loss.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    bsc = data[dataset]['bsc']
    awgn = data[dataset]['awgn']

    cbr_target = '0.5'  # Fixed CBR for comparison

    # BSC: PSNR at each BER
    ber_list = sorted(bsc.keys(), key=float)
    bsc_psnrs = [bsc[ber][cbr_target]['psnr'] for ber in ber_list]
    bsc_bers = [float(b) for b in ber_list]

    # AWGN: convert SNR to equivalent BER, then plot
    snr_list = sorted(awgn.keys(), key=float)
    awgn_bers = [ber_from_snr_awgn(float(s)) for s in snr_list]
    awgn_psnrs = [awgn[s][cbr_target]['psnr'] for s in snr_list]

    ax.plot(bsc_bers, bsc_psnrs, 'o-', color='#2563eb', linewidth=2.5,
            markersize=8, label='BSC (native binary)', zorder=5)
    ax.plot(awgn_bers, awgn_psnrs, 's--', color='#dc2626', linewidth=2,
            markersize=7, label='AWGN→BPSK→hard decision', zorder=4)

    ax.set_xlabel('Effective BER', fontsize=12)
    ax.set_ylabel(f'PSNR (dB) at CBR={cbr_target}', fontsize=12)
    ax.set_title(f'{dataset}: BSC vs AWGN at Matched BER (CBR={cbr_target})\n'
                 'Binary SNN spikes see identical channels → no quantization needed',
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.annotate('Binary spikes: AWGN+BPSK+hard decision ≡ BSC\n'
               '→ No quantization loss (spikes are natively binary)',
               xy=(0.15, min(bsc_psnrs) + 1), fontsize=9,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f'reconstruction_bsc_vs_awgn_{dataset.lower()}.pdf')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")


def print_comparison_table(data, dataset):
    """Print BSC vs AWGN comparison at matched BER."""
    bsc = data[dataset]['bsc']
    awgn = data[dataset]['awgn']

    print(f"\n{'='*70}")
    print(f"BSC vs AWGN (matched BER) — {dataset}, CBR=0.5")
    print(f"{'='*70}")
    print(f"{'SNR (dB)':>10} {'BER_equiv':>10} {'AWGN PSNR':>10} {'BSC PSNR':>10} {'Δ (dB)':>10}")
    print("-" * 55)

    ber_vals = sorted([float(b) for b in bsc.keys()])

    for snr in sorted(awgn.keys(), key=float):
        ber_eq = ber_from_snr_awgn(float(snr))
        awgn_p = awgn[snr]['0.5']['psnr']
        # Find closest BSC BER
        closest_ber = min(ber_vals, key=lambda b: abs(b - ber_eq))
        bsc_p = bsc[str(closest_ber)]['0.5']['psnr']
        delta = awgn_p - bsc_p
        print(f"{float(snr):>10.1f} {ber_eq:>10.4f} {awgn_p:>10.1f} {bsc_p:>10.1f} {delta:>+10.1f}")


def main():
    data = load_v6_results()
    outdir = 'eval/figures'

    for dataset in data:
        print(f"\n{'#'*60}")
        print(f"  {dataset}")
        print(f"{'#'*60}")

        plot_bsc_rd_curves(data, dataset, outdir)
        plot_awgn_rd_curves(data, dataset, outdir)
        plot_noise_resilience(data, dataset, outdir)
        plot_bsc_vs_awgn_matched(data, dataset, outdir)
        print_comparison_table(data, dataset)

    print("\nDone! All reconstruction figures saved to eval/figures/")


if __name__ == '__main__':
    main()
