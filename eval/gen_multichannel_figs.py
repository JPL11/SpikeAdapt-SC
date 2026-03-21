"""Generate paper-grade multichannel comparison figures.

Creates:
1. fig7_multichannel_snr.pdf — Extended SNR sweep (BSC/AWGN/Rayleigh) for both datasets
2. fig7_multichannel_matched.pdf — Matched-BER comparison across all 3 channels
"""

import os, sys, json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

OUT_DIR = 'paper/figures'

# Load results
with open('eval/multichannel_results_v2.json') as f:
    data = json.load(f)

# Color palette
C_BSC = '#1a5276'
C_AWGN = '#c0392b'
C_RAY = '#27ae60'

# ===================================================================
# Figure 1: Matched-BER comparison (dual panel)
# ===================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=False)

for ax, ds, title in [(ax1, 'aid', 'AID (50/50)'), (ax2, 'resisc45', 'RESISC45 (20/80)')]:
    matched = data[ds]['matched_ber']
    bers = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    bsc_accs = [matched[str(b)]['bsc_acc'] for b in bers]
    awgn_accs = [matched[str(b)]['awgn_acc'] for b in bers]
    ray_accs = [matched[str(b)]['ray_acc'] for b in bers]
    
    ax.plot(bers, bsc_accs, 'o-', color=C_BSC, label='BSC', markersize=4, linewidth=1.5)
    ax.plot(bers, awgn_accs, 's--', color=C_AWGN, label='AWGN', markersize=4, linewidth=1.5)
    ax.plot(bers, ray_accs, '^:', color=C_RAY, label='Rayleigh', markersize=4, linewidth=1.5)
    
    ax.set_xlabel('Equivalent BER')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', framealpha=0.9)

ax1.set_ylabel('Accuracy (%)')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig7_multichannel_matched.png')
plt.savefig(f'{OUT_DIR}/fig7_multichannel_matched.pdf')
plt.close()
print("✅ fig7_multichannel_matched saved")


# ===================================================================
# Figure 2: Extended SNR sweep (dual panel)
# ===================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=False)

for ax, ds, title in [(ax1, 'aid', 'AID (50/50)'), (ax2, 'resisc45', 'RESISC45 (20/80)')]:
    # AWGN
    awgn = data[ds]['awgn']
    snrs_a = [int(s) for s in awgn['results'].keys()]
    accs_a = [awgn['results'][s]['acc'] for s in awgn['results']]
    eq_bers_a = [awgn['results'][s]['eq_ber'] for s in awgn['results']]
    
    # Rayleigh
    ray = data[ds]['rayleigh']
    snrs_r = [int(s) for s in ray['results'].keys()]
    accs_r = [ray['results'][s]['acc'] for s in ray['results']]
    eq_bers_r = [ray['results'][s]['eq_ber'] for s in ray['results']]
    
    ax.plot(snrs_a, accs_a, 's-', color=C_AWGN, label='AWGN', markersize=4, linewidth=1.5)
    ax.plot(snrs_r, accs_r, '^-', color=C_RAY, label='Rayleigh', markersize=4, linewidth=1.5)
    
    # Add equivalent BER annotations at key points
    for snr_idx in [4, 7, 9]:  # 0dB, -6dB, -10dB
        if snr_idx < len(snrs_a):
            ax.annotate(f'BER={eq_bers_a[snr_idx]:.2f}', 
                       (snrs_a[snr_idx], accs_a[snr_idx]),
                       textcoords="offset points", xytext=(0, 8),
                       fontsize=6, color=C_AWGN, ha='center')
        if snr_idx < len(snrs_r):
            ax.annotate(f'BER={eq_bers_r[snr_idx]:.2f}',
                       (snrs_r[snr_idx], accs_r[snr_idx]),
                       textcoords="offset points", xytext=(0, -12),
                       fontsize=6, color=C_RAY, ha='center')
    
    ax.set_xlabel('SNR (dB)')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', framealpha=0.9)
    ax.invert_xaxis()  # SNR decreases left-to-right (worse conditions)

ax1.set_ylabel('Accuracy (%)')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig7_multichannel_snr.png')
plt.savefig(f'{OUT_DIR}/fig7_multichannel_snr.pdf')
plt.close()
print("✅ fig7_multichannel_snr saved")


# ===================================================================
# Figure 3: Combined 3-channel BER robustness (single panel per dataset, overlay)
# ===================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

for ax, ds, title in [(ax1, 'aid', 'AID'), (ax2, 'resisc45', 'RESISC45')]:
    # BSC
    bsc = data[ds]['bsc']
    bers_bsc = [float(b) for b in bsc['results'].keys()]
    accs_bsc = list(bsc['results'].values())
    
    # AWGN (use equivalent BER as x-axis)
    awgn = data[ds]['awgn']
    bers_awgn = [awgn['results'][s]['eq_ber'] for s in awgn['results']]
    accs_awgn = [awgn['results'][s]['acc'] for s in awgn['results']]
    
    # Rayleigh (use equivalent BER as x-axis)
    ray = data[ds]['rayleigh']
    bers_ray = [ray['results'][s]['eq_ber'] for s in ray['results']]
    accs_ray = [ray['results'][s]['acc'] for s in ray['results']]
    
    ax.plot(bers_bsc, accs_bsc, 'o-', color=C_BSC, label='BSC', markersize=3.5, linewidth=1.5)
    ax.plot(bers_awgn, accs_awgn, 's-', color=C_AWGN, label='AWGN (eq BER)', markersize=3.5, linewidth=1.5)
    ax.plot(bers_ray, accs_ray, '^-', color=C_RAY, label='Rayleigh (eq BER)', markersize=3.5, linewidth=1.5)
    
    ax.set_xlabel('Equivalent BER')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=7, framealpha=0.9)
    ax.set_xlim(-0.01, 0.42)

ax1.set_ylabel('Accuracy (%)')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig7_unified_ber.png')
plt.savefig(f'{OUT_DIR}/fig7_unified_ber.pdf')
plt.close()
print("✅ fig7_unified_ber saved")

print("\n✅ All multichannel figures generated!")
