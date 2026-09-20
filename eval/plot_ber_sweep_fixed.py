#!/usr/bin/env python3
"""Fixed BER-sweep baselines figure for the paper.

Issues with the original (paper/figures/fig_ber_sweep_baselines.png):
  1. SNN-SC (the external Wang et al. baseline) is missing entirely
  2. The two green curves (SpikeAdapt rho=0.75 and SNN no-mask rho=1.0) overlap
     because they're the SAME model at different rates — looks misleading
  3. No visual separation at the critical BER=0.20-0.30 region

Fixes:
  - Adds SNN-SC explicitly (drops to 82% at BER=0.30 — clear gap)
  - Adds SpikeAdapt at rho=1.0 (matched bandwidth vs SNN-SC) with 10-seed error bars
  - Zoomed inset shows BER 0.20-0.30 with all separators visible
  - Distinct colors per method family

Output: paper/figures/fig_ber_sweep_baselines_v2.{pdf,png}
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

os.makedirs('paper/figures', exist_ok=True)

# ===== Data from paper Table 2 (seed-42, full BER sweep) =====
data = {
    'AID': {
        'SpikeAdapt-SC (rho=0.75)': {0.0: 95.46, 0.05: 95.63, 0.10: 95.66,
                                       0.15: 95.78, 0.20: 95.70, 0.25: 95.16, 0.30: 93.43},
        'SNN-SC (rho=1.0)':        {0.0: 95.40, 0.05: 95.39, 0.10: 95.21,
                                       0.15: 94.86, 0.20: 94.29, 0.25: 92.22, 0.30: 82.42},
        'CNN-Uni (8-bit)':         {0.0: 91.78, 0.05: 91.90, 0.10: 90.82,
                                       0.15: 88.46, 0.20: 83.40, 0.25: 72.96, 0.30: 52.80},
        'CNN-NonUni (8-bit)':      {0.0: 91.74, 0.05: 91.50, 0.10: 90.50,
                                       0.15: 89.00, 0.20: 85.56, 0.25: 78.68, 0.30: 63.58},
        'MLP-FC (8-bit)':          {0.0: 91.98, 0.05: 88.46, 0.10: 84.16,
                                       0.15: 76.84, 0.20: 66.32, 0.25: 51.48, 0.30: 36.12},
        'JPEG+Conv':               {0.0: 95.16, 0.05: 2.66, 0.10: 2.68,
                                       0.15: 2.68, 0.20: 2.68, 0.25: 2.68, 0.30: 2.68},
    },
    'RESISC45': {
        'SpikeAdapt-SC (rho=0.75)': {0.0: 92.00, 0.05: 92.14, 0.10: 92.34,
                                       0.15: 92.42, 0.20: 92.31, 0.25: 91.38, 0.30: 87.00},
        'SNN-SC (rho=1.0)':        {0.0: 92.39, 0.05: 92.43, 0.10: 92.35,
                                       0.15: 92.12, 0.20: 91.51, 0.25: 89.60, 0.30: 80.55},
        'CNN-Uni (8-bit)':         {0.0: 78.32, 0.05: 81.38, 0.10: 82.68,
                                       0.15: 81.20, 0.20: 77.00, 0.25: 67.10, 0.30: 49.32},
        'CNN-NonUni (8-bit)':      {0.0: 78.32, 0.05: 78.63, 0.10: 77.96,
                                       0.15: 76.42, 0.20: 72.21, 0.25: 64.45, 0.30: 50.77},
        'MLP-FC (8-bit)':          {0.0: 82.23, 0.05: 74.73, 0.10: 65.58,
                                       0.15: 54.77, 0.20: 43.11, 0.25: 31.27, 0.30: 20.89},
        'JPEG+Conv':               {0.0: 93.55, 0.05: 1.80, 0.10: 1.80,
                                       0.15: 1.80, 0.20: 1.80, 0.25: 1.80, 0.30: 1.80},
    },
}

# 10-seed SpikeAdapt at rho=1.0 (matched-bandwidth comparison vs SNN-SC)
spikeadapt_r1_10seed = {
    'AID': {0.0: (95.49, 0.31), 0.15: (95.60, 0.30), 0.30: (91.89, 3.69)},
    'RESISC45': {0.0: (92.59, 0.16), 0.15: (92.87, 0.30), 0.30: (84.85, 5.58)},
}

BERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# Distinct, accessible color palette
COLORS = {
    'SpikeAdapt-SC (rho=0.75)':   '#1565c0',  # blue
    'SpikeAdapt-SC (rho=1.0)':    '#1976d2',  # lighter blue
    'SNN-SC (rho=1.0)':           '#7b1fa2',  # purple
    'CNN-Uni (8-bit)':            '#d32f2f',  # red
    'CNN-NonUni (8-bit)':         '#f57c00',  # orange
    'MLP-FC (8-bit)':             '#c2185b',  # pink/magenta
    'JPEG+Conv':                  '#616161',  # gray
}

MARKERS = {
    'SpikeAdapt-SC (rho=0.75)':   ('o', 7),
    'SpikeAdapt-SC (rho=1.0)':    ('D', 6),
    'SNN-SC (rho=1.0)':           ('s', 6),
    'CNN-Uni (8-bit)':            ('^', 5),
    'CNN-NonUni (8-bit)':         ('v', 5),
    'MLP-FC (8-bit)':             ('P', 5),
    'JPEG+Conv':                  ('x', 5),
}

LINESTYLES = {
    'SpikeAdapt-SC (rho=0.75)':   '-',
    'SpikeAdapt-SC (rho=1.0)':    '-',
    'SNN-SC (rho=1.0)':           '--',
    'CNN-Uni (8-bit)':            '-.',
    'CNN-NonUni (8-bit)':         '-.',
    'MLP-FC (8-bit)':             '-.',
    'JPEG+Conv':                  ':',
}

# Plot order (later = on top)
ORDER = [
    'JPEG+Conv', 'MLP-FC (8-bit)', 'CNN-Uni (8-bit)', 'CNN-NonUni (8-bit)',
    'SNN-SC (rho=1.0)', 'SpikeAdapt-SC (rho=1.0)', 'SpikeAdapt-SC (rho=0.75)',
]

plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12,
                     'axes.titlesize': 13, 'legend.fontsize': 8})

fig, axes = plt.subplots(2, 2, figsize=(13, 9), gridspec_kw={'height_ratios': [1.4, 1]})

for col, ds in enumerate(['AID', 'RESISC45']):
    d = data[ds]
    r1 = spikeadapt_r1_10seed[ds]
    bers_r1 = sorted(r1.keys())
    means_r1 = [r1[b][0] for b in bers_r1]
    stds_r1 = [r1[b][1] for b in bers_r1]

    # ===== Top row: full BER sweep =====
    ax = axes[0, col]
    for method in ORDER:
        if method == 'SpikeAdapt-SC (rho=1.0)':
            continue
        if method not in d:
            continue
        accs = [d[method][b] for b in BERS]
        marker, msize = MARKERS[method]
        ax.plot(BERS, accs, marker=marker, markersize=msize,
                linestyle=LINESTYLES[method], color=COLORS[method],
                linewidth=2, label=method, zorder=3)

    ax.errorbar(bers_r1, means_r1, yerr=stds_r1,
                marker='D', markersize=7, linestyle='-',
                color=COLORS['SpikeAdapt-SC (rho=1.0)'], linewidth=2,
                capsize=4, capthick=1.5,
                label='SpikeAdapt-SC (rho=1.0, 10-seed)', zorder=5)

    n_cls = 30 if ds == 'AID' else 45
    ax.axhline(y=100/n_cls, color='gray', linestyle=':', linewidth=1, alpha=0.4)

    ax.set_xlabel('Bit Error Rate (BER)', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    label = '(a) AID (50/50)' if ds == 'AID' else '(b) RESISC45 (20/80)'
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.set_xlim(-0.015, 0.32)
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.3)
    if col == 0:
        ax.legend(fontsize=7.5, loc='lower left', ncol=1, framealpha=0.95)
    else:
        ax.legend(fontsize=7.5, loc='lower left', ncol=1, framealpha=0.95)

    # ===== Bottom row: zoomed BER 0.20-0.30 =====
    ax_z = axes[1, col]
    for method in ORDER:
        if method == 'SpikeAdapt-SC (rho=1.0)':
            continue
        if method not in d:
            continue
        accs = [d[method][b] for b in BERS]
        marker, msize = MARKERS[method]
        ax_z.plot(BERS, accs, marker=marker, markersize=msize+1,
                  linestyle=LINESTYLES[method], color=COLORS[method],
                  linewidth=2, label=method, zorder=3)

    ax_z.errorbar(bers_r1, means_r1, yerr=stds_r1,
                   marker='D', markersize=8, linestyle='-',
                   color=COLORS['SpikeAdapt-SC (rho=1.0)'], linewidth=2,
                   capsize=4, capthick=1.5,
                   label='SpikeAdapt-SC (rho=1.0, 10-seed)', zorder=5)

    # Annotate the critical separation
    snn_30 = d['SNN-SC (rho=1.0)'][0.30]
    sa_30 = d['SpikeAdapt-SC (rho=0.75)'][0.30]
    sa_10_30 = r1[0.30][0]
    gap_075 = sa_30 - snn_30
    gap_10 = sa_10_30 - snn_30

    # Vertical bracket showing gap
    ax_z.annotate('', xy=(0.302, sa_30), xytext=(0.302, snn_30),
                   arrowprops=dict(arrowstyle='<->', color='#0d47a1', lw=1.5))
    ax_z.text(0.304, (sa_30 + snn_30) / 2,
              f'$\\rho{{=}}0.75$:\n+{gap_075:.1f} pp',
              fontsize=9, fontweight='bold', va='center', ha='left',
              color='#0d47a1',
              bbox=dict(boxstyle='round,pad=0.25', fc='#e3f2fd', ec='#0d47a1', alpha=0.95))

    ax_z.set_xlabel('Bit Error Rate (BER)', fontsize=11)
    ax_z.set_ylabel('Accuracy (%)', fontsize=11)
    title_z = '(c) AID — zoomed (BER 0.20–0.30)' if ds == 'AID' \
              else '(d) RESISC45 — zoomed (BER 0.20–0.30)'
    ax_z.set_title(title_z, fontsize=12, fontweight='bold')
    ax_z.set_xlim(0.195, 0.32)
    if ds == 'AID':
        ax_z.set_ylim(30, 100)
    else:
        ax_z.set_ylim(15, 95)
    ax_z.set_xticks([0.20, 0.25, 0.30])
    ax_z.grid(True, alpha=0.3)

plt.tight_layout()
out_pdf = 'paper/figures/fig_ber_sweep_baselines_v2.pdf'
out_png = 'paper/figures/fig_ber_sweep_baselines_v2.png'
fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
fig.savefig(out_png, dpi=200, bbox_inches='tight')
plt.close()
print(f'Saved: {out_pdf}')
print(f'Saved: {out_png}')

# Print key gaps for paper
print('\nKey separation at BER=0.30:')
for ds in ['AID', 'RESISC45']:
    d = data[ds]
    sa_075 = d['SpikeAdapt-SC (rho=0.75)'][0.30]
    snn_sc = d['SNN-SC (rho=1.0)'][0.30]
    sa_10 = spikeadapt_r1_10seed[ds][0.30][0]
    print(f'  {ds}:')
    print(f'    SpikeAdapt rho=0.75 (75% BW): {sa_075:.2f}%')
    print(f'    SpikeAdapt rho=1.0 (100% BW): {sa_10:.2f}% (10-seed mean)')
    print(f'    SNN-SC rho=1.0 (100% BW): {snn_sc:.2f}%')
    print(f'    Gap vs SNN-SC: rho=0.75 +{sa_075-snn_sc:.2f} pp (with -25% BW)')
    print(f'                   rho=1.0 +{sa_10-snn_sc:.2f} pp (matched BW)')
