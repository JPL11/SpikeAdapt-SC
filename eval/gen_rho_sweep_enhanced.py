#!/usr/bin/env python3
"""Enhanced ρ sweep: every ρ point labeled below BER=0.30 curve, 
clear spacing, best point highlighted."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 7.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
})

OUT = './paper/figures/'

with open('eval/ablation_final_results.json') as f:
    results = json.load(f)

C_BLUE = '#1565C0'
C_ORANGE = '#EF6C00'
C_RED = '#B71C1C'

rho_values = [0.10, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 1.0]
bw_save = [(1 - r) * 100 for r in rho_values]

fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.0))

for idx, (ds, title, ylim) in enumerate([
    ('aid', 'AID (50/50 split)', (55, 97.5)),
    ('resisc45', 'RESISC45 (20/80 split)', (45, 96)),
]):
    ax = axes[idx]
    clean = [results[ds]['rho_sweep'][str(r)]['0.0'] for r in rho_values]
    ber15 = [results[ds]['rho_sweep'][str(r)]['0.15'] for r in rho_values]
    ber30 = [results[ds]['rho_sweep'][str(r)]['0.3'] for r in rho_values]

    ax.plot(bw_save, clean, 'o-', color=C_BLUE, label='Clean (BER=0)', zorder=3, lw=2, ms=5)
    ax.plot(bw_save, ber15, 's--', color=C_ORANGE, label='BER=0.15', zorder=3, lw=2, ms=5)
    ax.plot(bw_save, ber30, 'D-', color=C_RED, label='BER=0.30', zorder=3, lw=2, ms=5)

    baseline_30 = ber30[-1]  # ρ=1.0 at BER=0.30

    # ── Labels below BER=0.30 curve (alternating offset to avoid overlap) ──
    for i, r in enumerate(rho_values):
        bw = bw_save[i]
        acc = ber30[i]
        delta = acc - baseline_30

        if r == 0.10:
            # Too far down, skip or minimal
            continue

        # Format delta
        if abs(delta) < 0.3:
            d_str = 'ref'
        elif delta > 0:
            d_str = f'+{delta:.1f}pp'
        else:
            d_str = f'{delta:.1f}pp'

        # Set styles
        if r == 0.625:
            # Best: star + bold red
            ax.scatter([bw], [acc], s=140, c='gold', edgecolor=C_RED,
                      linewidth=1.5, zorder=5, marker='*')
            fs, fw, fc = 7, 'bold', C_RED
        elif r == 1.0:
            fs, fw, fc = 6, 'normal', '#888'
        else:
            fs, fw, fc = 6, 'normal', '#555'

        # ── Per-dataset, per-point offsets (hand-tuned to avoid overlap) ──
        if ds == 'aid':
            offsets = {
                0.25: (0, -20), 0.375: (0, -20), 0.50: (0, -20),
                0.625: (2, -22), 0.75: (0, -20), 0.875: (0, -20),
                1.0: (0, -18),
            }
        else:  # resisc45
            offsets = {
                0.25: (0, -20), 0.375: (0, -20), 0.50: (0, -20),
                0.625: (2, -22), 0.75: (-2, 8), 0.875: (-2, 8),
                1.0: (-5, 8),
            }

        xt, yt = offsets.get(r, (0, -20))

        label = f'ρ={r}' if r == 1.0 else f'ρ={r}\n{d_str}'
        if r == 1.0:
            label = f'ρ=1.0\n(base)'

        ax.annotate(label, (bw, acc),
                   textcoords="offset points", xytext=(xt, yt),
                   fontsize=fs, color=fc, fontweight=fw,
                   ha='center', va='top' if yt < 0 else 'bottom')

    ax.set_xlabel('Bandwidth Savings (%)')
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title(title, fontweight='bold')
    ax.set_ylim(ylim)
    ax.set_xlim(-3, 93)
    ax.legend(loc='lower left', frameon=True, fancybox=False, edgecolor='#ccc')
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.xaxis.set_major_locator(MultipleLocator(25))

plt.tight_layout(w_pad=2.5)
plt.savefig(f'{OUT}/fig5_rho_sweep_pareto.pdf')
plt.savefig(f'{OUT}/fig5_rho_sweep_pareto.png', dpi=200)
plt.close()
print(f"✅ {OUT}fig5_rho_sweep_pareto.png")
