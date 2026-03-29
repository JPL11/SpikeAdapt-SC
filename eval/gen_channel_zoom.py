#!/usr/bin/env python3
"""Zoomed-in channel comparison: show BSC vs AWGN vs Rayleigh differences.
Zoom into the region where differences are visible."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

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

with open('eval/multichannel_results_v2.json') as f:
    snn_mc = json.load(f)

C_BSC = '#1a5276'
C_AWGN = '#c0392b'
C_RAY = '#27ae60'

# ============================================================================
# Version 1: Extra-zoomed into BER 0.20-0.40 region where curves diverge
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))

for ax, ds, title in [(axes[0], 'aid', 'AID'), (axes[1], 'resisc45', 'RESISC45')]:
    bsc = snn_mc[ds]['bsc']['results']
    bers_bsc = [float(b) for b in bsc.keys()]
    accs_bsc = list(bsc.values())

    awgn = snn_mc[ds]['awgn']['results']
    bers_awgn = [awgn[s]['eq_ber'] for s in awgn]
    accs_awgn = [awgn[s]['acc'] for s in awgn]

    ray = snn_mc[ds]['rayleigh']['results']
    bers_ray = [ray[s]['eq_ber'] for s in ray]
    accs_ray = [ray[s]['acc'] for s in ray]

    ax.plot(bers_bsc, accs_bsc, 'o-', color=C_BSC, label='BSC', ms=5, lw=2)
    ax.plot(bers_awgn, accs_awgn, 's--', color=C_AWGN, label='AWGN (eq BER)', ms=5, lw=1.8)
    ax.plot(bers_ray, accs_ray, '^:', color=C_RAY, label='Rayleigh (eq BER)', ms=5, lw=1.8)

    # Shade the "divergence zone" where differences emerge
    ax.axvspan(0.25, 0.40, alpha=0.06, color='orange', zorder=0)

    # Compute and annotate max difference at key BER points
    # Find closest matching BER points for comparison
    for target_ber in [0.25, 0.30, 0.35]:
        bsc_val = bsc.get(str(target_ber))
        if bsc_val is None:
            continue
        # Find awgn/ray at closest eq_ber
        awgn_val = None
        ray_val = None
        for s in awgn:
            if abs(awgn[s]['eq_ber'] - target_ber) < 0.02:
                awgn_val = awgn[s]['acc']
                break
        for s in ray:
            if abs(ray[s]['eq_ber'] - target_ber) < 0.02:
                ray_val = ray[s]['acc']
                break

        if awgn_val and ray_val:
            vals = [bsc_val, awgn_val, ray_val]
            delta = max(vals) - min(vals)
            if delta > 0.3 and target_ber >= 0.28:
                # Annotate the spread
                y_mid = np.mean(vals)
                ax.annotate(
                    f'Δ={delta:.1f}pp',
                    (target_ber, y_mid),
                    textcoords="offset points", xytext=(15, 8),
                    fontsize=6.5, color='#555',
                    arrowprops=dict(arrowstyle='->', color='#999', lw=0.6),
                )

    ax.set_xlabel('Equivalent BER')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=7, framealpha=0.9)
    ax.set_xlim(-0.01, 0.42)
    ax.set_ylim(75, 97)

axes[0].set_ylabel('Accuracy (%)')

plt.tight_layout()
plt.savefig(f'{OUT}/fig7_unified_ber_zoomed.png', dpi=200)
plt.savefig(f'{OUT}/fig7_unified_ber_zoomed.pdf')
plt.close()
print(f"✅ {OUT}fig7_unified_ber_zoomed.png (standard zoom)")


# ============================================================================
# Version 2: EXTRA ZOOMED — focus on BER 0.20-0.38, y-axis 72-97%
# This version makes the small differences between channels more visible
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))

for ax, ds, title in [(axes[0], 'aid', 'AID'), (axes[1], 'resisc45', 'RESISC45')]:
    bsc = snn_mc[ds]['bsc']['results']
    awgn = snn_mc[ds]['awgn']['results']
    ray = snn_mc[ds]['rayleigh']['results']

    # Filter to BER >= 0.15 for zoom
    bers_bsc_f = [(float(b), v) for b, v in bsc.items() if float(b) >= 0.15]
    bers_awgn_f = [(awgn[s]['eq_ber'], awgn[s]['acc']) for s in awgn if awgn[s]['eq_ber'] >= 0.15]
    bers_ray_f = [(ray[s]['eq_ber'], ray[s]['acc']) for s in ray if ray[s]['eq_ber'] >= 0.15]

    # Sort
    bers_bsc_f.sort(); bers_awgn_f.sort(); bers_ray_f.sort()

    ax.plot([b for b, a in bers_bsc_f], [a for b, a in bers_bsc_f],
            'o-', color=C_BSC, label='BSC', ms=6, lw=2.2)
    ax.plot([b for b, a in bers_awgn_f], [a for b, a in bers_awgn_f],
            's--', color=C_AWGN, label='AWGN (eq BER)', ms=6, lw=2)
    ax.plot([b for b, a in bers_ray_f], [a for b, a in bers_ray_f],
            '^:', color=C_RAY, label='Rayleigh (eq BER)', ms=6, lw=2)

    # Annotate max channel spread at each BER point
    for target_ber in [0.25, 0.30, 0.35]:
        bsc_val = bsc.get(str(target_ber))
        if bsc_val is None: continue
        awgn_val = ray_val = None
        for s in awgn:
            if abs(awgn[s]['eq_ber'] - target_ber) < 0.015:
                awgn_val = awgn[s]['acc']; break
        for s in ray:
            if abs(ray[s]['eq_ber'] - target_ber) < 0.015:
                ray_val = ray[s]['acc']; break
        if awgn_val and ray_val:
            vals = [bsc_val, awgn_val, ray_val]
            delta = max(vals) - min(vals)
            y_mid = np.mean(vals)
            offset_x = 12 if target_ber < 0.35 else -35
            offset_y = -12 if target_ber < 0.30 else 8
            ax.annotate(
                f'Δ={delta:.1f}pp',
                (target_ber, y_mid),
                textcoords="offset points", xytext=(offset_x, offset_y),
                fontsize=7, color='#333', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#666', lw=0.8),
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='#ccc', alpha=0.8),
            )

    ax.set_xlabel('Equivalent BER')
    ax.set_title(f'{title} (High-BER Detail)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=7, framealpha=0.9)

    if ds == 'aid':
        ax.set_xlim(0.14, 0.42)
        ax.set_ylim(72, 97)
    else:
        ax.set_xlim(0.14, 0.38)
        ax.set_ylim(72, 94)

axes[0].set_ylabel('Accuracy (%)')

plt.tight_layout()
plt.savefig(f'{OUT}/fig7_channel_detail_zoom.png', dpi=200)
plt.savefig(f'{OUT}/fig7_channel_detail_zoom.pdf')
plt.close()
print(f"✅ {OUT}fig7_channel_detail_zoom.png (extra zoom BER≥0.15)")

# Print exact differences
print("\n  Channel differences at key BER points:")
for ds in ['aid', 'resisc45']:
    bsc = snn_mc[ds]['bsc']['results']
    awgn = snn_mc[ds]['awgn']['results']
    ray = snn_mc[ds]['rayleigh']['results']
    print(f"\n  {ds.upper()}:")
    print(f"  {'BER':>5}  {'BSC':>7}  {'AWGN':>7}  {'Ray':>7}  {'MaxΔ':>6}")
    for target_ber in [0.15, 0.20, 0.25, 0.30, 0.35]:
        b_val = bsc.get(str(target_ber))
        if b_val is None: continue
        a_val = r_val = None
        for s in awgn:
            if abs(awgn[s]['eq_ber'] - target_ber) < 0.015:
                a_val = awgn[s]['acc']; break
        for s in ray:
            if abs(ray[s]['eq_ber'] - target_ber) < 0.015:
                r_val = ray[s]['acc']; break
        if a_val and r_val:
            vals = [b_val, a_val, r_val]
            print(f"  {target_ber:>5.2f}  {b_val:>7.2f}  {a_val:>7.2f}  {r_val:>7.2f}  {max(vals)-min(vals):>6.2f}")
