#!/usr/bin/env python3
"""Generate enhanced ρ sweep figure with 7 BER curves + zoomed channel comparison."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'legend.fontsize': 6.5, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'axes.linewidth': 0.8,
})

OUT = './paper/figures/'

# ============================================================================
# FIGURE 1: Extended ρ sweep with 7 BER curves
# ============================================================================
d = json.load(open(f'{OUT}/rho_sweep_extended.json'))

rho_values = [0.10, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 1.0]
ber_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
bw_save = [(1 - r) * 100 for r in rho_values]

colors = ['#1565C0', '#1E88E5', '#43A047', '#FDD835', '#FF8F00', '#E65100', '#B71C1C']
markers = ['o', 'D', 's', '^', 'v', 'P', '*']
linestyles = ['-', '--', '-.', ':', '-', '--', '-.']

fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.0))

for idx, (ds, title, ylim) in enumerate([
    ('aid', 'AID (50/50 split)', (22, 80)),
    ('resisc45', 'RESISC45 (20/80 split)', (15, 65)),
]):
    ax = axes[idx]
    ds_data = d[ds]

    for bi, ber in enumerate(ber_levels):
        accs = [ds_data[str(r)][str(ber)]['mean'] for r in rho_values]
        label = 'Clean' if ber == 0 else f'BER={ber:.2f}'
        lw = 2.0 if bi in (0, 6) else 1.4
        ms = 5 if bi in (0, 6) else 3.5
        ax.plot(bw_save, accs, linestyles[bi], marker=markers[bi], color=colors[bi],
                label=label, lw=lw, ms=ms, zorder=7 - bi)

    # Annotate ρ=0.625 best under BER=0.30
    best_acc = ds_data['0.625']['0.3']['mean']
    base_acc = ds_data['1.0']['0.3']['mean']
    delta = best_acc - base_acc
    ax.scatter([37.5], [best_acc], s=120, c='gold', edgecolor='#B71C1C',
              linewidth=1.5, zorder=10, marker='*')
    ax.annotate(f'ρ=0.625\n{delta:+.1f}pp', (37.5, best_acc),
               textcoords='offset points', xytext=(15, -15), fontsize=6.5,
               color='#B71C1C', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='#B71C1C', lw=0.8))

    ax.set_xlabel('Bandwidth Savings (%)')
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title(title, fontweight='bold')
    ax.set_ylim(ylim)
    ax.set_xlim(-3, 93)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.xaxis.set_major_locator(MultipleLocator(25))

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=7, fontsize=6.5,
           bbox_to_anchor=(0.5, -0.06), frameon=True, fancybox=True,
           framealpha=0.95, edgecolor='#ccc', columnspacing=0.8, handletextpad=0.3)
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.savefig(f'{OUT}/fig5_rho_sweep_7ber.pdf')
plt.savefig(f'{OUT}/fig5_rho_sweep_7ber.png', dpi=200)
plt.close()
print(f'✅ {OUT}fig5_rho_sweep_7ber.png')

# ============================================================================
# FIGURE 2: Zoomed-in channel comparison (BSC vs AWGN vs Rayleigh)
# ============================================================================
snn_mc = json.load(open('eval/multichannel_results_v2.json'))
C_BSC, C_AWGN, C_RAY = '#1a5276', '#c0392b', '#27ae60'

# Standard zoom
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
for ax, ds, title in [(axes[0], 'aid', 'AID'), (axes[1], 'resisc45', 'RESISC45')]:
    bsc = snn_mc[ds]['bsc']['results']
    awgn = snn_mc[ds]['awgn']['results']
    ray = snn_mc[ds]['rayleigh']['results']
    ax.plot([float(b) for b in bsc], list(bsc.values()), 'o-', color=C_BSC,
            label='BSC', ms=4, lw=1.8)
    ax.plot([awgn[s]['eq_ber'] for s in awgn], [awgn[s]['acc'] for s in awgn],
            's--', color=C_AWGN, label='AWGN (eq BER)', ms=4, lw=1.5)
    ax.plot([ray[s]['eq_ber'] for s in ray], [ray[s]['acc'] for s in ray],
            '^:', color=C_RAY, label='Rayleigh (eq BER)', ms=4, lw=1.5)
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
print(f'✅ {OUT}fig7_unified_ber_zoomed.png (standard)')

# Extra zoomed — BER >= 0.15
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
for ax, ds, title in [(axes[0], 'aid', 'AID'), (axes[1], 'resisc45', 'RESISC45')]:
    bsc = snn_mc[ds]['bsc']['results']
    awgn = snn_mc[ds]['awgn']['results']
    ray = snn_mc[ds]['rayleigh']['results']
    bf = sorted([(float(b), v) for b, v in bsc.items() if float(b) >= 0.15])
    af = sorted([(awgn[s]['eq_ber'], awgn[s]['acc']) for s in awgn if awgn[s]['eq_ber'] >= 0.15])
    rf = sorted([(ray[s]['eq_ber'], ray[s]['acc']) for s in ray if ray[s]['eq_ber'] >= 0.15])
    ax.plot([b for b, a in bf], [a for b, a in bf], 'o-', color=C_BSC, label='BSC', ms=6, lw=2.2)
    ax.plot([b for b, a in af], [a for b, a in af], 's--', color=C_AWGN, label='AWGN (eq BER)', ms=6, lw=2)
    ax.plot([b for b, a in rf], [a for b, a in rf], '^:', color=C_RAY, label='Rayleigh (eq BER)', ms=6, lw=2)

    for tb in [0.30, 0.35]:
        bv = bsc.get(str(tb))
        if not bv:
            continue
        av = rv = None
        for s in awgn:
            if abs(awgn[s]['eq_ber'] - tb) < 0.015:
                av = awgn[s]['acc']
                break
        for s in ray:
            if abs(ray[s]['eq_ber'] - tb) < 0.015:
                rv = ray[s]['acc']
                break
        if av and rv:
            vals = [bv, av, rv]
            delta = max(vals) - min(vals)
            ym = np.mean(vals)
            ox = 12 if tb < 0.35 else -35
            oy = -14 if tb < 0.35 else 8
            ax.annotate(f'max Δ={delta:.1f}pp', (tb, ym),
                       textcoords='offset points', xytext=(ox, oy),
                       fontsize=7, color='#333', fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='#666', lw=0.8),
                       bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='#ccc', alpha=0.8))
    ax.set_xlabel('Equivalent BER')
    ax.set_title(f'{title} (Detail)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right' if ds == 'resisc45' else 'lower left', fontsize=7, framealpha=0.9)
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
print(f'✅ {OUT}fig7_channel_detail_zoom.png (extra zoom)')

# Print data
print('\n=== Extended ρ sweep data ===')
for ds in ['aid', 'resisc45']:
    print(f'\n{ds.upper()}:')
    hdr = f'{"rho":>6} {"BW%":>5}  ' + '  '.join(f'{b:.2f}' for b in ber_levels)
    print(hdr)
    for r in rho_values:
        vals = [d[ds][str(r)][str(b)]['mean'] for b in ber_levels]
        bw = (1 - r) * 100
        print(f'{r:>6.3f} {bw:>5.1f}  ' + '  '.join(f'{v:>5.2f}' for v in vals))

print('\n=== Channel differences ===')
for ds in ['aid', 'resisc45']:
    bsc = snn_mc[ds]['bsc']['results']
    awgn = snn_mc[ds]['awgn']['results']
    ray = snn_mc[ds]['rayleigh']['results']
    print(f'\n{ds.upper()}:')
    for tb in [0.20, 0.25, 0.30, 0.35]:
        bv = bsc.get(str(tb))
        av = rv = None
        for s in awgn:
            if abs(awgn[s]['eq_ber'] - tb) < 0.015:
                av = awgn[s]['acc']
                break
        for s in ray:
            if abs(ray[s]['eq_ber'] - tb) < 0.015:
                rv = ray[s]['acc']
                break
        if bv and av and rv:
            vals = [bv, av, rv]
            print(f'  BER={tb:.2f}: BSC={bv:.2f} AWGN={av:.2f} Ray={rv:.2f} Δ={max(vals) - min(vals):.2f}')
