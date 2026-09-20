#!/usr/bin/env python3
"""Alpha vs rho from the 10-seed sweep (replaces single-seed version).

alpha(rho, BER) = (A(rho) - A(rho=1)) / A(rho=1), computed PER SEED (paired)
then averaged; shaded band = paired std for the BER=0.30 curve.

Data: eval/seed_results/per_rho_10seed_valproto.json  (shared scorer, rate override,
matched per-seed data splits/backbones; see eval/eval_multiseed_rho_sweep.py)

Output: paper/figures/fig_alpha_vs_rho_zoomed.{pdf,png}  (same path the
paper includes; overwrites the old seed-42 figure)

Usage:
    python eval/plot_alpha_10seed.py
"""

import json
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('eval/seed_results/per_rho_10seed_valproto.json') as f:
    DATA = json.load(f)

RHOS = [0.50, 0.625, 0.75, 0.875, 1.0]
BERS = [0.0, 0.15, 0.30]
STYLES = {0.0: dict(color='#2166ac', marker='o', ls='-'),
          0.15: dict(color='#f4a582', marker='s', ls='--'),
          0.30: dict(color='#b2182b', marker='^', ls='-')}

plt.rcParams.update({'font.size': 13, 'axes.labelsize': 14,
                     'axes.titlesize': 14, 'legend.fontsize': 12})


def key(x):
    return str(x) if str(x) != '0.1' else '0.1'


T975_DF9 = 2.262   # two-sided 97.5% Student-t quantile, n=10 seeds


def paired_alpha(ds, ber):
    """Per-seed paired alpha curves -> (mean, 95% CI half-width) over RHOS, %."""
    d = DATA[ds]
    seeds = sorted(d.keys(), key=int)
    curves = []
    for s in seeds:
        bk = next(k for k in d[s]['1.0'] if float(k) == ber)
        a1 = d[s]['1.0'][bk]
        row = []
        for r in RHOS:
            rk = next(k for k in d[s] if float(k) == r)
            bk2 = next(k for k in d[s][rk] if float(k) == ber)
            row.append(100.0 * (d[s][rk][bk2] - a1) / a1)
        curves.append(row)
    arr = np.array(curves)
    n = arr.shape[0]
    ci = T975_DF9 * arr.std(0, ddof=1) / np.sqrt(n)
    return arr.mean(0), ci


fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8))

for col, ds in enumerate(['aid', 'resisc45']):
    ax = axes[col]
    ds_label = 'AID' if ds == 'aid' else 'RESISC45'

    for i, ber in enumerate(BERS):
        mean, ci = paired_alpha(ds, ber)
        st = STYLES[ber]
        label = 'Clean' if ber == 0 else f'BER={ber:.2f}'
        ax.plot(RHOS, mean, marker=st['marker'], ls=st['ls'],
                color=st['color'], linewidth=2.4, markersize=8,
                label=label, zorder=4)
        if ber == 0.30:  # 95% CI band, headline curve only
            ax.fill_between(RHOS, mean - ci, mean + ci,
                            color=st['color'], alpha=0.12, zorder=2)

    # threshold + peak annotations from the BER=0.30 mean curve
    mean30, ci30 = paired_alpha(ds, 0.30)
    peak_idx = int(np.argmax(mean30))
    rho_star, alpha_star = RHOS[peak_idx], mean30[peak_idx]
    rho_th = None
    for k in range(len(RHOS) - 1):
        if mean30[k] < 0 <= mean30[k + 1]:
            x1, x2, y1, y2 = RHOS[k], RHOS[k + 1], mean30[k], mean30[k + 1]
            rho_th = x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
            break

    if rho_th is not None:
        ax.axvline(x=rho_th, color='#1976d2', linestyle='--', linewidth=1.2,
                   alpha=0.7, zorder=3)
        ax.text(rho_th - 0.012, 2.6 if ds == 'aid' else 4.3,
                f'$\\rho_{{\\mathrm{{th}}}}\\!\\approx\\!{rho_th:.2f}$',
                fontsize=12, color='#1976d2', fontweight='bold',
                ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.25', fc='white',
                          ec='#1976d2', alpha=0.95))

    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.set_xlabel('Transmission rate $\\rho$', fontsize=14)
    ax.set_ylabel('$\\alpha(\\rho)$  [%]', fontsize=14)
    ax.set_title(ds_label + ' (10-seed paired)', fontsize=14,
                 fontweight='bold')
    ax.set_xlim(0.48, 1.02)
    ax.set_xticks(RHOS)
    ax.set_ylim(-3, 3) if ds == 'aid' else ax.set_ylim(-5, 5)
    ax.grid(True, alpha=0.3)

handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles, labels, loc='lower right', fontsize=12,
               framealpha=0.95)
plt.tight_layout()
fig.savefig('paper/figures/fig_alpha_vs_rho_zoomed.pdf', dpi=300,
            bbox_inches='tight')
fig.savefig('paper/figures/fig_alpha_vs_rho_zoomed.png', dpi=200,
            bbox_inches='tight')
plt.close()
print('Saved: paper/figures/fig_alpha_vs_rho_zoomed.{pdf,png}')

for ds in ['aid', 'resisc45']:
    mean30, ci30 = paired_alpha(ds, 0.30)
    pk = int(np.argmax(mean30))
    print(f'{ds}: peak alpha={mean30[pk]:+.2f}% (95% CI half-width '
          f'{ci30[pk]:.2f}) at rho={RHOS[pk]:g} (BER=0.30, 10-seed paired)')
