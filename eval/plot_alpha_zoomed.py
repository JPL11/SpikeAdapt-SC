#!/usr/bin/env python3
"""Alpha vs rho — zoomed plot with rho axis 0 to 1, BER <= 0.30 only.

Per the sketch: shows the threshold/peak structure of alpha_1,
where alpha_1 = (A(rho) - A(rho_max)) / A(rho_max).

  - For rho < rho_th: alpha < 0 (bandwidth too tight, accuracy hurts)
  - At rho = rho_max: alpha peaks (optimal operating point)
  - At rho = 1: alpha returns to baseline (reference)

Output: paper/figures/fig_alpha_vs_rho_zoomed.{pdf,png}
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('paper/figures', exist_ok=True)

with open('eval/seed_results/per_rho_scorer_results.json') as f:
    data = json.load(f)

# Add rho=1.0 reference: use the fixed rho=1.0 paper Table 2 numbers (seed 42)
# These are the "no-mask" SpikeAdapt at full rate from the paper
RHO_1_DATA = {
    'aid': {0.0: 95.49, 0.05: 95.49, 0.10: 95.49, 0.15: 95.60,
            0.20: 95.49, 0.25: 95.49, 0.30: 91.89},
    'resisc45': {0.0: 92.59, 0.05: 92.59, 0.10: 92.59, 0.15: 92.87,
                 0.20: 92.59, 0.25: 92.59, 0.30: 84.85},
}

# rho=0 endpoint: nothing transmitted -> chance accuracy
# AID has 30 classes, RESISC45 has 45 classes
RHO_0_DATA = {
    'aid':      {b: 100.0 / 30 for b in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]},
    'resisc45': {b: 100.0 / 45 for b in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]},
}

# Use 10-seed for rho=1.0 anchor at BER=0, 0.15, 0.30
# For other BERs we approximate with the rho=0.875 result (closest available)
for ds in ['aid', 'resisc45']:
    rho_1_dict = {str(k): v for k, v in RHO_1_DATA[ds].items()}
    data[ds]['1.0'] = rho_1_dict
    rho_0_dict = {str(k): v for k, v in RHO_0_DATA[ds].items()}
    data[ds]['0.0'] = rho_0_dict

BERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
COLORS = plt.cm.RdYlBu_r(np.linspace(0.05, 0.95, len(BERS)))


plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12,
                     'axes.titlesize': 13, 'legend.fontsize': 9})

fig, axes = plt.subplots(1, 2, figsize=(14, 6.0))

for col, ds in enumerate(['aid', 'resisc45']):
    ax = axes[col]
    ds_label = 'AID' if ds == 'aid' else 'RESISC45'
    d = data[ds]
    rhos = sorted([float(r) for r in d.keys()])
    rho_max_str = '1.0'  # reference: full rate

    # Plot alpha curves for BER <= 0.30
    for i, ber in enumerate(BERS):
        a_max = d[rho_max_str].get(str(ber), 0)
        if a_max < 0.1:
            continue
        alphas = [100 * (d[str(r)].get(str(ber), 0) - a_max) / a_max for r in rhos]
        label = 'Clean' if ber == 0 else f'BER={ber:.2f}'
        ax.plot(rhos, alphas, 'o-', color=COLORS[i],
                linewidth=2.2, markersize=7, label=label, zorder=4)

    # Annotate the peak for BER=0.30 (the most interesting curve)
    ber_peak = 0.30
    a_max_peak = d[rho_max_str][str(ber_peak)]
    alphas_peak = [100 * (d[str(r)].get(str(ber_peak), 0) - a_max_peak) / a_max_peak
                   for r in rhos]
    peak_idx = int(np.argmax(alphas_peak))
    rho_star = rhos[peak_idx]
    alpha_star = alphas_peak[peak_idx]

    # Find threshold rho_th (where alpha crosses 0 for BER=0.30, going up)
    # interp between consecutive points where alpha crosses zero from below
    rho_th = None
    for k in range(len(rhos) - 1):
        if alphas_peak[k] < 0 and alphas_peak[k + 1] >= 0:
            x1, x2 = rhos[k], rhos[k + 1]
            y1, y2 = alphas_peak[k], alphas_peak[k + 1]
            rho_th = x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
            break

    # Mark rho_max (peak) with star
    ax.scatter([rho_star], [alpha_star], color='#d32f2f',
               marker='*', s=350, zorder=10, edgecolors='black', linewidths=1.2)
    # Annotation placed BELOW the curve to avoid title overlap
    ax.annotate(f'$\\rho^*\\!\\approx\\!{rho_star:.3f}$\n'
                f'$\\alpha\\!\\approx\\!{alpha_star:+.1f}\\%$',
                xy=(rho_star, alpha_star),
                xytext=(rho_star + 0.07, alpha_star - 1.5),
                fontsize=10, fontweight='bold', color='#d32f2f',
                bbox=dict(boxstyle='round,pad=0.3', fc='#ffebee',
                          ec='#d32f2f', alpha=0.95),
                arrowprops=dict(arrowstyle='-', color='#d32f2f', lw=0.7))

    # Mark rho_th (threshold) where alpha=0
    if rho_th is not None:
        ax.axvline(x=rho_th, color='#1976d2', linestyle='--',
                    linewidth=1.2, alpha=0.7, zorder=3)

    # Zero line
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.7)

    # Shaded region: alpha > 0 (masking helps)
    rhos_arr = np.array(rhos)
    alphas_arr = np.array(alphas_peak)
    ax.fill_between(rhos_arr, 0, alphas_arr,
                    where=alphas_arr >= 0,
                    color='#4caf50', alpha=0.10,
                    label='_masking helps')
    ax.fill_between(rhos_arr, 0, alphas_arr,
                    where=alphas_arr < 0,
                    color='#f44336', alpha=0.08,
                    label='_masking hurts')

    ax.set_xlabel('Transmission rate $\\rho$', fontsize=12)
    ax.set_ylabel('$\\alpha = (A(\\rho) - A(\\rho{=}1))\\, /\\, A(\\rho{=}1)$  [%]',
                  fontsize=11)
    ax.set_title(f'{ds_label}: $\\alpha$ vs $\\rho$ (BER $\\leq 0.30$)',
                 fontsize=12, fontweight='bold')

    # Axis settings
    ax.set_xlim(0, 1.0)
    ax.set_xticks(np.arange(0, 1.05, 0.1))

    # Y-axis zoom (percent): per-dataset request
    if ds == 'aid':
        ax.set_ylim(-3, 3)
    else:
        ax.set_ylim(-5, 5)
    ax.grid(True, alpha=0.3)

    # Place rho_th label inside plot at top
    if rho_th is not None:
        y_label_pos = ax.get_ylim()[1] - 0.4
        ax.text(rho_th + 0.005, y_label_pos,
                f'$\\rho_{{\\mathrm{{th}}}}\\!\\approx\\!{rho_th:.3f}$',
                fontsize=9.5, color='#1976d2', fontweight='bold',
                ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.25', fc='white',
                          ec='#1976d2', alpha=0.95))

    # Region labels
    ax.text(0.15, ax.get_ylim()[0] + 0.4,
            'masking\nhurts',
            fontsize=9, color='#c62828', alpha=0.7, ha='center', va='bottom',
            style='italic')
    ax.text(0.78, ax.get_ylim()[1] - 0.4,
            'masking\nhelps',
            fontsize=9, color='#2e7d32', alpha=0.8, ha='center', va='top',
            style='italic')

# Shared legend at the bottom (outside the axes so it doesn't cover points)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(labels),
           fontsize=9.5, framealpha=0.95, bbox_to_anchor=(0.5, -0.01))

plt.tight_layout(rect=[0, 0.07, 1, 1])
fig.savefig('paper/figures/fig_alpha_vs_rho_zoomed.pdf',
            dpi=300, bbox_inches='tight')
fig.savefig('paper/figures/fig_alpha_vs_rho_zoomed.png',
            dpi=200, bbox_inches='tight')
plt.close()
print('Saved: paper/figures/fig_alpha_vs_rho_zoomed.pdf')
print('Saved: paper/figures/fig_alpha_vs_rho_zoomed.png')

# Print key threshold/peak values for paper text
print('\n=== KEY VALUES (BER=0.30 alpha curve) ===')
for ds in ['aid', 'resisc45']:
    d = data[ds]
    rhos = sorted([float(r) for r in d.keys()])
    a_max = d['1.0']['0.3']
    alphas = [(d[str(r)].get('0.3', 0) - a_max) / a_max for r in rhos]
    peak_idx = int(np.argmax(alphas))
    print(f'\n{ds.upper()}:')
    print(f'  rho_max (full rate ref): A=1.0, A(1)={a_max:.2f}%')
    print(f'  rho_optimal: rho={rhos[peak_idx]:.3f}, '
          f'A={d[str(rhos[peak_idx])]["0.3"]:.2f}%, '
          f'alpha={alphas[peak_idx]*100:+.2f}%')
    # threshold
    for k in range(len(rhos) - 1):
        if alphas[k] < 0 and alphas[k + 1] >= 0:
            x1, x2 = rhos[k], rhos[k + 1]
            y1, y2 = alphas[k], alphas[k + 1]
            rho_th = x1 + (0 - y1) * (x2 - x1) / (y2 - y1)
            print(f'  rho_th (threshold): {rho_th:.3f}  '
                  f'(below this, masking hurts)')
            break
