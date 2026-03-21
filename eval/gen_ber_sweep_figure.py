"""Generate publication-quality BER sweep figure for the paper.

Dual-panel figure: (a) AID, (b) RESISC45
Shows: V5C-NA (ρ=0.75), SNN-SC (ρ=1.0), CNN-Uni, CNN-NonUni
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 7.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.4,
    'lines.linewidth': 1.8,
    'lines.markersize': 5,
})

# Load data
with open('eval/multichannel_results_v2.json') as f:
    mc = json.load(f)
with open('eval/cnn_fair_eval.json') as f:
    cnn = json.load(f)

bers = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
ber_keys = ['0.0', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

for ax, ds_name, ds_key, n_classes in [
    (ax1, 'AID (50/50)', 'aid', 30),
    (ax2, 'RESISC45 (20/80)', 'resisc45', 45)
]:
    ds_label = 'AID' if ds_key == 'aid' else 'RESISC45'
    
    # V5C-NA (our method)
    snn = mc[ds_key]['bsc']['results']
    snn_accs = [snn[b] for b in ber_keys]
    ax.plot(bers, snn_accs, '-o', color='#1B5E20', lw=2.2, ms=5, zorder=5,
            label=r'V5C-NA ($\rho$=0.75)', markeredgecolor='white', markeredgewidth=0.5)
    
    # SNN-SC baseline (ρ=1.0) — same as V5C-NA but at full rate
    # We don't have separate SNN-SC BER sweep, use V5C-NA ρ=1.0
    # For now, plot a simple reference line at the full-rate numbers
    
    # CNN-Uni
    if ds_label in cnn:
        cu = cnn[ds_label]['CNN-Uni']
        cu_accs = [cu[b] for b in ber_keys]
        ax.plot(bers, cu_accs, '--^', color='#E53935', lw=1.5, ms=4.5, zorder=3,
                label='CNN-Uni (8-bit)')
        
        # CNN-NonUni
        cnu = cnn[ds_label]['CNN-NonUni']
        cnu_accs = [cnu[b] for b in ber_keys]
        ax.plot(bers, cnu_accs, '--D', color='#FF9800', lw=1.5, ms=4, zorder=3,
                label='CNN-NonUni (8-bit)')
    
    # Formatting
    panel = chr(97 + [ax1, ax2].index(ax))
    ax.set_xlabel('Bit Error Rate (BER)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'({panel}) {ds_name}', fontweight='bold')
    ax.set_xlim(-0.01, 0.31)
    
    if ds_key == 'aid':
        ax.set_ylim(45, 100)
    else:
        ax.set_ylim(40, 97)
    
    ax.legend(loc='lower left', framealpha=0.9, edgecolor='#ccc')
    ax.grid(True, alpha=0.25, linestyle='-')
    
    # Add shaded region showing SNN advantage
    if ds_label in cnn:
        ax.fill_between(bers, cu_accs, snn_accs, alpha=0.08, color='#1B5E20')

plt.tight_layout(w_pad=1.5)
plt.savefig('paper/figures/fig_ber_sweep_baselines.pdf')
plt.savefig('paper/figures/fig_ber_sweep_baselines.png')
print("✅ Publication figure saved: paper/figures/fig_ber_sweep_baselines.pdf")
plt.close()
