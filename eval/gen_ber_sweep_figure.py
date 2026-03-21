"""Generate comprehensive multichannel BER sweep figures for the paper.

Creates two figures:
1. BER sweep with all baselines (BSC channel, dual-panel: AID + RESISC45)
2. Matched-BER multichannel comparison (BSC vs AWGN vs Rayleigh)
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
    'legend.fontsize': 7,
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
    snn_mc = json.load(f)
with open('eval/cnn_multichannel.json') as f:
    cnn_mc = json.load(f)

bers = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
ber_keys = ['0.0', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3']

# ─── Figure 1: BER sweep with all baselines (BSC) ───
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

for ax, ds_label, ds_key in [(ax1, 'AID', 'aid'), (ax2, 'RESISC45', 'resisc45')]:
    # V5C-NA (SNN)
    snn_bsc = snn_mc[ds_key]['bsc']['results']
    snn_accs = [snn_bsc[b] for b in ber_keys]
    ax.plot(bers, snn_accs, '-o', color='#1B5E20', lw=2.2, ms=5, zorder=5,
            label=r'V5C-NA ($\rho$=0.75)', markeredgecolor='white', markeredgewidth=0.5)

    # CNN-Uni BSC
    cu_bsc = cnn_mc[ds_label]['CNN-Uni']['bsc']
    cu_accs = [cu_bsc[str(b)] for b in bers]
    ax.plot(bers, cu_accs, '--^', color='#E53935', lw=1.5, ms=4.5, zorder=3,
            label='CNN-Uni (8-bit)')

    # CNN-NonUni BSC
    cnu_bsc = cnn_mc[ds_label]['CNN-NonUni']['bsc']
    cnu_accs = [cnu_bsc[str(b)] for b in bers]
    ax.plot(bers, cnu_accs, '--D', color='#FF9800', lw=1.5, ms=4, zorder=3,
            label='CNN-NonUni (8-bit)')

    # Shaded advantage region
    ax.fill_between(bers, cu_accs, snn_accs, alpha=0.08, color='#1B5E20')

    panel = chr(97 + [ax1, ax2].index(ax))
    split = '50/50' if ds_key == 'aid' else '20/80'
    ax.set_xlabel('Bit Error Rate (BER)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'({panel}) {ds_label} ({split})', fontweight='bold')
    ax.set_xlim(-0.01, 0.31)
    ax.set_ylim(40 if ds_key == 'resisc45' else 45, 100)
    ax.legend(loc='lower left', framealpha=0.9, edgecolor='#ccc')
    ax.grid(True, alpha=0.25)

plt.tight_layout(w_pad=1.5)
plt.savefig('paper/figures/fig_ber_sweep_baselines.pdf')
plt.savefig('paper/figures/fig_ber_sweep_baselines.png')
print("✅ Figure 1: BER sweep with baselines saved")
plt.close()

# ─── Figure 2: Matched-BER multichannel comparison ───
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))

for col, (ds_label, ds_key) in enumerate([('AID', 'aid'), ('RESISC45', 'resisc45')]):
    for row, method in enumerate(['CNN-Uni', 'CNN-NonUni']):
        ax = axes[row, col]
        mb = cnn_mc[ds_label][method]['matched_ber']

        bsc_accs = [mb[str(b)]['bsc'] for b in bers]
        awgn_accs = [mb[str(b)]['awgn'] for b in bers]
        ray_accs = [mb[str(b)]['rayleigh'] for b in bers]

        ax.plot(bers, bsc_accs, '-o', color='#1565C0', lw=1.8, ms=4, label='BSC')
        ax.plot(bers, awgn_accs, '--s', color='#E65100', lw=1.5, ms=3.5, label='AWGN')
        ax.plot(bers, ray_accs, '-.^', color='#6A1B9A', lw=1.5, ms=3.5, label='Rayleigh')

        # Also show SNN for reference
        snn_mb = snn_mc[ds_key]['matched_ber']
        snn_bsc = [snn_mb[b]['bsc_acc'] for b in ['0.01', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3']]
        # Prepend clean
        snn_clean = snn_mc[ds_key]['bsc']['results']['0.0']

        panel = chr(97 + row*2 + col)
        split = '50/50' if ds_key == 'aid' else '20/80'
        ax.set_title(f'({panel}) {method} — {ds_label}', fontweight='bold', fontsize=9)
        ax.set_xlabel('Equivalent BER')
        ax.set_ylabel('Accuracy (%)')
        ax.set_xlim(-0.01, 0.31)
        ax.legend(fontsize=6.5, loc='lower left')
        ax.grid(True, alpha=0.25)

plt.tight_layout(h_pad=1.5, w_pad=1.5)
plt.savefig('paper/figures/fig_multichannel_cnn.pdf')
plt.savefig('paper/figures/fig_multichannel_cnn.png')
print("✅ Figure 2: Multichannel CNN comparison saved")
plt.close()

# ─── Figure 3: Combined SNN+CNN multichannel (one per dataset) ───
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

for ax, ds_label, ds_key in [(ax1, 'AID', 'aid'), (ax2, 'RESISC45', 'resisc45')]:
    # SNN BSC
    snn_bsc = snn_mc[ds_key]['bsc']['results']
    snn_accs = [snn_bsc[b] for b in ber_keys]
    ax.plot(bers, snn_accs, '-o', color='#1B5E20', lw=2.2, ms=5, zorder=5,
            label='V5C-NA (BSC)', markeredgecolor='white', markeredgewidth=0.5)

    # SNN AWGN and Rayleigh at matched BER
    snn_mb = snn_mc[ds_key]['matched_ber']
    snn_awgn = [snn_bsc['0.0']] + [snn_mb[b]['awgn_acc'] for b in ['0.05', '0.1', '0.15', '0.2', '0.25', '0.3']]
    snn_ray = [snn_bsc['0.0']] + [snn_mb[b]['ray_acc'] for b in ['0.05', '0.1', '0.15', '0.2', '0.25', '0.3']]
    ax.plot(bers, snn_awgn, '--s', color='#388E3C', lw=1.5, ms=3.5, zorder=4, label='V5C-NA (AWGN)')
    ax.plot(bers, snn_ray, '-.^', color='#66BB6A', lw=1.5, ms=3.5, zorder=4, label='V5C-NA (Rayleigh)')

    # CNN-Uni BSC/AWGN/Rayleigh
    cu = cnn_mc[ds_label]['CNN-Uni']['matched_ber']
    cu_bsc = [cu[str(b)]['bsc'] for b in bers]
    cu_awgn = [cu[str(b)]['awgn'] for b in bers]
    cu_ray = [cu[str(b)]['rayleigh'] for b in bers]
    ax.plot(bers, cu_bsc, '-^', color='#E53935', lw=1.5, ms=4, zorder=3, label='CNN-Uni (BSC)')
    ax.plot(bers, cu_awgn, '--v', color='#EF5350', lw=1.0, ms=3, zorder=2, alpha=0.7, label='CNN-Uni (AWGN)')
    ax.plot(bers, cu_ray, '-.x', color='#EF9A9A', lw=1.0, ms=3, zorder=2, alpha=0.7, label='CNN-Uni (Rayleigh)')

    panel = chr(97 + [ax1, ax2].index(ax))
    split = '50/50' if ds_key == 'aid' else '20/80'
    ax.set_xlabel('Equivalent BER')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'({panel}) {ds_label} ({split})', fontweight='bold')
    ax.set_xlim(-0.01, 0.31)
    ax.set_ylim(40 if ds_key == 'resisc45' else 45, 100)
    ax.legend(fontsize=5.8, loc='lower left', ncol=2, framealpha=0.9, edgecolor='#ccc')
    ax.grid(True, alpha=0.25)

plt.tight_layout(w_pad=1.5)
plt.savefig('paper/figures/fig_multichannel_unified.pdf')
plt.savefig('paper/figures/fig_multichannel_unified.png')
print("✅ Figure 3: Unified multichannel comparison saved")
plt.close()

print("\n✅ All publication figures generated:")
print("  paper/figures/fig_ber_sweep_baselines.pdf — BER sweep with CNN baselines")
print("  paper/figures/fig_multichannel_cnn.pdf — CNN channel comparison grid")
print("  paper/figures/fig_multichannel_unified.pdf — SNN vs CNN under all channels")
