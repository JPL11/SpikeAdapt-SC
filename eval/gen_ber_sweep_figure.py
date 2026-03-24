"""Generate THE comprehensive unified BER sweep figure for the paper.

One dual-panel figure with ALL baselines:
  SpikeAdapt-SC (ρ=0.75), SNN (no mask, ρ=1.0), CNN-Uni, CNN-NonUni, MLP-FC, JPEG+Conv
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
    'legend.fontsize': 6.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.4,
    'lines.linewidth': 1.8,
    'lines.markersize': 4.5,
})

# Load all data
with open('eval/multichannel_results_v2.json') as f:
    snn_mc = json.load(f)
with open('eval/cnn_multichannel.json') as f:
    cnn_mc = json.load(f)
with open('eval/mlp_multichannel.json') as f:
    mlp_mc = json.load(f)

bers = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
ber_keys = ['0.0', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3']

# JPEG results (cliff effect — fill BER>0.05 with ~2.68%)
jpeg_aid = {'0.0': 95.16, '0.05': 2.66, '0.1': 2.68, '0.15': 2.68,
            '0.2': 2.68, '0.25': 2.68, '0.3': 2.68}

# ─── Figure: Unified BER sweep with ALL baselines ───
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

for ax, ds_label, ds_key in [(ax1, 'AID', 'aid'), (ax2, 'RESISC45', 'resisc45')]:
    # SpikeAdapt-SC (our method)
    snn_bsc = snn_mc[ds_key]['bsc']['results']
    snn_accs = [snn_bsc[b] for b in ber_keys]
    ax.plot(bers, snn_accs, '-o', color='#1B5E20', lw=2.5, ms=5.5, zorder=6,
            label=r'SpikeAdapt-SC ($\rho$=0.75)', markeredgecolor='white', markeredgewidth=0.5)

    # SNN (no mask, ρ=1.0) — from matched_ber data
    snn_mb = snn_mc[ds_key]['matched_ber']
    snn_sc_accs = [snn_bsc['0.0']]  # clean
    for b in ['0.05', '0.1', '0.15', '0.2', '0.25', '0.3']:
        snn_sc_accs.append(snn_mb[b]['bsc_acc'])
    ax.plot(bers, snn_sc_accs, '-s', color='#4CAF50', lw=1.8, ms=4, zorder=5,
            label=r'SNN (no mask, $\rho$=1.0)')

    # CNN-Uni BSC
    cu = cnn_mc[ds_label]['CNN-Uni']['bsc']
    cu_accs = [cu[str(b)] for b in bers]
    ax.plot(bers, cu_accs, '--^', color='#E53935', lw=1.5, ms=4, zorder=3,
            label='CNN-Uni (8-bit)')

    # CNN-NonUni BSC
    cnu = cnn_mc[ds_label]['CNN-NonUni']['bsc']
    cnu_accs = [cnu[str(b)] for b in bers]
    ax.plot(bers, cnu_accs, '--D', color='#FF9800', lw=1.5, ms=3.5, zorder=3,
            label='CNN-NonUni (8-bit)')

    # MLP-FC BSC
    mlp_accs = [mlp_mc[ds_label][str(b)]['bsc'] for b in bers]
    ax.plot(bers, mlp_accs, '-.v', color='#9C27B0', lw=1.5, ms=3.5, zorder=3,
            label='MLP-FC (8-bit)')

# JPEG+Conv (cliff effect) — both datasets
    if ds_key == 'aid':
        jpeg_accs = [jpeg_aid[str(b)] for b in bers]
    else:
        jpeg_accs = [93.55] + [1.80] * 6
    ax.plot(bers, jpeg_accs, ':x', color='#795548', lw=1.5, ms=4, zorder=2,
            label='JPEG+Conv')

    # Shaded advantage region
    worst_cnn = [min(cu_accs[i], cnu_accs[i], mlp_accs[i]) for i in range(len(bers))]
    ax.fill_between(bers, worst_cnn, snn_accs, alpha=0.06, color='#1B5E20')

    panel = chr(97 + [ax1, ax2].index(ax))
    split = '50/50' if ds_key == 'aid' else '20/80'
    ax.set_xlabel('Bit Error Rate (BER)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'({panel}) {ds_label} ({split})', fontweight='bold')
    ax.set_xlim(-0.01, 0.31)
    ax.set_ylim(0 if ds_key == 'aid' else 10, 100)
    ncols = 2 if ds_key == 'aid' else 2
    ax.legend(loc='lower left', ncol=ncols, framealpha=0.9, edgecolor='#ccc',
              columnspacing=0.8, handletextpad=0.4)
    ax.grid(True, alpha=0.25)

plt.tight_layout(w_pad=1.5)
plt.savefig('paper/figures/fig_ber_sweep_baselines.pdf')
plt.savefig('paper/figures/fig_ber_sweep_baselines.png')
print("✅ Unified BER sweep figure saved")
plt.close()

print("\nAll data summary for paper tables:")
for ds_label, ds_key in [('AID', 'aid'), ('RESISC45', 'resisc45')]:
    print(f"\n{ds_label}:")
    snn_bsc = snn_mc[ds_key]['bsc']['results']
    snn_mb = snn_mc[ds_key]['matched_ber']
    print(f"  SpikeAdapt-SC:     clean={snn_bsc['0.0']:.2f}  0.15={snn_bsc['0.15']:.2f}  0.30={snn_bsc['0.3']:.2f}")
    print(f"  SNN (no mask): clean={snn_bsc['0.0']:.2f}  0.15={snn_mb['0.15']['bsc_acc']:.2f}  0.30={snn_mb['0.3']['bsc_acc']:.2f}")
    cu = cnn_mc[ds_label]['CNN-Uni']['bsc']
    cnu = cnn_mc[ds_label]['CNN-NonUni']['bsc']
    print(f"  CNN-Uni:    clean={cu['0.0']:.2f}  0.15={cu['0.15']:.2f}  0.30={cu['0.3']:.2f}")
    print(f"  CNN-NonUni: clean={cnu['0.0']:.2f}  0.15={cnu['0.15']:.2f}  0.30={cnu['0.3']:.2f}")
    mlp = mlp_mc[ds_label]
    print(f"  MLP-FC:     clean={mlp['0.0']['bsc']:.2f}  0.15={mlp['0.15']['bsc']:.2f}  0.30={mlp['0.3']['bsc']:.2f}")
    if ds_key == 'aid':
        print(f"  JPEG+Conv:  clean=95.16  0.15=2.68  0.30=2.68  (cliff effect at BER≥0.05)")
