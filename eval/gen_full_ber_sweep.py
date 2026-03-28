#!/usr/bin/env python3
"""Generate final dual-panel BER sweep figure with ALL baselines.

Uses ONLY pre-evaluated data. No GPU evaluation needed.

Baselines with correct data:
  - SpikeAdapt-SC      → ber_sweep_all_corrected.json (separate model)
  - SNN-SC             → ber_sweep_all_corrected.json (separate model)
  - CNN-Uni (8-bit)    → cnn_multichannel.json
  - CNN-NonUni (8-bit) → cnn_multichannel.json
  - CNN-1bit           → archive/eval/1bit_baseline_results.json (dedicated BinaryCNN_SC)
  - MLP-FC (8-bit)     → mlp_multichannel.json
  - JPEG+Conv          → hardcoded cliff effect

Legend: placed BELOW the plot area (outside axes) — like the user's reference.
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 7,
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

OUT = './paper/figures/'
BER_POINTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# ============================================================================
# LOAD ALL DATA
# ============================================================================
print("Loading data...")

with open(f'{OUT}/ber_sweep_all_corrected.json') as f:
    snn_data = json.load(f)

with open('eval/cnn_multichannel.json') as f:
    cnn_mc = json.load(f)

with open('eval/mlp_multichannel.json') as f:
    mlp_mc = json.load(f)

with open('archive/eval/1bit_baseline_results.json') as f:
    cnn1bit = json.load(f)

# JPEG+Conv (cliff effect — collapses to random at BER ≥ 0.05)
jpeg = {
    'AID': {'0.0': 95.16, '0.05': 2.66, '0.1': 2.68, '0.15': 2.68,
            '0.2': 2.68, '0.25': 2.68, '0.3': 2.68},
    'RESISC45': {'0.0': 93.55, '0.05': 1.80, '0.1': 1.80, '0.15': 1.80,
                 '0.2': 1.80, '0.25': 1.80, '0.3': 1.80},
}

# Build unified structure
all_data = {}
for ds_name, ds_key in [('AID', 'aid'), ('RESISC45', 'resisc45')]:
    sa = snn_data[ds_key]['spikeadapt_sc']
    sc = snn_data[ds_key]['snn_sc']
    cu = cnn_mc[ds_name]['CNN-Uni']['bsc']
    cnu = cnn_mc[ds_name]['CNN-NonUni']['bsc']
    mlp = {str(b): mlp_mc[ds_name][str(b)]['bsc'] for b in BER_POINTS}
    c1 = cnn1bit[ds_key]
    jp = jpeg[ds_name]

    all_data[ds_name] = {
        'SpikeAdapt-SC': {str(b): sa[str(b)]['mean'] for b in BER_POINTS},
        'SNN-SC': {str(b): sc[str(b)]['mean'] for b in BER_POINTS},
        'CNN-Uni': {str(b): cu[str(b)] for b in BER_POINTS},
        'CNN-NonUni': {str(b): cnu[str(b)] for b in BER_POINTS},
        'CNN-1bit': {str(b): c1[str(b)] for b in BER_POINTS},
        'MLP-FC': mlp,
        'JPEG+Conv': jp,
    }


# ============================================================================
# FIGURE STYLES
# ============================================================================
STYLES = {
    'SpikeAdapt-SC': {'color': '#1B5E20', 'marker': 'o', 'ls': '-',   'lw': 2.5, 'ms': 5.5, 'z': 6,
                       'label': r'SpikeAdapt-SC ($\rho$=0.75)'},
    'SNN-SC':        {'color': '#4CAF50', 'marker': 's', 'ls': '-',   'lw': 1.8, 'ms': 4.5, 'z': 5,
                       'label': r'SNN-SC (no mask, $\rho$=1.0)'},
    'CNN-Uni':       {'color': '#E53935', 'marker': '^', 'ls': '--',  'lw': 1.5, 'ms': 4,   'z': 3,
                       'label': 'CNN-Uni (8-bit)'},
    'CNN-NonUni':    {'color': '#FF9800', 'marker': 'D', 'ls': '--',  'lw': 1.5, 'ms': 3.5, 'z': 3,
                       'label': 'CNN-NonUni (8-bit)'},
    'CNN-1bit':      {'color': '#0D47A1', 'marker': 'P', 'ls': '--',  'lw': 1.5, 'ms': 4.5, 'z': 4,
                       'label': 'CNN-1bit'},
    'MLP-FC':        {'color': '#9C27B0', 'marker': 'v', 'ls': '-.',  'lw': 1.5, 'ms': 3.5, 'z': 2,
                       'label': 'MLP-FC (8-bit)'},
    'JPEG+Conv':     {'color': '#795548', 'marker': 'x', 'ls': ':',   'lw': 1.5, 'ms': 4,   'z': 2,
                       'label': 'JPEG+Conv'},
}

PLOT_ORDER = ['SpikeAdapt-SC', 'SNN-SC', 'CNN-Uni', 'CNN-NonUni', 'CNN-1bit',
              'MLP-FC', 'JPEG+Conv']


# ============================================================================
# FIGURE 1: Full range
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))

for ax, ds_name, split in [(ax1, 'AID', '50/50'), (ax2, 'RESISC45', '20/80')]:
    dd = all_data[ds_name]
    for method in PLOT_ORDER:
        s = STYLES[method]
        accs = [dd[method][str(b)] for b in BER_POINTS]
        mec = 'white' if method == 'SpikeAdapt-SC' else s['color']
        mew = 0.5 if method == 'SpikeAdapt-SC' else 0.3
        ax.plot(BER_POINTS, accs, f"{s['ls']}", marker=s['marker'], color=s['color'],
                lw=s['lw'], ms=s['ms'], zorder=s['z'], label=s['label'],
                markeredgecolor=mec, markeredgewidth=mew)

    panel = chr(97 + [ax1, ax2].index(ax))
    ax.set_xlabel('Bit Error Rate (BER)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'({panel}) {ds_name} ({split})', fontweight='bold')
    ax.set_xlim(-0.01, 0.31)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=6.5,
           bbox_to_anchor=(0.5, -0.08), frameon=True, fancybox=True,
           framealpha=0.95, edgecolor='#ccc', columnspacing=0.8, handletextpad=0.3)
plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
plt.savefig(f'{OUT}/fig7_all_baselines.pdf')
plt.savefig(f'{OUT}/fig7_all_baselines.png', dpi=200)
plt.close()
print(f"✅ {OUT}fig7_all_baselines.png")


# ============================================================================
# FIGURE 2: Zoomed (75-97%) — SNN and CNN methods only
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))

zoom_methods = ['SpikeAdapt-SC', 'SNN-SC', 'CNN-Uni', 'CNN-NonUni', 'CNN-1bit', 'MLP-FC']
for ax, ds_name, split in [(ax1, 'AID', '50/50'), (ax2, 'RESISC45', '20/80')]:
    dd = all_data[ds_name]
    for method in zoom_methods:
        s = STYLES[method]
        accs = [dd[method][str(b)] for b in BER_POINTS]
        mec = 'white' if method == 'SpikeAdapt-SC' else s['color']
        mew = 0.5 if method == 'SpikeAdapt-SC' else 0.3
        ax.plot(BER_POINTS, accs, f"{s['ls']}", marker=s['marker'], color=s['color'],
                lw=s['lw'], ms=s['ms'], zorder=s['z'], label=s['label'],
                markeredgecolor=mec, markeredgewidth=mew)

    panel = chr(97 + [ax1, ax2].index(ax))
    ax.set_xlabel('Bit Error Rate (BER)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'({panel}) {ds_name} ({split})', fontweight='bold')
    ax.set_xlim(-0.01, 0.31)
    ax.set_ylim(75, 97)
    ax.grid(True, alpha=0.25)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=6.5,
           bbox_to_anchor=(0.5, -0.08), frameon=True, fancybox=True,
           framealpha=0.95, edgecolor='#ccc', columnspacing=0.8, handletextpad=0.3)
plt.tight_layout()
plt.subplots_adjust(bottom=0.24)
plt.savefig(f'{OUT}/fig7_all_baselines_zoomed.pdf')
plt.savefig(f'{OUT}/fig7_all_baselines_zoomed.png', dpi=200)
plt.close()
print(f"✅ {OUT}fig7_all_baselines_zoomed.png")


# ============================================================================
# TABLE
# ============================================================================
print(f"\n{'='*70}")
print("  PAPER TABLE: All baselines BER sweep")
print(f"{'='*70}")

for ds_name in ['AID', 'RESISC45']:
    dd = all_data[ds_name]
    print(f"\n  {ds_name}:")
    header = f"    {'Model':<28} {'Clean':>7} {'BER=0.1':>8} {'BER=0.2':>8} {'BER=0.3':>8}"
    print(header)
    print("    " + "-" * 58)
    for m in PLOT_ORDER:
        s = STYLES[m]
        d = dd[m]
        print(f"    {s['label']:<28} {d['0.0']:>7.2f} {d['0.1']:>8.2f} {d['0.2']:>8.2f} {d['0.3']:>8.2f}")

with open(f'{OUT}/all_baselines_data.json', 'w') as f:
    json.dump(all_data, f, indent=2)

print(f"\n✅ All done!")
