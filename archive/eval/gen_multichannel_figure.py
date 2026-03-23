"""Generate updated multichannel figure showing ALL baselines under BSC/AWGN/Rayleigh.

Dual-panel: AID and RESISC45, with V5C-NA, CNN-Uni, CNN-NonUni, MLP-FC
each showing BSC/AWGN/Rayleigh curves to demonstrate channel convergence.
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
    'legend.fontsize': 6,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.4,
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
})

# Load all multichannel data
with open('eval/multichannel_results_v2.json') as f:
    snn_mc = json.load(f)
with open('eval/cnn_multichannel.json') as f:
    cnn_mc = json.load(f)
with open('eval/mlp_multichannel.json') as f:
    mlp_mc = json.load(f)

bers = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
ber_keys_snn = ['0.0', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3']

# Method configs: (name, color, data_source, ds_key_map)
methods = [
    ('V5C-NA', '#1B5E20', 2.5),
    ('CNN-Uni', '#E53935', 1.5),
    ('CNN-NonUni', '#FF9800', 1.5),
    ('MLP-FC', '#9C27B0', 1.5),
]

channel_styles = {
    'BSC': ('-', 'o'),
    'AWGN': ('--', 's'),
    'Rayleigh': ('-.', '^'),
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))

for ax, ds_label, ds_key in [(ax1, 'AID', 'aid'), (ax2, 'RESISC45', 'resisc45')]:
    for method_name, color, lw in methods:
        for ch_name, (ls, mk) in channel_styles.items():
            accs = []
            for ber in bers:
                if method_name == 'V5C-NA':
                    # SNN data
                    if ber == 0:
                        accs.append(snn_mc[ds_key]['bsc']['results']['0.0'])
                    else:
                        bk = str(ber) if str(ber) in snn_mc[ds_key]['matched_ber'] else f"{ber:.2f}"
                        # Use the proper key format
                        for k in snn_mc[ds_key]['matched_ber']:
                            if abs(float(k) - ber) < 0.001:
                                bk = k; break
                        mb = snn_mc[ds_key]['matched_ber'].get(bk, None)
                        if mb:
                            if ch_name == 'BSC': accs.append(mb['bsc_acc'])
                            elif ch_name == 'AWGN': accs.append(mb['awgn_acc'])
                            else: accs.append(mb['ray_acc'])
                        else:
                            accs.append(snn_mc[ds_key]['bsc']['results'].get(str(ber), 0))
                elif method_name in ('CNN-Uni', 'CNN-NonUni'):
                    mb = cnn_mc[ds_label][method_name]['matched_ber'][str(ber)]
                    if ch_name == 'BSC': accs.append(mb['bsc'])
                    elif ch_name == 'AWGN': accs.append(mb['awgn'])
                    else: accs.append(mb['rayleigh'])
                elif method_name == 'MLP-FC':
                    mb = mlp_mc[ds_label][str(ber)]
                    if ch_name == 'BSC': accs.append(mb['bsc'])
                    elif ch_name == 'AWGN': accs.append(mb['awgn'])
                    else: accs.append(mb['rayleigh'])

            # Only label BSC for each method (others are visually grouped)
            label = f'{method_name} ({ch_name})' if ch_name == 'BSC' else None
            alpha = 1.0 if ch_name == 'BSC' else 0.5
            ax.plot(bers, accs, ls, marker=mk, color=color, lw=lw,
                    ms=3 if ch_name != 'BSC' else 4,
                    alpha=alpha, label=label, zorder=5 if method_name == 'V5C-NA' else 3)

    panel = chr(97 + [ax1, ax2].index(ax))
    split = '50/50' if ds_key == 'aid' else '20/80'
    ax.set_xlabel('Equivalent BER')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'({panel}) {ds_label} ({split})', fontweight='bold')
    ax.set_xlim(-0.01, 0.31)
    ax.set_ylim(10 if ds_key == 'resisc45' else 25, 100)

    # Add channel style legend
    handles, labels = ax.get_legend_handles_labels()
    # Add manual entries for channel styles
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], ls='-', color='gray', lw=1, label='BSC'))
    handles.append(Line2D([0], [0], ls='--', color='gray', lw=1, label='AWGN'))
    handles.append(Line2D([0], [0], ls='-.', color='gray', lw=1, label='Rayleigh'))
    ax.legend(handles=handles, loc='lower left', ncol=2, framealpha=0.9,
              edgecolor='#ccc', columnspacing=0.5, handletextpad=0.3)
    ax.grid(True, alpha=0.25)

plt.tight_layout(w_pad=1.5)
plt.savefig('paper/figures/fig7_unified_ber.pdf')
plt.savefig('paper/figures/fig7_unified_ber.png')
print("✅ Updated multichannel figure saved (fig7_unified_ber)")
plt.close()
