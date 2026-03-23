"""Generate publication-quality figures for SpikeAdapt-SC paper.

Uses all v2→v5 experimental data + ablation results.
Generates 5 new figures:
  Fig 5: ρ sweep Pareto curves (accuracy vs bandwidth for V2, V4-A, V5C)
  Fig 6: BER robustness comparison across versions (bar chart)
  Fig 7: Learned vs Random vs Uniform mask comparison
  Fig 8: Version evolution — clean + BER=0.30 + firing rate
  Fig 9: SNN-native ablation technique impact

Usage:
  python paper/generate_v5_figures.py
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# IEEE style matching existing figures
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
})

OUT_DIR = "./paper/figures/"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================================
# DATA (hardcoded from experiment results)
# ============================================================================

# ρ sweep results
RHO_VALUES = [0.25, 0.50, 0.625, 0.75, 0.875, 1.0]
BW_BITS = [14112, 28224, 35136, 42336, 49248, 56448]
BW_SAVINGS = [75.0, 50.0, 37.8, 25.0, 12.8, 0.0]

RHO_DATA = {
    'V2': {
        0.00: [87.80, 95.30, 96.00, 96.30, 96.20, 96.30],
        0.10: [80.40, 94.85, 95.75, 96.05, 96.15, 96.10],
        0.20: [63.25, 93.80, 94.95, 95.05, 95.45, 95.50],
        0.30: [29.90, 85.55, 91.05, 92.85, 93.25, 93.35],
    },
    'V4-A': {
        0.00: [91.55, 95.60, 96.15, 96.45, 96.30, 96.20],
        0.10: [87.45, 95.50, 96.10, 96.40, 96.50, 96.25],
        0.20: [78.60, 95.20, 95.90, 96.35, 96.50, 96.55],
        0.30: [51.15, 94.00, 94.75, 95.55, 95.65, 96.05],
    },
    'V5C': {
        0.00: [92.15, 95.60, 96.05, 96.55, 96.40, 96.35],
        0.10: [91.50, 95.65, 96.00, 96.25, 96.40, 96.35],
        0.20: [89.50, 95.15, 95.85, 96.00, 96.35, 96.25],
        0.30: [82.30, 94.55, 95.40, 95.45, 95.75, 95.40],
    },
}

# Cross-version BER results (at ρ=0.75)
VERSION_BER = {
    'V2 (IF+BN)':    {'clean': 96.35, 0.05: 96.30, 0.10: 96.30, 0.15: 95.85, 0.20: 95.65, 0.25: 93.55, 0.30: 92.90, 'fr': 0.277},
    'V4-A (LIF+BNTT)': {'clean': 96.45, 0.05: 96.50, 0.10: 96.40, 0.15: 96.45, 0.20: 96.35, 0.25: 96.00, 0.30: 95.85, 'fr': 0.266},
    'V5C (LIF+MPBN)': {'clean': 96.55, 0.05: 96.45, 0.10: 96.45, 0.15: 96.45, 0.20: 96.25, 0.25: 96.00, 0.30: 95.75, 'fr': 0.148},
}

# Mask comparison data (V4-A)
MASK_DATA_V4A = {
    (0.50, 0.00): {'learned': 95.60, 'random': 92.05, 'uniform': 92.10},
    (0.50, 0.15): {'learned': 95.40, 'random': 81.67, 'uniform': 78.70},
    (0.50, 0.30): {'learned': 93.35, 'random': 48.00, 'uniform': 44.50},
    (0.75, 0.00): {'learned': 96.45, 'random': 96.08, 'uniform': 95.70},
}

# SNN-native ablation data
ABLATION_DATA = {
    'BASELINE': {'clean': 96.60, 'ber30': 94.70},
    'A (Slope)':  {'clean': 96.70, 'ber30': 94.35},
    'B (LIF)':    {'clean': 96.50, 'ber30': 95.10},
    'C (BNTT)':   {'clean': 96.35, 'ber30': 95.50},
    'D (MemInit)': {'clean': 96.40, 'ber30': 94.10},
    'E (SpikeReg)': {'clean': 96.55, 'ber30': 94.75},
    'F (STDP)':   {'clean': 96.65, 'ber30': 94.65},
    'ALL':        {'clean': 96.50, 'ber30': 95.05},
}

# V5 complete results
V5_DATA = {
    'V5A (KD)':      {'clean': 95.75, 'ber30': 95.00, 'fr': 0.256, 'desc': 'Feature MSE KD'},
    'V5B (Gate)':     {'clean': 96.00, 'ber30': 95.50, 'fr': 0.268, 'desc': 'Channel Gate v2'},
    'V5C (MPBN)':     {'clean': 96.55, 'ber30': 95.75, 'fr': 0.148, 'desc': 'MPBN (no KD)'},
    'V5D (Attn)':     {'clean': 95.25, 'ber30': 93.20, 'fr': 0.266, 'desc': 'Spike Attn Refine'},
    'V5E (CG+Attn)':  {'clean': 94.15, 'ber30': 93.90, 'fr': 0.267, 'desc': 'CG + Attn combo'},
}

# Color palette
COLORS = {
    'V2': '#3498DB',    # blue
    'V4-A': '#E74C3C',  # red
    'V5C': '#2ECC71',   # green
    'random': '#95A5A6', # gray
    'uniform': '#F39C12', # orange
    'learned': '#8E44AD', # purple
}

# ============================================================================
# FIGURE 5: ρ Sweep — Accuracy vs Bandwidth Pareto Curves
# ============================================================================
def fig5_rho_sweep():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))
    fig.subplots_adjust(wspace=0.35)

    # Panel (a): Clean accuracy vs ρ
    ax = axes[0]
    for ver, color in [('V2', COLORS['V2']), ('V4-A', COLORS['V4-A']), ('V5C', COLORS['V5C'])]:
        ax.plot(RHO_VALUES, RHO_DATA[ver][0.00], 'o-', color=color, label=ver, markersize=5, zorder=3)
    ax.set_xlabel('Mask Rate ρ')
    ax.set_ylabel('Clean Accuracy (%)')
    ax.set_title('(a) Clean Accuracy vs ρ', fontweight='bold')
    ax.set_xlim(0.2, 1.05); ax.set_ylim(85, 97.5)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.axhline(y=96.35, color=COLORS['V2'], linestyle=':', alpha=0.4, linewidth=0.8)
    ax.text(0.52, 96.0, 'V2 clean baseline', fontsize=6.5, color=COLORS['V2'], alpha=0.6)

    # Add bandwidth savings axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(RHO_VALUES)
    ax2.set_xticklabels([f'{s:.0f}%' for s in BW_SAVINGS])
    ax2.set_xlabel('Bandwidth Saved', fontsize=7.5)
    ax2.tick_params(labelsize=6.5)

    # Panel (b): BER=0.30 accuracy vs ρ
    ax = axes[1]
    for ver, color in [('V2', COLORS['V2']), ('V4-A', COLORS['V4-A']), ('V5C', COLORS['V5C'])]:
        ax.plot(RHO_VALUES, RHO_DATA[ver][0.30], 's-', color=color, label=ver, markersize=5, zorder=3)

    # Key highlight: V5C at ρ=0.625 > V2 at ρ=1.0
    ax.annotate('V5C@ρ=0.625\nexceeds\nV2@ρ=1.0',
                xy=(0.625, 95.40), xytext=(0.35, 95.5),
                fontsize=6.5, fontweight='bold', color=COLORS['V5C'],
                arrowprops=dict(arrowstyle='->', color=COLORS['V5C'], lw=1.0),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=COLORS['V5C'], alpha=0.9))

    ax.axhline(y=93.35, color=COLORS['V2'], linestyle=':', alpha=0.4, linewidth=0.8)
    ax.text(0.77, 92.8, 'V2@ρ=1.0', fontsize=6.5, color=COLORS['V2'], alpha=0.6)

    ax.set_xlabel('Mask Rate ρ')
    ax.set_ylabel('Accuracy at BER=0.30 (%)')
    ax.set_title('(b) BER=0.30 Robustness vs ρ', fontweight='bold')
    ax.set_xlim(0.2, 1.05); ax.set_ylim(25, 97.5)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(RHO_VALUES)
    ax2.set_xticklabels([f'{s:.0f}%' for s in BW_SAVINGS])
    ax2.set_xlabel('Bandwidth Saved', fontsize=7.5)
    ax2.tick_params(labelsize=6.5)

    plt.savefig(os.path.join(OUT_DIR, 'fig5_rho_sweep.pdf'), format='pdf')
    plt.savefig(os.path.join(OUT_DIR, 'fig5_rho_sweep.png'), format='png')
    print("✓ Fig 5: ρ sweep Pareto curves")
    plt.close()


# ============================================================================
# FIGURE 6: BER Robustness Across Versions
# ============================================================================
def fig6_ber_comparison():
    fig, ax = plt.subplots(figsize=(7.16, 3.0))

    ber_vals = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    ber_labels = ['Clean', '0.05', '0.10', '0.15', '0.20', '0.25', '0.30']

    styles = {
        'V2 (IF+BN)':       {'color': COLORS['V2'],   'marker': 'o', 'ls': '--'},
        'V4-A (LIF+BNTT)':  {'color': COLORS['V4-A'], 'marker': 's', 'ls': '-'},
        'V5C (LIF+MPBN)':   {'color': COLORS['V5C'],  'marker': '^', 'ls': '-'},
    }

    for ver, style in styles.items():
        data = VERSION_BER[ver]
        accs = [data['clean']] + [data[b] for b in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]]
        ax.plot(range(len(ber_vals)), accs,
                marker=style['marker'], color=style['color'], linestyle=style['ls'],
                label=f"{ver} (FR={data['fr']:.3f})", markersize=6, zorder=3)

    # CNN-Uni reference
    cnn_accs = [92.50, None, None, 92.40, None, None, 67.25]
    cnn_plot_x = [0, 3, 6]
    cnn_plot_y = [92.50, 92.40, 67.25]
    ax.plot(cnn_plot_x, cnn_plot_y, 'x--', color='#7F8C8D', markersize=8, linewidth=1.0,
            label='CNN-Uni 8-bit', zorder=2)

    # Highlight region
    ax.fill_between(range(len(ber_vals)),
                    [VERSION_BER['V4-A (LIF+BNTT)'][b] if b != 'clean' else VERSION_BER['V4-A (LIF+BNTT)']['clean']
                     for b in ['clean', 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]],
                    [VERSION_BER['V2 (IF+BN)'][b] if b != 'clean' else VERSION_BER['V2 (IF+BN)']['clean']
                     for b in ['clean', 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]],
                    alpha=0.1, color=COLORS['V4-A'],
                    label='V4-A improvement over V2')

    ax.set_xticks(range(len(ber_vals)))
    ax.set_xticklabels(ber_labels)
    ax.set_xlabel('Bit Error Rate (BER)')
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title('BER Robustness: V2 vs V4-A vs V5C (ρ=0.75)', fontweight='bold')
    ax.set_ylim(64, 97.5)
    ax.legend(loc='lower left', framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Annotation: gap at BER=0.30
    ax.annotate(f'+2.95pp', xy=(6, 95.85), xytext=(5.3, 94.5),
                fontsize=8, fontweight='bold', color=COLORS['V4-A'],
                arrowprops=dict(arrowstyle='->', color=COLORS['V4-A'], lw=1.2))

    plt.savefig(os.path.join(OUT_DIR, 'fig6_ber_comparison.pdf'), format='pdf')
    plt.savefig(os.path.join(OUT_DIR, 'fig6_ber_comparison.png'), format='png')
    print("✓ Fig 6: BER robustness comparison")
    plt.close()


# ============================================================================
# FIGURE 7: Learned vs Random vs Uniform Mask
# ============================================================================
def fig7_mask_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))
    fig.subplots_adjust(wspace=0.35)

    # Panel (a): ρ=0.50 across BER
    ax = axes[0]
    ber_vals = [0.00, 0.15, 0.30]
    x = np.arange(len(ber_vals))
    width = 0.25

    learned = [MASK_DATA_V4A[(0.50, b)]['learned'] for b in ber_vals]
    random_m = [MASK_DATA_V4A[(0.50, b)]['random'] for b in ber_vals]
    uniform_m = [MASK_DATA_V4A[(0.50, b)]['uniform'] for b in ber_vals]

    bars1 = ax.bar(x - width, learned, width, label='Learned', color=COLORS['learned'],
                   edgecolor='black', linewidth=0.5, zorder=3)
    bars2 = ax.bar(x, random_m, width, label='Random', color=COLORS['random'],
                   edgecolor='black', linewidth=0.5, zorder=3)
    bars3 = ax.bar(x + width, uniform_m, width, label='Uniform', color=COLORS['uniform'],
                   edgecolor='black', linewidth=0.5, zorder=3)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 60:
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                        f'{h:.1f}', ha='center', va='bottom', fontsize=6)

    # Delta annotations
    for i, ber in enumerate(ber_vals):
        delta = learned[i] - random_m[i]
        ax.text(x[i], max(learned[i], random_m[i]) + 3,
                f'+{delta:.1f}pp', ha='center', fontsize=6.5, fontweight='bold',
                color=COLORS['learned'])

    ax.set_xticks(x)
    ax.set_xticklabels([f'BER={b:.2f}' for b in ber_vals])
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('(a) V4-A at ρ=0.50', fontweight='bold')
    ax.set_ylim(35, 105)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.2, axis='y', linewidth=0.5)

    # Panel (b): accuracy gap (learned - random) heatmap-style
    ax = axes[1]

    # Construct full comparison data
    rho_vals = [0.50, 0.75]
    ber_all = [0.00, 0.15, 0.30]
    gaps = []
    labels_txt = []

    for rho in rho_vals:
        row = []
        for ber in ber_all:
            if (rho, ber) in MASK_DATA_V4A:
                gap = MASK_DATA_V4A[(rho, ber)]['learned'] - MASK_DATA_V4A[(rho, ber)]['random']
                row.append(gap)
            else:
                row.append(0)
        gaps.append(row)

    gaps_arr = np.array(gaps)

    im = ax.imshow(gaps_arr, cmap='RdYlGn', aspect='auto', vmin=0, vmax=50)
    ax.set_xticks(range(len(ber_all)))
    ax.set_xticklabels([f'BER={b:.2f}' for b in ber_all])
    ax.set_yticks(range(len(rho_vals)))
    ax.set_yticklabels([f'ρ={r:.2f}' for r in rho_vals])
    ax.set_title('(b) Δ Accuracy: Learned − Random (pp)', fontweight='bold')

    for i in range(len(rho_vals)):
        for j in range(len(ber_all)):
            val = gaps_arr[i, j]
            color = 'white' if val > 20 else 'black'
            ax.text(j, i, f'+{val:.1f}', ha='center', va='center',
                    fontsize=9, fontweight='bold', color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label='Accuracy Gap (pp)')

    plt.savefig(os.path.join(OUT_DIR, 'fig7_mask_comparison.pdf'), format='pdf')
    plt.savefig(os.path.join(OUT_DIR, 'fig7_mask_comparison.png'), format='png')
    print("✓ Fig 7: Mask comparison")
    plt.close()


# ============================================================================
# FIGURE 8: Version Evolution — Multi-metric
# ============================================================================
def fig8_version_evolution():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.0))
    fig.subplots_adjust(wspace=0.4)

    # Panel (a): Clean + BER=0.30 grouped bars
    versions = ['V2', 'V4-A', 'V5C']
    clean_accs = [96.35, 96.45, 96.55]
    ber30_accs = [92.90, 95.85, 95.75]
    colors_bar = [COLORS['V2'], COLORS['V4-A'], COLORS['V5C']]

    x = np.arange(len(versions))
    width = 0.35

    bars1 = ax1.bar(x - width/2, clean_accs, width, label='Clean',
                    color=colors_bar, edgecolor='black', linewidth=0.5, alpha=0.9)
    bars2 = ax1.bar(x + width/2, ber30_accs, width, label='BER=0.30',
                    color=colors_bar, edgecolor='black', linewidth=0.5, alpha=0.5,
                    hatch='///')

    # Value labels
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7)

    ax1.set_xticks(x)
    ax1.set_xticklabels(versions, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('(a) Clean & BER=0.30 Accuracy', fontweight='bold')
    ax1.set_ylim(91, 97.5)
    ax1.legend(loc='lower right', framealpha=0.9)
    ax1.grid(True, alpha=0.2, axis='y', linewidth=0.5)

    # Improvement arrow
    ax1.annotate('+2.95pp', xy=(1.175, 95.85), xytext=(1.5, 94.0),
                fontsize=8, fontweight='bold', color=COLORS['V4-A'],
                arrowprops=dict(arrowstyle='->', color=COLORS['V4-A'], lw=1.2))

    # Panel (b): Firing rate comparison (energy proxy)
    frs = [0.277, 0.266, 0.148]
    synops_ratio = [fr / 0.277 for fr in frs]  # relative to V2

    bars = ax2.bar(x, frs, 0.5, color=colors_bar, edgecolor='black', linewidth=0.5)
    for i, bar in enumerate(bars):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{frs[i]:.3f}\n({synops_ratio[i]*100:.0f}%)',
                ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax2.set_xticks(x)
    ax2.set_xticklabels(versions, fontweight='bold')
    ax2.set_ylabel('Average Firing Rate')
    ax2.set_title('(b) Energy Efficiency (Firing Rate)', fontweight='bold')
    ax2.set_ylim(0, 0.35)
    ax2.grid(True, alpha=0.2, axis='y', linewidth=0.5)

    # Energy savings annotation
    ax2.annotate('47% fewer\nspikes → energy', xy=(2, 0.148), xytext=(1.3, 0.05),
                fontsize=7, fontweight='bold', color=COLORS['V5C'],
                arrowprops=dict(arrowstyle='->', color=COLORS['V5C'], lw=1.2),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor=COLORS['V5C'], alpha=0.9))

    plt.savefig(os.path.join(OUT_DIR, 'fig8_version_evolution.pdf'), format='pdf')
    plt.savefig(os.path.join(OUT_DIR, 'fig8_version_evolution.png'), format='png')
    print("✓ Fig 8: Version evolution")
    plt.close()


# ============================================================================
# FIGURE 9: SNN-Native Ablation Technique Impact
# ============================================================================
def fig9_ablation():
    fig, ax = plt.subplots(figsize=(7.16, 3.0))

    techniques = list(ABLATION_DATA.keys())
    clean_vals = [ABLATION_DATA[t]['clean'] for t in techniques]
    ber30_vals = [ABLATION_DATA[t]['ber30'] for t in techniques]

    x = np.arange(len(techniques))
    width = 0.35

    # Color coding: green for best, orange for good, red for weak
    tech_colors = []
    for t in techniques:
        d = ABLATION_DATA[t]
        if d['ber30'] >= 95.0:
            tech_colors.append('#2ECC71')  # green - excellent BER
        elif d['ber30'] >= 94.5:
            tech_colors.append('#F1C40F')  # yellow - good
        else:
            tech_colors.append('#E74C3C')  # red - weaker

    ax.bar(x - width/2, clean_vals, width, label='Clean',
           color=tech_colors, edgecolor='black', linewidth=0.5, alpha=0.9)
    ax.bar(x + width/2, ber30_vals, width, label='BER=0.30',
           color=tech_colors, edgecolor='black', linewidth=0.5, alpha=0.5, hatch='///')

    # Reference lines
    ax.axhline(y=96.35, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    ax.text(7.5, 96.15, 'V2 clean', fontsize=6, color='gray')
    ax.axhline(y=92.90, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.text(7.5, 92.65, 'V2 BER=0.30', fontsize=6, color='gray')

    ax.set_xticks(x)
    ax.set_xticklabels(techniques, rotation=30, ha='right', fontsize=7.5)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('SNN-Native Technique Ablation (AID-30, ρ=0.75)', fontweight='bold')
    ax.set_ylim(93, 97.5)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.2, axis='y', linewidth=0.5)

    # Highlight best BER technique
    ax.annotate('Best BER\nrobustness', xy=(3, 95.5), xytext=(4.5, 96.8),
                fontsize=7, fontweight='bold', color='#2ECC71',
                arrowprops=dict(arrowstyle='->', color='#2ECC71', lw=1.0),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='#2ECC71', alpha=0.9))

    plt.savefig(os.path.join(OUT_DIR, 'fig9_ablation.pdf'), format='pdf')
    plt.savefig(os.path.join(OUT_DIR, 'fig9_ablation.png'), format='png')
    print("✓ Fig 9: SNN-native ablation")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("Generating publication-quality figures...")
    print(f"Output: {OUT_DIR}")
    print()

    fig5_rho_sweep()
    fig6_ber_comparison()
    fig7_mask_comparison()
    fig8_version_evolution()
    fig9_ablation()

    print(f"\n✅ All 5 figures saved to {OUT_DIR}")
    print("   PDF versions ready for LaTeX \\includegraphics")
    print("   PNG versions ready for README.md")
