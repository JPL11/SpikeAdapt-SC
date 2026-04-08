#!/usr/bin/env python3
"""Generate publication-quality RD curves and gap heatmaps for Globecom paper."""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9.5,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2.0,
    'lines.markersize': 7,
})

OUT = '/home/jpli/.gemini/antigravity/brain/220715a1-1974-4d45-862f-590a0f95a261'

with open('eval/seed_results/kitti_image_v2_instereo2k.json') as f:
    IS = json.load(f)
with open('eval/seed_results/kitti_image_v2_kitti.json') as f:
    KT = json.load(f)

cbrs = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0]
cbrs_mask = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75]

# Colors
C_LEARN = '#1a56db'
C_RAND  = '#dc2626'
C_KITTI = '#059669'
C_KRAND = '#ea580c'

# =========================================================================
# Figure 1: RD Curves — InStereo2K + KITTI side by side (2 panels)
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

for ax, data, title, cl, cr in [
    (ax1, IS, 'InStereo2K', C_LEARN, C_RAND),
    (ax2, KT, 'KITTI Stereo 2015 (cross-dataset)', C_KITTI, C_KRAND),
]:
    for snr, ls, alpha, label_sfx in [
        ('1', '-', 1.0, ' (1 dB)'),
        ('7', '-', 1.0, ' (7 dB)'),
        ('19', '--', 0.6, ' (19 dB)'),
    ]:
        lvals = [data['awgn'][snr][str(c)]['learned']['psnr'] for c in cbrs]
        rvals = [data['awgn'][snr][str(c)]['random']['psnr'] for c in cbrs]
        if snr == '7':
            ax.plot(cbrs, lvals, 'o'+ls, color=cl, alpha=alpha, label='Learned'+label_sfx, zorder=5)
            ax.plot(cbrs, rvals, 's'+ls, color=cr, alpha=alpha, label='Random'+label_sfx, zorder=4)
            ax.fill_between(cbrs, rvals, lvals, alpha=0.08, color=cl,
                          where=[l >= r for l, r in zip(lvals, rvals)])
        elif snr == '1':
            ax.plot(cbrs, lvals, 'v'+ls, color=cl, alpha=0.5, markersize=5, label='Learned'+label_sfx, zorder=3)
            ax.plot(cbrs, rvals, '^'+ls, color=cr, alpha=0.4, markersize=5, label='Random'+label_sfx, zorder=2)
        else:
            ax.plot(cbrs, lvals, 'D'+ls, color=cl, alpha=alpha, markersize=5, label='Learned'+label_sfx, zorder=3)
            ax.plot(cbrs, rvals, 'x'+ls, color=cr, alpha=alpha, markersize=5, label='Random'+label_sfx, zorder=2)

    # Annotate peak gap at SNR=7
    snr = '7'
    lvals7 = [data['awgn'][snr][str(c)]['learned']['psnr'] for c in cbrs]
    rvals7 = [data['awgn'][snr][str(c)]['random']['psnr'] for c in cbrs]
    gaps = [l-r for l,r in zip(lvals7, rvals7)]
    max_idx = max(range(len(gaps)), key=lambda i: gaps[i])
    if gaps[max_idx] > 0.3:
        ax.annotate(f'+{gaps[max_idx]:.1f} dB',
                   xy=(cbrs[max_idx], (lvals7[max_idx]+rvals7[max_idx])/2),
                   fontsize=10, fontweight='bold', color=cl, ha='center',
                   bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=cl, alpha=0.8))

    ax.set_xlabel('Channel Bandwidth Ratio (ρ)')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='lower right', ncol=2, framealpha=0.9, edgecolor='#ccc')
    ax.grid(True, alpha=0.2, linestyle='-')
    ax.set_xlim(0.05, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('SpikeAdapt-SC V2: Rate-Distortion Performance (AWGN Channel)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'paper_rd_curves_awgn.png'), dpi=300)
print(f'✓ Saved paper_rd_curves_awgn.png')
plt.close()

# =========================================================================
# Figure 2: Gap Heatmaps — AWGN + BSC (2x2 grid)
# =========================================================================
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle('Masking Advantage: Learned − Random (Δ PSNR in dB)', 
             fontsize=14, fontweight='bold', y=1.01)

snr_list = ['1', '4', '7', '10', '13', '19']
ber_list = ['0.0', '0.05', '0.1', '0.15', '0.2', '0.3']

configs = [
    (axes[0,0], IS, 'awgn', snr_list, 'InStereo2K — AWGN', [f'{s} dB' for s in snr_list], 'CSNR'),
    (axes[0,1], KT, 'awgn', snr_list, 'KITTI — AWGN', [f'{s} dB' for s in snr_list], 'CSNR'),
    (axes[1,0], IS, 'bsc', ber_list, 'InStereo2K — BSC', [f'{b}' for b in ber_list], 'BER'),
    (axes[1,1], KT, 'bsc', ber_list, 'KITTI — BSC', [f'{b}' for b in ber_list], 'BER'),
]

for ax, data, ch, params, title, ylabels, ylabel in configs:
    g = np.array([[data[ch][p][str(c)]['gap_psnr'] for c in cbrs_mask] for p in params])
    vmax = max(2.5, g.max() + 0.2)
    im = ax.imshow(g, aspect='auto', cmap='RdYlGn', vmin=-0.5, vmax=vmax, origin='lower')
    ax.set_xticks(range(len(cbrs_mask)))
    ax.set_xticklabels([str(c) for c in cbrs_mask])
    ax.set_yticks(range(len(params)))
    ax.set_yticklabels(ylabels)
    ax.set_xlabel('CBR (ρ)')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold', fontsize=11)
    for i in range(len(params)):
        for j in range(len(cbrs_mask)):
            v = g[i, j]
            color = 'white' if abs(v) > 1.5 else 'black'
            ax.text(j, i, f'{v:+.1f}', ha='center', va='center', 
                   fontsize=8, fontweight='bold', color=color)
    plt.colorbar(im, ax=ax, shrink=0.8, label='Δ dB')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'paper_gap_heatmaps.png'), dpi=300)
print(f'✓ Saved paper_gap_heatmaps.png')
plt.close()

# =========================================================================
# Figure 3: BSC RD Curves — InStereo2K (important for binary spike story)
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

for ax, data, title, cl, cr in [
    (ax1, IS, 'InStereo2K', C_LEARN, C_RAND),
    (ax2, KT, 'KITTI (cross-dataset)', C_KITTI, C_KRAND),
]:
    for ber, ls, alpha, label_sfx in [
        ('0.0', '-', 1.0, ' (BER=0)'),
        ('0.1', '-', 1.0, ' (BER=0.1)'),
        ('0.3', '--', 0.6, ' (BER=0.3)'),
    ]:
        lvals = [data['bsc'][ber][str(c)]['learned']['psnr'] for c in cbrs]
        rvals = [data['bsc'][ber][str(c)]['random']['psnr'] for c in cbrs]
        if ber == '0.1':
            ax.plot(cbrs, lvals, 'o'+ls, color=cl, alpha=alpha, label='Learned'+label_sfx, zorder=5)
            ax.plot(cbrs, rvals, 's'+ls, color=cr, alpha=alpha, label='Random'+label_sfx, zorder=4)
            ax.fill_between(cbrs, rvals, lvals, alpha=0.08, color=cl,
                          where=[l >= r for l, r in zip(lvals, rvals)])
        elif ber == '0.0':
            ax.plot(cbrs, lvals, 'v'+ls, color=cl, alpha=0.5, markersize=5, label='Learned'+label_sfx, zorder=3)
        else:
            ax.plot(cbrs, lvals, 'D'+ls, color=cl, alpha=alpha, markersize=5, label='Learned'+label_sfx, zorder=3)
            ax.plot(cbrs, rvals, 'x'+ls, color=cr, alpha=alpha, markersize=5, label='Random'+label_sfx, zorder=2)

    ax.set_xlabel('Channel Bandwidth Ratio (ρ)')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='#ccc')
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0.05, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('SpikeAdapt-SC V2: Rate-Distortion Performance (BSC Channel)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'paper_rd_curves_bsc.png'), dpi=300)
print(f'✓ Saved paper_rd_curves_bsc.png')
plt.close()

# =========================================================================
# Figure 4: Single summary figure — gap vs CBR at multiple SNR/BER
# =========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

colors_snr = ['#e11d48', '#f59e0b', '#10b981', '#3b82f6', '#6366f1', '#8b5cf6']
for i, snr in enumerate(snr_list):
    gaps = [IS['awgn'][snr][str(c)]['gap_psnr'] for c in cbrs_mask]
    ax1.plot(cbrs_mask, gaps, 'o-', color=colors_snr[i], label=f'{snr} dB', markersize=5)
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('CBR (ρ)')
ax1.set_ylabel('Δ PSNR (dB)')
ax1.set_title('InStereo2K — AWGN Gap vs CBR', fontweight='bold')
ax1.legend(title='CSNR', ncol=2, framealpha=0.9)
ax1.grid(True, alpha=0.2)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

colors_ber = ['#059669', '#0ea5e9', '#8b5cf6', '#f59e0b', '#ef4444', '#991b1b']
for i, ber in enumerate(ber_list):
    gaps = [IS['bsc'][ber][str(c)]['gap_psnr'] for c in cbrs_mask]
    ax2.plot(cbrs_mask, gaps, 's-', color=colors_ber[i], label=f'{ber}', markersize=5)
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('CBR (ρ)')
ax2.set_ylabel('Δ PSNR (dB)')
ax2.set_title('InStereo2K — BSC Gap vs CBR', fontweight='bold')
ax2.legend(title='BER', ncol=2, framealpha=0.9)
ax2.grid(True, alpha=0.2)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

fig.suptitle('Content-Adaptive Masking Advantage Across Channel Conditions', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'paper_gap_vs_cbr.png'), dpi=300)
print(f'✓ Saved paper_gap_vs_cbr.png')
plt.close()

print('\nDone! All 4 publication figures generated.')
