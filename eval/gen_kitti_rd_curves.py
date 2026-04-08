#!/usr/bin/env python3
"""Generate publication-quality Rate-Distortion curves for KITTI image reconstruction.

Produces:
  1. PSNR vs CBR at multiple CSNR (AWGN) — Learned vs Random
  2. PSNR vs CBR at multiple BER (BSC) — Learned vs Random
  3. SSIM vs CBR at representative conditions
  4. Gap (Δ PSNR) summary table
"""

import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

RESULTS_FILE = 'eval/seed_results/kitti_image_results.json'
OUT_DIR = os.environ.get('ARTIFACT_DIR', '.')


def load_results():
    with open(RESULTS_FILE) as f:
        return json.load(f)


def plot_awgn_rd(results, out_path):
    """PSNR vs CBR at multiple CSNR values — Learned vs Random."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    fig.suptitle('PSNR vs Channel Bandwidth Ratio (CBR) — AWGN Channel\n'
                 'Learned Masking vs Random Masking on InStereo2K', 
                 fontsize=13, fontweight='bold', y=0.98)
    
    snr_list = ['1', '4', '7', '10', '13', '19']
    cbr_list = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0]
    
    colors_l = '#2563eb'  # blue
    colors_r = '#dc2626'  # red
    
    for idx, snr in enumerate(snr_list):
        ax = axes[idx // 3][idx % 3]
        
        learned_psnr = [results['awgn'][snr][str(c)]['learned']['psnr'] for c in cbr_list]
        random_psnr = [results['awgn'][snr][str(c)]['random']['psnr'] for c in cbr_list]
        
        ax.plot(cbr_list, learned_psnr, 'o-', color=colors_l, linewidth=2,
                markersize=6, label='Learned (Ours)', zorder=5)
        ax.plot(cbr_list, random_psnr, 's--', color=colors_r, linewidth=1.5,
                markersize=5, label='Random', zorder=4, alpha=0.8)
        
        # Shade the gap
        ax.fill_between(cbr_list, random_psnr, learned_psnr, 
                        alpha=0.12, color=colors_l, zorder=1)
        
        # Annotate max gap
        gaps = [l - r for l, r in zip(learned_psnr, random_psnr)]
        max_gap_idx = np.argmax(gaps)
        if gaps[max_gap_idx] > 0.5:
            ax.annotate(f'+{gaps[max_gap_idx]:.1f} dB', 
                       xy=(cbr_list[max_gap_idx], random_psnr[max_gap_idx]),
                       xytext=(0, -15), textcoords='offset points',
                       fontsize=8, fontweight='bold', color=colors_l,
                       ha='center')
        
        ax.set_title(f'CSNR = {snr} dB', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='-')
        ax.set_xlim(0.05, 1.05)
        ax.set_ylim(14, 26)
        
        if idx % 3 == 0:
            ax.set_ylabel('PSNR (dB)')
        if idx >= 3:
            ax.set_xlabel('CBR (ρ)')
        
        if idx == 0:
            ax.legend(loc='lower right', framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path)
    plt.close()
    print(f'  ✓ Saved {out_path}')


def plot_bsc_rd(results, out_path):
    """PSNR vs CBR at multiple BER values — Learned vs Random."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    fig.suptitle('PSNR vs Channel Bandwidth Ratio (CBR) — BSC Channel\n'
                 'Learned Masking vs Random Masking on InStereo2K', 
                 fontsize=13, fontweight='bold', y=0.98)
    
    ber_list = ['0.0', '0.05', '0.1', '0.15', '0.2', '0.3']
    ber_labels = ['BER = 0.00', 'BER = 0.05', 'BER = 0.10', 
                  'BER = 0.15', 'BER = 0.20', 'BER = 0.30']
    cbr_list = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0]
    
    colors_l = '#059669'  # green
    colors_r = '#ea580c'  # orange
    
    for idx, (ber, label) in enumerate(zip(ber_list, ber_labels)):
        ax = axes[idx // 3][idx % 3]
        
        learned_psnr = [results['bsc'][ber][str(c)]['learned']['psnr'] for c in cbr_list]
        random_psnr = [results['bsc'][ber][str(c)]['random']['psnr'] for c in cbr_list]
        
        ax.plot(cbr_list, learned_psnr, 'o-', color=colors_l, linewidth=2,
                markersize=6, label='Learned (Ours)', zorder=5)
        ax.plot(cbr_list, random_psnr, 's--', color=colors_r, linewidth=1.5,
                markersize=5, label='Random', zorder=4, alpha=0.8)
        
        ax.fill_between(cbr_list, random_psnr, learned_psnr, 
                        alpha=0.12, color=colors_l, zorder=1,
                        where=[l >= r for l, r in zip(learned_psnr, random_psnr)])
        
        gaps = [l - r for l, r in zip(learned_psnr, random_psnr)]
        max_gap_idx = np.argmax(gaps)
        if gaps[max_gap_idx] > 0.5:
            ax.annotate(f'+{gaps[max_gap_idx]:.1f} dB', 
                       xy=(cbr_list[max_gap_idx], random_psnr[max_gap_idx]),
                       xytext=(0, -15), textcoords='offset points',
                       fontsize=8, fontweight='bold', color=colors_l,
                       ha='center')
        
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='-')
        ax.set_xlim(0.05, 1.05)
        ax.set_ylim(14, 26)
        
        if idx % 3 == 0:
            ax.set_ylabel('PSNR (dB)')
        if idx >= 3:
            ax.set_xlabel('CBR (ρ)')
        
        if idx == 0:
            ax.legend(loc='lower right', framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path)
    plt.close()
    print(f'  ✓ Saved {out_path}')


def plot_gap_heatmap(results, out_path):
    """Heatmap of Δ PSNR (Learned - Random) across all conditions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('PSNR Advantage of Learned Masking Over Random (Δ dB)\n'
                 'SpikeAdapt-SC Image Codec on InStereo2K', 
                 fontsize=13, fontweight='bold', y=1.02)
    
    cbr_list = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75]
    
    # AWGN
    snr_list = ['1', '4', '7', '10', '13', '19']
    awgn_gaps = []
    for snr in snr_list:
        row = [results['awgn'][snr][str(c)]['gap_psnr'] for c in cbr_list]
        awgn_gaps.append(row)
    awgn_gaps = np.array(awgn_gaps)
    
    im1 = ax1.imshow(awgn_gaps, aspect='auto', cmap='RdYlGn', 
                     vmin=-1, vmax=2.5, origin='lower')
    ax1.set_xticks(range(len(cbr_list)))
    ax1.set_xticklabels([str(c) for c in cbr_list])
    ax1.set_yticks(range(len(snr_list)))
    ax1.set_yticklabels([f'{s} dB' for s in snr_list])
    ax1.set_xlabel('CBR (ρ)')
    ax1.set_ylabel('CSNR')
    ax1.set_title('AWGN Channel', fontweight='bold')
    
    for i in range(len(snr_list)):
        for j in range(len(cbr_list)):
            val = awgn_gaps[i, j]
            color = 'white' if abs(val) > 1.5 else 'black'
            ax1.text(j, i, f'{val:+.1f}', ha='center', va='center', 
                    fontsize=8, fontweight='bold', color=color)
    
    plt.colorbar(im1, ax=ax1, shrink=0.8, label='Δ PSNR (dB)')
    
    # BSC
    ber_list = ['0.0', '0.05', '0.1', '0.15', '0.2', '0.3']
    bsc_gaps = []
    for ber in ber_list:
        row = [results['bsc'][ber][str(c)]['gap_psnr'] for c in cbr_list]
        bsc_gaps.append(row)
    bsc_gaps = np.array(bsc_gaps)
    
    im2 = ax2.imshow(bsc_gaps, aspect='auto', cmap='RdYlGn', 
                     vmin=-1, vmax=2.5, origin='lower')
    ax2.set_xticks(range(len(cbr_list)))
    ax2.set_xticklabels([str(c) for c in cbr_list])
    ax2.set_yticks(range(len(ber_list)))
    ax2.set_yticklabels([str(b) for b in ber_list])
    ax2.set_xlabel('CBR (ρ)')
    ax2.set_ylabel('BER')
    ax2.set_title('BSC Channel', fontweight='bold')
    
    for i in range(len(ber_list)):
        for j in range(len(cbr_list)):
            val = bsc_gaps[i, j]
            color = 'white' if abs(val) > 1.5 else 'black'
            ax2.text(j, i, f'{val:+.1f}', ha='center', va='center', 
                    fontsize=8, fontweight='bold', color=color)
    
    plt.colorbar(im2, ax=ax2, shrink=0.8, label='Δ PSNR (dB)')
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f'  ✓ Saved {out_path}')


def plot_combined_rd(results, out_path):
    """Single figure: AWGN (SNR=7dB) and BSC (BER=0.10) side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle('Rate-Distortion: Learned vs Random Masking │ SpikeAdapt-SC Image Codec',
                 fontsize=12, fontweight='bold')
    
    cbr_list = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0]
    
    # AWGN SNR=7dB
    awgn_l = [results['awgn']['7'][str(c)]['learned']['psnr'] for c in cbr_list]
    awgn_r = [results['awgn']['7'][str(c)]['random']['psnr'] for c in cbr_list]
    
    ax1.plot(cbr_list, awgn_l, 'o-', color='#2563eb', linewidth=2.5, 
             markersize=7, label='Learned (Ours)')
    ax1.plot(cbr_list, awgn_r, 's--', color='#dc2626', linewidth=2, 
             markersize=6, label='Random', alpha=0.85)
    ax1.fill_between(cbr_list, awgn_r, awgn_l, alpha=0.1, color='#2563eb',
                     where=[l >= r for l, r in zip(awgn_l, awgn_r)])
    ax1.set_title('AWGN, CSNR = 7 dB', fontweight='bold')
    ax1.set_xlabel('CBR (ρ)')
    ax1.set_ylabel('PSNR (dB)')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.05, 1.05)
    
    # Annotate gap at CBR=0.375
    gap_375 = results['awgn']['7']['0.375']['gap_psnr']
    ax1.annotate(f'Δ = +{gap_375:.1f} dB', xy=(0.375, 17.97), xytext=(0.5, 16.5),
                arrowprops=dict(arrowstyle='->', color='#2563eb', lw=1.5),
                fontsize=9, fontweight='bold', color='#2563eb')
    
    # BSC BER=0.10
    bsc_l = [results['bsc']['0.1'][str(c)]['learned']['psnr'] for c in cbr_list]
    bsc_r = [results['bsc']['0.1'][str(c)]['random']['psnr'] for c in cbr_list]
    
    ax2.plot(cbr_list, bsc_l, 'o-', color='#059669', linewidth=2.5, 
             markersize=7, label='Learned (Ours)')
    ax2.plot(cbr_list, bsc_r, 's--', color='#ea580c', linewidth=2, 
             markersize=6, label='Random', alpha=0.85)
    ax2.fill_between(cbr_list, bsc_r, bsc_l, alpha=0.1, color='#059669',
                     where=[l >= r for l, r in zip(bsc_l, bsc_r)])
    ax2.set_title('BSC, BER = 0.10', fontweight='bold')
    ax2.set_xlabel('CBR (ρ)')
    ax2.set_ylabel('PSNR (dB)')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.05, 1.05)
    
    gap_375_bsc = results['bsc']['0.1']['0.375']['gap_psnr']
    ax2.annotate(f'Δ = +{gap_375_bsc:.1f} dB', xy=(0.375, 18.07), xytext=(0.5, 16.5),
                arrowprops=dict(arrowstyle='->', color='#059669', lw=1.5),
                fontsize=9, fontweight='bold', color='#059669')
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f'  ✓ Saved {out_path}')


def print_summary_table(results):
    """Print the key results as a table."""
    print('\n=== PSNR Gap Summary (Learned - Random) ===')
    print(f'{"Condition":<20} {"CBR=0.125":>10} {"CBR=0.25":>10} {"CBR=0.375":>10} '
          f'{"CBR=0.5":>10} {"CBR=0.625":>10} {"CBR=0.75":>10}')
    print('-' * 85)
    
    for snr in ['1', '7', '13', '19']:
        row = f'AWGN SNR={snr:>2}dB    '
        for cbr in ['0.125', '0.25', '0.375', '0.5', '0.625', '0.75']:
            gap = results['awgn'][snr][cbr]['gap_psnr']
            row += f'{gap:+10.2f}'
        print(row)
    
    print('-' * 85)
    for ber in ['0.0', '0.1', '0.2', '0.3']:
        row = f'BSC BER={ber:<5}      '
        for cbr in ['0.125', '0.25', '0.375', '0.5', '0.625', '0.75']:
            gap = results['bsc'][ber][cbr]['gap_psnr']
            row += f'{gap:+10.2f}'
        print(row)


def main():
    results = load_results()
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    plot_awgn_rd(results, os.path.join(OUT_DIR, 'kitti_awgn_rd.png'))
    plot_bsc_rd(results, os.path.join(OUT_DIR, 'kitti_bsc_rd.png'))
    plot_gap_heatmap(results, os.path.join(OUT_DIR, 'kitti_gap_heatmap.png'))
    plot_combined_rd(results, os.path.join(OUT_DIR, 'kitti_rd_combined.png'))
    print_summary_table(results)


if __name__ == '__main__':
    main()
