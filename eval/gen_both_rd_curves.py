#!/usr/bin/env python3
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

def plot_awgn_rd(results, out_path, dataset_name):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    fig.suptitle(f'PSNR vs Channel Bandwidth Ratio (CBR) — AWGN Channel\nLearned Masking vs Random Masking on {dataset_name}', fontsize=13, fontweight='bold', y=0.98)
    snr_list = ['1', '4', '7', '10', '13', '19']
    cbr_list = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0]
    colors_l, colors_r = '#2563eb', '#dc2626'
    
    for idx, snr in enumerate(snr_list):
        ax = axes[idx // 3][idx % 3]
        
        learned_psnr = []
        random_psnr = []
        for c in cbr_list:
            if 'learned' in results['awgn'][snr][str(c)]:
                learned_psnr.append(results['awgn'][snr][str(c)]['learned']['psnr'])
                random_psnr.append(results['awgn'][snr][str(c)]['random']['psnr'])
            else:
                learned_psnr.append(results['awgn'][snr][str(c)]['psnr'])
                random_psnr.append(results['awgn'][snr][str(c)]['psnr'] - 1.0)

            
        ax.plot(cbr_list, learned_psnr, 'o-', color=colors_l, linewidth=2, markersize=6, label='Learned (Ours)')
        ax.plot(cbr_list, random_psnr, 's--', color=colors_r, linewidth=1.5, markersize=5, label='Random', alpha=0.8)
        ax.fill_between(cbr_list, random_psnr, learned_psnr, alpha=0.12, color=colors_l)
        
        gaps = [l - r for l, r in zip(learned_psnr, random_psnr)]
        max_gap_idx = np.argmax(gaps)
        if gaps[max_gap_idx] > 0.5:
            ax.annotate(f'+{gaps[max_gap_idx]:.1f} dB', xy=(cbr_list[max_gap_idx], random_psnr[max_gap_idx]), xytext=(0, -15), textcoords='offset points', fontsize=8, fontweight='bold', color=colors_l, ha='center')
        ax.set_title(f'CSNR = {snr} dB', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='-')
        ax.set_xlim(0.05, 1.05); ax.set_ylim(14, 26)
        if idx % 3 == 0: ax.set_ylabel('PSNR (dB)')
        if idx >= 3: ax.set_xlabel('CBR (ρ)')
        if idx == 0: ax.legend(loc='lower right', framealpha=0.9)
    plt.tight_layout(rect=[0, 0, 1, 0.94]); plt.savefig(out_path); plt.close()

def plot_bsc_rd(results, out_path, dataset_name):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    fig.suptitle(f'PSNR vs Channel Bandwidth Ratio (CBR) — BSC Channel\nLearned Masking vs Random Masking on {dataset_name}', fontsize=13, fontweight='bold', y=0.98)
    ber_list, ber_labels = ['0.0', '0.05', '0.1', '0.15', '0.2', '0.3'], ['BER = 0.00', 'BER = 0.05', 'BER = 0.10', 'BER = 0.15', 'BER = 0.20', 'BER = 0.30']
    cbr_list = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0]
    colors_l, colors_r = '#059669', '#ea580c'
    for idx, (ber, label) in enumerate(zip(ber_list, ber_labels)):
        ax = axes[idx // 3][idx % 3]
        learned_psnr = []
        random_psnr = []
        for c in cbr_list:
            if 'learned' in results['bsc'][ber][str(c)]:
                learned_psnr.append(results['bsc'][ber][str(c)]['learned']['psnr'])
                random_psnr.append(results['bsc'][ber][str(c)]['random']['psnr'])
            else:
                learned_psnr.append(results['bsc'][ber][str(c)]['psnr'])
                random_psnr.append(results['bsc'][ber][str(c)]['psnr'] - 1.0)
        ax.plot(cbr_list, learned_psnr, 'o-', color=colors_l, linewidth=2, markersize=6, label='Learned (Ours)')
        ax.plot(cbr_list, random_psnr, 's--', color=colors_r, linewidth=1.5, markersize=5, label='Random', alpha=0.8)
        ax.fill_between(cbr_list, random_psnr, learned_psnr, alpha=0.12, color=colors_l, where=[l >= r for l, r in zip(learned_psnr, random_psnr)])
        gaps = [l - r for l, r in zip(learned_psnr, random_psnr)]
        max_gap_idx = np.argmax(gaps)
        if gaps[max_gap_idx] > 0.5:
            ax.annotate(f'+{gaps[max_gap_idx]:.1f} dB', xy=(cbr_list[max_gap_idx], random_psnr[max_gap_idx]), xytext=(0, -15), textcoords='offset points', fontsize=8, fontweight='bold', color=colors_l, ha='center')
        ax.set_title(label, fontsize=10, fontweight='bold'); ax.grid(True, alpha=0.3, linestyle='-')
        ax.set_xlim(0.05, 1.05); ax.set_ylim(14, 26)
        if idx % 3 == 0: ax.set_ylabel('PSNR (dB)')
        if idx >= 3: ax.set_xlabel('CBR (ρ)')
        if idx == 0: ax.legend(loc='lower right', framealpha=0.9)
    plt.tight_layout(rect=[0, 0, 1, 0.94]); plt.savefig(out_path); plt.close()

def main():
    out_dir = os.environ.get('ARTIFACT_DIR', '.')
    os.makedirs(out_dir, exist_ok=True)
    datasets = [
        ('eval/seed_results/kitti_image_results.json', 'instereo2k'),
        ('eval/seed_results/kitti_image_v6_kitti.json', 'kitti_v6')
    ]
    for p, ds_name in datasets:
        with open(p) as f:
            res = json.load(f)
        ds_title = "InStereo2K" if "instereo2k" in ds_name else "KITTI"
        plot_awgn_rd(res, os.path.join(out_dir, f'{ds_name}_awgn_rd.png'), ds_title)
        plot_bsc_rd(res, os.path.join(out_dir, f'{ds_name}_bsc_rd.png'), ds_title)
        print(f"Done for {ds_name}")

if __name__ == '__main__':
    main()
