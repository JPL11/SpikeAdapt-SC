#!/usr/bin/env python3
"""Analyze V2 results and generate comparison plots."""
import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.environ.get('ARTIFACT_DIR', '/home/jpli/.gemini/antigravity/brain/220715a1-1974-4d45-862f-590a0f95a261')

# Load all results
with open('eval/seed_results/kitti_image_v2_instereo2k.json') as f:
    v2_is = json.load(f)
with open('eval/seed_results/kitti_image_v2_kitti.json') as f:
    v2_kt = json.load(f)
with open('eval/seed_results/kitti_image_results.json') as f:
    v1_is = json.load(f)

cbrs = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0]

print("=" * 90)
print("V2 InStereo2K vs V1 InStereo2K — AWGN SNR=7dB")
print(f"{'CBR':>6} | {'V2 Learned':>12} {'V2 Random':>12} {'V2 Gap':>8} | {'V1 Learned':>12} {'V1 Random':>12} {'V1 Gap':>8}")
print("-" * 90)
for c in cbrs:
    v2l = v2_is['awgn']['7'][str(c)]['learned']['psnr']
    v2r = v2_is['awgn']['7'][str(c)]['random']['psnr']
    v2g = v2l - v2r
    v1l = v1_is['awgn']['7'][str(c)]['learned']['psnr']
    v1r = v1_is['awgn']['7'][str(c)]['random']['psnr']
    v1g = v1l - v1r
    print(f"{c:>6.3f} | {v2l:>10.2f}dB {v2r:>10.2f}dB {v2g:>+7.2f}dB | {v1l:>10.2f}dB {v1r:>10.2f}dB {v1g:>+7.2f}dB")

print()
print("=" * 90)
print("V2 KITTI (cross-dataset) — AWGN SNR=7dB")
print(f"{'CBR':>6} | {'Learned':>12} {'Random':>12} {'Gap':>8}")
print("-" * 50)
for c in cbrs:
    l = v2_kt['awgn']['7'][str(c)]['learned']['psnr']
    r = v2_kt['awgn']['7'][str(c)]['random']['psnr']
    g = l - r
    print(f"{c:>6.3f} | {l:>10.2f}dB {r:>10.2f}dB {g:>+7.2f}dB")

print()
print("=" * 90)
print("V2 InStereo2K Full-bandwidth (CBR=1.0) across noise")
for snr in ['1', '7', '13', '19']:
    v = v2_is['awgn'][snr]['1.0']['learned']['psnr']
    print(f"  AWGN SNR={snr:>2}dB: {v:.2f}dB")
for ber in ['0.0', '0.1', '0.2', '0.3']:
    v = v2_is['bsc'][ber]['1.0']['learned']['psnr']
    print(f"  BSC BER={ber}: {v:.2f}dB")

# Average masking gap
print()
print("=" * 90)
print("Average masking gap (Learned - Random)")
for name, data in [("V2 InStereo2K", v2_is), ("V2 KITTI", v2_kt), ("V1 InStereo2K", v1_is)]:
    gaps = []
    for snr in ['7']:
        for c in [0.25, 0.375, 0.5, 0.625, 0.75]:
            g = data['awgn'][snr][str(c)]['gap_psnr']
            gaps.append(g)
    print(f"  {name}: avg gap = {np.mean(gaps):+.2f} dB (range: {min(gaps):+.2f} to {max(gaps):+.2f})")

# --- Generate comparison plot ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('SpikeAdapt-SC V2: Rate-Distortion Comparison', fontsize=13, fontweight='bold')

plt.rcParams.update({'font.size': 10})

# Panel 1: V2 InStereo2K
ax = axes[0]
snr = '7'
v2l = [v2_is['awgn'][snr][str(c)]['learned']['psnr'] for c in cbrs]
v2r = [v2_is['awgn'][snr][str(c)]['random']['psnr'] for c in cbrs]
ax.plot(cbrs, v2l, 'o-', color='#2563eb', linewidth=2.5, markersize=7, label='Learned', zorder=5)
ax.plot(cbrs, v2r, 's--', color='#dc2626', linewidth=2, markersize=6, label='Random', alpha=0.8, zorder=4)
ax.fill_between(cbrs, v2r, v2l, alpha=0.12, color='#2563eb',
                where=[l >= r for l, r in zip(v2l, v2r)])
ax.set_title('V2 InStereo2K (AWGN 7dB)', fontweight='bold')
ax.set_xlabel('CBR (ρ)')
ax.set_ylabel('PSNR (dB)')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0.05, 1.05)

# Annotate peak gap
gaps = [l-r for l,r in zip(v2l, v2r)]
max_idx = np.argmax(gaps)
ax.annotate(f'+{gaps[max_idx]:.1f} dB', xy=(cbrs[max_idx], v2r[max_idx]),
           xytext=(0, -18), textcoords='offset points', fontsize=9,
           fontweight='bold', color='#2563eb', ha='center')

# Panel 2: V2 KITTI (cross-dataset)
ax = axes[1]
ktl = [v2_kt['awgn'][snr][str(c)]['learned']['psnr'] for c in cbrs]
ktr = [v2_kt['awgn'][snr][str(c)]['random']['psnr'] for c in cbrs]
ax.plot(cbrs, ktl, 'o-', color='#059669', linewidth=2.5, markersize=7, label='Learned', zorder=5)
ax.plot(cbrs, ktr, 's--', color='#ea580c', linewidth=2, markersize=6, label='Random', alpha=0.8, zorder=4)
ax.fill_between(cbrs, ktr, ktl, alpha=0.12, color='#059669',
                where=[l >= r for l, r in zip(ktl, ktr)])
ax.set_title('V2 KITTI (cross-dataset, AWGN 7dB)', fontweight='bold')
ax.set_xlabel('CBR (ρ)')
ax.set_ylabel('PSNR (dB)')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0.05, 1.05)

gaps_kt = [l-r for l,r in zip(ktl, ktr)]
max_idx_kt = np.argmax(gaps_kt)
ax.annotate(f'+{gaps_kt[max_idx_kt]:.1f} dB', xy=(cbrs[max_idx_kt], ktr[max_idx_kt]),
           xytext=(0, -18), textcoords='offset points', fontsize=9,
           fontweight='bold', color='#059669', ha='center')

# Panel 3: V1 vs V2 comparison
ax = axes[2]
v1l = [v1_is['awgn'][snr][str(c)]['learned']['psnr'] for c in cbrs]
ax.plot(cbrs, v2l, 'o-', color='#2563eb', linewidth=2.5, markersize=7, label='V2 Learned', zorder=5)
ax.plot(cbrs, v1l, 'D--', color='#7c3aed', linewidth=2, markersize=6, label='V1 Learned', alpha=0.8, zorder=4)
ax.set_title('V1 vs V2 Learned (InStereo2K, AWGN 7dB)', fontweight='bold')
ax.set_xlabel('CBR (ρ)')
ax.set_ylabel('PSNR (dB)')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0.05, 1.05)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'v2_comparison.png'), dpi=200, bbox_inches='tight')
print(f"\n✓ Saved {os.path.join(OUT_DIR, 'v2_comparison.png')}")

# --- Generate gap heatmap for V2 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('V2 PSNR Advantage: Learned over Random (Δ dB)', fontsize=13, fontweight='bold', y=1.02)
cbrs_mask = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75]

# InStereo2K
snr_list = ['1', '4', '7', '10', '13', '19']
g = np.array([[v2_is['awgn'][s][str(c)]['gap_psnr'] for c in cbrs_mask] for s in snr_list])
im1 = ax1.imshow(g, aspect='auto', cmap='RdYlGn', vmin=-0.5, vmax=2.5, origin='lower')
ax1.set_xticks(range(len(cbrs_mask))); ax1.set_xticklabels([str(c) for c in cbrs_mask])
ax1.set_yticks(range(len(snr_list))); ax1.set_yticklabels([f'{s} dB' for s in snr_list])
ax1.set_xlabel('CBR (ρ)'); ax1.set_ylabel('CSNR'); ax1.set_title('InStereo2K', fontweight='bold')
for i in range(len(snr_list)):
    for j in range(len(cbrs_mask)):
        ax1.text(j, i, f'{g[i,j]:+.1f}', ha='center', va='center', fontsize=8, fontweight='bold')
plt.colorbar(im1, ax=ax1, shrink=0.8, label='Δ PSNR (dB)')

# KITTI
g2 = np.array([[v2_kt['awgn'][s][str(c)]['gap_psnr'] for c in cbrs_mask] for s in snr_list])
im2 = ax2.imshow(g2, aspect='auto', cmap='RdYlGn', vmin=-0.5, vmax=2.5, origin='lower')
ax2.set_xticks(range(len(cbrs_mask))); ax2.set_xticklabels([str(c) for c in cbrs_mask])
ax2.set_yticks(range(len(snr_list))); ax2.set_yticklabels([f'{s} dB' for s in snr_list])
ax2.set_xlabel('CBR (ρ)'); ax2.set_ylabel('CSNR'); ax2.set_title('KITTI (cross-dataset)', fontweight='bold')
for i in range(len(snr_list)):
    for j in range(len(cbrs_mask)):
        ax2.text(j, i, f'{g2[i,j]:+.1f}', ha='center', va='center', fontsize=8, fontweight='bold')
plt.colorbar(im2, ax=ax2, shrink=0.8, label='Δ PSNR (dB)')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'v2_gap_heatmap.png'), dpi=200, bbox_inches='tight')
print(f"✓ Saved {os.path.join(OUT_DIR, 'v2_gap_heatmap.png')}")
plt.close('all')
