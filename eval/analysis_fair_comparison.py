#!/usr/bin/env python3
"""Analysis: Why SpikeAdapt-SC V6 reconstruction PSNR is lower than DHF-JSCC,
and how to build a fair comparison.

THE CORE PROBLEM:
  DHF-JSCC transmits CONTINUOUS floats through AWGN → exploits full channel capacity
  SpikeAdapt-SC transmits TERNARY {-1,0,1} spikes → hard-decision → ~1.58 bits/symbol max

At CR=0.065:
  - DHF-JSCC effective rate: CR × log2(1+SNR) bits/pixel (grows with SNR!)
    At CSNR=10dB: 0.065 × 3.46 = 0.225 bits/pixel
  - V6 effective rate: CR × log2(3) = 0.065 × 1.58 = 0.103 bits/pixel (FIXED, SNR-independent)
  → DHF-JSCC has 2.2× more information capacity at CSNR=10dB

But the gap is LARGER than 2.2×. Why?
  1. DHF-JSCC's hyperprior captures image structure (learned prior)
  2. Continuous symbols can be soft-decoded (no quantization loss)
  3. DHF-JSCC is jointly optimized end-to-end for reconstruction
  4. Our V6 was trained at target_cbr=0.5, evaluated at 0.065 via override (suboptimal)

THREE APPROACHES FOR FAIR COMPARISON:
  (A) Capacity-matched: plot PSNR vs "effective bits/pixel"
  (B) Channel-matched: force DHF-JSCC onto binary/ternary channel (quantize + BSC)
  (C) Energy-normalized: PSNR per joule

Output: printed analysis + eval/figures/fair_comparison_analysis.pdf
"""

import json, os, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('eval/figures', exist_ok=True)

# =============================================================================
# 1. Load our V6 results
# =============================================================================
def load_v6():
    results = {}
    for ds_file, ds_name in [('kitti_image_v6_instereo2k', 'InStereo2K'),
                              ('kitti_image_v6_kitti', 'KITTI')]:
        path = f'eval/seed_results/{ds_file}.json'
        if os.path.exists(path):
            with open(path) as f:
                results[ds_name] = json.load(f)
    return results

# =============================================================================
# 2. DHF-JSCC paper numbers (read from figures)
# =============================================================================
# From Fig.10 (KITTI, CR=0.065) and Fig.15 (InStereo2K, CR=0.065)
DHF_JSCC_KITTI = {  # CSNR(dB) → PSNR(dB), read from Fig.10
    0: 25.5, 2.5: 27.3, 5: 28.3, 7.5: 29.0, 10: 29.5
}
DHF_JSCC_INSTEREO = {  # CSNR(dB) → PSNR(dB), read from Fig.15
    0: 27.5, 2.5: 29.5, 5: 30.3, 7.5: 31.2, 10: 32.0
}
# Deep JSCC [16] baselines from same figures
DEEP_JSCC_KITTI = {0: 24.2, 2.5: 25.8, 5: 27.0, 7.5: 27.8, 10: 28.2}
DEEP_JSCC_INSTEREO = {0: 26.0, 2.5: 27.7, 5: 28.7, 7.5: 29.5, 10: 30.2}

# =============================================================================
# 3. Information-theoretic analysis
# =============================================================================
def awgn_capacity(snr_db):
    """Shannon capacity: C = 0.5 × log2(1 + SNR) bits/channel use."""
    snr_lin = 10 ** (snr_db / 10.0)
    return 0.5 * math.log2(1 + snr_lin)

def binary_capacity(ber):
    """BSC capacity: C = 1 - H(ber) bits/channel use."""
    if ber <= 0 or ber >= 1:
        return 1.0 if ber <= 0 else 0.0
    return 1 + ber * math.log2(ber) + (1-ber) * math.log2(1-ber)

def ternary_capacity():
    """Ternary noiseless: log2(3) ≈ 1.585 bits/symbol."""
    return math.log2(3)

def bpsk_ber(snr_db):
    """BER for BPSK over AWGN."""
    snr_lin = 10 ** (snr_db / 10.0)
    return 0.5 * math.erfc(math.sqrt(snr_lin))

def effective_bits_continuous(cr, snr_db):
    """Effective bits/pixel for continuous JSCC at given CR and SNR."""
    return cr * awgn_capacity(snr_db) * 2  # ×2 for complex channel

def effective_bits_ternary(cbr, snr_db=None):
    """Effective bits/pixel for ternary spikes at given CBR.
    SNR-independent for hard-decision: fixed at log2(3) bits/symbol."""
    return cbr * ternary_capacity()

def effective_bits_binary(cbr, snr_db=None):
    """Effective bits/pixel for binary spikes: 1 bit/symbol."""
    return cbr * 1.0


# =============================================================================
# 4. MAIN ANALYSIS
# =============================================================================
def main():
    v6 = load_v6()

    print("=" * 80)
    print("FAIR COMPARISON ANALYSIS: SpikeAdapt-SC V6 vs DHF-JSCC")
    print("=" * 80)

    # --- Information capacity comparison ---
    print("\n1. INFORMATION CAPACITY AT CR=0.065:")
    print("-" * 60)
    for snr in [0, 2.5, 5, 7.5, 10]:
        cap_cont = effective_bits_continuous(0.065, snr)
        cap_tern = effective_bits_ternary(0.065)
        cap_bin  = effective_bits_binary(0.065)
        ratio = cap_cont / cap_tern if cap_tern > 0 else float('inf')
        print(f"  CSNR={snr:>4.1f} dB: Continuous={cap_cont:.4f} bpp, "
              f"Ternary={cap_tern:.4f} bpp, Binary={cap_bin:.4f} bpp, "
              f"Ratio={ratio:.2f}x")

    # --- PSNR comparison ---
    print("\n2. PSNR GAP AT CR=0.065 (InStereo2K):")
    print("-" * 60)
    print(f"  {'CSNR':>6} {'DHF-JSCC':>10} {'V6@CBR=0.125':>13} {'V6@CBR=0.25':>12} {'Gap(0.125)':>11}")

    if 'InStereo2K' in v6:
        for snr_str in sorted(v6['InStereo2K']['awgn'].keys(), key=float):
            snr = float(snr_str)
            v6_125 = v6['InStereo2K']['awgn'][snr_str].get('0.125', {}).get('psnr', 0)
            v6_25  = v6['InStereo2K']['awgn'][snr_str].get('0.25', {}).get('psnr', 0)
            # Find closest DHF-JSCC CSNR
            dhf_csnrs = sorted(DHF_JSCC_INSTEREO.keys())
            closest = min(dhf_csnrs, key=lambda c: abs(c - snr))
            dhf_psnr = DHF_JSCC_INSTEREO[closest]
            gap = dhf_psnr - v6_125
            print(f"  {snr:>5.0f}dB {dhf_psnr:>9.1f}dB {v6_125:>12.1f}dB {v6_25:>11.1f}dB {gap:>10.1f}dB")

    # --- Capacity-matched comparison ---
    print("\n3. CAPACITY-MATCHED COMPARISON:")
    print("   (Find V6 CBR that gives same effective bits/pixel as DHF-JSCC)")
    print("-" * 60)
    for snr in [0, 5, 10]:
        dhf_ebpp = effective_bits_continuous(0.065, snr)
        matched_cbr_ternary = dhf_ebpp / ternary_capacity()
        matched_cbr_binary = dhf_ebpp / 1.0
        print(f"  CSNR={snr:>2d}dB: DHF@CR=0.065 → {dhf_ebpp:.3f} bpp")
        print(f"    → V6 needs CBR={matched_cbr_ternary:.3f} (ternary) or "
              f"CBR={matched_cbr_binary:.3f} (binary) to match")

        if 'InStereo2K' in v6:
            # Interpolate V6 PSNR at matched CBR
            cbrs = sorted([float(c) for c in v6['InStereo2K']['bsc']['0.0'].keys()])
            psnrs = [v6['InStereo2K']['bsc']['0.0'][str(c)]['psnr'] for c in cbrs]
            v6_matched = np.interp(matched_cbr_ternary, cbrs, psnrs)
            dhf_psnr = DHF_JSCC_INSTEREO.get(snr, DHF_JSCC_INSTEREO[min(DHF_JSCC_INSTEREO.keys(), key=lambda c: abs(c-snr))])
            print(f"    → V6 PSNR ≈ {v6_matched:.1f}dB vs DHF-JSCC {dhf_psnr:.1f}dB "
                  f"(gap: {dhf_psnr - v6_matched:.1f}dB)")

    # --- WHY quantization hurts ---
    print("\n4. WHY QUANTIZATION HURTS (AND WHY YOU MUST QUANTIZE):")
    print("-" * 60)
    print("""
  The channel model determines everything:

  DHF-JSCC (continuous AWGN):
    encoder → z ∈ ℝ^n → power_norm(z) → z + N(0,σ²) → decoder
    Each symbol carries INFINITE precision → log2(1+SNR) bits/use

  SpikeAdapt-SC (ternary spikes + hard decision):
    encoder → s ∈ {-1,0,1}^n → channel → hard_decision → decoder
    Each symbol carries log2(3) = 1.58 bits/use MAX (SNR-independent!)

  The quantization from continuous → ternary is LOSSY but NECESSARY because:
    1. SNN neurons natively produce spikes (discrete events)
    2. Binary/ternary encoding enables energy-efficient neuromorphic hardware
    3. Hard-decision decoding is robust (BSC equivalent, no error propagation)
    4. Sparse spike encoding → 37× energy savings

  The cost: at CR=0.065, you lose ~10-12 dB PSNR vs continuous JSCC.
  The benefit: 37× energy savings, robust to channel mismatch, neuromorphic-ready.
    """)

    # --- Approach B: How to faithfully adapt DHF-JSCC ---
    print("5. HOW TO BUILD A FAIR 'DHF-JSCC ON BINARY CHANNEL' BASELINE:")
    print("-" * 60)
    print("""
  To show the gap is NOT architectural but channel-fundamental:

  Option (B1): DHF-JSCC-Binary
    - Take DHF-JSCC architecture (hyperprior encoder + entropy mask + decoder)
    - REPLACE continuous channel with: sign_quantize(z) → BSC → decode
    - Train end-to-end with STE through sign quantization
    - This gives DHF-JSCC the same 1 bit/symbol constraint as SNN
    - Compare at matched CBR

  Option (B2): Continuous SpikeAdapt (ablation)
    - Take V6 architecture but SKIP spike threshold
    - Send raw membrane potentials through AWGN (continuous)
    - Power-normalize before channel (like JSCC)
    - This gives SpikeAdapt the same continuous advantage as DHF-JSCC
    - Compare at matched CR

  Option (B3): Capacity-normalized plot (easiest, no retraining)
    - x-axis: effective bits per pixel = CBR × capacity_per_symbol
    - y-axis: PSNR
    - Plot both V6 and DHF-JSCC on same axes
    - If curves overlap → gap is purely channel capacity
    - If V6 is lower → architecture/training also contributes

  RECOMMENDATION: Do (B3) first (easy, uses existing data), then (B1) if needed.
    """)

    # === PLOT: Capacity-normalized comparison ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax_idx, (ds_name, dhf_data, deep_data) in enumerate([
        ('KITTI', DHF_JSCC_KITTI, DEEP_JSCC_KITTI),
        ('InStereo2K', DHF_JSCC_INSTEREO, DEEP_JSCC_INSTEREO),
    ]):
        ax = axes[ax_idx]

        if ds_name not in v6:
            continue

        # V6 BSC BER=0 (noiseless binary): plot effective bpp vs PSNR
        cbrs = sorted([float(c) for c in v6[ds_name]['bsc']['0.0'].keys()])
        v6_psnrs = [v6[ds_name]['bsc']['0.0'][str(c)]['psnr'] for c in cbrs]
        v6_ebpp_tern = [effective_bits_ternary(c) for c in cbrs]
        v6_ebpp_bin = [effective_bits_binary(c) for c in cbrs]

        ax.plot(v6_ebpp_tern, v6_psnrs, 'o-', color='#2563eb', linewidth=2.5,
                markersize=7, label='V6 (ternary spikes, BER=0)', zorder=5)

        # V6 BSC BER=0.1
        v6_psnrs_ber01 = [v6[ds_name]['bsc']['0.1'][str(c)]['psnr'] for c in cbrs]
        ax.plot(v6_ebpp_tern, v6_psnrs_ber01, 's--', color='#60a5fa', linewidth=1.5,
                markersize=5, label='V6 (ternary, BER=0.1)')

        # DHF-JSCC at CR=0.065 across CSNR (each CSNR = different effective bpp)
        dhf_csnrs = sorted(dhf_data.keys())
        dhf_ebpp = [effective_bits_continuous(0.065, snr) for snr in dhf_csnrs]
        dhf_psnrs = [dhf_data[snr] for snr in dhf_csnrs]
        ax.plot(dhf_ebpp, dhf_psnrs, '^-', color='#dc2626', linewidth=2.5,
                markersize=8, label='DHF-JSCC (continuous, CR=0.065)', zorder=4)

        # Deep JSCC baseline
        deep_ebpp = [effective_bits_continuous(0.065, snr) for snr in sorted(deep_data.keys())]
        deep_psnrs = [deep_data[snr] for snr in sorted(deep_data.keys())]
        ax.plot(deep_ebpp, deep_psnrs, 'x--', color='#16a34a', linewidth=1.5,
                markersize=7, label='Deep JSCC (continuous, CR=0.065)')

        ax.set_xlabel('Effective bits per pixel', fontsize=12)
        ax.set_ylabel('PSNR (dB)', fontsize=12)
        ax.set_title(f'{ds_name}: Capacity-Normalized Comparison', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(max(v6_ebpp_tern), max(dhf_ebpp)) * 1.1)

        # Annotate the capacity gap
        ax.annotate('DHF-JSCC gets more\nbpp at same CR\n(continuous channel)',
                    xy=(dhf_ebpp[-1], dhf_psnrs[-1]),
                    xytext=(dhf_ebpp[-1]+0.05, dhf_psnrs[-1]-3),
                    fontsize=8, color='#dc2626',
                    arrowprops=dict(arrowstyle='->', color='#dc2626'))

    plt.tight_layout()
    fig.savefig('eval/figures/fair_comparison_capacity_normalized.pdf', dpi=300, bbox_inches='tight')
    fig.savefig('eval/figures/fair_comparison_capacity_normalized.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: eval/figures/fair_comparison_capacity_normalized.pdf")

    # === PLOT: Raw CBR comparison (showing the gap honestly) ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax_idx, (ds_name, dhf_data) in enumerate([
        ('KITTI', DHF_JSCC_KITTI),
        ('InStereo2K', DHF_JSCC_INSTEREO),
    ]):
        ax = axes[ax_idx]
        if ds_name not in v6:
            continue

        # V6 at different CBR, CSNR=1dB (AWGN)
        cbrs = sorted([float(c) for c in v6[ds_name]['awgn']['1'].keys()])
        v6_psnrs = [v6[ds_name]['awgn']['1'][str(c)]['psnr'] for c in cbrs]
        ax.plot(cbrs, v6_psnrs, 'o-', color='#2563eb', linewidth=2.5,
                markersize=7, label='V6 (ternary, CSNR=1dB)')

        # V6 at CSNR=10dB
        v6_psnrs_10 = [v6[ds_name]['awgn']['10'][str(c)]['psnr'] for c in cbrs]
        ax.plot(cbrs, v6_psnrs_10, 's-', color='#60a5fa', linewidth=2,
                markersize=6, label='V6 (ternary, CSNR=10dB)')

        # DHF-JSCC at CR=0.065 (single point per CSNR)
        for snr, psnr_val in dhf_data.items():
            marker = '^' if snr in [0, 10] else 'v'
            ax.plot(0.065, psnr_val, marker, color='#dc2626', markersize=10,
                    label=f'DHF-JSCC (CR=0.065, CSNR={snr}dB)' if snr in [0, 10] else None,
                    zorder=10)

        # Arrow showing the gap
        v6_at_065 = np.interp(0.065, cbrs, v6_psnrs)
        dhf_at_065 = dhf_data[0]
        ax.annotate('', xy=(0.065, dhf_at_065), xytext=(0.065, v6_at_065),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        ax.text(0.085, (dhf_at_065 + v6_at_065)/2,
                f'{dhf_at_065 - v6_at_065:.0f} dB gap\n(channel capacity)',
                fontsize=9, color='red', fontweight='bold')

        ax.set_xlabel('Channel Bandwidth Ratio (CBR / CR)', fontsize=12)
        ax.set_ylabel('PSNR (dB)', fontsize=12)
        ax.set_title(f'{ds_name}: Raw CBR Comparison', fontsize=13, fontweight='bold')
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1.05)

    plt.tight_layout()
    fig.savefig('eval/figures/fair_comparison_raw_cbr.pdf', dpi=300, bbox_inches='tight')
    fig.savefig('eval/figures/fair_comparison_raw_cbr.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: eval/figures/fair_comparison_raw_cbr.pdf")


if __name__ == '__main__':
    main()
