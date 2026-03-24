"""Compute SynOps, energy ratios, and latency estimates for SpikeAdapt-SC.

Uses the Horowitz energy model:
  - 32-bit MAC: 4.6 pJ
  - SynOp (accumulate-only): 0.9 pJ
  
SynOps = firing_rate × fan_in × output_elements
MACs = fan_in × output_elements (full precision equivalent)

Usage:
  python eval/compute_synops.py
"""

import os, sys, json
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))

# Architecture parameters
T = 8  # timesteps
H, W = 14, 14  # spatial grid
C_in = 1024  # input channels from ResNet50
C1 = 256  # encoder L1 output
C2 = 36   # encoder L2 output (bottleneck)
C_dec1 = 256  # decoder L1
C_dec2 = 512  # decoder L2
C_dec3 = 1024  # decoder L3

# Energy constants (Horowitz 2014)
E_MAC = 4.6   # pJ per 32-bit floating-point MAC
E_SYNOP = 0.9  # pJ per spike accumulate operation

def compute_layer_ops(fan_in, out_h, out_w, out_c, firing_rate):
    """Compute SynOps and equivalent MACs for one layer per timestep."""
    output_elements = out_h * out_w * out_c
    macs = fan_in * output_elements  # full precision
    synops = firing_rate * fan_in * output_elements  # spike-driven
    return synops, macs

def compute_model_synops(version, firing_rate, rho=0.75):
    """Compute total SynOps for one forward pass (T timesteps)."""
    
    # Encoder L1: Conv2d(C_in, C1, 3×3) → LIF → spike
    enc_l1_synops, enc_l1_macs = compute_layer_ops(
        fan_in=C_in * 9, out_h=H, out_w=W, out_c=C1, firing_rate=firing_rate
    )
    
    # Encoder L2: Conv2d(C1, C2, 3×3) → LIF → spike
    enc_l2_synops, enc_l2_macs = compute_layer_ops(
        fan_in=C1 * 9, out_h=H, out_w=W, out_c=C2, firing_rate=firing_rate
    )
    
    # After masking: only ρ fraction of spatial locations transmitted
    # Channel: BSC bit flips (no computation)
    
    # Decoder L1: Conv2d(C2, C_dec1, 3×3)
    dec_l1_synops, dec_l1_macs = compute_layer_ops(
        fan_in=C2 * 9, out_h=H, out_w=W, out_c=C_dec1, firing_rate=firing_rate * rho
    )
    
    # Decoder L2: Conv2d(C_dec1, C_dec2, 3×3) → LIF → spike
    dec_l2_synops, dec_l2_macs = compute_layer_ops(
        fan_in=C_dec1 * 9, out_h=H, out_w=W, out_c=C_dec2, firing_rate=firing_rate * rho
    )
    
    # Decoder L3: Conv2d(C_dec2, C_dec3, 3×3)
    dec_l3_synops, dec_l3_macs = compute_layer_ops(
        fan_in=C_dec2 * 9, out_h=H, out_w=W, out_c=C_dec3, firing_rate=firing_rate * rho
    )
    
    # Scorer: Conv2d(C2, 32, 3×3) + Conv2d(32, 1, 1×1) — runs once (averaged over T)
    scorer_synops, scorer_macs = compute_layer_ops(
        fan_in=C2 * 9, out_h=H, out_w=W, out_c=32, firing_rate=firing_rate
    )
    scorer_synops2, scorer_macs2 = compute_layer_ops(
        fan_in=32, out_h=H, out_w=W, out_c=1, firing_rate=0.5  # ReLU output
    )
    
    # Total per timestep
    enc_synops = enc_l1_synops + enc_l2_synops
    enc_macs = enc_l1_macs + enc_l2_macs
    dec_synops = dec_l1_synops + dec_l2_synops + dec_l3_synops
    dec_macs = dec_l1_macs + dec_l2_macs + dec_l3_macs
    
    # Total over T timesteps
    total_synops = T * (enc_synops + dec_synops) + scorer_synops + scorer_synops2
    total_macs = T * (enc_macs + dec_macs) + scorer_macs + scorer_macs2
    
    # Energy
    energy_synops = total_synops * E_SYNOP  # pJ
    energy_macs = total_macs * E_MAC  # pJ
    energy_ratio = energy_macs / energy_synops if energy_synops > 0 else float('inf')
    energy_savings = (1 - energy_synops / energy_macs) * 100
    
    # Payload
    payload_bits = int(rho * H * W) * C2 * T
    payload_full = H * W * C2 * T
    bw_savings = (1 - payload_bits / payload_full) * 100
    
    return {
        'version': version,
        'firing_rate': firing_rate,
        'rho': rho,
        'enc_synops': int(enc_synops * T),
        'dec_synops': int(dec_synops * T),
        'scorer_synops': int(scorer_synops + scorer_synops2),
        'total_synops': int(total_synops),
        'total_macs': int(total_macs),
        'energy_synops_pJ': energy_synops,
        'energy_macs_pJ': energy_macs,
        'energy_ratio': energy_ratio,
        'energy_savings_pct': energy_savings,
        'payload_bits': payload_bits,
        'bw_savings_pct': bw_savings,
    }


if __name__ == "__main__":
    print("=" * 80)
    print("SpikeAdapt-SC — SynOps & Energy Analysis")
    print("=" * 80)
    
    # Model configurations with measured firing rates
    configs = [
        ("V2 (IF baseline)", 0.266, 0.75),
        ("V4-A (LIF+BNTT)", 0.268, 0.75),
        ("V5C (MPBN)", 0.167, 0.75),
        ("V5C (MPBN, ρ=0.50)", 0.167, 0.50),
        ("V5C (MPBN, ρ=0.625)", 0.167, 0.625),
    ]
    
    results = []
    print(f"\n{'Version':<25} {'FR':>5} {'ρ':>5} {'Enc SynOps':>12} {'Dec SynOps':>12} "
          f"{'Total SynOps':>12} {'MACs':>12} {'Energy ×':>10} {'BW Save':>8}")
    print("-" * 110)
    
    for name, fr, rho in configs:
        r = compute_model_synops(name, fr, rho)
        results.append(r)
        print(f"{name:<25} {fr:>5.3f} {rho:>5.3f} {r['enc_synops']:>12,d} {r['dec_synops']:>12,d} "
              f"{r['total_synops']:>12,d} {r['total_macs']:>12,d} {r['energy_ratio']:>9.1f}× "
              f"{r['bw_savings_pct']:>6.1f}%")
    
    # LaTeX table
    print(f"\n{'='*80}")
    print("LaTeX Table (for main.tex)")
    print(f"{'='*80}")
    print(r"\begin{tabular}{l c c c c c}")
    print(r"\toprule")
    print(r"\textbf{Version} & \textbf{FR} & $\rho$ & \textbf{SynOps} & \textbf{Energy} $\times$ & \textbf{BW Save} \\")
    print(r"\midrule")
    for r in results:
        synops_M = r['total_synops'] / 1e6
        name_tex = r['version'].replace('ρ', r'$\rho$')
        print(f"{name_tex} & {r['firing_rate']:.3f} & {r['rho']:.2f} & "
              f"{synops_M:.1f}M & {r['energy_ratio']:.1f}$\\times$ & {r['bw_savings_pct']:.0f}\\% \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    
    # Save results
    with open("eval/synops_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ SynOps results saved to eval/synops_results.json")
