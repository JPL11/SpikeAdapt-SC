#!/usr/bin/env python3
"""Learned non-spiking codec + soft-decode LDPC separation baseline (reviewer
fix; user-selected). Answers: is SpikeAdapt-SC's benefit from SPIKING, or just
from learned binary coding? The strongest separation baseline uses the LEARNED
CNN-1bit codec as the source coder (not JPEG) and protects its bits with a
soft-decode (sum-product) LDPC channel code, at MATCHED channel uses.

Pipeline: ResNet-50 feats -> CNN-1bit encoder -> 7056 source bits
          -> LDPC(rate r) -> BSC(p) -> sum-product BP -> CNN-1bit decoder -> class.
The CNN-1bit clean accuracy (val-selected, leakage-free, from
valproto_cnn1bit.json) is the source-decode-success accuracy; below the LDPC
threshold the receiver sees clean bits, above it the frame is lost. Analytic
all-or-nothing frame model (same analytic_accuracy as the JPEG baseline), so
this is CPU-only and needs no GPU.

Channel uses = 7056 / r. Matched to SpikeAdapt-SC rho=0.75 (42336 uses) at
rate r=1/6; lighter protection (r=1/2, 1/3) traces the separation tradeoff.

Usage: python eval/eval_learned_ldpc.py
Output: eval/seed_results/learned_ldpc_results.json
"""

import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eval.ldpc import make_regular_ldpc, characterize          # noqa: E402
from eval.eval_jpeg_ldpc import analytic_accuracy              # noqa: E402

OUT = os.path.join(ROOT, 'eval/seed_results/learned_ldpc_results.json')
CNN1BIT = os.path.join(ROOT, 'eval/seed_results/valproto_cnn1bit.json')
SOURCE_BITS = 7056                       # CNN-1bit payload (14x14x36, rho=1.0)
N_CLASSES = {'aid': 30, 'resisc45': 45}
BER_GRID = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
# regular LDPC codes: (name, n, dv, dc) -> rate = 1 - dv/dc
CODES = [('r12', 648, 3, 6), ('r13', 648, 4, 6), ('r16', 648, 5, 6)]


def hbin(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def shannon_cliff_ber(rate):
    """BSC crossover p at which capacity 1-H(p) == rate (ideal-separation
    cliff): the best possible operating point for ANY rate-`rate` code."""
    lo, hi = 0.0, 0.5
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if 1 - hbin(mid) > rate:      # capacity above rate -> can push p higher
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    cnn = json.load(open(CNN1BIT))
    # clean (BER=0) CNN-1bit accuracy, averaged over the 10 val-selected seeds
    clean = {ds: float(np.mean([cnn[ds][s]['0.0'] for s in cnn[ds]]))
             for ds in cnn}
    print('CNN-1bit clean acc (val-selected, 10-seed mean):', clean)
    # CNN-1bit JOINT (no FEC) robustness vs BER, 10-seed mean -- the key
    # "is graceful degradation from spiking or just binary?" reference.
    bers_cnn = sorted({k for s in cnn['aid'].values() for k in s
                       if k not in ('selected',)}, key=float)
    cnn_joint = {ds: {b: round(float(np.mean([cnn[ds][s][b] for s in cnn[ds]
                                              if b in cnn[ds][s]])), 2)
                      for b in bers_cnn} for ds in cnn}

    # characterize each LDPC rate once (all-zero-cw sum-product MC)
    ldpc = {}
    for name, n, dv, dc in CODES:
        H = make_regular_ldpc(n, dv, dc, seed=0)
        k = n - (n * dv // dc)
        rate = k / n
        stats = characterize(H, BER_GRID, n_frames=2000, method='sumprod')
        ldpc[name] = dict(n=n, k=k, rate=rate, channel_uses=SOURCE_BITS / rate,
                          fer={p: stats[f'{p:.4f}']['fer'] for p in BER_GRID})
        print(f'{name}: rate {rate:.3f}, {SOURCE_BITS/rate:.0f} channel uses; '
              f'FER@0.20={ldpc[name]["fer"][0.20]:.3f}')

    out = {'source_bits': SOURCE_BITS, 'clean_acc': clean,
           'cnn1bit_joint_no_fec_acc_vs_ber': cnn_joint,
           'note': 'CNN-1bit(learned, non-spiking) source + soft-BP LDPC; '
                   'analytic all-or-nothing frame survival; leakage-free '
                   'clean acc from valproto_cnn1bit.json. Also reports the '
                   'IDEAL (Shannon-limit) separation cliff per rate and the '
                   'CNN-1bit JOINT (no-FEC) curve for the joint-vs-separation '
                   'and spiking-vs-binary comparisons.', 'results': {}}
    for ds in clean:
        out['results'][ds] = {}
        acc_dead = 100.0 / N_CLASSES[ds]
        for name in ldpc:
            k = ldpc[name]['k']
            rate = ldpc[name]['rate']
            p_cliff = shannon_cliff_ber(rate)
            curve, ideal = {}, {}
            for p in BER_GRID:
                curve[f'{p:.2f}'] = round(analytic_accuracy(
                    SOURCE_BITS, clean[ds] / 100.0, ldpc[name]['fer'][p], k,
                    acc_dead), 2)
                # ideal separation: clean below the Shannon cliff, dead above
                ideal[f'{p:.2f}'] = round(clean[ds] if p < p_cliff else acc_dead, 2)
            out['results'][ds][name] = dict(
                rate=round(rate, 4),
                channel_uses=round(ldpc[name]['channel_uses']),
                shannon_cliff_ber=round(p_cliff, 4),
                acc_vs_ber=curve, acc_vs_ber_ideal_separation=ideal)
    out['ldpc'] = {n: {'rate': round(v['rate'], 4),
                       'channel_uses': round(v['channel_uses'])}
                   for n, v in ldpc.items()}
    json.dump(out, open(OUT, 'w'), indent=1)

    # headline: matched to SpikeAdapt rho=0.75 (42336 uses) = rate 1/6
    print('\n=== matched 42336 channel uses (rate 1/6), accuracy vs BER ===')
    for ds in clean:
        r16 = out['results'][ds]['r16']
        c, idl = r16['acc_vs_ber'], r16['acc_vs_ber_ideal_separation']
        j = cnn_joint[ds]
        print(f"  {ds} (Shannon cliff p={r16['shannon_cliff_ber']}):")
        print(f"    CNN-1bit+LDPC (practical) : 0.0 {c['0.00']} | 0.10 {c['0.10']} "
              f"| 0.20 {c['0.20']} | 0.30 {c['0.30']}")
        print(f"    ideal separation (Shannon): 0.0 {idl['0.00']} | 0.10 {idl['0.10']} "
              f"| 0.20 {idl['0.20']} | 0.30 {idl['0.30']}")
        print(f"    CNN-1bit JOINT (no FEC)   : 0.0 {j.get('0.0')} | 0.10 {j.get('0.1')} "
              f"| 0.20 {j.get('0.2')} | 0.30 {j.get('0.3')}")
    print('\nFinding: separation (learned source + LDPC) shows a DIGITAL CLIFF '
          '(dead by its Shannon cliff); the JOINT binary scheme degrades '
          'GRACEFULLY -> graceful degradation is from joint binary coding, not '
          'uniquely spiking. Spiking-specific gains = training-free temporal '
          'axis + event-driven energy (shown elsewhere).')
    print('saved:', OUT)


if __name__ == '__main__':
    main()
