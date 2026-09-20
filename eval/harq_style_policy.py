#!/usr/bin/env python3
"""SNN-SC-HARQ-style policy simulation from per-sample records.

Stopping rule: transmit T'=3, then retransmit increments (4,5,6,8) until the
RX confidence pmax >= tau or depth is exhausted (their similarity-gated,
spatially-uniform, single-axis policy, on our codec). tau is selected on VAL
(cheapest mission payload within eps=0.5pp of the best val mission accuracy),
scored once on TEST -- the same leakage-free criterion as the joint loop.

Reports 3-seed mission accuracy @ payload + mean retransmission rounds, next
to the one-shot pilot-driven joint loop for the same (taug) codec.

Usage: python eval/harq_style_policy.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'eval'))
from eval.eval_trajectory_closedloop import mission_profile     # noqa: E402

SCHEDULE = np.array([3, 4, 5, 6, 8])
BERS = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
TAUS = np.linspace(0.05, 0.999, 120)
EPS = 0.5
_, _, _, MISSION_BERS = mission_profile()
MB = np.clip(np.asarray(MISSION_BERS, dtype=float), 0, 0.30)


def curves(correct, pmax, tau):
    """acc/payload/rounds per BER for stopping threshold tau."""
    hit = pmax.astype(np.float32) >= tau            # (N, B, S)
    first = np.where(hit.any(-1), hit.argmax(-1), len(SCHEDULE) - 1)
    acc = np.take_along_axis(correct, first[..., None], -1)[..., 0]
    tp = SCHEDULE[first]
    return acc.mean(0) * 100, (tp / 8.0).mean(0), (first + 1.0).mean(0)


def mission(vec):
    return float(np.interp(MB, BERS, vec).mean())


def run(ds):
    p = os.path.join(ROOT, f'eval/seed_results/harq_style_persample_{ds}.npz')
    if not os.path.exists(p):
        print(f'{ds}: no persample file yet'); return None
    z = np.load(p)
    accs, pays, rnds, taus = [], [], [], []
    for seed in [42, 123, 456]:
        if f's{seed}_test_correct' not in z:
            continue
        cv, pv = z[f's{seed}_val_correct'], z[f's{seed}_val_pmax']
        ct, pt = z[f's{seed}_test_correct'], z[f's{seed}_test_pmax']
        # tau selection on VAL mission curves
        val = [(mission(a), mission(pl))
               for a, pl, _ in (curves(cv, pv, t) for t in TAUS)]
        best = max(a for a, _ in val)
        cand = [(pl, t) for (a, pl), t in zip(val, TAUS) if a >= best - EPS]
        _, tau = min(cand)
        a, pl, r = curves(ct, pt, tau)
        accs.append(mission(a)); pays.append(mission(pl))
        rnds.append(mission(r)); taus.append(tau)
    if not accs:
        return None
    return (np.mean(accs), np.std(accs), np.mean(pays), np.mean(rnds),
            np.mean(taus), len(accs))


def main():
    print('=' * 70)
    print('SNN-SC-HARQ-STYLE policy (RX pmax-gated incremental-T\', rho=1) '
          'on the taug codec')
    print('mission-average, val-selected tau, test scored once')
    print('=' * 70)
    for ds in ['aid', 'resisc45']:
        r = run(ds)
        if r is None:
            continue
        a, sd, pl, rd, tau, n = r
        print(f'{ds:9}: acc {a:.2f}±{sd:.2f} @ {pl*100:5.1f}% payload, '
              f'{rd:.2f} rounds/frame (tau~{tau:.2f}, {n} seeds)')
    print('\nreference (same codec, one-shot pilot-driven JOINT loop, '
          '1 round, + spatial axis):')
    print('  aid      : 95.86 @ 34.0%')
    print('  resisc45 : 91.95 @ 31.4%')


if __name__ == '__main__':
    main()
