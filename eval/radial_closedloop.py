#!/usr/bin/env python3
"""Closed-loop (rho, T') adaptation on RADIal from the measured grid, with
A2G physical grounding — the radar-track analog of the ICC paper's
trajectory result.

Policy: for each channel BER, choose the (rho, T') cell minimizing payload
subject to mAP >= best_mAP_at_that_BER - epsilon (grid-argmax policy;
LOSO-style held-out policy transfer is future work, noted in output).

Trajectory: UAV at h=100 m, suburban A2G (eval/channel_a2g.py profile);
ground distance sweep maps to BER via the Rician link; closed loop applies
the policy per position (nearest grid BER); compared against the fixed
full-rate (rho=1, T'=8) configuration on mission-average mAP and payload.

Usage: python eval/radial_closedloop.py [--eps 0.01]
Output: eval/seed_results/radial_closedloop.json
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eval.channel_a2g import distance_profile  # noqa: E402

GRID = os.path.join(ROOT, 'eval/seed_results/radial_grid_rho_T.json')
OUT = os.path.join(ROOT, 'eval/seed_results/radial_closedloop.json')
RHOS = [0.5, 0.75, 1.0]
TPRIMES = [4, 6, 8]
BERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eps', type=float, default=0.01,
                    help='allowed mAP sacrifice vs best cell at each BER')
    args = ap.parse_args()
    grid = json.load(open(GRID))

    def cell(rho, tp, ber):
        return grid[f'rho{rho}_T{tp}_ber{ber}']

    # ---- per-BER policy: min payload within eps of the best F1-style score
    # (harmonic mean of mAP and mAR — mAP alone is gameable: some cells hold
    # precision while recall collapses, e.g. rho=1.0/T'=4 at high BER)
    def f1(m):
        p, r = m['mAP'], m['mAR']
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    policy = {}
    for ber in BERS:
        cells = [(rho, tp, cell(rho, tp, ber)) for rho in RHOS for tp in TPRIMES]
        best = max(f1(c[2]) for c in cells)
        ok = [c for c in cells if f1(c[2]) >= best - args.eps]
        rho, tp, m = min(ok, key=lambda c: c[2]['payload_bits'])
        policy[str(ber)] = dict(rho=rho, T=tp, mAP=m['mAP'], mAR=m['mAR'],
                                f1=round(f1(m), 4), payload_bits=m['payload_bits'],
                                best_f1=round(best, 4))

    full = {str(b): cell(1.0, 8, b) for b in BERS}
    full_payload = full['0.0']['payload_bits']

    # ---- A2G trajectory: h=100 m suburban, d_ground 200 m .. 3 km
    prof = [r for r in distance_profile(100.0, 'suburban')
            if 200 <= r['d_ground_m'] <= 3000]
    traj = []
    for r in prof:
        ber_grid = min(BERS, key=lambda b: abs(b - min(r['ber'], 0.30)))
        p = policy[str(ber_grid)]
        traj.append(dict(d_m=r['d_ground_m'], snr_db=r['snr_db'],
                         ber_link=round(r['ber'], 4), ber_grid=ber_grid,
                         rho=p['rho'], T=p['T'],
                         mAP_adaptive=p['mAP'], mAR_adaptive=p['mAR'],
                         mAP_fixed=full[str(ber_grid)]['mAP'],
                         mAR_fixed=full[str(ber_grid)]['mAR'],
                         payload_adaptive=p['payload_bits'],
                         payload_fixed=full_payload))

    mean_ad = float(np.mean([t['mAP_adaptive'] for t in traj]))
    mean_fx = float(np.mean([t['mAP_fixed'] for t in traj]))
    mar_ad = float(np.mean([t['mAR_adaptive'] for t in traj]))
    mar_fx = float(np.mean([t['mAR_fixed'] for t in traj]))
    pay_ad = float(np.mean([t['payload_adaptive'] for t in traj]))
    pay_fx = float(np.mean([t['payload_fixed'] for t in traj]))
    summary = dict(
        eps=args.eps,
        policy=policy,
        trajectory=traj,
        mission=dict(mAP_adaptive=round(mean_ad, 4),
                     mAP_fixed=round(mean_fx, 4),
                     mAR_adaptive=round(mar_ad, 4),
                     mAR_fixed=round(mar_fx, 4),
                     payload_adaptive_bits=int(pay_ad),
                     payload_fixed_bits=int(pay_fx),
                     payload_saving=round(1 - pay_ad / pay_fx, 4)),
        caveat='grid-argmax policy on the evaluation grid; held-out (LOSO) '
               'policy transfer as in the ICC paper is future work')
    json.dump(summary, open(OUT, 'w'), indent=1)
    print('Per-BER policy (eps=%.3f):' % args.eps)
    for b in BERS:
        p = policy[str(b)]
        print(f"  BER {b:4.2f}: rho={p['rho']} T'={p['T']} "
              f"mAP {p['mAP']:.4f} mAR {p['mAR']:.4f} "
              f"(best F1 {p['best_f1']:.4f}) "
              f"payload {p['payload_bits']/1e6:.2f} Mbit")
    m = summary['mission']
    print(f"\nMission (h=100 m suburban, 0.2-3 km): "
          f"adaptive mAP {m['mAP_adaptive']} vs fixed {m['mAP_fixed']}, "
          f"payload saving {m['payload_saving']*100:.1f}%")
    print(f'saved: {OUT}')


if __name__ == '__main__':
    main()
