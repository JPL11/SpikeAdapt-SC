#!/usr/bin/env python3
"""LOSO policy transfer for the RADIal closed loop: for each held-out seed,
derive the per-BER (rho, T') policy from the OTHER two seeds' grids (mean F1,
min payload within eps), then evaluate that policy on the held-out seed's
grid — accuracy the policy would deliver on a model it never saw.

Requires eval/seed_results/radial_grid_rho_T{,_s123,_s456}.json.
Usage: python eval/radial_loso_policy.py [--eps 0.01]
Output: eval/seed_results/radial_loso_policy.json
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eval.channel_a2g import distance_profile  # noqa: E402

GRIDS = {42: 'radial_grid_rho_T.json',
         123: 'radial_grid_rho_T_s123.json',
         456: 'radial_grid_rho_T_s456.json'}
OUT = os.path.join(ROOT, 'eval/seed_results/radial_loso_policy.json')
RHOS = [0.5, 0.75, 1.0]
TPRIMES = [4, 6, 8]
BERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def f1(m):
    p, r = m['mAP'], m['mAR']
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eps', type=float, default=0.01)
    args = ap.parse_args()
    grids = {s: json.load(open(os.path.join(ROOT, 'eval/seed_results', f)))
             for s, f in GRIDS.items()}

    prof = [r for r in distance_profile(100.0, 'suburban')
            if 200 <= r['d_ground_m'] <= 3000]

    out = {'eps': args.eps, 'per_heldout_seed': {}}
    mission_ad, mission_fx, savings = [], [], []
    for held in GRIDS:
        train_seeds = [s for s in GRIDS if s != held]
        # policy from mean F1 over training seeds
        policy = {}
        for ber in BERS:
            cells = []
            for rho in RHOS:
                for tp in TPRIMES:
                    key = f'rho{rho}_T{tp}_ber{ber}'
                    mf1 = float(np.mean([f1(grids[s][key]) for s in train_seeds]))
                    pay = grids[train_seeds[0]][key]['payload_bits']
                    cells.append((rho, tp, mf1, pay))
            best = max(c[2] for c in cells)
            ok = [c for c in cells if c[2] >= best - args.eps]
            rho, tp, _, pay = min(ok, key=lambda c: c[3])
            policy[str(ber)] = dict(rho=rho, T=tp, payload_bits=pay)

        # evaluate on held-out seed along the trajectory
        traj_ad, traj_fx, pay_ad, pay_fx = [], [], [], []
        for r in prof:
            ber = min(BERS, key=lambda b: abs(b - min(r['ber'], 0.30)))
            p = policy[str(ber)]
            held_cell = grids[held][f"rho{p['rho']}_T{p['T']}_ber{ber}"]
            full_cell = grids[held][f'rho1.0_T8_ber{ber}']
            traj_ad.append(held_cell['mAP'])
            traj_fx.append(full_cell['mAP'])
            pay_ad.append(p['payload_bits'])
            pay_fx.append(full_cell['payload_bits'])
        m_ad, m_fx = float(np.mean(traj_ad)), float(np.mean(traj_fx))
        sv = 1 - float(np.mean(pay_ad)) / float(np.mean(pay_fx))
        out['per_heldout_seed'][str(held)] = dict(
            policy=policy, mAP_adaptive=round(m_ad, 4),
            mAP_fixed=round(m_fx, 4), payload_saving=round(sv, 4))
        mission_ad.append(m_ad); mission_fx.append(m_fx); savings.append(sv)
        print(f'held-out seed {held}: adaptive mAP {m_ad:.4f} vs fixed '
              f'{m_fx:.4f}, payload saving {sv*100:.1f}%')

    out['summary'] = dict(
        mAP_adaptive_mean=round(float(np.mean(mission_ad)), 4),
        mAP_adaptive_std=round(float(np.std(mission_ad, ddof=1)), 4),
        mAP_fixed_mean=round(float(np.mean(mission_fx)), 4),
        payload_saving_mean=round(float(np.mean(savings)), 4))
    json.dump(out, open(OUT, 'w'), indent=1)
    s = out['summary']
    print(f"\nLOSO mission: adaptive {s['mAP_adaptive_mean']}±"
          f"{s['mAP_adaptive_std']} vs fixed {s['mAP_fixed_mean']}, "
          f"saving {s['payload_saving_mean']*100:.1f}%")
    print(f'saved: {OUT}')


if __name__ == '__main__':
    main()
