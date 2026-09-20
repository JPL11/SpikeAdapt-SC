#!/usr/bin/env python3
"""Leakage-free closed-loop mission for the FREE-SPACE SEGMENTATION task
(journal deferred track #1). Same spiking bottleneck and (rho,T') axes as the
detection loop, scored on mIoU instead of mAP -- demonstrating the two-axis
adaptive primitive is TASK-general on a fixed backbone.

Reads radial_grid_valtest.json ([seed][val|test][cell]); mIoU is already
populated for every cell (run_FullEvaluation reports free-space mIoU). Policy
is built on the VAL surface, every mIoU is scored on the TEST surface. A2G
h=100 m suburban; pilot-noise MC (256-bit, no oracle); p<=0.40 envelope.

Usage: python eval/radial_seg_closedloop_valtest.py [--eps 1.0] [--pilot 256] [--mc 200]
Output: eval/seed_results/radial_seg_closedloop_valtest.json
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eval.channel_a2g import distance_profile  # noqa: E402

GRID = os.path.join(ROOT, 'eval/seed_results/radial_grid_valtest.json')
OUT = os.path.join(ROOT, 'eval/seed_results/radial_seg_closedloop_valtest.json')
SEEDS = ['42', '123', '456']
RHOS = [0.5, 0.75, 1.0]
TPRIMES = [4, 6, 8]
BERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
BER_MAX = 0.40
T_FULL = 8.0


def build_policy(gv_seeds, eps, mode='joint'):
    """Policy per BER from the mean mIoU over the given VAL grids; min payload
    within eps (mIoU points) of best. mode: joint | ronly (T'=8) | tonly (rho=1)."""
    policy = {}
    for ber in BERS:
        cells = []
        for rho in RHOS:
            for tp in TPRIMES:
                if mode == 'ronly' and tp != 8:
                    continue
                if mode == 'tonly' and rho != 1.0:
                    continue
                key = f'rho{rho}_T{tp}_ber{ber}'
                miou = float(np.mean([g[key]['mIoU'] for g in gv_seeds]))
                pay = gv_seeds[0][key]['payload_bits']
                cells.append((rho, tp, miou, pay))
        best = max(c[2] for c in cells)
        ok = [c for c in cells if c[2] >= best - eps]
        rho, tp, _, pay = min(ok, key=lambda c: c[3])
        policy[ber] = dict(rho=rho, T=tp, payload_bits=pay)
    return policy


def run_mission(policy, score_grid, prof, pilot_n, mc, rng):
    s_ad, pay_ad, s_fx = [], [], []
    clamped = 0
    full_pay = score_grid[f'rho1.0_T8_ber{BERS[0]}']['payload_bits']
    for r in prof:
        p_true = r['ber']
        if p_true > BER_MAX:
            clamped += 1
            continue
        near = min(BERS, key=lambda b: abs(b - p_true))
        s_fx.append(score_grid[f'rho1.0_T8_ber{near}']['mIoU'])
        mious, pays = [], []
        for _ in range(mc):
            p_hat = rng.binomial(pilot_n, p_true) / pilot_n
            pol = policy[min(BERS, key=lambda b: abs(b - p_hat))]
            cell = score_grid[f"rho{pol['rho']}_T{pol['T']}_ber{near}"]
            mious.append(cell['mIoU'])
            pays.append(pol['payload_bits'])
        s_ad.append(np.mean(mious))
        pay_ad.append(np.mean(pays))
    return dict(mIoU_adaptive=round(float(np.mean(s_ad)), 4),
                mIoU_fixed=round(float(np.mean(s_fx)), 4),
                payload_saving=round(1 - np.mean(pay_ad) / full_pay, 4),
                waypoints_used=len(s_ad), waypoints_truncated=clamped)


def best_fixed(score_grid, prof, max_frac):
    """Best fixed (rho,T') with payload fraction <= max_frac, by mission mIoU."""
    best = None
    for rho in RHOS:
        for tp in TPRIMES:
            if rho * tp / T_FULL > max_frac + 1e-9:
                continue
            accs = [score_grid[f'rho{rho}_T{tp}_ber'
                              f'{min(BERS, key=lambda b: abs(b - r["ber"]))}']['mIoU']
                    for r in prof if r['ber'] <= BER_MAX]
            m = float(np.mean(accs))
            if best is None or m > best[2]:
                best = (rho, tp, m, rho * tp / T_FULL)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eps', type=float, default=1.0)
    ap.add_argument('--pilot', type=int, default=256)
    ap.add_argument('--mc', type=int, default=200)
    ap.add_argument('--h', type=float, default=100.0)
    args = ap.parse_args()
    g = json.load(open(GRID))
    rng = np.random.default_rng(3)
    prof = [r for r in distance_profile(args.h, 'suburban')
            if 200 <= r['d_ground_m'] <= 3000]

    out = {'params': vars(args), 'task': 'free-space segmentation (mIoU)',
           'protocol': 'policy on VAL surface, mIoU on TEST surface; '
           'A2G h=100m suburban; pilot-noise MC; envelope p<=0.40'}
    for mode in ['joint', 'ronly', 'tonly']:
        per = []
        for s in SEEDS:
            pol = build_policy([g[s]['val']], args.eps, mode)
            per.append(run_mission(pol, g[s]['test'], prof, args.pilot,
                                   args.mc, rng))
        out[mode] = {k: round(float(np.mean([p[k] for p in per])), 4)
                     for k in ['mIoU_adaptive', 'mIoU_fixed', 'payload_saving']}
        out[mode]['per_seed_mIoU_adaptive'] = [p['mIoU_adaptive'] for p in per]
    jfrac = 1 - out['joint']['payload_saving']
    bf = [best_fixed(g[s]['test'], prof, jfrac) for s in SEEDS]
    out['best_fixed_below'] = dict(
        mIoU=round(float(np.mean([b[2] for b in bf])), 4),
        payload_frac=round(bf[0][3], 4), cell=f'rho{bf[0][0]}_T{bf[0][1]}')
    loso = []
    for held in SEEDS:
        others = [s for s in SEEDS if s != held]
        pol = build_policy([g[o]['val'] for o in others], args.eps)
        loso.append(run_mission(pol, g[held]['test'], prof, args.pilot,
                                args.mc, rng))
    out['loso'] = {k: round(float(np.mean([p[k] for p in loso])), 4)
                   for k in ['mIoU_adaptive', 'mIoU_fixed', 'payload_saving']}
    out['loso']['worst_mIoU'] = round(float(min(p['mIoU_adaptive']
                                               for p in loso)), 4)
    json.dump(out, open(OUT, 'w'), indent=1)
    for k in ['joint', 'ronly', 'tonly', 'loso']:
        print(f"{k}: mIoU {out[k]['mIoU_adaptive']} vs fixed "
              f"{out[k]['mIoU_fixed']}, saving {out[k]['payload_saving']}")
    print('best fixed <= joint payload:', out['best_fixed_below'])
    print('saved:', OUT)


if __name__ == '__main__':
    main()
