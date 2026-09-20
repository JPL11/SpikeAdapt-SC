#!/usr/bin/env python3
"""Leakage-free closed-loop mission for the radar paper (review fix #4).

Reads radial_grid_valtest.json ([seed][val|test][cell]); the (rho,T')
policy is built on the VALIDATION surface and every accuracy is scored on
the TEST surface. Mirrors the ICC valproto protocol. A2G profile h=100 m
suburban; pilot-noise MC (256-bit, no oracle); envelope truncation at
p<=0.40 (disclosed); mAP and mAR reported for both arms; static + LOSO +
single-axis factorial rows.

Usage: python eval/radial_closedloop_valtest.py [--eps 0.01] [--pilot 256] [--mc 200]
Output: eval/seed_results/radial_closedloop_valtest.json
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
OUT = os.path.join(ROOT, 'eval/seed_results/radial_closedloop_valtest.json')
SEEDS = ['42', '123', '456']
RHOS = [0.5, 0.75, 1.0]
TPRIMES = [4, 6, 8]
BERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
BER_MAX = 0.40
T_FULL = 8.0


def f1(m):
    p, r = m['mAP'], m['mAR']
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def build_policy(gv_seeds, eps, mode='joint'):
    """Policy per BER from the mean F1 over the given VAL grids; min payload
    within eps of best. mode: joint | ronly (T'=8) | tonly (rho=1)."""
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
                mf1 = float(np.mean([f1(g[key]) for g in gv_seeds]))
                pay = gv_seeds[0][key]['payload_bits']
                cells.append((rho, tp, mf1, pay))
        best = max(c[2] for c in cells)
        ok = [c for c in cells if c[2] >= best - eps]
        rho, tp, _, pay = min(ok, key=lambda c: c[3])
        policy[ber] = dict(rho=rho, T=tp, payload_bits=pay)
    return policy


def run_mission(policy, score_grid, prof, pilot_n, mc, rng):
    m_ad, r_ad, pay_ad, m_fx, r_fx = [], [], [], [], []
    clamped = 0
    full_pay = score_grid[f'rho1.0_T8_ber{BERS[0]}']['payload_bits']
    for r in prof:
        p_true = r['ber']
        if p_true > BER_MAX:
            clamped += 1
            continue
        near = min(BERS, key=lambda b: abs(b - p_true))
        fx = score_grid[f'rho1.0_T8_ber{near}']
        m_fx.append(fx['mAP']); r_fx.append(fx['mAR'])
        maps, mars, pays = [], [], []
        for _ in range(mc):
            p_hat = rng.binomial(pilot_n, p_true) / pilot_n
            pol = policy[min(BERS, key=lambda b: abs(b - p_hat))]
            cell = score_grid[f"rho{pol['rho']}_T{pol['T']}_ber{near}"]
            maps.append(cell['mAP']); mars.append(cell['mAR'])
            pays.append(pol['payload_bits'])
        m_ad.append(np.mean(maps)); r_ad.append(np.mean(mars))
        pay_ad.append(np.mean(pays))
    return dict(mAP_adaptive=round(float(np.mean(m_ad)), 4),
                mAR_adaptive=round(float(np.mean(r_ad)), 4),
                mAP_fixed=round(float(np.mean(m_fx)), 4),
                mAR_fixed=round(float(np.mean(r_fx)), 4),
                payload_saving=round(1 - np.mean(pay_ad) / full_pay, 4),
                waypoints_used=len(m_ad), waypoints_truncated=clamped)


def best_fixed(score_grid, prof, max_frac):
    """Best fixed (rho,T') with payload fraction <= max_frac, by mission mAP."""
    best = None
    for rho in RHOS:
        for tp in TPRIMES:
            if rho * tp / T_FULL > max_frac + 1e-9:
                continue
            accs = [score_grid[f'rho{rho}_T{tp}_ber'
                              f'{min(BERS, key=lambda b: abs(b - r["ber"]))}']['mAP']
                    for r in prof if r['ber'] <= BER_MAX]
            m = float(np.mean(accs))
            if best is None or m > best[2]:
                best = (rho, tp, m, rho * tp / T_FULL)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eps', type=float, default=0.01)
    ap.add_argument('--pilot', type=int, default=256)
    ap.add_argument('--mc', type=int, default=200)
    ap.add_argument('--h', type=float, default=100.0)
    ap.add_argument('--grid', default=GRID,
                    help='grid json (e.g. radial_ylink_grid_valtest.json)')
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()
    g = json.load(open(args.grid))
    rng = np.random.default_rng(3)
    prof = [r for r in distance_profile(args.h, 'suburban')
            if 200 <= r['d_ground_m'] <= 3000]

    out = {'params': vars(args), 'protocol':
           'policy on VAL surface, accuracy on TEST surface; '
           'A2G h=100m suburban; pilot-noise MC; envelope p<=0.40'}
    # 3-seed joint + single-axis, per-seed own val policy / own test scoring
    for mode in ['joint', 'ronly', 'tonly']:
        per = []
        for s in SEEDS:
            pol = build_policy([g[s]['val']], args.eps, mode)
            per.append(run_mission(pol, g[s]['test'], prof, args.pilot,
                                   args.mc, rng))
        out[mode] = {k: round(float(np.mean([p[k] for p in per])), 4)
                     for k in ['mAP_adaptive', 'mAR_adaptive', 'mAP_fixed',
                               'mAR_fixed', 'payload_saving']}
        out[mode]['per_seed_mAP_adaptive'] = [p['mAP_adaptive'] for p in per]
    # best fixed at the joint loop's mission payload (from below)
    jfrac = 1 - out['joint']['payload_saving']
    bf = [best_fixed(g[s]['test'], prof, jfrac) for s in SEEDS]
    out['best_fixed_below'] = dict(
        mAP=round(float(np.mean([b[2] for b in bf])), 4),
        payload_frac=round(bf[0][3], 4), cell=f'rho{bf[0][0]}_T{bf[0][1]}')
    # LOSO: policy from 2 seeds' val, scored on held-out seed's test
    loso = []
    for held in SEEDS:
        others = [s for s in SEEDS if s != held]
        pol = build_policy([g[o]['val'] for o in others], args.eps)
        loso.append(run_mission(pol, g[held]['test'], prof, args.pilot,
                                args.mc, rng))
    out['loso'] = {k: round(float(np.mean([p[k] for p in loso])), 4)
                   for k in ['mAP_adaptive', 'mAR_adaptive', 'mAP_fixed',
                             'mAR_fixed', 'payload_saving']}
    out['loso']['worst_mAP'] = round(float(min(p['mAP_adaptive']
                                               for p in loso)), 4)
    json.dump(out, open(args.out, 'w'), indent=1)
    for k in ['joint', 'ronly', 'tonly', 'loso']:
        print(f"{k}: mAP {out[k]['mAP_adaptive']} vs fixed "
              f"{out[k]['mAP_fixed']}, saving {out[k]['payload_saving']}")
    print('best fixed <= joint payload:', out['best_fixed_below'])
    print('saved:', args.out)


if __name__ == '__main__':
    main()
