#!/usr/bin/env python3
"""E3 (review round 1): hardened closed-loop evaluation.

Changes vs radial_closedloop.py (panel finding F3):
  - grid extended to p in {0.35, 0.40}: the mission is truncated at the
    measured envelope (link BER <= 0.40, ~2.2 km at h=100 m) and any residual
    clamp is DISCLOSED in the output (clamped waypoint count).
  - pilot-noise Monte Carlo: the receiver estimates p_hat from N pilot bits
    (binomial draw) and indexes the policy at the nearest grid BER; results
    average over --mc draws per waypoint. No oracle BER.
  - mission mAR reported alongside mAP for both arms.
  - LOSO variant included (policy from 2 seeds' grids, scored on held-out).

Requires grids WITH the extended BERs for all three seeds.
Usage: python eval/radial_closedloop_v2.py [--eps 0.01] [--pilot 256] [--mc 200]
Output: eval/seed_results/radial_closedloop_v2.json
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
OUT = os.path.join(ROOT, 'eval/seed_results/radial_closedloop_v2.json')
RHOS = [0.5, 0.75, 1.0]
TPRIMES = [4, 6, 8]
BERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
BER_MAX = 0.40


def f1(m):
    p, r = m['mAP'], m['mAR']
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def build_policy(grids, seeds, eps):
    policy = {}
    for ber in BERS:
        cells = []
        for rho in RHOS:
            for tp in TPRIMES:
                key = f'rho{rho}_T{tp}_ber{ber}'
                mf1 = float(np.mean([f1(grids[s][key]) for s in seeds]))
                pay = grids[seeds[0]][key]['payload_bits']
                cells.append((rho, tp, mf1, pay))
        best = max(c[2] for c in cells)
        ok = [c for c in cells if c[2] >= best - eps]
        rho, tp, _, pay = min(ok, key=lambda c: c[3])
        policy[ber] = dict(rho=rho, T=tp, payload_bits=pay)
    return policy


def run_mission(policy, score_grid, prof, pilot_n, mc, rng, collect=False):
    """Pilot-noise MC mission average. Returns dict of means + clamp count."""
    m_ad, r_ad, pay_ad, m_fx, r_fx = [], [], [], [], []
    clamped = 0
    traj = []
    full_pay = score_grid[f'rho1.0_T8_ber{BERS[0]}']['payload_bits']
    for r in prof:
        p_true = r['ber']
        if p_true > BER_MAX:
            clamped += 1
            continue                      # truncate at measured envelope
        fixed_cell = score_grid[
            f'rho1.0_T8_ber{min(BERS, key=lambda b: abs(b - p_true))}']
        m_fx.append(fixed_cell['mAP']); r_fx.append(fixed_cell['mAR'])
        maps, mars, pays = [], [], []
        for _ in range(mc):
            p_hat = rng.binomial(pilot_n, p_true) / pilot_n
            ber_g = min(BERS, key=lambda b: abs(b - p_hat))
            pol = policy[ber_g]
            # channel is still p_true: score the chosen cell at the true BER
            cell = score_grid[
                f"rho{pol['rho']}_T{pol['T']}_ber"
                f"{min(BERS, key=lambda b: abs(b - p_true))}"]
            maps.append(cell['mAP']); mars.append(cell['mAR'])
            pays.append(pol['payload_bits'])
        m_ad.append(float(np.mean(maps))); r_ad.append(float(np.mean(mars)))
        pay_ad.append(float(np.mean(pays)))
        if collect:
            traj.append(dict(d_m=r['d_ground_m'], ber=round(p_true, 4),
                             mAP_adaptive=round(m_ad[-1], 4),
                             mAR_adaptive=round(r_ad[-1], 4),
                             mAP_fixed=fixed_cell['mAP'],
                             mAR_fixed=fixed_cell['mAR'],
                             payload_adaptive=int(pay_ad[-1]),
                             payload_fixed=full_pay))
    res = dict(mAP_adaptive=round(float(np.mean(m_ad)), 4),
               mAR_adaptive=round(float(np.mean(r_ad)), 4),
               mAP_fixed=round(float(np.mean(m_fx)), 4),
               mAR_fixed=round(float(np.mean(r_fx)), 4),
               payload_saving=round(1 - np.mean(pay_ad) / full_pay, 4),
               waypoints_used=len(m_ad), waypoints_truncated=clamped)
    if collect:
        res['trajectory'] = traj
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eps', type=float, default=0.01)
    ap.add_argument('--pilot', type=int, default=256)
    ap.add_argument('--mc', type=int, default=200)
    ap.add_argument('--h', type=float, default=100.0)
    ap.add_argument('--suffix', default='',
                    help="grid-file suffix, e.g. '_so' for spikes-only models")
    args = ap.parse_args()
    rng = np.random.default_rng(3)
    global OUT
    if args.suffix:
        OUT = OUT.replace('.json', f'{args.suffix}.json')
    grids = {s: json.load(open(os.path.join(
                 ROOT, 'eval/seed_results',
                 f.replace('.json', f'{args.suffix}.json'))))
             for s, f in GRIDS.items()}
    for s, g in grids.items():
        missing = [f'rho{r}_T{t}_ber{b}' for r in RHOS for t in TPRIMES
                   for b in BERS if f'rho{r}_T{t}_ber{b}' not in g]
        if missing:
            sys.exit(f'seed {s} grid missing {len(missing)} extended points '
                     f'(run radial_grid_rho_T.py --ext): {missing[:3]}')

    prof = [r for r in distance_profile(args.h, 'suburban')
            if 200 <= r['d_ground_m'] <= 3000]

    out = {'params': vars(args), 'ber_max': BER_MAX}
    # main result: 3-seed policy scored per seed, averaged
    pol_all = build_policy(grids, list(grids), args.eps)
    per_seed = {s: run_mission(pol_all, grids[s], prof, args.pilot, args.mc,
                               rng, collect=(s == 42)) for s in grids}
    out['policy'] = {str(k): v for k, v in pol_all.items()}
    out['per_seed'] = {str(s): v for s, v in per_seed.items()}
    out['mission_mean'] = {
        k: round(float(np.mean([per_seed[s][k] for s in grids])), 4)
        for k in ['mAP_adaptive', 'mAR_adaptive', 'mAP_fixed', 'mAR_fixed',
                  'payload_saving']}

    # LOSO: policy from 2 seeds, scored on held-out
    loso = {}
    for held in grids:
        train_seeds = [s for s in grids if s != held]
        pol = build_policy(grids, train_seeds, args.eps)
        loso[str(held)] = run_mission(pol, grids[held], prof, args.pilot,
                                      args.mc, rng)
    out['loso'] = loso
    out['loso_mean'] = {
        k: round(float(np.mean([loso[s][k] for s in loso])), 4)
        for k in ['mAP_adaptive', 'mAR_adaptive', 'mAP_fixed', 'mAR_fixed',
                  'payload_saving']}
    json.dump(out, open(OUT, 'w'), indent=1)
    print('mission (3-seed mean, pilot-noise MC):', out['mission_mean'])
    print('LOSO mean:', out['loso_mean'])
    print('waypoints used/truncated:',
          per_seed[42]['waypoints_used'], per_seed[42]['waypoints_truncated'])
    print(f'saved: {OUT}')


if __name__ == '__main__':
    main()
