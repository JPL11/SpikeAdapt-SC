#!/usr/bin/env python3
"""Complete Table II statistics for the ICASSP radar paper (reviewer fix):
per-row mAP, mAR, F1, and std across seeds, plus a best-fixed comparator
selected by the SAME criterion the policy uses (harmonic-mean F1), reported
with its mAR/F1 (previously only mAP was shown).

Reuses build_policy / run_mission from the leakage-free closed loop; the
best-fixed row is computed by F1 over fixed (rho,T') cells whose average
payload does not exceed the joint loop's.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eval.channel_a2g import distance_profile
from eval.radial_closedloop_valtest import (GRID, SEEDS, RHOS, TPRIMES, BERS,
                                            BER_MAX, T_FULL, build_policy,
                                            run_mission)

ap = argparse.ArgumentParser()
ap.add_argument('--grid', default=GRID,
                help='SNN grid json (e.g. radial_ylink_grid_valtest.json)')
ap.add_argument('--out', default=os.path.join(
    ROOT, 'eval/seed_results/radial_table2_stats.json'))
args = ap.parse_args()
g = json.load(open(args.grid))
rng = np.random.default_rng(3)
prof = [r for r in distance_profile(100.0, 'suburban') if 200 <= r['d_ground_m'] <= 3000]
PILOT, MC = 256, 200


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def mission_fixed(score_grid, rho, tp):
    """Mission mean mAP/mAR for a FIXED (rho,tp) cell over the trajectory."""
    aps, ars = [], []
    for w in prof:
        if w['ber'] > BER_MAX:
            continue
        near = min(BERS, key=lambda b: abs(b - w['ber']))
        c = score_grid[f'rho{rho}_T{tp}_ber{near}']
        aps.append(c['mAP']); ars.append(c['mAR'])
    return float(np.mean(aps)), float(np.mean(ars)), rho * tp / T_FULL


def stat(rows, key):
    v = [r[key] for r in rows]
    return float(np.mean(v)), float(np.std(v, ddof=1))


# ---- adaptive policies (joint / temporal-only / spatial-only), per seed ----
out = {}
for mode in ['joint', 'ronly', 'tonly']:
    per = []
    for s in SEEDS:
        pol = build_policy([g[s]['val']], 0.01, mode)
        m = run_mission(pol, g[s]['test'], prof, PILOT, MC, rng)
        m['F1'] = f1(m['mAP_adaptive'], m['mAR_adaptive'])
        per.append(m)
    out[mode] = {
        'mAP': stat(per, 'mAP_adaptive'), 'mAR': stat(per, 'mAR_adaptive'),
        'F1': stat(per, 'F1'),
        'saving': float(np.mean([p['payload_saving'] for p in per]))}

# fair 1-bit control through the identical validation-selected joint loop
gf = json.load(open(os.path.join(
    ROOT, 'eval/seed_results/radial_fair1bit_grid_valtest.json')))
per = []
for s in SEEDS:
    pol = build_policy([gf[s]['val']], 0.01, 'joint')
    m = run_mission(pol, gf[s]['test'], prof, PILOT, MC, rng)
    m['F1'] = f1(m['mAP_adaptive'], m['mAR_adaptive'])
    per.append(m)
out['fair_joint'] = {
    'mAP': stat(per, 'mAP_adaptive'), 'mAR': stat(per, 'mAR_adaptive'),
    'F1': stat(per, 'F1'),
    'saving': float(np.mean([p['payload_saving'] for p in per]))}

# full-rate fixed (rho=1,T=8), per seed
per = []
for s in SEEDS:
    ap, ar, _ = mission_fixed(g[s]['test'], 1.0, 8)
    per.append({'mAP': ap, 'mAR': ar, 'F1': f1(ap, ar)})
out['fixed_full'] = {'mAP': stat(per, 'mAP'), 'mAR': stat(per, 'mAR'),
                     'F1': stat(per, 'F1'), 'saving': 0.0}

# best-fixed by F1, avg payload <= joint's avg payload
jfrac = 1 - out['joint']['saving']
per = []
for s in SEEDS:
    best = None
    for rho in RHOS:
        for tp in TPRIMES:
            frac = rho * tp / T_FULL
            if frac > jfrac + 1e-9:
                continue
            ap, ar, _ = mission_fixed(g[s]['test'], rho, tp)
            fv = f1(ap, ar)
            if best is None or fv > best['F1']:
                best = {'mAP': ap, 'mAR': ar, 'F1': fv, 'frac': frac,
                        'cell': f'rho{rho}_T{tp}'}
    per.append(best)
out['best_fixed'] = {'mAP': stat(per, 'mAP'), 'mAR': stat(per, 'mAR'),
                     'F1': stat(per, 'F1'),
                     'saving': 1 - float(np.mean([p['frac'] for p in per])),
                     'cell': per[0]['cell']}

print(f"{'row':<12}{'mAP':>16}{'mAR':>16}{'F1':>16}{'saving':>9}")
for k in ['joint', 'tonly', 'ronly', 'fair_joint', 'best_fixed', 'fixed_full']:
    r = out[k]
    c = f"  {r.get('cell','')}"
    print(f"{k:<12}{r['mAP'][0]:.3f}+/-{r['mAP'][1]:.3f}   "
          f"{r['mAR'][0]:.3f}+/-{r['mAR'][1]:.3f}   "
          f"{r['F1'][0]:.3f}+/-{r['F1'][1]:.3f}   {r['saving']*100:5.1f}%{c}")
json.dump({k: {kk: (vv if not isinstance(vv, tuple) else list(vv))
               for kk, vv in v.items()} for k, v in out.items()},
          open(args.out, 'w'), indent=1)
print('saved:', args.out)
