#!/usr/bin/env python3
"""Head-to-head: SpikeAdapt-SC vs. Adaptive-1bit-Grouped on the JOINT (rho,T')
axis, using the IDENTICAL Table-III policy/mission machinery for both.

Both grids share the shape [seed][split][rho][tprime][ber]. For each method we
build the val-selected policy and score on test over the ergodic A2G mission,
in three modes: joint (rho,T'), rho-only (T'=8), T'-only (rho=1.0). Same eps,
same budget (rho*T'/T<=0.75), same 200 pilot draws, same seeds (42/123/456).

If SpikeAdapt's joint mission beats Adaptive-1bit's at equal/less payload, the
free-temporal-axis co-design thesis holds against the strongest fair ANN.

Usage: python eval/compare_joint_baseline.py
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'eval'))
from eval.valproto_missions import acc_at, T_FULL, BUDGET, N_PILOT, RHOS, TPRIMES
from eval.valproto_table3_rows import policy, static_mission, fixed_mission, MISSION_BERS

SEEDS = ['42', '123', '456']


def load_grid(path):
    g = json.load(open(path))
    tables = {}
    for s in SEEDS:
        if s not in g:
            continue
        tables[s] = {}
        for split in ['val', 'test']:
            t = {}
            for rho in RHOS:
                for tp in TPRIMES:
                    d = g[s][split][str(rho)][str(tp)]
                    items = sorted(d.items(), key=lambda kv: float(kv[0]))
                    t[(rho, tp)] = (np.array([float(b) for b, _ in items]),
                                    np.array([a for _, a in items]))
            tables[s][split] = t
    return tables


def missions(tables, eps=0.5):
    seeds = [s for s in SEEDS if s in tables]
    out = {}
    for label, mode in [('joint', 'joint'), ('rho_only', 'ronly'),
                        ('tprime_only', 'tonly')]:
        rng = np.random.default_rng(7)
        r = [static_mission(tables[s]['val'], tables[s]['test'], rng, eps, mode)
             for s in seeds]
        out[label] = (round(float(np.mean([x[0] for x in r])), 2),
                      round(float(np.mean([x[1] for x in r])), 4))
    # fixed full-rate reference
    r = [fixed_mission(tables[s]['test'], (1.0, 8)) for s in seeds]
    out['fixed_full'] = (round(float(np.mean([x[0] for x in r])), 2), 1.0)
    return out


def main():
    print(f"{'='*74}\nJOINT-AXIS HEAD-TO-HEAD (3-seed, val policy / test score, "
          f"eps=0.5)\n{'='*74}")
    for ds in ['aid', 'resisc45']:
        sa = load_grid(os.path.join(
            ROOT, f'eval/seed_results/valproto_joint_grid_{ds}.json'))
        ab_path = os.path.join(
            ROOT, f'eval/seed_results/adaptive1bit_joint_grid_{ds}.json')
        if not os.path.exists(ab_path):
            print(f'\n{ds}: adaptive1bit grid not ready yet'); continue
        ab = load_grid(ab_path)
        m_sa, m_ab = missions(sa), missions(ab)
        print(f"\n===== {ds.upper()} =====")
        print(f"{'policy':14} {'SpikeAdapt acc/pay':>26} "
              f"{'Adaptive-1bit acc/pay':>26}")
        for k in ['joint', 'rho_only', 'tprime_only', 'fixed_full']:
            a1, p1 = m_sa[k]; a2, p2 = m_ab[k]
            print(f"{k:14} {a1:8.2f} @ {p1*100:5.1f}%{'':>8} "
                  f"{a2:8.2f} @ {p2*100:5.1f}%")
        # headline: joint vs joint
        ja_sa, jp_sa = m_sa['joint']; ja_ab, jp_ab = m_ab['joint']
        print(f"  -> JOINT: SpikeAdapt {ja_sa:.2f}@{jp_sa*100:.1f}%  vs  "
              f"Adaptive-1bit {ja_ab:.2f}@{jp_ab*100:.1f}%  "
              f"(dAcc={ja_sa-ja_ab:+.2f}pp, dPay={ (jp_sa-jp_ab)*100:+.1f}pp)")


if __name__ == '__main__':
    main()
