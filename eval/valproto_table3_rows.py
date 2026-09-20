#!/usr/bin/env python3
"""Panel fix F1/F6/F7: committed generator for every Table III row and the
dynamic-channel prose numbers of main_icc.tex.

All rows: 3-seed (42/123/456) means; policies built on the VAL surface of
valproto_joint_grid_{ds}.json, accuracies scored on the TEST surface;
budget rho*T'/T <= 0.75; eps in pp; 200 binomial pilot draws (n=256) per
mission step on the ergodic profile. Dynamic rows: 100 correlated-channel
realizations per seed on a SINGLE matched stream per seed (same realizations
for every policy — F7), full-slot scoring on the grid measured to p=0.50,
two-block hold-down; genie variant = current-block true BER (F6).

Usage: python eval/valproto_table3_rows.py
Output: eval/seed_results/valproto_table3_rows.json
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'eval'))
from eval.valproto_missions import load_grids, acc_at, T_FULL, BUDGET, N_PILOT  # noqa: E402
from eval.eval_trajectory_closedloop import mission_profile                     # noqa: E402
from eval.eval_trajectory_dynamic import simulate_channel                       # noqa: E402

SEEDS = ['42', '123', '456']
OUT = os.path.join(ROOT, 'eval/seed_results/valproto_table3_rows.json')
_, _, _, MISSION_BERS = mission_profile()


def policy(table, p, eps, mode='joint'):
    p = float(np.clip(p, 0, 0.30))
    ks = [k for k in table if k[0] * k[1] / T_FULL <= BUDGET + 1e-9]
    if mode == 'tonly':
        ks = [k for k in ks if k[0] == 1.0]
    elif mode == 'ronly':
        ks = [k for k in ks if k[1] == T_FULL]
    accs = {k: acc_at(table, k, p) for k in ks}
    best = max(accs.values())
    return min((k[0] * k[1] / T_FULL, k) for k, v in accs.items()
               if v >= best - eps)


def static_mission(pol_table, score_table, rng, eps, mode='joint',
                   oracle=False):
    accs, pays = [], []
    for p_true in MISSION_BERS:
        a_d, p_d = [], []
        for _ in range(1 if oracle else 200):
            p_hat = p_true if oracle else \
                rng.binomial(N_PILOT, min(max(p_true, 0), 1)) / N_PILOT
            pay, key = policy(pol_table, p_hat, eps, mode)
            a_d.append(acc_at(score_table, key, p_true))
            p_d.append(pay)
        accs.append(np.mean(a_d)); pays.append(np.mean(p_d))
    return float(np.mean(accs)), float(np.mean(pays))


def fixed_mission(score_table, key):
    return float(np.mean([acc_at(score_table, key, p)
                          for p in MISSION_BERS])), key[0] * key[1] / T_FULL


def dynamic_matched(T, seed, eps):
    """One matched realization stream per seed; every policy variant scored
    on the SAME realizations (F7). Full-slot scoring to p=0.50."""
    rng = np.random.default_rng(int(seed))
    streams = []
    for _ in range(100):
        _, ber = simulate_channel(rng)
        est = rng.binomial(N_PILOT, np.clip(ber, 0, 1)) / N_PILOT
        est_stale = np.concatenate([[est[0]], est[:-1]])
        est_hold = np.maximum(est_stale,
                              np.concatenate([[est_stale[0]], est_stale[:-1]]))
        streams.append((ber, est_stale, est_hold))
    out = {}
    for name, drive_idx, mode in [('joint_hold', 2, 'joint'),
                                  ('joint_stale', 1, 'joint'),
                                  ('joint_genie', 0, 'joint'),
                                  ('tonly_hold', 2, 'tonly')]:
        accs, pays = [], []
        for ber, est_stale, est_hold in streams:
            drive = (ber, est_stale, est_hold)[drive_idx]
            keys = [policy(T[seed]['val'], p, eps, mode) for p in drive]
            a = np.mean([acc_at(T[seed]['test'], k, p, ber_max=0.50)
                         for (pl, k), p in zip(keys, ber)])
            accs.append(a); pays.append(np.mean([pl for pl, _ in keys]))
        out[name] = dict(acc=round(float(np.mean(accs)), 2),
                         payload=round(float(np.mean(pays)), 4))
    accs = [np.mean([acc_at(T[seed]['test'], (1.0, T_FULL), p, ber_max=0.50)
                     for p in ber]) for ber, _, _ in streams]
    out['fixed_full'] = dict(acc=round(float(np.mean(accs)), 2), payload=1.0)
    # conditional (p<=0.30) views on the same streams
    for name, mode, di in [('joint_hold_cond', 'joint', 2),
                           ('joint_genie_cond', 'joint', 0)]:
        accs = []
        for ber, est_stale, est_hold in streams:
            drive = (ber, est_stale, est_hold)[di]
            keys = [policy(T[seed]['val'], p, eps, mode) for p in drive]
            env = ber <= 0.30 + 1e-9
            a = np.array([acc_at(T[seed]['test'], k, p)
                          for (pl, k), p in zip(keys, ber)])
            accs.append(a[env].mean())
        out[name] = round(float(np.mean(accs)), 2)
    accs = []
    for ber, _, _ in streams:
        env = ber <= 0.30 + 1e-9
        a = np.array([acc_at(T[seed]['test'], (1.0, T_FULL), p) for p in ber])
        accs.append(a[env].mean())
    out['fixed_full_cond'] = round(float(np.mean(accs)), 2)
    return out


def main():
    res = {'protocol': '3-seed means (42/123/456); VAL policy, TEST scoring; '
                       'budget<=0.75; 200 pilot draws; dynamic: matched '
                       'streams, full-slot to p=0.50, hold-down'}
    for ds in ['aid', 'resisc45']:
        T = load_grids(ds)
        node = {}
        # fixed rows
        for label, key in [('fixed_full', (1.0, 8)), ('fixed_r075', (0.75, 8)),
                           ('bracket_below', (0.625, 6)),
                           ('bracket_above', (0.75, 6))]:
            r = [fixed_mission(T[s]['test'], key) for s in SEEDS]
            node[label] = dict(acc=round(float(np.mean([x[0] for x in r])), 2),
                               payload=round(r[0][1], 4))
        # adaptive rows
        for label, eps, mode in [('rho_only_e05', 0.5, 'ronly'),
                                 ('tprime_only_e05', 0.5, 'tonly'),
                                 ('joint_e05', 0.5, 'joint'),
                                 ('joint_e10', 1.0, 'joint')]:
            rng = np.random.default_rng(7)
            r = [static_mission(T[s]['val'], T[s]['test'], rng, eps, mode)
                 for s in SEEDS]
            node[label] = dict(acc=round(float(np.mean([x[0] for x in r])), 2),
                               payload=round(float(np.mean([x[1] for x in r])), 4))
        # oracle vs pilot payload gap (F10)
        rng = np.random.default_rng(7)
        r_p = [static_mission(T[s]['val'], T[s]['test'], rng, 0.5, 'joint')
               for s in SEEDS]
        r_o = [static_mission(T[s]['val'], T[s]['test'], rng, 0.5, 'joint',
                              oracle=True) for s in SEEDS]
        node['oracle_gap_payload_pp'] = round(100 * abs(
            float(np.mean([x[1] for x in r_p]))
            - float(np.mean([x[1] for x in r_o]))), 2)
        # dynamic (matched streams per seed)
        dyn = [dynamic_matched(T, s, 0.5) for s in SEEDS]
        node['dynamic'] = {k: dict(
            acc=round(float(np.mean([d[k]['acc'] for d in dyn])), 2),
            payload=round(float(np.mean([d[k]['payload'] for d in dyn])), 4))
            if isinstance(dyn[0][k], dict) else
            round(float(np.mean([d[k] for d in dyn])), 2)
            for k in dyn[0]}
        res[ds] = node
        print(ds, json.dumps(node, indent=1), flush=True)
    json.dump(res, open(OUT, 'w'), indent=1)
    print('saved:', OUT)


if __name__ == '__main__':
    main()
