#!/usr/bin/env python3
"""Path A stage 4b: mission suite under the validation protocol.

Reuses the ICC mission machinery (eval_trajectory_closedloop /
eval_trajectory_dynamic: A2G profile, pilot estimator, AMC baseline shape,
budget clamp, envelope conditioning) with the selection/reporting split:

  POLICY built from VAL-surface grids (valproto_joint_grid val split),
  ACCURACY scored on TEST-surface grids (same cells, test split).

Missions:
  static  — ergodic trajectory, per-BER expected accuracy (as Table trajectory)
  loso    — policy from two seeds' VAL grids, scored on held-out seed's TEST grid
  dynamic — correlated channel w/ stale feedback (imports simulate_channel,
            post-Rayleigh-fix), envelope-conditioned, hold-down variant

Requires valproto_joint_grid_{ds}.json (stage 4a) for seeds 42/123/456.
Usage: python eval/valproto_missions.py
Output: eval/seed_results/valproto_missions.json
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'eval'))
from eval.eval_trajectory_closedloop import (mission_profile,  # noqa: E402
                                             T_FULL, N_PILOT)
EPS_PP = 0.5   # paper's ex-ante headline epsilon
from eval.eval_trajectory_dynamic import simulate_channel              # noqa: E402

SEEDS = ['42', '123', '456']
RHOS = [0.625, 0.75, 0.875, 1.0]
TPRIMES = [3, 4, 5, 6, 8]
BERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
BUDGET = 0.75          # hard airtime cap (rho*T'/T <= 0.75)
N_DRAWS = 200
N_RUNS_DYN = 100
OUT = os.path.join(ROOT, 'eval/seed_results/valproto_missions.json')


def load_grids(ds):
    g = json.load(open(os.path.join(
        ROOT, f'eval/seed_results/valproto_joint_grid_{ds}.json')))
    # tables[seed][split][(rho,tp)] = (bers_array, accs_array)
    tables = {}
    for s in SEEDS:
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


def acc_at(table, key, ber, ber_max=0.30):
    b, a = table[key]
    return float(np.interp(np.clip(ber, 0, ber_max), b, a))


def policy_from(table, ber, eps=EPS_PP):
    """Cheapest budget-feasible (rho,T') within eps of best (on THIS table)."""
    ber = float(np.clip(ber, 0.0, 0.30))
    accs = {k: acc_at(table, k, ber) for k in table
            if k[0] * k[1] / T_FULL <= BUDGET + 1e-9}
    best = max(accs.values())
    feas = [(k[0] * k[1] / T_FULL, k) for k, v in accs.items()
            if v >= best - eps]
    pay, key = min(feas)
    return key, pay


def static_mission(pol_table, score_table, rng):
    """Ergodic trajectory with pilot-noise MC; policy on pol_table,
    accuracy on score_table."""
    _, _, _, bers = mission_profile()
    accs, pays = [], []
    for p_true in bers:
        a_d, p_d = [], []
        for _ in range(N_DRAWS):
            p_hat = rng.binomial(N_PILOT, min(max(p_true, 0), 1)) / N_PILOT
            key, pay = policy_from(pol_table, p_hat)
            a_d.append(acc_at(score_table, key, p_true))
            p_d.append(pay)
        accs.append(np.mean(a_d)); pays.append(np.mean(p_d))
    fixed = [acc_at(score_table, (1.0, T_FULL), p) for p in bers]
    return dict(acc=round(float(np.mean(accs)), 2),
                payload=round(float(np.mean(pays)), 4),
                fixed_acc=round(float(np.mean(fixed)), 2))


def dynamic_mission(pol_table, score_table, rng):
    """Correlated channel, stale feedback + hold-down. FULL-SLOT scoring on
    the extended test surface (measured to p=0.50; policy selection still
    indexes the p<=0.30 table). Also reports the p<=0.30-conditional view."""
    accs, pays, fixed, cond, beyond40 = [], [], [], [], []
    all_steps, all_fixed = [], []
    for _ in range(N_RUNS_DYN):
        _, ber = simulate_channel(rng)
        est = rng.binomial(N_PILOT, np.clip(ber, 0, 1)) / N_PILOT
        est_stale = np.concatenate([[est[0]], est[:-1]])
        est_hold = np.maximum(est_stale,
                              np.concatenate([[est_stale[0]], est_stale[:-1]]))
        keys = [policy_from(pol_table, p) for p in est_hold]
        a = np.array([acc_at(score_table, k, p, ber_max=0.50)
                      for (k, _), p in zip(keys, ber)])
        f = np.array([acc_at(score_table, (1.0, T_FULL), p, ber_max=0.50)
                      for p in ber])
        env = ber <= 0.30 + 1e-9
        accs.append(a.mean())
        pays.append(np.mean([pl for _, pl in keys]))
        fixed.append(f.mean())
        cond.append(a[env].mean())
        beyond40.append((ber > 0.40).mean())
        all_steps.append(a); all_fixed.append(f)
    a_cat = np.concatenate(all_steps); f_cat = np.concatenate(all_fixed)
    clean = acc_at(score_table, (1.0, T_FULL), 0.0)
    return dict(acc=round(float(np.mean(accs)), 2),
                acc_std=round(float(np.std(accs)), 2),
                payload=round(float(np.mean(pays)), 4),
                fixed_acc=round(float(np.mean(fixed)), 2),
                cond_acc_le030=round(float(np.mean(cond)), 2),
                frac_beyond_train_envelope=round(float(np.mean(beyond40)), 4),
                tail=dict(frac_below_half_clean=round(float(
                              (a_cat < 0.5 * clean).mean()), 4),
                          min_step_acc=round(float(a_cat.min()), 2),
                          fixed_min=round(float(f_cat.min()), 2)))


def main():
    out = {'protocol': 'policy on VAL surface, accuracy on TEST surface; '
                       'budget rho*T\'/T<=0.75; pilot 256b; envelope p<=0.30',
           'eps_pp': EPS_PP}
    for ds in ['aid', 'resisc45']:
        tables = load_grids(ds)
        rng = np.random.default_rng(7)
        node = {}
        # static, matched (per seed: own val policy, own test scores)
        st = [static_mission(tables[s]['val'], tables[s]['test'], rng)
              for s in SEEDS]
        node['static'] = dict(
            acc=round(float(np.mean([x['acc'] for x in st])), 2),
            acc_std=round(float(np.std([x['acc'] for x in st], ddof=1)), 2),
            payload=round(float(np.mean([x['payload'] for x in st])), 4),
            fixed_acc=round(float(np.mean([x['fixed_acc'] for x in st])), 2))
        # LOSO: pooled-min val policy from two seeds, held-out seed's test grid
        lo = []
        for held in SEEDS:
            others = [s for s in SEEDS if s != held]
            pooled = {}
            for k in tables[others[0]]['val']:
                b0, a0 = tables[others[0]]['val'][k]
                _, a1 = tables[others[1]]['val'][k]
                pooled[k] = (b0, np.minimum(a0, a1))
            lo.append(static_mission(pooled, tables[held]['test'], rng))
        node['loso'] = dict(
            acc=round(float(np.mean([x['acc'] for x in lo])), 2),
            acc_worst=round(float(min(x['acc'] for x in lo)), 2),
            payload=round(float(np.mean([x['payload'] for x in lo])), 4),
            fixed_acc=round(float(np.mean([x['fixed_acc'] for x in lo])), 2))
        # dynamic, matched per seed
        dy = [dynamic_mission(tables[s]['val'], tables[s]['test'], rng)
              for s in SEEDS]
        node['dynamic'] = dict(
            acc=round(float(np.mean([x['acc'] for x in dy])), 2),
            payload=round(float(np.mean([x['payload'] for x in dy])), 4),
            fixed_acc=round(float(np.mean([x['fixed_acc'] for x in dy])), 2),
            cond_acc_le030=round(float(np.mean(
                [x['cond_acc_le030'] for x in dy])), 2),
            frac_beyond_train_envelope=dy[0]['frac_beyond_train_envelope'],
            tail=dict(
                frac_below_half_clean=round(float(np.mean(
                    [x['tail']['frac_below_half_clean'] for x in dy])), 4),
                min_step_acc=round(float(min(
                    x['tail']['min_step_acc'] for x in dy)), 2),
                fixed_min=round(float(min(
                    x['tail']['fixed_min'] for x in dy)), 2)))
        out[ds] = node
        print(ds, json.dumps(node, indent=1), flush=True)
    json.dump(out, open(OUT, 'w'), indent=1)
    print('saved:', OUT)


if __name__ == '__main__':
    main()
