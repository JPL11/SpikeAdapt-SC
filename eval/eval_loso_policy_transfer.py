#!/usr/bin/env python3
"""Leave-one-seed-out (LOSO) policy transfer for the joint (rho,T') loop.

Addresses the circularity objection: in the original simulation, the policy
is selected on and scored against the same accuracy grid. Here, for each
ordered pair of seeds (i -> j), the policy table is built from seed i's
measured grid and the mission is SCORED on seed j's grid (an independently
trained model with its own data split). Reported: mean, std, and worst case
over the 6 transfer pairs, plus the matched (i -> i) reference.

Uses the dynamic-channel mission (correlated shadowing, LoS/NLoS Markov,
block fading, one-block-stale feedback, two-block hold-down) -- i.e. the
hardest scenario, not the ergodic one.

Output: eval/seed_results/loso_policy_transfer.json

Usage:
    python eval/eval_loso_policy_transfer.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from eval.eval_trajectory_closedloop import (load_joint_options, joint_policy,
                                             joint_acc, T_FULL)
from eval.eval_trajectory_dynamic import simulate_channel, N_PILOT

SEEDS = ['42', '123', '456']
EPS_PP = 0.5
N_RUNS = 100


def run_mission(policy_opts, score_opts, rng):
    """Mission with policy from policy_opts, scored on score_opts."""
    accs, pays = [], []
    for _ in range(N_RUNS):
        d, ber = simulate_channel(rng)
        est = rng.binomial(N_PILOT, np.clip(ber, 0, 1)) / N_PILOT
        est_stale = np.concatenate([[est[0]], est[:-1]])
        est_hold = np.maximum(est_stale,
                              np.concatenate([[est_stale[0]], est_stale[:-1]]))
        keys = [joint_policy(policy_opts, p, EPS_PP) for p in est_hold]
        accs.append(np.mean([joint_acc(score_opts, k, p)
                             for (k, _), p in zip(keys, ber)]))
        pays.append(np.mean([pl for _, pl in keys]))
    return float(np.mean(accs)), float(np.mean(pays))


def main():
    results = {}
    for ds in ['aid', 'resisc45']:
        opts = {s: load_joint_options(ds, seed=s) for s in SEEDS}
        missing = [s for s in SEEDS if opts[s] is None or len(opts[s]) < 20]
        if missing:
            print(f'{ds}: grids incomplete for seeds {missing} '
                  f'({[len(opts[s] or {}) for s in SEEDS]} options); abort')
            return

        rng = np.random.default_rng(23)
        matched, transfer, fixed_full, pooled = [], [], [], []
        pair_table = {}
        for i in SEEDS:
            for j in SEEDS:
                acc, pay = run_mission(opts[i], opts[j],
                                       np.random.default_rng(23))
                pair_table[f'{i}->{j}'] = dict(acc=round(acc, 2),
                                               payload=round(pay, 4))
                (matched if i == j else transfer).append((acc, pay, i, j))

        # conservative held-out pooling: policy from elementwise MIN of the
        # other seeds' grids, evaluated on the held-out seed
        for j in SEEDS:
            others = [s for s in SEEDS if s != j]
            pool = {}
            for k in opts[others[0]]:
                if all(k in opts[s] for s in others):
                    bers = opts[others[0]][k][0]
                    accs = np.minimum.reduce([opts[s][k][1] for s in others])
                    pool[k] = (bers, accs)
            acc, pay = run_mission(pool, opts[j], np.random.default_rng(23))
            pooled.append((acc, pay, j))
            pair_table[f'min(others)->{j}'] = dict(acc=round(acc, 2),
                                                   payload=round(pay, 4))
        for j in SEEDS:
            rngf = np.random.default_rng(23)
            accs = []
            for _ in range(N_RUNS):
                d, ber = simulate_channel(rngf)
                accs.append(np.mean([joint_acc(opts[j], (1.0, T_FULL), p)
                                     for p in ber]))
            fixed_full.append(float(np.mean(accs)))

        t_acc = [a for a, _, _, _ in transfer]
        t_pay = [p for _, p, _, _ in transfer]
        m_acc = [a for a, _, _, _ in matched]
        worst = min(transfer)
        p_acc = [a for a, _, _ in pooled]
        p_pay = [p for _, p, _ in pooled]
        res = dict(
            pairs=pair_table,
            matched_acc_mean=float(np.mean(m_acc)),
            transfer_acc_mean=float(np.mean(t_acc)),
            transfer_acc_std=float(np.std(t_acc, ddof=1)),
            transfer_payload_mean=float(np.mean(t_pay)),
            worst_pair=dict(acc=round(worst[0], 2), policy_seed=worst[2],
                            eval_seed=worst[3]),
            pooled_acc_mean=float(np.mean(p_acc)),
            pooled_acc_worst=float(np.min(p_acc)),
            pooled_payload_mean=float(np.mean(p_pay)),
            fixed_full_acc_mean=float(np.mean(fixed_full)),
        )
        results[ds] = res
        print(f'\n===== {ds.upper()} LOSO policy transfer '
              f'(dynamic channel, eps={EPS_PP}) =====')
        for k, v in pair_table.items():
            print(f'  {k}: acc={v["acc"]}%  payload={100*v["payload"]:.1f}%')
        print(f'matched (i->i):   {np.mean(m_acc):.2f}%')
        print(f'transfer (i!=j):  {np.mean(t_acc):.2f}% ± '
              f'{np.std(t_acc, ddof=1):.2f} (worst {worst[0]:.2f}%, '
              f'{worst[2]}->{worst[3]})')
        print(f'pooled-min held-out: {np.mean(p_acc):.2f}% '
              f'(worst {np.min(p_acc):.2f}%) at {100*np.mean(p_pay):.1f}% payload')
        print(f'fixed full-rate:  {np.mean(fixed_full):.2f}%')
        print(f'transfer payload: {100*np.mean(t_pay):.1f}%')

    with open('eval/seed_results/loso_policy_transfer.json', 'w') as f:
        json.dump(results, f, indent=1)
    print('\nSaved: eval/seed_results/loso_policy_transfer.json')


if __name__ == '__main__':
    main()
