#!/usr/bin/env python3
"""Closed-loop adaptation under a DYNAMIC (temporally correlated) A2G channel.

Hardens the trajectory result against the standard objection that the
ergodic simulation (eval_trajectory_closedloop.py) assumes i.i.d.
per-symbol fading and instantaneous, noiseless-side-information feedback.
This version adds, per mission step (= one image transmission):

  1. Correlated shadowing: log-normal, Gudmundson AR(1) model,
     sigma = 4 dB (suburban), decorrelation distance 100 m.
  2. LoS/NLoS state process: two-state Markov chain whose stationary
     distribution matches P_LoS(elevation) at each step, mean sojourn
     ~30 steps; excess path loss switches 0.1 <-> 21 dB with the state.
  3. Block fading: at 2.4 GHz and v = 15 m/s, Doppler ~120 Hz gives a
     coherence time of ~3.5 ms, comparable to one image payload
     (42 kbit / 10 MHz BPSK ~ 4.2 ms). Each image therefore sees a single
     Rician fade |h|^2 (K from elevation); the instantaneous hard-decision
     BER is Q(sqrt(2 * gamma_mean * |h|^2)).
  4. Stale feedback: the policy at step t uses the pilot estimate from
     step t-1 (one-block feedback delay), n_pilot = 256.

Accuracy is read from the same measured grids as the ergodic study:
joint (rho, T') grid (seed 42) and the BSC equivalence (Table I of the
paper) applied at the *instantaneous* BER.

Output: eval/seed_results/trajectory_dynamic.json

Usage:
    python eval/eval_trajectory_dynamic.py
"""

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from eval.channel_a2g import (ENVIRONMENTS, LINK, elevation_deg,
                              distance_3d_m, fspl_db, p_los, k_factor_db)
from eval.eval_trajectory_closedloop import (load_joint_options, joint_policy,
                                             joint_acc, T_FULL,
                                             D_MIN, D_MAX, N_STEPS, ALT_M, ENV)

SIGMA_SH_DB = 4.0          # shadowing std (suburban)
DECORR_M = 100.0           # Gudmundson decorrelation distance
MEAN_SOJOURN = 30          # mean LoS/NLoS state sojourn (steps)
N_PILOT = 256
N_RUNS = 100               # independent channel realizations
EPS_PP = 0.5


def q_func(x):
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def simulate_channel(rng):
    """One realization of the dynamic channel along the mission."""
    half = N_STEPS // 2
    d = np.concatenate([np.linspace(D_MIN, D_MAX, half),
                        np.linspace(D_MAX, D_MIN, half)])
    step_m = (D_MAX - D_MIN) / half
    a_sh = math.exp(-step_m / DECORR_M)          # AR(1) coefficient

    e = ENVIRONMENTS[ENV]
    noise_dbm = -174.0 + 10 * math.log10(LINK['bw_mhz'] * 1e6) + LINK['nf_db']

    sh = rng.normal(0, SIGMA_SH_DB)
    elev0 = elevation_deg(ALT_M, d[0])
    los_state = rng.random() < p_los(elev0, ENV)

    ber_inst = np.zeros(N_STEPS)
    for t in range(N_STEPS):
        elev = elevation_deg(ALT_M, d[t])
        # --- shadowing AR(1)
        sh = a_sh * sh + math.sqrt(1 - a_sh**2) * rng.normal(0, SIGMA_SH_DB)
        # --- LoS/NLoS Markov chain with stationary prob = p_los(elev)
        pl = p_los(elev, ENV)
        if los_state:
            p_leave = (1 - pl) / MEAN_SOJOURN
        else:
            p_leave = pl / MEAN_SOJOURN
        if rng.random() < p_leave:
            los_state = not los_state
        excess = e['eta_los'] if los_state else e['eta_nlos']
        # --- mean SNR with shadowing + LoS state
        pl_db = fspl_db(distance_3d_m(ALT_M, d[t]), LINK['fc_ghz']) + excess
        snr_mean_db = (LINK['ptx_dbm'] + LINK['gain_db'] - pl_db
                       - noise_dbm + sh)
        gamma = 10 ** (snr_mean_db / 10.0)
        # --- block Rician fade for this image; K conditioned on LoS state
        # (LoS: elevation-dependent K; NLoS: Rayleigh, K = 0 linear)
        K = 10 ** (k_factor_db(elev) / 10.0) if los_state else 0.0
        los_amp = math.sqrt(K / (K + 1.0))
        scale = math.sqrt(1.0 / (2.0 * (K + 1.0)))
        h2 = ((los_amp + scale * rng.normal())**2
              + (scale * rng.normal())**2)
        ber_inst[t] = q_func(math.sqrt(max(2.0 * gamma * h2, 0.0)))
    return d, ber_inst


def run(ds):
    options = load_joint_options(ds)
    opts_t8 = {k: v for k, v in options.items() if k[1] == T_FULL}
    rng = np.random.default_rng(11)

    agg = {name: dict(acc=[], pay=[]) for name in
           ['fixed_full', 'joint_stale', 'joint_stale_ucb', 'joint_genie',
            'rho_only_stale']}
    per_step = {name: [] for name in agg}   # per-image expected accuracies
    for run_i in range(N_RUNS):
        d, ber = simulate_channel(rng)
        # pilot estimates (current block) -> used at NEXT step (stale)
        est = rng.binomial(N_PILOT, np.clip(ber, 0, 1)) / N_PILOT
        est_stale = np.concatenate([[est[0]], est[:-1]])
        # robust variant: act on the max of the last two pilot estimates
        # (conservative hold-down against LoS->NLoS transitions)
        est_ucb = np.maximum(est_stale,
                             np.concatenate([[est_stale[0]], est_stale[:-1]]))

        env = ber <= 0.30 + 1e-9   # measured-grid envelope; scored steps only
        for name, opts, drive in [('joint_stale', options, est_stale),
                                  ('joint_stale_ucb', options, est_ucb),
                                  ('joint_genie', options, ber),
                                  ('rho_only_stale', opts_t8, est_stale)]:
            keys = [joint_policy(opts, p, EPS_PP) for p in drive]
            accs = np.array(
                [joint_acc(opts, k, p) for (k, _), p in zip(keys, ber)])
            per_step[name].append(accs[env])
            agg[name]['acc'].append(accs[env].mean())
            agg[name]['pay'].append(np.mean(
                [pl for (_, pl), e in zip(keys, env) if e]))
        accs_fx = np.array(
            [joint_acc(options, (1.0, T_FULL), p) for p in ber])
        per_step['fixed_full'].append(accs_fx[env])
        agg['fixed_full']['acc'].append(accs_fx[env].mean())
        agg['fixed_full']['pay'].append(1.0)
        agg.setdefault('_beyond', []).append(1.0 - env.mean())

    clean_acc = joint_acc(options, (1.0, T_FULL), 0.0)
    beyond = agg.pop('_beyond', [0.0])
    out = {'frac_beyond_envelope': float(np.mean(beyond))}
    for name, v in agg.items():
        a = np.concatenate(per_step[name])      # env-conditioned steps, all runs
        w10 = [np.lib.stride_tricks.sliding_window_view(x, 10).mean(-1)
               for x in per_step[name] if len(x) >= 10]
        w = np.concatenate(w10) if w10 else a[None]
        out[name] = dict(acc=float(np.mean(v['acc'])),
                         acc_std=float(np.std(v['acc'])),
                         payload=float(np.mean(v['pay'])),
                         tail=dict(
                             clean_ref=float(clean_acc),
                             frac_below_half_clean=float(
                                 (a < 0.5 * clean_acc).mean()),
                             min_image_acc=float(a.min()),
                             worst_window10_min=float(w.min())))
    return out


def main():
    results = {}
    for ds in ['aid', 'resisc45']:
        res = run(ds)
        results[ds] = res
        print(f'\n===== {ds.upper()} dynamic channel '
              f'({N_RUNS} realizations, eps={EPS_PP}) =====')
        print(f'{"policy":<18} {"acc":>14} {"payload":>8}')
        for name in ['fixed_full', 'rho_only_stale', 'joint_stale',
                     'joint_stale_ucb', 'joint_genie']:
            r = res[name]
            print(f'{name:<18} {r["acc"]:>7.2f}±{r["acc_std"]:<5.2f} '
                  f'{100*r["payload"]:>7.1f}%')
        js, fx = res['joint_stale'], res['fixed_full']
        print(f'joint(stale fb) vs fixed full: {js["acc"]-fx["acc"]:+.2f} pp '
              f'at {100*js["payload"]:.1f}% payload; '
              f'genie gap: {res["joint_genie"]["acc"]-js["acc"]:+.2f} pp')

    cfg = dict(sigma_sh_db=SIGMA_SH_DB, decorr_m=DECORR_M,
               mean_sojourn=MEAN_SOJOURN, n_pilot=N_PILOT,
               n_runs=N_RUNS, eps_pp=EPS_PP)
    with open('eval/seed_results/trajectory_dynamic.json', 'w') as f:
        json.dump(dict(config=cfg, results=results), f, indent=1)
    print('\nSaved: eval/seed_results/trajectory_dynamic.json')


if __name__ == '__main__':
    main()
