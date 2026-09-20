#!/usr/bin/env python3
"""Closed-loop adaptive-rho semantic communication over a UAV A2G trajectory.

System simulation (the ICC headline experiment):

  UAV mission: out-and-back survey at h=300 m (suburban A2G), ground
  distance 0.4 -> 2.6 km -> 0.4 km. At each step the physical channel gives
  SNR(d) and Rician K(elevation) -> true BER (eval/channel_a2g.py).

  Closed loop: the receiver estimates BER from n_pilot known pilot bits
  (binomial noise) and feeds it back; the transmitter selects the
  transmission rate rho from a policy table built on the 10-seed accuracy
  grid A(rho, BER) (eval/seed_results/per_rho_10seed.json):

      rho*(p) = min { rho : A(rho, p) >= max_rho' A(rho', p) - eps }

  i.e. the cheapest rate within eps pp of the best achievable accuracy.

  Baselines: fixed rho = 1.0 (no adaptation), fixed rho = 0.75 (paper),
  oracle (true BER, same policy), and genie-best-fixed-rho-per-mission.

  Metrics: mission-mean accuracy (10-seed mean grid, bilinear interp at
  true BER) and mission-mean bandwidth, averaged over pilot-noise draws.

Output: eval/seed_results/trajectory_closedloop.json
        paper/figures/fig_trajectory_closedloop.{pdf,png}

Usage:
    python eval/eval_trajectory_closedloop.py
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from eval.channel_a2g import (snr_db, k_factor_db, elevation_deg,
                              ber_bpsk_rician)

RHOS = np.array([0.10, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 1.0])
BERS = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
EPS_PP = 0.25          # accuracy tolerance for rate selection (pp)
N_PILOT = 256          # pilot bits per frame for BER estimation
N_DRAWS = 200          # pilot-noise realizations
ALT_M = 300.0
ENV = 'suburban'
D_MIN, D_MAX, N_STEPS = 400.0, 2600.0, 600


def load_grid(ds):
    """10-seed mean accuracy grid A[rho, ber]."""
    with open('eval/seed_results/per_rho_10seed.json') as f:
        d = json.load(f)[ds]
    seeds = list(d.keys())
    A = np.zeros((len(RHOS), len(BERS)))
    for i, r in enumerate(RHOS):
        for j, b in enumerate(BERS):
            rk = str(r) if str(r) in d[seeds[0]] else f'{r:g}'
            bk = str(b) if str(b) in d[seeds[0]][rk] else f'{b:g}'
            A[i, j] = np.mean([d[s][rk][bk] for s in seeds])
    return A


def interp_acc(A, rho, ber):
    """Bilinear interpolation of the accuracy grid at (rho, ber)."""
    ber = np.clip(ber, BERS[0], BERS[-1])
    rho = np.clip(rho, RHOS[0], RHOS[-1])
    j = np.searchsorted(BERS, ber) - 1
    j = np.clip(j, 0, len(BERS) - 2)
    i = np.searchsorted(RHOS, rho) - 1
    i = np.clip(i, 0, len(RHOS) - 2)
    tb = (ber - BERS[j]) / (BERS[j + 1] - BERS[j])
    tr = (rho - RHOS[i]) / (RHOS[i + 1] - RHOS[i])
    return ((1 - tr) * (1 - tb) * A[i, j] + tr * (1 - tb) * A[i + 1, j]
            + (1 - tr) * tb * A[i, j + 1] + tr * tb * A[i + 1, j + 1])


def policy_rho(A, ber, eps_pp=EPS_PP):
    """Cheapest rho within eps_pp of the best accuracy at this BER."""
    accs = np.array([interp_acc(A, r, ber) for r in RHOS])
    best = accs.max()
    ok = np.nonzero(accs >= best - eps_pp)[0]
    return RHOS[ok[0]]


def mission_profile():
    """Out-and-back ground distance profile + channel along it."""
    half = N_STEPS // 2
    d_out = np.linspace(D_MIN, D_MAX, half)
    d = np.concatenate([d_out, d_out[::-1]])
    snr = np.array([snr_db(ALT_M, di, ENV) for di in d])
    kdb = np.array([k_factor_db(elevation_deg(ALT_M, di)) for di in d])
    ber = np.array([ber_bpsk_rician(s, k) for s, k in zip(snr, kdb)])
    return d, snr, kdb, ber


# ------------------------------------------------ joint (rho, T') policy
T_FULL = 8
FULL_PAYLOAD_BITS = 1.0 * T_FULL * 36 * 14 * 14   # rho=1, T'=8 reference


def load_joint_options(ds, seed='42'):
    """Seed-42 accuracy tables A(ber) for each (rho, T') option.

    T' in {3..6} from joint_rho_T_grid.json; T'=8 from per_rho_10seed
    (same seed, same protocol). Returns {(rho, tp): (bers, accs)}."""
    path = 'eval/seed_results/joint_rho_T_grid.json'
    if not os.path.exists(path):
        return None
    with open(path) as f:
        grid = json.load(f).get(ds, {}).get(seed, {})
    with open('eval/seed_results/per_rho_10seed.json') as f:
        full = json.load(f)[ds][seed]

    options = {}
    for rho_s, tps in grid.items():
        for tp_s, bdict in tps.items():
            items = sorted(bdict.items(), key=lambda kv: float(kv[0]))
            options[(float(rho_s), int(tp_s))] = (
                np.array([float(b) for b, _ in items]),
                np.array([a for _, a in items]))
    for rho_s, bdict in full.items():
        if float(rho_s) < 0.6:
            continue
        items = sorted(bdict.items(), key=lambda kv: float(kv[0]))
        options[(float(rho_s), T_FULL)] = (
            np.array([float(b) for b, _ in items]),
            np.array([a for _, a in items]))
    return options


def joint_policy(options, ber, eps_pp=EPS_PP):
    """Cheapest (rho, T') within eps of the best accuracy at this BER.
    Returns ((rho, tp), payload_fraction)."""
    ber = float(np.clip(ber, 0.0, 0.30))
    # hard airtime budget: adaptive policies may only select candidates
    # within rho*T'/T <= 0.75 (42,336 uses); (1,8) stays available to
    # explicit unconstrained baselines via joint_acc.
    accs = {k: float(np.interp(ber, b, a)) for k, (b, a) in options.items()
            if k[0] * k[1] / T_FULL <= 0.75 + 1e-9}
    best = max(accs.values())
    feas = [(rho * tp / T_FULL, k) for k, v in accs.items()
            if v >= best - eps_pp for rho, tp in [k]]
    payload, key = min(feas)
    return key, payload


def joint_acc(options, key, ber):
    b, a = options[key]
    return float(np.interp(np.clip(ber, 0, 0.30), b, a))


# ------------------------------------------------ AMC separation baseline
def load_amc_schemes(ds):
    """Adaptive separation: JPEG + selectable LDPC rate (1/2 or 1/3).

    Returns list of (name, bers, accs, channel_bits)."""
    with open('eval/seed_results/jpeg_ldpc_results.json') as f:
        d = json.load(f)[ds]
    src_bits = d['mean_bits_q50']
    schemes = []
    for key, rate, name in [('r12_n648_dv3dc6_q50', 0.5, 'LDPC-1/2'),
                            ('r13_n648_dv4dc6_q50', 1 / 3, 'LDPC-1/3')]:
        sw = d['sweeps'][key]
        items = sorted(sw.items(), key=lambda kv: float(kv[0]))
        schemes.append((name,
                        np.array([float(b) for b, _ in items]),
                        np.array([a for _, a in items]),
                        src_bits / rate))
    return schemes


def amc_policy(schemes, ber):
    """Pick the scheme with the best expected accuracy at estimated BER;
    ties broken by fewer channel bits. Returns (idx, acc_fn ready)."""
    best = None
    for i, (name, b, a, bits) in enumerate(schemes):
        acc = float(np.interp(np.clip(ber, 0, 0.30), b, a))
        cand = (-acc, bits, i)
        if best is None or cand < best:
            best = cand
    return best[2]


def run(ds):
    A = load_grid(ds)
    d, snr, kdb, ber_true = mission_profile()
    rng = np.random.default_rng(7)

    res = {}
    # --- fixed-rho baselines (no estimation needed)
    for rho_fix in RHOS:
        acc = np.mean([interp_acc(A, rho_fix, b) for b in ber_true])
        res[f'fixed_{rho_fix:g}'] = dict(acc=float(acc), bw=float(rho_fix))
    # genie best fixed rho for this mission
    best_fix = max((res[f'fixed_{r:g}']['acc'], r) for r in RHOS)
    res['best_fixed'] = dict(acc=float(best_fix[0]), bw=float(best_fix[1]))

    # --- oracle adaptive (true BER)
    rho_oracle = np.array([policy_rho(A, b) for b in ber_true])
    acc_oracle = np.mean([interp_acc(A, r, b)
                          for r, b in zip(rho_oracle, ber_true)])
    res['oracle'] = dict(acc=float(acc_oracle), bw=float(rho_oracle.mean()))

    # --- closed loop with pilot-based estimation (N_DRAWS realizations),
    #     swept over the accuracy tolerance eps
    rho_example = None
    ber_est_example = None
    res['closed_loop_eps'] = {}
    for eps in [0.1, 0.25, 0.5, 1.0]:
        accs, bws = [], []
        for it in range(N_DRAWS):
            n_err = rng.binomial(N_PILOT, ber_true)
            ber_est = n_err / N_PILOT
            rho_cl = np.array([policy_rho(A, b, eps) for b in ber_est])
            accs.append(np.mean([interp_acc(A, r, b)
                                 for r, b in zip(rho_cl, ber_true)]))
            bws.append(rho_cl.mean())
            if it == 0 and eps == EPS_PP:
                rho_example = rho_cl
                ber_est_example = ber_est
        res['closed_loop_eps'][str(eps)] = dict(
            acc=float(np.mean(accs)), acc_std=float(np.std(accs)),
            bw=float(np.mean(bws)), n_pilot=N_PILOT)
    res['closed_loop'] = res['closed_loop_eps'][str(EPS_PP)]

    # --- AMC separation baseline (adaptive LDPC rate selection).
    # Fairness provisions: the policy may also declare OUTAGE (skip the
    # transmission entirely, spending zero payload, scoring chance accuracy)
    # instead of transmitting into a channel it knows is beyond the cliff.
    schemes = load_amc_schemes(ds)
    chance = {'aid': 100 / 30, 'resisc45': 100 / 45}[ds]
    amc_accs, amc_bits = [], []
    amc_outages = 0
    for it in range(20):   # deterministic given estimate; 20 draws suffice
        n_err = rng.binomial(N_PILOT, ber_true)
        ber_est = n_err / N_PILOT
        for p_hat, p_true in zip(ber_est, ber_true):
            i = amc_policy(schemes, p_hat)
            name, b, a, bits = schemes[i]
            exp_acc = float(np.interp(min(p_hat, 0.30), b, a))
            if exp_acc <= chance + 0.5:          # outage: skip, save payload
                amc_accs.append(chance)
                amc_bits.append(0.0)
                amc_outages += 1
            else:
                amc_accs.append(float(np.interp(min(p_true, 0.30), b, a)))
                amc_bits.append(bits)
    res['amc_separation'] = dict(
        acc=float(np.mean(amc_accs)),
        channel_bits=float(np.mean(amc_bits)),
        bits_vs_spikeadapt_full=float(np.mean(amc_bits) / FULL_PAYLOAD_BITS),
        outage_fraction=float(amc_outages / len(amc_accs)))

    # --- joint (rho, T') closed loop on the seed-42 grid
    options = load_joint_options(ds)
    joint_example = None
    if options is not None and len(options) > 4:
        # rho-only policy evaluated on the SAME seed-42 T'=8 tables,
        # for an internally consistent comparison
        opts_t8 = {k: v for k, v in options.items() if k[1] == T_FULL}
        res['joint_eps'] = {}
        for eps in [0.25, 0.5, 1.0]:
            j_accs, j_pay, r_accs, r_pay = [], [], [], []
            for it in range(N_DRAWS):
                n_err = rng.binomial(N_PILOT, ber_true)
                ber_est = n_err / N_PILOT
                keys = [joint_policy(options, p, eps) for p in ber_est]
                j_accs.append(np.mean([joint_acc(options, k, p)
                                       for (k, _), p in zip(keys, ber_true)]))
                j_pay.append(np.mean([pl for _, pl in keys]))
                keys8 = [joint_policy(opts_t8, p, eps) for p in ber_est]
                r_accs.append(np.mean([joint_acc(opts_t8, k, p)
                                       for (k, _), p in zip(keys8, ber_true)]))
                r_pay.append(np.mean([pl for _, pl in keys8]))
                if it == 0 and eps == 0.5:
                    joint_example = [
                        dict(rho=k[0], tp=k[1], payload=pl)
                        for (k, pl) in keys]
            res['joint_eps'][str(eps)] = dict(
                joint_acc=float(np.mean(j_accs)),
                joint_payload=float(np.mean(j_pay)),
                rho_only_acc=float(np.mean(r_accs)),
                rho_only_payload=float(np.mean(r_pay)))

    traj = dict(d_m=d.tolist(), snr_db=snr.tolist(), k_db=kdb.tolist(),
                ber_true=ber_true.tolist(),
                ber_est_example=ber_est_example.tolist(),
                rho_oracle=rho_oracle.tolist(),
                rho_closed_loop_example=rho_example.tolist(),
                joint_example=joint_example)
    return res, traj, A


def make_figure(all_res, all_traj):
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 4.6), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1, 1.25]})
    traj = all_traj['resisc45']   # RESISC45 policy actually varies rho
    t = np.arange(len(traj['d_m']))

    ax = axes[0]
    ax.plot(t, np.array(traj['d_m']) / 1000, color='#455a64', lw=2)
    ax.set_ylabel('Ground dist. (km)')
    ax2 = ax.twinx()
    ax2.plot(t, traj['snr_db'], color='#1565c0', lw=1.6, alpha=0.8)
    ax2.set_ylabel('SNR (dB)', color='#1565c0')
    ax2.tick_params(axis='y', colors='#1565c0')
    ax.set_title('UAV out-and-back mission, h=300 m, suburban A2G',
                 fontsize=11, fontweight='bold')

    ax = axes[1]
    ax.plot(t, traj['ber_true'], color='#b71c1c', lw=2, label='true BER')
    ax.plot(t, traj['ber_est_example'], color='#ef9a9a', lw=0.7,
            label=f'pilot estimate (n={N_PILOT})')
    ax.set_ylabel('BER')
    ax.legend(fontsize=8.5, loc='upper center')

    ax = axes[2]
    ax.step(t, traj['rho_closed_loop_example'], where='mid',
            color='#1b5e20', lw=1.6, alpha=0.8,
            label=r'$\rho$-only loop (payload $=\rho$)')
    if traj.get('joint_example'):
        payload = [e['payload'] for e in traj['joint_example']]
        ax.step(t, payload, where='mid', color='#e65100', lw=2.0,
                label=r"joint $(\rho, T')$ loop (payload $=\rho T'/8$)")
    ax.axhline(1.0, color='#9e9e9e', ls=':', lw=1.2,
               label=r'fixed $\rho{=}1.0$, $T{=}8$')
    ax.set_ylabel('Payload fraction')
    ax.set_xlabel('Mission time step')
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8.5, ncol=2, loc='lower center')

    for a in axes:
        a.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs('paper/figures', exist_ok=True)
    for ext in ['pdf', 'png']:
        fig.savefig(f'paper/figures/fig_trajectory_closedloop.{ext}',
                    dpi=300 if ext == 'pdf' else 200, bbox_inches='tight')
    plt.close()
    print('Saved: paper/figures/fig_trajectory_closedloop.{pdf,png}')


def main():
    all_res, all_traj = {}, {}
    for ds in ['aid', 'resisc45']:
        res, traj, A = run(ds)
        all_res[ds] = res
        all_traj[ds] = traj
        print(f'\n===== {ds.upper()} mission summary =====')
        print(f'{"policy":<18} {"mean acc":>9} {"mean BW":>8} {"BW saved":>9}')
        for name in ['fixed_1', 'fixed_0.75', 'best_fixed', 'oracle',
                     'closed_loop']:
            r = res[name]
            print(f'{name:<18} {r["acc"]:>8.2f}% {r["bw"]:>7.3f} '
                  f'{100*(1-r["bw"]):>8.1f}%')
        fx = res['fixed_1']
        print('eps sweep (closed loop vs fixed rho=1):')
        for eps, cl in res['closed_loop_eps'].items():
            print(f'  eps={eps:>4} pp: {cl["acc"]-fx["acc"]:+.2f} pp acc, '
                  f'{100*(1-cl["bw"]):.1f}% BW saved '
                  f'(std {cl["acc_std"]:.3f})')
        amc = res['amc_separation']
        print(f'AMC separation: acc={amc["acc"]:.2f}%, '
              f'{amc["bits_vs_spikeadapt_full"]:.1f}x SpikeAdapt full payload')
        if 'joint_eps' in res:
            print('joint (rho,T\') loop vs rho-only (seed-42 grid):')
            for eps, j in res['joint_eps'].items():
                print(f'  eps={eps:>4} pp: joint acc={j["joint_acc"]:.2f}% '
                      f'payload={100*j["joint_payload"]:.1f}% | rho-only '
                      f'acc={j["rho_only_acc"]:.2f}% '
                      f'payload={100*j["rho_only_payload"]:.1f}%')

    with open('eval/seed_results/trajectory_closedloop.json', 'w') as f:
        json.dump(dict(results=all_res, trajectories=all_traj), f)
    print('\nSaved: eval/seed_results/trajectory_closedloop.json')
    make_figure(all_res, all_traj)


if __name__ == '__main__':
    main()
