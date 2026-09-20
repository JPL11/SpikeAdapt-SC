#!/usr/bin/env python3
"""Leakage-free closed-loop mission for KITTI image RECONSTRUCTION
(cross-modality deferred-track result). Mirrors eval/radial_closedloop_valtest.py
but the metric is PSNR (higher is better), single-metric, one trained codec.

Reads recon_grid_valtest.json [seed][val|test][cell]; the (CBR,T') policy is
built on the VAL surface and every PSNR is scored on the TEST surface -- this
is the leakage fix: the codec checkpoint was best-on-InStereo2K-test, but the
RATE-ADAPTATION POLICY is now val-selected / test-scored, and KITTI is a pure
transfer set the codec never trained on. A2G h=100 m suburban; pilot-noise MC
(256-bit, no oracle); envelope truncation at p<=0.40 (disclosed).

Usage: python eval/recon_closedloop_valtest.py [--eps 0.3] [--pilot 256] [--mc 200]
Output: eval/seed_results/recon_closedloop_valtest.json
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eval.channel_a2g import distance_profile  # noqa: E402

GRID = os.path.join(ROOT, 'eval/seed_results/recon_grid_valtest.json')
OUT = os.path.join(ROOT, 'eval/seed_results/recon_closedloop_valtest.json')
SEED = 'v6'
CBRS = [0.125, 0.25, 0.5, 0.75, 1.0]
TPRIMES = [1, 2, 4, 8]
BERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
BER_MAX = 0.40
T_FULL = 8


def build_policy(gv, eps, mode='joint'):
    """Policy per BER from VAL PSNR; min payload within eps (dB) of best.
    mode: joint | ronly (T'=8) | tonly (CBR=1.0)."""
    policy = {}
    for ber in BERS:
        cells = []
        for cbr in CBRS:
            for tp in TPRIMES:
                if mode == 'ronly' and tp != T_FULL:
                    continue
                if mode == 'tonly' and cbr != 1.0:
                    continue
                c = gv[f'cbr{cbr}_T{tp}_ber{ber}']
                cells.append((cbr, tp, c['psnr'], c['payload_bits']))
        best = max(c[2] for c in cells)
        ok = [c for c in cells if c[2] >= best - eps]
        cbr, tp, _, pay = min(ok, key=lambda c: c[3])
        policy[ber] = dict(cbr=cbr, T=tp, payload_bits=pay)
    return policy


def run_mission(policy, sg, prof, pilot_n, mc, rng):
    s_ad, pay_ad, s_fx = [], [], []
    clamped = 0
    full_pay = sg[f'cbr1.0_T{T_FULL}_ber{BERS[0]}']['payload_bits']
    for r in prof:
        p_true = r['ber']
        if p_true > BER_MAX:
            clamped += 1
            continue
        near = min(BERS, key=lambda b: abs(b - p_true))
        s_fx.append(sg[f'cbr1.0_T{T_FULL}_ber{near}']['psnr'])
        psnrs, pays = [], []
        for _ in range(mc):
            p_hat = rng.binomial(pilot_n, p_true) / pilot_n
            pol = policy[min(BERS, key=lambda b: abs(b - p_hat))]
            cell = sg[f"cbr{pol['cbr']}_T{pol['T']}_ber{near}"]
            psnrs.append(cell['psnr'])
            pays.append(pol['payload_bits'])
        s_ad.append(np.mean(psnrs))
        pay_ad.append(np.mean(pays))
    return dict(psnr_adaptive=round(float(np.mean(s_ad)), 4),
                psnr_fixed=round(float(np.mean(s_fx)), 4),
                payload_saving=round(1 - np.mean(pay_ad) / full_pay, 4),
                waypoints_used=len(s_ad), waypoints_truncated=clamped)


def best_fixed(sg, prof, max_frac):
    """Best fixed (CBR,T') with payload fraction <= max_frac, by mission PSNR."""
    full_pay = sg[f'cbr1.0_T{T_FULL}_ber{BERS[0]}']['payload_bits']
    best = None
    for cbr in CBRS:
        for tp in TPRIMES:
            frac = sg[f'cbr{cbr}_T{tp}_ber{BERS[0]}']['payload_bits'] / full_pay
            if frac > max_frac + 1e-9:
                continue
            accs = [sg[f'cbr{cbr}_T{tp}_ber'
                       f'{min(BERS, key=lambda b: abs(b - r["ber"]))}']['psnr']
                    for r in prof if r['ber'] <= BER_MAX]
            m = float(np.mean(accs))
            if best is None or m > best[2]:
                best = (cbr, tp, m, round(frac, 4))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eps', type=float, default=0.3)
    ap.add_argument('--pilot', type=int, default=256)
    ap.add_argument('--mc', type=int, default=200)
    ap.add_argument('--h', type=float, default=100.0)
    ap.add_argument('--grid', default=GRID)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--tag', default='KITTI (transfer)')
    args = ap.parse_args()
    g = json.load(open(args.grid))[SEED]
    rng = np.random.default_rng(3)
    prof = [r for r in distance_profile(args.h, 'suburban')
            if 200 <= r['d_ground_m'] <= 3000]

    out = {'params': vars(args),
           'task': f'image reconstruction (PSNR dB) — {args.tag}',
           'protocol': 'policy on VAL surface, PSNR on TEST surface; '
           'A2G h=100m suburban; pilot-noise MC; envelope p<=0.40'}
    for mode in ['joint', 'ronly', 'tonly']:
        pol = build_policy(g['val'], args.eps, mode)
        out[mode] = run_mission(pol, g['test'], prof, args.pilot, args.mc, rng)
    jfrac = 1 - out['joint']['payload_saving']
    bf = best_fixed(g['test'], prof, jfrac)
    out['best_fixed_below'] = dict(psnr=round(bf[2], 4), payload_frac=bf[3],
                                   cell=f'cbr{bf[0]}_T{bf[1]}')
    json.dump(out, open(args.out, 'w'), indent=1)
    for k in ['joint', 'ronly', 'tonly']:
        print(f"{k}: psnr {out[k]['psnr_adaptive']} vs fixed "
              f"{out[k]['psnr_fixed']}, saving {out[k]['payload_saving']}")
    print('best fixed <= joint payload:', out['best_fixed_below'])
    print('saved:', args.out)


if __name__ == '__main__':
    main()
