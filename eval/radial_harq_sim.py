#!/usr/bin/env python3
"""Task-aware incremental-T' HARQ simulation (journal deferred track).

Reads per-frame stats from radial_harq_persample_s<seed>.npz. HARQ rides the
TEMPORAL axis (rho=1): each frame starts at T0 timesteps and the transmitter
sends one more timestep whenever the RX-side decodability proxy is below a
NACK threshold tau, up to T=8. The frame's detections are then those decoded
at its final T'. Mission F1 (the meaningful metric here -- RADIal "mAP" is a
non-monotonic mean-precision that a few confident detections game) is composed
from the per-frame TP/FP/FN at each frame's final T'.

Quality metric: F1 = 2P.R/(P+R), P,R = mean-over-threshold precision/recall.
F1 PEAKS at intermediate T' (recall rises with T', precision falls as FPs
accumulate), so HARQ genuinely has an optimum to adapt toward.

Leakage-free: tau is calibrated on VAL, every reported number is on TEST.
Strawman: bit-exact CRC-HARQ NACKs on ANY payload bit error -> at BER>=0.1
its per-frame success prob (1-BER)^payload is ~0, so it escalates to T=8 for
(almost) every frame => no payload saving.

Usage: python eval/radial_harq_sim.py [--seed 42] [--t0 3] [--eps 0.01]
Output: eval/seed_results/radial_harq_sim.json
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(ROOT, 'eval/seed_results/radial_harq_persample_s{seed}.npz')
OUT = os.path.join(ROOT, 'eval/seed_results/radial_harq_sim.json')
TPRIMES = [1, 2, 3, 4, 5, 6, 7, 8]
C_SPIKE = 36
DIMS = {'x2': (128, 64), 'x3': (64, 32), 'x4': (32, 16)}
FULL_SPATIAL = sum(C_SPIKE * h * w for h, w in DIMS.values())   # bits at T'=1


def load(seed):
    d = dict(np.load(NPZ.format(seed=seed), allow_pickle=True))
    return d


def metrics_from(TP, FP, FN):
    """mAP, mAR, F1 from [F,9] per-threshold counts summed over frames."""
    P, R = [], []
    for j in range(TP.shape[1]):
        tp, fp, fn = TP[:, j].sum(), FP[:, j].sum(), FN[:, j].sum()
        P.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        R.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    mAP, mAR = float(np.mean(P)), float(np.mean(R))
    f1 = 2 * mAP * mAR / (mAP + mAR) if (mAP + mAR) > 0 else 0.0
    return dict(mAP=round(mAP, 4), mAR=round(mAR, 4), F1=round(f1, 4))


def stack(d, split, ber, arr):
    """[F, nT] proxy or [F, nT, 9] counts across T' for one (split,ber)."""
    return np.stack([d[f'{split}|{ber}|{tp}|{arr}'] for tp in TPRIMES],
                    axis=1)


def harq_final_tp(proxy_fT, tau, t0):
    """Per-frame final T' index: smallest T'>=t0 with proxy>=tau, else 8."""
    F, nT = proxy_fT.shape
    i0 = TPRIMES.index(t0)
    out = np.full(F, nT - 1, int)
    for f in range(F):
        for i in range(i0, nT):
            if proxy_fT[f, i] >= tau:
                out[f] = i
                break
    return out


def compose(TP_fT, FP_fT, FN_fT, final_idx):
    F = len(final_idx)
    TP = np.stack([TP_fT[f, final_idx[f]] for f in range(F)])
    FP = np.stack([FP_fT[f, final_idx[f]] for f in range(F)])
    FN = np.stack([FN_fT[f, final_idx[f]] for f in range(F)])
    m = metrics_from(TP, FP, FN)
    m['avg_T'] = round(float(np.mean([TPRIMES[i] for i in final_idx])), 3)
    return m


def proxy_auc(proxy_f, TP_f, FP_f, FN_f, thr_idx=4):
    """AUC of proxy vs per-frame 'well-decoded' (frame F1>=0.5 at ref thr)."""
    q = 2 * TP_f[:, thr_idx] / np.maximum(
        2 * TP_f[:, thr_idx] + FP_f[:, thr_idx] + FN_f[:, thr_idx], 1e-9)
    good = q >= 0.5
    if good.all() or (~good).all():
        return None
    order = np.argsort(proxy_f)
    ranks = np.empty_like(order, float)
    ranks[order] = np.arange(1, len(proxy_f) + 1)
    n1 = good.sum()
    n0 = len(good) - n1
    auc = (ranks[good].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return round(float(auc), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', default='42')
    ap.add_argument('--t0', type=int, default=3)
    ap.add_argument('--eps', type=float, default=0.01)  # F1 tolerance
    ap.add_argument('--proxy', default='pmax',
                    choices=['pmax', 'pmean3', 'pndet'])
    args = ap.parse_args()
    d = load(args.seed)
    out = {'params': vars(args),
           'note': 'quality=F1; tau on VAL, scored on TEST; HARQ=incremental-T'
           " on temporal axis (rho=1)"}

    for ber in [0.0, 0.1, 0.2, 0.3]:
        rec = {}
        # fixed-T' Pareto frontier (per split)
        for split in ['val', 'test']:
            fx = {}
            for tp in TPRIMES:
                m = metrics_from(d[f'{split}|{ber}|{tp}|TP'],
                                 d[f'{split}|{ber}|{tp}|FP'],
                                 d[f'{split}|{ber}|{tp}|FN'])
                m['T'] = tp
                fx[tp] = m
            rec[f'fixed_{split}'] = fx
        # proxy AUC (val, start T0) — does the NACK signal separate?
        i0 = TPRIMES.index(args.t0)
        rec['proxy_auc_val'] = proxy_auc(
            stack(d, 'val', ber, args.proxy)[:, i0],
            d[f'val|{ber}|{args.t0}|TP'], d[f'val|{ber}|{args.t0}|FP'],
            d[f'val|{ber}|{args.t0}|FN'])
        # calibrate tau on VAL: smallest avg_T' with F1 >= best_val_F1 - eps
        pv = stack(d, 'val', ber, args.proxy)
        TPv, FPv, FNv = (stack(d, 'val', ber, a) for a in ['TP', 'FP', 'FN'])
        best_f1 = max(rec['fixed_val'][tp]['F1'] for tp in TPRIMES)
        cand = []
        taus = np.round(np.linspace(0.0, 1.0, 51), 3) if args.proxy != 'pndet' \
            else np.arange(0, 20)
        for tau in taus:
            fi = harq_final_tp(pv, tau, args.t0)
            m = compose(TPv, FPv, FNv, fi)
            cand.append((tau, m['F1'], m['avg_T']))
        ok = [c for c in cand if c[1] >= best_f1 - args.eps]
        tau_star = min(ok, key=lambda c: c[2])[0] if ok else \
            max(cand, key=lambda c: c[1])[0]
        # evaluate tau* on TEST
        pt = stack(d, 'test', ber, args.proxy)
        TPt, FPt, FNt = (stack(d, 'test', ber, a) for a in ['TP', 'FP', 'FN'])
        fi = harq_final_tp(pt, tau_star, args.t0)
        harq = compose(TPt, FPt, FNt, fi)
        harq['tau'] = float(tau_star)
        rec['harq_test'] = harq
        # best fixed-T' at matched-or-lower avg payload (test)
        bf = [rec['fixed_test'][tp] for tp in TPRIMES if tp <= harq['avg_T']]
        rec['best_fixed_le_payload'] = max(bf, key=lambda m: m['F1']) if bf \
            else rec['fixed_test'][TPRIMES[0]]
        # CRC strawman: any payload bit error -> escalate; success prob ~0 for
        # ber>0, payload huge -> avg_T' = 8 (max), F1 = F1 at T'=8.
        p_succ_perT = [(1 - ber) ** (FULL_SPATIAL * tp) for tp in TPRIMES]
        rec['crc_strawman'] = dict(
            avg_T=8.0 if ber > 0 else float(args.t0),
            F1=rec['fixed_test'][8]['F1'] if ber > 0
            else rec['fixed_test'][args.t0]['F1'],
            per_frame_success_prob_at_T8=float(p_succ_perT[-1]))
        out[f'ber{ber}'] = rec
        h, bfd = rec['harq_test'], rec['best_fixed_le_payload']
        print(f"BER {ber}: HARQ F1 {h['F1']} @avgT {h['avg_T']} (tau {h['tau']}) "
              f"| best-fixed<=payload F1 {bfd['F1']} @T{bfd['T']} "
              f"| CRC avgT {rec['crc_strawman']['avg_T']} "
              f"| AUC {rec['proxy_auc_val']}", flush=True)
    json.dump(out, open(OUT, 'w'), indent=1)
    print('saved:', OUT, flush=True)


if __name__ == '__main__':
    main()
