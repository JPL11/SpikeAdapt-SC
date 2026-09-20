#!/usr/bin/env python3
"""Hybrid calibration-resample check: is the certified-CSI hybrid's contract
violation (v1@0.15, v2@0.05) stable across calibration draws, or single-draw
luck? Subsample 80% of calibration points 12x, rebuild per-(tier, BER-cell)
quantiles, replay the escalation on every seed matrix."""
import json

import numpy as np

DELTA = 0.15
RHOS = [0.25, 0.5, 0.75, 1.0]
FULL = 3
GRID = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])

def quant(ex, delta):
    ex = np.sort(ex)
    k = int(np.ceil((1 - delta) * (len(ex) + 1)))
    return np.inf if k > len(ex) else float(ex[k - 1])

OUTJ = {}
for SUB, seeds, eps in [("v1", 6, 0.15), ("v2", 4, 0.05)]:
    Z = np.load(f"eval/seed_results/det_loop3_{SUB}.npz")
    CAL_EX = {ri: Z[f"cal_ex_{ri}"] for ri in range(4)}
    CAL_BER = Z["cal_ber"]
    NCAL = len(CAL_BER)
    ber_path = Z["ber_path"]
    T = len(Z["refs_0"])
    near = [int(np.argmin(np.abs(GRID - b))) for b in ber_path]
    draws = []
    for d in range(12):
        rng = np.random.default_rng([7, d])
        idx = np.arange(NCAL) if d == 0 else rng.choice(NCAL, int(0.8 * NCAL), replace=False)
        HYB = {ri: [quant(CAL_EX[ri][idx][CAL_BER[idx] == b], DELTA)
                    if (CAL_BER[idx] == b).sum() else np.inf for b in GRID]
               for ri in range(4)}
        outs = []
        for s in range(seeds):
            L, PH, refs = Z[f"L_{s}"], Z[f"PH_{s}"], Z[f"refs_{s}"]
            prev = 0; losses = []
            for t in range(T):
                i = max(prev - 1, 0); ri = FULL
                while True:
                    if HYB[i][near[t]] <= eps or i == FULL:
                        ri = i; break
                    i += 1
                losses.append(L[t, ri]); prev = ri
            outs.append(float(np.mean(np.array(losses) > refs + eps)))
        draws.append(float(np.mean(outs)))
    OUTJ[SUB] = {"eps": eps, "full_draw": draws[0], "sub_draws": draws[1:],
                 "sub_mean": float(np.mean(draws[1:])),
                 "sub_min": float(np.min(draws[1:])),
                 "sub_max": float(np.max(draws[1:])),
                 "frac_violating": float(np.mean([o > DELTA for o in draws[1:]]))}
    print(SUB, OUTJ[SUB], flush=True)
json.dump(OUTJ, open("eval/seed_results/hybrid_resample.json", "w"), indent=1)
print("WROTE hybrid_resample.json")
