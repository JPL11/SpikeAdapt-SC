#!/usr/bin/env python3
"""Detection offline analysis from det_loop3_{v1,v2}.npz (no GPU):
heur_bin (per-bin empirical rule) on detection, selection-free coverage,
calibration-size sweep (conformal vs empirical), eps-frontier, sticky-capped
Mondrian. Output: eval/seed_results/det_offline_h.json"""
import json
import time

import numpy as np

t0 = time.time()
DELTA = 0.15
NBINS = 5
RHOS = [0.25, 0.5, 0.75, 1.0]
FULL = 3
GRID = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
EPS_SWEEP = [0.03, 0.05, 0.08, 0.10, 0.15]
OUTJ = {}

def quant(ex, delta):
    ex = np.sort(ex)
    k = int(np.ceil((1 - delta) * (len(ex) + 1)))
    if k > len(ex):
        return np.inf
    return float(ex[k - 1])

for SUB, seeds in [("v1_mc6", 6), ("v2", 4)]:
    Z = np.load(f"eval/seed_results/det_loop3_{SUB}.npz")
    CAL_F = {ri: Z[f"cal_f_{ri}"] for ri in range(4)}
    CAL_EX = {ri: Z[f"cal_ex_{ri}"] for ri in range(4)}
    CAL_BER = Z["cal_ber"]
    ber_path = Z["ber_path"]
    NCAL = len(CAL_BER)
    T = len(Z["refs_0"])
    near = [int(np.argmin(np.abs(GRID - b))) for b in ber_path]

    def build_bounds(idx, eps_acc=0.10):
        sq, me, mq, hb = {}, {}, {}, {}
        for ri in range(4):
            ex = CAL_EX[ri][idx]
            f0 = CAL_F[ri][idx, 0]
            sq[ri] = quant(ex, DELTA)
            e = np.quantile(f0, np.linspace(0, 1, NBINS + 1))
            e[0], e[-1] = -np.inf, np.inf
            me[ri] = e
            qs, accs = [], []
            for b in range(NBINS):
                m = (f0 > e[b]) & (f0 <= e[b + 1])
                if m.sum() == 0:
                    qs.append(np.inf); accs.append(False); continue
                qs.append(quant(ex[m], DELTA))
                accs.append(bool(np.mean(ex[m] > eps_acc) <= DELTA))
            mq[ri] = qs
            hb[ri] = accs
        return sq, me, mq, hb

    def mb(B_, ri, phi):
        _, me, mq, _ = B_
        b = int(np.searchsorted(me[ri], phi[0], side="left")) - 1
        return mq[ri][min(max(b, 0), NBINS - 1)]

    def hb_ok(B_, ri, phi):
        _, me, _, hb = B_
        b = int(np.searchsorted(me[ri], phi[0], side="left")) - 1
        return hb[ri][min(max(b, 0), NBINS - 1)]

    FULLB = build_bounds(np.arange(NCAL))
    FULLB_E = {e: build_bounds(np.arange(NCAL), e) for e in EPS_SWEEP}
    # certified-CSI hybrid: per-(tier, BER-cell) conformal quantiles
    HYB = {ri: [quant(CAL_EX[ri][CAL_BER == b], DELTA)
                if (CAL_BER == b).sum() else np.inf for b in GRID]
           for ri in range(4)}

    def hyb_bound(ri, t):
        return HYB[ri][near[t]]
    # CSI LUT + heuristic taus (det convention: high mean-conf = healthy)
    CSI_LUT, HEUR_TAU = {}, {}
    for eps in EPS_SWEEP:
        lut = []
        for ber in GRID:
            m = CAL_BER == ber
            pick = FULL
            for ri in range(4):
                if quant(CAL_EX[ri][m], DELTA) <= eps:
                    pick = ri; break
            lut.append(pick)
        CSI_LUT[eps] = lut
        taus = []
        for ri in range(4):
            o = np.argsort(-CAL_F[ri][:, 0])
            exs = CAL_EX[ri][o]
            viol = np.cumsum(exs > eps) / (np.arange(len(exs)) + 1)
            ok = np.where(viol <= DELTA)[0]
            taus.append(float(CAL_F[ri][o[ok[-1]], 0]) if len(ok) else np.inf)
        HEUR_TAU[eps] = taus

    def run_escal(L, PH, refs, eps, decide, sticky=False, auto=False, W=16):
        # auto: label-free engage rule -- run sticky mode only while the
        # trailing-W mean per-block payment exceeds full rate (observable
        # at runtime; no labels, no CSI). sticky: always-on fallback.
        prev = 0; pay = 0.0; losses = []; tiers = []
        recent = []
        engaged = False
        for t in range(T):
            if auto:
                trail = np.mean(recent[-W:]) if recent else 0.0
                engaged = trail > RHOS[FULL]
            mode = sticky or (auto and engaged)
            if mode and prev == FULL and (t % 8) != 0:
                start = FULL
            else:
                start = max(prev - 1, 0)
            i = start; ri = FULL
            paid = 0.0
            while True:
                paid += RHOS[i]
                if decide(t, i) or i == FULL:
                    ri = i; break
                i = FULL if mode else i + 1
            pay += paid; recent.append(paid)
            losses.append(L[t, ri]); tiers.append(ri); prev = ri
        losses = np.array(losses)
        vio = losses > refs + eps
        return {"payload": pay / T / RHOS[FULL], "outage": float(vio.mean())}

    per_seed = []
    for s in range(seeds):
        L, PH, refs = Z[f"L_{s}"], Z[f"PH_{s}"], Z[f"refs_{s}"]
        o = {}
        for eps in EPS_SWEEP:
            o[f"mond@{eps}"] = run_escal(L, PH, refs, eps,
                lambda t, i, e=eps: mb(FULLB, i, PH[t, i]) <= e)
            o[f"mond_sticky@{eps}"] = run_escal(L, PH, refs, eps,
                lambda t, i, e=eps: mb(FULLB, i, PH[t, i]) <= e, sticky=True)
            o[f"mond_auto@{eps}"] = run_escal(L, PH, refs, eps,
                lambda t, i, e=eps: mb(FULLB, i, PH[t, i]) <= e, auto=True)
            o[f"heur_bin@{eps}"] = run_escal(L, PH, refs, eps,
                lambda t, i, e=eps: hb_ok(FULLB_E[e], i, PH[t, i]))
            o[f"heur@{eps}"] = run_escal(L, PH, refs, eps,
                lambda t, i, e=eps: PH[t, i, 0] >= HEUR_TAU[e][i])
            o[f"oracle0@{eps}"] = run_escal(L, PH, refs, eps,
                lambda t, i, e=eps: L[t, i] <= refs[t] + e)
            o[f"hybrid@{eps}"] = run_escal(L, PH, refs, eps,
                lambda t, i, e=eps: hyb_bound(i, t) <= e)
            tiers = [CSI_LUT[eps][near[t]] for t in range(T)]
            losses = np.array([L[t, tiers[t]] for t in range(T)])
            o[f"csi@{eps}"] = {"payload": float(np.mean([RHOS[i] for i in tiers])) / RHOS[FULL],
                               "outage": float(np.mean(losses > refs + eps))}
            sat = np.full(T, np.nan)
            for t in range(T):
                for i in range(4):
                    if L[t, i] <= refs[t] + eps:
                        sat[t] = RHOS[i]; break
            forced = np.isnan(sat)
            cost = np.where(forced, RHOS[0], sat)
            budget = int(np.floor(DELTA * T)) - int(forced.sum())
            if budget > 0:
                sav = np.where(forced, -1.0, sat - RHOS[0])
                cheat = np.argsort(-sav)[:budget]; cheat = cheat[sav[cheat] > 0]
                cost[cheat] = RHOS[0]
            o[f"genie@{eps}"] = {"payload": float(cost.mean()) / RHOS[FULL],
                                 "outage": DELTA}
        # calibration-size sweep at 0.10: NSUB subsets SHARED across seeds
        # so draw-level (seed-averaged) statistics are well defined
        NSUB = 8
        for ncal in [NCAL, 84, 42, 21]:
            runs_m, runs_e = [], []
            for d_ in range(1 if ncal >= NCAL else NSUB):
                rngc = np.random.default_rng([99, d_])
                idx = np.arange(NCAL) if ncal >= NCAL else rngc.choice(NCAL, ncal, replace=False)
                B_ = build_bounds(idx)
                runs_m.append(run_escal(L, PH, refs, 0.10,
                    lambda t, i, B=B_: mb(B, i, PH[t, i]) <= 0.10))
                runs_e.append(run_escal(L, PH, refs, 0.10,
                    lambda t, i, B=B_: hb_ok(B, i, PH[t, i])))
            o[f"calsz_mond@{ncal}"] = {m: float(np.mean([r[m] for r in runs_m])) for m in runs_m[0]}
            o[f"calsz_emp@{ncal}"] = {m: float(np.mean([r[m] for r in runs_e])) for m in runs_e[0]}
            o[f"calsz_mond@{ncal}"]["draw_out"] = [float(r["outage"]) for r in runs_m]
            o[f"calsz_emp@{ncal}"]["draw_out"] = [float(r["outage"]) for r in runs_e]
        # selection-free coverage
        covs = []
        for i in range(FULL):
            bd = np.array([mb(FULLB, i, PH[t, i]) for t in range(T)])
            covs.append(np.mean((L[:, i] - refs) <= bd))
        o["selfree_cov"] = {"payload": float(np.mean(covs)), "outage": None}
        per_seed.append(o)
    agg = {}
    for pol in per_seed[0]:
        agg[pol] = {}
        for m in per_seed[0][pol]:
            vals = [r[pol][m] for r in per_seed if r[pol].get(m) is not None]
            if m == "draw_out":
                M = np.array(vals)              # seeds x draws (shared)
                dl = M.mean(0)
                agg[pol][m] = [float(dl.mean()), float(dl.max()),
                               float(np.mean(dl > DELTA))]
                agg[pol][m + "_cellmax"] = float(M.max())
            else:
                agg[pol][m] = [float(np.mean(vals)), float(np.std(vals))] if vals else None
    OUTJ[SUB] = agg
    print(SUB, "done", flush=True)

json.dump(OUTJ, open("eval/seed_results/det_offline_h.json", "w"), indent=1)
print("WROTE det_offline_h.json", round(time.time() - t0, 1), "s", flush=True)
