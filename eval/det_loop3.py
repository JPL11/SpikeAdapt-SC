#!/usr/bin/env python3
"""Chapter A analysis pass: delta-constrained oracle + weighted conformal.
Design change vs det_loop2: per seed we evaluate the FULL (block x tier)
loss/phi matrix once (plus denoised refs), then compute ALL policies offline
from the matrix with paired channel realizations:
  - mond@eps, split@eps (unweighted; sanity vs det_loop2)
  - mond_w@eps, split_w@eps: WEIGHTED conformal quantiles, weights =
    trajectory-planned mission BER histogram over the calibration grid
    (known pre-flight from geometry; no runtime CSI), conservative +infty
    mass of max-weight (Tibshirani-style).
  - csi@eps, heur@eps (from recomputed calibration)
  - oracle0@eps: zero-outage escalation reference (as before)
  - genie_delta@eps: DELTA-CONSTRAINED payload-optimal genie — cheapest
    satisfying tier per block, with the floor(delta*T) violation budget spent
    on the blocks with the largest savings (separable optimum; no HARQ climb
    costs; a strict payload lower bound).
Calibration arrays and matrices are SAVED (npz) for future analysis.
SUB=v1 (DOTA-v1, 6 seeds) / SUB=v2 (DOTA-v2, 4 seeds).
Output: eval/seed_results/det_loop3_{SUB}.json (+ .npz)
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import channel_a2g as ch
from ultralytics.utils.metrics import batch_probiou
from ultralytics.utils.ops import xyxyxyxy2xywhr

SUB = os.environ.get("SUB", "v1")
device = torch.device("cuda")
t0 = time.time()

if SUB == "v1":
    from eval_v8a_paper import load_v8a, build_and_load
    ck, snn, cfg = load_v8a()
    yolo, _ = build_and_load(snn, ck, ber=0.0)
    DSET = Path("/home/jpli/SemCom/datasets_ul/DOTAv1")
    IMGSZ = 640
    B, T_MISSION, SEEDS = 24, 80, 6
    M_CAL, M_REF, N_CALB = 6, 5, 24
    EPSES = [0.05, 0.10]
    LABELED_ONLY = False
else:
    import train.train_dota_v8 as v8
    v8.BASELINE = "runs/obb/runs/obb/dotav2_baseline/weights/best.pt"
    from train.train_dota_v8 import SpikeAdaptSC_Det_Multi, build_v8_model
    # PINNED paper checkpoint (read-only copy; runs/yolo26_snn_dotav2.pth
    # was overwritten 2026-09-01 by unrelated training and no longer works
    # under this harness)
    ckp = torch.load(os.environ.get("DET_CKPT",
                     "runs/yolo26_snn_dotav2_tccn_paper.pth"),
                     map_location=device, weights_only=False)
    cfg = ckp["config"]
    snn = SpikeAdaptSC_Det_Multi(cfg["channel_sizes"],
                                 C_spike=cfg["C_spike"], T=cfg["T"],
                                 target_rate=cfg["target_rate"]).to(device)
    snn.load_state_dict(ckp["snn_state"])
    snn.eval()
    yolo, extra = build_v8_model(snn, ber=0.0)
    for k, sd in ckp.get("extra_modules", {}).items():
        if k in extra:
            try:
                extra[k].load_state_dict(sd)
            except Exception:
                pass
    for lk, sd in ckp.get("yolo_head_state", {}).items():
        i = int(lk.split("_")[1])
        if i < len(yolo.model.model):
            try:
                yolo.model.model[i].load_state_dict(sd)
            except Exception:
                pass
    DSET = Path("/home/jpli/SemCom/datasets/DOTAv2")
    IMGSZ = 1024
    B, T_MISSION, SEEDS = 24, 60, 4
    M_CAL, M_REF, N_CALB = 5, 5, 20
    EPSES = [0.10]
    LABELED_ONLY = True

# env overrides (TCCN round-1: seed expansion + MC harmonization) with a
# non-clobbering output tag so archived results stay canonical
SEEDS = int(os.environ.get("DET_SEEDS", SEEDS))
M_CAL = int(os.environ.get("DET_MCAL", M_CAL))
M_REF = int(os.environ.get("DET_MREF", M_REF))
_TAG = os.environ.get("DET_TAG", "")
OUT = f"eval/seed_results/det_loop3_{SUB}{_TAG}.json"
NPZ = f"eval/seed_results/det_loop3_{SUB}{_TAG}.npz"
DELTA = 0.15
BER_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
RHOS = [0.25, 0.5, 0.75, 1.0]
FULL = 3
CONF = 0.25

img_dir = DSET / "images/val"
lbl_dir = DSET / "labels/val"
imgs_all = sorted(q for q in img_dir.glob("*")
                  if q.suffix.lower() in (".jpg", ".jpeg", ".png"))
GT = {}
imgs = []
for p in imgs_all:
    lp = lbl_dir / (p.stem + ".txt")
    boxes, clss = [], []
    if lp.exists():
        for line in open(lp):
            v = line.split()
            if len(v) >= 9:
                clss.append(int(float(v[0])))
                boxes.append([float(x) for x in v[1:9]])
    if LABELED_ONLY and not boxes:
        continue
    GT[p] = (np.array(boxes, dtype=np.float32).reshape(-1, 8),
             np.array(clss, dtype=np.int64))
    imgs.append(p)
print(f"{SUB}: {len(imgs)} tiles ({time.time()-t0:.0f}s)", flush=True)

rng0 = np.random.default_rng(0)
perm = rng0.permutation(len(imgs))
cal_pool = perm[:len(imgs) // 2]
mis_pool = perm[len(imgs) // 2:]

hooks = [m for m in yolo.model.modules()
         if hasattr(m, "ber") and hasattr(m, "snn")]
maskers = [m for m in yolo.model.modules()
           if "Masker" in type(m).__name__ and hasattr(m, "target_rate")]
assert hooks and maskers


def set_knobs(rho, ber):
    for h in hooks:
        h.ber = float(ber)
    for m in maskers:
        m.target_rate = float(rho)


def eval_block(idxs, rho, ber):
    set_knobs(rho, ber)
    res = yolo.predict([str(imgs[i]) for i in idxs], imgsz=IMGSZ, device=0,
                       conf=CONF, verbose=False)
    f1s, confs, maxcs, ndets = [], [], [], []
    for p_i, r in zip(idxs, res):
        gt_poly, gt_cls = GT[imgs[p_i]]
        ob = r.obb
        if ob is None or len(ob) == 0:
            pred_xywhr = np.zeros((0, 5), dtype=np.float32)
            pred_cls = np.zeros(0, dtype=np.int64)
            pred_conf = np.zeros(0, dtype=np.float32)
        else:
            pred_xywhr = ob.xywhr.cpu().numpy()
            pred_cls = ob.cls.cpu().numpy().astype(np.int64)
            pred_conf = ob.conf.cpu().numpy()
        h0, w0 = r.orig_shape
        if len(gt_poly):
            g = gt_poly.copy()
            g[:, 0::2] *= w0
            g[:, 1::2] *= h0
            gt_xywhr = xyxyxyxy2xywhr(
                torch.from_numpy(g).view(-1, 4, 2)).numpy()
        else:
            gt_xywhr = np.zeros((0, 5), dtype=np.float32)
        tp = 0
        matched = np.zeros(len(gt_xywhr), dtype=bool)
        if len(pred_xywhr) and len(gt_xywhr):
            iou = batch_probiou(torch.from_numpy(gt_xywhr),
                                torch.from_numpy(pred_xywhr)).numpy()
            for j in np.argsort(-pred_conf):
                cand = np.where((gt_cls == pred_cls[j]) & (~matched)
                                & (iou[:, j] >= 0.5))[0]
                if len(cand):
                    gi = cand[np.argmax(iou[cand, j])]
                    matched[gi] = True
                    tp += 1
        fp = len(pred_xywhr) - tp
        fn = len(gt_xywhr) - tp
        f1s.append(1.0 if (fp + fn + tp) == 0
                   else 2 * tp / max(2 * tp + fp + fn, 1e-9))
        confs.extend(pred_conf.tolist())
        maxcs.append(float(pred_conf.max()) if len(pred_conf) else 0.0)
        ndets.append(len(pred_conf))
    confs = np.array(confs) if confs else np.zeros(1)
    phi = np.array([float(confs.mean()), float((confs < 0.5).mean()),
                    float(np.mean(ndets)) / 50.0, float(np.mean(maxcs))])
    return 1.0 - float(np.mean(f1s)), phi


# ---------- calibration ----------
rcal = np.random.default_rng(2)
CAL_F = {i: [] for i in range(len(RHOS))}
CAL_EX = {i: [] for i in range(len(RHOS))}
CAL_BER = []
for bi in range(N_CALB):
    blk = rcal.choice(cal_pool, B, replace=False)
    for ber in BER_GRID:
        ref = float(np.mean([eval_block(blk, 1.0, ber)[0]
                             for _ in range(M_CAL)]))
        for ri, rho in enumerate(RHOS):
            loss, phi = eval_block(blk, rho, ber)
            CAL_F[ri].append(phi)
            CAL_EX[ri].append(loss - ref)
        CAL_BER.append(ber)
    print(f"cal {bi+1}/{N_CALB} ({time.time()-t0:.0f}s)", flush=True)
CAL_BER = np.array(CAL_BER)
for ri in CAL_F:
    CAL_F[ri] = np.array(CAL_F[ri])
    CAL_EX[ri] = np.array(CAL_EX[ri])
NCAL = len(CAL_BER)

# ---------- mission BER path + trajectory-planned weights ----------
seg = T_MISSION // 5
dg = np.concatenate([np.full(seg, 100.0), np.linspace(100, 1000, seg),
                     np.full(seg, 1000.0), np.linspace(1000, 150, seg),
                     np.full(T_MISSION - 4 * seg, 150.0)])
ber_path = np.clip([ch.ber_bpsk_rician(ch.snr_db(60.0, g, "suburban"),
                    ch.k_factor_db(ch.elevation_deg(60.0, g)))
                    for g in dg], 0, 0.30)
FADE = ber_path > 0.20
BENIGN = ber_path < 0.01
grid = np.array(BER_GRID)
near = np.array([int(np.argmin(np.abs(grid - b))) for b in ber_path])
pi_mis = np.bincount(near, minlength=len(grid)) / len(near)
W_CAL = pi_mis[np.array([int(np.argmin(np.abs(grid - b)))
                         for b in CAL_BER])] * len(grid)  # /(1/7)
print("pi_mis:", np.round(pi_mis, 3), flush=True)


def quant(ex, delta):
    ex = np.sort(ex)
    n = len(ex)
    if n == 0:
        return float("inf")  # empty Mondrian bin: no certificate, escalate
    j = min(n - 1, int(np.ceil((1 - delta) * (n + 1))) - 1)
    return float(ex[j])


def wquant(ex, w, delta):
    if len(ex) == 0:
        return float("inf")
    o = np.argsort(ex)
    ex, w = np.asarray(ex)[o], np.asarray(w)[o]
    cum = np.cumsum(w) / (w.sum() + w.max())
    idx = int(np.searchsorted(cum, 1 - delta, side="left"))
    return float(ex[idx]) if idx < len(ex) else float("inf")


NBINS = 5
BOUNDS = {}
for weighted in (False, True):
    tag = "w" if weighted else "u"
    sq, me, mq = {}, {}, {}
    for ri in CAL_F:
        w = W_CAL if weighted else np.ones(NCAL)
        sq[ri] = wquant(CAL_EX[ri], w, DELTA) if weighted \
            else quant(CAL_EX[ri], DELTA)
        e = np.quantile(CAL_F[ri][:, 0], np.linspace(0, 1, NBINS + 1))
        e[0], e[-1] = -np.inf, np.inf
        me[ri] = e
        qs = []
        for b in range(NBINS):
            m = (CAL_F[ri][:, 0] > e[b]) & (CAL_F[ri][:, 0] <= e[b + 1])
            qs.append(wquant(CAL_EX[ri][m], w[m], DELTA) if weighted
                      else quant(CAL_EX[ri][m], DELTA))
        mq[ri] = qs
    BOUNDS[tag] = (sq, me, mq)
print("split_q u:", {RHOS[k]: round(v, 3)
                     for k, v in BOUNDS["u"][0].items()}, flush=True)
print("split_q w:", {RHOS[k]: round(v, 3)
                     for k, v in BOUNDS["w"][0].items()}, flush=True)


def ucb(ri, phi, kind, tag):
    sq, me, mq = BOUNDS[tag]
    if kind == "split":
        return sq[ri]
    b = int(np.searchsorted(me[ri], phi[0], side="left")) - 1
    return mq[ri][min(max(b, 0), NBINS - 1)]


CSI_LUT, HEUR_TAU = {}, {}
for eps in EPSES:
    lut = []
    for ber in BER_GRID:
        m = CAL_BER == ber
        pick = FULL
        for ri in range(len(RHOS)):
            if quant(CAL_EX[ri][m], DELTA) <= eps:
                pick = ri
                break
        lut.append(pick)
    CSI_LUT[eps] = lut
    taus = []
    for ri in range(len(RHOS)):
        o = np.argsort(-CAL_F[ri][:, 0])
        exs = CAL_EX[ri][o]
        viol = np.cumsum(exs > eps) / (np.arange(len(exs)) + 1)
        ok = np.where(viol <= DELTA)[0]
        taus.append(float(CAL_F[ri][o[ok[-1]], 0]) if len(ok) else np.inf)
    HEUR_TAU[eps] = taus

# ---------- per-seed full matrix, then offline policies ----------
def seed_matrix(seed):
    r = np.random.default_rng([seed, 1])
    blocks = [r.choice(mis_pool, B, replace=False) for _ in range(T_MISSION)]
    L = np.zeros((T_MISSION, len(RHOS)))
    PH = np.zeros((T_MISSION, len(RHOS), 4))
    refs = np.zeros(T_MISSION)
    for t in range(T_MISSION):
        ber = float(ber_path[t])
        refs[t] = np.mean([eval_block(blocks[t], 1.0, ber)[0]
                           for _ in range(M_REF)])
        for ri, rho in enumerate(RHOS):
            L[t, ri], PH[t, ri] = eval_block(blocks[t], rho, ber)
    return L, PH, refs


def policies_from_matrix(L, PH, refs):
    out = {}
    for eps in EPSES:
        # escalation policies (warm-start), simulated from the matrix
        for name, decide in [
            ("oracle0", lambda t, i: L[t, i] <= refs[t] + eps),
            ("heur", lambda t, i: PH[t, i, 0] >= HEUR_TAU[eps][i]),
            ("split_u", lambda t, i: ucb(i, PH[t, i], "split", "u") <= eps),
            ("split_w", lambda t, i: ucb(i, PH[t, i], "split", "w") <= eps),
            ("mond_u", lambda t, i: ucb(i, PH[t, i], "mond", "u") <= eps),
            ("mond_w", lambda t, i: ucb(i, PH[t, i], "mond", "w") <= eps),
        ]:
            prev = 0
            pay = 0.0
            losses, tiers = [], []
            cov_ok = cov_n = 0
            for t in range(T_MISSION):
                start = max(prev - 1, 0)
                ri = FULL
                for i in range(start, len(RHOS)):
                    pay += RHOS[i]
                    ok = decide(t, i)
                    if name in ("split_u", "split_w", "mond_u", "mond_w") \
                            and ok and i < FULL:
                        cov_n += 1
                        cov_ok += (L[t, i] - refs[t]) <= ucb(
                            i, PH[t, i], name.split("_")[0], name[-1])
                    if ok or i == FULL:
                        ri = i
                        break
                losses.append(L[t, ri])
                tiers.append(ri)
                prev = ri
            losses = np.array(losses)
            vio = losses > refs + eps
            out[f"{name}@{eps}"] = {
                "payload": pay / T_MISSION / RHOS[FULL],
                "outage": float(vio.mean()),
                "outage_benign": float(vio[BENIGN].mean()),
                "outage_fade": float(vio[FADE].mean()),
                "loss": float(losses.mean()),
                "tier_hist": [int((np.array(tiers) == i).sum())
                              for i in range(len(RHOS))],
                "coverage": (cov_ok / cov_n) if cov_n else None,
            }
        # CSI lookup (no escalation cost)
        tiers = [CSI_LUT[eps][near[t]] for t in range(T_MISSION)]
        losses = np.array([L[t, tiers[t]] for t in range(T_MISSION)])
        vio = losses > refs + eps
        out[f"csi@{eps}"] = {
            "payload": float(np.mean([RHOS[i] for i in tiers])) / RHOS[FULL],
            "outage": float(vio.mean()),
            "outage_benign": float(vio[BENIGN].mean()),
            "outage_fade": float(vio[FADE].mean()),
            "loss": float(losses.mean()),
            "tier_hist": [int(np.sum(np.array(tiers) == i))
                          for i in range(len(RHOS))],
            "coverage": None,
        }
        # delta-constrained payload-optimal genie
        sat_cost = np.full(T_MISSION, np.nan)
        for t in range(T_MISSION):
            for i in range(len(RHOS)):
                if L[t, i] <= refs[t] + eps:
                    sat_cost[t] = RHOS[i]
                    break
        forced = np.isnan(sat_cost)
        budget = int(np.floor(DELTA * T_MISSION)) - int(forced.sum())
        cost = np.where(forced, RHOS[0], sat_cost)
        if budget > 0:
            savings = np.where(forced, -1.0, sat_cost - RHOS[0])
            cheat = np.argsort(-savings)[:budget]
            cheat = cheat[savings[cheat] > 0]
            cost[cheat] = RHOS[0]
            n_v = int(forced.sum()) + len(cheat)
        else:
            n_v = int(forced.sum())
        out[f"genie_delta@{eps}"] = {
            "payload": float(cost.mean()) / RHOS[FULL],
            "outage": n_v / T_MISSION,
            "outage_benign": None, "outage_fade": None,
            "loss": None, "tier_hist": None, "coverage": None,
        }
        # fixed full
        vio = L[:, FULL] > refs + eps
        out[f"fixed_full@{eps}"] = {
            "payload": 1.0, "outage": float(vio.mean()),
            "outage_benign": float(vio[BENIGN].mean()),
            "outage_fade": float(vio[FADE].mean()),
            "loss": float(L[:, FULL].mean()),
            "tier_hist": None, "coverage": None,
        }
    return out


results = []
mats = []
for s in range(SEEDS):
    L, PH, refs = seed_matrix(s)
    mats.append((L, PH, refs))
    results.append(policies_from_matrix(L, PH, refs))
    print(f"seed {s} done ({time.time()-t0:.0f}s)", flush=True)
    for pol, rr in results[-1].items():
        print(f"  {pol:16s} payload {rr['payload']:.3f} "
              f"outage {rr['outage']:.3f}", flush=True)

agg = {}
for pol in results[0]:
    agg[pol] = {}
    for m in results[0][pol]:
        vals = [r[pol][m] for r in results if r[pol][m] is not None]
        if not vals:
            agg[pol][m] = None
        elif m == "tier_hist":
            agg[pol][m] = list(np.mean(np.array(vals), 0).round(1))
        else:
            agg[pol][m] = [float(np.mean(vals)), float(np.std(vals))]

np.savez_compressed(NPZ,
                    cal_ber=CAL_BER, w_cal=W_CAL,
                    **{f"cal_f_{ri}": CAL_F[ri] for ri in CAL_F},
                    **{f"cal_ex_{ri}": CAL_EX[ri] for ri in CAL_F},
                    **{f"L_{s}": mats[s][0] for s in range(SEEDS)},
                    **{f"PH_{s}": mats[s][1] for s in range(SEEDS)},
                    **{f"refs_{s}": mats[s][2] for s in range(SEEDS)},
                    ber_path=ber_path)
json.dump({"sub": SUB, "B": B, "T": T_MISSION, "seeds": SEEDS,
           "delta": DELTA, "pi_mis": [float(x) for x in pi_mis],
           "split_q_u": BOUNDS["u"][0], "split_q_w": BOUNDS["w"][0],
           "agg": agg, "per_seed": results,
           "runtime_s": round(time.time() - t0, 1)},
          open(OUT, "w"), indent=1)
print("WROTE", OUT, round(time.time() - t0, 1), "s", flush=True)
