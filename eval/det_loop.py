#!/usr/bin/env python3
"""Certified detection mission (journal Chapter A): Mondrian-conformal rate
control of the SpikeAdapt v8a DOTA detector over the A2G trajectory.

Ladder: rho in {0.25, 0.5, 0.75, 1.0} via live DetMasker.target_rate override
(payload prop. to rho; full = rho 1.0). Per-block loss = 1 - mean per-image
F1@IoU0.5 (class-aware greedy OBB matching via batch_probiou, conf=0.25).
Damage stats phi (label-free, from detections): mean conf, frac conf<0.5,
dets/img (/50), mean per-image max conf. Denoised full-rate references.
Policies: fixed_full, csi@eps (genie BER->tier quantile LUT), heur@eps
(entropy... here conf-threshold analogue), oracle@eps, split@eps, mond@eps.
Missions: 8 seeds x 80 blocks over the h=60m A2G profile; blocks are random
12-tile draws from the mission split (i.i.d. tiles; disclosed).
Output: eval/seed_results/det_loop.json
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
from eval_v8a_paper import load_v8a, build_and_load
import channel_a2g as ch
from ultralytics.utils.metrics import batch_probiou
from ultralytics.utils.ops import xyxyxyxy2xywhr

OUT = "eval/seed_results/det_loop.json"
DSET = Path("/home/jpli/SemCom/datasets_ul/DOTAv1")
B = 12
T_MISSION = 80
SEEDS = 8
M_CAL, M_REF = 3, 4
DELTA = 0.15
BER_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
RHOS = [0.25, 0.5, 0.75, 1.0]
FULL = 3
CONF = 0.25
t0 = time.time()

# ---------- data ----------
img_dir = DSET / "images/val"
lbl_dir = DSET / "labels/val"
imgs = sorted(img_dir.glob("*"))
assert len(imgs) > 400, len(imgs)
GT = {}
for p in imgs:
    lp = lbl_dir / (p.stem + ".txt")
    boxes, clss = [], []
    if lp.exists():
        for line in open(lp):
            v = line.split()
            if len(v) >= 9:
                clss.append(int(float(v[0])))
                boxes.append([float(x) for x in v[1:9]])
    GT[p] = (np.array(boxes, dtype=np.float32).reshape(-1, 8),
             np.array(clss, dtype=np.int64))
print(f"{len(imgs)} val tiles, GT loaded ({time.time()-t0:.0f}s)", flush=True)

rng0 = np.random.default_rng(0)
perm = rng0.permutation(len(imgs))
cal_idx = perm[:len(imgs) // 2]
mis_idx = perm[len(imgs) // 2:]

# ---------- model + live knobs ----------
ck, snn, cfg = load_v8a()
yolo, _ = build_and_load(snn, ck, ber=0.0)
hooks = [m for m in yolo.model.modules()
         if hasattr(m, "ber") and hasattr(m, "snn")]
maskers = [m for m in yolo.model.modules()
           if "Masker" in type(m).__name__ and hasattr(m, "target_rate")]
assert hooks and maskers
print(f"hooks={len(hooks)} maskers={len(maskers)}", flush=True)


def set_knobs(rho, ber):
    for h in hooks:
        h.ber = float(ber)
    for m in maskers:
        m.target_rate = float(rho)


def eval_block(idxs, rho, ber):
    """Returns (loss = 1 - mean per-image F1, phi damage stats)."""
    set_knobs(rho, ber)
    paths = [str(imgs[i]) for i in idxs]
    res = yolo.predict(paths, imgsz=640, device=0, conf=CONF,
                       verbose=False)
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
        # class-aware greedy matching at probIoU >= 0.5
        tp = 0
        matched = np.zeros(len(gt_xywhr), dtype=bool)
        if len(pred_xywhr) and len(gt_xywhr):
            iou = batch_probiou(
                torch.from_numpy(gt_xywhr),
                torch.from_numpy(pred_xywhr)).numpy()  # [ngt, npred]
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
    phi = np.array([float(confs.mean()),
                    float((confs < 0.5).mean()),
                    float(np.mean(ndets)) / 50.0,
                    float(np.mean(maxcs))])
    return 1.0 - float(np.mean(f1s)), phi


# ---------- sanity ----------
sb = cal_idx[:B]
l0, p0 = eval_block(sb, 1.0, 0.0)
l3, _ = eval_block(sb, 1.0, 0.30)
lr, _ = eval_block(sb, 0.25, 0.0)
print(f"SANITY loss full/clean={l0:.3f} full/ber.3={l3:.3f} "
      f"rho.25/clean={lr:.3f} phi={np.round(p0,3)}", flush=True)

# ---------- calibration ----------
cal_blocks = [cal_idx[i:i + B] for i in range(0, len(cal_idx) - B + 1, B)]
CAL_F = {i: [] for i in range(len(RHOS))}
CAL_EX = {i: [] for i in range(len(RHOS))}
CAL_BER = []
for bi, blk in enumerate(cal_blocks):
    for ber in BER_GRID:
        refs = [eval_block(blk, 1.0, ber)[0] for _ in range(M_CAL)]
        ref = float(np.mean(refs))
        for ri, rho in enumerate(RHOS):
            loss, phi = eval_block(blk, rho, ber)
            CAL_F[ri].append(phi)
            CAL_EX[ri].append(loss - ref)
        CAL_BER.append(ber)
    print(f"cal block {bi+1}/{len(cal_blocks)} ({time.time()-t0:.0f}s)",
          flush=True)
CAL_BER = np.array(CAL_BER)
for ri in CAL_F:
    CAL_F[ri] = np.array(CAL_F[ri])
    CAL_EX[ri] = np.array(CAL_EX[ri])
NCAL = len(CAL_BER)
print(f"calibration {NCAL} pts/tier ({time.time()-t0:.0f}s)", flush=True)

# ---------- bounds ----------
jsplit = min(NCAL - 1, int(np.ceil((1 - DELTA) * (NCAL + 1))) - 1)
SPLIT_Q = {ri: float(np.sort(CAL_EX[ri])[jsplit]) for ri in CAL_F}
print("split_q:", {RHOS[k]: round(v, 3) for k, v in SPLIT_Q.items()},
      flush=True)
NBINS = 5
MOND_EDGES, MOND_Q = {}, {}
for ri in CAL_F:
    e = np.quantile(CAL_F[ri][:, 0], np.linspace(0, 1, NBINS + 1))
    e[0], e[-1] = -np.inf, np.inf
    MOND_EDGES[ri] = e
    qs = []
    for b in range(NBINS):
        m = (CAL_F[ri][:, 0] > e[b]) & (CAL_F[ri][:, 0] <= e[b + 1])
        ex = np.sort(CAL_EX[ri][m])
        nb = len(ex)
        j = min(nb - 1, int(np.ceil((1 - DELTA) * (nb + 1))) - 1)
        qs.append(float(ex[j]))
    MOND_Q[ri] = qs


def ucb(ri, phi, kind):
    if kind == "split":
        return SPLIT_Q[ri]
    b = int(np.searchsorted(MOND_EDGES[ri], phi[0], side="left")) - 1
    return MOND_Q[ri][min(max(b, 0), NBINS - 1)]


CSI_LUT = {}
for eps in (0.05, 0.10):
    lut = []
    for ber in BER_GRID:
        m = CAL_BER == ber
        pick = FULL
        for ri in range(len(RHOS)):
            ex = np.sort(CAL_EX[ri][m])
            nb = len(ex)
            j = min(nb - 1, int(np.ceil((1 - DELTA) * (nb + 1))) - 1)
            if ex[j] <= eps:
                pick = ri
                break
        lut.append(pick)
    CSI_LUT[eps] = lut
print("csi_lut:", CSI_LUT, flush=True)

HEUR_TAU = {}
for eps in (0.05, 0.10):
    taus = []
    for ri in range(len(RHOS)):
        o = np.argsort(-CAL_F[ri][:, 0])   # high mean-conf = healthy
        exs = CAL_EX[ri][o]
        viol = np.cumsum(exs > eps) / (np.arange(len(exs)) + 1)
        ok = np.where(viol <= DELTA)[0]
        taus.append(float(CAL_F[ri][o[ok[-1]], 0]) if len(ok) else np.inf)
    HEUR_TAU[eps] = taus

# ---------- mission ----------
seg = T_MISSION // 5
dg = np.concatenate([np.full(seg, 100.0), np.linspace(100, 1000, seg),
                     np.full(seg, 1000.0), np.linspace(1000, 150, seg),
                     np.full(T_MISSION - 4 * seg, 150.0)])
ber_path = np.array([ch.ber_bpsk_rician(ch.snr_db(60.0, g, "suburban"),
                     ch.k_factor_db(ch.elevation_deg(60.0, g))) for g in dg])
ber_path = np.clip(ber_path, 0, 0.30)
FADE = ber_path > 0.20
BENIGN = ber_path < 0.01
POLS = ["fixed_full", "csi@0.05", "heur@0.05", "oracle@0.05", "split@0.05",
        "mond@0.05", "mond@0.10", "oracle@0.10"]


def run_seed(seed):
    r = np.random.default_rng([seed, 1])
    blocks = [r.choice(mis_idx, B, replace=False) for _ in range(T_MISSION)]
    refs = []
    for t in range(T_MISSION):
        refs.append(float(np.mean([eval_block(blocks[t], 1.0,
                    float(ber_path[t]))[0] for _ in range(M_REF)])))
    out = {}
    for pol in POLS:
        eps = float(pol.split("@")[1]) if "@" in pol else 0.05
        kind = pol.split("@")[0]
        prev = 0
        pay = 0.0
        losses, tiers, forced = [], [], []
        cov_ok = cov_n = 0
        for t in range(T_MISSION):
            ber = float(ber_path[t])
            if kind == "fixed_full":
                loss, _ = eval_block(blocks[t], 1.0, ber)
                ri = FULL
                pay += RHOS[FULL]
                frc = False
            elif kind == "csi":
                ri = CSI_LUT[eps][int(np.argmin(
                    np.abs(np.array(BER_GRID) - ber)))]
                loss, _ = eval_block(blocks[t], RHOS[ri], ber)
                pay += RHOS[ri]
                frc = ri == FULL
            else:
                start = max(prev - 1, 0)
                ri = FULL
                loss = None
                frc = True
                for i in range(start, len(RHOS)):
                    l_i, phi = eval_block(blocks[t], RHOS[i], ber)
                    pay += RHOS[i]
                    if kind == "oracle":
                        ok = l_i <= refs[t] + eps
                    elif kind == "heur":
                        ok = phi[0] >= HEUR_TAU[eps][i]
                    else:
                        bnd = ucb(i, phi, kind)
                        ok = bnd <= eps
                        if ok and i < FULL:
                            cov_n += 1
                            cov_ok += (l_i - refs[t]) <= bnd
                    if ok or i == FULL:
                        ri = i
                        loss = l_i
                        frc = (i == FULL and not ok)
                        break
            losses.append(loss)
            tiers.append(ri)
            forced.append(frc)
            prev = ri
        losses = np.array(losses)
        vio = losses > np.array(refs) + eps
        out[pol] = {
            "payload": pay / T_MISSION / RHOS[FULL],
            "loss": float(losses.mean()),
            "loss_fade": float(losses[FADE].mean()),
            "outage": float(vio.mean()),
            "outage_fade": float(vio[FADE].mean()),
            "outage_benign": float(vio[BENIGN].mean()),
            "n_forced": int(np.sum(forced)),
            "tier_hist": [int((np.array(tiers) == i).sum())
                          for i in range(len(RHOS))],
            "coverage": (cov_ok / cov_n) if cov_n else None,
        }
    return out


results = []
for s in range(SEEDS):
    results.append(run_seed(s))
    print(f"seed {s} done ({time.time()-t0:.0f}s)", flush=True)
    for pol in POLS:
        rr = results[-1][pol]
        print(f"  {pol:12s} payload {rr['payload']:.3f} loss {rr['loss']:.3f}"
              f" outage {rr['outage']:.3f} hist {rr['tier_hist']}",
              flush=True)

agg = {}
for pol in POLS:
    agg[pol] = {}
    for m in results[0][pol]:
        vals = [r[pol][m] for r in results if r[pol][m] is not None]
        if not vals:
            agg[pol][m] = None
        elif m == "tier_hist":
            agg[pol][m] = list(np.mean(np.array(vals), 0).round(1))
        else:
            agg[pol][m] = [float(np.mean(vals)), float(np.std(vals))]

json.dump({"B": B, "T": T_MISSION, "seeds": SEEDS, "delta": DELTA,
           "rhos": RHOS, "conf_thresh": CONF, "split_q": SPLIT_Q,
           "csi_lut": CSI_LUT, "n_cal": NCAL,
           "ber_max": float(ber_path.max()), "agg": agg,
           "per_seed": results, "runtime_s": round(time.time() - t0, 1)},
          open(OUT, "w"), indent=1)
print("WROTE", OUT, round(time.time() - t0, 1), "s", flush=True)
