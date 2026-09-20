#!/usr/bin/env python3
"""Probe: does a given v2 checkpoint respond to the rate knob (target_rate)?
Usage: SUB must stay v2 semantics. CKPT=<path> python eval/probe_ckpt_rho.py
Evaluates 12 val tiles at (rho=0.25, ber=0.15) vs (rho=1.0, ber=0.15) and
(rho=1.0, ber=0.0); prints mean 1-F1 for each. Read-only."""
import os, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import train.train_dota_v8 as v8
v8.BASELINE = "runs/obb/runs/obb/dotav2_baseline/weights/best.pt"
from train.train_dota_v8 import SpikeAdaptSC_Det_Multi, build_v8_model
from ultralytics.utils.metrics import batch_probiou
from ultralytics.utils.ops import xyxyxyxy2xywhr
device = torch.device("cuda")
CKPT = os.environ["CKPT"]
ckp = torch.load(CKPT, map_location=device, weights_only=False)
cfg = ckp["config"]
snn = SpikeAdaptSC_Det_Multi(cfg["channel_sizes"], C_spike=cfg["C_spike"],
                             T=cfg["T"], target_rate=cfg["target_rate"]).to(device)
snn.load_state_dict(ckp["snn_state"]); snn.eval()
yolo, extra = build_v8_model(snn, ber=0.0)
nl_extra = nl_head = 0
for k, sd in ckp.get("extra_modules", {}).items():
    if k in extra:
        try: extra[k].load_state_dict(sd); nl_extra += 1
        except Exception as e: print("extra FAIL", k, str(e)[:80])
for lk, sd in ckp.get("yolo_head_state", {}).items():
    i = int(lk.split("_")[1])
    if i < len(yolo.model.model):
        try: yolo.model.model[i].load_state_dict(sd); nl_head += 1
        except Exception as e: print("head FAIL", lk, str(e)[:80])
print("loaded extra", nl_extra, "head", nl_head)
DSET = Path("/home/jpli/SemCom/datasets/DOTAv2")
img_dir, lbl_dir = DSET/"images/val", DSET/"labels/val"
imgs_all = sorted(q for q in img_dir.glob("*") if q.suffix.lower() in (".jpg",".jpeg",".png"))
GT = {}
imgs = []
for p in imgs_all:
    lp = lbl_dir/(p.stem+".txt"); boxes, clss = [], []
    if lp.exists():
        for line in open(lp):
            v = line.split()
            if len(v) >= 9:
                clss.append(int(float(v[0]))); boxes.append([float(x) for x in v[1:9]])
    GT[p] = (np.array(boxes, dtype=np.float32).reshape(-1,8), np.array(clss, dtype=np.int64))
    imgs.append(p)
rng0 = np.random.default_rng(0)
perm = rng0.permutation(len(imgs))
idxs = perm[:12]
hooks = [m for m in yolo.model.modules() if hasattr(m,"ber") and hasattr(m,"snn")]
maskers = [m for m in yolo.model.modules() if "Masker" in type(m).__name__ and hasattr(m,"target_rate")]
print("hooks", len(hooks), "maskers", len(maskers))
def eval_block(rho, ber):
    for h in hooks: h.ber = float(ber)
    for m in maskers: m.target_rate = float(rho)
    res = yolo.predict([str(imgs[i]) for i in idxs], imgsz=1024, device=0,
                       conf=0.25, verbose=False)
    f1s = []
    for r, i in zip(res, idxs):
        p = imgs[i]; gt_poly, gt_cls = GT[p]
        ob = r.obb
        if ob is None or ob.xywhr is None or len(ob.xywhr) == 0:
            pred = np.zeros((0,5)); pc = np.zeros(0, dtype=np.int64); conf = np.zeros(0)
        else:
            pred = ob.xywhr.cpu().numpy(); pc = ob.cls.cpu().numpy().astype(np.int64); conf = ob.conf.cpu().numpy()
        h0, w0 = r.orig_shape
        if len(gt_poly):
            g = gt_poly.copy(); g[:,0::2] *= w0; g[:,1::2] *= h0
            gtx = xyxyxyxy2xywhr(torch.from_numpy(g).view(-1,4,2)).numpy()
        else:
            gtx = np.zeros((0,5), dtype=np.float32)
        tp = 0; matched = np.zeros(len(gtx), dtype=bool)
        if len(pred) and len(gtx):
            iou = batch_probiou(torch.from_numpy(gtx), torch.from_numpy(pred)).numpy()
            for j in np.argsort(-conf):
                cand = np.where((gt_cls == pc[j]) & (~matched) & (iou[:,j] >= 0.5))[0]
                if len(cand):
                    gi = cand[np.argmax(iou[cand,j])]; matched[gi] = True; tp += 1
        fp = len(pred)-tp; fn = len(gtx)-tp
        f1s.append(1.0 if (fp+fn+tp)==0 else 2*tp/max(2*tp+fp+fn,1e-9))
    return 1.0-float(np.mean(f1s))
for rho, ber in [(1.0,0.0),(1.0,0.15),(0.25,0.15),(0.25,0.0)]:
    print("rho %.2f ber %.2f -> loss %.4f" % (rho, ber, eval_block(rho, ber)), flush=True)
