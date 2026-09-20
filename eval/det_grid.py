#!/usr/bin/env python3
"""Certified-detection prerequisite: mAP50 over rho x BER grid for the v8a
SNN-bottleneck DOTA detector, using LIVE knobs (hook.ber + masker.target_rate
override) on a single built model. Ladder-viability check for the journal's
certified-detection chapter. Output: eval/seed_results/det_grid.json
Run from ~/SemCom with the semcom conda python."""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from eval_v8a_paper import load_v8a, build_and_load

OUT = "eval/seed_results/det_grid.json"
RHOS = [0.25, 0.5, 0.75, 1.0]
BERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

t0 = time.time()
ck, snn, cfg = load_v8a()
yolo, extra = build_and_load(snn, ck, ber=0.0)

# introspect live knobs
hooks = [m for m in yolo.model.modules()
         if hasattr(m, "ber") and hasattr(m, "snn")]
maskers = [m for m in yolo.model.modules()
           if "Masker" in type(m).__name__ and hasattr(m, "target_rate")]
print(f"hooks={len(hooks)} maskers={len(maskers)}", flush=True)
assert hooks and maskers
base_rates = [m.target_rate for m in maskers]

results = {}
if os.path.exists(OUT):
    results = json.load(open(OUT))

for rho in RHOS:
    for ber in BERS:
        key = f"rho{rho}_ber{ber}"
        if key in results:
            continue
        for h in hooks:
            h.ber = ber
        for m in maskers:
            m.target_rate = rho
        r = yolo.val(data="DOTAv1.yaml", imgsz=640, batch=8, device=0,
                     verbose=False, plots=False)
        results[key] = {"map50": float(r.box.map50), "map": float(r.box.map)}
        json.dump(results, open(OUT, "w"), indent=1)
        print(f"{key}: mAP50={results[key]['map50']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        torch.cuda.empty_cache()

# restore
for m, br in zip(maskers, base_rates):
    m.target_rate = br
print("DONE", round(time.time() - t0, 1), "s", flush=True)
