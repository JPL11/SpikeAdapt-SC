#!/usr/bin/env python3
"""Detection mAP vs BER sweep for the V8A SNN-bottleneck detector on DOTA.

Strengthens the paper's task-generality section: instead of two points
(clean, BER=0.20), produce the full graceful-degradation curve mAP50(BER)
for the SNN-bottleneck detector, plus the unconstrained baseline reference.

Output: eval/seed_results/detection_ber_sweep.json

Usage:
    python eval/eval_detection_ber_sweep.py
"""

import os, sys, json
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from eval_v8a_paper import load_v8a, build_and_load, BASELINE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
# CLI: [data_yaml] [imgsz] [out_json]; defaults reproduce the original run
DATA = sys.argv[1] if len(sys.argv) > 1 else 'DOTAv1.yaml'
IMGSZ = int(sys.argv[2]) if len(sys.argv) > 2 else 640
OUT = sys.argv[3] if len(sys.argv) > 3 \
    else 'eval/seed_results/detection_ber_sweep.json'


def main():
    results = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            results = json.load(f)

    # --- unconstrained baseline (no SNN bottleneck), once
    if 'baseline' not in results:
        from ultralytics import YOLO
        yolo_base = YOLO(BASELINE)
        r = yolo_base.val(data=DATA, imgsz=IMGSZ, batch=8,
                          device=0, verbose=False, plots=False)
        results['baseline'] = dict(map50=float(r.box.map50),
                                   map=float(r.box.map))
        with open(OUT, 'w') as f:
            json.dump(results, f, indent=1)
        print(f"baseline: mAP50={results['baseline']['map50']:.4f}",
              flush=True)
        del yolo_base
        torch.cuda.empty_cache()

    # --- V8A SNN bottleneck across BER
    ck, snn, cfg = load_v8a()
    results.setdefault('v8a', {})
    for ber in BERS:
        if str(ber) in results['v8a']:
            continue
        yolo, _ = build_and_load(snn, ck, ber=ber)
        r = yolo.val(data=DATA, imgsz=IMGSZ, batch=8,
                     device=0, verbose=False, plots=False)
        results['v8a'][str(ber)] = dict(map50=float(r.box.map50),
                                        map=float(r.box.map))
        with open(OUT, 'w') as f:
            json.dump(results, f, indent=1)
        print(f"v8a BER={ber}: mAP50={float(r.box.map50):.4f} "
              f"mAP50-95={float(r.box.map):.4f}", flush=True)
        del yolo
        torch.cuda.empty_cache()

    print('Done.', flush=True)


if __name__ == '__main__':
    main()
