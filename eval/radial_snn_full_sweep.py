#!/usr/bin/env python3
"""BER sweep of the trained FFTRadNet SNN bottleneck under the FULL evaluation
protocol (run_FullEvaluation: threshold-swept mAP/mAR), matching the baseline
0.9849/0.9170/74.64 numbers in eval/fftradnet_baseline.log.

Usage: python eval/radial_snn_full_sweep.py
Output: eval/seed_results/radial_snn_full_sweep.json
"""

import contextlib
import io
import json
import os
import re
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'radial_repo', 'FFTRadNet'))

from utils.evaluation import run_FullEvaluation              # noqa: E402
from train.train_dota_v6d2d import SpikeAdaptSC_Det_Multi    # noqa: E402
from train.train_radial_snn import (BERS, CHANNEL_SIZES, CONFIG, SAVE,  # noqa: E402
                                    FFTRadNetSNN, build_loaders, load_base)

OUT_JSON = os.path.join(ROOT, 'eval/seed_results/radial_snn_full_sweep.json')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=SAVE)
    ap.add_argument('--out', default=OUT_JSON)
    args = ap.parse_args()
    config = json.load(open(CONFIG))
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    base = load_base(config)
    snn = SpikeAdaptSC_Det_Multi(CHANNEL_SIZES, C_spike=36, T=8,
                                 target_rate=0.75)
    net = FFTRadNetSNN(base, snn).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    net.snn.load_state_dict(ck['snn_state'])
    net.RA_decoder.load_state_dict(ck['decoder_state'])
    net.detection_header.load_state_dict(ck['det_head_state'])
    net.freespace.load_state_dict(ck['seg_head_state'])
    net.eval()
    enc, _, _, test_loader = build_loaders(config)

    results = {'protocol': 'run_FullEvaluation (threshold-swept)',
               'baseline_no_snn': dict(mAP=0.9849, mAR=0.9170, mIoU=74.64)}
    for ber in BERS:
        net.ber = ber
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_FullEvaluation(net, test_loader, enc)
        text = buf.getvalue()
        nums = {}
        for key in ['mAP', 'mAR', 'mIoU']:
            m = re.search(rf'{key}:?\s+([0-9.]+)', text)
            nums[key] = float(m.group(1)) if m else None
        results[f'ber{ber}'] = nums
        print(f"BER {ber:.2f}: mAP {nums['mAP']:.4f}  mAR {nums['mAR']:.4f}  "
              f"mIoU {nums['mIoU']:.2f}", flush=True)
    json.dump(results, open(args.out, 'w'), indent=1)
    print(f'saved: {args.out}')


if __name__ == '__main__':
    main()
