#!/usr/bin/env python3
"""(rho, T', BER) grid on RADIal for the trained SNN bottleneck — the joint
spatial+temporal rate-adaptation surface that feeds the closed-loop policy.

rho: importance-mask keep fraction override (spatial rate knob).
T': temporal truncation — only the first T' of T=8 timesteps are transmitted
    (training-free, as in the ICC paper; decoder consumes truncated lists).
Full evaluation protocol (run_FullEvaluation) throughout.

Incremental: each grid point is appended to the output JSON as it completes,
so an interrupted run resumes by skipping finished points.

Usage: python eval/radial_grid_rho_T.py [--ckpt ...] [--out ...]
Output: eval/seed_results/radial_grid_rho_T.json
"""

import argparse
import contextlib
import io
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'radial_repo', 'FFTRadNet'))

from utils.evaluation import run_FullEvaluation            # noqa: E402
from train.train_dota_v6d2d import SpikeAdaptSC_Det_Multi  # noqa: E402
from train.train_radial_snn import (CHANNEL_SIZES, CONFIG, SAVE,  # noqa: E402
                                    SNN_LEVELS, FFTRadNetSNN,
                                    build_loaders, load_base)

OUT_DEFAULT = os.path.join(ROOT, 'eval/seed_results/radial_grid_rho_T.json')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RHOS = [0.5, 0.75, 1.0]
TPRIMES = [4, 6, 8]
BERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
C_SPIKE, T = 36, 8
DIMS = {'x2': (128, 64), 'x3': (64, 32), 'x4': (32, 16)}


class GridNet(nn.Module):
    """FFTRadNetSNN with explicit (rho, T') knobs; run_FullEvaluation-compatible."""

    def __init__(self, base_net):
        super().__init__()
        self.net = base_net
        self.rho = 1.0
        self.tprime = T
        self.ber = 0.0

    def forward(self, x):
        n = self.net
        with torch.no_grad():
            feats = n.FPN(x)
        f = dict(feats)
        for k, lev in zip(SNN_LEVELS, n.snn.levels):
            spikes, mems = lev.encoder(feats[k])
            imp = lev.scorer(spikes, self.ber)
            msp, mmem, mask = lev.masker(spikes, mems, imp, False, self.rho)
            msp, mmem = msp[:self.tprime], mmem[:self.tprime]   # T' truncation
            recv_sp = [lev.channel(s, self.ber) for s in msp]
            if os.environ.get('SPIKES_ONLY', '') == '1':
                recv_mem = [torch.zeros_like(m) for m in mmem]
            else:
                recv_mem = [m + torch.randn_like(m) * self.ber * 0.5
                            if self.ber > 0 else m for m in mmem]
            f[k] = lev.decoder(recv_sp, recv_mem)
        RA = n.RA_decoder(f)
        out = {'Detection': n.detection_header(RA)}
        out['Segmentation'] = n.freespace(F.interpolate(RA, (256, 224)))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=SAVE)
    ap.add_argument('--out', default=OUT_DEFAULT)
    ap.add_argument('--ext', action='store_true',
                    help='extend BER grid with 0.35 and 0.40 (E3, round-1 review)')
    args = ap.parse_args()
    if args.ext:
        BERS.extend([0.35, 0.40])
    config = json.load(open(CONFIG))
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    base = load_base(config)
    snn = SpikeAdaptSC_Det_Multi(CHANNEL_SIZES, C_spike=C_SPIKE, T=T,
                                 target_rate=0.75)
    inner = FFTRadNetSNN(base, snn).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    inner.snn.load_state_dict(ck['snn_state'])
    inner.RA_decoder.load_state_dict(ck['decoder_state'])
    inner.detection_header.load_state_dict(ck['det_head_state'])
    inner.freespace.load_state_dict(ck['seg_head_state'])
    net = GridNet(inner).to(device).eval()
    enc, _, _, test_loader = build_loaders(config)

    results = {}
    if os.path.exists(args.out):
        results = json.load(open(args.out))
        print(f'resuming: {len(results)} points done', flush=True)
    results.setdefault('config', dict(ckpt=args.ckpt, C_spike=C_SPIKE, T=T,
                                      rhos=RHOS, tprimes=TPRIMES, bers=BERS))

    full_spatial = sum(C_SPIKE * h * w for h, w in DIMS.values())
    for rho in RHOS:
        for tp in TPRIMES:
            for ber in BERS:
                key = f'rho{rho}_T{tp}_ber{ber}'
                if key in results:
                    continue
                net.rho, net.tprime, net.ber = rho, tp, ber
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    run_FullEvaluation(net, test_loader, enc)
                text = buf.getvalue()
                nums = {}
                for mk in ['mAP', 'mAR', 'mIoU']:
                    m = re.search(rf'{mk}:?\s+([0-9.]+)', text)
                    nums[mk] = float(m.group(1)) if m else None
                nums['payload_bits'] = int(full_spatial * rho * tp)
                results[key] = nums
                json.dump(results, open(args.out, 'w'), indent=1)
                print(f"{key}: mAP {nums['mAP']:.4f} mAR {nums['mAR']:.4f} "
                      f"mIoU {nums['mIoU']:.2f} "
                      f"payload {nums['payload_bits']/1e6:.2f} Mbit", flush=True)
    print(f'grid complete: {args.out}')


if __name__ == '__main__':
    main()
