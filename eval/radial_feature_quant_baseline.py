#!/usr/bin/env python3
"""Separation baseline for the RADIal track: uniform-quantized float features
over a BSC, decoded by the UNMODIFIED pretrained FFTRadNet decoder+heads
(no retraining) — the digital-transport counterpart of the SNN bottleneck.

Per level (x2/x3/x4): per-sample symmetric uniform quantization to `bits`
bits/value; the integer codes' bits are flipped i.i.d. with prob BER; codes
are dequantized and fed onward. Full evaluation protocol (run_FullEvaluation)
to stay comparable with baseline 0.9849 and the SNN sweeps.

Usage: python eval/radial_feature_quant_baseline.py [--bits 8 4]
Output: eval/seed_results/radial_quant_baseline.json
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
from train.train_radial_snn import (CONFIG, SNN_LEVELS,    # noqa: E402
                                    build_loaders, load_base)

OUT_JSON = os.path.join(ROOT, 'eval/seed_results/radial_quant_baseline.json')
BERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def quantize_bsc(feat, bits, ber, gen):
    """Per-sample symmetric uniform quantization + i.i.d. bit flips."""
    B = feat.shape[0]
    scale = feat.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
    levels = 2 ** bits
    idx = torch.round((feat / scale + 1) / 2 * (levels - 1)).long()
    idx = idx.clamp(0, levels - 1)
    if ber > 0:
        flip = torch.zeros_like(idx)
        for k in range(bits):
            flip += (torch.rand(idx.shape, generator=gen,
                                device=idx.device) < ber).long() << k
        idx = idx ^ flip
    return (idx.float() / (levels - 1) * 2 - 1) * scale


class FFTRadNetQuant(nn.Module):
    def __init__(self, base, bits):
        super().__init__()
        self.FPN = base.FPN
        self.RA_decoder = base.RA_decoder
        self.detection_header = base.detection_header
        self.freespace = base.freespace
        self.bits = bits
        self.ber = 0.0
        self.gen = torch.Generator(device=device).manual_seed(3)

    def forward(self, x):
        with torch.no_grad():
            feats = self.FPN(x)
            f = dict(feats)
            for k in SNN_LEVELS:
                f[k] = quantize_bsc(feats[k], self.bits, self.ber, self.gen)
            RA = self.RA_decoder(f)
            out = {'Detection': self.detection_header(RA)}
            out['Segmentation'] = self.freespace(F.interpolate(RA, (256, 224)))
        return out


class CappedEncoder:
    """Proxy for ra_encoder that caps decoded detections at top-K by
    confidence — collapsed transports otherwise emit thousands of garbage
    detections per frame and the polygon-IoU metric grinds for hours.
    Standard practice (COCO caps at 100/image); does not affect healthy
    operating points, which emit far fewer than K."""

    def __init__(self, enc, k=200):
        self._enc = enc
        self.k = k

    def __getattr__(self, name):
        return getattr(self._enc, name)

    def decode(self, pred_map, threshold):
        coords = self._enc.decode(pred_map, threshold)
        if len(coords) > self.k:
            coords = sorted(coords, key=lambda c: -c[2])[:self.k]
        return coords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bits', type=int, nargs='+', default=[8, 4])
    args = ap.parse_args()
    config = json.load(open(CONFIG))
    base = load_base(config).to(device).eval()
    enc, _, _, test_loader = build_loaders(config)
    enc = CappedEncoder(enc, k=200)

    # payload per frame: quantized float features vs full float32
    elems = {'x2': 160 * 128 * 64, 'x3': 192 * 64 * 32, 'x4': 224 * 32 * 16}
    total_elems = sum(elems.values())
    results = {'protocol': 'run_FullEvaluation (threshold-swept)',
               'feature_elems': elems,
               'float32_bits_per_frame': total_elems * 32}
    for bits in args.bits:
        net = FFTRadNetQuant(base, bits).to(device).eval()
        row = {'payload_bits_per_frame': total_elems * bits}
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
            row[f'ber{ber}'] = nums
            print(f"bits={bits} BER {ber:.2f}: mAP {nums['mAP']:.4f}  "
                  f"mAR {nums['mAR']:.4f}  mIoU {nums['mIoU']:.2f}", flush=True)
        results[f'q{bits}'] = row
    json.dump(results, open(OUT_JSON, 'w'), indent=1)
    print(f'saved: {OUT_JSON}')


if __name__ == '__main__':
    main()
