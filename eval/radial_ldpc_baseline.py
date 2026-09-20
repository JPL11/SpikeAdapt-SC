#!/usr/bin/env python3
"""LDPC-protected separation baseline for RADIal: 8-bit quantized features
protected by the ICC paper's regular LDPC codes (rate 1/2 and 1/3) over the
BSC, decoded by the unmodified pretrained FFTRadNet decoder.

Uses the offline code characterization (eval/seed_results/
ldpc_characterization.json): each 648-bit block independently fails with
FER(p); bits inside failed blocks are flipped with res_ber_failed(p)
(block-correlated corruption — same MC methodology as the ICC JPEG baseline).
Payload = quantized bits / code rate.

Usage: python eval/radial_ldpc_baseline.py
Output: eval/seed_results/radial_ldpc_baseline.json
"""

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

from utils.evaluation import run_FullEvaluation                    # noqa: E402
from eval.radial_feature_quant_baseline import CappedEncoder       # noqa: E402
from train.train_radial_snn import (CONFIG, SNN_LEVELS,            # noqa: E402
                                    build_loaders, load_base)

CHAR = os.path.join(ROOT, 'eval/seed_results/ldpc_characterization.json')
OUT_JSON = os.path.join(ROOT, 'eval/seed_results/radial_ldpc_baseline.json')
BERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
BITS = 8
BLOCK = 648
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def code_stats(char, code, p):
    """FER and residual BER of failed frames at channel BER p (nearest grid)."""
    stats = char[code]['stats']
    keys = sorted(stats.keys(), key=float)
    k = min(keys, key=lambda q: abs(float(q) - p))
    return stats[k]['fer'], stats[k]['res_ber_failed']


class FFTRadNetQuantLDPC(nn.Module):
    def __init__(self, base, fer, res_ber):
        super().__init__()
        self.FPN = base.FPN
        self.RA_decoder = base.RA_decoder
        self.detection_header = base.detection_header
        self.freespace = base.freespace
        self.fer, self.res_ber = fer, res_ber
        self.gen = torch.Generator(device=device).manual_seed(3)

    def corrupt(self, feat):
        """8-bit quantize; block-correlated post-LDPC bit flips; dequantize."""
        scale = feat.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
        levels = 2 ** BITS
        idx = torch.round((feat / scale + 1) / 2 * (levels - 1)).long()
        idx = idx.clamp(0, levels - 1)
        if self.fer > 0 and self.res_ber > 0:
            flat = idx.view(idx.shape[0], -1)
            n_vals = flat.shape[1]
            vals_per_block = max(1, BLOCK // BITS)
            n_blocks = (n_vals + vals_per_block - 1) // vals_per_block
            fail = (torch.rand(idx.shape[0], n_blocks, generator=self.gen,
                               device=device) < self.fer)
            fail_vals = fail.repeat_interleave(vals_per_block, dim=1)[:, :n_vals]
            flip = torch.zeros_like(flat)
            for k in range(BITS):
                flip += ((torch.rand(flat.shape, generator=self.gen,
                                     device=device) < self.res_ber)
                         & fail_vals).long() << k
            idx = (flat ^ flip).view(idx.shape)
        return (idx.float() / (levels - 1) * 2 - 1) * scale

    def forward(self, x):
        with torch.no_grad():
            feats = self.FPN(x)
            f = dict(feats)
            for k in SNN_LEVELS:
                f[k] = self.corrupt(feats[k])
            RA = self.RA_decoder(f)
            out = {'Detection': self.detection_header(RA)}
            out['Segmentation'] = self.freespace(F.interpolate(RA, (256, 224)))
        return out


def main():
    char = json.load(open(CHAR))
    config = json.load(open(CONFIG))
    base = load_base(config).to(device).eval()
    enc, _, _, test_loader = build_loaders(config)
    enc = CappedEncoder(enc, k=200)

    elems = 160 * 128 * 64 + 192 * 64 * 32 + 224 * 32 * 16
    results = {'protocol': 'run_FullEvaluation, top-200 cap',
               'quant_bits_per_frame': elems * BITS}
    if os.path.exists(OUT_JSON):
        results.update(json.load(open(OUT_JSON)))
    for code, label in [('r12_n648_dv3dc6', 'ldpc_r12'),
                        ('r13_n648_dv4dc6', 'ldpc_r13')]:
        rate = char[code]['rate']
        row = results.setdefault(label, {'payload_bits_per_frame':
                                         int(elems * BITS / rate)})
        for ber in BERS:
            key = f'ber{ber}'
            if key in row:
                continue
            fer, res = code_stats(char, code, ber)
            net = FFTRadNetQuantLDPC(base, fer, res).to(device).eval()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run_FullEvaluation(net, test_loader, enc)
            text = buf.getvalue()
            nums = {'fer': fer, 'res_ber_failed': res}
            for mk in ['mAP', 'mAR', 'mIoU']:
                m = re.search(rf'{mk}:?\s+([0-9.]+)', text)
                nums[mk] = float(m.group(1)) if m else None
            row[key] = nums
            json.dump(results, open(OUT_JSON, 'w'), indent=1)
            print(f"{label} BER {ber:.2f} (FER {fer:.3f}): mAP {nums['mAP']:.4f} "
                  f"mAR {nums['mAR']:.4f} mIoU {nums['mIoU']:.2f}", flush=True)
    print(f'saved: {OUT_JSON}')


if __name__ == '__main__':
    main()
