#!/usr/bin/env python3
"""E1 (review round 1): membrane side-channel ablation for the RADIal SNN.

The trained bottleneck transmits, besides binary spikes, an 18-ch continuous
membrane summary per timestep that the original evaluation passed through
AWGN(0.5*BER) and counted at zero bits. This script measures, WITHOUT
retraining, three receiver variants on the seed-42 model:

  asis   : original behavior (spikes BSC, mems AWGN 0.5*BER)  [reference]
  zeroed : mems zeroed at the receiver (spikes-only transport)
  qbsc   : mems quantized to --mem-bits (default 4) per value, bits through
           the SAME BSC as the spikes; payload counted.

Full threshold-swept protocol, 744-frame test split.
Usage: python eval/radial_mem_ablation.py [--mem-bits 4]
Output: eval/seed_results/radial_mem_ablation.json (incremental)
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

OUT = os.path.join(ROOT, 'eval/seed_results/radial_mem_ablation.json')
BERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
C_SPIKE, T = 36, 8
DIMS = {'x2': (128, 64), 'x3': (64, 32), 'x4': (32, 16)}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def quant_bsc_mem(m, bits, ber, gen):
    """Uniform quantization of tanh-bounded mems + i.i.d. bit flips."""
    levels = 2 ** bits
    idx = torch.round((m.clamp(-1, 1) + 1) / 2 * (levels - 1)).long()
    if ber > 0:
        flip = torch.zeros_like(idx)
        for k in range(bits):
            flip += (torch.rand(idx.shape, generator=gen,
                                device=idx.device) < ber).long() << k
        idx = idx ^ flip
    return idx.float() / (levels - 1) * 2 - 1


class AblationNet(nn.Module):
    def __init__(self, inner, mode, mem_bits):
        super().__init__()
        self.net = inner
        self.mode = mode
        self.mem_bits = mem_bits
        self.ber = 0.0
        self.gen = torch.Generator(device=device).manual_seed(7)

    def forward(self, x):
        n = self.net
        with torch.no_grad():
            feats = n.FPN(x)
        f = dict(feats)
        for k, lev in zip(SNN_LEVELS, n.snn.levels):
            spikes, mems = lev.encoder(feats[k])
            imp = lev.scorer(spikes, self.ber)
            msp, mmem, mask = lev.masker(spikes, mems, imp, False, None)
            recv_sp = [lev.channel(s, self.ber) for s in msp]
            if self.mode == 'asis':
                recv_mem = [m + torch.randn_like(m) * self.ber * 0.5
                            if self.ber > 0 else m for m in mmem]
            elif self.mode == 'zeroed':
                recv_mem = [torch.zeros_like(m) for m in mmem]
            elif self.mode == 'qbsc':
                recv_mem = [quant_bsc_mem(m, self.mem_bits, self.ber, self.gen)
                            * (mask if mask.shape == m.shape else 1.0)
                            for m in mmem]
            f[k] = lev.decoder(recv_sp, recv_mem)
        RA = n.RA_decoder(f)
        out = {'Detection': n.detection_header(RA)}
        out['Segmentation'] = n.freespace(F.interpolate(RA, (256, 224)))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mem-bits', dest='mem_bits', type=int, default=4)
    ap.add_argument('--modes', nargs='+', default=['zeroed', 'qbsc', 'asis'])
    args = ap.parse_args()
    config = json.load(open(CONFIG))
    torch.manual_seed(config['seed'])
    base = load_base(config)
    snn = SpikeAdaptSC_Det_Multi(CHANNEL_SIZES, C_spike=C_SPIKE, T=T,
                                 target_rate=0.75)
    inner = FFTRadNetSNN(base, snn).to(device)
    ck = torch.load(SAVE, map_location=device, weights_only=False)
    inner.snn.load_state_dict(ck['snn_state'])
    inner.RA_decoder.load_state_dict(ck['decoder_state'])
    inner.detection_header.load_state_dict(ck['det_head_state'])
    inner.freespace.load_state_dict(ck['seg_head_state'])
    inner.eval()
    enc, _, _, test_loader = build_loaders(config)

    # payload accounting: mems add C_MEM=18 channels x tx_rate x T' values
    c_mem = C_SPIKE // 2
    mem_vals = sum(c_mem * h * w for h, w in DIMS.values()) * T * 0.75
    results = {'config': dict(mem_bits=args.mem_bits, seed=42,
                              mem_values_per_frame=int(mem_vals),
                              mem_payload_bits_qbsc=int(mem_vals * args.mem_bits),
                              spike_payload_bits=2322432)}
    if os.path.exists(OUT):
        results.update(json.load(open(OUT)))
    for mode in args.modes:
        row = results.setdefault(mode, {})
        net = AblationNet(inner, mode, args.mem_bits).to(device).eval()
        for ber in BERS:
            key = f'ber{ber}'
            if key in row:
                continue
            net.ber = ber
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run_FullEvaluation(net, test_loader, enc)
            text = buf.getvalue()
            nums = {}
            for mk in ['mAP', 'mAR', 'mIoU']:
                m = re.search(rf'{mk}:?\s+([0-9.]+)', text)
                nums[mk] = float(m.group(1)) if m else None
            row[key] = nums
            json.dump(results, open(OUT, 'w'), indent=1)
            print(f"{mode} BER {ber:.2f}: mAP {nums['mAP']:.4f} "
                  f"mAR {nums['mAR']:.4f} mIoU {nums['mIoU']:.2f}", flush=True)
    print(f'saved: {OUT}')


if __name__ == '__main__':
    main()
