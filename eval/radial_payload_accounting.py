#!/usr/bin/env python3
"""Payload accounting for the RADIal SNN bottleneck vs digital alternatives.

Spike payload per frame = sum over levels of C_spike * (H*W) * T * tx_rate,
where tx_rate is the measured spatial importance-mask keep fraction (target
0.75). Compared against float32 feature transport and b-bit quantized
features. Also reports the spike firing rate (fraction of 1s among
transmitted bits) for the energy/sparsity discussion.

Usage: python eval/radial_payload_accounting.py [--ckpt runs/fftradnet_snn_radial.pth]
Output: eval/seed_results/radial_payload_accounting.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'radial_repo', 'FFTRadNet'))

from train.train_dota_v6d2d import SpikeAdaptSC_Det_Multi   # noqa: E402
from train.train_radial_snn import (CHANNEL_SIZES, CONFIG, SAVE,  # noqa: E402
                                    SNN_LEVELS, FFTRadNetSNN,
                                    build_loaders, load_base)

OUT_JSON = os.path.join(ROOT, 'eval/seed_results/radial_payload_accounting.json')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

C_SPIKE, T = 36, 8
DIMS = {'x2': (128, 64), 'x3': (64, 32), 'x4': (32, 16)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=SAVE)
    ap.add_argument('--batches', type=int, default=25)
    args = ap.parse_args()
    config = json.load(open(CONFIG))
    torch.manual_seed(config['seed'])
    base = load_base(config)
    snn = SpikeAdaptSC_Det_Multi(CHANNEL_SIZES, C_spike=C_SPIKE, T=T,
                                 target_rate=0.75)
    net = FFTRadNetSNN(base, snn).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    net.snn.load_state_dict(ck['snn_state'])
    net.RA_decoder.load_state_dict(ck['decoder_state'])
    net.detection_header.load_state_dict(ck['det_head_state'])
    net.freespace.load_state_dict(ck['seg_head_state'])
    net.eval()
    _, _, _, test_loader = build_loaders(config)

    tx_rates = {k: [] for k in SNN_LEVELS}
    fire_rates = {k: [] for k in SNN_LEVELS}
    net.ber = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if i >= args.batches:
                break
            x = data[0].to(device).float()
            feats = net.FPN(x)
            for k, lev in zip(SNN_LEVELS, net.snn.levels):
                spikes, mems = lev.encoder(feats[k])
                imp = lev.scorer(spikes, 0.0)
                msp, _, mask = lev.masker(spikes, mems, imp, False, None)
                tx_rates[k].append(mask.mean().item())
                kept = mask.sum() * C_SPIKE * len(msp)
                ones = sum(s.sum() for s in msp)
                fire_rates[k].append((ones / kept.clamp_min(1)).item())

    spike_bits = 0
    per_level = {}
    for k in SNN_LEVELS:
        h, w = DIMS[k]
        tx = float(np.mean(tx_rates[k]))
        bits = C_SPIKE * h * w * T * tx
        per_level[k] = dict(tx_rate=round(tx, 4),
                            firing_rate=round(float(np.mean(fire_rates[k])), 4),
                            bits_per_frame=int(bits))
        spike_bits += bits

    float_bits = sum(c * h * w * 32 for c, (h, w) in
                     zip(CHANNEL_SIZES, DIMS.values()))
    q8_bits = float_bits // 4
    q4_bits = float_bits // 8
    res = dict(per_level=per_level,
               spike_bits_per_frame=int(spike_bits),
               float32_bits_per_frame=int(float_bits),
               quant8_bits_per_frame=int(q8_bits),
               quant4_bits_per_frame=int(q4_bits),
               ratio_vs_float32=round(float_bits / spike_bits, 2),
               ratio_vs_quant8=round(q8_bits / spike_bits, 2),
               ratio_vs_quant4=round(q4_bits / spike_bits, 2),
               T=T, C_spike=C_SPIKE, ckpt=args.ckpt)
    json.dump(res, open(OUT_JSON, 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
