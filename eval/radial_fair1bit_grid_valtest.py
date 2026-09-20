#!/usr/bin/env python3
"""P0-3 (ICASSP panel): (rho, T') x BER val/test grid for the FAIR 1-bit radar
control, in the exact format of radial_grid_valtest.json so
radial_closedloop_valtest-style mission machinery can consume it unchanged.

Loads the encoder-matched, truncation-trained grouped-1bit checkpoints
(runs/fftradnet_adaptive1bit_radial_s{42,123,456}.pth), evaluates the RADIal
FULL protocol per cell on BOTH the validation (786) and test (744) splits.
Incremental/resumable (saves after every cell).

Grid: rho in {0.5, 0.75, 1.0} x T' in {4, 6, 8} x BER in {0..0.40} (9 pts).
Payload convention matches the SNN grid: full_spatial * rho * T' with
full_spatial = 36 * (128*64 + 64*32 + 32*16) = 387,072 bits per group.

Usage: python eval/radial_fair1bit_grid_valtest.py
Output: eval/seed_results/radial_fair1bit_grid_valtest.json
        keyed [seed][val|test]['rho{r}_T{t}_ber{b}'] -> mAP/mAR/mIoU/payload_bits
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

from utils.evaluation import run_FullEvaluation                     # noqa: E402
from eval.radial_feature_quant_baseline import CappedEncoder        # noqa: E402
from train.train_radial_snn import CONFIG, CHANNEL_SIZES, build_loaders, load_base  # noqa: E402
from train.train_radial_adaptive1bit import (Adaptive1bitDet_Multi,  # noqa: E402
                                             FFTRadNetAdaptive1bit)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RHOS = [0.5, 0.75, 1.0]
TPRIMES = [4, 6, 8]
BERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
FULL_SPATIAL = 36 * (128 * 64 + 64 * 32 + 32 * 16)   # 387,072 bits per group
SEED_CKPT = {'42': 'runs/fftradnet_adaptive1bit_radial_s42.pth',
             '123': 'runs/fftradnet_adaptive1bit_radial_s123.pth',
             '456': 'runs/fftradnet_adaptive1bit_radial_s456.pth'}
OUT = os.path.join(ROOT, 'eval/seed_results/radial_fair1bit_grid_valtest.json')


def full_eval(net, loader, enc_c):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_FullEvaluation(net, loader, enc_c)
    text = buf.getvalue()
    out = {}
    for mk in ['mAP', 'mAR', 'mIoU']:
        m = re.search(rf'{mk}:?\s+([0-9.]+)', text)
        out[mk] = float(m.group(1)) if m else None
    return out


def main():
    config = json.load(open(CONFIG))
    enc, _, val_loader, test_loader = build_loaders(config, 4)
    enc_c = CappedEncoder(enc, k=200)
    loaders = {'val': val_loader, 'test': test_loader}
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for seed, ck_path in SEED_CKPT.items():
        node = res.setdefault(seed, {})
        base = load_base(config)
        codec = Adaptive1bitDet_Multi(CHANNEL_SIZES, target_rate=0.75)
        net = FFTRadNetAdaptive1bit(base, codec).to(device)
        ck = torch.load(os.path.join(ROOT, ck_path), map_location=device,
                        weights_only=False)
        net.codec.load_state_dict(ck['codec_state'])
        net.RA_decoder.load_state_dict(ck['decoder_state'])
        net.detection_header.load_state_dict(ck['det_head_state'])
        net.freespace.load_state_dict(ck['seg_head_state'])
        net.eval()
        for split in ['val', 'test']:
            sn = node.setdefault(split, {})
            for rho in RHOS:
                for tp in TPRIMES:
                    for ber in BERS:
                        key = f'rho{rho}_T{tp}_ber{ber}'
                        if key in sn:
                            continue
                        net.ber = ber; net.rho = rho; net.tprime = tp
                        m = full_eval(net, loaders[split], enc_c)
                        m['payload_bits'] = int(round(FULL_SPATIAL * rho * tp))
                        sn[key] = m
                        json.dump(res, open(OUT, 'w'), indent=1)
                        print(f's{seed} {split} {key}: mAP {m["mAP"]:.4f} '
                              f'mAR {m["mAR"]:.4f}', flush=True)
        del net, base, codec; torch.cuda.empty_cache()
    print(f'saved {OUT}', flush=True)


if __name__ == '__main__':
    main()
