#!/usr/bin/env python3
"""Leakage-migration (array-paper review fix #4): (rho, T', BER) grids on the
RADIal spikes-only models evaluated on BOTH the validation and test splits.

Policy is later built on the VAL surface; accuracy is reported on the TEST
surface (mirrors the ICC valproto protocol). No retraining: the RADIal SNN
checkpoints save the final model (no test-based selection), and the official
sequence split already exposes a validation loader.

Grid: rho in {0.5,0.75,1.0} x T' in {4,6,8} x BER in [0,0.40]; 3 seeds.
Incremental per (seed,split,cell). Output keyed [seed][val|test][key].

Usage: python eval/radial_grid_valtest.py
Output: eval/seed_results/radial_grid_valtest_{so,s123_so,s456_so as tags}.json
        -> single file eval/seed_results/radial_grid_valtest.json keyed by seed
"""

import contextlib
import io
import json
import os
import re
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'radial_repo', 'FFTRadNet'))

from utils.evaluation import run_FullEvaluation                     # noqa: E402
from train.train_dota_v6d2d import SpikeAdaptSC_Det_Multi           # noqa: E402
from train.train_radial_snn import (CHANNEL_SIZES, CONFIG, SNN_LEVELS,  # noqa: E402
                                    FFTRadNetSNN, build_loaders, load_base)
from eval.radial_grid_rho_T import GridNet, C_SPIKE, T, DIMS        # noqa: E402

OUT = os.path.join(ROOT, 'eval/seed_results/radial_grid_valtest.json')
RHOS = [0.5, 0.75, 1.0]
TPRIMES = [4, 6, 8]
BERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
SEED_CKPT = {'42': 'runs/fftradnet_snn_radial_so.pth',
             '123': 'runs/fftradnet_snn_radial_so_s123.pth',
             '456': 'runs/fftradnet_snn_radial_so_s456.pth'}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ['SPIKES_ONLY'] = '1'   # models are spikes-only


def load_net(config, ckpt):
    base = load_base(config)
    snn = SpikeAdaptSC_Det_Multi(CHANNEL_SIZES, C_spike=C_SPIKE, T=T,
                                 target_rate=0.75)
    inner = FFTRadNetSNN(base, snn).to(device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    inner.snn.load_state_dict(ck['snn_state'])
    inner.RA_decoder.load_state_dict(ck['decoder_state'])
    inner.detection_header.load_state_dict(ck['det_head_state'])
    inner.freespace.load_state_dict(ck['seg_head_state'])
    inner.eval()
    return GridNet(inner).to(device).eval()


def main():
    config = json.load(open(CONFIG))
    torch.manual_seed(config['seed'])
    enc, _, val_loader, test_loader = build_loaders(config)
    loaders = {'val': val_loader, 'test': test_loader}
    full_spatial = sum(C_SPIKE * h * w for h, w in DIMS.values())
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for seed, ckpt in SEED_CKPT.items():
        if not os.path.exists(ckpt):
            print(f'seed {seed}: {ckpt} missing, skip', flush=True)
            continue
        net = None
        node = res.setdefault(seed, {})
        for split in ['val', 'test']:
            sp = node.setdefault(split, {})
            for rho in RHOS:
                for tp in TPRIMES:
                    for ber in BERS:
                        key = f'rho{rho}_T{tp}_ber{ber}'
                        if key in sp:
                            continue
                        if net is None:
                            net = load_net(config, ckpt)
                        net.rho, net.tprime, net.ber = rho, tp, ber
                        buf = io.StringIO()
                        with contextlib.redirect_stdout(buf):
                            run_FullEvaluation(net, loaders[split], enc)
                        text = buf.getvalue()
                        nums = {'payload_bits': int(full_spatial * rho * tp)}
                        for mk in ['mAP', 'mAR', 'mIoU']:
                            m = re.search(rf'{mk}:?\s+([0-9.]+)', text)
                            nums[mk] = float(m.group(1)) if m else None
                        sp[key] = nums
                        json.dump(res, open(OUT, 'w'))
                        print(f'{seed} {split} r{rho} T{tp} b{ber}: '
                              f"mAP {nums['mAP']:.3f}", flush=True)
        del net
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    print('saved:', OUT)


if __name__ == '__main__':
    main()
