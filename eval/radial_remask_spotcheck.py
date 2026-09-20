#!/usr/bin/env python3
"""Eq(3) reviewer check: receiver re-masking spot check.

The shipped pipeline applies the BSC to ALL positions of the masked spike
maps (flips at unselected positions reach the decoder). The physically
consistent receiver scatters received bits into the selected support and
zero-fills the rest: y = m * (s XOR e). This script evaluates the SAME
spikes-only checkpoints (all 3 seeds) under the re-masking receiver on the
RADIal test split for a few (rho, T', BER) cells and prints the delta
against the cached all-position numbers in radial_grid_valtest.json.

No retraining; eval-only receiver change.

Usage: SPIKES_ONLY=1 python eval/radial_remask_spotcheck.py
Output: eval/seed_results/radial_remask_spotcheck.json
"""
import contextlib
import io
import json
import os
import re
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'radial_repo', 'FFTRadNet'))

os.environ['SPIKES_ONLY'] = '1'

from utils.evaluation import run_FullEvaluation                     # noqa: E402
from train.train_dota_v6d2d import SpikeAdaptSC_Det_Multi           # noqa: E402
from train.train_radial_snn import (CHANNEL_SIZES, CONFIG, SNN_LEVELS,  # noqa: E402
                                    FFTRadNetSNN, build_loaders, load_base)
from eval.radial_grid_rho_T import GridNet, C_SPIKE, T              # noqa: E402

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CELLS = [(0.75, 8, 0.30), (0.75, 8, 0.15), (0.5, 8, 0.40)]
# Panel-3 P0-a extension. (a) Mission-visited cells: union over seeds of the
# validation-selected joint policy (radial_closedloop_valtest.build_policy,
# eps=0.01) at the profile-visited BER bins with p>0 (p=0 is an identity
# under re-masking: no flips to scatter). (b) Two extra rho0.75/T8 cells to
# locate the SNN-vs-fair crossover under y_link. The fair 1-bit control
# needs NO re-scoring here: its codec forward already computes
# recv = ((send+flip)%2)*mask (train/train_radial_adaptive1bit.py), i.e. the
# support-aware receiver y = m*(s XOR e) by construction, so
# radial_fair1bit_grid_valtest.json is y_link already.
MISSION_CELLS = [(0.5, 6, 0.05), (0.5, 6, 0.1), (0.5, 6, 0.15), (0.5, 6, 0.2),
                 (0.5, 8, 0.25), (0.5, 8, 0.3), (0.5, 8, 0.35)]
CROSSOVER_CELLS = [(0.75, 8, 0.2), (0.75, 8, 0.25)]
ALL_CELLS = CELLS + MISSION_CELLS + CROSSOVER_CELLS
SEED_CKPT = {'42': 'runs/fftradnet_snn_radial_so.pth',
             '123': 'runs/fftradnet_snn_radial_so_s123.pth',
             '456': 'runs/fftradnet_snn_radial_so_s456.pth'}
CACHED = os.path.join(ROOT, 'eval/seed_results/radial_grid_valtest.json')
OUT = os.path.join(ROOT, 'eval/seed_results/radial_remask_spotcheck.json')


class RemaskNet(GridNet):
    """GridNet with the physically consistent receiver: y = m * (s XOR e)."""

    def forward(self, x):
        n = self.net
        with torch.no_grad():
            feats = n.FPN(x)
        f = dict(feats)
        for k, lev in zip(SNN_LEVELS, n.snn.levels):
            spikes, mems = lev.encoder(feats[k])
            imp = lev.scorer(spikes, self.ber)
            msp, mmem, mask = lev.masker(spikes, mems, imp, False, self.rho)
            msp, mmem = msp[:self.tprime], mmem[:self.tprime]
            recv_sp = [mask * lev.channel(s, self.ber) for s in msp]
            recv_mem = [torch.zeros_like(m) for m in mmem]
            f[k] = lev.decoder(recv_sp, recv_mem)
        RA = n.RA_decoder(f)
        out = {'Detection': n.detection_header(RA)}
        out['Segmentation'] = n.freespace(F.interpolate(RA, (256, 224)))
        return out


def full_eval(net, loader, enc):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_FullEvaluation(net, loader, enc)
    text = buf.getvalue()
    return {mk: (float(m.group(1)) if (m := re.search(
        rf'{mk}:?\s+([0-9.]+)', text)) else None)
        for mk in ['mAP', 'mAR', 'mIoU']}


def main():
    config = json.load(open(CONFIG))
    torch.manual_seed(config['seed'])
    enc, _, _, test_loader = build_loaders(config)
    cached_all = json.load(open(CACHED))

    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    if res and not any(k in SEED_CKPT for k in res):
        res = {'42': res}   # migrate the flat single-seed layout
        json.dump(res, open(OUT, 'w'), indent=1)

    for seed, ck_rel in SEED_CKPT.items():
        node = res.setdefault(seed, {})
        if all(f'rho{r}_T{t}_ber{b}' in node for r, t, b in ALL_CELLS):
            print(f'seed {seed}: done, skip', flush=True)
            continue
        base = load_base(config)
        snn = SpikeAdaptSC_Det_Multi(CHANNEL_SIZES, C_spike=C_SPIKE, T=T,
                                     target_rate=0.75)
        inner = FFTRadNetSNN(base, snn).to(device)
        ck = torch.load(os.path.join(ROOT, ck_rel), map_location=device,
                        weights_only=False)
        inner.snn.load_state_dict(ck['snn_state'])
        inner.RA_decoder.load_state_dict(ck['decoder_state'])
        inner.detection_header.load_state_dict(ck['det_head_state'])
        inner.freespace.load_state_dict(ck['seg_head_state'])
        inner.eval()
        net = RemaskNet(inner).to(device).eval()

        cached = cached_all[seed]['test']
        for rho, tp, ber in ALL_CELLS:
            key = f'rho{rho}_T{tp}_ber{ber}'
            if key in node:
                continue
            net.rho, net.tprime, net.ber = rho, tp, ber
            m = full_eval(net, test_loader, enc)
            m['cached_allpos'] = {mk: cached[key][mk]
                                  for mk in ['mAP', 'mAR', 'mIoU']}
            node[key] = m
            json.dump(res, open(OUT, 'w'), indent=1)
            print(f's{seed} {key}: remask mAP {m["mAP"]:.4f} '
                  f'mAR {m["mAR"]:.4f} '
                  f'| allpos mAP {m["cached_allpos"]["mAP"]:.4f} '
                  f'mAR {m["cached_allpos"]["mAR"]:.4f}', flush=True)
        del net, inner, base, snn
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    print('saved', OUT, flush=True)


if __name__ == '__main__':
    main()
