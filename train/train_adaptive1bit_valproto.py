#!/usr/bin/env python3
"""Adaptive-1bit baseline (validation protocol): the strongest fair learned
DISCRETE task-oriented competitor for SpikeAdapt-SC.

It is the binary analogue of the (continuous) Adaptive-JSCC baseline: a learned
1-bit CNN codec (STE binarization, the same encoder/decoder as CNN-1bit) plus
the *same* BER-conditioned NoiseAwareScorer + LearnedBlockMask that SpikeAdapt-SC
uses. So the only difference from the main method is spike-code + free temporal
axis (T') vs. a plain binary CNN bottleneck: this isolates what the spiking
temporal axis buys once the discrete baseline is given identical adaptive spatial
machinery.

Construction (parallel to train_jscc_adaptive.py extending fixed JSCC):
  frozen front-end (valp backbone) -> frozen CNN-1bit encoder/decoder (valp 1bit
  checkpoint) -> TRAIN only {scorer, block_mask, back}. BSC-native, BER curriculum.

Channel/mask order matches models/spikeadapt_sc.py exactly: mask -> BSC(masked)
-> decoder re-masks, so only kept bits traverse the channel and flips in dropped
blocks are discarded (RX knows the mask bitmap, per the paper's side channel).

Val protocol: 10% per-seed val carve (test transforms), checkpoint selected on
val, TEST scored once over the full (rho, BER) grid. VAL grid is also dumped so
a val-selected spatial rho-policy can be built downstream (Table III).

Usage: /home/jpli/miniconda/envs/semcom/bin/python train/train_adaptive1bit_valproto.py \
           --seeds 42 123 456 ... --datasets aid resisc45
Output: eval/seed_results/valproto_adaptive1bit.json
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'models'))

# harness helpers (loaders, backbone loading, val-select loop) reused verbatim
from train_baselines_valproto import (make_loaders, load_front_back, acc_on,
                                       select_loop, BERS, device, ROOT)
from train_1bit_baseline import BinaryCNN_Encoder, BinaryCNN_Decoder
from noise_aware_scorer import NoiseAwareScorer
from train_jscc_adaptive import LearnedBlockMask   # standalone copy, == spikeadapt_sc

RHOS = [0.625, 0.75, 0.875, 1.0]           # matches the paper spatial grid C
# BER curriculum: include high-BER so the scorer learns real channel adaptation
TRAIN_BERS = [0.0, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


class Adaptive1bit(nn.Module):
    """CNN-1bit codec + learned BER-conditioned spatial masking."""

    def __init__(self, C_in=1024, C1=256, C2=36, target_rate=0.75):
        super().__init__()
        self.encoder = BinaryCNN_Encoder(C_in, C1, C2)
        self.decoder = BinaryCNN_Decoder(C_in, C1, C2)
        self.scorer = NoiseAwareScorer(C_spike=C2, hidden=32)
        self.block_mask = LearnedBlockMask(target_rate, 0.5)

    def forward(self, feat, ber=0.0, target_rate_override=None):
        z = self.encoder(feat)                       # binary {0,1}, B x C2 x H x W
        importance = self.scorer([z], ber)           # scorer expects a list of tensors
        if target_rate_override is not None:
            old = self.block_mask.target_rate
            self.block_mask.target_rate = target_rate_override
            mask, tx = self.block_mask(importance, training=False)
            self.block_mask.target_rate = old
        else:
            mask, tx = self.block_mask(importance, training=self.training)
        z_masked = z * mask                          # drop non-selected blocks
        # BSC on the masked tensor, then re-mask (matches spikeadapt_sc.py):
        # flips inside dropped blocks are discarded; only kept bits go through.
        flip = (torch.rand_like(z_masked) < ber).float()
        recv = ((z_masked + flip) % 2) * mask
        Fp = self.decoder(recv)
        return Fp, {'tx_rate': tx.item(), 'mask': mask, 'importance': importance}


def _latest_ckpt(snap_dir, prefix):
    if not os.path.isdir(snap_dir):
        return None
    cks = sorted([f for f in os.listdir(snap_dir) if f.startswith(prefix)],
                 key=lambda x: float(x.split('_')[-1][:-4]))
    return os.path.join(snap_dir, cks[-1]) if cks else None


def run_adaptive1bit(ds, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_loader, val_loader, test_loader, ncls = make_loaders(ds, seed)
    front, back = load_front_back(ds, ncls, seed)
    if front is None:
        print(f'  {ds} seed {seed}: valp backbone missing, skip'); return None

    # load the frozen CNN-1bit encoder/decoder for this (ds, seed)
    one_bit = _latest_ckpt(f'./snapshots_{ds}valp_1bit_seed{seed}/', '1bit_best_')
    if one_bit is None:
        print(f'  {ds} seed {seed}: valp 1bit checkpoint missing, skip'); return None
    obck = torch.load(one_bit, map_location=device, weights_only=False)

    model = Adaptive1bit(C_in=1024, C1=256, C2=36, target_rate=0.75).to(device)
    enc = {k.replace('encoder.', ''): v for k, v in obck['model'].items()
           if k.startswith('encoder.')}
    dec = {k.replace('decoder.', ''): v for k, v in obck['model'].items()
           if k.startswith('decoder.')}
    # init codec from the pretrained CNN-1bit, but train the WHOLE pipeline
    # end-to-end (encoder+decoder+scorer+mask+back) with masking active, so the
    # decoder learns to handle masked/noisy input -- exactly as SpikeAdapt-SC
    # trains its own codec. Only the ResNet front-end stays frozen (all methods).
    model.encoder.load_state_dict(enc)
    model.decoder.load_state_dict(dec)
    back.load_state_dict(obck['back'])

    opt = optim.AdamW(list(model.parameters()) + list(back.parameters()),
                      lr=1e-4, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60, eta_min=1e-6)
    crit = nn.CrossEntropyLoss()

    def sample_noise():
        # SpikeAdapt-SC's curriculum: 50/50 mix of U(0,0.15) and U(0.15,0.40)
        return (random.uniform(0.0, 0.15) if random.random() < 0.5
                else random.uniform(0.15, 0.40))

    def train_epoch():
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            ber = sample_noise()
            Fp, info = model(feat, ber=ber)
            loss_ce = crit(back(Fp), labels)
            loss_rate = (info['tx_rate'] - 0.75) ** 2
            with torch.no_grad():
                z = model.encoder(feat)
            loss_div = model.scorer.compute_diversity_loss([z.detach()])
            loss = loss_ce + 0.5 * loss_rate + 0.05 * loss_div
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

    val_fwd = lambda imgs: back(model(front(imgs), ber=0.0)[0])
    snap = f'./snapshots_{ds}valp_adaptive1bit_e2e_seed{seed}/'
    cks = select_loop(snap, 'adaptive1bit_e2e_best_', 60, 10, train_epoch,
                      val_fwd, model, back, val_loader)
    ck = torch.load(os.path.join(snap, cks[-1]), map_location=device,
                    weights_only=False)
    model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
    model.eval(); back.eval()

    def grid_on(loader):
        g = {}
        for rho in RHOS:
            g[str(rho)] = {}
            for ber in BERS:
                g[str(rho)][str(ber)] = round(acc_on(
                    loader, lambda im: back(model(front(im), ber=ber,
                                                  target_rate_override=rho)[0])), 2)
        return g

    res = {'selected': cks[-1]}
    res['test'] = grid_on(test_loader)
    res['val'] = grid_on(val_loader)
    print(f"  TEST {ds} s{seed} adaptive1bit rho0.75: "
          f"clean {res['test']['0.75']['0.0']} / ber0.3 "
          f"{res['test']['0.75']['0.3']}", flush=True)
    del model, front, back; torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+',
                    default=[42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144])
    ap.add_argument('--datasets', nargs='+', default=['aid', 'resisc45'])
    args = ap.parse_args()
    out = os.path.join(ROOT, 'eval/seed_results/valproto_adaptive1bit.json')
    res = json.load(open(out)) if os.path.exists(out) else {}
    for ds in args.datasets:
        node = res.setdefault(ds, {})
        for seed in args.seeds:
            if str(seed) in node:
                print(f'  {ds} seed {seed}: already done, skip', flush=True)
                continue
            print(f'== adaptive1bit {ds} seed {seed}', flush=True)
            r = run_adaptive1bit(ds, seed)
            if r is not None:
                node[str(seed)] = r
                json.dump(res, open(out, 'w'), indent=1)
    print(f'adaptive1bit: saved {out}', flush=True)


if __name__ == '__main__':
    main()
