#!/usr/bin/env python3
"""SNN-SC (prior spiking baseline) upgraded to the 10-seed validation protocol.

Replaces the legacy seed-42 single-run SNN-SC row in Table II. SNN-SC is the
SpikeAdapt-SC codec ablated to the prior-art configuration: EncoderV5/DecoderV5
with use_mpbn=False, NO importance scorer, NO spatial masking (fixed rho=1.0,
56,448 bits), T=8. Trained with the SAME data splits, curriculum, and val
protocol as every other baseline so the "adaptation + MPBN vs. fixed-rate SNN"
comparison is apples-to-apples and multi-seed.

Only the ResNet front-end is frozen. 60 epochs, Adam 1e-4 cosine, sample_noise
(50/50 U(0,.15)/U(.15,.40)) curriculum. 10% per-seed val carve; checkpoint
selected on val; TEST scored once over the BER grid.

Usage: /home/jpli/miniconda/envs/semcom/bin/python train/train_snnsc_valproto.py
Output: eval/seed_results/valproto_snnsc.json
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

from train_baselines_valproto import (make_loaders, load_front_back, acc_on,
                                       select_loop, BERS, device, ROOT)
from train_aid_v2 import BSC_Channel
from train_aid_v5 import EncoderV5, DecoderV5


def sample_noise():
    """SpikeAdapt curriculum: 50/50 mix of U(0,0.15) and U(0.15,0.40)."""
    return (random.uniform(0.0, 0.15) if random.random() < 0.5
            else random.uniform(0.15, 0.40))

T_STEPS = 8


class SNNSC(nn.Module):
    """Fixed-rate spiking codec: no scorer, no masking, no MPBN (rho=1.0)."""

    def __init__(self, C_in=1024, C1=256, C2=36, T=T_STEPS):
        super().__init__()
        self.T = T
        self.encoder = EncoderV5(C_in, C1, C2, T, use_mpbn=False)
        self.decoder = DecoderV5(C_in, C1, C2, T, use_mpbn=False)
        self.channel = BSC_Channel()

    def forward(self, feat, ber=0.0):
        all_S2, m1, m2 = [], None, None
        for t in range(self.T):
            _, s2, m1, m2 = self.encoder(feat, m1, m2, t=t)
            all_S2.append(s2)
        mask = torch.ones(feat.size(0), 1, all_S2[0].size(-2),
                          all_S2[0].size(-1), device=feat.device)  # rho=1.0
        recv = [self.channel(all_S2[t] * mask, ber) for t in range(self.T)]
        Fp = self.decoder(recv, mask)
        return Fp, {}


def run_snnsc(ds, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_loader, val_loader, test_loader, ncls = make_loaders(ds, seed)
    front, back = load_front_back(ds, ncls, seed)
    if front is None:
        print(f'  {ds} seed {seed}: valp backbone missing, skip'); return None

    model = SNNSC(C_in=1024, C1=256, C2=36, T=T_STEPS).to(device)
    opt = optim.Adam(list(model.parameters()) + list(back.parameters()),
                     lr=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60, eta_min=1e-6)
    crit = nn.CrossEntropyLoss()

    def train_epoch():
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            ber = sample_noise()
            Fp, _ = model(feat, ber=ber)
            loss = crit(back(Fp), labels)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

    val_fwd = lambda imgs: back(model(front(imgs), ber=0.0)[0])
    snap = f'./snapshots_{ds}valp_snnsc_seed{seed}/'
    cks = select_loop(snap, 'snnsc_best_', 60, 10, train_epoch, val_fwd,
                      model, back, val_loader)
    ck = torch.load(os.path.join(snap, cks[-1]), map_location=device,
                    weights_only=False)
    model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
    model.eval(); back.eval()

    res = {'selected': cks[-1]}
    for ber in BERS:
        res[str(ber)] = round(acc_on(
            test_loader, lambda im: back(model(front(im), ber=ber)[0])), 2)
    print(f"  TEST {ds} s{seed} snnsc clean {res['0.0']} / ber0.3 "
          f"{res['0.3']}", flush=True)
    del model, front, back; torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+',
                    default=[42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144])
    ap.add_argument('--datasets', nargs='+', default=['aid', 'resisc45'])
    args = ap.parse_args()
    out = os.path.join(ROOT, 'eval/seed_results/valproto_snnsc.json')
    res = json.load(open(out)) if os.path.exists(out) else {}
    for ds in args.datasets:
        node = res.setdefault(ds, {})
        for seed in args.seeds:
            if str(seed) in node:
                print(f'  {ds} seed {seed}: done, skip', flush=True); continue
            print(f'== snnsc {ds} seed {seed}', flush=True)
            r = run_snnsc(ds, seed)
            if r is not None:
                node[str(seed)] = r
                json.dump(res, open(out, 'w'), indent=1)
    print(f'snnsc: saved {out}', flush=True)


if __name__ == '__main__':
    main()
