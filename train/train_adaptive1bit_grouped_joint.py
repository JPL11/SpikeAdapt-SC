#!/usr/bin/env python3
"""Adaptive-1bit-Grouped: the MAXIMALLY strong learned-discrete JOINT baseline.

Gives the fair binary competitor BOTH rate axes SpikeAdapt-SC has:
  * learned BER-conditioned spatial masking (rho), same scorer/mask as the method
  * grouped temporal truncation (T'): C2 = 8 groups x 36 = 288 binary maps, keep
    the first T' groups (payload-matched: rho*T'*36*14*14, == 42,336 at .75/8).

Crucially, unlike the cnn1bit_grouped_T2 control (which trains at full depth and
truncates only at inference), this baseline is TRAINED WITH random-T' truncation
augmentation, so the CNN gets its best possible shot at the temporal axis. If
SpikeAdapt's joint loop still dominates it, the temporal-axis-for-free thesis is
proven against the strongest fair ANN; if it matches, we learn that too.

Only the ResNet front-end is frozen (all methods). Val protocol: 10% per-seed
val carve, checkpoint selected on val, TEST scored once. Grid output matches the
SpikeAdapt joint-grid contract (valproto_missions.load_grids) so the SAME Table
III policy/mission machinery scores it.

Usage: /home/jpli/miniconda/envs/semcom/bin/python \
         train/train_adaptive1bit_grouped_joint.py --seeds 42 123 456
Output: eval/seed_results/adaptive1bit_joint_grid_{aid,resisc45}.json
        keyed [seed][split][rho][tprime][ber]  (== SpikeAdapt grid shape)
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

from train_baselines_valproto import (make_loaders, load_front_back,
                                       select_loop, device, ROOT)
from train_1bit_baseline import BinaryCNN_Encoder, BinaryCNN_Decoder
from noise_aware_scorer import NoiseAwareScorer
from train_jscc_adaptive import LearnedBlockMask

N_GROUPS = 8
C2_GROUP = 36
C2_TOTAL = N_GROUPS * C2_GROUP           # 288
RHOS = [0.625, 0.75, 0.875, 1.0]
TPRIMES = [3, 4, 5, 6, 8]
BERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


class Adaptive1bitGrouped(nn.Module):
    """Binary CNN codec with learned spatial mask + grouped temporal truncation."""

    def __init__(self, C_in=1024, C1=256, target_rate=0.75):
        super().__init__()
        self.encoder = BinaryCNN_Encoder(C_in, C1, C2_TOTAL)
        self.decoder = BinaryCNN_Decoder(C_in, C1, C2_TOTAL)
        self.scorer = NoiseAwareScorer(C_spike=C2_GROUP, hidden=32)
        self.block_mask = LearnedBlockMask(target_rate, 0.5)

    def forward(self, feat, ber=0.0, tprime=N_GROUPS, target_rate_override=None):
        z = self.encoder(feat)                       # (B, 288, 14, 14) binary
        B = z.size(0)
        z8 = z.view(B, N_GROUPS, C2_GROUP, 14, 14)   # (B, 8, 36, 14, 14)
        groups = [z8[:, g] for g in range(N_GROUPS)]  # temporal analog of spikes
        importance = self.scorer(groups, ber)        # (B, 1, 14, 14)
        if target_rate_override is not None:
            old = self.block_mask.target_rate
            self.block_mask.target_rate = target_rate_override
            mask, tx = self.block_mask(importance, training=False)
            self.block_mask.target_rate = old
        else:
            mask, tx = self.block_mask(importance, training=self.training)
        m = mask.unsqueeze(1)                        # (B, 1, 1, 14, 14) broadcast
        # transmit only kept spatial blocks x first T' groups
        send = (z8 * m).clone()
        send[:, tprime:] = 0.0
        flip = (torch.rand_like(send) < ber).float()
        recv = ((send + flip) % 2) * m               # BSC on kept, re-mask spatial
        recv[:, tprime:] = 0.0                        # re-zero truncated groups
        Fp = self.decoder(recv.reshape(B, C2_TOTAL, 14, 14))
        return Fp, {'tx_rate': tx.item(), 'kept': mask.mean().item()}


def run_grouped(ds, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_loader, val_loader, test_loader, ncls = make_loaders(ds, seed)
    front, back = load_front_back(ds, ncls, seed)
    if front is None:
        print(f'  {ds} seed {seed}: valp backbone missing, skip'); return None

    model = Adaptive1bitGrouped(C_in=1024, C1=256, target_rate=0.75).to(device)
    opt = optim.AdamW(list(model.parameters()) + list(back.parameters()),
                      lr=1e-4, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60, eta_min=1e-6)
    crit = nn.CrossEntropyLoss()

    def sample_noise():
        return (random.uniform(0.0, 0.15) if random.random() < 0.5
                else random.uniform(0.15, 0.40))

    def train_epoch():
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            ber = sample_noise()
            tp = random.choice(TPRIMES)              # truncation augmentation
            Fp, info = model(feat, ber=ber, tprime=tp)
            loss_ce = crit(back(Fp), labels)
            loss_rate = (info['tx_rate'] - 0.75) ** 2
            with torch.no_grad():
                z = model.encoder(feat)
                B = z.size(0)
                g0 = z.view(B, N_GROUPS, C2_GROUP, 14, 14)
                grp = [g0[:, g].detach() for g in range(N_GROUPS)]
            loss_div = model.scorer.compute_diversity_loss(grp)
            loss = loss_ce + 0.5 * loss_rate + 0.05 * loss_div
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

    # val-select at the nominal operating point (rho=0.75, full depth, clean)
    val_fwd = lambda imgs: back(model(front(imgs), ber=0.0, tprime=N_GROUPS,
                                      target_rate_override=0.75)[0])
    snap = f'./snapshots_{ds}valp_adaptive1bitG8_seed{seed}/'
    cks = select_loop(snap, 'adaptive1bitG8_best_', 60, 10, train_epoch,
                      val_fwd, model, back, val_loader)
    ck = torch.load(os.path.join(snap, cks[-1]), map_location=device,
                    weights_only=False)
    model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
    model.eval(); back.eval()

    def acc_on(loader, rho, tp, ber):
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                Fp, _ = model(front(imgs), ber=ber, tprime=tp,
                              target_rate_override=rho)
                correct += back(Fp).argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        return round(100. * correct / total, 2)

    def grid_on(loader):
        g = {}
        for rho in RHOS:
            g[str(rho)] = {}
            for tp in TPRIMES:
                g[str(rho)][str(tp)] = {str(b): acc_on(loader, rho, tp, b)
                                        for b in BERS}
        return g

    out = {'test': grid_on(test_loader), 'val': grid_on(val_loader),
           'selected': cks[-1]}
    print(f"  {ds} s{seed} G8: test rho.75/T8 clean "
          f"{out['test']['0.75']['8']['0.0']} / ber.3 "
          f"{out['test']['0.75']['8']['0.3']}; rho.75/T4 ber.3 "
          f"{out['test']['0.75']['4']['0.3']}", flush=True)
    del model, front, back; torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    ap.add_argument('--datasets', nargs='+', default=['aid', 'resisc45'])
    args = ap.parse_args()
    for ds in args.datasets:
        out = os.path.join(ROOT,
                           f'eval/seed_results/adaptive1bit_joint_grid_{ds}.json')
        res = json.load(open(out)) if os.path.exists(out) else {}
        for seed in args.seeds:
            if str(seed) in res:
                print(f'  {ds} seed {seed}: done, skip', flush=True); continue
            print(f'== adaptive1bitG8 {ds} seed {seed}', flush=True)
            r = run_grouped(ds, seed)
            if r is not None:
                res[str(seed)] = r
                json.dump(res, open(out, 'w'), indent=1)
        print(f'saved {out}', flush=True)


if __name__ == '__main__':
    main()
