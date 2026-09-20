#!/usr/bin/env python3
"""Rebuild the budget-matched G8-ext grid from saved checkpoints (eval-only).

The ext training saved snapshots_{ds}valp_adaptive1bitG8ext_seed{seed}/ but a
quoting bug killed the process before the grid was written. This re-runs only
the (rho, T', ber) grid eval on val+test and saves
eval/seed_results/adaptive1bit_joint_grid_{ds}_ext.json (same shape as the rest).

Usage: python eval/rebuild_g8ext_grid.py [--datasets aid] [--seeds 42 123 456]
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'train'))
sys.path.insert(0, os.path.join(ROOT, 'models'))

from train_baselines_valproto import make_loaders, load_front_back, device  # noqa: E402
from train_adaptive1bit_grouped_joint import (Adaptive1bitGrouped, RHOS,     # noqa: E402
                                              TPRIMES, BERS)


def grid_on(model, back, front, loader):
    g = {}
    with torch.no_grad():
        for rho in RHOS:
            g[str(rho)] = {}
            for tp in TPRIMES:
                g[str(rho)][str(tp)] = {}
                for b in BERS:
                    c = t = 0
                    for imgs, labels in loader:
                        imgs, labels = imgs.to(device), labels.to(device)
                        Fp, _ = model(front(imgs), ber=b, tprime=tp,
                                      target_rate_override=rho)
                        c += back(Fp).argmax(1).eq(labels).sum().item()
                        t += labels.size(0)
                    g[str(rho)][str(tp)][str(b)] = round(100. * c / t, 2)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['aid'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    args = ap.parse_args()
    for ds in args.datasets:
        out_p = os.path.join(
            ROOT, f'eval/seed_results/adaptive1bit_joint_grid_{ds}_ext.json')
        res = json.load(open(out_p)) if os.path.exists(out_p) else {}
        for seed in args.seeds:
            if str(seed) in res:
                print(f'{ds} s{seed}: done, skip', flush=True); continue
            snap = f'./snapshots_{ds}valp_adaptive1bitG8ext_seed{seed}/'
            if not os.path.isdir(snap):
                print(f'{ds} s{seed}: no ext snapshot, skip', flush=True)
                continue
            cks = sorted([f for f in os.listdir(snap)
                          if f.startswith('ext_best_')],
                         key=lambda x: float(x.split('_')[-1][:-4]))
            if not cks:
                print(f'{ds} s{seed}: no ext ckpt, skip', flush=True); continue
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            train_loader, val_loader, test_loader, ncls = make_loaders(ds, seed)
            front, back = load_front_back(ds, ncls, seed)
            model = Adaptive1bitGrouped(C_in=1024, C1=256,
                                        target_rate=0.75).to(device)
            ck = torch.load(os.path.join(snap, cks[-1]), map_location=device,
                            weights_only=False)
            model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
            model.eval(); back.eval()
            out = {'selected': cks[-1],
                   'test': grid_on(model, back, front, test_loader),
                   'val': grid_on(model, back, front, val_loader)}
            key = out['test']['0.75']['8']['0.3']
            print(f'{ds} s{seed} G8ext: r.75/T8 ber.3 = {key}', flush=True)
            res[str(seed)] = out
            json.dump(res, open(out_p, 'w'), indent=1)
            del model, front, back; torch.cuda.empty_cache()
        print(f'saved {out_p}', flush=True)


if __name__ == '__main__':
    main()
