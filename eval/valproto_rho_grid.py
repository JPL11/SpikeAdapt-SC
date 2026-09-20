#!/usr/bin/env python3
"""Path A: rho x BER accuracy grids for the val-protocol checkpoints.

For each finished valp seed, evaluates the val-selected model at
rho in RHOS x BER in BERS on BOTH splits:
  - 'val'  grid -> policy construction (selection surface)
  - 'test' grid -> reported accuracy (scored once per pre-declared cell)

Incremental: one JSON per dataset, keyed [seed][split][rho][ber]; finished
cells are skipped, so the script can run repeatedly as seeds finish.

Usage: python eval/valproto_rho_grid.py [--seeds 42 123 ...] [--quick]
Output: eval/seed_results/valproto_rho_grid_{ds}.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'train'))
from run_final_pipeline import (AIDDataset5050, RESISC45Dataset,   # noqa: E402
                                ResNet50Front, ResNet50Back,
                                SpikeAdaptSC_v5c_NA, T_STEPS, device)
from run_final_pipeline_valproto import (TF_TEST, VAL_FRAC)         # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RHOS = [0.5, 0.625, 0.75, 0.875, 1.0]
BERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
ALL_SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]


def loaders(ds, seed):
    if ds == 'aid':
        tr = AIDDataset5050('./data', TF_TEST, 'train', seed=seed)
        te = AIDDataset5050('./data', TF_TEST, 'test', seed=seed)
        ncls = 30
    else:
        tr = RESISC45Dataset('./data', TF_TEST, 'train', train_ratio=0.20, seed=seed)
        te = RESISC45Dataset('./data', TF_TEST, 'test', train_ratio=0.20, seed=seed)
        ncls = 45
    idx = np.random.default_rng(seed).permutation(len(tr))
    n_val = int(round(VAL_FRAC * len(tr)))
    val = Subset(tr, idx[-n_val:].tolist())     # same carve as training
    mk = lambda d: DataLoader(d, 64, False, num_workers=4, pin_memory=True)
    return {'val': mk(val), 'test': mk(te)}, ncls


def load_model(ds, ncls, seed):
    name = f'{ds}valp'
    bb = torch.load(f'./snapshots_{name}_5050_seed{seed}/backbone_best.pth',
                    map_location=device)
    front = ResNet50Front(grid_size=14).to(device)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.',
                                                'spatial_pool.'))}, strict=False)
    front.eval()
    snap = f'./snapshots_{name}_v5cna_seed{seed}/'
    cks = [f for f in os.listdir(snap) if f.startswith('v5cna_best_')]
    best = max(cks, key=lambda f: float(f.split('_')[-1][:-4]))
    ck = torch.load(os.path.join(snap, best), map_location=device,
                    weights_only=False)
    back = ResNet50Back(ncls).to(device)
    back.load_state_dict(ck['back'])
    model = SpikeAdaptSC_v5c_NA(C_in=1024, C1=256, C2=36, T=T_STEPS,
                                target_rate=0.75, grid_size=14).to(device)
    model.load_state_dict(ck['model'])
    model.eval(); back.eval()
    return front, model, back


def acc_at(front, model, back, loader, ber, rho):
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            Fp, _ = model(front(imgs), noise_param=ber,
                          target_rate_override=rho)
            correct += back(Fp).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return round(100. * correct / total, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=ALL_SEEDS)
    ap.add_argument('--quick', action='store_true',
                    help='2 rhos x 2 bers smoke')
    args = ap.parse_args()
    rhos = [0.75, 1.0] if args.quick else RHOS
    bers = [0.0, 0.30] if args.quick else BERS
    for ds in ['aid', 'resisc45']:
        out = os.path.join(ROOT, f'eval/seed_results/valproto_rho_grid_{ds}.json')
        res = json.load(open(out)) if os.path.exists(out) else {}
        for seed in args.seeds:
            if not os.path.isdir(f'./snapshots_{ds}valp_v5cna_seed{seed}'):
                print(f'{ds} seed {seed}: checkpoints not ready, skipping',
                      flush=True)
                continue
            lds, ncls = loaders(ds, seed)
            fmb = None
            node = res.setdefault(str(seed), {})
            for split in ['val', 'test']:
                sp = node.setdefault(split, {})
                for rho in rhos:
                    rp = sp.setdefault(str(rho), {})
                    for ber in bers:
                        if str(ber) in rp:
                            continue
                        if fmb is None:
                            fmb = load_model(ds, ncls, seed)
                        rp[str(ber)] = acc_at(*fmb, lds[split], ber, rho)
                        json.dump(res, open(out, 'w'))
                        print(f'{ds} s{seed} {split} rho{rho} ber{ber}: '
                              f'{rp[str(ber)]:.2f}', flush=True)
        print(f'{ds}: saved {out}', flush=True)


if __name__ == '__main__':
    main()
