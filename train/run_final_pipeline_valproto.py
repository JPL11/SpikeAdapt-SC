#!/usr/bin/env python3
"""Validation-protocol retraining (external-review fix: test-set selection).

Identical three-stage pipeline to run_final_pipeline.py, with one change:
checkpoint selection (backbone best, v5cna best) is driven by a held-out
VALIDATION split (10% of train, deterministic per seed, test transforms),
and the TEST split is evaluated exactly once, at the end, over the BER grid.

Snapshots go to snapshots_{ds}valp_* dirs; historical runs untouched.
Usage: python train/run_final_pipeline_valproto.py --dataset aid --seed 42
Output: eval/seed_results/valproto_{ds}_seed{seed}.json
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_final_pipeline import (AIDDataset5050, RESISC45Dataset,   # noqa: E402
                                ResNet50Front, ResNet50Back,
                                SpikeAdaptSC_v5c_NA, T_STEPS,
                                train_backbone, train_v5c_na, device)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
VAL_FRAC = 0.10

TF_TRAIN = T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
                      T.ColorJitter(0.2, 0.2, 0.2),
                      T.ToTensor(), T.Normalize((.485, .456, .406), (.229, .224, .225))])
TF_TEST = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                     T.Normalize((.485, .456, .406), (.229, .224, .225))])


def make_loaders(ds, seed):
    if ds == 'aid':
        tr_aug = AIDDataset5050('./data', TF_TRAIN, 'train', seed=seed)
        tr_plain = AIDDataset5050('./data', TF_TEST, 'train', seed=seed)
        test_ds = AIDDataset5050('./data', TF_TEST, 'test', seed=seed)
        ncls = 30
    else:
        tr_aug = RESISC45Dataset('./data', TF_TRAIN, 'train', train_ratio=0.20, seed=seed)
        tr_plain = RESISC45Dataset('./data', TF_TEST, 'train', train_ratio=0.20, seed=seed)
        test_ds = RESISC45Dataset('./data', TF_TEST, 'test', train_ratio=0.20, seed=seed)
        ncls = 45
    n = len(tr_aug)
    idx = np.random.default_rng(seed).permutation(n)
    n_val = int(round(VAL_FRAC * n))
    train_sub = Subset(tr_aug, idx[:-n_val].tolist())
    val_sub = Subset(tr_plain, idx[-n_val:].tolist())
    print(f'{ds}: train {len(train_sub)}, val {len(val_sub)}, test {len(test_ds)}')
    mk = lambda d, sh: DataLoader(d, 32, sh, num_workers=4, pin_memory=True)
    return mk(train_sub, True), mk(val_sub, False), mk(test_ds, False), ncls


def final_test_eval(ds, ncls, seed, test_loader):
    """One-shot test evaluation of the val-selected checkpoints over the BER grid."""
    name = f'{ds}valp'
    bb_path = f'./snapshots_{name}_5050_seed{seed}/backbone_best.pth'
    snap = f'./snapshots_{name}_v5cna_seed{seed}/'
    cks = [f for f in os.listdir(snap) if f.startswith('v5cna_best_')]
    best = max(cks, key=lambda f: float(f.split('_')[-1][:-4]))
    ck = torch.load(os.path.join(snap, best), map_location=device, weights_only=False)

    front = ResNet50Front(grid_size=14).to(device)
    bb = torch.load(bb_path, map_location=device)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))},
                          strict=False)
    front.eval()
    back = ResNet50Back(ncls).to(device)
    back.load_state_dict(ck['back'])
    model = SpikeAdaptSC_v5c_NA(C_in=1024, C1=256, C2=36, T=T_STEPS,
                                target_rate=0.75, grid_size=14).to(device)
    model.load_state_dict(ck['model'])
    model.eval(); back.eval()

    res = {'selected_ckpt': best}
    with torch.no_grad():
        for ber in BERS:
            correct, total = 0, 0
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                Fp, _ = model(front(imgs), noise_param=ber)
                correct += back(Fp).argmax(1).eq(labels).sum().item()
                total += labels.size(0)
            res[f'ber{ber}'] = round(100. * correct / total, 2)
            print(f'  TEST {ds} seed{seed} BER {ber:.2f}: {res[f"ber{ber}"]:.2f}%', flush=True)
    out = os.path.join(ROOT, f'eval/seed_results/valproto_{ds}_seed{seed}.json')
    json.dump(res, open(out, 'w'), indent=1)
    print('saved:', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['aid', 'resisc45', 'both'], default='both')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs_bb', type=int, default=50)
    ap.add_argument('--epochs_s2', type=int, default=60)
    ap.add_argument('--epochs_s3', type=int, default=40)
    ap.add_argument('--eval-only', action='store_true')
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    for ds in (['aid', 'resisc45'] if args.dataset == 'both' else [args.dataset]):
        train_loader, val_loader, test_loader, ncls = make_loaders(ds, args.seed)
        name = f'{ds}valp'
        if not args.eval_only:
            # selection loader = VALIDATION (passed where the original passed test)
            train_backbone(name, ncls, train_loader, val_loader, args.seed,
                           args.epochs_bb)
            bb_path = f'./snapshots_{name}_5050_seed{args.seed}/backbone_best.pth'
            train_v5c_na(name, ncls, train_loader, val_loader, args.seed,
                         bb_path, args.epochs_s2, args.epochs_s3)
        final_test_eval(ds, ncls, args.seed, test_loader)


if __name__ == '__main__':
    main()
