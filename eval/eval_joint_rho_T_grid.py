#!/usr/bin/env python3
"""(rho, T', BER) accuracy grid for the joint closed-loop policy.

Two passes:
  1. POLICY GRID (seed 42): rho in {0.625, 0.75, 0.875, 1.0} x
     T' in {3,4,5,6} x BER in {0,...,0.30}. T'=8 values come from the
     existing 10-seed sweep (eval/seed_results/per_rho_10seed.json) and are
     not recomputed.
  2. REPLICATION (seeds 123, 456): T' in {2,3,4,5,6} at rho=0.75,
     BER in {0, 0.15, 0.30} -- 3-seed mean+-std for the paper's
     temporal-truncation table (with seed 42 from pass 1 / prior run).

Early-exit semantics identical to eval_temporal_truncation.py: encoder runs
only T' steps, scorer sees the truncated time-average, decoder zero-pads.
Per-seed checkpoints, backbones, and data splits follow multi_seed_pipeline.

Output (incremental, resumable): eval/seed_results/joint_rho_T_grid.json
    {dataset: {seed: {rho: {T': {ber: acc}}}}}

Usage:
    python eval/eval_joint_rho_T_grid.py
"""

import os, sys, json, glob, random, time
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train_aid_v2 import ResNet50Front, ResNet50Back
from run_final_pipeline import AIDDataset5050, RESISC45Dataset, SpikeAdaptSC_v5c_NA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_FULL = 8
OUT = 'eval/seed_results/joint_rho_T_grid.json'

GRID_PASS = dict(
    seeds=[42, 123, 456],
    rhos=[0.625, 0.75, 0.875, 1.0],
    tprimes=[3, 4, 5, 6],
    bers=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
)
REPL_PASS = dict(
    seeds=[123, 456],
    rhos=[0.75],
    tprimes=[2, 3, 4, 5, 6],
    bers=[0.0, 0.15, 0.30],
)


def pick_checkpoint(snap_dir):
    manifest = os.path.join(snap_dir, 'selected_checkpoint.txt')
    if os.path.exists(manifest):
        with open(manifest) as f:
            name = f.read().strip()
        p = os.path.join(snap_dir, name)
        if os.path.exists(p):
            return p
    cands = glob.glob(os.path.join(snap_dir, 'v5cna_best_*.pth'))
    return max(cands, key=lambda p: float(p.rsplit('_', 1)[1][:-4])) \
        if cands else None


@torch.no_grad()
def evaluate_truncated(model, back, front, loader, ber, rho, t_prime):
    model.eval(); back.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        feat = front(imgs)
        all_S2, m1, m2 = [], None, None
        for t in range(t_prime):
            _, s2, m1, m2 = model.encoder(feat, m1, m2, t=t)
            all_S2.append(s2)
        importance = model.scorer(all_S2, ber).squeeze(1)
        old = model.block_mask.target_rate
        model.block_mask.target_rate = rho
        mask, _ = model.block_mask(importance, training=False)
        model.block_mask.target_rate = old
        recv = [model.channel(all_S2[t] * mask, ber) for t in range(t_prime)]
        Fp = model.decoder(recv, mask)
        correct += back(Fp).argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return 100. * correct / total


def load_results():
    if os.path.exists(OUT):
        with open(OUT) as f:
            return json.load(f)
    return {}


def save_results(res):
    tmp = OUT + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(res, f, indent=1)
    os.replace(tmp, OUT)


def main():
    print(f'Device: {device}', flush=True)
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                         T.Normalize((.485, .456, .406), (.229, .224, .225))])
    results = load_results()

    for pass_name, cfg in [('grid', GRID_PASS), ('replication', REPL_PASS)]:
        print(f'\n##### {pass_name.upper()} pass #####', flush=True)
        for ds_name, ds_key, n_classes in [('AID', 'aid', 30),
                                           ('RESISC45', 'resisc45', 45)]:
            for seed in cfg['seeds']:
                seed_str = str(seed)
                done = results.get(ds_key, {}).get(seed_str, {})
                todo = [(r, tp, b)
                        for r in cfg['rhos'] for tp in cfg['tprimes']
                        for b in cfg['bers']
                        if str(b) not in done.get(str(r), {}).get(str(tp), {})]
                if not todo:
                    print(f'[{ds_name} seed {seed}] done, skip', flush=True)
                    continue

                bb_path = f'./snapshots_{ds_key}_5050_seed{seed}/backbone_best.pth'
                ck_path = pick_checkpoint(f'./snapshots_{ds_key}_v5cna_seed{seed}')
                if ck_path is None or not os.path.exists(bb_path):
                    print(f'[{ds_name} seed {seed}] MISSING, skip', flush=True)
                    continue

                torch.manual_seed(1234 + seed)
                np.random.seed(1234 + seed)
                random.seed(1234 + seed)
                if ds_key == 'aid':
                    test_ds = AIDDataset5050('./data', tf_test, 'test', seed=seed)
                else:
                    test_ds = RESISC45Dataset('./data', tf_test, 'test',
                                              train_ratio=0.20, seed=seed)
                loader = DataLoader(test_ds, 64, False, num_workers=6,
                                    pin_memory=True, persistent_workers=True)

                front = ResNet50Front(grid_size=14).to(device)
                bb = torch.load(bb_path, map_location=device, weights_only=False)
                front.load_state_dict(
                    {k: v for k, v in bb.items()
                     if not k.startswith(('layer4.', 'fc.', 'avgpool.',
                                          'spatial_pool.'))}, strict=False)
                front.eval()
                for p in front.parameters():
                    p.requires_grad = False
                back = ResNet50Back(n_classes).to(device)
                model = SpikeAdaptSC_v5c_NA(C_in=1024, C1=256, C2=36, T=T_FULL,
                                            target_rate=0.75,
                                            grid_size=14).to(device)
                ck = torch.load(ck_path, map_location=device, weights_only=False)
                model.load_state_dict(ck['model'])
                back.load_state_dict(ck['back'])
                print(f'[{ds_name} seed {seed}] {os.path.basename(ck_path)}, '
                      f'{len(todo)} combos', flush=True)

                for rho, tp, ber in todo:
                    t0 = time.time()
                    acc = evaluate_truncated(model, back, front, loader,
                                             ber, rho, tp)
                    results.setdefault(ds_key, {}).setdefault(seed_str, {}) \
                           .setdefault(str(rho), {}).setdefault(str(tp), {})[
                               str(ber)] = round(acc, 4)
                    save_results(results)
                    print(f'  rho={rho:g} T\'={tp} ber={ber:.2f}: {acc:.2f}%'
                          f'  ({time.time()-t0:.0f}s)', flush=True)

                del model, back, front, loader, test_ds
                torch.cuda.empty_cache()

    print('\nAll done.', flush=True)


if __name__ == '__main__':
    main()
