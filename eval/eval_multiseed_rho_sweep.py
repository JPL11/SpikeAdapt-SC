#!/usr/bin/env python3
"""10-seed rho x BER sweep for SpikeAdapt-SC (V5C-NA) on AID and RESISC45.

Fills the gaps behind the alpha-vs-rho figure: the existing per-seed JSONs
(eval/seed_results/{ds}_seed{S}.json) only cover rho in {0.625, 0.75, 1.0};
the full rho grid existed only for seed 42 (per_rho_scorer_results.json).
This script evaluates every seed's own checkpoint (matched data split seed,
matched backbone, selected_checkpoint.txt convention from
train/multi_seed_pipeline.py) across the full rho grid.

Priority pass: BER in {0.0, 0.15, 0.30} (paper operating points).
Extended pass: BER in {0.05, 0.10, 0.20, 0.25} (smoother alpha curves).

Output (incremental, resumable): eval/seed_results/per_rho_10seed.json
    {dataset: {seed: {rho: {ber: acc}}}}

Usage:
    python eval/eval_multiseed_rho_sweep.py
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
T_STEPS = 8
SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]
RHOS = [0.10, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 1.0]
BERS_PRIORITY = [0.0, 0.15, 0.30]
BERS_EXTENDED = [0.05, 0.10, 0.20, 0.25]
OUT = 'eval/seed_results/per_rho_10seed.json'


def pick_checkpoint(snap_dir):
    """Replicate multi_seed_pipeline.py selection: manifest, else highest acc."""
    manifest = os.path.join(snap_dir, 'selected_checkpoint.txt')
    if os.path.exists(manifest):
        with open(manifest) as f:
            name = f.read().strip()
        path = os.path.join(snap_dir, name)
        if os.path.exists(path):
            return path
    cands = glob.glob(os.path.join(snap_dir, 'v5cna_best_*.pth'))
    if not cands:
        return None
    return max(cands, key=lambda p: float(p.rsplit('_', 1)[1][:-4]))


@torch.no_grad()
def evaluate(model, back, front, loader, ber, rho):
    model.eval(); back.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        feat = front(imgs)
        Fp, _ = model(feat, noise_param=ber, target_rate_override=rho)
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


def seed_existing(res):
    """Import already-computed values from the original per-seed JSONs
    (same harness: multi_seed_pipeline.py) so we don't recompute them."""
    for ds in ['aid', 'resisc45']:
        res.setdefault(ds, {})
        for s in SEEDS:
            p = f'eval/seed_results/{ds}_seed{s}.json'
            if not os.path.exists(p):
                continue
            with open(p) as f:
                d = json.load(f)
            res[ds].setdefault(str(s), {})
            for rho_str, bers in d.items():
                res[ds][str(s)].setdefault(rho_str, {})
                for ber_str, acc in bers.items():
                    res[ds][str(s)][rho_str].setdefault(ber_str, acc)
    return res


def main():
    print(f'Device: {device}', flush=True)
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                         T.Normalize((.485, .456, .406), (.229, .224, .225))])

    results = seed_existing(load_results())
    save_results(results)

    for ber_pass, bers in [('priority', BERS_PRIORITY), ('extended', BERS_EXTENDED)]:
        print(f'\n##### {ber_pass.upper()} pass: BERs={bers} #####', flush=True)
        for ds_name, ds_key, n_classes in [('AID', 'aid', 30),
                                           ('RESISC45', 'resisc45', 45)]:
            for seed in SEEDS:
                seed_str = str(seed)
                done = results.get(ds_key, {}).get(seed_str, {})
                todo = [(r, b) for r in RHOS for b in bers
                        if str(b) not in done.get(str(r), {})]
                if not todo:
                    print(f'[{ds_name} seed {seed}] all done, skip', flush=True)
                    continue

                bb_path = f'./snapshots_{ds_key}_5050_seed{seed}/backbone_best.pth'
                snap_dir = f'./snapshots_{ds_key}_v5cna_seed{seed}'
                ck_path = pick_checkpoint(snap_dir)
                if ck_path is None or not os.path.exists(bb_path):
                    print(f'[{ds_name} seed {seed}] MISSING ckpt, skip', flush=True)
                    continue

                torch.manual_seed(1234 + seed)
                np.random.seed(1234 + seed)
                random.seed(1234 + seed)

                # Per-seed data split (matched to training, see multi_seed_pipeline)
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
                model = SpikeAdaptSC_v5c_NA(C_in=1024, C1=256, C2=36, T=T_STEPS,
                                            target_rate=0.75, grid_size=14).to(device)
                ck = torch.load(ck_path, map_location=device, weights_only=False)
                model.load_state_dict(ck['model'])
                back.load_state_dict(ck['back'])
                print(f'[{ds_name} seed {seed}] {os.path.basename(ck_path)}, '
                      f'{len(todo)} combos', flush=True)

                for rho, ber in todo:
                    t0 = time.time()
                    acc = evaluate(model, back, front, loader, ber, rho)
                    results.setdefault(ds_key, {}).setdefault(seed_str, {}) \
                           .setdefault(str(rho), {})[str(ber)] = round(acc, 4)
                    save_results(results)
                    print(f'  rho={rho:.3f} ber={ber:.2f}: {acc:.2f}%  '
                          f'({time.time()-t0:.0f}s)', flush=True)

                del model, back, front, loader, test_ds
                torch.cuda.empty_cache()

    print('\nAll done.', flush=True)


if __name__ == '__main__':
    main()
