#!/usr/bin/env python3
"""Checkpoint ensemble: average top-K checkpoints per seed, re-evaluate.

Purpose: Reduce 10-seed variance at BER=0.30 without any retraining.

Method (Stochastic Weight Averaging-style):
  For each seed, load the top-K best checkpoints from training.
  Average their weights -> single ensembled model.
  Evaluate at BER={0, 0.15, 0.30} for rho={0.625, 0.75, 1.0}.

Expected outcome: variance reduction ~40-50%, slight accuracy gain.

Usage:
    python eval/eval_checkpoint_ensemble.py --dataset aid --k 3
    python eval/eval_checkpoint_ensemble.py --dataset both --k 5
"""

import os, sys, glob, json, argparse, random
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from copy import deepcopy

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from train_aid_v2 import ResNet50Front, ResNet50Back
from run_final_pipeline import AIDDataset5050, RESISC45Dataset, SpikeAdaptSC_v5c_NA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_STEPS = 8
SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]
BER_EVAL = [0.0, 0.15, 0.30]
RHO_EVAL = [0.625, 0.75, 1.0]

DATASET_CONFIGS = {
    'aid': {
        'n_classes': 30, 'ds_cls': AIDDataset5050,
        'ds_kwargs_template': lambda s: dict(seed=s),
        'snap_template': './snapshots_aid_v5cna_seed{}',
        'bb_template': './snapshots_aid_5050_seed{}/backbone_best.pth',
    },
    'resisc45': {
        'n_classes': 45, 'ds_cls': RESISC45Dataset,
        'ds_kwargs_template': lambda s: dict(train_ratio=0.20, seed=s),
        'snap_template': './snapshots_resisc45_v5cna_seed{}',
        'bb_template': './snapshots_resisc45_5050_seed{}/backbone_best.pth',
    },
}


def average_checkpoints(ckpt_paths):
    """Load checkpoints and average their weights component-wise.

    Args:
        ckpt_paths: list of paths
    Returns:
        (avg_model_state, avg_back_state)
    """
    avg_model = None
    avg_back = None
    n = len(ckpt_paths)

    for path in ckpt_paths:
        ck = torch.load(path, map_location=device, weights_only=False)
        if avg_model is None:
            avg_model = {k: v.detach().clone().float() / n for k, v in ck['model'].items()}
            avg_back = {k: v.detach().clone().float() / n for k, v in ck['back'].items()}
        else:
            for k, v in ck['model'].items():
                avg_model[k] += v.detach().float() / n
            for k, v in ck['back'].items():
                avg_back[k] += v.detach().float() / n

    return avg_model, avg_back


def load_ensembled_model(ckpt_paths, n_classes, target_rate=0.75):
    """Build model from checkpoint-averaged weights."""
    avg_model, avg_back = average_checkpoints(ckpt_paths)

    model = SpikeAdaptSC_v5c_NA(C_in=1024, C1=256, C2=36, T=T_STEPS,
                                 target_rate=target_rate, grid_size=14).to(device)
    back = ResNet50Back(n_classes).to(device)

    # Load (keys may have float cast, need to match dtypes)
    model_sd = model.state_dict()
    for k, v in avg_model.items():
        if k in model_sd:
            model_sd[k] = v.to(model_sd[k].dtype)
    model.load_state_dict(model_sd, strict=False)

    back_sd = back.state_dict()
    for k, v in avg_back.items():
        if k in back_sd:
            back_sd[k] = v.to(back_sd[k].dtype)
    back.load_state_dict(back_sd, strict=False)

    model.eval(); back.eval()
    return model, back


def evaluate(model, back, front, loader, ber, rho):
    """Evaluate at (ber, rho)."""
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            Fp, _ = model(front(imgs), noise_param=ber, target_rate_override=rho)
            correct += back(Fp).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total


def process_seed(ds_name, cfg, seed, k):
    """Process one seed: ensemble top-K checkpoints and evaluate."""
    snap_dir = cfg['snap_template'].format(seed)
    bb_path = cfg['bb_template'].format(seed)

    if not os.path.exists(snap_dir):
        print(f"  seed {seed}: no checkpoints dir, skipping")
        return None

    ckpts = sorted(glob.glob(os.path.join(snap_dir, 'v5cna_best_*.pth')),
                   key=lambda p: float(p.split('_')[-1].replace('.pth', '')))
    if len(ckpts) < 2:
        print(f"  seed {seed}: only {len(ckpts)} checkpoints, using single best")
        ckpts_use = ckpts[-1:]
    else:
        # Top-K by accuracy (last K in sorted list)
        ckpts_use = ckpts[-k:]

    accs = [float(p.split('_')[-1].replace('.pth', '')) for p in ckpts_use]
    print(f"  seed {seed}: ensembling {len(ckpts_use)} checkpoints "
          f"[{min(accs):.2f}, {max(accs):.2f}]")

    # Build ensembled model
    model, back = load_ensembled_model(ckpts_use, cfg['n_classes'])

    # Load backbone
    front = ResNet50Front(grid_size=14).to(device)
    if not os.path.exists(bb_path):
        # Fallback to seed 42 backbone
        bb_path = cfg['bb_template'].format(42)
    bb = torch.load(bb_path, map_location=device, weights_only=False)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))},
                          strict=False)
    front.eval()
    for p in front.parameters(): p.requires_grad = False

    # Test loader
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])
    test_ds = cfg['ds_cls']("./data", tf_test, 'test', **cfg['ds_kwargs_template'](seed))
    loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)

    # Evaluate at each (ber, rho)
    results = {}
    for rho in RHO_EVAL:
        results[str(rho)] = {}
        for ber in BER_EVAL:
            acc = evaluate(model, back, front, loader, ber, rho)
            results[str(rho)][str(ber)] = round(acc, 2)
        print(f"    rho={rho}: clean={results[str(rho)]['0.0']:.2f}, "
              f"BER=0.15={results[str(rho)]['0.15']:.2f}, "
              f"BER=0.30={results[str(rho)]['0.3']:.2f}")

    del model, back, front
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='both', choices=['aid', 'resisc45', 'both'])
    parser.add_argument('--k', type=int, default=3,
                        help='Number of top checkpoints to average per seed')
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='Specific seeds to process (default: all 10)')
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else SEEDS
    ds_list = ['aid', 'resisc45'] if args.dataset == 'both' else [args.dataset]

    torch.manual_seed(42); np.random.seed(42); random.seed(42)

    all_results = {}

    for ds_name in ds_list:
        print(f'\n{"#"*60}\n  {ds_name.upper()} — Checkpoint Ensemble (top-{args.k})\n{"#"*60}')
        cfg = DATASET_CONFIGS[ds_name]

        all_results[ds_name] = {}
        for seed in seeds:
            print(f'\n  Processing seed {seed}...')
            res = process_seed(ds_name, cfg, seed, args.k)
            if res is not None:
                all_results[ds_name][str(seed)] = res

    # Summary: compute mean/std over seeds
    print(f'\n{"="*80}')
    print(f'  ENSEMBLE RESULTS SUMMARY (top-{args.k} checkpoints averaged)')
    print(f'{"="*80}')

    # Load baseline 10-seed stats for comparison
    with open('eval/seed_results/summary_10seed.json') as f:
        baseline = json.load(f)

    summary = {}
    for ds_name in ds_list:
        summary[ds_name] = {}
        print(f'\n  {ds_name.upper()}:')
        print(f'  {"rho":<8} {"BER":<8} {"Baseline (mean±std)":<22} {"Ensemble (mean±std)":<22} {"Δstd":>8}')
        print('  ' + '-' * 75)

        for rho in RHO_EVAL:
            for ber in BER_EVAL:
                # Gather ensemble per-seed values
                ens_vals = [all_results[ds_name][s][str(rho)][str(ber)]
                            for s in all_results[ds_name] if s in all_results[ds_name]]
                if not ens_vals:
                    continue
                ens_mean = np.mean(ens_vals)
                ens_std = np.std(ens_vals)

                # Baseline (from existing summary)
                bl_key_rho = '1.0' if rho == 1.0 else f'{rho}'
                bl_key_ber = '0.0' if ber == 0 else f'{ber}' if ber != 0.3 else '0.3'

                bl_rho_dict = baseline.get(ds_name, {}).get(bl_key_rho, {})
                bl_stats = bl_rho_dict.get(bl_key_ber, None)
                if bl_stats:
                    bl_mean, bl_std = bl_stats['mean'], bl_stats['std']
                    delta_std = ens_std - bl_std
                    label = f'{ens_mean:.2f}±{ens_std:.2f}'
                    bl_label = f'{bl_mean:.2f}±{bl_std:.2f}'
                    print(f'  {rho:<8} {ber:<8} {bl_label:<22} {label:<22} {delta_std:>+8.2f}')
                    summary[ds_name][f'rho{rho}_ber{ber}'] = {
                        'ensemble_mean': ens_mean, 'ensemble_std': ens_std,
                        'baseline_mean': bl_mean, 'baseline_std': bl_std,
                        'delta_std': delta_std,
                    }

    # Save
    out = f'eval/seed_results/ensemble_top{args.k}_results.json'
    with open(out, 'w') as f:
        json.dump({'per_seed': all_results, 'summary': summary}, f, indent=2, default=float)
    print(f'\nSaved to {out}')


if __name__ == '__main__':
    main()
