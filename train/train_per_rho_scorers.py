#!/usr/bin/env python3
"""Train per-ρ scorers for fair comparison.

For each (dataset, ρ) pair, trains a dedicated scorer (Stage 4 + Stage 5)
from the shared Stage 2 encoder/decoder checkpoint. This ensures each ρ
has its own optimized masking strategy.

Training matrix:
  2 datasets (AID, RESISC45) × 7 ρ values = 14 scorer trainings
  Each: Stage 4 (20 ep, scorer only) + Stage 5 (15 ep, joint FT)

Usage:
    python train/train_per_rho_scorers.py                    # all
    python train/train_per_rho_scorers.py --dataset aid       # AID only
    python train/train_per_rho_scorers.py --rho 0.25 0.5     # specific rhos
    python train/train_per_rho_scorers.py --eval-only         # just evaluate
"""

import os, sys, random, json, math, glob, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from train_aid_v2 import ResNet50Front, ResNet50Back, BSC_Channel, LearnedBlockMask, sample_noise
from train_aid_v5 import EncoderV5, DecoderV5, LIFNeuron, MPBN
from noise_aware_scorer import NoiseAwareScorer
from run_final_pipeline import AIDDataset5050, RESISC45Dataset, SpikeAdaptSC_v5c_NA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_STEPS = 8
BER_SWEEP = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
RHO_VALUES = [0.10, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875]

DATASET_CONFIGS = {
    'aid': {
        'n_classes': 30,
        'ds_cls': AIDDataset5050,
        'ds_kwargs_train': dict(seed=42),
        'ds_kwargs_test': dict(seed=42),
        'bb_path': './snapshots_aid_5050_seed42/backbone_best.pth',
        'v5_ck_path': './snapshots_aid_v5cna_seed42/v5cna_best_95.42.pth',
        'snap_base': './snapshots_aid_per_rho',
    },
    'resisc45': {
        'n_classes': 45,
        'ds_cls': RESISC45Dataset,
        'ds_kwargs_train': dict(train_ratio=0.20, seed=42),
        'ds_kwargs_test': dict(train_ratio=0.20, seed=42),
        'bb_path': './snapshots_resisc45_5050_seed42/backbone_best.pth',
        'v5_ck_path': './snapshots_resisc45_v5cna_seed42/v5cna_best_92.01.pth',
        'snap_base': './snapshots_resisc45_per_rho',
    },
}


def evaluate(model, back, front, loader, ber=0.0, target_rate=None):
    model.eval(); back.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            if target_rate is not None:
                Fp, _ = model(feat, noise_param=ber, target_rate_override=target_rate)
            else:
                Fp, _ = model(feat, noise_param=ber)
            correct += back(Fp).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total


def train_scorer_for_rho(model, back, front, train_loader, test_loader,
                          target_rho, snap_dir, ds_name, epochs_s4=20, epochs_s5=15):
    """Train scorer + joint FT for a specific rho."""
    rho_tag = f'rho{target_rho:.3f}'
    criterion = nn.CrossEntropyLoss()

    # ===== Stage 4: Scorer only =====
    print(f"\n  S4: Scorer training (rho={target_rho})")
    for n, p in model.named_parameters():
        p.requires_grad = 'scorer' in n
    for p in back.parameters():
        p.requires_grad = False

    scorer_params = list(model.scorer.parameters())
    opt = optim.Adam(scorer_params, lr=1e-3)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_s4, eta_min=1e-5)
    best = 0

    for ep in range(1, epochs_s4 + 1):
        model.train(); model.encoder.eval(); model.decoder.eval()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            ber = sample_noise('bsc')
            Fp, stats = model(feat, noise_param=ber, target_rate_override=target_rho)
            loss = criterion(back(Fp), labels)
            # Rate penalty
            if 'importance' in stats and stats['importance'] is not None:
                imp = stats['importance']
                if imp.dim() == 3:
                    imp = imp.unsqueeze(1)
                rate_loss = (imp.mean() - target_rho) ** 2
                loss = loss + 15.0 * rate_loss
            # Diversity
            div = model.scorer.compute_diversity_loss(stats['all_S2'], 0.0, 0.30)
            loss = loss + 0.05 * div
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(scorer_params, 1.0); opt.step()
        sch.step()

        if ep % 5 == 0 or ep == epochs_s4:
            acc = evaluate(model, back, front, test_loader, 0.15, target_rho)
            acc_c = evaluate(model, back, front, test_loader, 0.0, target_rho)
            print(f"    E{ep:02d}: Clean={acc_c:.2f}%, BER=0.15={acc:.2f}%")
            if acc > best:
                best = acc
                torch.save({'model': model.state_dict(), 'back': back.state_dict()},
                           os.path.join(snap_dir, f's4_{rho_tag}_best_{best:.2f}.pth'))

    # ===== Stage 5: Joint fine-tune =====
    print(f"  S5: Joint FT (rho={target_rho})")
    # Load best S4
    cks = sorted(glob.glob(os.path.join(snap_dir, f's4_{rho_tag}_best_*.pth')),
                 key=lambda x: float(x.split('_')[-1].replace('.pth', '')))
    if cks:
        ck = torch.load(cks[-1], map_location=device, weights_only=False)
        model.load_state_dict(ck['model'], strict=False)
        back.load_state_dict(ck['back'])

    for p in model.parameters(): p.requires_grad = True
    for p in back.parameters(): p.requires_grad = True
    all_params = list(model.parameters()) + list(back.parameters())
    opt = optim.Adam(all_params, lr=1e-5)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_s5, eta_min=1e-7)
    best_s5 = 0

    for ep in range(1, epochs_s5 + 1):
        model.train(); back.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            ber = sample_noise('bsc')
            Fp, stats = model(feat, noise_param=ber, target_rate_override=target_rho)
            loss = criterion(back(Fp), labels)
            if 'importance' in stats and stats['importance'] is not None:
                imp = stats['importance']
                if imp.dim() == 3: imp = imp.unsqueeze(1)
                rate_loss = (imp.mean() - target_rho) ** 2
                loss = loss + 10.0 * rate_loss
            div = model.scorer.compute_diversity_loss(stats['all_S2'], 0.0, 0.30)
            loss = loss + 0.05 * div
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()
        sch.step()

        if ep % 5 == 0 or ep == epochs_s5:
            acc = evaluate(model, back, front, test_loader, 0.15, target_rho)
            acc_c = evaluate(model, back, front, test_loader, 0.0, target_rho)
            print(f"    E{ep:02d}: Clean={acc_c:.2f}%, BER=0.15={acc:.2f}%")
            if acc > best_s5:
                best_s5 = acc
                torch.save({'model': model.state_dict(), 'back': back.state_dict()},
                           os.path.join(snap_dir, f's5_{rho_tag}_best_{best_s5:.2f}.pth'))

    return best_s5


def eval_all_rhos(ds_name, cfg, rho_list, test_loader, front):
    """Evaluate all per-rho trained scorers with full BER sweep."""
    snap_dir = cfg['snap_base']
    results = {}

    for rho in rho_list:
        rho_tag = f'rho{rho:.3f}'
        # Load per-rho checkpoint
        back = ResNet50Back(cfg['n_classes']).to(device)
        model = SpikeAdaptSC_v5c_NA(C_in=1024, C1=256, C2=36, T=T_STEPS,
                                     target_rate=rho, grid_size=14).to(device)

        loaded = False
        for tag in [f's5_{rho_tag}', f's4_{rho_tag}']:
            cks = sorted(glob.glob(os.path.join(snap_dir, f'{tag}_best_*.pth')),
                         key=lambda x: float(x.split('_')[-1].replace('.pth', '')))
            if cks:
                ck = torch.load(cks[-1], map_location=device, weights_only=False)
                model.load_state_dict(ck['model'], strict=False)
                back.load_state_dict(ck['back'])
                print(f"  Loaded {cks[-1]}")
                loaded = True
                break

        if not loaded:
            print(f"  WARNING: No checkpoint for rho={rho}, skipping")
            continue

        results[str(rho)] = {}
        for ber in BER_SWEEP:
            acc = evaluate(model, back, front, test_loader, ber, rho)
            results[str(rho)][str(ber)] = round(acc, 2)
            label = 'Clean' if ber == 0 else f'BER={ber:.2f}'
            print(f"    rho={rho:.3f} {label}: {acc:.2f}%")

        del model, back
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='both', choices=['aid', 'resisc45', 'both'])
    parser.add_argument('--rho', nargs='+', type=float, default=None,
                       help='Specific rho values (default: all)')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    rho_list = args.rho if args.rho else RHO_VALUES
    ds_list = ['aid', 'resisc45'] if args.dataset == 'both' else [args.dataset]

    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])
    tf_train = T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
                           T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))])

    all_results = {}

    for ds_name in ds_list:
        cfg = DATASET_CONFIGS[ds_name]
        snap_dir = cfg['snap_base']
        os.makedirs(snap_dir, exist_ok=True)

        print(f"\n{'#'*60}\n  {ds_name.upper()}\n{'#'*60}")

        # Load dataset
        if ds_name == 'aid':
            train_ds = cfg['ds_cls']("./data", tf_train, 'train', **cfg['ds_kwargs_train'])
            test_ds = cfg['ds_cls']("./data", tf_test, 'test', **cfg['ds_kwargs_test'])
        else:
            train_ds = cfg['ds_cls']("./data", tf_train, 'train', **cfg['ds_kwargs_train'])
            test_ds = cfg['ds_cls']("./data", tf_test, 'test', **cfg['ds_kwargs_test'])

        train_loader = DataLoader(train_ds, 32, True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)

        # Load backbone
        front = ResNet50Front(grid_size=14).to(device)
        bb = torch.load(cfg['bb_path'], map_location=device, weights_only=False)
        front.load_state_dict({k: v for k, v in bb.items()
                               if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))},
                              strict=False)
        front.eval()
        for p in front.parameters(): p.requires_grad = False

        if not args.eval_only:
            # Train each rho
            for rho in rho_list:
                print(f"\n{'='*60}")
                print(f"  {ds_name.upper()}: Training scorer for rho={rho}")
                print(f"{'='*60}")

                # Fresh model from V5C-NA checkpoint for each rho
                back = ResNet50Back(cfg['n_classes']).to(device)
                model = SpikeAdaptSC_v5c_NA(C_in=1024, C1=256, C2=36, T=T_STEPS,
                                             target_rate=rho, grid_size=14).to(device)

                ck = torch.load(cfg['v5_ck_path'], map_location=device, weights_only=False)
                model.load_state_dict(ck['model'], strict=False)
                back.load_state_dict(ck['back'])
                print(f"  Loaded base: {cfg['v5_ck_path']}")

                train_scorer_for_rho(model, back, front, train_loader, test_loader,
                                    rho, snap_dir, ds_name)

                del model, back
                torch.cuda.empty_cache()

        # Evaluate all rhos
        print(f"\n{'='*60}")
        print(f"  {ds_name.upper()}: Evaluating all per-rho scorers")
        print(f"{'='*60}")
        all_results[ds_name] = eval_all_rhos(ds_name, cfg, rho_list, test_loader, front)

        del front
        torch.cuda.empty_cache()

    # Save results
    os.makedirs('eval/seed_results', exist_ok=True)
    out = 'eval/seed_results/per_rho_scorer_results.json'
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out}")

    # Print summary tables
    for ds_name in all_results:
        print(f"\n{'='*80}")
        print(f"  {ds_name.upper()} — Per-ρ trained scorer results")
        print(f"{'='*80}")
        d = all_results[ds_name]
        rhos = sorted([float(r) for r in d.keys()])
        header = f"{'BER':<8}" + ''.join(f' rho={r:.3f}' for r in rhos)
        print(header)
        print('-' * len(header))
        for ber in BER_SWEEP:
            row = f"{'Clean' if ber==0 else f'{ber:.2f}':<8}"
            for rho in rhos:
                val = d.get(str(rho), {}).get(str(ber), 0)
                row += f' {val:>8.2f}%'
            print(row)


if __name__ == '__main__':
    main()
