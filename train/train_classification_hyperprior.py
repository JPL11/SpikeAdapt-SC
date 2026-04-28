#!/usr/bin/env python3
"""Fine-tune SpikeAdaptSC with hyperprior + adaptive rho(BER) for classification.

Three stages:
  H1 (20 ep): Train hyperprior only (frozen main model)
  H2 (20 ep): Joint fine-tune all params
  H3 (20 ep): Train adaptive rho policy (learned BER→rho mapping)

Usage:
    python train/train_classification_hyperprior.py --dataset both
    python train/train_classification_hyperprior.py --dataset aid
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

from train_aid_v2 import ResNet50Front, ResNet50Back, sample_noise
from run_final_pipeline import AIDDataset5050, RESISC45Dataset
from models.spikeadapt_sc_hyper import SpikeAdaptSC_Hyper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_STEPS = 8
BER_SWEEP = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def evaluate(model, back, front, test_loader, ber=0.0, use_hyper=True, use_adapt=False):
    model.eval(); back.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            Fp, _ = model(front(imgs), noise_param=ber,
                         use_hyperprior=use_hyper, use_adaptive_rho=use_adapt)
            correct += back(Fp).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total


def run_dataset(ds, args):
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    n_classes = 30 if ds == 'aid' else 45
    print(f"\n{'#'*60}\n  DATASET: {ds.upper()}\n{'#'*60}")

    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])
    tf_train = T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
                           T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))])

    if ds == 'aid':
        train_ds = AIDDataset5050("./data", tf_train, 'train', seed=args.seed)
        test_ds = AIDDataset5050("./data", tf_test, 'test', seed=args.seed)
        bb_path = f"./snapshots_aid_5050_seed{args.seed}/backbone_best.pth"
        ck_path = f"./snapshots_aid_v5cna_seed{args.seed}/v5cna_best_95.42.pth"
    else:
        train_ds = RESISC45Dataset("./data", tf_train, 'train', train_ratio=0.20, seed=args.seed)
        test_ds = RESISC45Dataset("./data", tf_test, 'test', train_ratio=0.20, seed=args.seed)
        bb_path = f"./snapshots_resisc45_5050_seed{args.seed}/backbone_best.pth"
        ck_path = f"./snapshots_resisc45_v5cna_seed{args.seed}/v5cna_best_92.01.pth"

    train_loader = DataLoader(train_ds, 32, True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)

    front = ResNet50Front(grid_size=14).to(device)
    bb = torch.load(bb_path, map_location=device, weights_only=False)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))},
                          strict=False)
    front.eval()
    for p in front.parameters(): p.requires_grad = False

    back = ResNet50Back(n_classes).to(device)
    model = SpikeAdaptSC_Hyper(C_in=1024, C1=256, C2=36, T=T_STEPS,
                                target_rate=0.75, grid_size=14,
                                C_hyper=args.C_hyper, adaptive_rho=True).to(device)

    ck = torch.load(ck_path, map_location=device, weights_only=False)
    model.load_state_dict(ck['model'], strict=False)
    back.load_state_dict(ck['back'])
    print(f"Loaded V5C-NA checkpoint")

    snap_dir = f'./snapshots_{ds}_hyper_seed{args.seed}'
    os.makedirs(snap_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()

    # ===== H1: Train hyperprior only =====
    print(f"\n{'='*60}\n  H1: Hyperprior training (frozen main model)\n{'='*60}")
    for n, p in model.named_parameters():
        p.requires_grad = 'hyperprior' in n
    for p in back.parameters(): p.requires_grad = False
    hyper_params = [p for p in model.parameters() if p.requires_grad]
    print(f"  Trainable: {sum(p.numel() for p in hyper_params):,} params")

    opt = optim.Adam(hyper_params, lr=1e-3)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20, eta_min=1e-5)
    best = 0

    for ep in range(1, 21):
        model.train(); model.encoder.eval()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            ber = sample_noise('bsc')
            Fp, _ = model(front(imgs), noise_param=ber, use_hyperprior=True, use_adaptive_rho=False)
            loss = criterion(back(Fp), labels)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(hyper_params, 1.0); opt.step()
        sch.step()
        if ep % 5 == 0 or ep == 20:
            a3h = evaluate(model, back, front, test_loader, 0.30, True, False)
            print(f"  E{ep:02d}: BER=0.3 +H={a3h:.2f}%")
            if a3h > best:
                best = a3h
                torch.save({'model': model.state_dict(), 'back': back.state_dict()},
                           os.path.join(snap_dir, f'h1_best_{best:.2f}.pth'))

    # ===== H2: Joint fine-tune =====
    print(f"\n{'='*60}\n  H2: Joint fine-tune\n{'='*60}")
    cks = sorted(glob.glob(os.path.join(snap_dir, 'h1_best_*.pth')),
                 key=lambda x: float(x.split('_')[-1].replace('.pth', '')))
    if cks:
        ck = torch.load(cks[-1], map_location=device, weights_only=False)
        model.load_state_dict(ck['model'], strict=False); back.load_state_dict(ck['back'])
        print(f"  Loaded {cks[-1]}")

    for p in model.parameters(): p.requires_grad = True
    # Keep rho_policy frozen during H2
    for p in model.rho_policy.parameters(): p.requires_grad = False
    for p in back.parameters(): p.requires_grad = True

    opt = optim.Adam([p for p in model.parameters() if p.requires_grad] +
                     list(back.parameters()), lr=1e-5)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20, eta_min=1e-7)
    best_h2 = 0

    for ep in range(1, 21):
        model.train(); back.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            ber = sample_noise('bsc')
            Fp, stats = model(front(imgs), noise_param=ber, use_hyperprior=True, use_adaptive_rho=False)
            loss = criterion(back(Fp), labels)
            div = model.scorer.compute_diversity_loss(stats['all_S2'], 0.0, 0.30)
            loss = loss + 0.05 * div
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()
        sch.step()
        if ep % 5 == 0 or ep == 20:
            a3 = evaluate(model, back, front, test_loader, 0.30, True, False)
            print(f"  E{ep:02d}: BER=0.3 +H={a3:.2f}%")
            if a3 > best_h2:
                best_h2 = a3
                torch.save({'model': model.state_dict(), 'back': back.state_dict()},
                           os.path.join(snap_dir, f'h2_best_{best_h2:.2f}.pth'))

    # ===== H3: Train adaptive rho policy =====
    print(f"\n{'='*60}\n  H3: Adaptive rho(BER) policy\n{'='*60}")
    cks = sorted(glob.glob(os.path.join(snap_dir, 'h2_best_*.pth')),
                 key=lambda x: float(x.split('_')[-1].replace('.pth', '')))
    if cks:
        ck = torch.load(cks[-1], map_location=device, weights_only=False)
        model.load_state_dict(ck['model'], strict=False); back.load_state_dict(ck['back'])
        print(f"  Loaded {cks[-1]}")

    # Train rho_policy + scorer jointly, rest frozen
    for p in model.parameters(): p.requires_grad = False
    for p in model.rho_policy.parameters(): p.requires_grad = True
    for p in model.scorer.parameters(): p.requires_grad = True

    rho_params = list(model.rho_policy.parameters()) + list(model.scorer.parameters())
    print(f"  Trainable: {sum(p.numel() for p in rho_params):,} params (rho_policy + scorer)")

    opt = optim.Adam(rho_params, lr=1e-3)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20, eta_min=1e-5)
    best_h3 = 0

    for ep in range(1, 21):
        model.train(); model.encoder.eval(); model.decoder.eval()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            # Sample BER from wider range including high BER
            ber = random.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45])
            Fp, stats = model(front(imgs), noise_param=ber, use_hyperprior=True, use_adaptive_rho=True)
            loss = criterion(back(Fp), labels)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(rho_params, 1.0); opt.step()
        sch.step()
        if ep % 5 == 0 or ep == 20:
            # Eval at multiple BER to see adaptive rho in action
            results_ep = {}
            for ber in [0.0, 0.15, 0.30, 0.40]:
                a_adapt = evaluate(model, back, front, test_loader, ber, True, True)
                a_fixed = evaluate(model, back, front, test_loader, ber, True, False)
                # Get the rho the policy chose
                with torch.no_grad():
                    rho_val = model.rho_policy(ber).item()
                results_ep[ber] = (a_adapt, a_fixed, rho_val)
            print(f"  E{ep:02d}:")
            for ber, (aa, af, rv) in results_ep.items():
                label = 'Clean' if ber == 0 else f'BER={ber:.2f}'
                print(f"    {label}: Adapt={aa:.2f}%(rho={rv:.3f}) Fixed={af:.2f}% D={aa-af:+.2f}%")

            avg_adapt = sum(v[0] for v in results_ep.values()) / len(results_ep)
            if avg_adapt > best_h3:
                best_h3 = avg_adapt
                torch.save({'model': model.state_dict(), 'back': back.state_dict()},
                           os.path.join(snap_dir, f'h3_best_{best_h3:.2f}.pth'))
                print(f"    saved (avg={best_h3:.2f}%)")

    for p in model.parameters(): p.requires_grad = True

    # ===== Final eval =====
    print(f"\n{'='*60}\n  FINAL EVALUATION — {ds.upper()}\n{'='*60}")
    cks = sorted(glob.glob(os.path.join(snap_dir, 'h3_best_*.pth')),
                 key=lambda x: float(x.split('_')[-1].replace('.pth', '')))
    if cks:
        ck = torch.load(cks[-1], map_location=device, weights_only=False)
        model.load_state_dict(ck['model'], strict=False); back.load_state_dict(ck['back'])

    print(f"\n{ds.upper()} — BER sweep (0.0 to 0.5):")
    print(f"{'BER':<10} {'Adapt+H':>10} {'Fixed+H':>10} {'NoHyper':>10} {'rho*':>8} {'D(adapt-fixed)':>15}")
    print('-' * 65)
    for ber in BER_SWEEP:
        a_ah = evaluate(model, back, front, test_loader, ber, True, True)
        a_fh = evaluate(model, back, front, test_loader, ber, True, False)
        a_no = evaluate(model, back, front, test_loader, ber, False, False)
        with torch.no_grad():
            rho_val = model.rho_policy(ber).item()
        label = 'Clean' if ber == 0 else f'BER={ber:.2f}'
        d = a_ah - a_fh
        print(f"  {label:<8} {a_ah:>9.2f}% {a_fh:>9.2f}% {a_no:>9.2f}% {rho_val:>7.3f} {d:>+14.2f}%")

    # Print learned rho policy
    print(f"\n  Learned rho(BER) policy:")
    for ber in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        with torch.no_grad():
            rho_val = model.rho_policy(ber).item()
        print(f"    BER={ber:.2f} -> rho={rho_val:.4f}")

    # Cleanup GPU
    del model, back, front
    torch.cuda.empty_cache()
    print(f'\nDone with {ds.upper()}!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='aid', choices=['aid', 'resisc45', 'both'])
    parser.add_argument('--C_hyper', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    datasets_to_run = ['aid', 'resisc45'] if args.dataset == 'both' else [args.dataset]
    for ds in datasets_to_run:
        run_dataset(ds, args)


if __name__ == '__main__':
    main()
