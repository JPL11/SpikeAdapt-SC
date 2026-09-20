#!/usr/bin/env python3
"""SYMMETRIC EXPERIMENT: SpikeAdapt-SC fine-tuned WITH truncation augmentation.

The fair grouped-1bit ANN baseline was trained with random-T' truncation
augmentation; SpikeAdapt never was (its truncation is post-hoc). This script
completes the symmetry: fine-tune the existing val-protocol SpikeAdapt
checkpoints with per-batch random T' in {3,4,5,6,8} (same truncated-encoder /
zero-pad semantics as eval_joint_rho_T_grid.evaluate_truncated, but with
gradients and the Gumbel mask), same 50/50 BER curriculum, then emit the
standard [seed][split][rho][tprime][ber] joint grid so the identical Table-III
mission machinery scores it.

If augmented SpikeAdapt beats the grouped-1bit joint frontier, the original
spiking-forward framing partially survives; if parity persists, the
code-agnostic reframe stands on firm ground.

Usage: /home/jpli/miniconda/envs/semcom/bin/python train/train_spikeadapt_taug.py \
          --datasets aid --seeds 42 123 456 [--epochs 60]
Outputs: snapshots_{ds}valp_v5cna_taug_seed{seed}/
         eval/seed_results/spikeadapt_taug_joint_grid_{ds}.json
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

from train_baselines_valproto import make_loaders, device, ROOT   # noqa: E402
from train_aid_v2 import ResNet50Front, ResNet50Back              # noqa: E402
from run_final_pipeline import SpikeAdaptSC_v5c_NA, T_STEPS       # noqa: E402

RHOS = [0.625, 0.75, 0.875, 1.0]
TPRIMES = [3, 4, 5, 6, 8]
BERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def load_v5cna(ds, ncls, seed):
    """Frozen front + trained v5cna model/back from the valproto snapshots."""
    name = f'{ds}valp'
    bb = torch.load(f'./snapshots_{name}_5050_seed{seed}/backbone_best.pth',
                    map_location=device, weights_only=False)
    front = ResNet50Front(grid_size=14).to(device)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.',
                                                'spatial_pool.'))}, strict=False)
    front.eval()
    for p in front.parameters():
        p.requires_grad = False
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
    return front, model, back


def fwd_truncated(model, feat, ber, tprime, training):
    """Truncated forward matching evaluate_truncated, gradient-friendly."""
    all_S2, m1, m2 = [], None, None
    for t in range(tprime):
        _, s2, m1, m2 = model.encoder(feat, m1, m2, t=t)
        all_S2.append(s2)
    importance = model.scorer(all_S2, ber).squeeze(1)
    mask, tx = model.block_mask(importance, training=training)
    recv = [model.channel(all_S2[t] * mask, ber) for t in range(tprime)]
    Fp = model.decoder(recv, mask)
    return Fp, tx, all_S2


def eval_grid(model, back, front, loader):
    """[rho][tp][ber] accuracy grid; encoder amortized (run once at T=8,
    truncation = prefix slice, identical recurrence)."""
    model.eval(); back.eval()
    correct = {(r, tp, b): 0 for r in RHOS for tp in TPRIMES for b in BERS}
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            all_S2, m1, m2 = [], None, None
            for t in range(T_STEPS):
                _, s2, m1, m2 = model.encoder(feat, m1, m2, t=t)
                all_S2.append(s2)
            for tp in TPRIMES:
                S = all_S2[:tp]
                for ber in BERS:
                    imp = model.scorer(S, ber).squeeze(1)
                    for rho in RHOS:
                        old = model.block_mask.target_rate
                        model.block_mask.target_rate = rho
                        mask, _ = model.block_mask(imp, training=False)
                        model.block_mask.target_rate = old
                        recv = [model.channel(S[t] * mask, ber)
                                for t in range(tp)]
                        Fp = model.decoder(recv, mask)
                        correct[(rho, tp, ber)] += back(Fp).argmax(1).eq(
                            labels).sum().item()
            total += labels.size(0)
    return {str(r): {str(tp): {str(b): round(100. * correct[(r, tp, b)] / total, 2)
                               for b in BERS} for tp in TPRIMES} for r in RHOS}


def run_taug(ds, seed, epochs, lr):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_loader, val_loader, test_loader, ncls = make_loaders(ds, seed)
    front, model, back = load_v5cna(ds, ncls, seed)

    snap = f'./snapshots_{ds}valp_v5cna_taug_seed{seed}/'
    os.makedirs(snap, exist_ok=True)
    existing = sorted([f for f in os.listdir(snap) if f.startswith('taug_best_')],
                      key=lambda x: float(x.split('_')[-1][:-4]))
    if not existing:
        opt = optim.Adam(list(model.parameters()) + list(back.parameters()),
                         lr=lr)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                     eta_min=1e-6)
        crit = nn.CrossEntropyLoss()

        def sample_noise():
            return (random.uniform(0.0, 0.15) if random.random() < 0.5
                    else random.uniform(0.15, 0.40))

        best = 0
        for epoch in range(1, epochs + 1):
            model.train(); back.train()
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                feat = front(imgs)
                ber = sample_noise()
                tp = random.choice(TPRIMES)         # truncation augmentation
                Fp, tx, all_S2 = fwd_truncated(model, feat, ber, tp,
                                               training=True)
                loss = crit(back(Fp), labels) + 2.0 * (tx - 0.75) ** 2
                loss = loss + 0.05 * model.scorer.compute_diversity_loss(
                    [s.detach() for s in all_S2])
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            if epoch % 10 == 0 or epoch == epochs:
                model.eval(); back.eval()
                correct = tot = 0
                with torch.no_grad():
                    for imgs, labels in val_loader:
                        imgs, labels = imgs.to(device), labels.to(device)
                        Fp, _, _ = fwd_truncated(model, front(imgs), 0.0,
                                                 T_STEPS, training=False)
                        correct += back(Fp).argmax(1).eq(labels).sum().item()
                        tot += labels.size(0)
                acc = 100. * correct / tot
                print(f'  E{epoch:02d} (val): {acc:.2f}%', flush=True)
                if acc > best:
                    best = acc
                    torch.save({'model': model.state_dict(),
                                'back': back.state_dict()},
                               os.path.join(snap, f'taug_best_{acc:.2f}.pth'))
        existing = sorted([f for f in os.listdir(snap)
                           if f.startswith('taug_best_')],
                          key=lambda x: float(x.split('_')[-1][:-4]))
    else:
        print(f'  found existing {existing[-1]}', flush=True)
    ck = torch.load(os.path.join(snap, existing[-1]), map_location=device,
                    weights_only=False)
    model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])

    out = {'selected': existing[-1]}
    out['test'] = eval_grid(model, back, front, test_loader)
    out['val'] = eval_grid(model, back, front, val_loader)
    print(f"  {ds} s{seed} taug: test r.75/T8 clean "
          f"{out['test']['0.75']['8']['0.0']} / ber.3 "
          f"{out['test']['0.75']['8']['0.3']}; T4 ber.3 "
          f"{out['test']['0.75']['4']['0.3']}", flush=True)
    del model, front, back; torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    ap.add_argument('--datasets', nargs='+', default=['aid'])
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--lr', type=float, default=5e-5)
    args = ap.parse_args()
    for ds in args.datasets:
        out = os.path.join(ROOT,
                           f'eval/seed_results/spikeadapt_taug_joint_grid_{ds}.json')
        res = json.load(open(out)) if os.path.exists(out) else {}
        for seed in args.seeds:
            if str(seed) in res:
                print(f'  {ds} seed {seed}: done, skip', flush=True); continue
            print(f'== spikeadapt-taug {ds} seed {seed}', flush=True)
            res[str(seed)] = run_taug(ds, seed, args.epochs, args.lr)
            json.dump(res, open(out, 'w'), indent=1)
        print(f'saved {out}', flush=True)


if __name__ == '__main__':
    main()
