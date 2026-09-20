#!/usr/bin/env python3
"""Independent re-verification of the SpikeAdapt+Taug winning checkpoints.

Loads each selected taug checkpoint and re-runs a FRESH test evaluation
(new channel randomness, fresh dataloader pass) at the key operating points,
then prints the delta against the saved grid values. Confirms the large
improvements are real, not an evaluation artifact.

Usage: python eval/reverify_taug.py
"""
import json
import os
import random
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'train'))
sys.path.insert(0, os.path.join(ROOT, 'models'))

from train_baselines_valproto import make_loaders, device       # noqa: E402
from train_spikeadapt_taug import load_v5cna, fwd_truncated     # noqa: E402
from run_final_pipeline import T_STEPS                          # noqa: E402

CELLS = [(0.75, 8, 0.0), (0.75, 8, 0.30), (0.75, 4, 0.30), (0.625, 4, 0.0)]


def eval_cell(model, back, front, loader, rho, tp, ber):
    model.eval(); back.eval()
    c = t = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            all_S2, m1, m2 = [], None, None
            for s in range(tp):
                _, s2, m1, m2 = model.encoder(feat, m1, m2, t=s)
                all_S2.append(s2)
            imp = model.scorer(all_S2, ber).squeeze(1)
            old = model.block_mask.target_rate
            model.block_mask.target_rate = rho
            mask, _ = model.block_mask(imp, training=False)
            model.block_mask.target_rate = old
            recv = [model.channel(all_S2[s] * mask, ber) for s in range(tp)]
            Fp = model.decoder(recv, mask)
            c += back(Fp).argmax(1).eq(labels).sum().item()
            t += labels.size(0)
    return round(100. * c / t, 2)


def main():
    # fresh, different RNG stream than training/grid runs
    torch.manual_seed(20260911); np.random.seed(20260911); random.seed(20260911)
    for ds in ['aid', 'resisc45']:
        grid = json.load(open(os.path.join(
            ROOT, f'eval/seed_results/spikeadapt_taug_joint_grid_{ds}.json')))
        print(f'===== {ds.upper()} =====', flush=True)
        for seed in [42, 123, 456]:
            _, _, test_loader, ncls = make_loaders(ds, seed)
            front, model, back = load_v5cna(ds, ncls, seed)
            snap = f'./snapshots_{ds}valp_v5cna_taug_seed{seed}/'
            cks = sorted([f for f in os.listdir(snap)
                          if f.startswith('taug_best_')],
                         key=lambda x: float(x.split('_')[-1][:-4]))
            ck = torch.load(os.path.join(snap, cks[-1]), map_location=device,
                            weights_only=False)
            model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
            for rho, tp, ber in CELLS:
                fresh = eval_cell(model, back, front, test_loader, rho, tp, ber)
                saved = grid[str(seed)]['test'][str(rho)][str(tp)][str(ber)]
                print(f'  s{seed} rho{rho}/T{tp}/BER{ber:g}: fresh {fresh:.2f} '
                      f'vs saved {saved:.2f}  (d={fresh-saved:+.2f})', flush=True)
            del model, front, back; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
