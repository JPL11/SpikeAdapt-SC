#!/usr/bin/env python3
"""Per-sample records for the SNN-SC-HARQ-style baseline (visual).

SNN-SC-HARQ (Wang et al., TVT 2025) grows rate by RETRANSMISSION, gated by a
receiver-side similarity score, spatially uniform, single-axis. We port that
POLICY onto our own (taug) codec so the comparison isolates the adaptation
mechanism: incremental-T' retransmission with an RX confidence proxy (pmax)
versus our one-shot pilot-driven joint (rho,T') loop.

This script records, for every sample, correctness and RX confidence at each
prefix depth T' in {3,4,5,6,8} under one consistent channel draw per timestep
(earlier receptions are reused when more steps arrive -- true HARQ semantics),
at rho=1 (spatially uniform), for each grid BER, on BOTH val (threshold
selection) and test (scored once). The stopping rule for ANY threshold tau can
then be simulated offline (eval/harq_style_policy.py) without re-running.

Usage: python eval/harq_style_persample.py --datasets aid resisc45
Output: eval/seed_results/harq_style_persample_{ds}.npz
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'train'))
sys.path.insert(0, os.path.join(ROOT, 'models'))

from train_baselines_valproto import make_loaders, device       # noqa: E402
from train_spikeadapt_taug import load_v5cna                    # noqa: E402
from run_final_pipeline import T_STEPS                          # noqa: E402

SCHEDULE = [3, 4, 5, 6, 8]
BERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def record(model, back, front, loader):
    """-> correct[N, len(BERS), len(SCHEDULE)] uint8,
          pmax[N, len(BERS), len(SCHEDULE)] float16"""
    model.eval(); back.eval()
    C, P = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            all_S2, m1, m2 = [], None, None
            for t in range(T_STEPS):
                _, s2, m1, m2 = model.encoder(feat, m1, m2, t=t)
                all_S2.append(s2)
            imp = model.scorer(all_S2, 0.0).squeeze(1)
            old = model.block_mask.target_rate
            model.block_mask.target_rate = 1.0          # spatially uniform
            mask, _ = model.block_mask(imp, training=False)
            model.block_mask.target_rate = old
            B = imgs.size(0)
            cb = np.zeros((B, len(BERS), len(SCHEDULE)), dtype=np.uint8)
            pb = np.zeros((B, len(BERS), len(SCHEDULE)), dtype=np.float16)
            for bi, ber in enumerate(BERS):
                # ONE channel draw per timestep; prefixes reuse receptions
                recv_all = [model.channel(all_S2[t] * mask, ber)
                            for t in range(T_STEPS)]
                for si, tp in enumerate(SCHEDULE):
                    Fp = model.decoder(recv_all[:tp], mask)
                    logits = back(Fp)
                    prob = F.softmax(logits, dim=1)
                    pmax, pred = prob.max(1)
                    cb[:, bi, si] = pred.eq(labels).cpu().numpy()
                    pb[:, bi, si] = pmax.cpu().numpy().astype(np.float16)
            C.append(cb); P.append(pb)
    return np.concatenate(C), np.concatenate(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['aid', 'resisc45'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    args = ap.parse_args()
    for ds in args.datasets:
        out_p = os.path.join(ROOT,
                             f'eval/seed_results/harq_style_persample_{ds}.npz')
        store = dict(np.load(out_p)) if os.path.exists(out_p) else {}
        for seed in args.seeds:
            if f's{seed}_test_correct' in store:
                print(f'{ds} s{seed}: done, skip', flush=True); continue
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            _, val_loader, test_loader, ncls = make_loaders(ds, seed)
            front, model, back = load_v5cna(ds, ncls, seed)
            snap = f'./snapshots_{ds}valp_v5cna_taug_seed{seed}/'
            cks = sorted([f for f in os.listdir(snap)
                          if f.startswith('taug_best_')],
                         key=lambda x: float(x.split('_')[-1][:-4]))
            ck = torch.load(os.path.join(snap, cks[-1]), map_location=device,
                            weights_only=False)
            model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
            for split, loader in [('val', val_loader), ('test', test_loader)]:
                c, p = record(model, back, front, loader)
                store[f's{seed}_{split}_correct'] = c
                store[f's{seed}_{split}_pmax'] = p
                print(f'{ds} s{seed} {split}: {c.shape[0]} samples; e.g. '
                      f'T8@BER0.3 acc {c[:, -1, -1].mean()*100:.2f}', flush=True)
            np.savez_compressed(out_p, **store)
            del model, front, back; torch.cuda.empty_cache()
        print(f'saved {out_p}', flush=True)


if __name__ == '__main__':
    main()
