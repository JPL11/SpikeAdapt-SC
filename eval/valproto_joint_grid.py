#!/usr/bin/env python3
"""Path A stage 4a: joint (rho, T') x BER grids on the val-protocol
checkpoints, evaluated on BOTH the validation surface (policy construction)
and the test surface (reporting). Truncation semantics identical to
eval_joint_rho_T_grid.py (encoder runs T' steps, scorer sees truncated
average, decoder zero-pads via mask path).

Grid: rho in {0.625, 0.75, 0.875, 1.0} x T' in {3,4,5,6,8} x
      BER in {0, 0.05, ..., 0.30}; seeds 42/123/456 (matching the original
      joint-grid protocol; the rho-only axis has all 10 seeds in
      valproto_rho_grid_{ds}.json).

Usage: python eval/valproto_joint_grid.py [--seeds 42 123 456]
Output: eval/seed_results/valproto_joint_grid_{ds}.json
        keyed [seed][split][rho][tprime][ber]
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'train'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_final_pipeline import (ResNet50Front, ResNet50Back,        # noqa: E402
                                SpikeAdaptSC_v5c_NA, T_STEPS, device)
from valproto_rho_grid import loaders                               # noqa: E402
from eval_joint_rho_T_grid import evaluate_truncated                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RHOS = [0.625, 0.75, 0.875, 1.0]
TPRIMES = [3, 4, 5, 6, 8]
BERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
BERS_EXT = [0.35, 0.40, 0.45, 0.50]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    ap.add_argument('--ext', action='store_true',
                    help='evaluate the extended BER set 0.35-0.50 (review fix: full-slot dynamic scoring)')
    args = ap.parse_args()
    global BERS
    if args.ext:
        BERS = BERS_EXT
    for ds in ['aid', 'resisc45']:
        out = os.path.join(ROOT, f'eval/seed_results/valproto_joint_grid_{ds}.json')
        res = json.load(open(out)) if os.path.exists(out) else {}
        for seed in args.seeds:
            if not os.path.isdir(f'./snapshots_{ds}valp_v5cna_seed{seed}'):
                print(f'{ds} s{seed}: not ready, skip', flush=True)
                continue
            lds, ncls = loaders(ds, seed)
            fmb = None
            node = res.setdefault(str(seed), {})
            for split in ['val', 'test']:
                sp = node.setdefault(split, {})
                for rho in RHOS:
                    rp = sp.setdefault(str(rho), {})
                    for tp in TPRIMES:
                        tpd = rp.setdefault(str(tp), {})
                        for ber in BERS:
                            if str(ber) in tpd:
                                continue
                            if fmb is None:
                                fmb = load_model(ds, ncls, seed)
                            front, model, back = fmb
                            acc = evaluate_truncated(model, back, front,
                                                     lds[split], ber, rho, tp)
                            tpd[str(ber)] = round(acc, 3)
                            json.dump(res, open(out, 'w'))
                            print(f'{ds} s{seed} {split} r{rho} T{tp} '
                                  f'b{ber}: {acc:.2f}', flush=True)
        print(f'{ds}: saved {out}', flush=True)


if __name__ == '__main__':
    main()
