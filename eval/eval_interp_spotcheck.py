#!/usr/bin/env python3
"""Spot-validation of the bilinear grid interpolation used by the closed loop.

Fresh per-image inference at OFF-GRID BERs (midpoints of the policy grid's
p-axis) and one off-grid rho, seed 42, compared against linear interpolation
of eval/seed_results/joint_rho_T_grid.json along p.

Output: eval/seed_results/interp_spotcheck.json
    {dataset: {"rho": r, "tprime": tp, "points": [{ber, fresh, interp, delta}]}}

Usage: python eval/eval_interp_spotcheck.py
"""

import os, sys, json, glob, random
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train_aid_v2 import ResNet50Front
from run_final_pipeline import AIDDataset5050, RESISC45Dataset, SpikeAdaptSC_v5c_NA
from eval_joint_rho_T_grid import pick_checkpoint, evaluate_truncated

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = 'eval/seed_results/interp_spotcheck.json'
SEED = 42
OFF_BERS = [0.025, 0.075, 0.125, 0.175, 0.225, 0.275]
GRID_BERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
CONFIGS = [(0.75, 6), (0.875, 4)]          # on-grid (rho, T') rows
OFF_RHO = (0.70, 6, [0.125, 0.275])        # off-grid rho spot points


def interp_p(grid_row, p):
    xs, ys = zip(*sorted((float(b), a) for b, a in grid_row.items()))
    return float(np.interp(p, xs, ys))


def interp_rho(grid_seed, tp, p, rho):
    pairs = sorted((float(r), r) for r in grid_seed if str(tp) in grid_seed[r])
    rhos = [fr for fr, _ in pairs]
    vals = [interp_p(grid_seed[key][str(tp)], p) for _, key in pairs]
    return float(np.interp(rho, rhos, vals))


def main():
    print(f'Device: {device}', flush=True)
    grid = json.load(open('eval/seed_results/joint_rho_T_grid.json'))
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                         T.Normalize((.485, .456, .406), (.229, .224, .225))])
    results = {}
    if os.path.exists(OUT):
        results = json.load(open(OUT))

    for ds_name, ds_key, n_classes in [('AID', 'aid', 30),
                                       ('RESISC45', 'resisc45', 45)]:
        bb_path = f'./snapshots_{ds_key}_5050_seed{SEED}/backbone_best.pth'
        ck_path = pick_checkpoint(f'./snapshots_{ds_key}_v5cna_seed{SEED}')
        assert ck_path and os.path.exists(bb_path), f'missing ckpt for {ds_key}'

        torch.manual_seed(1234 + SEED); np.random.seed(1234 + SEED)
        random.seed(1234 + SEED)
        if ds_key == 'aid':
            test_ds = AIDDataset5050('./data', tf_test, 'test', seed=SEED)
        else:
            test_ds = RESISC45Dataset('./data', tf_test, 'test',
                                      train_ratio=0.20, seed=SEED)
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

        from train_aid_v2 import ResNet50Back
        back = ResNet50Back(n_classes).to(device)
        model = SpikeAdaptSC_v5c_NA(C_in=1024, C1=256, C2=36, T=8,
                                    target_rate=0.75, grid_size=14).to(device)
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        back.load_state_dict(ck['back'])
        model.eval(); back.eval()

        ds_res = results.setdefault(ds_key, {})
        gseed = grid[ds_key][str(SEED)]

        for rho, tp in CONFIGS:
            key = f'rho{rho}_T{tp}'
            pts = ds_res.setdefault(key, [])
            done = {round(p['ber'], 3) for p in pts}
            for ber in OFF_BERS:
                if round(ber, 3) in done:
                    continue
                torch.manual_seed(42)  # deterministic noise, same as grid
                fresh = evaluate_truncated(model, back, front, loader,
                                           ber, rho, tp)
                itp = interp_p(gseed[f'{rho:g}'][str(tp)], ber)
                pts.append(dict(ber=ber, fresh=fresh, interp=itp,
                                delta=fresh - itp))
                print(f'[{ds_name}] rho={rho} T={tp} p={ber}: '
                      f'fresh={fresh:.2f} interp={itp:.2f} '
                      f'delta={fresh-itp:+.2f}', flush=True)
                json.dump(results, open(OUT, 'w'), indent=1)

        rho_o, tp_o, bers_o = OFF_RHO
        key = f'rho{rho_o}_T{tp_o}_offrho'
        pts = ds_res.setdefault(key, [])
        done = {round(p['ber'], 3) for p in pts}
        for ber in bers_o:
            if round(ber, 3) in done:
                continue
            torch.manual_seed(42)
            fresh = evaluate_truncated(model, back, front, loader,
                                       ber, rho_o, tp_o)
            itp = interp_rho(gseed, tp_o, ber, rho_o)
            pts.append(dict(ber=ber, fresh=fresh, interp=itp,
                            delta=fresh - itp))
            print(f'[{ds_name}] OFFRHO rho={rho_o} T={tp_o} p={ber}: '
                  f'fresh={fresh:.2f} interp={itp:.2f} '
                  f'delta={fresh-itp:+.2f}', flush=True)
            json.dump(results, open(OUT, 'w'), indent=1)

    all_d = [abs(p['delta']) for ds in results.values()
             for pts in ds.values() for p in pts]
    print(f'\nSpot-check complete: n={len(all_d)}, '
          f'max|delta|={max(all_d):.2f} pp, mean|delta|={np.mean(all_d):.2f} pp',
          flush=True)


if __name__ == '__main__':
    main()
