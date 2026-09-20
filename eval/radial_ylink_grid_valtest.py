#!/usr/bin/env python3
"""Physical-receiver (y_link) val/test grid for the RADIal SNN (panel-3
round-2 fix): the FULL (rho, T', BER) surface under the support-aware
receiver y = m * (s XOR e), so the closed-loop policy can be RESELECTED on a
physical validation surface instead of merely re-scoring stress-selected
cells. Same layout as radial_grid_valtest.json, so the closed-loop and
Table-II machinery consume it via --grid.

Prefills (no GPU cost):
 - every ber=0.0 cell from radial_grid_valtest.json (y_link == y_stress at
   p=0: no flips to scatter; identical deterministic eval);
 - the 12x3 TEST cells already evaluated by radial_remask_spotcheck.py.
Eval order: (1) test rho0.75/T8 (Fig-2 curve), (2) full val (policy
surface), (3) remaining test. Incremental/resumable per cell.

Usage: SPIKES_ONLY=1 python eval/radial_ylink_grid_valtest.py
Output: eval/seed_results/radial_ylink_grid_valtest.json
"""
import contextlib
import io
import json
import os
import re
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'radial_repo', 'FFTRadNet'))

os.environ['SPIKES_ONLY'] = '1'

from utils.evaluation import run_FullEvaluation                     # noqa: E402
from train.train_radial_snn import CONFIG, build_loaders            # noqa: E402
from eval.radial_grid_rho_T import DIMS, C_SPIKE                    # noqa: E402
from eval.radial_grid_valtest import RHOS, TPRIMES, BERS, SEED_CKPT, load_net  # noqa: E402
from eval.radial_remask_spotcheck import RemaskNet                  # noqa: E402

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STRESS = os.path.join(ROOT, 'eval/seed_results/radial_grid_valtest.json')
SPOT = os.path.join(ROOT, 'eval/seed_results/radial_remask_spotcheck.json')
OUT = os.path.join(ROOT, 'eval/seed_results/radial_ylink_grid_valtest.json')


def prefill(res, full_spatial):
    stress = json.load(open(STRESS))
    spot = json.load(open(SPOT)) if os.path.exists(SPOT) else {}
    n = 0
    for seed in SEED_CKPT:
        node = res.setdefault(seed, {})
        for split in ['val', 'test']:
            sp = node.setdefault(split, {})
            for rho in RHOS:
                for tp in TPRIMES:
                    key = f'rho{rho}_T{tp}_ber0.0'
                    if key not in sp and key in stress[seed][split]:
                        sp[key] = dict(stress[seed][split][key])
                        sp[key]['prefill'] = 'ber0-identity'
                        n += 1
        # spot-check cells are TEST-split evals under the same receiver
        for key, m in spot.get(seed, {}).items():
            sp = node['test']
            if key not in sp:
                rho, tp = re.match(r'rho([\d.]+)_T(\d+)_', key).groups()
                sp[key] = {'payload_bits': int(full_spatial * float(rho) * int(tp)),
                           'mAP': m['mAP'], 'mAR': m['mAR'], 'mIoU': m['mIoU'],
                           'prefill': 'remask-spotcheck'}
                n += 1
    return n


def cell_order():
    """(split, rho, tp, ber) in priority order."""
    order = []
    for ber in BERS:                                   # 1) Fig-2 curve cells
        order.append(('test', 0.75, 8, ber))
    for rho in RHOS:                                   # 2) policy surface
        for tp in TPRIMES:
            for ber in BERS:
                order.append(('val', rho, tp, ber))
    for rho in RHOS:                                   # 3) remaining test
        for tp in TPRIMES:
            for ber in BERS:
                order.append(('test', rho, tp, ber))
    return order


def main():
    config = json.load(open(CONFIG))
    torch.manual_seed(config['seed'])
    enc, _, val_loader, test_loader = build_loaders(config)
    loaders = {'val': val_loader, 'test': test_loader}
    full_spatial = sum(C_SPIKE * h * w for h, w in DIMS.values())
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    n = prefill(res, full_spatial)
    if n:
        json.dump(res, open(OUT, 'w'))
        print(f'prefilled {n} cells', flush=True)
    for seed, ckpt in SEED_CKPT.items():
        net = None
        node = res.setdefault(seed, {})
        for split, rho, tp, ber in cell_order():
            sp = node.setdefault(split, {})
            key = f'rho{rho}_T{tp}_ber{ber}'
            if key in sp:
                continue
            if net is None:
                inner = load_net(config, ckpt).net
                net = RemaskNet(inner).to(device).eval()
            net.rho, net.tprime, net.ber = rho, tp, ber
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run_FullEvaluation(net, loaders[split], enc)
            text = buf.getvalue()
            nums = {'payload_bits': int(full_spatial * rho * tp)}
            for mk in ['mAP', 'mAR', 'mIoU']:
                m = re.search(rf'{mk}:?\s+([0-9.]+)', text)
                nums[mk] = float(m.group(1)) if m else None
            sp[key] = nums
            json.dump(res, open(OUT, 'w'))
            print(f"{seed} {split} r{rho} T{tp} b{ber}: mAP {nums['mAP']:.3f} "
                  f"mAR {nums['mAR']:.3f}", flush=True)
        del net
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    print('saved:', OUT, flush=True)


if __name__ == '__main__':
    main()
