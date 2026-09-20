#!/usr/bin/env python3
"""Extra Monte Carlo verification points for the JPEG+LDPC analytic model.

Reviewer fix: the all-or-nothing survival model was MC-verified only at
p in {0.06, 0.08} (rate 1/2). Add transition-region points for BOTH codes:
rate-1/2 at p in {0.04, 0.05}, rate-1/3 at p in {0.07, 0.08}.
Merges results into eval/seed_results/jpeg_ldpc_results.json under
'mc_check_extra'.

Usage:
    python eval/eval_jpeg_mc_extra.py
"""

import json
import os
import sys

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from train_aid_v2 import ResNet50Front, ResNet50Back
from run_final_pipeline import AIDDataset5050, RESISC45Dataset
from eval.ldpc import make_regular_ldpc, MinSumDecoder
from eval_jpeg_ldpc import mc_spot_check

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_IMAGES = 300

CODES = {
    'r12_n648_dv3dc6': (dict(n=648, dv=3, dc=6), [0.04, 0.05]),
    'r13_n648_dv4dc6': (dict(n=648, dv=4, dc=6), [0.07, 0.08]),
}


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                         T.Normalize((.485, .456, .406), (.229, .224, .225))])

    with open('eval/seed_results/jpeg_ldpc_results.json') as f:
        results = json.load(f)

    for ds_name, ds_key, n_classes, DsCls, ds_kwargs, bb_path in [
        ('AID', 'aid', 30, AIDDataset5050, dict(seed=42),
         './snapshots_aid_5050_seed42/backbone_best.pth'),
        ('RESISC45', 'resisc45', 45, RESISC45Dataset,
         dict(train_ratio=0.20, seed=42),
         './snapshots_resisc45_5050_seed42/backbone_best.pth'),
    ]:
        print(f'===== {ds_name} =====', flush=True)
        test_ds = DsCls('./data', tf_test, 'test', **ds_kwargs)
        loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)
        front = ResNet50Front(grid_size=14).to(device)
        bb = torch.load(bb_path, map_location=device, weights_only=False)
        front.load_state_dict({k: v for k, v in bb.items()
                               if not k.startswith(('layer4.', 'fc.',
                                                    'avgpool.', 'spatial_pool.'))},
                              strict=False)
        back = ResNet50Back(n_classes).to(device)
        back.load_state_dict({k: v for k, v in bb.items()
                              if k.startswith(('layer4.', 'fc.', 'avgpool.'))},
                             strict=False)
        front.eval(); back.eval()

        extra = results[ds_key].setdefault('mc_check_extra', {})
        for code_name, (params, points) in CODES.items():
            H = make_regular_ldpc(seed=0, **params)
            dec = MinSumDecoder(H, max_iter=50, method='sumprod')
            for p in points:
                key = f'{code_name}_p{p}'
                if key in extra:
                    continue
                acc = mc_spot_check(front, back, loader, H, dec, p, 50,
                                    N_IMAGES)
                ana = results[ds_key]['sweeps'][f'{code_name}_q50'].get(str(p))
                extra[key] = dict(mc=round(acc, 2), analytic=ana)
                with open('eval/seed_results/jpeg_ldpc_results.json', 'w') as f:
                    json.dump(results, f, indent=1)
                print(f'  {key}: MC={acc:.2f}%  analytic={ana}%', flush=True)

        del front, back, loader
        torch.cuda.empty_cache()
    print('Done.', flush=True)


if __name__ == '__main__':
    main()
