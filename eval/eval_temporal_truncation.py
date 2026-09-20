#!/usr/bin/env python3
"""Temporal truncation (anytime transmission): SNN-specific rate control.

The spiking encoder produces T=8 spike tensors sequentially; the payload is
rho * T' * C2 * H * W bits when only the first T' timesteps are transmitted.
Unlike spatial masking (rho), this axis exists ONLY for spiking codes: a
single-pass CNN-1bit cannot be truncated. The DecoderV5 converter already
zero-pads missing timesteps, so no retraining is needed.

True early-exit semantics: the encoder runs only T' steps (energy also
scales), the scorer sees the time-average of the first T' steps, and the
decoder receives the truncated sequence.

Sweep: T' in {1..8} x BER in {0, 0.15, 0.30} at rho = 0.75 and rho = 1.0,
seed-42 checkpoints, both datasets.

Output: eval/seed_results/temporal_truncation.json

Usage:
    python eval/eval_temporal_truncation.py
"""

import os, sys, json, random
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train_aid_v2 import ResNet50Front, ResNet50Back
from run_final_pipeline import AIDDataset5050, RESISC45Dataset, SpikeAdaptSC_v5c_NA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_FULL = 8
T_PRIMES = [1, 2, 3, 4, 5, 6, 8]
BERS = [0.0, 0.15, 0.30]
RHOS = [0.75, 1.0]


@torch.no_grad()
def evaluate_truncated(model, back, front, loader, ber, rho, t_prime):
    """Manual pipeline with early-exit encoding at T' timesteps."""
    model.eval(); back.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        feat = front(imgs)

        # encode only T' steps
        all_S2, m1, m2 = [], None, None
        for t in range(t_prime):
            _, s2, m1, m2 = model.encoder(feat, m1, m2, t=t)
            all_S2.append(s2)

        # scorer on the truncated time-average
        importance = model.scorer(all_S2, ber).squeeze(1)

        old = model.block_mask.target_rate
        model.block_mask.target_rate = rho
        mask, _ = model.block_mask(importance, training=False)
        model.block_mask.target_rate = old

        recv = [model.channel(all_S2[t] * mask, ber) for t in range(t_prime)]
        Fp = model.decoder(recv, mask)
        correct += back(Fp).argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return 100. * correct / total


def main():
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    print(f'Device: {device}', flush=True)
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                         T.Normalize((.485, .456, .406), (.229, .224, .225))])

    results = {}
    for ds_name, ds_key, n_classes, DsCls, ds_kwargs, bb_path, ck_path in [
        ('AID', 'aid', 30, AIDDataset5050, dict(seed=42),
         './snapshots_aid_5050_seed42/backbone_best.pth',
         './snapshots_aid_v5cna_seed42/v5cna_best_95.42.pth'),
        ('RESISC45', 'resisc45', 45, RESISC45Dataset,
         dict(train_ratio=0.20, seed=42),
         './snapshots_resisc45_5050_seed42/backbone_best.pth',
         './snapshots_resisc45_v5cna_seed42/v5cna_best_92.01.pth'),
    ]:
        print(f'\n===== {ds_name} =====', flush=True)
        test_ds = DsCls('./data', tf_test, 'test', **ds_kwargs)
        loader = DataLoader(test_ds, 64, False, num_workers=4, pin_memory=True)

        front = ResNet50Front(grid_size=14).to(device)
        bb = torch.load(bb_path, map_location=device, weights_only=False)
        front.load_state_dict({k: v for k, v in bb.items()
                               if not k.startswith(('layer4.', 'fc.',
                                                    'avgpool.', 'spatial_pool.'))},
                              strict=False)
        front.eval()
        for p in front.parameters():
            p.requires_grad = False

        back = ResNet50Back(n_classes).to(device)
        model = SpikeAdaptSC_v5c_NA(C_in=1024, C1=256, C2=36, T=T_FULL,
                                    target_rate=0.75, grid_size=14).to(device)
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])

        results[ds_key] = {}
        tp_hdr = "T'"
        print(f'{"rho":>5} {tp_hdr:>3} ' + ' '.join(f'BER={b:>4}' for b in BERS),
              flush=True)
        for rho in RHOS:
            results[ds_key][str(rho)] = {}
            for tp in T_PRIMES:
                row = {}
                for ber in BERS:
                    acc = evaluate_truncated(model, back, front, loader,
                                             ber, rho, tp)
                    row[str(ber)] = round(acc, 2)
                results[ds_key][str(rho)][str(tp)] = row
                payload = rho * tp * 36 * 14 * 14
                print(f'{rho:>5} {tp:>3} ' +
                      ' '.join(f'{row[str(b)]:>8.2f}' for b in BERS) +
                      f'   ({payload/1000:.1f} kbit)', flush=True)

        del model, back, front, loader
        torch.cuda.empty_cache()

    out = 'eval/seed_results/temporal_truncation.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=1)
    print(f'\nSaved: {out}', flush=True)


if __name__ == '__main__':
    main()
