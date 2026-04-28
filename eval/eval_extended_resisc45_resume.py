#!/usr/bin/env python3
"""Resume extended ρ×BER evaluation from where it stopped.

Loads eval/extended_ber_rho_results.json and fills in any missing (rho, BER) pairs.
Supports extending BER range beyond what was previously computed.
Output: eval/extended_ber_rho_results.json
"""

import os, sys, json, random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from train_aid_v2 import ResNet50Front, ResNet50Back, BSC_Channel, LearnedBlockMask, sample_noise
from train_aid_v5 import EncoderV5, DecoderV5, SpikeFunction_Learnable, LIFNeuron, BNTT, MPBN
from noise_aware_scorer import NoiseAwareScorer
from run_final_pipeline import AIDDataset5050, RESISC45Dataset, SpikeAdaptSC_v5c_NA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_STEPS = 8


def evaluate(model, back, front, test_loader, ber=0.0, target_rate=None):
    model.eval(); back.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            if target_rate is not None:
                Fp, _ = model(feat, noise_param=ber, target_rate_override=target_rate)
            else:
                Fp, _ = model(feat, noise_param=ber)
            correct += back(Fp).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total


def main():
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    print(f"Device: {device}")

    # Load existing results
    results_path = 'eval/extended_ber_rho_results.json'
    if os.path.exists(results_path):
        results = json.load(open(results_path))
    else:
        results = {'aid': {}, 'resisc45': {}}

    rho_values = [0.10, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 1.0]
    ber_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])

    configs = [
        ('aid', 'AID', 30, AIDDataset5050, "./data", 'test',
         "./snapshots_aid_5050_seed42/backbone_best.pth",
         "./snapshots_aid_v5cna_seed42/v5cna_best_95.42.pth",
         dict(seed=42)),
        ('resisc45', 'RESISC45', 45, RESISC45Dataset, "./data", 'test',
         "./snapshots_resisc45_5050_seed42/backbone_best.pth",
         "./snapshots_resisc45_v5cna_seed42/v5cna_best_92.01.pth",
         dict(train_ratio=0.20, seed=42)),
    ]

    for ds_key, ds_name, n_classes, DsCls, ds_root, ds_split, bb_path, ck_path, ds_kwargs in configs:
        print(f"\n{'='*60}\n{ds_name} — resuming\n{'='*60}")

        if ds_key == 'aid':
            test_ds = DsCls(ds_root, tf_test, ds_split, **ds_kwargs)
        else:
            test_ds = DsCls(ds_root, tf_test, ds_split, **ds_kwargs)
        test_loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)

        front = ResNet50Front(grid_size=14).to(device)
        bb = torch.load(bb_path, map_location=device, weights_only=False)
        front.load_state_dict({k: v for k, v in bb.items()
                               if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))},
                              strict=False)
        front.eval()
        for p in front.parameters(): p.requires_grad = False

        back = ResNet50Back(n_classes).to(device)
        model = SpikeAdaptSC_v5c_NA(C_in=1024, C1=256, C2=36, T=T_STEPS,
                                     target_rate=0.75, grid_size=14).to(device)
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        back.load_state_dict(ck['back'])

        if ds_key not in results:
            results[ds_key] = {}

        for rho in rho_values:
            rho_key = f"{rho:.3f}" if rho != 1.0 else "1.000"
            if rho_key not in results[ds_key]:
                results[ds_key][rho_key] = {}

            for ber in ber_values:
                ber_key = str(ber)
                if ber_key in results[ds_key][rho_key]:
                    print(f"  [{ds_name}] rho={rho:.3f} BER={ber} -> {results[ds_key][rho_key][ber_key]:.2f}% (cached)")
                    continue

                acc = evaluate(model, back, front, test_loader, ber=ber, target_rate=rho)
                results[ds_key][rho_key][ber_key] = round(acc, 4)
                lab = "Clean" if ber == 0 else f"BER={ber}"
                print(f"  [{ds_name}] rho={rho:.3f} {lab}: {acc:.2f}%")

            # Save incrementally
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)

        del model, back, front
        torch.cuda.empty_cache()

    print(f"\nSaved to {results_path}")


if __name__ == '__main__':
    main()
