#!/usr/bin/env python3
"""Extended ρ×BER evaluation for V5C-NA (SpikeAdapt-SC) on AID and RESISC45.

Extends BER range to 0.40 with finer BER granularity for the α plot:
    α(ρ, BER) = (A(ρ) - A(ρ=1)) / A(ρ=1)

Output: eval/extended_ber_rho_results.json

Usage:
    python eval/eval_extended_ber_rho.py
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
    """Evaluate model at given BER and optional rate override."""
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


def run_extended_sweep(dataset_name, n_classes, model, back, front, test_loader):
    """Run extended ρ×BER sweep."""
    rho_values = [0.10, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 1.0]
    ber_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    results = {}
    for rho in rho_values:
        results[str(rho)] = {}
        for ber in ber_values:
            acc = evaluate(model, back, front, test_loader, ber=ber, target_rate=rho)
            results[str(rho)][str(ber)] = round(acc, 4)
            lab = "Clean" if ber == 0 else f"BER={ber}"
            print(f"  [{dataset_name}] ρ={rho:.3f} {lab}: {acc:.2f}%")

    return results


def main():
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    print(f"Device: {device}")

    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])

    all_results = {}

    # ========== AID ==========
    print(f"\n{'='*60}\nAID (50/50)\n{'='*60}")
    test_ds = AIDDataset5050("./data", tf_test, 'test', seed=42)
    test_loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)

    front = ResNet50Front(grid_size=14).to(device)
    bb = torch.load("./snapshots_aid_5050_seed42/backbone_best.pth",
                     map_location=device, weights_only=False)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))},
                          strict=False)
    front.eval()
    for p in front.parameters(): p.requires_grad = False

    back = ResNet50Back(30).to(device)
    model = SpikeAdaptSC_v5c_NA(C_in=1024, C1=256, C2=36, T=T_STEPS,
                                 target_rate=0.75, grid_size=14).to(device)
    ck = torch.load("./snapshots_aid_v5cna_seed42/v5cna_best_95.42.pth",
                     map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])
    back.load_state_dict(ck['back'])

    all_results['aid'] = run_extended_sweep('AID', 30, model, back, front, test_loader)
    del model, back, front; torch.cuda.empty_cache()

    # ========== RESISC45 ==========
    print(f"\n{'='*60}\nRESISC45 (20/80)\n{'='*60}")
    test_ds = RESISC45Dataset("./data", tf_test, 'test', train_ratio=0.20, seed=42)
    test_loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)

    front = ResNet50Front(grid_size=14).to(device)
    bb = torch.load("./snapshots_resisc45_5050_seed42/backbone_best.pth",
                     map_location=device, weights_only=False)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))},
                          strict=False)
    front.eval()
    for p in front.parameters(): p.requires_grad = False

    back = ResNet50Back(45).to(device)
    model = SpikeAdaptSC_v5c_NA(C_in=1024, C1=256, C2=36, T=T_STEPS,
                                 target_rate=0.75, grid_size=14).to(device)
    ck = torch.load("./snapshots_resisc45_v5cna_seed42/v5cna_best_92.01.pth",
                     map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])
    back.load_state_dict(ck['back'])

    all_results['resisc45'] = run_extended_sweep('RESISC45', 45, model, back, front, test_loader)

    # Save
    out = 'eval/extended_ber_rho_results.json'
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == '__main__':
    main()
