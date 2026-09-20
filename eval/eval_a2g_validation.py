#!/usr/bin/env python3
"""Validate the BSC abstraction against the physical Rician A2G channel.

Hard-decision BPSK over Rician fading with per-symbol perfect CSI is
information-theoretically a BSC with BER(SNR, K). This script verifies the
equivalence INSIDE the full SpikeAdapt-SC pipeline: for each paper BER anchor,
it evaluates (a) BSC at BER p, and (b) Rician fading at the SNR for which
BER(SNR, K) = p, with the noise-aware scorer fed the equivalent BER both ways
(deployment: receiver estimates BER from pilots and feeds it back).

Expected: |acc_BSC - acc_Rician| < ~0.3 pp (Monte Carlo noise only).

Output: eval/seed_results/a2g_validation.json

Usage:
    python eval/eval_a2g_validation.py
"""

import os, sys, json, random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train_aid_v2 import ResNet50Front, ResNet50Back
from run_final_pipeline import AIDDataset5050, RESISC45Dataset, SpikeAdaptSC_v5c_NA
from models.snn_modules import Rician_Channel, BSC_Channel
from eval.channel_a2g import snr_for_ber

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_STEPS = 8
RHO = 0.75
BER_ANCHORS = [0.05, 0.15, 0.30]
K_FACTORS_DB = [5.0, 10.0]


class PhysicalChannelAdapter(nn.Module):
    """Applies a fading channel at a fixed SNR while the model's noise_param
    (consumed by the scorer as BER) is ignored by the channel itself."""

    def __init__(self, channel, snr_db):
        super().__init__()
        self.channel = channel
        self.snr_db = snr_db

    def forward(self, x, _noise_param):
        return self.channel(x, self.snr_db)


@torch.no_grad()
def evaluate(model, back, front, loader, ber_for_scorer):
    model.eval(); back.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        Fp, _ = model(front(imgs), noise_param=ber_for_scorer,
                      target_rate_override=RHO)
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
                               if not k.startswith(('layer4.', 'fc.', 'avgpool.',
                                                    'spatial_pool.'))}, strict=False)
        front.eval()
        for p in front.parameters():
            p.requires_grad = False

        back = ResNet50Back(n_classes).to(device)
        model = SpikeAdaptSC_v5c_NA(C_in=1024, C1=256, C2=36, T=T_STEPS,
                                    target_rate=RHO, grid_size=14).to(device)
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])

        bsc = BSC_Channel()
        results[ds_key] = {}
        print(f'{"BER":>5} {"K(dB)":>6} {"SNR(dB)":>8} {"BSC":>7} '
              f'{"Rician":>7} {"Delta":>7}', flush=True)
        for ber in BER_ANCHORS:
            model.channel = bsc
            acc_bsc = evaluate(model, back, front, loader, ber)
            row = {'bsc': round(acc_bsc, 2), 'rician': {}}
            for kdb in K_FACTORS_DB:
                snr = snr_for_ber(ber, kdb)
                model.channel = PhysicalChannelAdapter(
                    Rician_Channel(k_factor_db=kdb), snr).to(device)
                acc_ric = evaluate(model, back, front, loader, ber)
                row['rician'][f'K{kdb:.0f}dB'] = {
                    'snr_db': round(snr, 2), 'acc': round(acc_ric, 2)}
                print(f'{ber:>5.2f} {kdb:>6.0f} {snr:>8.2f} {acc_bsc:>6.2f}% '
                      f'{acc_ric:>6.2f}% {acc_ric-acc_bsc:>+6.2f}', flush=True)
            results[ds_key][str(ber)] = row

        del model, back, front, loader
        torch.cuda.empty_cache()

    out = 'eval/seed_results/a2g_validation.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=1)
    print(f'\nSaved: {out}', flush=True)


if __name__ == '__main__':
    main()
