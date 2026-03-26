#!/usr/bin/env python3
"""Non-spiking Conv Autoencoder baseline for fair comparison with SpikeAdapt-SC.

Same bottleneck shape (C_in → C2=36 → C_in), same BSC channel,
but NO SNN components: no LIF neurons, no scorer, no adaptive mask.

This establishes whether SNN-specific features (spike coding, noise-aware
scoring, adaptive masking) provide robustness benefits over vanilla compression.

Usage:
  python train_dota_ae_baseline.py --phase 2   # Train
  python train_dota_ae_baseline.py --phase 3   # mAP vs BER sweep
"""

import os, sys, json, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.snn_modules import BSC_Channel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASELINE = 'runs/obb/runs/dota/yolo26n_obb_baseline/weights/best.pt'
AE_SAVE = 'runs/yolo26_ae_baseline.pth'
HOOK_LAYERS = [4, 6, 10]
CHANNEL_SIZES = [128, 128, 256]


class ConvAE_Level(nn.Module):
    """Non-spiking Conv autoencoder for one FPN level.
    
    Encoder: Conv(C_in→C2) + BN + SiLU → binarize (hard sigmoid to [0,1])
    Channel: BSC (bit flips)
    Decoder: Conv(C2→C1→C_in) + BN + SiLU
    
    No LIF neurons, no scorer, no adaptive mask.
    Uses fixed rate=1.0 (transmit everything) or fixed random mask.
    """
    def __init__(self, C_in, C2=36):
        super().__init__()
        C1 = min(128, C_in)
        # Encoder: compress to C2 channels
        self.encoder = nn.Sequential(
            nn.Conv2d(C_in, C1, 3, 1, 1),
            nn.BatchNorm2d(C1),
            nn.SiLU(inplace=True),
            nn.Conv2d(C1, C2, 3, 1, 1),
            nn.BatchNorm2d(C2),
        )
        # Binarizer: hard sigmoid + STE for binary transmission
        self.bsc = BSC_Channel()
        
        # Decoder: reconstruct from C2
        self.decoder = nn.Sequential(
            nn.Conv2d(C2, C1, 3, 1, 1),
            nn.BatchNorm2d(C1),
            nn.SiLU(inplace=True),
            nn.Conv2d(C1, C_in, 3, 1, 1),
            nn.BatchNorm2d(C_in),
            nn.SiLU(inplace=True),
        )
    
    def forward(self, feat, ber=0.0):
        # Encode
        z = self.encoder(feat)
        # Binarize with STE (sigmoid → round)
        z_bin = torch.sigmoid(z)
        if self.training:
            z_hard = (z_bin > 0.5).float()
            z_bin = z_bin + (z_hard - z_bin).detach()  # STE
        else:
            z_bin = (z_bin > 0.5).float()
        # Transmit through BSC (same channel as SpikeAdapt-SC)
        z_recv = self.bsc(z_bin, ber)
        # Decode
        recon = self.decoder(z_recv)
        tx_rate = z_bin.mean().item()
        return recon, {'tx_rate': tx_rate}


class ConvAE_Multi(nn.Module):
    """Multi-level Conv AE for YOLO FPN."""
    def __init__(self, channel_sizes, C2=36):
        super().__init__()
        self.levels = nn.ModuleList([
            ConvAE_Level(c, C2) for c in channel_sizes
        ])
    
    def forward(self, features, ber=0.0):
        recons, infos = [], []
        for feat, level in zip(features, self.levels):
            r, i = level(feat, ber=ber)
            recons.append(r)
            infos.append(i)
        return recons, infos


class AE_Hook(nn.Module):
    """Hook to inject Conv AE into YOLO backbone."""
    def __init__(self, original_layer, ae_level, ber=0.0):
        super().__init__()
        self.original_layer = original_layer
        self.ae = ae_level
        self.ber = ber
        self.f = getattr(original_layer, 'f', -1)
        self.i = getattr(original_layer, 'i', 0)
        self.type = getattr(original_layer, 'type', type(original_layer).__name__)
        self.np = getattr(original_layer, 'np', 0)
        self.last_info = {}
    
    def forward(self, x):
        out = self.original_layer(x)
        recon, info = self.ae(out, ber=self.ber)
        self.last_info = info
        return recon


def get_backbone_features(yolo_model, imgs):
    """Extract backbone features without hooks."""
    save = {}
    x = imgs
    with torch.no_grad():
        for i, layer in enumerate(yolo_model.model[:max(HOOK_LAYERS)+1]):
            if hasattr(layer, 'f') and isinstance(layer.f, list):
                x_in = [x if j == -1 else save[j] for j in layer.f]
                x = layer(x_in)
            elif hasattr(layer, 'f') and isinstance(layer.f, int) and layer.f != -1:
                x = layer(save[layer.f])
            else:
                x = layer(x)
            save[i] = x
    return [save[l] for l in HOOK_LAYERS]


# ===================================================================
# Phase 2: Train
# ===================================================================
def phase2_train(args):
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data import YOLODataset
    from torch.utils.data import DataLoader

    print("=" * 70)
    print("  Non-Spiking Conv AE Baseline (same bottleneck, same BSC)")
    print("=" * 70)

    yolo_orig = YOLO(BASELINE)
    yolo_orig.model.to(device).eval()

    ae = ConvAE_Multi(CHANNEL_SIZES, C2=args.c2).to(device)
    n_params = sum(p.numel() for p in ae.parameters())
    print(f"  AE params: {n_params:,}")
    print(f"  Config: C2={args.c2}, NO SNN (no LIF, no scorer, no mask)")

    optimizer = torch.optim.AdamW(ae.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    data_dict = check_det_dataset('DOTAv1.yaml')
    train_dataset = YOLODataset(
        img_path=data_dict['train'], imgsz=640,
        augment=True, data=data_dict, task='obb',
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=YOLODataset.collate_fn,
    )

    best_loss = float('inf')
    level_weights = [0.2, 0.3, 0.5]

    for epoch in range(args.epochs):
        ae.train()
        epoch_mse, n_batches = 0, 0
        ber = np.random.uniform(0, args.ber_max)

        for batch_data in train_loader:
            imgs = batch_data['img'].to(device).float() / 255.0
            clean = get_backbone_features(yolo_orig.model, imgs)
            recons, infos = ae([f.detach() for f in clean], ber=ber)

            mse_loss = sum(
                w * F.mse_loss(r, c.detach())
                for w, r, c in zip(level_weights, recons, clean)
            )

            optimizer.zero_grad()
            mse_loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            optimizer.step()

            epoch_mse += mse_loss.item()
            n_batches += 1
            if n_batches >= 200:
                break

        scheduler.step()
        avg_mse = epoch_mse / max(n_batches, 1)
        print(f"  Epoch {epoch+1:3d}/{args.epochs}  MSE={avg_mse:.6f}  BER={ber:.2f}")

        if avg_mse < best_loss:
            best_loss = avg_mse
            torch.save({
                'ae_state': ae.state_dict(),
                'epoch': epoch + 1,
                'loss': best_loss,
                'config': {'C2': args.c2, 'channel_sizes': CHANNEL_SIZES},
            }, AE_SAVE)

    print(f"\n✅ Best MSE: {best_loss:.6f}, saved: {AE_SAVE}")


# ===================================================================
# Phase 3: Full BER sweep
# ===================================================================
def phase3_eval(args):
    from ultralytics import YOLO

    print("=" * 70)
    print("  mAP vs BER: YOLO26 + Conv AE Baseline")
    print("=" * 70)

    ck = torch.load(AE_SAVE, map_location=device, weights_only=False)
    cfg = ck['config']
    ae = ConvAE_Multi(cfg['channel_sizes'], C2=cfg['C2']).to(device)
    ae.load_state_dict(ck['ae_state'])
    ae.eval()
    print(f"  Loaded epoch {ck['epoch']}, MSE={ck['loss']:.6f}")

    # Baseline
    yolo_base = YOLO(BASELINE)
    base_res = yolo_base.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0,
                              verbose=True, plots=False)
    base_map = float(base_res.results_dict.get('metrics/mAP50(B)', 0))
    print(f"  Baseline: mAP@50={base_map:.4f}")

    bers = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    results = {'baseline': base_map, 'ae': {}}

    for ber in bers:
        print(f"\n  BER={ber:.2f}...")
        yolo = YOLO(BASELINE)
        yolo.model.fuse = lambda *a, **kw: yolo.model
        for idx, (lid, lev) in enumerate(zip(HOOK_LAYERS, ae.levels)):
            yolo.model.model[lid] = AE_Hook(
                yolo.model.model[lid], lev, ber=ber
            ).to(device)
        res = yolo.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0,
                       verbose=False, plots=False)
        m50 = float(res.results_dict.get('metrics/mAP50(B)', 0))
        d = m50 - base_map
        results['ae'][str(ber)] = {'mAP50': m50}
        print(f"    mAP@50={m50:.4f} ({d:+.4f})")

    print("\n" + "=" * 70)
    print(f"  {'BER':>6s}  {'mAP@50':>8s}  {'Δ':>8s}")
    print(f"  {'none':>6s}  {base_map:8.4f}  {'—':>8s}")
    for ber in bers:
        r = results['ae'][str(ber)]
        d = r['mAP50'] - base_map
        print(f"  {ber:6.2f}  {r['mAP50']:8.4f}  {d:+8.4f}")

    with open('eval/dota_ae_baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved: eval/dota_ae_baseline_results.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, default=2, choices=[2, 3])
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--c2', type=int, default=36)
    parser.add_argument('--ber-max', type=float, default=0.30)
    args = parser.parse_args()

    {2: phase2_train, 3: phase3_eval}[args.phase](args)
