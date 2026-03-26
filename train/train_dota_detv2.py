#!/usr/bin/env python3
"""SpikeAdapt-SC Detection V2: Faithful encoder + detection-adapted decoder.

Hybrid architecture:
  ✅ FAITHFUL: LIF+MPBN Encoder, NoiseAwareScorer, LearnedBlockMask, BSC Channel
  🔧 ADAPTED: Continuous Conv decoder (replaces IHF+converter)

The decoder change is justified because:
  - IHF neurons produce discrete spikes, destroying spatial precision
  - Detection needs continuous features for bounding box regression
  - The encoder→scorer→mask→BSC pipeline IS the SpikeAdapt-SC novelty
  - The decoder is just signal reconstruction — Conv is standard

Usage:
  python train_dota_detv2.py --phase 2   # Train
  python train_dota_detv2.py --phase 3   # mAP vs BER
  python train_dota_detv2.py --phase 4   # Visualization
"""

import os, sys, json, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.spikeadapt_sc import Encoder, LearnedBlockMask
from models.noise_aware_scorer import NoiseAwareScorer
from models.snn_modules import get_channel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASELINE = 'runs/obb/runs/dota/yolo26n_obb_baseline/weights/best.pt'
SNN_SAVE = 'runs/yolo26_snn_detv2.pth'
HOOK_LAYERS = [4, 6, 10]     # YOLO26: P3, P4, P5
CHANNEL_SIZES = [128, 128, 256]


# ===================================================================
# Detection-Adapted Decoder (continuous output)
# ===================================================================
class DetectionDecoder(nn.Module):
    """Continuous Conv decoder for detection feature reconstruction.
    
    Takes T timesteps of received (masked+noised) spikes,
    averages them temporally, then reconstructs via Conv layers.
    
    Unlike IHF decoder: outputs continuous features, preserving
    the spatial precision detection needs for OBB regression.
    """
    def __init__(self, C_out, C1, C2, T=8):
        super().__init__()
        self.T = T
        # Temporal aggregation: average T spike frames + learnable weighting
        self.temporal_weight = nn.Parameter(torch.ones(T) / T)
        
        # 2-layer Conv decoder: C2 → C1 → C_out
        self.decoder = nn.Sequential(
            nn.Conv2d(C2, C1, 3, 1, 1),
            nn.BatchNorm2d(C1),
            nn.SiLU(inplace=True),
            nn.Conv2d(C1, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            nn.SiLU(inplace=True),
        )
    
    def forward(self, recv_all, mask):
        """Decode received spikes to continuous features.
        
        Args:
            recv_all: list of T tensors [B, C2, H, W] (received spikes)
            mask: [B, 1, H, W] spatial mask
        
        Returns:
            Fp: [B, C_out, H, W] reconstructed features
        """
        # Apply mask and weighted temporal average
        w = torch.softmax(self.temporal_weight, dim=0)
        avg = sum(w[t] * recv_all[t] * mask for t in range(self.T))
        return self.decoder(avg)


# ===================================================================
# Hybrid SpikeAdaptSC (faithful encoder + detection decoder)
# ===================================================================
class SpikeAdaptSC_DetV2(nn.Module):
    """SpikeAdapt-SC with detection-adapted decoder.
    
    Faithful components: Encoder (LIF+MPBN), NoiseAwareScorer, LearnedBlockMask, BSC
    Adapted component: DetectionDecoder (continuous Conv output)
    """
    def __init__(self, C_in, C1=128, C2=36, T=8,
                 target_rate=0.75, channel_type='bsc'):
        super().__init__()
        self.T = T
        self.encoder = Encoder(C_in, min(C1, C_in), C2, T)
        self.scorer = NoiseAwareScorer(C_spike=C2, hidden=32)
        self.block_mask = LearnedBlockMask(target_rate, 0.5)
        self.decoder = DetectionDecoder(C_in, min(C1, C_in), C2, T)
        self.channel = get_channel(channel_type)
    
    def forward(self, feat, noise_param=0.0, target_rate_override=None):
        # Encode over T timesteps (FAITHFUL: LIF + MPBN)
        all_S1, all_S2, m1, m2 = [], [], None, None
        for t in range(self.T):
            s1, s2, m1, m2 = self.encoder(feat, m1, m2, t=t)
            all_S1.append(s1)
            all_S2.append(s2)
        
        # Score importance (FAITHFUL: BER-conditioned channel gates)
        importance = self.scorer(all_S2, ber=noise_param if isinstance(noise_param, float) else 0.0)
        
        # Generate mask (FAITHFUL: Gumbel-sigmoid / top-k)
        if target_rate_override is not None:
            old = self.block_mask.target_rate
            self.block_mask.target_rate = target_rate_override
            mask, tx = self.block_mask(importance, training=False)
            self.block_mask.target_rate = old
        else:
            mask, tx = self.block_mask(importance, training=self.training)
        
        # Transmit through BSC (FAITHFUL)
        recv = [self.channel(self.block_mask.apply_mask(all_S2[t], mask), noise_param)
                for t in range(self.T)]
        
        # Decode (ADAPTED: continuous Conv, not IHF)
        Fp = self.decoder(recv, mask)
        
        return Fp, {
            'tx_rate': tx.item() if isinstance(tx, torch.Tensor) else tx,
            'mask': mask,
            'importance': importance,
            'all_S2': all_S2,
        }


class SNN_DetV2_Hook(nn.Module):
    """Hook to inject SpikeAdaptSC_DetV2 into YOLO backbone."""
    def __init__(self, original_layer, snn_level, ber=0.0):
        super().__init__()
        self.original_layer = original_layer
        self.snn = snn_level
        self.ber = ber
        self.f = getattr(original_layer, 'f', -1)
        self.i = getattr(original_layer, 'i', 0)
        self.type = getattr(original_layer, 'type', type(original_layer).__name__)
        self.np = getattr(original_layer, 'np', 0)
        self.last_info = {}
    
    def forward(self, x):
        out = self.original_layer(x)
        recon, info = self.snn(out, noise_param=self.ber)
        self.last_info = info
        return recon


# ===================================================================
# Multi-level wrapper
# ===================================================================
class SpikeAdaptSC_DetV2_Multi(nn.Module):
    """Per-level SpikeAdaptSC_DetV2 for YOLO FPN."""
    def __init__(self, channel_sizes, C1=128, C2=36, T=8,
                 target_rate=0.75, channel_type='bsc'):
        super().__init__()
        self.levels = nn.ModuleList([
            SpikeAdaptSC_DetV2(c, C1, C2, T, target_rate, channel_type)
            for c in channel_sizes
        ])
    
    def forward(self, features, ber=0.0, target_rate_override=None):
        recons, infos = [], []
        for feat, level in zip(features, self.levels):
            r, i = level(feat, noise_param=ber, target_rate_override=target_rate_override)
            recons.append(r)
            infos.append(i)
        return recons, infos


def get_backbone_features(yolo_model, imgs):
    """Extract TRUE backbone features (no SNN hooks)."""
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
    print("  SpikeAdapt-SC DetV2: Faithful Encoder + Detection Decoder")
    print("=" * 70)

    yolo_orig = YOLO(BASELINE)
    yolo_orig.model.to(device).eval()

    snn = SpikeAdaptSC_DetV2_Multi(
        CHANNEL_SIZES, C1=128, C2=args.c2, T=args.T,
        target_rate=args.target_rate, channel_type='bsc',
    ).to(device)

    n_params = sum(p.numel() for p in snn.parameters())
    print(f"  SNN params: {n_params:,}")
    print(f"  Config: C2={args.c2}, T={args.T}, target_rate={args.target_rate}")

    optimizer = torch.optim.AdamW(snn.parameters(), lr=args.lr, weight_decay=1e-4)
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
    level_weights = [0.2, 0.3, 0.5]  # P3, P4, P5

    for epoch in range(args.epochs):
        snn.train()
        epoch_mse, epoch_rate, epoch_div = 0, 0, 0
        n_batches = 0

        ber = np.random.uniform(0, args.ber_max)

        for batch_data in train_loader:
            imgs = batch_data['img'].to(device).float() / 255.0
            clean = get_backbone_features(yolo_orig.model, imgs)
            recons, infos = snn([f.detach() for f in clean], ber=ber)

            # MSE loss (per-level weighted)
            mse_loss = sum(
                w * F.mse_loss(r, c.detach())
                for w, r, c in zip(level_weights, recons, clean)
            )
            # Rate loss
            rate_loss = sum(
                (info['tx_rate'] - args.target_rate) ** 2
                for info in infos
            ) / len(infos)
            # Diversity loss
            div_loss = torch.tensor(0.0, device=device)
            for level_snn, info in zip(snn.levels, infos):
                if info.get('all_S2'):
                    div_loss = div_loss + level_snn.scorer.compute_diversity_loss(info['all_S2'])

            loss = mse_loss + 0.1 * rate_loss + 0.02 * div_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(snn.parameters(), 1.0)
            optimizer.step()

            epoch_mse += mse_loss.item()
            epoch_rate += sum(i['tx_rate'] for i in infos) / len(infos)
            epoch_div += div_loss.item()
            n_batches += 1
            if n_batches >= 200:
                break

        scheduler.step()
        avg_mse = epoch_mse / max(n_batches, 1)
        avg_rate = epoch_rate / max(n_batches, 1)

        print(f"  Epoch {epoch+1:3d}/{args.epochs}  MSE={avg_mse:.6f}  "
              f"Rate={avg_rate:.3f}  BER={ber:.2f}")

        if avg_mse < best_loss:
            best_loss = avg_mse
            torch.save({
                'snn_state': snn.state_dict(),
                'epoch': epoch + 1,
                'loss': best_loss,
                'config': {
                    'C2': args.c2, 'T': args.T,
                    'target_rate': args.target_rate,
                    'channel_sizes': CHANNEL_SIZES,
                },
            }, SNN_SAVE)

    print(f"\n✅ Best MSE: {best_loss:.6f}, saved: {SNN_SAVE}")


# ===================================================================
# Phase 3: Full BER sweep
# ===================================================================
def phase3_eval(args):
    from ultralytics import YOLO

    print("=" * 70)
    print("  mAP vs BER: YOLO26 + SpikeAdaptSC DetV2")
    print("=" * 70)

    ck = torch.load(SNN_SAVE, map_location=device, weights_only=False)
    cfg = ck['config']
    snn = SpikeAdaptSC_DetV2_Multi(
        cfg['channel_sizes'], C1=128, C2=cfg['C2'], T=cfg['T'],
        target_rate=cfg['target_rate'], channel_type='bsc',
    ).to(device)
    snn.load_state_dict(ck['snn_state'])
    snn.eval()
    print(f"  Loaded epoch {ck['epoch']}, MSE={ck['loss']:.6f}")

    # Baseline
    yolo_base = YOLO(BASELINE)
    base_res = yolo_base.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0,
                              verbose=True, plots=False)
    base_map = float(base_res.results_dict.get('metrics/mAP50(B)', 0))
    print(f"  Baseline: mAP@50={base_map:.4f}")

    bers = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    results = {'baseline': base_map, 'snn': {}}

    for ber in bers:
        print(f"\n  BER={ber:.2f}...")
        yolo = YOLO(BASELINE)
        yolo.model.fuse = lambda *a, **kw: yolo.model
        for idx, (lid, lev) in enumerate(zip(HOOK_LAYERS, snn.levels)):
            yolo.model.model[lid] = SNN_DetV2_Hook(
                yolo.model.model[lid], lev, ber=ber
            ).to(device)
        res = yolo.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0,
                       verbose=False, plots=False)
        m50 = float(res.results_dict.get('metrics/mAP50(B)', 0))
        tx = np.mean([yolo.model.model[l].last_info.get('tx_rate', 0) for l in HOOK_LAYERS])
        d = m50 - base_map
        results['snn'][str(ber)] = {'mAP50': m50, 'tx_rate': float(tx)}
        print(f"    mAP@50={m50:.4f} ({d:+.4f}), Rate={tx:.3f}")

    print("\n" + "=" * 70)
    print(f"  {'BER':>6s}  {'mAP@50':>8s}  {'Δ':>8s}  {'Rate':>6s}")
    print(f"  {'none':>6s}  {base_map:8.4f}  {'—':>8s}  {'1.00':>6s}")
    for ber in bers:
        r = results['snn'][str(ber)]
        d = r['mAP50'] - base_map
        print(f"  {ber:6.2f}  {r['mAP50']:8.4f}  {d:+8.4f}  {r['tx_rate']:6.3f}")

    with open('eval/dota_detv2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved: eval/dota_detv2_results.json")


# ===================================================================
# Phase 4: 6-image visualization
# ===================================================================
def phase4_viz(args):
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("  Detection Visualization: SpikeAdaptSC DetV2 (6 images)")
    print("=" * 70)

    ck = torch.load(SNN_SAVE, map_location=device, weights_only=False)
    cfg = ck['config']
    snn = SpikeAdaptSC_DetV2_Multi(
        cfg['channel_sizes'], C1=128, C2=cfg['C2'], T=cfg['T'],
        target_rate=cfg['target_rate'], channel_type='bsc',
    ).to(device)
    snn.load_state_dict(ck['snn_state'])
    snn.eval()

    data_dict = check_det_dataset('DOTAv1.yaml')
    val_dir = Path(data_dict['val'])
    all_imgs = sorted(list(val_dir.glob('*.png')) + list(val_dir.glob('*.jpg')))
    n = len(all_imgs)
    indices = [0, n//6, n//3, n//2, 2*n//3, 5*n//6]
    test_imgs = [str(all_imgs[i]) for i in indices if i < n]

    ber_levels = [0.0, 0.10, 0.20, 0.30]
    os.makedirs('paper/figures', exist_ok=True)

    for img_idx, img_path in enumerate(test_imgs):
        n_cols = len(ber_levels) + 1
        fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4.5))
        name = Path(img_path).stem

        # Baseline
        yolo = YOLO(BASELINE)
        r = yolo.predict(img_path, imgsz=640, device=0, verbose=False, conf=0.25)
        ax = axes[0]
        ax.imshow(r[0].plot()[:, :, ::-1])
        nb = len(r[0].obb) if r[0].obb is not None else 0
        ax.set_title(f'Baseline\n({nb} det)', fontsize=10, fontweight='bold')
        ax.axis('off')

        for bi, ber in enumerate(ber_levels):
            yolo = YOLO(BASELINE)
            yolo.model.fuse = lambda *a, **kw: yolo.model
            for idx, (lid, lev) in enumerate(zip(HOOK_LAYERS, snn.levels)):
                yolo.model.model[lid] = SNN_DetV2_Hook(
                    yolo.model.model[lid], lev, ber=ber
                ).to(device)
            r = yolo.predict(img_path, imgsz=640, device=0, verbose=False, conf=0.25)
            ax = axes[bi + 1]
            ax.imshow(r[0].plot()[:, :, ::-1])
            nb = len(r[0].obb) if r[0].obb is not None else 0
            ax.set_title(f'SpikeAdapt-SC\nBER={ber:.2f} ({nb} det)',
                        fontsize=10, fontweight='bold')
            ax.axis('off')

        plt.suptitle(f'YOLO26 + SpikeAdapt-SC DetV2 on DOTA ({name})',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        out = f'paper/figures/dota_detv2_viz_{img_idx+1}.png'
        plt.savefig(out, dpi=200, bbox_inches='tight')
        plt.savefig(out.replace('.png', '.pdf'), dpi=200, bbox_inches='tight')
        print(f"  Saved: {out}")
        plt.close()

    print(f"\n✅ {len(test_imgs)} visualizations saved!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, default=2, choices=[2, 3, 4])
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--c2', type=int, default=36)
    parser.add_argument('--T', type=int, default=8)
    parser.add_argument('--target-rate', type=float, default=0.75)
    parser.add_argument('--ber-max', type=float, default=0.30)
    args = parser.parse_args()

    {2: phase2_train, 3: phase3_eval, 4: phase4_viz}[args.phase](args)
