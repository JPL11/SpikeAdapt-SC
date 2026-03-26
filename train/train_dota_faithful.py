#!/usr/bin/env python3
"""Faithful SpikeAdapt-SC × YOLO26-OBB on DOTA.

Uses the EXACT same SNN components as the classification pipeline:
  - Encoder: 2-layer Conv + MPBN + LIF neurons
  - NoiseAwareScorer: BER-conditioned channel gates
  - LearnedBlockMask: Gumbel-sigmoid / top-k
  - BSC Channel
  - Decoder: IHF neurons + spike-to-feature converter

Phases:
  1. YOLO26 baseline (already done — mAP@50=42.8%)
  2. Train per-level SpikeAdaptSC with feature distillation
  3. Full val-set BER sweep (458 images, 7 BER levels)
  4. Visualization (6+ diverse images)

Usage:
  python train_dota_faithful.py --phase 2   # Train SNN
  python train_dota_faithful.py --phase 3   # mAP vs BER
  python train_dota_faithful.py --phase 4   # Visualization
"""

import os, sys, json, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.snn_detection import SpikeAdaptSC_Detection, SNN_Detection_Hook

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASELINE = 'runs/obb/runs/dota/yolo26n_obb_baseline/weights/best.pt'
SNN_SAVE = 'runs/yolo26_snn_faithful.pth'
HOOK_LAYERS = [4, 6, 10]  # YOLO26: P3, P4, P5
CHANNEL_SIZES = [128, 128, 256]


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
# Phase 2: Train SpikeAdaptSC for Detection
# ===================================================================
def phase2_train(args):
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data import YOLODataset
    from torch.utils.data import DataLoader

    print("=" * 70)
    print("  FAITHFUL SpikeAdapt-SC Training on DOTA")
    print("  (Encoder + NoiseAwareScorer + LearnedBlockMask + BSC + Decoder)")
    print("=" * 70)

    yolo_orig = YOLO(BASELINE)
    yolo_orig.model.to(device)
    yolo_orig.model.eval()

    # Create per-level SpikeAdaptSC (faithful architecture)
    snn = SpikeAdaptSC_Detection(
        channel_sizes=CHANNEL_SIZES,
        C1=128,          # Intermediate encoder channels (adapt to detection scale)
        C2=args.c2,      # Bottleneck channels (36 = paper default)
        T=args.T,        # Timesteps (8 = paper default)
        target_rate=args.target_rate,
        channel_type='bsc',
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

    for epoch in range(args.epochs):
        snn.train()
        epoch_mse = 0
        epoch_rate = 0
        epoch_div = 0
        n_batches = 0

        ber = np.random.uniform(0, args.ber_max)

        for batch_data in train_loader:
            imgs = batch_data['img'].to(device).float() / 255.0

            # Get TRUE backbone features
            clean_features = get_backbone_features(yolo_orig.model, imgs)

            # SpikeAdaptSC: encode → score → mask → BSC → decode
            recon_features, all_info = snn(
                [f.detach() for f in clean_features], ber=ber
            )

            # Loss 1: Feature distillation MSE (per-level weighted)
            level_weights = [0.2, 0.3, 0.5]  # P3, P4, P5
            mse_loss = sum(
                w * F.mse_loss(r, c.detach())
                for w, r, c in zip(level_weights, recon_features, clean_features)
            )

            # Loss 2: Rate loss — push tx_rate toward target
            rate_losses = []
            for info in all_info:
                rate_losses.append((info['tx_rate'] - args.target_rate) ** 2)
            rate_loss = sum(rate_losses) / len(rate_losses)

            # Loss 3: Scorer diversity — masks must differ at different BER
            div_loss = torch.tensor(0.0, device=device)
            for level_snn, feat in zip(snn.levels, clean_features):
                # Encode again for diversity loss (using cached spikes from forward)
                all_S2 = all_info[0].get('all_S2', None)
                if all_S2 is not None:
                    div_loss = div_loss + level_snn.scorer.compute_diversity_loss(all_S2)

            # Combined loss
            loss = mse_loss + 0.1 * rate_loss + 0.05 * div_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(snn.parameters(), 1.0)
            optimizer.step()

            epoch_mse += mse_loss.item()
            epoch_rate += sum(info['tx_rate'] for info in all_info) / len(all_info)
            epoch_div += div_loss.item()
            n_batches += 1
            if n_batches >= 200:
                break

        scheduler.step()
        avg_mse = epoch_mse / max(n_batches, 1)
        avg_rate = epoch_rate / max(n_batches, 1)
        avg_div = epoch_div / max(n_batches, 1)

        print(f"  Epoch {epoch+1:3d}/{args.epochs}  "
              f"MSE={avg_mse:.6f}  Rate={avg_rate:.3f}  "
              f"Div={avg_div:.4f}  BER={ber:.2f}")

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

    print(f"\n✅ Best MSE: {best_loss:.6f}")
    print(f"  Saved: {SNN_SAVE}")


# ===================================================================
# Phase 3: Full val-set BER sweep
# ===================================================================
def phase3_eval(args):
    from ultralytics import YOLO

    print("=" * 70)
    print("  FAITHFUL mAP vs BER: YOLO26 + SpikeAdaptSC (full val set)")
    print("=" * 70)

    ck = torch.load(SNN_SAVE, map_location=device, weights_only=False)
    cfg = ck['config']
    snn = SpikeAdaptSC_Detection(
        channel_sizes=cfg['channel_sizes'],
        C1=128, C2=cfg['C2'], T=cfg['T'],
        target_rate=cfg['target_rate'],
        channel_type='bsc',
    ).to(device)
    snn.load_state_dict(ck['snn_state'])
    snn.eval()
    print(f"  SNN loaded (epoch {ck['epoch']}, MSE={ck['loss']:.6f})")
    print(f"  Config: C2={cfg['C2']}, T={cfg['T']}, rate={cfg['target_rate']}")

    # Baseline
    print("\n  Baseline (no SNN)...")
    yolo_clean = YOLO(BASELINE)
    base_res = yolo_clean.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0,
                              verbose=True, plots=False)
    base_map = float(base_res.results_dict.get('metrics/mAP50(B)', 0))
    base_map95 = float(base_res.results_dict.get('metrics/mAP50-95(B)', 0))

    # Per-class baseline
    per_class_base = {}
    if hasattr(base_res, 'ap_class_index'):
        for i, cls_idx in enumerate(base_res.ap_class_index):
            per_class_base[int(cls_idx)] = float(base_res.maps[i])

    print(f"  Baseline: mAP@50={base_map:.4f}, mAP@50-95={base_map95:.4f}")

    bers = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    results = {
        'model': 'YOLO26n-obb + SpikeAdaptSC (faithful)',
        'baseline': {'mAP50': base_map, 'mAP50_95': base_map95},
        'snn': {},
    }

    for ber in bers:
        print(f"\n  BER={ber:.2f}...")
        yolo = YOLO(BASELINE)
        yolo.model.fuse = lambda *a, **kw: yolo.model  # prevent fusion

        # Install faithful SNN hooks
        for idx, (lid, level_snn) in enumerate(zip(HOOK_LAYERS, snn.levels)):
            hook = SNN_Detection_Hook(
                yolo.model.model[lid], level_snn, ber=ber, alpha=args.alpha
            ).to(device)
            yolo.model.model[lid] = hook

        res = yolo.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0,
                       verbose=False, plots=False)
        m50 = float(res.results_dict.get('metrics/mAP50(B)', 0))
        m95 = float(res.results_dict.get('metrics/mAP50-95(B)', 0))

        # Collect hook info
        avg_tx = np.mean([
            yolo.model.model[l].last_info.get('tx_rate', 0)
            for l in HOOK_LAYERS
        ])

        delta = m50 - base_map
        results['snn'][str(ber)] = {
            'mAP50': m50, 'mAP50_95': m95,
            'tx_rate': float(avg_tx),
        }
        print(f"    mAP@50={m50:.4f} ({delta:+.4f}), "
              f"mAP@50-95={m95:.4f}, Rate={avg_tx:.3f}")

    # Summary table
    print("\n" + "=" * 70)
    print(f"  {'BER':>6s}  {'mAP@50':>8s}  {'mAP@50-95':>10s}  {'Δ':>8s}  {'Rate':>6s}")
    print(f"  {'none':>6s}  {base_map:8.4f}  {base_map95:10.4f}  {'—':>8s}  {'1.00':>6s}")
    for ber in bers:
        r = results['snn'].get(str(ber), {})
        if 'mAP50' in r:
            d = r['mAP50'] - base_map
            print(f"  {ber:6.2f}  {r['mAP50']:8.4f}  {r['mAP50_95']:10.4f}  "
                  f"{d:+8.4f}  {r.get('tx_rate', 0):6.3f}")

    os.makedirs('eval', exist_ok=True)
    with open('eval/dota_faithful_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved: eval/dota_faithful_results.json")


# ===================================================================
# Phase 4: Visualization (6+ diverse images)
# ===================================================================
def phase4_viz(args):
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("  Detection Visualization: Faithful SpikeAdaptSC (6 images)")
    print("=" * 70)

    ck = torch.load(SNN_SAVE, map_location=device, weights_only=False)
    cfg = ck['config']
    snn = SpikeAdaptSC_Detection(
        channel_sizes=cfg['channel_sizes'],
        C1=128, C2=cfg['C2'], T=cfg['T'],
        target_rate=cfg['target_rate'],
        channel_type='bsc',
    ).to(device)
    snn.load_state_dict(ck['snn_state'])
    snn.eval()

    data_dict = check_det_dataset('DOTAv1.yaml')
    val_dir = Path(data_dict['val'])
    all_imgs = sorted(list(val_dir.glob('*.png')) + list(val_dir.glob('*.jpg')))

    # Sample 6 diverse images (spread across the val set)
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

        # SNN @ each BER
        for bi, ber in enumerate(ber_levels):
            yolo = YOLO(BASELINE)
            yolo.model.fuse = lambda *a, **kw: yolo.model
            for idx, (lid, level_snn) in enumerate(zip(HOOK_LAYERS, snn.levels)):
                yolo.model.model[lid] = SNN_Detection_Hook(
                    yolo.model.model[lid], level_snn, ber=ber, alpha=args.alpha
                ).to(device)

            r = yolo.predict(img_path, imgsz=640, device=0, verbose=False, conf=0.25)
            ax = axes[bi + 1]
            ax.imshow(r[0].plot()[:, :, ::-1])
            nb = len(r[0].obb) if r[0].obb is not None else 0
            ax.set_title(f'SpikeAdapt-SC\nBER={ber:.2f} ({nb} det)',
                        fontsize=10, fontweight='bold')
            ax.axis('off')

        plt.suptitle(f'YOLO26 + SpikeAdapt-SC on DOTA ({name})',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        out = f'paper/figures/dota_faithful_viz_{img_idx+1}.png'
        plt.savefig(out, dpi=200, bbox_inches='tight')
        plt.savefig(out.replace('.png', '.pdf'), dpi=200, bbox_inches='tight')
        print(f"  Saved: {out}")
        plt.close()

    print(f"\n✅ {len(test_imgs)} visualizations saved!")


# ===================================================================
# Main
# ===================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, default=2, choices=[2, 3, 4])
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--c2', type=int, default=36,
                        help='Bottleneck channels (paper default: 36)')
    parser.add_argument('--T', type=int, default=8,
                        help='SNN timesteps (paper default: 8)')
    parser.add_argument('--target-rate', type=float, default=0.75,
                        help='Target transmission rate (paper default: 0.75)')
    parser.add_argument('--ber-max', type=float, default=0.30)
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Residual mixing: alpha*original + (1-alpha)*SNN')
    args = parser.parse_args()

    {2: phase2_train, 3: phase3_eval, 4: phase4_viz}[args.phase](args)
