#!/usr/bin/env python3
"""YOLO26+SNN: Fixed SNN training + BER eval + visualization.

Root cause fix: Extract clean features from ORIGINAL backbone (no hooks),
then train SNN to reconstruct those. Previous version accidentally extracted
features through the SNN hook (with ber=-1).

Usage:
  python fix_dota_snn.py train    # Retrain SNN properly
  python fix_dota_snn.py eval     # mAP vs BER
  python fix_dota_snn.py viz      # Detection visualization
"""

import os, sys, json, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from train.train_dota_yolo26 import (
    SNN_Bottleneck, SNN_Hook_Layer, bsc_channel, LIF_Encoder, SNN_Decoder
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASELINE = 'runs/obb/runs/dota/yolo26n_obb_baseline/weights/best.pt'
SNN_SAVE = 'runs/yolo26_snn_fixed.pth'
HOOK_LAYERS = [4, 6, 10]  # YOLO26 P3/P4/P5
CHANNEL_SIZES = [128, 128, 256]


def get_original_features(yolo_model, imgs):
    """Extract features from original YOLO backbone (NO SNN hooks)."""
    save = {}
    x = imgs
    with torch.no_grad():
        for i, layer in enumerate(yolo_model.model[:max(HOOK_LAYERS)+1]):
            if hasattr(layer, 'f') and isinstance(layer.f, list):
                x_in = []
                for j in layer.f:
                    if j == -1:
                        x_in.append(x)
                    else:
                        x_in.append(save[j])
                x = layer(x_in)
            elif hasattr(layer, 'f') and isinstance(layer.f, int) and layer.f != -1:
                x = layer(save[layer.f])
            else:
                x = layer(x)
            save[i] = x
    return [save[l] for l in HOOK_LAYERS]


def train_snn(args):
    """Train SNN bottleneck by extracting TRUE backbone features."""
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data import YOLODataset
    from torch.utils.data import DataLoader
    
    print("=" * 70)
    print("  FIXED SNN Training: True backbone feature distillation")
    print("=" * 70)
    
    # Load ORIGINAL model (no hooks!)
    yolo_orig = YOLO(BASELINE)
    yolo_orig.model.to(device)
    yolo_orig.model.eval()
    
    # Create SNN bottleneck
    snn = SNN_Bottleneck(CHANNEL_SIZES, C_bot=args.c_bot, T=args.T).to(device)
    print(f"  SNN params: {sum(p.numel() for p in snn.parameters()):,}")
    
    optimizer = torch.optim.AdamW(snn.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    data_dict = check_det_dataset('DOTAv1.yaml')
    train_dataset = YOLODataset(
        img_path=data_dict['train'], imgsz=640,
        augment=True, data=data_dict, task='obb',
    )
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=YOLODataset.collate_fn,
    )
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        snn.train()
        epoch_loss = 0
        n_batches = 0
        ber = np.random.uniform(0, 0.30)
        
        for batch_data in train_loader:
            imgs = batch_data['img'].to(device).float() / 255.0
            
            # TRUE backbone features (no SNN!)
            clean = get_original_features(yolo_orig.model, imgs)
            
            # SNN encode → BSC → decode
            recon = []
            for idx, (enc, dec, feat) in enumerate(
                zip(snn.encoders, snn.decoders, clean)):
                spikes = enc(feat.detach())
                received = [bsc_channel(s, ber) for s in spikes]
                recon.append(dec(received))
            
            loss = sum(F.mse_loss(r, c.detach()) for r, c in zip(recon, clean))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(snn.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            if n_batches >= 200:
                break
        
        scheduler.step()
        avg = epoch_loss / max(n_batches, 1)
        
        print(f"  Epoch {epoch+1:3d}/{args.epochs}  MSE={avg:.6f}  BER={ber:.2f}")
        
        if avg < best_loss:
            best_loss = avg
            torch.save({
                'snn_state': snn.state_dict(),
                'epoch': epoch + 1,
                'loss': best_loss,
            }, SNN_SAVE)
    
    print(f"\n✅ Best MSE: {best_loss:.6f}")
    print(f"  Saved: {SNN_SAVE}")


def eval_ber(args):
    """Evaluate mAP vs BER using YOLO26 with SNN hooks."""
    from ultralytics import YOLO
    
    print("=" * 70)
    print("  mAP vs BER: YOLO26 + SNN (fixed)")
    print("=" * 70)
    
    ck = torch.load(SNN_SAVE, map_location=device, weights_only=False)
    snn = SNN_Bottleneck(CHANNEL_SIZES, C_bot=args.c_bot, T=args.T).to(device)
    snn.load_state_dict(ck['snn_state'])
    snn.eval()
    print(f"  SNN loaded (epoch {ck['epoch']}, MSE={ck['loss']:.6f})")
    
    # Baseline
    yolo_clean = YOLO(BASELINE)
    base_res = yolo_clean.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0,
                              verbose=False, plots=False)
    base_map = float(base_res.results_dict.get('metrics/mAP50(B)', 0))
    print(f"  Baseline: mAP@50={base_map:.4f}")
    
    bers = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    results = {'baseline': base_map, 'snn': {}}
    
    for ber in bers:
        print(f"\n  BER={ber:.2f}...")
        yolo = YOLO(BASELINE)
        # Prevent fusion (SNN trained on unfused features)
        yolo.model.fuse = lambda *a, **kw: yolo.model
        
        for idx, (lid, enc, dec) in enumerate(
            zip(HOOK_LAYERS, snn.encoders, snn.decoders)):
            yolo.model.model[lid] = SNN_Hook_Layer(
                yolo.model.model[lid], enc, dec, ber=ber
            ).to(device)
        
        res = yolo.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0,
                       verbose=False, plots=False)
        m50 = float(res.results_dict.get('metrics/mAP50(B)', 0))
        frs = [yolo.model.model[l].last_fr for l in HOOK_LAYERS]
        delta = m50 - base_map
        results['snn'][str(ber)] = {'mAP50': m50, 'fr': float(np.mean(frs))}
        print(f"    mAP@50={m50:.4f} ({delta:+.4f}), FR={np.mean(frs):.3f}")
    
    # Summary
    print("\n" + "=" * 70)
    print(f"  {'BER':>6s}  {'mAP@50':>8s}  {'Δ':>8s}")
    print(f"  {'none':>6s}  {base_map:8.4f}  {'—':>8s}")
    for ber in bers:
        r = results['snn'].get(str(ber), {})
        if 'mAP50' in r:
            d = r['mAP50'] - base_map
            print(f"  {ber:6.2f}  {r['mAP50']:8.4f}  {d:+8.4f}")
    
    with open('eval/dota_yolo26_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved: eval/dota_yolo26_results.json")
    
    return results


def visualize(args):
    """Detection visualization: baseline vs SNN at different BER levels."""
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    import matplotlib.pyplot as plt
    
    print("=" * 70)
    print("  Detection Visualization")
    print("=" * 70)
    
    ck = torch.load(SNN_SAVE, map_location=device, weights_only=False)
    snn = SNN_Bottleneck(CHANNEL_SIZES, C_bot=args.c_bot, T=args.T).to(device)
    snn.load_state_dict(ck['snn_state'])
    snn.eval()
    
    data_dict = check_det_dataset('DOTAv1.yaml')
    val_dir = Path(data_dict['val'])
    all_imgs = sorted(list(val_dir.glob('*.png')) + list(val_dir.glob('*.jpg')))
    
    # Pick 2 images at different positions for diversity
    test_imgs = [str(all_imgs[5]), str(all_imgs[len(all_imgs)//3])]
    ber_levels = [0.0, 0.10, 0.20, 0.30]
    
    os.makedirs('paper/figures', exist_ok=True)
    
    for img_idx, img_path in enumerate(test_imgs):
        n_cols = len(ber_levels) + 1
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
        name = Path(img_path).stem
        
        # Baseline
        yolo = YOLO(BASELINE)
        r = yolo.predict(img_path, imgsz=640, device=0, verbose=False, conf=0.25)
        ax = axes[0]
        ax.imshow(r[0].plot()[:, :, ::-1])
        nb = len(r[0].obb) if r[0].obb is not None else 0
        ax.set_title(f'Baseline\n({nb} det)', fontsize=11, fontweight='bold')
        ax.axis('off')
        
        # SNN @ each BER
        for bi, ber in enumerate(ber_levels):
            yolo = YOLO(BASELINE)
            yolo.model.fuse = lambda *a, **kw: yolo.model
            
            for idx, (lid, enc, dec) in enumerate(
                zip(HOOK_LAYERS, snn.encoders, snn.decoders)):
                yolo.model.model[lid] = SNN_Hook_Layer(
                    yolo.model.model[lid], enc, dec, ber=ber
                ).to(device)
            
            r = yolo.predict(img_path, imgsz=640, device=0, verbose=False, conf=0.25)
            ax = axes[bi + 1]
            ax.imshow(r[0].plot()[:, :, ::-1])
            nb = len(r[0].obb) if r[0].obb is not None else 0
            ax.set_title(f'SNN BER={ber:.2f}\n({nb} det)', fontsize=11, fontweight='bold')
            ax.axis('off')
        
        plt.suptitle(f'YOLO26 + SNN Detection on DOTA ({name})',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        out = f'paper/figures/dota_yolo26_viz_{img_idx+1}.png'
        plt.savefig(out, dpi=200, bbox_inches='tight')
        plt.savefig(out.replace('.png', '.pdf'), dpi=200, bbox_inches='tight')
        print(f"  Saved: {out}")
        plt.close()
    
    print("\n✅ Visualization complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'eval', 'viz'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--c-bot', type=int, default=36)
    parser.add_argument('--T', type=int, default=4)
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_snn(args)
    elif args.mode == 'eval':
        eval_ber(args)
    elif args.mode == 'viz':
        visualize(args)
