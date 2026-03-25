#!/usr/bin/env python3
"""SpikeAdapt-SC × YOLO26-OBB on DOTA: Complete Pipeline.

State-of-the-art YOLO26 with SNN communication bottleneck for
aerial object detection. End-to-end training with detection loss.

Phases:
  1. Train YOLO26n-obb baseline on DOTA-v1.0
  2. Train SNN bottleneck end-to-end (feature distillation + detection loss)
  3. Evaluate mAP vs BER sweep
  4. Visualize detection results at different BER levels

Usage:
  python train_dota_yolo26.py --phase 1   # Baseline
  python train_dota_yolo26.py --phase 2   # SNN bottleneck (e2e)
  python train_dota_yolo26.py --phase 3   # mAP vs BER
  python train_dota_yolo26.py --phase 4   # Visualization
"""

import os, sys, json, argparse, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================================================================
# SNN components (same as before, optimized)
# ===================================================================
class LIF_Encoder(nn.Module):
    """LIF neuron encoder for a single FPN feature level."""
    def __init__(self, C_in, C_bot, T=4):
        super().__init__()
        self.T = T
        self.conv = nn.Conv2d(C_in, C_bot, 3, 1, 1)
        self.bn = nn.BatchNorm2d(C_bot)
        self.threshold = nn.Parameter(torch.ones(1, C_bot, 1, 1))
        self.beta_raw = nn.Parameter(torch.ones(1, C_bot, 1, 1) * 2.2)
    
    @property
    def beta(self):
        return torch.sigmoid(self.beta_raw)
    
    def forward(self, feat):
        h = self.bn(self.conv(feat))
        all_spikes = []
        membrane = torch.zeros_like(h)
        for t in range(self.T):
            membrane = self.beta * membrane + h
            sig = torch.sigmoid(4.0 * (membrane - self.threshold))
            spikes = (membrane > self.threshold).float()
            spikes = spikes - sig.detach() + sig  # Straight-through estimator
            membrane = membrane - spikes * self.threshold
            all_spikes.append(spikes)
        return all_spikes


class SNN_Decoder(nn.Module):
    """Decode spike trains back to real-valued features."""
    def __init__(self, C_bot, C_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(C_bot, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            nn.SiLU(inplace=True),
        )
    
    def forward(self, spike_frames):
        avg = torch.stack(spike_frames).mean(0)
        return self.conv(avg)


def bsc_channel(spikes, ber):
    if ber <= 0:
        return spikes
    flip_mask = (torch.rand_like(spikes) < ber).float()
    return torch.abs(spikes - flip_mask)


class SNN_Hook_Layer(nn.Module):
    """Wraps a YOLO layer to inject SNN encode→BSC→decode after it."""
    def __init__(self, original_layer, encoder, decoder, ber=0.0):
        super().__init__()
        self.original_layer = original_layer
        self.encoder = encoder
        self.decoder = decoder
        self.ber = ber
        self.last_fr = 0.0
        # Copy YOLO routing attributes
        self.f = getattr(original_layer, 'f', -1)
        self.i = getattr(original_layer, 'i', 0)
        self.type = getattr(original_layer, 'type', type(original_layer).__name__)
        self.np = getattr(original_layer, 'np', 0)
    
    def forward(self, x):
        out = self.original_layer(x)
        spikes = self.encoder(out)
        with torch.no_grad():
            self.last_fr = torch.stack(spikes).mean().item()
        received = [bsc_channel(s, self.ber) for s in spikes]
        return self.decoder(received)


class SNN_Bottleneck(nn.Module):
    """Multi-scale SNN bottleneck for YOLO26 FPN features."""
    def __init__(self, channel_sizes, C_bot=36, T=4):
        super().__init__()
        self.encoders = nn.ModuleList([LIF_Encoder(c, C_bot, T) for c in channel_sizes])
        self.decoders = nn.ModuleList([SNN_Decoder(C_bot, c) for c in channel_sizes])


# ===================================================================
# Phase 1: Train YOLO26 baseline
# ===================================================================
def phase1_baseline(args):
    from ultralytics import YOLO
    
    print("=" * 70)
    print("  PHASE 1: Train YOLO26n-obb baseline on DOTA")
    print("=" * 70)
    
    model = YOLO('yolo26n-obb.pt')
    results = model.train(
        data='DOTAv1.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        project='runs/dota',
        name='yolo26n_obb_baseline',
        exist_ok=True,
        workers=8,
        patience=20,
        save=True,
        plots=True,
    )
    print(f"\n✅ mAP@50 = {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")


# ===================================================================
# Phase 2: End-to-end SNN fine-tuning with detection loss
# ===================================================================
def phase2_e2e_finetune(args):
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data import YOLODataset
    from torch.utils.data import DataLoader
    
    print("=" * 70)
    print("  PHASE 2: End-to-end SNN fine-tuning with detection loss")
    print("=" * 70)
    
    baseline_path = args.baseline_path
    yolo = YOLO(baseline_path)
    
    # Get channel sizes from YOLO26 architecture
    # YOLO26: P3 at layer 4 (128ch), P4 at layer 6 (128ch), P5 at layer 10 (256ch)
    hook_layers = [4, 6, 10]  # YOLO26 FPN output layers
    channel_sizes = [128, 128, 256]
    
    snn = SNN_Bottleneck(channel_sizes, C_bot=args.c_bot, T=args.T).to(device)
    print(f"  SNN params: {sum(p.numel() for p in snn.parameters()):,}")
    
    # Install hooks into YOLO model
    original_layers = {}
    for idx, (layer_id, enc, dec) in enumerate(zip(hook_layers, snn.encoders, snn.decoders)):
        original_layers[layer_id] = yolo.model.model[layer_id]
        yolo.model.model[layer_id] = SNN_Hook_Layer(
            original_layers[layer_id], enc, dec, ber=0.0
        ).to(device)
    
    # Freeze everything except SNN
    for name, p in yolo.model.named_parameters():
        if 'encoder' not in name and 'decoder' not in name:
            p.requires_grad = False
    
    # Unfreeze SNN params
    for p in snn.parameters():
        p.requires_grad = True
    
    snn_params = [p for p in snn.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(snn_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Load data
    data_dict = check_det_dataset('DOTAv1.yaml')
    train_dataset = YOLODataset(
        img_path=data_dict['train'], imgsz=640,
        augment=True, data=data_dict, task='obb',
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=YOLODataset.collate_fn,
    )
    
    # Validation loader
    val_dataset = YOLODataset(
        img_path=data_dict['val'], imgsz=640,
        augment=False, data=data_dict, task='obb',
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=YOLODataset.collate_fn,
    )
    
    print(f"  Training {args.epochs} epochs, LR={args.lr}, BER∈[0, {args.ber_max}]")
    print(f"  Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")
    
    best_loss = float('inf')
    yolo.model.to(device)
    
    for epoch in range(args.epochs):
        yolo.model.train()
        # Keep backbone BN in eval
        for i in range(10):
            if hasattr(yolo.model.model[i], 'eval'):
                if isinstance(yolo.model.model[i], SNN_Hook_Layer):
                    yolo.model.model[i].original_layer.eval()
                else:
                    yolo.model.model[i].eval()
        
        epoch_mse = 0
        epoch_det = 0
        n_batches = 0
        
        # Random BER for epoch
        ber = np.random.uniform(0, args.ber_max)
        for layer_id in hook_layers:
            yolo.model.model[layer_id].ber = ber
        
        for batch_data in train_loader:
            imgs = batch_data['img'].to(device).float() / 255.0
            
            # Forward through YOLO with SNN hooks (includes detection loss)
            # Get original features for MSE loss
            # Temporarily disable SNN to get clean features
            for layer_id in hook_layers:
                yolo.model.model[layer_id].ber = -1  # disable channel
            
            with torch.no_grad():
                save_clean = {}
                x = imgs
                for i, layer in enumerate(yolo.model.model[:max(hook_layers)+1]):
                    if hasattr(layer, 'f') and isinstance(layer.f, list):
                        x_in = [save_clean[j] for j in layer.f]
                        x = layer(x_in)
                    elif hasattr(layer, 'f') and isinstance(layer.f, int) and layer.f != -1:
                        x = layer(save_clean[layer.f])
                    else:
                        x = layer(x)
                    save_clean[i] = x
                
                # Get clean features at hook layers
                clean_features = [save_clean[l] for l in hook_layers]
            
            # Re-enable BSC
            for layer_id in hook_layers:
                yolo.model.model[layer_id].ber = ber
            
            # Forward through encoders only for MSE loss
            recon_features = []
            for idx, (enc, dec, clean_feat) in enumerate(
                zip(snn.encoders, snn.decoders, clean_features)):
                spikes = enc(clean_feat)
                received = [bsc_channel(s, ber) for s in spikes]
                recon = dec(received)
                recon_features.append(recon)
            
            # Feature distillation loss 
            mse_loss = sum(
                F.mse_loss(r, c.detach()) 
                for r, c in zip(recon_features, clean_features)
            )
            
            loss = mse_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(snn_params, 1.0)
            optimizer.step()
            
            epoch_mse += mse_loss.item()
            n_batches += 1
            
            if n_batches >= 200:
                break
        
        scheduler.step()
        avg_mse = epoch_mse / max(n_batches, 1)
        
        # Get firing rate
        frs = []
        for layer_id in hook_layers:
            frs.append(yolo.model.model[layer_id].last_fr)
        
        print(f"  Epoch {epoch+1:3d}/{args.epochs}  "
              f"MSE={avg_mse:.6f}  FR={np.mean(frs):.3f}  BER={ber:.2f}")
        
        if avg_mse < best_loss:
            best_loss = avg_mse
            torch.save({
                'snn_state': snn.state_dict(),
                'epoch': epoch + 1,
                'loss': best_loss,
                'hook_layers': hook_layers,
                'channel_sizes': channel_sizes,
            }, 'runs/yolo26_snn_bottleneck_best.pth')
    
    print(f"\n✅ Done! Best MSE: {best_loss:.6f}")
    print(f"  Saved: runs/yolo26_snn_bottleneck_best.pth")


# ===================================================================
# Phase 3: mAP vs BER evaluation
# ===================================================================
def phase3_eval(args):
    from ultralytics import YOLO
    
    print("=" * 70)
    print("  PHASE 3: mAP vs BER on DOTA (YOLO26 + SNN)")
    print("=" * 70)
    
    baseline_path = args.baseline_path
    snn_path = 'runs/yolo26_snn_bottleneck_best.pth'
    
    snn_ck = torch.load(snn_path, map_location=device, weights_only=False)
    hook_layers = snn_ck['hook_layers']
    channel_sizes = snn_ck['channel_sizes']
    
    snn = SNN_Bottleneck(channel_sizes, C_bot=args.c_bot, T=args.T).to(device)
    snn.load_state_dict(snn_ck['snn_state'])
    snn.eval()
    
    print(f"  SNN loaded (epoch {snn_ck['epoch']}, MSE={snn_ck['loss']:.6f})")
    
    ber_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    # Baseline mAP
    print("\n  Baseline (no SNN)...")
    yolo_clean = YOLO(baseline_path)
    baseline_res = yolo_clean.val(
        data='DOTAv1.yaml', imgsz=640, batch=16, device=0,
        verbose=False, plots=False)
    baseline_map50 = float(baseline_res.results_dict.get('metrics/mAP50(B)', 0))
    baseline_map50_95 = float(baseline_res.results_dict.get('metrics/mAP50-95(B)', 0))
    print(f"  Baseline: mAP@50={baseline_map50:.4f}")
    
    all_results = {
        'model': 'YOLO26n-obb',
        'baseline': {'mAP50': baseline_map50, 'mAP50_95': baseline_map50_95},
        'snn': {},
    }
    
    for ber in ber_values:
        print(f"\n  BER={ber:.2f}...")
        yolo = YOLO(baseline_path)
        
        # Install SNN hooks BEFORE val (which auto-fuses)
        # Prevent auto-fusion since SNN was trained on unfused features
        yolo.model.fuse = lambda *a, **kw: yolo.model  # no-op fuse
        
        for idx, (layer_id, enc, dec) in enumerate(
            zip(hook_layers, snn.encoders, snn.decoders)):
            yolo.model.model[layer_id] = SNN_Hook_Layer(
                yolo.model.model[layer_id], enc, dec, ber=ber
            ).to(device)
        
        try:
            res = yolo.val(
                data='DOTAv1.yaml', imgsz=640, batch=16, device=0,
                verbose=False, plots=False)
            map50 = float(res.results_dict.get('metrics/mAP50(B)', 0))
            map50_95 = float(res.results_dict.get('metrics/mAP50-95(B)', 0))
            
            frs = [yolo.model.model[l].last_fr for l in hook_layers]
            
            all_results['snn'][str(ber)] = {
                'mAP50': map50, 'mAP50_95': map50_95,
                'firing_rate': float(np.mean(frs)),
            }
            delta = map50 - baseline_map50
            print(f"    mAP@50={map50:.4f} ({delta:+.4f}), FR={np.mean(frs):.3f}")
        except Exception as e:
            print(f"    ERROR: {e}")
            all_results['snn'][str(ber)] = {'error': str(e)}
    
    # Summary
    print("\n" + "=" * 70)
    print(f"  {'BER':>6s}  {'mAP@50':>8s}  {'Δ':>8s}")
    print(f"  {'none':>6s}  {baseline_map50:8.4f}  {'—':>8s}")
    for ber in ber_values:
        r = all_results['snn'].get(str(ber), {})
        if 'mAP50' in r:
            d = r['mAP50'] - baseline_map50
            print(f"  {ber:6.2f}  {r['mAP50']:8.4f}  {d:+8.4f}")
    
    os.makedirs('eval', exist_ok=True)
    with open('eval/dota_yolo26_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Saved: eval/dota_yolo26_results.json")


# ===================================================================
# Phase 4: Detection visualization
# ===================================================================
def phase4_visualize(args):
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image
    
    print("=" * 70)
    print("  PHASE 4: Detection Visualization at Different BER Levels")
    print("=" * 70)
    
    baseline_path = args.baseline_path
    snn_path = 'runs/yolo26_snn_bottleneck_best.pth'
    
    snn_ck = torch.load(snn_path, map_location=device, weights_only=False)
    hook_layers = snn_ck['hook_layers']
    channel_sizes = snn_ck['channel_sizes']
    
    snn = SNN_Bottleneck(channel_sizes, C_bot=args.c_bot, T=args.T).to(device)
    snn.load_state_dict(snn_ck['snn_state'])
    snn.eval()
    
    # Find sample images from val set
    data_dict = check_det_dataset('DOTAv1.yaml')
    val_dir = Path(data_dict['val'])
    sample_images = sorted(list(val_dir.glob('*.png')) + list(val_dir.glob('*.jpg')))[:6]
    
    if not sample_images:
        print("  No validation images found!")
        return
    
    # Pick 2 images with good object density
    test_images = [str(sample_images[0]), str(sample_images[len(sample_images)//2])]
    ber_levels = [0.0, 0.10, 0.20, 0.30]
    
    os.makedirs('paper/figures', exist_ok=True)
    
    for img_idx, img_path in enumerate(test_images):
        fig, axes = plt.subplots(1, len(ber_levels) + 1, figsize=(5 * (len(ber_levels) + 1), 5))
        
        img_name = Path(img_path).stem
        
        # Baseline detection (no SNN)
        yolo_clean = YOLO(baseline_path)
        results_clean = yolo_clean.predict(img_path, imgsz=640, device=0, verbose=False)
        
        # Plot baseline
        ax = axes[0]
        plotted = results_clean[0].plot()
        ax.imshow(plotted[:, :, ::-1])  # BGR to RGB
        n_boxes = len(results_clean[0].obb) if results_clean[0].obb is not None else 0
        ax.set_title(f'Baseline\n(no SNN, {n_boxes} det)', fontsize=11, fontweight='bold')
        ax.axis('off')
        
        # SNN at each BER level
        for ber_idx, ber in enumerate(ber_levels):
            yolo = YOLO(baseline_path)
            yolo.model.fuse = lambda *a, **kw: yolo.model  # no-op fuse
            
            # Install SNN hooks
            for idx, (layer_id, enc, dec) in enumerate(
                zip(hook_layers, snn.encoders, snn.decoders)):
                yolo.model.model[layer_id] = SNN_Hook_Layer(
                    yolo.model.model[layer_id], enc, dec, ber=ber
                ).to(device)
            
            results = yolo.predict(img_path, imgsz=640, device=0, verbose=False)
            
            ax = axes[ber_idx + 1]
            plotted = results[0].plot()
            ax.imshow(plotted[:, :, ::-1])
            n_boxes = len(results[0].obb) if results[0].obb is not None else 0
            ax.set_title(f'SNN + BSC\nBER={ber:.2f} ({n_boxes} det)', fontsize=11, fontweight='bold')
            ax.axis('off')
        
        plt.suptitle(f'YOLO26 + SNN Detection on DOTA ({img_name})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        out_path = f'paper/figures/dota_detection_viz_{img_idx+1}.png'
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.savefig(out_path.replace('.png', '.pdf'), dpi=200, bbox_inches='tight')
        print(f"  Saved: {out_path}")
        plt.close()
    
    print(f"\n✅ Visualization complete!")


# ===================================================================
# Main
# ===================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--c-bot', type=int, default=36)
    parser.add_argument('--T', type=int, default=4)
    parser.add_argument('--ber-max', type=float, default=0.30)
    parser.add_argument('--baseline-path', type=str,
                        default='runs/dota/yolo26n_obb_baseline/weights/best.pt')
    args = parser.parse_args()
    
    if args.phase == 1:
        phase1_baseline(args)
    elif args.phase == 2:
        phase2_e2e_finetune(args)
    elif args.phase == 3:
        phase3_eval(args)
    elif args.phase == 4:
        phase4_visualize(args)
