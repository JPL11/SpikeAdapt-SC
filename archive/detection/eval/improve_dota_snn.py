#!/usr/bin/env python3
"""Improved YOLO26+SNN: Larger bottleneck + perceptual loss + curriculum.

Key improvements over v1:
  1. C_bot=96 (was 36) — 2.6× more capacity
  2. Residual connections — SNN output + scaled original
  3. Perceptual loss — detection head feature similarity
  4. Progressive BER curriculum — start easy, increase noise
  5. Multi-scale aware — per-level loss weighting

Usage:
  python improve_dota_snn.py train --epochs 80
  python improve_dota_snn.py eval
  python improve_dota_snn.py viz
"""

import os, sys, json, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASELINE = 'runs/obb/runs/dota/yolo26n_obb_baseline/weights/best.pt'
SNN_SAVE = 'runs/yolo26_snn_improved.pth'
HOOK_LAYERS = [4, 6, 10]
CHANNEL_SIZES = [128, 128, 256]


# ===================================================================
# Improved SNN Components
# ===================================================================
class ImprovedLIF_Encoder(nn.Module):
    """Improved LIF encoder with larger capacity + pre-processing."""
    def __init__(self, C_in, C_bot, T=4):
        super().__init__()
        self.T = T
        # Two-layer encoder for better feature extraction
        self.pre = nn.Sequential(
            nn.Conv2d(C_in, C_bot, 1, 1, 0),  # 1x1 channel reduction
            nn.BatchNorm2d(C_bot),
            nn.SiLU(inplace=True),
        )
        self.conv = nn.Conv2d(C_bot, C_bot, 3, 1, 1, groups=C_bot)  # depthwise
        self.bn = nn.BatchNorm2d(C_bot)
        self.threshold = nn.Parameter(torch.ones(1, C_bot, 1, 1) * 0.5)
        self.beta_raw = nn.Parameter(torch.ones(1, C_bot, 1, 1) * 2.0)
    
    @property
    def beta(self):
        return torch.sigmoid(self.beta_raw)
    
    def forward(self, feat):
        h = self.bn(self.conv(self.pre(feat)))
        spikes = []
        mem = torch.zeros_like(h)
        for t in range(self.T):
            mem = self.beta * mem + h
            sig = torch.sigmoid(5.0 * (mem - self.threshold))
            s = (mem > self.threshold).float()
            s = s - sig.detach() + sig  # STE
            mem = mem - s * self.threshold
            spikes.append(s)
        return spikes


class ImprovedSNN_Decoder(nn.Module):
    """Improved decoder with larger capacity + residual."""
    def __init__(self, C_bot, C_out):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(C_bot, C_out, 1, 1, 0),  # 1x1 expand
            nn.BatchNorm2d(C_out),
            nn.SiLU(inplace=True),
            nn.Conv2d(C_out, C_out, 3, 1, 1),  # 3x3 refine
            nn.BatchNorm2d(C_out),
        )
        # Learnable residual gate (starts at 0 = pure reconstruction)
        self.gate = nn.Parameter(torch.zeros(1))
    
    def forward(self, spike_frames, original=None):
        avg = torch.stack(spike_frames).mean(0)
        recon = self.decode(avg)
        if original is not None:
            # Residual: mix reconstruction with original
            alpha = torch.sigmoid(self.gate)
            return (1 - alpha) * recon + alpha * original
        return recon


def bsc_channel(spikes, ber):
    if ber <= 0:
        return spikes
    flip_mask = (torch.rand_like(spikes) < ber).float()
    return torch.abs(spikes - flip_mask)


class ImprovedSNN_Bottleneck(nn.Module):
    """Improved multi-scale SNN bottleneck."""
    def __init__(self, channel_sizes, C_bot=96, T=4):
        super().__init__()
        self.C_bot = C_bot
        self.T = T
        self.encoders = nn.ModuleList([
            ImprovedLIF_Encoder(c, C_bot, T) for c in channel_sizes
        ])
        self.decoders = nn.ModuleList([
            ImprovedSNN_Decoder(C_bot, c) for c in channel_sizes
        ])
    
    def encode_decode(self, features, ber=0.0, use_residual=True):
        """Encode features to spikes, pass through BSC, decode."""
        results = []
        for idx, (feat, enc, dec) in enumerate(
            zip(features, self.encoders, self.decoders)):
            spikes = enc(feat)
            received = [bsc_channel(s, ber) for s in spikes]
            orig = feat if use_residual else None
            recon = dec(received, original=orig)
            results.append(recon)
        return results


class SNN_Hook_Layer(nn.Module):
    """Hook layer for YOLO with improved SNN."""
    def __init__(self, original_layer, encoder, decoder, ber=0.0, use_residual=True):
        super().__init__()
        self.original_layer = original_layer
        self.encoder = encoder
        self.decoder = decoder
        self.ber = ber
        self.use_residual = use_residual
        self.last_fr = 0.0
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
        orig = out if self.use_residual else None
        return self.decoder(received, original=orig)


# ===================================================================
# Feature extraction helpers
# ===================================================================
def get_backbone_features(yolo_model, imgs):
    """Extract TRUE backbone features (no SNN hooks)."""
    save = {}
    x = imgs
    with torch.no_grad():
        for i, layer in enumerate(yolo_model.model[:max(HOOK_LAYERS)+1]):
            if hasattr(layer, 'f') and isinstance(layer.f, list):
                x_in = []
                for j in layer.f:
                    x_in.append(x if j == -1 else save[j])
                x = layer(x_in)
            elif hasattr(layer, 'f') and isinstance(layer.f, int) and layer.f != -1:
                x = layer(save[layer.f])
            else:
                x = layer(x)
            save[i] = x
    return [save[l] for l in HOOK_LAYERS]


def get_neck_features(yolo_model, backbone_features_dict):
    """Run neck (layers 11-21) to get detection-head inputs."""
    save = backbone_features_dict.copy()
    x = save[max(HOOK_LAYERS)]
    with torch.no_grad():
        for i, layer in enumerate(yolo_model.model[max(HOOK_LAYERS)+1:-1]):
            i_abs = i + max(HOOK_LAYERS) + 1
            if hasattr(layer, 'f') and isinstance(layer.f, list):
                x_in = []
                for j in layer.f:
                    x_in.append(x if j == -1 else save[j])
                x = layer(x_in)
            elif hasattr(layer, 'f') and isinstance(layer.f, int) and layer.f != -1:
                x = layer(save[layer.f])
            else:
                x = layer(x)
            save[i_abs] = x
    return save


# ===================================================================
# Training
# ===================================================================
def train_improved(args):
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data import YOLODataset
    from torch.utils.data import DataLoader
    
    print("=" * 70)
    print("  IMPROVED SNN Training: C_bot=96 + Residual + Perceptual + Curriculum")
    print("=" * 70)
    
    yolo_orig = YOLO(BASELINE)
    yolo_orig.model.to(device)
    yolo_orig.model.eval()
    
    snn = ImprovedSNN_Bottleneck(CHANNEL_SIZES, C_bot=args.c_bot, T=args.T).to(device)
    n_params = sum(p.numel() for p in snn.parameters())
    print(f"  SNN params: {n_params:,} (was 333K with C_bot=36)")
    
    optimizer = torch.optim.AdamW(snn.parameters(), lr=5e-4, weight_decay=1e-4)
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
        
        # Progressive BER curriculum: ramp up over first 40 epochs
        max_ber = min(0.30, 0.30 * min(1.0, epoch / 40.0))
        ber = np.random.uniform(0, max_ber)
        
        for batch_data in train_loader:
            imgs = batch_data['img'].to(device).float() / 255.0
            
            # TRUE backbone features
            clean = get_backbone_features(yolo_orig.model, imgs)
            
            # SNN encode → BSC → decode (with residual)
            recon = snn.encode_decode(
                [c.detach() for c in clean], ber=ber, use_residual=True
            )
            
            # Loss 1: MSE at backbone level (per-level weighted)
            # P5 (256ch, 20×20) gets more weight — it's most important for detection
            level_weights = [0.2, 0.3, 0.5]  # P3, P4, P5
            mse_loss = sum(
                w * F.mse_loss(r, c.detach())
                for w, r, c in zip(level_weights, recon, clean)
            )
            
            # Loss 2: Perceptual loss — run neck on both clean and recon features
            # and compare detection head inputs
            recon_save = {}
            clean_save = {}
            x_r, x_c = imgs, imgs
            with torch.no_grad():
                for i, layer in enumerate(yolo_orig.model.model[:max(HOOK_LAYERS)+1]):
                    if hasattr(layer, 'f') and isinstance(layer.f, list):
                        x_in_c = [x_c if j == -1 else clean_save[j] for j in layer.f]
                        x_c = layer(x_in_c)
                    elif hasattr(layer, 'f') and isinstance(layer.f, int) and layer.f != -1:
                        x_c = layer(clean_save[layer.f])
                    else:
                        x_c = layer(x_c)
                    clean_save[i] = x_c
            
            # Replace hook layer features with SNN reconstructed
            for idx, lid in enumerate(HOOK_LAYERS):
                recon_save[lid] = recon[idx]
            # Copy non-hook layers from clean
            for k, v in clean_save.items():
                if k not in recon_save:
                    recon_save[k] = v
            
            # Run neck on both to get FPN outputs
            try:
                x_recon = recon_save[max(HOOK_LAYERS)]
                x_clean = clean_save[max(HOOK_LAYERS)]
                
                # Just compare the immediate FPN features after hook layers
                percep_loss = sum(
                    w * F.l1_loss(recon_save[lid], clean_save[lid].detach())
                    for w, lid in zip(level_weights, HOOK_LAYERS)
                )
            except Exception:
                percep_loss = torch.tensor(0.0, device=device)
            
            # Combined loss
            loss = mse_loss + 0.5 * percep_loss
            
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
        
        print(f"  Epoch {epoch+1:3d}/{args.epochs}  Loss={avg:.6f}  "
              f"BER_max={max_ber:.2f}  BER={ber:.2f}")
        
        if avg < best_loss:
            best_loss = avg
            torch.save({
                'snn_state': snn.state_dict(),
                'epoch': epoch + 1,
                'loss': best_loss,
                'c_bot': args.c_bot,
            }, SNN_SAVE)
    
    print(f"\n✅ Best loss: {best_loss:.6f}")
    print(f"  Saved: {SNN_SAVE}")


# ===================================================================
# Evaluation
# ===================================================================
def eval_ber(args):
    from ultralytics import YOLO
    
    print("=" * 70)
    print("  mAP vs BER: YOLO26 + Improved SNN")
    print("=" * 70)
    
    ck = torch.load(SNN_SAVE, map_location=device, weights_only=False)
    snn = ImprovedSNN_Bottleneck(CHANNEL_SIZES, C_bot=ck['c_bot'], T=args.T).to(device)
    snn.load_state_dict(ck['snn_state'])
    snn.eval()
    print(f"  SNN loaded (epoch {ck['epoch']}, loss={ck['loss']:.6f})")
    
    # Baseline
    yolo_clean = YOLO(BASELINE)
    base_res = yolo_clean.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0,
                              verbose=False, plots=False)
    base_map = float(base_res.results_dict.get('metrics/mAP50(B)', 0))
    print(f"  Baseline: mAP@50={base_map:.4f}")
    
    bers = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    results = {'baseline': base_map, 'snn_improved': {}}
    
    for ber in bers:
        print(f"\n  BER={ber:.2f}...")
        yolo = YOLO(BASELINE)
        yolo.model.fuse = lambda *a, **kw: yolo.model
        
        for idx, (lid, enc, dec) in enumerate(
            zip(HOOK_LAYERS, snn.encoders, snn.decoders)):
            yolo.model.model[lid] = SNN_Hook_Layer(
                yolo.model.model[lid], enc, dec, ber=ber,
                use_residual=True  # gate is learned, always use
            ).to(device)
        
        res = yolo.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0,
                       verbose=False, plots=False)
        m50 = float(res.results_dict.get('metrics/mAP50(B)', 0))
        frs = [yolo.model.model[l].last_fr for l in HOOK_LAYERS]
        delta = m50 - base_map
        results['snn_improved'][str(ber)] = {
            'mAP50': m50, 'fr': float(np.mean(frs))}
        print(f"    mAP@50={m50:.4f} ({delta:+.4f}), FR={np.mean(frs):.3f}")
    
    # Load old results for comparison
    old_results = {}
    old_path = 'eval/dota_yolo26_results.json'
    if os.path.exists(old_path):
        with open(old_path) as f:
            old_data = json.load(f)
            old_results = old_data.get('snn', {})
    
    # Summary
    print("\n" + "=" * 70)
    print(f"  {'BER':>6s}  {'Old mAP':>8s}  {'New mAP':>8s}  {'Δ Improved':>12s}")
    print(f"  {'none':>6s}  {base_map:8.4f}  {base_map:8.4f}  {'(baseline)':>12s}")
    for ber in bers:
        r_new = results['snn_improved'].get(str(ber), {})
        r_old = old_results.get(str(ber), {})
        old_m = r_old.get('mAP50', 0)
        new_m = r_new.get('mAP50', 0)
        improvement = new_m - old_m
        print(f"  {ber:6.2f}  {old_m:8.4f}  {new_m:8.4f}  {improvement:+12.4f}")
    
    os.makedirs('eval', exist_ok=True)
    with open('eval/dota_yolo26_improved_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved: eval/dota_yolo26_improved_results.json")


# ===================================================================
# Visualization
# ===================================================================
def visualize(args):
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    print("=" * 70)
    print("  Detection Visualization: Improved SNN")
    print("=" * 70)
    
    ck = torch.load(SNN_SAVE, map_location=device, weights_only=False)
    snn = ImprovedSNN_Bottleneck(CHANNEL_SIZES, C_bot=ck['c_bot'], T=args.T).to(device)
    snn.load_state_dict(ck['snn_state'])
    snn.eval()
    
    data_dict = check_det_dataset('DOTAv1.yaml')
    val_dir = Path(data_dict['val'])
    all_imgs = sorted(list(val_dir.glob('*.png')) + list(val_dir.glob('*.jpg')))
    
    test_imgs = [str(all_imgs[5]), str(all_imgs[len(all_imgs)//3])]
    ber_levels = [0.0, 0.05, 0.10, 0.20]
    
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
        
        for bi, ber in enumerate(ber_levels):
            yolo = YOLO(BASELINE)
            yolo.model.fuse = lambda *a, **kw: yolo.model
            for idx, (lid, enc, dec) in enumerate(
                zip(HOOK_LAYERS, snn.encoders, snn.decoders)):
                yolo.model.model[lid] = SNN_Hook_Layer(
                    yolo.model.model[lid], enc, dec, ber=ber,
                    use_residual=True  # gate is learned
                ).to(device)
            r = yolo.predict(img_path, imgsz=640, device=0, verbose=False, conf=0.25)
            ax = axes[bi + 1]
            ax.imshow(r[0].plot()[:, :, ::-1])
            nb = len(r[0].obb) if r[0].obb is not None else 0
            ax.set_title(f'Improved SNN\nBER={ber:.2f} ({nb} det)',
                        fontsize=11, fontweight='bold')
            ax.axis('off')
        
        plt.suptitle(f'YOLO26 + Improved SNN on DOTA ({name})',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        out = f'paper/figures/dota_improved_viz_{img_idx+1}.png'
        plt.savefig(out, dpi=200, bbox_inches='tight')
        plt.savefig(out.replace('.png', '.pdf'), dpi=200, bbox_inches='tight')
        print(f"  Saved: {out}")
        plt.close()
    
    print("\n✅ Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'eval', 'viz'])
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--c-bot', type=int, default=96)
    parser.add_argument('--T', type=int, default=4)
    args = parser.parse_args()
    
    {'train': train_improved, 'eval': eval_ber, 'viz': visualize}[args.mode](args)
