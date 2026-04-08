#!/usr/bin/env python3
"""SpikeAdapt-SC V8 — Best-of-both-worlds: V6d-v2 SNN + P2/SE enhancements.

Option B: Load V6d-v2's proven 200-epoch SNN, add P2+SE, fine-tune 40ep
Option A: Full 200ep MSE retrain with best arch, then fine-tune

Usage:
  python train/train_dota_v8.py --option B --phase train   # Quick graft (40min)
  python train/train_dota_v8.py --option B --phase eval    # Eval
  python train/train_dota_v8.py --option A --phase train   # Full retrain (3.5hr)
  python train/train_dota_v8.py --option A --phase eval
"""

import os, sys, json, argparse, copy, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASELINE = 'runs/obb/runs/dota/yolo26n_obb_baseline/weights/best.pt'
V6D2_SNN = 'runs/yolo26_snn_v6d2.pth'    # V6d-v2: 200ep MSE, original encoder
V8B_SAVE = 'runs/yolo26_snn_v8_optB.pth'
V8A_SAVE = 'runs/yolo26_snn_v8_optA.pth'
V8A_STAGE1 = 'runs/yolo26_snn_v8A_stage1.pth'

HOOK_LAYERS = [4, 6, 10]
CHANNEL_SIZES = [128, 128, 256]
P2_LAYER = 2
P2_CHANNELS = 64

# Import components from v6d2d (has MaxFormer encoder)
from train.train_dota_v6d2d import (
    SpikeAdaptSC_Det_Multi, SpikeAdaptSC_Det,
    SNNEncoder, NoiseAwareScorerDet, DetMasker, DetDecoder,
    BSC_Channel, AdaptiveLIF, MPBN, SpikeFunction,
    yolo_forward_with_hooks,
)


# ############################################################################
# LIGHTWEIGHT MODULES (from V7)
# ############################################################################

class FeatureEnhancer(nn.Module):
    """Lightweight DWConv+SE post-SNN feature enhancement."""
    def __init__(self, C):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(C, C, 3, 1, 1, groups=C),
            nn.BatchNorm2d(C), nn.SiLU(),
            nn.Conv2d(C, C, 1),
            nn.BatchNorm2d(C))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C, max(C // 4, 8), 1), nn.SiLU(),
            nn.Conv2d(max(C // 4, 8), C, 1), nn.Sigmoid())
        self.alpha = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        refined = self.spatial(x)
        att = self.se(refined)
        enhanced = refined * att
        return x + torch.sigmoid(self.alpha) * enhanced


class P2Bridge(nn.Module):
    """Fuse P2 high-res features into P3 via learned downsampling."""
    def __init__(self, C_p2=64, C_p3=128):
        super().__init__()
        self.bridge = nn.Sequential(
            nn.Conv2d(C_p2, C_p3, 3, stride=2, padding=1),
            nn.BatchNorm2d(C_p3), nn.SiLU())
        self.gate = nn.Sequential(
            nn.Conv2d(C_p3 * 2, C_p3, 1), nn.Sigmoid())
    
    def forward(self, p2_feat, p3_feat):
        p2_down = self.bridge(p2_feat)
        gate = self.gate(torch.cat([p2_down, p3_feat], dim=1))
        return p3_feat + gate * p2_down


# ############################################################################
# HOOKS
# ############################################################################

class SNN_Hook_Enhanced(nn.Module):
    def __init__(self, original_layer, snn_level, ber=0.0, enhancer=None):
        super().__init__()
        self.original_layer = original_layer
        self.snn = snn_level
        self.ber = ber
        self.enhancer = enhancer
        self.f = getattr(original_layer, 'f', -1)
        self.i = getattr(original_layer, 'i', 0)
        self.type = getattr(original_layer, 'type', type(original_layer).__name__)
        self.np = getattr(original_layer, 'np', 0)
        self.last_info = {}

    def forward(self, x):
        out = self.original_layer(x)
        recon, info = self.snn(out, noise_param=self.ber)
        if self.enhancer is not None:
            recon = self.enhancer(recon)
        self.last_info = info
        return recon


class SNN_Hook_P2(nn.Module):
    def __init__(self, original_layer, snn_level, ber=0.0, enhancer=None):
        super().__init__()
        self.original_layer = original_layer
        self.snn = snn_level
        self.ber = ber
        self.enhancer = enhancer
        self.f = getattr(original_layer, 'f', -1)
        self.i = getattr(original_layer, 'i', 0)
        self.type = getattr(original_layer, 'type', type(original_layer).__name__)
        self.np = getattr(original_layer, 'np', 0)
        self.last_info = {}
        self.last_p2_recon = None

    def forward(self, x):
        out = self.original_layer(x)
        recon, info = self.snn(out, noise_param=self.ber)
        if self.enhancer is not None:
            recon = self.enhancer(recon)
        self.last_info = info
        self.last_p2_recon = recon
        return out  # Pass through original


class SNN_Hook_P3WithBridge(nn.Module):
    def __init__(self, original_layer, snn_level, p2_hook, p2_bridge, ber=0.0, enhancer=None):
        super().__init__()
        self.original_layer = original_layer
        self.snn = snn_level
        self.ber = ber
        self.enhancer = enhancer
        self.p2_hook = p2_hook
        self.p2_bridge = p2_bridge
        self.f = getattr(original_layer, 'f', -1)
        self.i = getattr(original_layer, 'i', 0)
        self.type = getattr(original_layer, 'type', type(original_layer).__name__)
        self.np = getattr(original_layer, 'np', 0)
        self.last_info = {}

    def forward(self, x):
        out = self.original_layer(x)
        recon, info = self.snn(out, noise_param=self.ber)
        if self.enhancer is not None:
            recon = self.enhancer(recon)
        if self.p2_hook.last_p2_recon is not None:
            recon = self.p2_bridge(self.p2_hook.last_p2_recon, recon)
        self.last_info = info
        return recon


# ############################################################################
# MODEL BUILDING
# ############################################################################

def build_v8_model(snn, ber=0.0, for_training=False):
    """Build YOLO with V6d-v2 SNN + P2 hook + SE enhancers."""
    from ultralytics import YOLO
    
    yolo = YOLO(BASELINE)
    yolo.model.to(device)
    yolo.model.fuse = lambda *a, **kw: yolo.model
    
    extra_modules = {}
    
    # P2 SNN level (new, trains from scratch)
    p2_snn = SpikeAdaptSC_Det(P2_CHANNELS, C_spike=36, T=8, target_rate=0.75).to(device)
    p2_enhancer = FeatureEnhancer(P2_CHANNELS).to(device)
    
    p2_hook = SNN_Hook_P2(
        yolo.model.model[P2_LAYER], p2_snn, ber=ber, enhancer=p2_enhancer
    ).to(device)
    yolo.model.model[P2_LAYER] = p2_hook
    
    p2_bridge = P2Bridge(P2_CHANNELS, CHANNEL_SIZES[0]).to(device)
    
    # P3 hook with bridge
    enhancers = nn.ModuleDict()
    p3_enhancer = FeatureEnhancer(CHANNEL_SIZES[0]).to(device)
    enhancers['e0'] = p3_enhancer
    lev0 = copy.deepcopy(snn.levels[0]) if not for_training else snn.levels[0]
    if not for_training:
        lev0.eval()
    yolo.model.model[HOOK_LAYERS[0]] = SNN_Hook_P3WithBridge(
        yolo.model.model[HOOK_LAYERS[0]], lev0, p2_hook, p2_bridge,
        ber=ber, enhancer=p3_enhancer
    ).to(device)
    
    # P4, P5 hooks with enhancers
    for idx in range(1, len(HOOK_LAYERS)):
        lid = HOOK_LAYERS[idx]
        lev = copy.deepcopy(snn.levels[idx]) if not for_training else snn.levels[idx]
        if not for_training:
            lev.eval()
        enhancer = FeatureEnhancer(CHANNEL_SIZES[idx]).to(device)
        enhancers[f'e{idx}'] = enhancer
        yolo.model.model[lid] = SNN_Hook_Enhanced(
            yolo.model.model[lid], lev, ber=ber, enhancer=enhancer
        ).to(device)
    
    extra_modules['enhancers'] = enhancers
    extra_modules['p2_snn'] = p2_snn
    extra_modules['p2_enhancer'] = p2_enhancer
    extra_modules['p2_bridge'] = p2_bridge
    
    return yolo, extra_modules


# ############################################################################
# OPTION B: GRAFT P2+SE ONTO V6d-v2's SNN
# ############################################################################

def train_option_b(args):
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data import YOLODataset
    from torch.utils.data import DataLoader

    save_path = V8B_SAVE
    print("=" * 70)
    print("  V8 Option B: V6d-v2 SNN + P2 Bridge + SE Enhancers")
    print("=" * 70)

    # Load V6d-v2's proven 200-epoch SNN
    ck_snn = torch.load(V6D2_SNN, map_location=device, weights_only=False)
    cfg = ck_snn['config']
    snn = SpikeAdaptSC_Det_Multi(
        cfg['channel_sizes'], C_spike=cfg['C_spike'], T=cfg['T'],
        target_rate=cfg['target_rate']
    ).to(device)
    
    # Load with strict=False — skip MaxFormer HF keys that V6d-v2 doesn't have
    missing, unexpected = snn.load_state_dict(ck_snn['snn_state'], strict=False)
    if missing:
        print(f"  Missing keys (MaxFormer HF): {len(missing)}")
        for k in missing[:5]:
            print(f"    {k}")
        # Zero-out the MaxFormer HF gate so it acts as identity (no HF injection)
        for lev in snn.levels:
            enc = lev.encoder
            if hasattr(enc, 'hf_gate'):
                # Set gate output to ~0 so x + gate*hf ≈ x (no HF contribution)
                nn.init.constant_(enc.hf_gate[0].bias, -10.0)
                print("    → HF gate biased to ~0 (identity bypass)")
    
    snn.eval()
    for param in snn.parameters():
        param.requires_grad = False
    print(f"  Loaded V6d-v2 SNN (epoch {ck_snn['epoch']}, loss {ck_snn['loss']:.6f})")
    print(f"  SNN params: {sum(p.numel() for p in snn.parameters()):,} (ALL FROZEN)")

    # Build model with P2+SE
    yolo, extra_modules = build_v8_model(snn, ber=0.0, for_training=True)
    yolo_model = yolo.model.to(device)

    # Collect trainable params: enhancers + P2 + neck + head
    trainable_params = []
    for i, layer in enumerate(yolo_model.model):
        if i <= max(HOOK_LAYERS) and i not in [P2_LAYER] + HOOK_LAYERS:
            for param in layer.parameters():
                param.requires_grad = False
        elif i in HOOK_LAYERS or i == P2_LAYER:
            for name, param in layer.named_parameters():
                if 'snn' in name and 'p2_snn' not in name:
                    param.requires_grad = False
                elif 'enhancer' in name or 'bridge' in name or 'p2_bridge' in name or 'p2_snn' in name:
                    param.requires_grad = True
                    trainable_params.append(param)
                else:
                    param.requires_grad = False
        else:
            for param in layer.parameters():
                param.requires_grad = True
                trainable_params.append(param)
    
    for key, mod in extra_modules.items():
        for param in mod.parameters():
            if param.requires_grad:
                trainable_params.append(param)
    
    # Deduplicate
    seen = set()
    trainable_params = [p for p in trainable_params if (pid := id(p)) not in seen and not seen.add(pid)]
    
    n_trainable = sum(p.numel() for p in trainable_params)
    n_extra = sum(p.numel() for m in extra_modules.values() for p in m.parameters())
    print(f"  Extra module params: {n_extra:,}")
    print(f"  Trainable: {n_trainable:,}")

    # Loss
    from types import SimpleNamespace
    existing_args = yolo_model.args if isinstance(yolo_model.args, dict) else vars(yolo_model.args)
    loss_hyp = {'box': 7.5, 'cls': 0.5, 'dfl': 1.5, 'angle': 1.0, 'overlap_mask': True}
    loss_hyp.update(existing_args)
    for k, v in [('box', 7.5), ('cls', 0.5), ('dfl', 1.5), ('angle', 1.0)]:
        if k not in loss_hyp: loss_hyp[k] = v
    yolo_model.args = SimpleNamespace(**loss_hyp)
    criterion = yolo_model.init_criterion()

    # Data
    data_dict = check_det_dataset('DOTAv1.yaml')
    train_dataset = YOLODataset(img_path=data_dict['train'], imgsz=640, augment=True, data=data_dict, task='obb')
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True, collate_fn=YOLODataset.collate_fn)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    warmup = 3
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - warmup, eta_min=1e-6)
    best_loss = float('inf')

    for epoch in range(args.epochs):
        yolo_model.train()
        for lid in HOOK_LAYERS:
            h = yolo_model.model[lid]
            if hasattr(h, 'snn'): h.snn.eval()
        if hasattr(yolo_model.model[P2_LAYER], 'snn'):
            yolo_model.model[P2_LAYER].snn.eval()
        
        epoch_loss, n_batches = 0, 0
        if epoch < warmup:
            for pg in optimizer.param_groups: pg['lr'] = args.lr * (epoch + 1) / warmup

        pbar = tqdm(train_loader, desc=f"V8B E{epoch+1}/{args.epochs}")
        for batch_data in pbar:
            if epoch < 20: ber = np.random.choice([0.0]*8 + [0.01, 0.02, 0.03, 0.05])
            elif epoch < 35: ber = np.random.uniform(0, 0.15)
            else: ber = np.random.uniform(0, 0.25)

            for lid in HOOK_LAYERS:
                h = yolo_model.model[lid]
                if hasattr(h, 'ber'): h.ber = ber
            if hasattr(yolo_model.model[P2_LAYER], 'ber'):
                yolo_model.model[P2_LAYER].ber = ber

            imgs = batch_data['img'].to(device).float() / 255.0
            preds = yolo_forward_with_hooks(yolo_model, imgs)
            loss_vec, loss_items = criterion(preds, batch_data)
            loss = loss_vec.sum()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 10.0)
            optimizer.step()

            epoch_loss += loss.item(); n_batches += 1
            if loss_items is not None and len(loss_items) >= 3:
                pbar.set_postfix({'L': f'{loss.item():.4f}', 'box': f'{loss_items[0]:.3f}', 'cls': f'{loss_items[1]:.3f}', 'ber': f'{ber:.2f}'})
            if n_batches >= args.max_batches: break

        if epoch >= warmup: scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  E{epoch+1:3d}/{args.epochs}  Loss={avg_loss:.6f}  LR={optimizer.param_groups[0]['lr']:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            extra_state = {k: m.state_dict() for k, m in extra_modules.items()}
            yolo_state = {f'layer_{i}': layer.state_dict() for i, layer in enumerate(yolo_model.model) if i > max(HOOK_LAYERS)}
            torch.save({'snn_state': snn.state_dict(), 'yolo_head_state': yolo_state, 'extra_modules': extra_state, 'epoch': epoch + 1, 'loss': avg_loss, 'config': cfg, 'option': 'B'}, save_path)

    print(f"\n✅ Best Loss: {best_loss:.6f}, saved: {save_path}")


# ############################################################################
# OPTION A: FULL 200ep MSE RETRAIN (Stage 1)
# ############################################################################

def train_option_a_stage1(args):
    """Train SNN encoder from scratch with best arch for 200 epochs MSE."""
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data import YOLODataset
    from torch.utils.data import DataLoader

    print("=" * 70)
    print("  V8 Option A Stage 1: Full 200-epoch MSE Training")
    print("  Architecture: MaxFormer HF + EMS encoder")
    print("=" * 70)

    snn = SpikeAdaptSC_Det_Multi(
        CHANNEL_SIZES, C_spike=36, T=8, target_rate=0.75
    ).to(device)
    cfg = {'channel_sizes': CHANNEL_SIZES, 'C_spike': 36, 'T': 8, 'target_rate': 0.75}
    print(f"  SNN params: {sum(p.numel() for p in snn.parameters()):,}")

    # Build hook model for feature extraction
    yolo = YOLO(BASELINE)
    yolo_model = yolo.model.to(device)
    yolo_model.eval()
    for p in yolo_model.parameters():
        p.requires_grad = False

    # Data
    data_dict = check_det_dataset('DOTAv1.yaml')
    train_dataset = YOLODataset(img_path=data_dict['train'], imgsz=640, augment=True, data=data_dict, task='obb')
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True, collate_fn=YOLODataset.collate_fn)

    optimizer = torch.optim.AdamW(snn.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - warmup, eta_min=1e-6)
    best_loss = float('inf')
    start_epoch = 0

    # Resume if exists
    if args.resume and os.path.exists(V8A_STAGE1):
        ck = torch.load(V8A_STAGE1, map_location=device, weights_only=False)
        snn.load_state_dict(ck['snn_state'])
        start_epoch = ck.get('epoch', 0)
        best_loss = ck.get('loss', float('inf'))
        print(f"  Resumed from epoch {start_epoch}, loss={best_loss:.6f}")

    for epoch in range(start_epoch, args.epochs):
        snn.train()
        epoch_loss, n_batches = 0, 0
        if epoch < warmup:
            for pg in optimizer.param_groups: pg['lr'] = 1e-3 * (epoch + 1) / warmup

        # BER curriculum
        if epoch < 80: ber = np.random.choice([0.0]*8 + [0.01, 0.02, 0.03, 0.05])
        elif epoch < 150: ber = np.random.uniform(0, 0.15)
        else: ber = np.random.uniform(0, 0.25)

        pbar = tqdm(train_loader, desc=f"V8A-S1 E{epoch+1}/{args.epochs}")
        for batch_data in pbar:
            imgs = batch_data['img'].to(device).float() / 255.0
            
            # Extract YOLO features at hook points
            with torch.no_grad():
                save = {}
                x = imgs
                for i, layer in enumerate(yolo_model.model):
                    if hasattr(layer, 'f') and isinstance(layer.f, list):
                        x = [save[j] if j != -1 else x for j in layer.f]
                    elif hasattr(layer, 'f') and layer.f != -1:
                        x = save[layer.f]
                    x = layer(x) if not isinstance(x, list) else layer(x)
                    save[i] = x
                    if i >= max(HOOK_LAYERS):
                        break
                feats = [save[lid] for lid in HOOK_LAYERS]

            # Forward through SNN
            recons, infos = snn(feats, ber=ber)
            
            # MSE loss
            loss = sum(F.mse_loss(r, f.detach()) for r, f in zip(recons, feats)) / len(feats)
            
            # Rate regularization
            rates = [info['mask'].mean() for info in infos]
            rate_loss = sum((r - 0.75)**2 for r in rates) / len(rates)
            loss = loss + 0.1 * rate_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(snn.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item(); n_batches += 1
            avg_rate = np.mean([r.item() for r in rates])
            pbar.set_postfix({'MSE': f'{loss.item():.5f}', 'r': f'{avg_rate:.2f}', 'ber': f'{ber:.2f}'})
            if n_batches >= args.max_batches: break

        if epoch >= warmup: scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  E{epoch+1:3d}/{args.epochs}  MSE={avg_loss:.6f}  LR={optimizer.param_groups[0]['lr']:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'snn_state': snn.state_dict(), 'epoch': epoch + 1, 'loss': avg_loss, 'config': cfg}, V8A_STAGE1)

        if (epoch + 1) % 50 == 0:
            torch.save({'snn_state': snn.state_dict(), 'epoch': epoch + 1, 'loss': avg_loss, 'config': cfg}, V8A_STAGE1.replace('.pth', f'_e{epoch+1}.pth'))

    print(f"\n✅ Best MSE: {best_loss:.6f}, saved: {V8A_STAGE1}")


def train_option_a_stage2(args):
    """Fine-tune with P2+SE on top of 200ep SNN (same as Option B but with better SNN)."""
    print("=" * 70)
    print("  V8 Option A Stage 2: Joint Fine-Tuning with P2+SE")
    print("=" * 70)

    ck = torch.load(V8A_STAGE1, map_location=device, weights_only=False)
    cfg = ck['config']
    snn = SpikeAdaptSC_Det_Multi(
        cfg['channel_sizes'], C_spike=cfg['C_spike'], T=cfg['T'],
        target_rate=cfg['target_rate']
    ).to(device)
    snn.load_state_dict(ck['snn_state'])
    snn.eval()
    for p in snn.parameters(): p.requires_grad = False
    print(f"  Loaded SNN from stage 1 (epoch {ck['epoch']}, MSE={ck['loss']:.6f})")

    # Reuse Option B training logic but with this better SNN
    args_b = argparse.Namespace(**vars(args))
    # Monkey-patch to use V8A_SAVE
    global V8B_SAVE
    orig = V8B_SAVE
    V8B_SAVE = V8A_SAVE
    
    # Directly call training with the better SNN
    # (copy-paste of train_option_b internals with different SNN)
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data import YOLODataset
    from torch.utils.data import DataLoader

    yolo, extra_modules = build_v8_model(snn, ber=0.0, for_training=True)
    yolo_model = yolo.model.to(device)

    trainable_params = []
    for i, layer in enumerate(yolo_model.model):
        if i <= max(HOOK_LAYERS) and i not in [P2_LAYER] + HOOK_LAYERS:
            for param in layer.parameters(): param.requires_grad = False
        elif i in HOOK_LAYERS or i == P2_LAYER:
            for name, param in layer.named_parameters():
                if 'snn' in name and 'p2_snn' not in name: param.requires_grad = False
                elif any(k in name for k in ['enhancer', 'bridge', 'p2_snn']):
                    param.requires_grad = True; trainable_params.append(param)
                else: param.requires_grad = False
        else:
            for param in layer.parameters():
                param.requires_grad = True; trainable_params.append(param)
    for mod in extra_modules.values():
        for p in mod.parameters():
            if p.requires_grad: trainable_params.append(p)
    seen = set()
    trainable_params = [p for p in trainable_params if (pid := id(p)) not in seen and not seen.add(pid)]
    print(f"  Trainable: {sum(p.numel() for p in trainable_params):,}")

    from types import SimpleNamespace
    ea = yolo_model.args if isinstance(yolo_model.args, dict) else vars(yolo_model.args)
    lh = {'box': 7.5, 'cls': 0.5, 'dfl': 1.5, 'angle': 1.0, 'overlap_mask': True}
    lh.update(ea)
    yolo_model.args = SimpleNamespace(**lh)
    criterion = yolo_model.init_criterion()

    data_dict = check_det_dataset('DOTAv1.yaml')
    ds = YOLODataset(img_path=data_dict['train'], imgsz=640, augment=True, data=data_dict, task='obb')
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True, collate_fn=YOLODataset.collate_fn)

    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    warmup = 3
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs - warmup, eta_min=1e-6)
    best_loss = float('inf')

    for epoch in range(args.epochs):
        yolo_model.train()
        for lid in HOOK_LAYERS:
            h = yolo_model.model[lid]
            if hasattr(h, 'snn'): h.snn.eval()
        epoch_loss, nb = 0, 0
        if epoch < warmup:
            for pg in opt.param_groups: pg['lr'] = args.lr * (epoch+1)/warmup
        pbar = tqdm(dl, desc=f"V8A-S2 E{epoch+1}/{args.epochs}")
        for bd in pbar:
            if epoch < 20: ber = np.random.choice([0.0]*8+[0.01,0.02,0.03,0.05])
            elif epoch < 35: ber = np.random.uniform(0,0.15)
            else: ber = np.random.uniform(0,0.25)
            for lid in HOOK_LAYERS:
                h = yolo_model.model[lid]
                if hasattr(h,'ber'): h.ber=ber
            if hasattr(yolo_model.model[P2_LAYER],'ber'):
                yolo_model.model[P2_LAYER].ber=ber
            imgs = bd['img'].to(device).float()/255.0
            preds = yolo_forward_with_hooks(yolo_model,imgs)
            lv,li = criterion(preds,bd); loss=lv.sum()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params,10.0); opt.step()
            epoch_loss+=loss.item(); nb+=1
            if li is not None and len(li)>=3:
                pbar.set_postfix({'L':f'{loss.item():.4f}','box':f'{li[0]:.3f}','cls':f'{li[1]:.3f}','ber':f'{ber:.2f}'})
            if nb>=args.max_batches: break
        if epoch>=warmup: sched.step()
        avg=epoch_loss/max(nb,1)
        if (epoch+1)%5==0 or epoch==0:
            print(f"  E{epoch+1:3d}/{args.epochs}  Loss={avg:.6f}")
        if avg<best_loss:
            best_loss=avg
            es={k:m.state_dict() for k,m in extra_modules.items()}
            ys={f'layer_{i}':l.state_dict() for i,l in enumerate(yolo_model.model) if i>max(HOOK_LAYERS)}
            torch.save({'snn_state':snn.state_dict(),'yolo_head_state':ys,'extra_modules':es,'epoch':epoch+1,'loss':avg,'config':cfg,'option':'A'},V8A_SAVE)
    
    V8B_SAVE = orig
    print(f"\n✅ Best Loss: {best_loss:.6f}, saved: {V8A_SAVE}")


# ############################################################################
# EVALUATION
# ############################################################################

def eval_v8(args):
    from ultralytics import YOLO
    
    save_path = V8B_SAVE if args.option == 'B' else V8A_SAVE
    print("=" * 70)
    print(f"  V8 Option {args.option}: mAP Evaluation")
    print("=" * 70)

    ck = torch.load(save_path, map_location=device, weights_only=False)
    cfg = ck['config']
    snn = SpikeAdaptSC_Det_Multi(
        cfg['channel_sizes'], C_spike=cfg['C_spike'], T=cfg['T'],
        target_rate=cfg['target_rate']
    ).to(device)
    
    # Handle V6d-v2 weight mismatch for Option B
    missing, _ = snn.load_state_dict(ck['snn_state'], strict=False)
    if missing:
        for lev in snn.levels:
            enc = lev.encoder
            if hasattr(enc, 'hf_gate'):
                nn.init.constant_(enc.hf_gate[0].bias, -10.0)
    snn.eval()
    print(f"  Loaded epoch {ck['epoch']}, loss={ck['loss']:.6f}")

    yolo_base = YOLO(BASELINE)
    base_res = yolo_base.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0, verbose=False, plots=False)
    base_map = float(base_res.results_dict.get('metrics/mAP50(B)', 0))
    print(f"  Baseline: mAP@50={base_map:.4f}")

    bers = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    key = f'v8_{args.option.lower()}'
    results = {'baseline': base_map, key: {}}

    for ber in bers:
        print(f"\n  BER={ber:.2f}...")
        yolo, extra = build_v8_model(snn, ber=ber)
        es = ck.get('extra_modules', {})
        for k, s in es.items():
            if k in extra:
                try: extra[k].load_state_dict(s)
                except: pass
        hs = ck.get('yolo_head_state', {})
        for lk, s in hs.items():
            i = int(lk.split('_')[1])
            if i < len(yolo.model.model):
                try: yolo.model.model[i].load_state_dict(s)
                except: pass
        res = yolo.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0, verbose=False, plots=False)
        m50 = float(res.results_dict.get('metrics/mAP50(B)', 0))
        results[key][str(ber)] = {'mAP50': m50}
        print(f"    mAP@50={m50:.4f} ({m50-base_map:+.4f})")

    print("\n" + "=" * 70)
    print(f"  {'BER':>6s}  {'mAP@50':>8s}  {'Δ':>8s}")
    print(f"  {'none':>6s}  {base_map:8.4f}  {'—':>8s}")
    for ber in bers:
        r = results[key][str(ber)]
        print(f"  {ber:6.2f}  {r['mAP50']:8.4f}  {r['mAP50']-base_map:+8.4f}")

    os.makedirs('eval', exist_ok=True)
    with open(f'eval/dota_v8_{args.option.lower()}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved: eval/dota_v8_{args.option.lower()}_results.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=str, required=True, choices=['A', 'B'])
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'eval', 'stage1', 'stage2'])
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max-batches', type=int, default=300)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    if args.option == 'B':
        if args.phase == 'train': train_option_b(args)
        elif args.phase == 'eval': eval_v8(args)
    elif args.option == 'A':
        if args.phase == 'stage1': train_option_a_stage1(args)
        elif args.phase in ['stage2', 'train']: train_option_a_stage2(args)
        elif args.phase == 'eval': eval_v8(args)
