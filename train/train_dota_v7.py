#!/usr/bin/env python3
"""SpikeAdapt-SC V7 — Dense Scene Improvements (3 approaches).

Builds incrementally on V6d-v2d (Max-Former) weights, adding:
  Approach 3: FeatureEnhancer — PixelShuffle post-SNN enhancement
  Approach 4: ContentAwareDecoder — CARAFE-inspired content-aware blending
  Approach 5: P2 High-Res Hook — Process layer 2 at 160×160 for small objects

Literature basis:
  - CARAFE (ICCV 2019), DySample (ICCV 2023), FeatUp (ICLR 2024)
  - SOD-YOLO, CPDD-YOLOv8, BFRI-YOLO (2024)
  - ESPCN Sub-pixel Conv (CVPR 2016)

Usage:
  python train/train_dota_v7.py --approach 3 --phase train   # Feature Enhancer
  python train/train_dota_v7.py --approach 4 --phase train   # CARAFE decoder
  python train/train_dota_v7.py --approach 5 --phase train   # P2 head
  python train/train_dota_v7.py --approach 5 --phase eval    # Eval best
  python train/train_dota_v7.py --approach 5 --phase viz     # Visualize
"""

import os, sys, json, argparse, random, copy, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASELINE = 'runs/obb/runs/dota/yolo26n_obb_baseline/weights/best.pt'

# Approach 3 & 4 build on Max-Former weights
MAXFORMER_STAGE1 = 'runs/yolo26_snn_v6_maxformer.pth'
MAXFORMER_STAGE2 = 'runs/yolo26_snn_v6d_maxformer.pth'

HOOK_LAYERS = [4, 6, 10]     # P3, P4, P5
CHANNEL_SIZES = [128, 128, 256]

# Import core SNN components from v6d2d
from train.train_dota_v6d2d import (
    SpikeAdaptSC_Det_Multi, SNN_Hook, SpikeAdaptSC_Det,
    SNNEncoder, NoiseAwareScorerDet, DetMasker, DetDecoder,
    BSC_Channel, AdaptiveLIF, MPBN, SpikeFunction,
    yolo_forward_with_hooks,
)


# ############################################################################
# APPROACH 3: FeatureEnhancer (PixelShuffle post-SNN)
# ############################################################################

class FeatureEnhancer(nn.Module):
    """Lightweight post-SNN feature enhancement via DWConv + SE attention.
    
    Uses depthwise separable convolution for spatial refinement with
    squeeze-and-excite channel attention. Initialized near-identity 
    to avoid disrupting pre-trained features.
    
    Ref: MobileNetV2, SE-Net, FeatUp (ICLR 2024)
    """
    def __init__(self, C):
        super().__init__()
        # Spatial refinement (depthwise conv — very lightweight)
        self.spatial = nn.Sequential(
            nn.Conv2d(C, C, 3, 1, 1, groups=C),  # DWConv
            nn.BatchNorm2d(C), nn.SiLU(),
            nn.Conv2d(C, C, 1),                   # Pointwise
            nn.BatchNorm2d(C))
        # Channel attention (SE-style)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C, max(C // 4, 8), 1), nn.SiLU(),
            nn.Conv2d(max(C // 4, 8), C, 1), nn.Sigmoid())
        # Initialize near-identity: gate starts at 0 → output ≈ input
        self.alpha = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        refined = self.spatial(x)
        att = self.se(refined)
        enhanced = refined * att
        return x + torch.sigmoid(self.alpha) * enhanced  # Starts as identity


class SNN_Hook_Enhanced(nn.Module):
    """SNN Hook with post-decoding feature enhancement."""
    def __init__(self, original_layer, snn_level, ber=0.0, enhancer=None):
        super().__init__()
        self.original_layer = original_layer
        self.snn = snn_level
        self.ber = ber
        self.enhancer = enhancer
        # Copy YOLO metadata
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


# ############################################################################
# APPROACH 4: ContentAwareDecoder (CARAFE-inspired)
# ############################################################################

class ContentAwareBlend(nn.Module):
    """CARAFE-inspired content-aware feature reassembly.
    
    Replaces the static blend convolution in DetDecoder with dynamic
    content-aware kernels that adapt to local feature patterns.
    
    Ref: CARAFE (ICCV 2019), DySample (ICCV 2023)
    """
    def __init__(self, C, k_up=5):
        super().__init__()
        self.k_up = k_up
        # Predict reassembly kernel weights from content
        self.kernel_enc = nn.Sequential(
            nn.Conv2d(C, C, 3, 1, 1, groups=min(C, 32)),
            nn.BatchNorm2d(C), nn.SiLU(),
            nn.Conv2d(C, k_up * k_up, 1),
            nn.Softmax(dim=1))  # Normalized kernel weights
        # Standard blend as fallback residual path
        self.blend_static = nn.Sequential(
            nn.Conv2d(C, C, 3, 1, 1),
            nn.BatchNorm2d(C), nn.SiLU())
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C, 1, 1),
            nn.Sigmoid())
    
    def forward(self, x):
        # Static path (original)
        static = self.blend_static(x)
        
        # Content-aware path (CARAFE-style)
        kernels = self.kernel_enc(x)  # [B, k*k, H, W]
        carafe = self._apply_carafe(x, kernels)
        
        # Blend static and CARAFE with learned gate
        g = self.gate(x)
        return g * carafe + (1 - g) * static
    
    def _apply_carafe(self, x, kernels):
        B, C, H, W = x.shape
        k = self.k_up
        pad = k // 2
        x_pad = F.pad(x, [pad]*4, mode='reflect')
        # Unfold to get local patches [B, C, H, W, k, k]
        patches = x_pad.unfold(2, k, 1).unfold(3, k, 1)
        patches = patches.contiguous().view(B, C, H, W, k*k)
        # Apply kernels: [B, k*k, H, W] → broadcast over C
        kernels = kernels.unsqueeze(1)  # [B, 1, k*k, H, W]
        kernels = kernels.permute(0, 1, 4, 3, 2)  # [B, 1, W, H, k*k]
        out = (patches * kernels.permute(0, 1, 3, 2, 4)).sum(dim=-1)
        return out


class ContentAwareDetDecoder(nn.Module):
    """DetDecoder with CARAFE-inspired content-aware blending."""
    def __init__(self, C_out, C_spike=36, T=8):
        super().__init__()
        self.T = T; C_mem = C_spike // 2
        self.spike_tw = nn.Parameter(torch.ones(T) / T)
        self.mem_tw = nn.Parameter(torch.ones(T) / T)
        self.spike_dec = nn.Sequential(
            nn.Conv2d(C_spike, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out), nn.SiLU())
        self.mem_dec = nn.Sequential(
            nn.Conv2d(C_mem, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out), nn.SiLU())
        self.gate = nn.Sequential(nn.Conv2d(C_out * 2, C_out, 1), nn.Sigmoid())
        # CARAFE blend instead of static conv
        self.blend = ContentAwareBlend(C_out, k_up=5)
        
    def forward(self, recv_spikes, recv_mems):
        sw = torch.softmax(self.spike_tw, 0)
        mw = torch.softmax(self.mem_tw, 0)
        spike_avg = sum(sw[t] * recv_spikes[t] for t in range(min(len(recv_spikes), self.T)))
        mem_avg = sum(mw[t] * recv_mems[t] for t in range(min(len(recv_mems), self.T)))
        spike_feat = self.spike_dec(spike_avg)
        mem_feat = self.mem_dec(mem_avg)
        gate = self.gate(torch.cat([spike_feat, mem_feat], dim=1))
        fused = gate * spike_feat + (1 - gate) * mem_feat
        return self.blend(fused)


# ############################################################################
# APPROACH 5: P2 High-Res Hook
# ############################################################################

# YOLO26n layer 2: C3k2 block, output = 64 channels at 160×160
# We hook this for high-res small object features, fused into P3 via P2Bridge

P2_LAYER = 2
P2_CHANNELS = 64  # YOLO26n width scale 0.25: 256*0.25=64

class P2Bridge(nn.Module):
    """Fuse P2 high-res SNN features into P3 level via learned downsampling.
    
    Ref: SOD-YOLO (2024), CPDD-YOLOv8 (TGRS 2024)
    """
    def __init__(self, C_p2=64, C_p3=128):
        super().__init__()
        self.bridge = nn.Sequential(
            nn.Conv2d(C_p2, C_p3, 3, stride=2, padding=1),  # Downsample 2×
            nn.BatchNorm2d(C_p3), nn.SiLU())
        self.gate = nn.Sequential(
            nn.Conv2d(C_p3 * 2, C_p3, 1), 
            nn.Sigmoid())
    
    def forward(self, p2_feat, p3_feat):
        p2_down = self.bridge(p2_feat)
        gate = self.gate(torch.cat([p2_down, p3_feat], dim=1))
        return p3_feat + gate * p2_down


class SNN_Hook_P2(nn.Module):
    """Hook for P2 layer that stores output for later P2Bridge fusion."""
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
        self.last_p2_recon = None  # Stored for P2Bridge fusion

    def forward(self, x):
        out = self.original_layer(x)
        recon, info = self.snn(out, noise_param=self.ber)
        if self.enhancer is not None:
            recon = self.enhancer(recon)
        self.last_info = info
        self.last_p2_recon = recon  # Store for later bridge fusion
        return out  # Pass through original (P2 not a detection head)


class SNN_Hook_P3WithBridge(nn.Module):
    """P3 hook that fuses P2 features via P2Bridge."""
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
        # Fuse P2 features
        if self.p2_hook.last_p2_recon is not None:
            recon = self.p2_bridge(self.p2_hook.last_p2_recon, recon)
        self.last_info = info
        return recon


# ############################################################################
# SAVE/LOAD PATHS
# ############################################################################

SAVE_PATHS = {
    3: 'runs/yolo26_snn_v7_approach3.pth',
    4: 'runs/yolo26_snn_v7_approach4.pth',
    5: 'runs/yolo26_snn_v7_approach5.pth',
}


def get_save_path(approach):
    return SAVE_PATHS[approach]


# ############################################################################
# BUILD MODEL WITH ENHANCEMENTS
# ############################################################################

def build_enhanced_model(approach, snn, ber=0.0, for_training=False):
    """Build YOLO model with SNN hooks + selected enhancement approach."""
    from ultralytics import YOLO
    
    yolo = YOLO(BASELINE)
    yolo.model.to(device)
    yolo.model.fuse = lambda *a, **kw: yolo.model
    
    extra_modules = {}  # Track extra trainable modules
    
    if approach == 3:
        # Approach 3: FeatureEnhancer after each SNN hook
        enhancers = nn.ModuleDict()
        for idx, (lid, lev) in enumerate(zip(HOOK_LAYERS, snn.levels)):
            lev_copy = copy.deepcopy(lev) if not for_training else lev
            if not for_training:
                lev_copy.eval()
            enhancer = FeatureEnhancer(CHANNEL_SIZES[idx]).to(device)
            enhancers[f'e{idx}'] = enhancer
            yolo.model.model[lid] = SNN_Hook_Enhanced(
                yolo.model.model[lid], lev_copy, ber=ber, enhancer=enhancer
            ).to(device)
        extra_modules['enhancers'] = enhancers
        
    elif approach == 4:
        # Approach 4: FeatureEnhancer + CARAFE content-aware blend (stacked)
        enhancers = nn.ModuleDict()
        for idx, (lid, lev) in enumerate(zip(HOOK_LAYERS, snn.levels)):
            lev_copy = copy.deepcopy(lev) if not for_training else lev
            if not for_training:
                lev_copy.eval()
            # Stack: lightweight spatial enhancer → CARAFE content-aware blend
            C = CHANNEL_SIZES[idx]
            enhancer = nn.Sequential(
                FeatureEnhancer(C),
                ContentAwareBlend(C, k_up=5),
            ).to(device)
            enhancers[f'e{idx}'] = enhancer
            yolo.model.model[lid] = SNN_Hook_Enhanced(
                yolo.model.model[lid], lev_copy, ber=ber, enhancer=enhancer
            ).to(device)
        extra_modules['enhancers'] = enhancers
        
    elif approach == 5:
        # Approach 5: P2 hook + P2Bridge + FeatureEnhancer on all levels
        # P2 SNN level
        p2_snn = SpikeAdaptSC_Det(P2_CHANNELS, C_spike=36, T=8, target_rate=0.75).to(device)
        p2_enhancer = FeatureEnhancer(P2_CHANNELS).to(device)
        
        p2_hook = SNN_Hook_P2(
            yolo.model.model[P2_LAYER], p2_snn, ber=ber, enhancer=p2_enhancer
        ).to(device)
        yolo.model.model[P2_LAYER] = p2_hook
        
        # P2Bridge for P3 fusion
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
        
        # P4, P5 hooks with enhancers (standard)
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
# TRAINING
# ############################################################################

def train_approach(args):
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data import YOLODataset
    from torch.utils.data import DataLoader

    approach = args.approach
    save_path = get_save_path(approach)
    
    print("=" * 70)
    print(f"  V7 Approach {approach}: Training")
    print("=" * 70)

    # Load Max-Former SNN weights
    ck_snn = torch.load(MAXFORMER_STAGE1, map_location=device, weights_only=False)
    cfg = ck_snn['config']
    snn = SpikeAdaptSC_Det_Multi(
        cfg['channel_sizes'], C_spike=cfg['C_spike'], T=cfg['T'],
        target_rate=cfg['target_rate']
    ).to(device)
    snn.load_state_dict(ck_snn['snn_state'])
    snn.eval()
    for param in snn.parameters():
        param.requires_grad = False
    print(f"  Loaded Max-Former SNN (epoch {ck_snn['epoch']})")

    # Build enhanced model
    yolo, extra_modules = build_enhanced_model(approach, snn, ber=0.0, for_training=True)
    yolo_model = yolo.model.to(device)
    
    # Load fine-tuned head weights from Max-Former Stage 2
    if os.path.exists(MAXFORMER_STAGE2):
        ck_head = torch.load(MAXFORMER_STAGE2, map_location=device, weights_only=False)
        head_state = ck_head.get('yolo_head_state', {})
        for layer_key, state in head_state.items():
            i = int(layer_key.split('_')[1])
            if i < len(yolo_model.model):
                try:
                    yolo_model.model[i].load_state_dict(state)
                except:
                    pass  # Skip if shape mismatch (new modules)
        print(f"  Loaded Max-Former Stage 2 head weights")

    # Collect trainable params: neck+head + extra modules
    trainable_params = []
    
    # Unfreeze neck + head
    for i, layer in enumerate(yolo_model.model):
        if i <= max(HOOK_LAYERS) and i not in [P2_LAYER] + HOOK_LAYERS:
            for param in layer.parameters():
                param.requires_grad = False
        elif i in HOOK_LAYERS or i == P2_LAYER:
            # Only train enhancers/bridges, not SNN
            for name, param in layer.named_parameters():
                if 'snn' in name:
                    param.requires_grad = False
                elif 'enhancer' in name or 'bridge' in name or 'p2_bridge' in name:
                    param.requires_grad = True
                    trainable_params.append(param)
                else:
                    param.requires_grad = False
        else:
            for param in layer.parameters():
                param.requires_grad = True
                trainable_params.append(param)
    
    # Add extra module params
    for key, mod in extra_modules.items():
        for param in mod.parameters():
            if param.requires_grad:
                trainable_params.append(param)
    
    # Deduplicate
    seen = set()
    unique_params = []
    for p in trainable_params:
        pid = id(p)
        if pid not in seen:
            seen.add(pid)
            unique_params.append(p)
    trainable_params = unique_params
    
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in yolo_model.parameters())
    n_extra = sum(p.numel() for m in extra_modules.values() for p in m.parameters())
    print(f"  Total params: {n_total:,}")
    print(f"  Extra module params: {n_extra:,}")
    print(f"  Trainable: {n_trainable:,}")

    # Detection loss
    from types import SimpleNamespace
    existing_args = yolo_model.args if isinstance(yolo_model.args, dict) else vars(yolo_model.args)
    loss_hyp = {'box': 7.5, 'cls': 0.5, 'dfl': 1.5, 'angle': 1.0, 'overlap_mask': True}
    loss_hyp.update(existing_args)
    for k, v in [('box', 7.5), ('cls', 0.5), ('dfl', 1.5), ('angle', 1.0)]:
        if k not in loss_hyp:
            loss_hyp[k] = v
    yolo_model.args = SimpleNamespace(**loss_hyp)
    criterion = yolo_model.init_criterion()

    # Data loader
    data_dict = check_det_dataset('DOTAv1.yaml')
    train_dataset = YOLODataset(
        img_path=data_dict['train'], imgsz=640,
        augment=True, data=data_dict, task='obb',
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=YOLODataset.collate_fn,
    )

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    warmup_epochs = 3
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6)

    best_loss = float('inf')

    for epoch in range(args.epochs):
        yolo_model.train()
        # Keep SNN in eval mode
        for lid in HOOK_LAYERS:
            hook = yolo_model.model[lid]
            if hasattr(hook, 'snn'):
                hook.snn.eval()
        if approach == 5 and hasattr(yolo_model.model[P2_LAYER], 'snn'):
            yolo_model.model[P2_LAYER].snn.eval()
            
        epoch_loss = 0
        n_batches = 0

        if epoch < warmup_epochs:
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr * (epoch + 1) / warmup_epochs

        pbar = tqdm(train_loader, desc=f"V7-A{approach} E{epoch+1}/{args.epochs}")

        for batch_data in pbar:
            # BER curriculum: clean-first
            if epoch < 20:
                ber = np.random.choice([0.0]*8 + [0.01, 0.02, 0.03, 0.05])
            elif epoch < 35:
                ber = np.random.uniform(0, 0.15)
            else:
                ber = np.random.uniform(0, 0.25)

            # Set BER on all hooks
            for lid in HOOK_LAYERS:
                hook = yolo_model.model[lid]
                if hasattr(hook, 'ber'):
                    hook.ber = ber
            if approach == 5 and hasattr(yolo_model.model[P2_LAYER], 'ber'):
                yolo_model.model[P2_LAYER].ber = ber

            imgs = batch_data['img'].to(device).float() / 255.0
            preds = yolo_forward_with_hooks(yolo_model, imgs)
            loss_vec, loss_items = criterion(preds, batch_data)
            loss = loss_vec.sum()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 10.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            if loss_items is not None and len(loss_items) >= 3:
                pbar.set_postfix({
                    'L': f'{loss.item():.4f}',
                    'box': f'{loss_items[0]:.3f}',
                    'cls': f'{loss_items[1]:.3f}',
                    'ber': f'{ber:.2f}',
                })
            if n_batches >= args.max_batches:
                break

        if epoch >= warmup_epochs:
            scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  E{epoch+1:3d}/{args.epochs}  Loss={avg_loss:.6f}  LR={optimizer.param_groups[0]['lr']:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save everything
            extra_state = {}
            for key, mod in extra_modules.items():
                extra_state[key] = mod.state_dict()
            
            yolo_state = {}
            for i, layer in enumerate(yolo_model.model):
                if i > max(HOOK_LAYERS):
                    yolo_state[f'layer_{i}'] = layer.state_dict()
            
            torch.save({
                'snn_state': snn.state_dict(),
                'yolo_head_state': yolo_state,
                'extra_modules': extra_state,
                'approach': approach,
                'epoch': epoch + 1,
                'loss': avg_loss,
                'config': cfg,
            }, save_path)

    print(f"\n✅ Best Loss: {best_loss:.6f}, saved: {save_path}")


# ############################################################################
# EVALUATION
# ############################################################################

def eval_approach(args):
    from ultralytics import YOLO
    
    approach = args.approach
    save_path = get_save_path(approach)
    
    print("=" * 70)
    print(f"  V7 Approach {approach}: mAP Evaluation")
    print("=" * 70)

    ck = torch.load(save_path, map_location=device, weights_only=False)
    cfg = ck['config']
    snn = SpikeAdaptSC_Det_Multi(
        cfg['channel_sizes'], C_spike=cfg['C_spike'], T=cfg['T'],
        target_rate=cfg['target_rate']
    ).to(device)
    snn.load_state_dict(ck['snn_state'])
    snn.eval()
    print(f"  Loaded epoch {ck['epoch']}, loss={ck['loss']:.6f}")

    # Baseline
    yolo_base = YOLO(BASELINE)
    base_res = yolo_base.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0, verbose=False, plots=False)
    base_map = float(base_res.results_dict.get('metrics/mAP50(B)', 0))
    print(f"  Baseline: mAP@50={base_map:.4f}")

    bers = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    results = {'baseline': base_map, f'v7_a{approach}': {}}

    for ber in bers:
        print(f"\n  BER={ber:.2f}...")
        yolo, extra_modules = build_enhanced_model(approach, snn, ber=ber)
        
        # Load extra module weights
        extra_state = ck.get('extra_modules', {})
        for key, state in extra_state.items():
            if key in extra_modules:
                try:
                    extra_modules[key].load_state_dict(state)
                except:
                    print(f"    Warning: Could not load {key}")
        
        # Load head weights
        head_state = ck.get('yolo_head_state', {})
        for layer_key, state in head_state.items():
            i = int(layer_key.split('_')[1])
            if i < len(yolo.model.model):
                try:
                    yolo.model.model[i].load_state_dict(state)
                except:
                    pass
        
        res = yolo.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0, verbose=False, plots=False)
        m50 = float(res.results_dict.get('metrics/mAP50(B)', 0))
        delta = m50 - base_map
        results[f'v7_a{approach}'][str(ber)] = {'mAP50': m50}
        print(f"    mAP@50={m50:.4f} ({delta:+.4f})")

    print("\n" + "=" * 70)
    print(f"  {'BER':>6s}  {'mAP@50':>8s}  {'Δ':>8s}")
    print(f"  {'none':>6s}  {base_map:8.4f}  {'—':>8s}")
    for ber in bers:
        r = results[f'v7_a{approach}'][str(ber)]
        d = r['mAP50'] - base_map
        print(f"  {ber:6.2f}  {r['mAP50']:8.4f}  {d:+8.4f}")

    os.makedirs('eval', exist_ok=True)
    with open(f'eval/dota_v7_a{approach}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved: eval/dota_v7_a{approach}_results.json")


# ############################################################################
# VISUALIZATION
# ############################################################################

def viz_approach(args):
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    approach = args.approach
    save_path = get_save_path(approach)
    
    print("=" * 70)
    print(f"  V7 Approach {approach}: Visualization")
    print("=" * 70)

    ck = torch.load(save_path, map_location=device, weights_only=False)
    cfg = ck['config']
    snn = SpikeAdaptSC_Det_Multi(
        cfg['channel_sizes'], C_spike=cfg['C_spike'], T=cfg['T'],
        target_rate=cfg['target_rate']
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
        fig, axes = plt.subplots(1, n_cols, figsize=(4.5*n_cols, 4.5))
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
            yolo, extra_modules = build_enhanced_model(approach, snn, ber=ber)
            extra_state = ck.get('extra_modules', {})
            for key, state in extra_state.items():
                if key in extra_modules:
                    try: extra_modules[key].load_state_dict(state)
                    except: pass
            head_state = ck.get('yolo_head_state', {})
            for layer_key, state in head_state.items():
                i = int(layer_key.split('_')[1])
                if i < len(yolo.model.model):
                    try: yolo.model.model[i].load_state_dict(state)
                    except: pass
            
            r = yolo.predict(img_path, imgsz=640, device=0, verbose=False, conf=0.25)
            ax = axes[bi + 1]
            ax.imshow(r[0].plot()[:, :, ::-1])
            nb = len(r[0].obb) if r[0].obb is not None else 0
            ax.set_title(f'V7-A{approach}\nBER={ber:.2f} ({nb} det)', fontsize=10, fontweight='bold')
            ax.axis('off')

        plt.suptitle(f'V7 Approach {approach} on DOTA ({name})', fontsize=13, fontweight='bold')
        plt.tight_layout()
        out = f'paper/figures/dota_v7_a{approach}_viz_{img_idx+1}.png'
        plt.savefig(out, dpi=200, bbox_inches='tight')
        print(f"  Saved: {out}")
        plt.close()

    print(f"\n✅ {len(test_imgs)} visualizations saved!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--approach', type=int, required=True, choices=[3, 4, 5])
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'eval', 'viz'])
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max-batches', type=int, default=300)
    args = parser.parse_args()

    {'train': train_approach, 'eval': eval_approach, 'viz': viz_approach}[args.phase](args)
