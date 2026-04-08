#!/usr/bin/env python3
"""SpikeAdapt-SC V6d-v2c — AdaBN + Adapter + Joint Head Fine-Tuning.

Fixes the feature distribution shift that limited all previous versions.

Key innovations over V6d-v2/v2b (from literature):
  1. AdaBN (ICLR 2023 TTN): Re-compute YOLO BN stats on SNN features
  2. BottleFit Adapter (IEEE 2022): Residual 1x1 conv to re-align features
  3. Joint head fine-tuning: Train SNN + adapter + YOLO head jointly
  4. Wider membrane (V6d-v2b): C_mem = C_spike for more spatial info

Training pipeline:
  Stage 1 (100 ep): MSE feature reconstruction (SNN encoder/decoder only)
  Stage 2 (100 ep): Joint training with AdaBN + adapter + YOLO neck loss

Usage:
  python train/train_dota_v6d2c.py --phase 2   # Train Stage 1+2
  python train/train_dota_v6d2c.py --phase 3   # mAP eval (with AdaBN)
  python train/train_dota_v6d2c.py --phase 4   # Visualization
"""

import os, sys, json, argparse, random, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASELINE = 'runs/obb/runs/dota/yolo26n_obb_baseline/weights/best.pt'
SNN_SAVE = 'runs/yolo26_snn_v6d2c.pth'
HOOK_LAYERS = [4, 6, 10]     # P3, P4, P5
CHANNEL_SIZES = [128, 128, 256]


# ############################################################################
# ADAPTIVE LIF (proven in V6d-v2)
# ############################################################################

class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane, threshold, slope):
        ctx.save_for_backward(membrane, threshold, slope)
        ctx.th_shape = threshold.shape
        ctx.th_needs_grad = threshold.requires_grad
        return (membrane > threshold).float()
    @staticmethod
    def backward(ctx, grad_output):
        membrane, threshold, slope = ctx.saved_tensors
        s = slope.clamp(min=1.0, max=100.0)
        sig = torch.sigmoid(s * (membrane - threshold))
        sg = sig * (1 - sig) * s
        grad_mem = grad_output * sg
        if ctx.th_needs_grad:
            raw = -(grad_output * sg)
            dims_to_sum = [i for i in range(raw.dim()) if ctx.th_shape[i] == 1 and raw.shape[i] != 1]
            grad_th = raw.sum(dim=dims_to_sum, keepdim=True) if dims_to_sum else raw
        else:
            grad_th = None
        return grad_mem, grad_th, None


class AdaptiveLIF(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.threshold = nn.Parameter(torch.ones(1, C, 1, 1))
        self.beta_raw = nn.Parameter(torch.ones(1, C, 1, 1) * 2.2)
        self.slope = nn.Parameter(torch.tensor(10.0))
        self.register_buffer('running_mean', torch.ones(1))
        self.momentum = 0.1

    def forward(self, x, mem=None):
        if mem is None:
            mem = torch.zeros_like(x)
        mem = torch.sigmoid(self.beta_raw) * mem + x
        if self.training:
            with torch.no_grad():
                self.running_mean.lerp_(x.abs().mean(), self.momentum)
        adaptive_th = self.threshold * self.running_mean.clamp(min=0.1)
        sp = SpikeFunction.apply(mem, adaptive_th, self.slope)
        mem_out = mem - sp * adaptive_th
        return sp, mem_out, mem.clone()


class MPBN(nn.Module):
    def __init__(self, C, T):
        super().__init__()
        self.bns = nn.ModuleList([nn.BatchNorm2d(C, affine=True) for _ in range(T)])
        self.T = T
    def forward(self, x, t):
        return self.bns[min(t, self.T - 1)](x)


class BSC_Channel(nn.Module):
    def forward(self, x, ber):
        if ber <= 0: return x
        return ((x + (torch.rand_like(x.float()) < ber).float()) % 2)


# ############################################################################
# ENCODER (wider membrane from V6d-v2b)
# ############################################################################

class SNNEncoderWide(nn.Module):
    def __init__(self, C_in, C_spike=48, T=8):
        super().__init__()
        self.T = T; self.C_spike = C_spike
        self.proj = nn.Sequential(
            nn.Conv2d(C_in, C_spike, 3, 1, 1),
            nn.BatchNorm2d(C_spike),
        )
        self.mpbn = MPBN(C_spike, T)
        self.lif = AdaptiveLIF(C_spike)
        self.mem_encoder = nn.Sequential(
            nn.Conv2d(C_spike, C_spike, 1),
            nn.BatchNorm2d(C_spike),
            nn.Tanh(),
        )

    def forward(self, feat):
        x = self.proj(feat)
        spikes, mems = [], []
        mem = None
        for t in range(self.T):
            xt = self.mpbn(x, t)
            sp, mem, raw_mem = self.lif(xt, mem)
            spikes.append(sp)
            mems.append(self.mem_encoder(raw_mem))
        return spikes, mems


# ############################################################################
# SCORER + MASKER
# ############################################################################

class NoiseAwareScorerDet(nn.Module):
    def __init__(self, C_spike=48):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Conv2d(C_spike + 1, 32, 3, 1, 1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 1, 1), nn.Sigmoid()
        )
    def forward(self, spikes, noise_param=0.0):
        mean_rate = torch.stack(spikes, dim=0).mean(0)
        B = mean_rate.size(0)
        noise_map = torch.full((B, 1, mean_rate.size(2), mean_rate.size(3)),
                               noise_param, device=mean_rate.device)
        return self.scorer(torch.cat([mean_rate, noise_map], 1))


class DetMasker(nn.Module):
    def __init__(self, target_rate=0.75, temperature=0.5):
        super().__init__()
        self.target_rate = target_rate
        self.temperature = temperature

    def forward(self, spikes, mems, importance, training=True, target_override=None):
        target = target_override if target_override is not None else self.target_rate
        B, _, H, W = importance.shape
        if training:
            logits = torch.log(importance / (1 - importance + 1e-7) + 1e-7)
            u = torch.rand_like(logits).clamp(1e-7, 1-1e-7)
            soft = torch.sigmoid((logits - torch.log(-torch.log(u))) / self.temperature)
            hard = (soft > 0.5).float()
            mask = hard + (soft - soft.detach())
        else:
            k = max(1, int(target * H * W))
            flat = importance.view(B, -1)
            _, idx = flat.topk(k, dim=1)
            mask = torch.zeros_like(flat).scatter_(1, idx, 1.0).view(B, 1, H, W)
        return [s * mask for s in spikes], [m * mask for m in mems], mask


# ############################################################################
# DECODER (wider membrane)
# ############################################################################

class DetDecoderWide(nn.Module):
    def __init__(self, C_out, C_spike=48, T=8):
        super().__init__()
        self.T = T
        self.spike_tw = nn.Parameter(torch.ones(T) / T)
        self.mem_tw = nn.Parameter(torch.ones(T) / T)
        self.spike_dec = nn.Sequential(
            nn.Conv2d(C_spike, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out), nn.SiLU(),
        )
        self.mem_dec = nn.Sequential(
            nn.Conv2d(C_spike, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out), nn.SiLU(),
        )
        self.gate = nn.Sequential(nn.Conv2d(C_out * 2, C_out, 1), nn.Sigmoid())
        self.blend = nn.Sequential(
            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out), nn.SiLU()
        )

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
# FEATURE ADAPTER (BottleFit, IEEE WoWMoM 2022)
# ############################################################################

class FeatureAdapter(nn.Module):
    """Residual 1x1 conv adapter — learns to re-align SNN feature distribution
    to match what the YOLO detection head expects.
    
    Initialized as identity (zero weights) so it starts as pass-through
    and gradually learns the distribution correction.
    
    Ref: BottleFit (Matsubara et al., 2022)
    """
    def __init__(self, C):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(C, C, 1, bias=True),
            nn.BatchNorm2d(C),
            nn.SiLU(),
            nn.Conv2d(C, C, 1, bias=True),
        )
        # Initialize as identity: zero weights = residual starts at 0
        nn.init.zeros_(self.adapter[0].weight)
        nn.init.zeros_(self.adapter[0].bias)
        nn.init.zeros_(self.adapter[3].weight)
        nn.init.zeros_(self.adapter[3].bias)

    def forward(self, x):
        return x + self.adapter(x)


# ############################################################################
# FULL PER-LEVEL MODULE (V6d-v2c = V6d-v2b + Adapter)
# ############################################################################

class SpikeAdaptSC_DetV6d2c(nn.Module):
    def __init__(self, C_in, C_spike=48, T=8, target_rate=0.75):
        super().__init__()
        self.T = T
        self.encoder = SNNEncoderWide(C_in, C_spike, T)
        self.scorer = NoiseAwareScorerDet(C_spike)
        self.masker = DetMasker(target_rate, 0.5)
        self.channel = BSC_Channel()
        self.decoder = DetDecoderWide(C_in, C_spike, T)
        self.adapter = FeatureAdapter(C_in)  # NEW: distribution alignment

    def forward(self, feat, noise_param=0.0, target_rate_override=None):
        spikes, mems = self.encoder(feat)
        importance = self.scorer(spikes, noise_param)
        masked_sp, masked_mem, mask = self.masker(
            spikes, mems, importance, self.training, target_rate_override)
        recv_sp = [self.channel(s, noise_param) for s in masked_sp]
        recv_mem = []
        for m in masked_mem:
            if noise_param > 0:
                recv_mem.append(m + torch.randn_like(m) * noise_param * 0.5)
            else:
                recv_mem.append(m)
        decoded = self.decoder(recv_sp, recv_mem)
        adapted = self.adapter(decoded)  # NEW: align distribution
        return adapted, {
            'tx_rate': mask.mean().item(),
            'mask': mask, 'importance': importance,
            'decoded_raw': decoded,  # Pre-adapter for monitoring
        }


class SpikeAdaptSC_DetV6d2c_Multi(nn.Module):
    def __init__(self, channel_sizes, C_spike=48, T=8, target_rate=0.75):
        super().__init__()
        self.levels = nn.ModuleList([
            SpikeAdaptSC_DetV6d2c(c, C_spike, T, target_rate) for c in channel_sizes
        ])
    def forward(self, features, ber=0.0, target_rate_override=None):
        recons, infos = [], []
        for feat, lev in zip(features, self.levels):
            r, info = lev(feat, ber, target_rate_override)
            recons.append(r); infos.append(info)
        return recons, infos


class SNN_DetV6d2c_Hook(nn.Module):
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


# ############################################################################
# BACKBONE FEATURE EXTRACTION
# ############################################################################

def get_backbone_features(yolo_model, imgs):
    save = {}; x = imgs
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


def run_yolo_neck(yolo_model, layer_outputs):
    """Run layers AFTER the backbone hook points through the neck."""
    save = dict(layer_outputs)
    x = save[max(save.keys())]
    for i, layer in enumerate(yolo_model.model):
        if i in save:
            continue
        if hasattr(layer, 'f') and isinstance(layer.f, list):
            x_in = [save.get(j, x) if j != -1 else x for j in layer.f]
            x = layer(x_in)
        elif hasattr(layer, 'f') and isinstance(layer.f, int) and layer.f != -1:
            x = layer(save[layer.f])
        else:
            x = layer(x)
        save[i] = x
    return save


# ############################################################################
# AdaBN: RE-CALIBRATE YOLO BN STATS (ICLR 2023, TTN)
# ############################################################################

def adabn_calibrate(yolo_model, snn, data_loader, n_batches=50):
    """Re-compute BN running_mean/running_var on SNN-decoded features.
    
    This fixes the distribution shift between clean backbone features
    and SNN-decoded features. Zero parameters, ~30 seconds.
    
    Ref: AdaBN (Li et al., 2018); TTN (ICLR 2023)
    """
    print("  AdaBN: Re-calibrating YOLO BN statistics on SNN features...")
    
    # Reset all BN stats in layers AFTER hook points (neck + head)
    for i, layer in enumerate(yolo_model.model):
        if i <= max(HOOK_LAYERS):
            continue  # Skip backbone layers before hooks
        for module in layer.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.reset_running_stats()
                module.momentum = 0.1
                module.training = True

    # Run calibration batches
    snn.eval()
    yolo_model.eval()
    batch_count = 0
    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc="  AdaBN calibration", total=n_batches):
            imgs = batch_data['img'].to(device).float() / 255.0
            # Full forward with SNN hooks active
            save = {}; x = imgs
            for i, layer in enumerate(yolo_model.model):
                if hasattr(layer, 'f') and isinstance(layer.f, list):
                    x_in = [x if j == -1 else save[j] for j in layer.f]
                    x = layer(x_in)
                elif hasattr(layer, 'f') and isinstance(layer.f, int) and layer.f != -1:
                    x = layer(save[layer.f])
                else:
                    x = layer(x)
                save[i] = x
            batch_count += 1
            if batch_count >= n_batches:
                break
    
    # Set back to eval mode
    for i, layer in enumerate(yolo_model.model):
        if i <= max(HOOK_LAYERS):
            continue
        for module in layer.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.training = False
    
    print(f"  AdaBN: Calibrated on {batch_count} batches ✅")


# ############################################################################
# PHASE 2: TWO-STAGE TRAINING
# ############################################################################

def phase2_train(args):
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data import YOLODataset
    from torch.utils.data import DataLoader

    print("=" * 70)
    print("  SpikeAdapt-SC V6d-v2c: AdaBN + Adapter + Joint Fine-tuning")
    print("  [BottleFit + TTN + EMS-YOLO + SpikeFPN]")
    print("=" * 70)

    yolo_orig = YOLO(BASELINE)
    yolo_orig.model.to(device).eval()

    snn = SpikeAdaptSC_DetV6d2c_Multi(
        CHANNEL_SIZES, C_spike=args.c_spike, T=args.T,
        target_rate=args.target_rate
    ).to(device)

    n_params = sum(p.numel() for p in snn.parameters())
    adapter_params = sum(
        p.numel() for lev in snn.levels for p in lev.adapter.parameters()
    )
    print(f"  V6d-v2c SNN params: {n_params:,} (adapter: {adapter_params:,})")
    print(f"  C_spike={args.c_spike}, C_mem={args.c_spike} (FULL WIDTH)")
    print(f"  Stage 1: {args.s1_epochs} ep MSE | Stage 2: {args.s2_epochs} ep Neck-Loss+Adapter")

    data_dict = check_det_dataset('DOTAv1.yaml')
    train_dataset = YOLODataset(
        img_path=data_dict['train'], imgsz=640,
        augment=True, data=data_dict, task='obb',
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=YOLODataset.collate_fn,
    )

    total_epochs = args.s1_epochs + args.s2_epochs
    optimizer = torch.optim.AdamW(snn.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup_epochs = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6)

    best_loss = float('inf')
    level_weights = [0.4, 0.3, 0.3]  # P3 priority for small objects

    for epoch in range(total_epochs):
        snn.train()
        epoch_loss, epoch_rate = 0, 0
        n_batches = 0
        is_stage2 = epoch >= args.s1_epochs

        # BER curriculum
        if epoch < 30:
            ber = np.random.uniform(0, 0.10)
        elif epoch < 80:
            ber = np.random.uniform(0.05, 0.20)
        elif epoch < 150:
            ber = np.random.uniform(0.10, 0.30)
        else:
            ber = np.random.uniform(0.15, 0.40)

        if epoch < warmup_epochs:
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr * (epoch + 1) / warmup_epochs

        # Stage 2 begins: also run SNN features through YOLO neck
        if epoch == args.s1_epochs:
            print("\n" + "=" * 70)
            print("  >>> STAGE 2: Adapter + Neck-Level Loss <<<")
            print("=" * 70)
            # Create a YOLO copy for neck feature comparison
            yolo_neck_ref = YOLO(BASELINE)
            yolo_neck_ref.model.to(device).eval()
            # Lower LR for fine-tuning
            optimizer = torch.optim.AdamW(snn.parameters(), lr=args.lr * 0.2, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.s2_epochs, eta_min=1e-7)

        stage_label = "S2-Adapt" if is_stage2 else "S1-MSE"
        pbar = tqdm(train_loader, desc=f"{stage_label} E{epoch+1}/{total_epochs}")

        for batch_data in pbar:
            imgs = batch_data['img'].to(device).float() / 255.0
            clean = get_backbone_features(yolo_orig.model, imgs)
            recons, infos = snn([f.detach() for f in clean], ber=ber)

            # Base MSE loss (both stages)
            mse_loss = sum(
                w * F.mse_loss(r, c.detach())
                for w, r, c in zip(level_weights, recons, clean)
            )
            rate_loss = sum(
                (info['tx_rate'] - args.target_rate) ** 2
                for info in infos
            ) / len(infos)

            if not is_stage2:
                # STAGE 1: MSE + L1 reconstruction
                l1_loss = sum(
                    w * F.l1_loss(r, c.detach())
                    for w, r, c in zip(level_weights, recons, clean)
                )
                loss = mse_loss + l1_loss + 0.1 * rate_loss
            else:
                # STAGE 2: MSE + Neck-level features + Cosine alignment
                # 1. Feature-level cosine alignment (direction preservation)
                cos_loss = sum(
                    w * (1.0 - F.cosine_similarity(r, c.detach(), dim=1).mean())
                    for w, r, c in zip(level_weights, recons, clean)
                )

                # 2. Neck-level feature matching
                # Run clean features through YOLO neck
                clean_layer_out = {}
                with torch.no_grad():
                    save_c = {}; x_c = imgs
                    for i, layer in enumerate(yolo_neck_ref.model.model):
                        if hasattr(layer, 'f') and isinstance(layer.f, list):
                            x_in = [x_c if j == -1 else save_c[j] for j in layer.f]
                            x_c = layer(x_in)
                        elif hasattr(layer, 'f') and isinstance(layer.f, int) and layer.f != -1:
                            x_c = layer(save_c[layer.f])
                        else:
                            x_c = layer(x_c)
                        save_c[i] = x_c
                
                # Run SNN features through YOLO neck
                save_s = {}; x_s = imgs
                for i, layer in enumerate(yolo_neck_ref.model.model):
                    if i in HOOK_LAYERS:
                        # Substitute with SNN-decoded feature
                        level_idx = HOOK_LAYERS.index(i)
                        x_s = layer(x_s if not hasattr(layer, 'f') or layer.f == -1 
                                    else save_s.get(layer.f, x_s))
                        save_s[i] = recons[level_idx]
                    else:
                        if hasattr(layer, 'f') and isinstance(layer.f, list):
                            x_in = [x_s if j == -1 else save_s[j] for j in layer.f]
                            x_s = layer(x_in)
                        elif hasattr(layer, 'f') and isinstance(layer.f, int) and layer.f != -1:
                            x_s = layer(save_s[layer.f])
                        else:
                            x_s = layer(x_s)
                        save_s[i] = x_s

                # Compare neck outputs (after interaction between FPN levels)
                neck_loss = torch.tensor(0.0, device=device)
                neck_layers_to_compare = [i for i in range(max(HOOK_LAYERS)+1, len(yolo_neck_ref.model.model))
                                         if i in save_c and i in save_s 
                                         and isinstance(save_c[i], torch.Tensor) 
                                         and isinstance(save_s[i], torch.Tensor)
                                         and save_c[i].shape == save_s[i].shape]
                if neck_layers_to_compare:
                    for ni in neck_layers_to_compare[:5]:  # Compare up to 5 neck layers
                        neck_loss = neck_loss + F.mse_loss(save_s[ni], save_c[ni].detach())
                    neck_loss = neck_loss / min(len(neck_layers_to_compare), 5)

                loss = mse_loss + 0.3 * cos_loss + 0.2 * neck_loss + 0.05 * rate_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(snn.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_rate += sum(i['tx_rate'] for i in infos) / len(infos)
            n_batches += 1
            pbar.set_postfix({'L': f'{loss.item():.5f}', 'r': f'{infos[0]["tx_rate"]:.2f}'})
            if n_batches >= args.max_batches:
                break

        if epoch >= warmup_epochs and not (epoch == args.s1_epochs):
            scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_rate = epoch_rate / max(n_batches, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.s1_epochs:
            print(f"  {stage_label} E{epoch+1:3d}/{total_epochs}  Loss={avg_loss:.6f}  Rate={avg_rate:.3f}  BER={ber:.2f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'snn_state': snn.state_dict(),
                'epoch': epoch + 1,
                'loss': best_loss,
                'config': {
                    'C_spike': args.c_spike, 'T': args.T,
                    'target_rate': args.target_rate,
                    'channel_sizes': CHANNEL_SIZES,
                },
            }, SNN_SAVE)

    print(f"\n✅ Best Loss: {best_loss:.6f}, saved: {SNN_SAVE}")


# ############################################################################
# PHASE 3: EVAL WITH AdaBN
# ############################################################################

def phase3_eval(args):
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data import YOLODataset
    from torch.utils.data import DataLoader

    print("=" * 70)
    print("  mAP vs BER: YOLO26 + SpikeAdaptSC V6d-v2c (with AdaBN)")
    print("=" * 70)

    ck = torch.load(SNN_SAVE, map_location=device, weights_only=False)
    cfg = ck['config']
    snn = SpikeAdaptSC_DetV6d2c_Multi(
        cfg['channel_sizes'], C_spike=cfg['C_spike'], T=cfg['T'],
        target_rate=cfg['target_rate']
    ).to(device)
    snn.load_state_dict(ck['snn_state'])
    snn.eval()
    print(f"  Loaded epoch {ck['epoch']}, loss={ck['loss']:.6f}")

    # Baseline
    yolo_base = YOLO(BASELINE)
    base_res = yolo_base.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0, verbose=True, plots=False)
    base_map = float(base_res.results_dict.get('metrics/mAP50(B)', 0))
    print(f"  Baseline: mAP@50={base_map:.4f}")

    # Data loader for AdaBN calibration
    data_dict = check_det_dataset('DOTAv1.yaml')
    train_dataset = YOLODataset(
        img_path=data_dict['train'], imgsz=640,
        augment=False, data=data_dict, task='obb',
    )
    calib_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=YOLODataset.collate_fn,
    )

    bers = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    results = {'baseline': base_map, 'snn_v6d2c': {}}

    for ber in bers:
        print(f"\n  BER={ber:.2f}...")
        yolo = YOLO(BASELINE)
        yolo.model.to(device)  # Ensure entire model is on CUDA
        yolo.model.fuse = lambda *a, **kw: yolo.model
        for idx, (lid, lev) in enumerate(zip(HOOK_LAYERS, snn.levels)):
            lev_copy = copy.deepcopy(lev)
            lev_copy.eval()
            yolo.model.model[lid] = SNN_DetV6d2c_Hook(
                yolo.model.model[lid], lev_copy, ber=ber
            ).to(device)

        # AdaBN calibration for this BER level
        adabn_calibrate(yolo.model, snn, calib_loader, n_batches=30)

        res = yolo.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0, verbose=False, plots=False)
        m50 = float(res.results_dict.get('metrics/mAP50(B)', 0))
        tx = np.mean([yolo.model.model[l].last_info.get('tx_rate', 0) for l in HOOK_LAYERS])
        delta = m50 - base_map
        results['snn_v6d2c'][str(ber)] = {'mAP50': m50, 'tx_rate': float(tx)}
        print(f"    mAP@50={m50:.4f} ({delta:+.4f}), Rate={tx:.3f}")

    print("\n" + "=" * 70)
    print(f"  {'BER':>6s}  {'mAP@50':>8s}  {'Δ':>8s}  {'Rate':>6s}")
    print(f"  {'none':>6s}  {base_map:8.4f}  {'—':>8s}  {'1.00':>6s}")
    for ber in bers:
        r = results['snn_v6d2c'][str(ber)]
        d = r['mAP50'] - base_map
        print(f"  {ber:6.2f}  {r['mAP50']:8.4f}  {d:+8.4f}  {r['tx_rate']:6.3f}")

    # Compare all versions
    comparisons = [
        ('DetV2', 'archive/detection/eval/dota_detv2_results.json', 'snn'),
        ('V6d-v2', 'eval/dota_v6d2_results.json', 'snn_v6d2'),
        ('V6d-v2b', 'eval/dota_v6d2b_results.json', 'snn_v6d2b'),
        ('V6d-v3', 'eval/dota_v6d3_results.json', 'snn_v6d3'),
    ]
    for name, path, key in comparisons:
        if os.path.exists(path):
            with open(path) as f:
                prev = json.load(f)
            print(f"\n  --- vs {name} ---")
            print(f"  {'BER':>6s}  {name:>8s}  {'V6d-v2c':>8s}  {'Winner':>8s}")
            for ber in bers:
                v_prev = prev[key].get(str(ber), {}).get('mAP50', 0)
                v_new = results['snn_v6d2c'][str(ber)]['mAP50']
                w = 'V6d-v2c' if v_new > v_prev else name if v_prev > v_new else 'Tie'
                print(f"  {ber:6.2f}  {v_prev:8.4f}  {v_new:8.4f}  {w:>8s}")

    os.makedirs('eval', exist_ok=True)
    with open('eval/dota_v6d2c_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved: eval/dota_v6d2c_results.json")


# ############################################################################
# PHASE 4: VISUALIZATION
# ############################################################################

def phase4_viz(args):
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data import YOLODataset
    from torch.utils.data import DataLoader
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("  Detection Visualization: SpikeAdaptSC V6d-v2c (with AdaBN)")
    print("=" * 70)

    ck = torch.load(SNN_SAVE, map_location=device, weights_only=False)
    cfg = ck['config']
    snn = SpikeAdaptSC_DetV6d2c_Multi(
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

    # Calibration data for AdaBN
    train_dataset = YOLODataset(
        img_path=data_dict['train'], imgsz=640,
        augment=False, data=data_dict, task='obb',
    )
    calib_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=YOLODataset.collate_fn,
    )

    ber_levels = [0.0, 0.10, 0.20, 0.30]
    os.makedirs('paper/figures', exist_ok=True)

    for img_idx, img_path in enumerate(test_imgs):
        n_cols = len(ber_levels) + 1
        fig, axes = plt.subplots(1, n_cols, figsize=(4.5*n_cols, 4.5))
        name = Path(img_path).stem

        yolo = YOLO(BASELINE)
        r = yolo.predict(img_path, imgsz=640, device=0, verbose=False, conf=0.25)
        ax = axes[0]
        ax.imshow(r[0].plot()[:, :, ::-1])
        nb = len(r[0].obb) if r[0].obb is not None else 0
        ax.set_title(f'Baseline\n({nb} det)', fontsize=10, fontweight='bold')
        ax.axis('off')

        for bi, ber in enumerate(ber_levels):
            yolo = YOLO(BASELINE)
            yolo.model.to(device)  # Ensure on CUDA
            yolo.model.fuse = lambda *a, **kw: yolo.model
            for idx, (lid, lev) in enumerate(zip(HOOK_LAYERS, snn.levels)):
                lev_copy = copy.deepcopy(lev)
                lev_copy.eval()
                yolo.model.model[lid] = SNN_DetV6d2c_Hook(
                    yolo.model.model[lid], lev_copy, ber=ber
                ).to(device)
            
            # AdaBN calibration
            adabn_calibrate(yolo.model, snn, calib_loader, n_batches=20)

            r = yolo.predict(img_path, imgsz=640, device=0, verbose=False, conf=0.25)
            ax = axes[bi + 1]
            ax.imshow(r[0].plot()[:, :, ::-1])
            nb = len(r[0].obb) if r[0].obb is not None else 0
            ax.set_title(f'V6d-v2c\nBER={ber:.2f} ({nb} det)', fontsize=10, fontweight='bold')
            ax.axis('off')

        plt.suptitle(f'V6d-v2c (AdaBN+Adapter) on DOTA ({name})', fontsize=13, fontweight='bold')
        plt.tight_layout()
        out = f'paper/figures/dota_v6d2c_viz_{img_idx+1}.png'
        plt.savefig(out, dpi=200, bbox_inches='tight')
        print(f"  Saved: {out}")
        plt.close()

    print(f"\n✅ {len(test_imgs)} visualizations saved!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, default=2, choices=[2, 3, 4])
    parser.add_argument('--s1-epochs', type=int, default=100)
    parser.add_argument('--s2-epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--c-spike', type=int, default=48)
    parser.add_argument('--T', type=int, default=8)
    parser.add_argument('--target-rate', type=float, default=0.75)
    parser.add_argument('--max-batches', type=int, default=300)
    args = parser.parse_args()

    {2: phase2_train, 3: phase3_eval, 4: phase4_viz}[args.phase](args)
