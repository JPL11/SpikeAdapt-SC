#!/usr/bin/env python3
"""SpikeAdapt-SC V6d-v2d — Joint YOLO Head Fine-Tuning with Detection Loss.

The key insight from EMS-YOLO (ICCV 2023), SpikeYOLO (ECCV 2024), and
BottleFit (IEEE 2022): the detection head MUST be jointly trained with
SNN features. Keeping it frozen causes catastrophic distribution shift.

Architecture: V6d-v2 SNN encoder/decoder (proven best)
Training:
  Stage 1: Load pre-trained V6d-v2 SNN weights (100 epochs MSE)
  Stage 2: Freeze SNN, unfreeze YOLO neck+head, train with OBB detection loss
  Stage 3: mAP eval
  Stage 4: Visualization

Usage:
  python train/train_dota_v6d2d.py --phase 2   # Stage 2: joint fine-tune
  python train/train_dota_v6d2d.py --phase 3   # mAP vs BER eval
  python train/train_dota_v6d2d.py --phase 4   # Visualization
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
SNN_V6D2_SAVE = 'runs/yolo26_snn_v6_maxformer.pth'     # Pre-trained SNN weights
SNN_V6D2D_SAVE = 'runs/yolo26_snn_v6d_maxformer.pth'    # Combined V6d-v2d weights
HOOK_LAYERS = [4, 6, 10]     # P3, P4, P5
CHANNEL_SIZES = [128, 128, 256]


# ############################################################################
# SNN COMPONENTS (identical to V6d-v2, copied for self-containment)
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
        if mem is None: mem = torch.zeros_like(x)
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


class SNNEncoder(nn.Module):
    def __init__(self, C_in, C_spike=36, T=8):
        super().__init__()
        self.T = T; self.C_spike = C_spike
        C_mem = C_spike // 2
        self.proj = nn.Sequential(
            nn.Conv2d(C_in, C_spike, 3, 1, 1), nn.BatchNorm2d(C_spike))
        
        # HIGH-FREQ BRANCH: Max-Pool + DWC to inject edge information (Max-Former)
        self.hf_maxpool = nn.MaxPool2d(3, stride=1, padding=1)
        self.hf_dwconv = nn.Conv2d(C_spike, C_spike, 3, 1, 1, groups=C_spike)
        self.hf_gate = nn.Sequential(
            nn.Conv2d(C_spike * 2, C_spike, 1), nn.Sigmoid())

        self.mpbn = MPBN(C_spike, T)
        self.lif = AdaptiveLIF(C_spike)
        self.mem_encoder = nn.Sequential(
            nn.Conv2d(C_spike, C_mem, 1), nn.BatchNorm2d(C_mem), nn.Tanh())
            
    def forward(self, feat):
        x = self.proj(feat)
        
        # High-frequency injection
        # hf = maxpool(x) - x isolates local high-frequency peaks/edges
        hf = self.hf_maxpool(x) - x
        hf = self.hf_dwconv(hf)
        gate = self.hf_gate(torch.cat([x, hf], dim=1))
        x = x + gate * hf  # Add high-freq detail before spiking
        
        spikes, mems = [], []
        mem = None
        for t in range(self.T):
            xt = self.mpbn(x, t)
            sp, mem, raw_mem = self.lif(xt, mem)
            spikes.append(sp)
            mems.append(self.mem_encoder(raw_mem))
        return spikes, mems


class NoiseAwareScorerDet(nn.Module):
    def __init__(self, C_spike=36):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Conv2d(C_spike + 1, 32, 3, 1, 1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 1, 1), nn.Sigmoid())
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


class DetDecoder(nn.Module):
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
        self.blend = nn.Sequential(
            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out), nn.SiLU())
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


class SpikeAdaptSC_Det(nn.Module):
    def __init__(self, C_in, C_spike=36, T=8, target_rate=0.75):
        super().__init__()
        self.encoder = SNNEncoder(C_in, C_spike, T)
        self.scorer = NoiseAwareScorerDet(C_spike)
        self.masker = DetMasker(target_rate, 0.5)
        self.channel = BSC_Channel()
        self.decoder = DetDecoder(C_in, C_spike, T)
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
        return decoded, {
            'tx_rate': mask.mean().item(),
            'mask': mask, 'importance': importance,
        }


class SpikeAdaptSC_Det_Multi(nn.Module):
    def __init__(self, channel_sizes, C_spike=36, T=8, target_rate=0.75):
        super().__init__()
        self.levels = nn.ModuleList([
            SpikeAdaptSC_Det(c, C_spike, T, target_rate) for c in channel_sizes
        ])
    def forward(self, features, ber=0.0, target_rate_override=None):
        recons, infos = [], []
        for feat, lev in zip(features, self.levels):
            r, info = lev(feat, ber, target_rate_override)
            recons.append(r); infos.append(info)
        return recons, infos


class SNN_Hook(nn.Module):
    """Hook that replaces a YOLO layer output with SNN-encoded version."""
    def __init__(self, original_layer, snn_level, ber=0.0):
        super().__init__()
        self.original_layer = original_layer
        self.snn = snn_level
        self.ber = ber
        # Copy YOLO metadata
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
# YOLO FORWARD PASS WITH SNN HOOKS
# ############################################################################

def yolo_forward_with_hooks(yolo_model, imgs):
    """Run full YOLO forward pass (backbone + neck + head) returning predictions.
    
    SNN hooks are already installed at HOOK_LAYERS, so SNN encoding/decoding
    happens transparently during the forward pass.
    """
    save = {}
    x = imgs
    for i, layer in enumerate(yolo_model.model):
        if hasattr(layer, 'f') and isinstance(layer.f, list):
            x_in = [x if j == -1 else save[j] for j in layer.f]
            x = layer(x_in)
        elif hasattr(layer, 'f') and isinstance(layer.f, int) and layer.f != -1:
            x = layer(save[layer.f])
        else:
            x = layer(x)
        save[i] = x
    return x  # Detection head output (raw predictions)


# ############################################################################
# PHASE 2: JOINT YOLO HEAD FINE-TUNING WITH DETECTION LOSS
# ############################################################################

def phase2_train(args):
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data import YOLODataset
    from torch.utils.data import DataLoader

    print("=" * 70)
    print("  SpikeAdapt-SC V6d-v2d: Joint YOLO Head Fine-Tuning")
    print("  [EMS-YOLO + SpikeYOLO + BottleFit]")
    print("=" * 70)

    # ---- Load pre-trained SNN encoder/decoder from V6d-v2 ----
    ck_snn = torch.load(SNN_V6D2_SAVE, map_location=device, weights_only=False)
    cfg = ck_snn['config']
    snn = SpikeAdaptSC_Det_Multi(
        cfg['channel_sizes'], C_spike=cfg['C_spike'], T=cfg['T'],
        target_rate=cfg['target_rate']
    ).to(device)
    snn.load_state_dict(ck_snn['snn_state'])
    snn.eval()  # SNN is FROZEN in Stage 2
    for param in snn.parameters():
        param.requires_grad = False
    print(f"  Loaded V6d-v2 SNN (epoch {ck_snn['epoch']}, loss {ck_snn['loss']:.6f})")
    print(f"  SNN params: {sum(p.numel() for p in snn.parameters()):,} (ALL FROZEN)")

    # ---- Load YOLO model ----
    yolo = YOLO(BASELINE)
    yolo_model = yolo.model.to(device)

    # ---- Install SNN hooks ----
    for idx, (lid, lev) in enumerate(zip(HOOK_LAYERS, snn.levels)):
        yolo_model.model[lid] = SNN_Hook(
            yolo_model.model[lid], lev, ber=0.0  # BER set per-batch
        ).to(device)

    # ---- Freeze backbone (layers before hooks stay frozen) ----
    # Unfreeze neck + head (layers after max hook point)
    trainable_params = []
    for i, layer in enumerate(yolo_model.model):
        if i <= max(HOOK_LAYERS):
            # Freeze backbone + SNN hook layers
            for param in layer.parameters():
                param.requires_grad = False
        else:
            # Unfreeze neck + head
            for param in layer.parameters():
                param.requires_grad = True
            trainable_params.extend(layer.parameters())

    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in yolo_model.parameters())
    print(f"  YOLO total params: {n_total:,}")
    print(f"  Trainable (neck+head): {n_trainable:,} ({100*n_trainable/n_total:.1f}%)")

    # ---- Initialize detection loss (OBB) ----
    # The loss function accesses self.hyp.box, .cls, .dfl, .angle via attribute access.
    # The loaded model.args (from weights) is a dict that may lack training hyperparams.
    # Create a proper object with all required attributes.
    from types import SimpleNamespace
    existing_args = yolo_model.args if isinstance(yolo_model.args, dict) else vars(yolo_model.args)
    loss_hyp = {
        'box': 7.5, 'cls': 0.5, 'dfl': 1.5, 'angle': 1.0,  # YOLO default loss weights
        'overlap_mask': True,
    }
    loss_hyp.update(existing_args)  # Merge with any existing args
    # Ensure loss weights are present (they may have been overwritten)
    for k, v in [('box', 7.5), ('cls', 0.5), ('dfl', 1.5), ('angle', 1.0)]:
        if k not in loss_hyp:
            loss_hyp[k] = v
    yolo_model.args = SimpleNamespace(**loss_hyp)
    criterion = yolo_model.init_criterion()
    print(f"  Loss: {type(criterion).__name__}")

    # ---- Data loader ----
    data_dict = check_det_dataset('DOTAv1.yaml')
    train_dataset = YOLODataset(
        img_path=data_dict['train'], imgsz=640,
        augment=True, data=data_dict, task='obb',
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=YOLODataset.collate_fn,
    )

    # ---- Optimizer (only neck+head) ----
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    warmup_epochs = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6)

    best_loss = float('inf')
    start_epoch = 0

    # Resume from checkpoint if exists
    if args.resume and os.path.exists(SNN_V6D2D_SAVE):
        ck_resume = torch.load(SNN_V6D2D_SAVE, map_location=device, weights_only=False)
        head_state = ck_resume.get('yolo_head_state', {})
        for layer_key, state in head_state.items():
            i = int(layer_key.split('_')[1])
            if i < len(yolo_model.model):
                yolo_model.model[i].load_state_dict(state)
        start_epoch = ck_resume.get('epoch', 0)
        best_loss = ck_resume.get('loss', float('inf'))
        print(f"  Resumed from epoch {start_epoch}, loss={best_loss:.6f}")

    for epoch in range(start_epoch, args.epochs):
        yolo_model.train()
        epoch_loss = 0
        n_batches = 0

        # LR warmup
        if epoch < warmup_epochs:
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr * (epoch + 1) / warmup_epochs

        pbar = tqdm(train_loader, desc=f"S2 E{epoch+1}/{args.epochs}")

        for batch_data in pbar:
            # Per-batch BER curriculum (much better diversity than per-epoch)
            # Phase 1 (epochs 0-39): mostly clean, occasional light noise
            # Phase 2 (epochs 40-69): moderate noise
            # Phase 3 (epochs 70+): full range
            if epoch < 40:
                ber = np.random.choice([0.0]*8 + [0.01, 0.02, 0.03, 0.05])  # 80% clean
            elif epoch < 70:
                ber = np.random.uniform(0, 0.15)
            else:
                ber = np.random.uniform(0, 0.25)

            # Set BER on hooks for this batch
            for lid in HOOK_LAYERS:
                if isinstance(yolo_model.model[lid], SNN_Hook):
                    yolo_model.model[lid].ber = ber

            imgs = batch_data['img'].to(device).float() / 255.0
            batch_size = imgs.size(0)

            # Forward through full YOLO (with SNN hooks active)
            preds = yolo_forward_with_hooks(yolo_model, imgs)

            # Compute OBB detection loss against ground truth
            loss_vec, loss_items = criterion(preds, batch_data)
            loss = loss_vec.sum()  # OBB loss returns [box, cls, dfl, angle], sum for scalar

            # Backward (gradients flow to neck+head only, SNN is frozen)
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
                    'dfl': f'{loss_items[2]:.3f}',
                    'ber': f'{ber:.2f}',
                })
            else:
                pbar.set_postfix({'L': f'{loss.item():.4f}'})
            if n_batches >= args.max_batches:
                break

        if epoch >= warmup_epochs:
            scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  S2 E{epoch+1:3d}/{args.epochs}  Loss={avg_loss:.6f}  LR={optimizer.param_groups[0]['lr']:.2e}")

        # Save best + periodic checkpoints
        def save_ckpt(path_):
            yolo_state = {}
            for i_, layer_ in enumerate(yolo_model.model):
                if i_ > max(HOOK_LAYERS):
                    yolo_state[f'layer_{i_}'] = layer_.state_dict()
            torch.save({
                'snn_state': snn.state_dict(),
                'yolo_head_state': yolo_state,
                'epoch': epoch + 1,
                'loss': avg_loss,
                'config': cfg,
            }, path_)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_ckpt(SNN_V6D2D_SAVE)

        # Save periodic checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            save_ckpt(SNN_V6D2D_SAVE.replace('.pth', f'_e{epoch+1}.pth'))

    print(f"\n✅ Best Loss: {best_loss:.6f}, saved: {SNN_V6D2D_SAVE}")


# ############################################################################
# HELPER: BUILD HOOKED YOLO MODEL
# ############################################################################

def build_hooked_yolo(snn, ber=0.0, load_head_weights=True):
    from ultralytics import YOLO
    yolo = YOLO(BASELINE)
    yolo.model.to(device)
    yolo.model.fuse = lambda *a, **kw: yolo.model

    # Install SNN hooks
    for idx, (lid, lev) in enumerate(zip(HOOK_LAYERS, snn.levels)):
        lev_copy = copy.deepcopy(lev)
        lev_copy.eval()
        yolo.model.model[lid] = SNN_Hook(
            yolo.model.model[lid], lev_copy, ber=ber
        ).to(device)

    # Load fine-tuned head weights
    if load_head_weights and os.path.exists(SNN_V6D2D_SAVE):
        ck = torch.load(SNN_V6D2D_SAVE, map_location=device, weights_only=False)
        head_state = ck.get('yolo_head_state', {})
        for layer_key, state in head_state.items():
            i = int(layer_key.split('_')[1])
            if i < len(yolo.model.model):
                yolo.model.model[i].load_state_dict(state)

    return yolo


# ############################################################################
# PHASE 3: EVAL
# ############################################################################

def phase3_eval(args):
    from ultralytics import YOLO

    print("=" * 70)
    print("  mAP vs BER: YOLO26 + SpikeAdaptSC V6d-v2d (Joint Fine-Tuned)")
    print("=" * 70)

    ck = torch.load(SNN_V6D2D_SAVE, map_location=device, weights_only=False)
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
    results = {'baseline': base_map, 'snn_v6d2d': {}}

    for ber in bers:
        print(f"\n  BER={ber:.2f}...")
        yolo = build_hooked_yolo(snn, ber=ber, load_head_weights=True)
        res = yolo.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0, verbose=False, plots=False)
        m50 = float(res.results_dict.get('metrics/mAP50(B)', 0))
        tx = np.mean([yolo.model.model[l].last_info.get('tx_rate', 0) for l in HOOK_LAYERS])
        delta = m50 - base_map
        results['snn_v6d2d'][str(ber)] = {'mAP50': m50, 'tx_rate': float(tx)}
        print(f"    mAP@50={m50:.4f} ({delta:+.4f}), Rate={tx:.3f}")

    print("\n" + "=" * 70)
    print(f"  {'BER':>6s}  {'mAP@50':>8s}  {'Δ':>8s}  {'Rate':>6s}")
    print(f"  {'none':>6s}  {base_map:8.4f}  {'—':>8s}  {'1.00':>6s}")
    for ber in bers:
        r = results['snn_v6d2d'][str(ber)]
        d = r['mAP50'] - base_map
        print(f"  {ber:6.2f}  {r['mAP50']:8.4f}  {d:+8.4f}  {r['tx_rate']:6.3f}")

    # Compare with V6d-v2
    comparisons = [
        ('DetV2', 'archive/detection/eval/dota_detv2_results.json', 'snn'),
        ('V6d-v2', 'eval/dota_v6d2_results.json', 'snn_v6d2'),
        ('V6d-v2b', 'eval/dota_v6d2b_results.json', 'snn_v6d2b'),
    ]
    for name, path, key in comparisons:
        if os.path.exists(path):
            with open(path) as f:
                prev = json.load(f)
            print(f"\n  --- vs {name} ---")
            print(f"  {'BER':>6s}  {name:>8s}  {'V6d-v2d':>8s}  {'Winner':>8s}")
            for ber in bers:
                v_prev = prev.get(key, {}).get(str(ber), {}).get('mAP50', 0)
                v_new = results['snn_v6d2d'][str(ber)]['mAP50']
                w = 'V6d-v2d' if v_new > v_prev else name if v_prev > v_new else 'Tie'
                print(f"  {ber:6.2f}  {v_prev:8.4f}  {v_new:8.4f}  {w:>8s}")

    os.makedirs('eval', exist_ok=True)
    with open('eval/dota_v6d2d_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved: eval/dota_v6d2d_results.json")


# ############################################################################
# PHASE 4: VISUALIZATION
# ############################################################################

def phase4_viz(args):
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("  Detection Visualization: V6d-v2d (Joint Fine-Tuned)")
    print("=" * 70)

    ck = torch.load(SNN_V6D2D_SAVE, map_location=device, weights_only=False)
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
            yolo = build_hooked_yolo(snn, ber=ber, load_head_weights=True)
            r = yolo.predict(img_path, imgsz=640, device=0, verbose=False, conf=0.25)
            ax = axes[bi + 1]
            ax.imshow(r[0].plot()[:, :, ::-1])
            nb = len(r[0].obb) if r[0].obb is not None else 0
            ax.set_title(f'V6d-v2d\nBER={ber:.2f} ({nb} det)', fontsize=10, fontweight='bold')
            ax.axis('off')

        plt.suptitle(f'V6d-v2d (Joint Fine-Tuned) on DOTA ({name})', fontsize=13, fontweight='bold')
        plt.tight_layout()
        out = f'paper/figures/dota_v6d2d_viz_{img_idx+1}.png'
        plt.savefig(out, dpi=200, bbox_inches='tight')
        print(f"  Saved: {out}")
        plt.close()

    print(f"\n✅ {len(test_imgs)} visualizations saved!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, default=2, choices=[2, 3, 4])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max-batches', type=int, default=500)
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()

    {2: phase2_train, 3: phase3_eval, 4: phase4_viz}[args.phase](args)
