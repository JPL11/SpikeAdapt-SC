#!/usr/bin/env python3
"""SpikeAdapt-SC V6d-v2 — Literature-Informed SNN Detection for DOTA.

Key improvements over V6d (based on CVPR/ICCV/ECCV/NeurIPS 2023-2024):
  1. SINGLE-SCALE per FPN level (SpikeYOLO: simpler = better)
  2. MEMBRANE SHORTCUTS (EMS-YOLO, ICCV 2023): transmit analog membrane
     potential alongside binary spikes for spatial precision
  3. ADAPTIVE THRESHOLDS per FPN level (SpikeFPN, IEEE 2024)
  4. Hybrid CNN+Spike approach (Spike-TransCNN): Conv encoder preserves
     local spatial features, spikes carry semantic content
  5. Longer training: 200 epochs with BER curriculum + warmup

Usage:
  python train/train_dota_v6d2.py --phase 2   # Train
  python train/train_dota_v6d2.py --phase 3   # mAP vs BER eval
  python train/train_dota_v6d2.py --phase 4   # Visualization
"""

import os, sys, json, argparse, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASELINE = 'runs/obb/runs/dota/yolo26n_obb_baseline/weights/best.pt'
SNN_SAVE = 'runs/yolo26_snn_v6_maxformer.pth'
HOOK_LAYERS = [4, 6, 10]     # P3, P4, P5
CHANNEL_SIZES = [128, 128, 256]


# ############################################################################
# ADAPTIVE LIF NEURON (SpikeFPN-inspired)
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
            # Sum over batch and spatial dims, keep channel dims
            dims_to_sum = [i for i in range(raw.dim()) if ctx.th_shape[i] == 1 and raw.shape[i] != 1]
            grad_th = raw.sum(dim=dims_to_sum, keepdim=True) if dims_to_sum else raw
        else:
            grad_th = None
        return grad_mem, grad_th, None


class AdaptiveLIF(nn.Module):
    """LIF neuron with ADAPTIVE threshold (SpikeFPN-inspired).
    
    Threshold adapts based on input statistics — different FPN levels
    (P3=fine, P5=coarse) naturally need different spiking sensitivity.
    """
    def __init__(self, C):
        super().__init__()
        self.threshold = nn.Parameter(torch.ones(1, C, 1, 1) * 1.0)
        self.beta_raw = nn.Parameter(torch.ones(1, C, 1, 1) * 2.2)
        self.slope = nn.Parameter(torch.tensor(10.0))
        # Adaptive threshold scaling based on input running mean
        self.register_buffer('running_mean', torch.ones(1) * 1.0)
        self.momentum = 0.1

    def forward(self, x, mem=None):
        if mem is None:
            mem = torch.zeros_like(x)
        mem = torch.sigmoid(self.beta_raw) * mem + x
        # Adaptive threshold: scale by input statistics
        if self.training:
            with torch.no_grad():
                batch_mean = x.abs().mean()
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
        adaptive_th = self.threshold * self.running_mean.clamp(min=0.1)
        sp = SpikeFunction.apply(mem, adaptive_th, self.slope)
        mem_out = mem - sp * adaptive_th
        return sp, mem_out, mem.clone()  # Return membrane potential too!


class MPBN(nn.Module):
    def __init__(self, C, T):
        super().__init__()
        self.bns = nn.ModuleList([nn.BatchNorm2d(C, affine=True) for _ in range(T)])
        self.T = T
    def forward(self, x, t):
        return self.bns[min(t, self.T - 1)](x)


# ############################################################################
# BSC CHANNEL
# ############################################################################

class BSC_Channel(nn.Module):
    def forward(self, x, ber):
        if ber <= 0: return x
        return ((x + (torch.rand_like(x.float()) < ber).float()) % 2)


# ############################################################################
# SINGLE-SCALE SNN ENCODER WITH MEMBRANE SHORTCUT (EMS-YOLO inspired)
# ############################################################################

class SNNEncoderEMS(nn.Module):
    """Single-scale SNN encoder with membrane shortcut.
    
    Key innovation: alongside binary spikes (for BSC channel),
    also outputs membrane potentials (analog) that carry spatial
    precision needed for bbox regression.
    
    Refs: EMS-YOLO (ICCV 2023), SpikeYOLO (ECCV 2024), Max-Former (NeurIPS 2025)
    """
    def __init__(self, C_in, C_spike=36, T=8):
        super().__init__()
        self.T = T; self.C_spike = C_spike
        # Feature projection (single-scale, no pooling!)
        self.proj = nn.Sequential(
            nn.Conv2d(C_in, C_spike, 3, 1, 1),
            nn.BatchNorm2d(C_spike),
        )
        
        # HIGH-FREQ BRANCH: Max-Pool + DWC to inject edge information (Max-Former)
        self.hf_maxpool = nn.MaxPool2d(3, stride=1, padding=1)
        self.hf_dwconv = nn.Conv2d(C_spike, C_spike, 3, 1, 1, groups=C_spike)
        self.hf_gate = nn.Sequential(
            nn.Conv2d(C_spike * 2, C_spike, 1), nn.Sigmoid())

        self.mpbn = MPBN(C_spike, T)
        self.lif = AdaptiveLIF(C_spike)
        
        # Membrane encoder: compress membrane to transmittable form
        self.mem_encoder = nn.Sequential(
            nn.Conv2d(C_spike, C_spike // 2, 1),
            nn.BatchNorm2d(C_spike // 2),
            nn.Tanh(),  # Bound to [-1, 1]
        )

    def forward(self, feat):
        """
        Returns:
            spikes: list of T binary tensors [B, C_spike, H, W]
            mems: list of T membrane potentials [B, C_spike//2, H, W]
        """
        x = self.proj(feat)
        
        # High-frequency injection
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
            # Membrane shortcut: encode membrane potential
            mems.append(self.mem_encoder(raw_mem))
        return spikes, mems


# ############################################################################
# NOISE-AWARE SCORER + MASKER
# ############################################################################

class NoiseAwareScorerDet(nn.Module):
    def __init__(self, C_spike=36):
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
        # Apply mask to spikes (binary) and mems (analog)
        masked_spikes = [s * mask for s in spikes]
        masked_mems = [m * mask for m in mems]  # membrane also masked
        return masked_spikes, masked_mems, mask


# ############################################################################
# DECODER WITH MEMBRANE FUSION (EMS-YOLO inspired)
# ############################################################################

class DetDecoderEMS(nn.Module):
    """Decoder that fuses binary spikes AND analog membrane potentials.
    
    The membrane shortcut provides continuous spatial information that
    spikes alone cannot carry — critical for bounding box regression.
    """
    def __init__(self, C_out, C_spike=36, T=8):
        super().__init__()
        self.T = T
        # Temporal weights for spike aggregation
        self.spike_tw = nn.Parameter(torch.ones(T) / T)
        self.mem_tw = nn.Parameter(torch.ones(T) / T)
        
        # Spike path: C_spike → C_out//2
        self.spike_dec = nn.Sequential(
            nn.Conv2d(C_spike, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            nn.SiLU(),
        )
        # Membrane path: C_spike//2 → C_out//2
        self.mem_dec = nn.Sequential(
            nn.Conv2d(C_spike // 2, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            nn.SiLU(),
        )
        # Fusion gate: learn how much to trust spikes vs membrane
        self.gate = nn.Sequential(
            nn.Conv2d(C_out * 2, C_out, 1),
            nn.Sigmoid()
        )
        # Final blend
        self.blend = nn.Sequential(
            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            nn.SiLU()
        )

    def forward(self, recv_spikes, recv_mems):
        sw = torch.softmax(self.spike_tw, 0)
        mw = torch.softmax(self.mem_tw, 0)
        spike_avg = sum(sw[t] * recv_spikes[t] for t in range(min(len(recv_spikes), self.T)))
        mem_avg = sum(mw[t] * recv_mems[t] for t in range(min(len(recv_mems), self.T)))
        
        spike_feat = self.spike_dec(spike_avg)    # semantic (from spikes)
        mem_feat = self.mem_dec(mem_avg)            # spatial (from membrane)
        
        # Gated fusion: gate decides spike vs membrane contribution
        gate = self.gate(torch.cat([spike_feat, mem_feat], dim=1))
        fused = gate * spike_feat + (1 - gate) * mem_feat
        return self.blend(fused)


# ############################################################################
# FULL V6d-v2 PER-LEVEL MODULE
# ############################################################################

class SpikeAdaptSC_DetV6d2(nn.Module):
    """V6d-v2: Single-scale SNN with membrane shortcuts for detection."""
    def __init__(self, C_in, C_spike=36, T=8, target_rate=0.75):
        super().__init__()
        self.T = T
        self.encoder = SNNEncoderEMS(C_in, C_spike, T)
        self.scorer = NoiseAwareScorerDet(C_spike)
        self.masker = DetMasker(target_rate, 0.5)
        self.channel = BSC_Channel()
        self.decoder = DetDecoderEMS(C_in, C_spike, T)

    def forward(self, feat, noise_param=0.0, target_rate_override=None):
        # Encode → spikes + membrane potentials
        spikes, mems = self.encoder(feat)
        importance = self.scorer(spikes, noise_param)
        
        # Mask both spikes and membranes
        masked_sp, masked_mem, mask = self.masker(
            spikes, mems, importance, self.training, target_rate_override)
        
        # Transmit spikes through BSC channel (binary)
        recv_sp = [self.channel(s, noise_param) for s in masked_sp]
        # Membrane transmitted through AWGN-like noise (analog)
        recv_mem = []
        for m in masked_mem:
            if noise_param > 0:
                # Analog channel: add Gaussian noise proportional to BER
                noise_std = noise_param * 0.5  # gentler than BSC
                recv_mem.append(m + torch.randn_like(m) * noise_std)
            else:
                recv_mem.append(m)
        
        # Decode with spike + membrane fusion
        Fp = self.decoder(recv_sp, recv_mem)
        
        return Fp, {
            'tx_rate': mask.mean().item(),
            'mask': mask,
            'importance': importance,
        }


class SpikeAdaptSC_DetV6d2_Multi(nn.Module):
    """Wrapper for all FPN levels."""
    def __init__(self, channel_sizes, C_spike=36, T=8, target_rate=0.75):
        super().__init__()
        self.levels = nn.ModuleList([
            SpikeAdaptSC_DetV6d2(c, C_spike, T, target_rate) for c in channel_sizes
        ])
    def forward(self, features, ber=0.0, target_rate_override=None):
        recons, infos = [], []
        for feat, lev in zip(features, self.levels):
            r, info = lev(feat, ber, target_rate_override)
            recons.append(r); infos.append(info)
        return recons, infos


class SNN_DetV6d2_Hook(nn.Module):
    """Hook to inject V6d-v2 into YOLO backbone."""
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


# ############################################################################
# PHASE 2: TRAIN (200 epochs with curriculum)
# ############################################################################

def phase2_train(args):
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data import YOLODataset
    from torch.utils.data import DataLoader

    print("=" * 70)
    print("  SpikeAdapt-SC V6d-v2: Membrane Shortcuts + Adaptive Thresholds")
    print("  [EMS-YOLO + SpikeFPN + SpikeYOLO inspired]")
    print("=" * 70)

    yolo_orig = YOLO(BASELINE)
    yolo_orig.model.to(device).eval()

    snn = SpikeAdaptSC_DetV6d2_Multi(
        CHANNEL_SIZES, C_spike=args.c_spike, T=args.T,
        target_rate=args.target_rate
    ).to(device)

    n_params = sum(p.numel() for p in snn.parameters())
    print(f"  V6d-v2 SNN params: {n_params:,}")
    print(f"  C_spike={args.c_spike}, T={args.T}, target_rate={args.target_rate}")
    print(f"  Epochs={args.epochs}, max_batches={args.max_batches}")

    data_dict = check_det_dataset('DOTAv1.yaml')
    train_dataset = YOLODataset(
        img_path=data_dict['train'], imgsz=640,
        augment=True, data=data_dict, task='obb',
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=YOLODataset.collate_fn,
    )

    optimizer = torch.optim.AdamW(snn.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup_epochs = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6)

    best_loss = float('inf')
    level_weights = [0.2, 0.3, 0.5]  # P3=fine, P4=mid, P5=semantic

    for epoch in range(args.epochs):
        snn.train()
        epoch_mse, epoch_rate = 0, 0
        n_batches = 0

        # BER curriculum: 4 phases
        if epoch < 30:
            ber = np.random.uniform(0, 0.10)          # gentle
        elif epoch < 80:
            ber = np.random.uniform(0.05, 0.20)       # moderate
        elif epoch < 150:
            ber = np.random.uniform(0.10, 0.30)       # aggressive
        else:
            ber = np.random.uniform(0.15, 0.40)       # hardening

        # Warmup LR
        if epoch < warmup_epochs:
            lr_scale = (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr * lr_scale

        pbar = tqdm(train_loader, desc=f"E{epoch+1}/{args.epochs}")
        for batch_data in pbar:
            imgs = batch_data['img'].to(device).float() / 255.0
            clean = get_backbone_features(yolo_orig.model, imgs)
            recons, infos = snn([f.detach() for f in clean], ber=ber)

            # Per-level weighted MSE
            mse_loss = sum(
                w * F.mse_loss(r, c.detach())
                for w, r, c in zip(level_weights, recons, clean)
            )
            # Rate regularization
            rate_loss = sum(
                (info['tx_rate'] - args.target_rate) ** 2
                for info in infos
            ) / len(infos)

            # Spatial consistency loss: encourage spatial smoothness
            spatial_loss = torch.tensor(0.0, device=device)
            for r in recons:
                dx = (r[:, :, :, 1:] - r[:, :, :, :-1]).abs().mean()
                dy = (r[:, :, 1:, :] - r[:, :, :-1, :]).abs().mean()
                spatial_loss = spatial_loss + (dx + dy) * 0.001

            loss = mse_loss + 0.1 * rate_loss + spatial_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(snn.parameters(), 1.0)
            optimizer.step()

            epoch_mse += mse_loss.item()
            epoch_rate += sum(i['tx_rate'] for i in infos) / len(infos)
            n_batches += 1
            pbar.set_postfix({'MSE': f'{mse_loss.item():.5f}', 'rate': f'{infos[0]["tx_rate"]:.2f}'})
            if n_batches >= args.max_batches:
                break

        if epoch >= warmup_epochs:
            scheduler.step()

        avg_mse = epoch_mse / max(n_batches, 1)
        avg_rate = epoch_rate / max(n_batches, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  E{epoch+1:3d}/{args.epochs}  MSE={avg_mse:.6f}  Rate={avg_rate:.3f}  BER={ber:.2f}")

        if avg_mse < best_loss:
            best_loss = avg_mse
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

    print(f"\n✅ Best MSE: {best_loss:.6f}, saved: {SNN_SAVE}")


# ############################################################################
# PHASE 3: EVAL
# ############################################################################

def phase3_eval(args):
    from ultralytics import YOLO

    print("=" * 70)
    print("  mAP vs BER: YOLO26 + SpikeAdaptSC V6d-v2")
    print("=" * 70)

    ck = torch.load(SNN_SAVE, map_location=device, weights_only=False)
    cfg = ck['config']
    snn = SpikeAdaptSC_DetV6d2_Multi(
        cfg['channel_sizes'], C_spike=cfg['C_spike'], T=cfg['T'],
        target_rate=cfg['target_rate']
    ).to(device)
    snn.load_state_dict(ck['snn_state'])
    snn.eval()
    print(f"  Loaded epoch {ck['epoch']}, MSE={ck['loss']:.6f}")

    yolo_base = YOLO(BASELINE)
    base_res = yolo_base.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0, verbose=True, plots=False)
    base_map = float(base_res.results_dict.get('metrics/mAP50(B)', 0))
    print(f"  Baseline: mAP@50={base_map:.4f}")

    bers = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    results = {'baseline': base_map, 'snn_v6d2': {}}

    for ber in bers:
        print(f"\n  BER={ber:.2f}...")
        yolo = YOLO(BASELINE)
        yolo.model.fuse = lambda *a, **kw: yolo.model
        for idx, (lid, lev) in enumerate(zip(HOOK_LAYERS, snn.levels)):
            yolo.model.model[lid] = SNN_DetV6d2_Hook(
                yolo.model.model[lid], lev, ber=ber
            ).to(device)
        res = yolo.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0, verbose=False, plots=False)
        m50 = float(res.results_dict.get('metrics/mAP50(B)', 0))
        tx = np.mean([yolo.model.model[l].last_info.get('tx_rate', 0) for l in HOOK_LAYERS])
        delta = m50 - base_map
        results['snn_v6d2'][str(ber)] = {'mAP50': m50, 'tx_rate': float(tx)}
        print(f"    mAP@50={m50:.4f} ({delta:+.4f}), Rate={tx:.3f}")

    # Comparison table
    print("\n" + "=" * 70)
    print(f"  {'BER':>6s}  {'mAP@50':>8s}  {'Δ':>8s}  {'Rate':>6s}")
    print(f"  {'none':>6s}  {base_map:8.4f}  {'—':>8s}  {'1.00':>6s}")
    for ber in bers:
        r = results['snn_v6d2'][str(ber)]
        d = r['mAP50'] - base_map
        print(f"  {ber:6.2f}  {r['mAP50']:8.4f}  {d:+8.4f}  {r['tx_rate']:6.3f}")

    # Load DetV2 for comparison
    detv2_path = 'archive/detection/eval/dota_detv2_results.json'
    if os.path.exists(detv2_path):
        with open(detv2_path) as f:
            dv2 = json.load(f)
        print(f"\n  --- Comparison vs DetV2 ---")
        print(f"  {'BER':>6s}  {'DetV2':>8s}  {'V6d-v2':>8s}  {'Winner':>8s}")
        for ber in bers:
            v2 = dv2['snn'].get(str(ber), {}).get('mAP50', 0)
            v6 = results['snn_v6d2'][str(ber)]['mAP50']
            w = 'V6d-v2' if v6 > v2 else 'DetV2' if v2 > v6 else 'Tie'
            print(f"  {ber:6.2f}  {v2:8.4f}  {v6:8.4f}  {w:>8s}")

    os.makedirs('eval', exist_ok=True)
    with open('eval/dota_v6d2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved: eval/dota_v6d2_results.json")


# ############################################################################
# PHASE 4: VISUALIZATION
# ############################################################################

def phase4_viz(args):
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("  Detection Visualization: SpikeAdaptSC V6d-v2")
    print("=" * 70)

    ck = torch.load(SNN_SAVE, map_location=device, weights_only=False)
    cfg = ck['config']
    snn = SpikeAdaptSC_DetV6d2_Multi(
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
                yolo.model.model[lid] = SNN_DetV6d2_Hook(
                    yolo.model.model[lid], lev, ber=ber
                ).to(device)
            r = yolo.predict(img_path, imgsz=640, device=0, verbose=False, conf=0.25)
            ax = axes[bi + 1]
            ax.imshow(r[0].plot()[:, :, ::-1])
            nb = len(r[0].obb) if r[0].obb is not None else 0
            ax.set_title(f'V6d-v2\nBER={ber:.2f} ({nb} det)', fontsize=10, fontweight='bold')
            ax.axis('off')

        plt.suptitle(f'YOLO26 + V6d-v2 (EMS+Adaptive) on DOTA ({name})', fontsize=13, fontweight='bold')
        plt.tight_layout()
        out = f'paper/figures/dota_v6d2_viz_{img_idx+1}.png'
        plt.savefig(out, dpi=200, bbox_inches='tight')
        print(f"  Saved: {out}")
        plt.close()

    print(f"\n✅ {len(test_imgs)} visualizations saved!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, default=2, choices=[2, 3, 4])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--c-spike', type=int, default=36)
    parser.add_argument('--T', type=int, default=8)
    parser.add_argument('--target-rate', type=float, default=0.75)
    parser.add_argument('--max-batches', type=int, default=300)
    args = parser.parse_args()

    {2: phase2_train, 3: phase3_eval, 4: phase4_viz}[args.phase](args)
