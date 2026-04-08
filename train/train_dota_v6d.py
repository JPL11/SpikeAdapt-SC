#!/usr/bin/env python3
"""SpikeAdapt-SC V6d — Multi-Scale Progressive JSCC for Object Detection (DOTA).

Extends V6's architecture to YOLO26-OBB detection:
  - Multi-scale SNN encoder within each FPN level (3 sub-scales)
  - SDSA at each sub-scale (spike-driven attention)
  - Cross-scale scorer for bandwidth allocation
  - Continuous Conv decoder (detection needs float features)
  - Coarse-scale protection for noise robustness
  - YOLO26 backbone hook injection at P3/P4/P5

Usage:
  python train/train_dota_v6d.py --phase 2   # Train SNN
  python train/train_dota_v6d.py --phase 3   # mAP vs BER evaluation
  python train/train_dota_v6d.py --phase 4   # Visualization
"""

import os, sys, json, argparse, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASELINE = 'runs/obb/runs/dota/yolo26n_obb_baseline/weights/best.pt'
SNN_SAVE = 'runs/yolo26_snn_v6d.pth'
HOOK_LAYERS = [4, 6, 10]     # YOLO26: P3, P4, P5
CHANNEL_SIZES = [128, 128, 256]


# ############################################################################
# SNN CORE MODULES
# ############################################################################

class SpikeFunction_Learnable(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane, threshold, slope):
        if isinstance(threshold, (int, float)):
            threshold = torch.tensor(float(threshold), device=membrane.device)
        ctx.save_for_backward(membrane, threshold, slope)
        ctx.th_needs_grad = threshold.requires_grad
        ctx.slope_needs_grad = slope.requires_grad
        return (membrane > threshold).float()
    @staticmethod
    def backward(ctx, grad_output):
        membrane, threshold, slope = ctx.saved_tensors
        s = slope.clamp(min=1.0, max=100.0)
        sig = torch.sigmoid(s * (membrane - threshold))
        sg = sig * (1 - sig) * s
        grad_mem = grad_output * sg
        grad_th = -(grad_output * sg).sum() if ctx.th_needs_grad else None
        grad_slope = (grad_output * sig * (1 - sig) * (membrane - threshold)).sum() if ctx.slope_needs_grad else None
        return grad_mem, grad_th, grad_slope


class LIFNeuron(nn.Module):
    def __init__(self, C, th=1.0):
        super().__init__()
        self.threshold = th
        self.beta_raw = nn.Parameter(torch.ones(1, C, 1, 1) * 2.2)
        self.slope = nn.Parameter(torch.tensor(10.0))
    def forward(self, x, mem=None):
        if mem is None: mem = torch.zeros_like(x)
        mem = torch.sigmoid(self.beta_raw) * mem + x
        sp = SpikeFunction_Learnable.apply(mem, self.threshold, self.slope)
        return sp, mem - sp * self.threshold


class MPBN(nn.Module):
    def __init__(self, C, T):
        super().__init__()
        self.bns = nn.ModuleList([nn.BatchNorm2d(C, affine=True) for _ in range(T)])
        self.T = T
    def forward(self, x, t):
        return self.bns[min(t, self.T - 1)](x)


# ############################################################################
# SDSA for Detection
# ############################################################################

class SpikeDrivenSelfAttention(nn.Module):
    """Spike-Driven Self-Attention with ternary STE."""
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.dim = dim; self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.k_conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.v_conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.q_bn = nn.BatchNorm2d(dim)
        self.k_bn = nn.BatchNorm2d(dim)
        self.v_bn = nn.BatchNorm2d(dim)

    @staticmethod
    def _ternary_ste(x):
        return x + (torch.sign(x) - x).detach()

    def forward(self, x):
        B, C, H, W = x.shape
        q = self._ternary_ste(self.q_bn(self.q_conv(x)))
        k = self._ternary_ste(self.k_bn(self.k_conv(x)))
        v = self.v_bn(self.v_conv(x))
        q = q.view(B, self.n_heads, self.head_dim, H*W)
        k = k.view(B, self.n_heads, self.head_dim, H*W)
        v = v.view(B, self.n_heads, self.head_dim, H*W)
        attn = torch.einsum('bhdn,bhen->bhde', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhde,bhen->bhdn', attn, v).reshape(B, C, H, W)
        return self.proj(out) + x


# ############################################################################
# MULTI-SCALE SNN ENCODER PER FPN LEVEL
# ############################################################################

class MultiScaleSNNEncoderDet(nn.Module):
    """Multi-scale SNN encoder for detection FPN features.
    
    Creates 3 sub-scales from input feature:
      Scale 1: H×W (full resolution)
      Scale 2: H/2×W/2 (mid)
      Scale 3: H/4×W/4 (semantic)
    """
    def __init__(self, C_in, C_tx=(12, 18, 24), T=8):
        super().__init__()
        self.T = T
        n_heads = max(1, min(C_in // 16, 8))
        
        # Scale 1: full res
        self.sdsa1 = SpikeDrivenSelfAttention(C_in, n_heads)
        self.conv1 = nn.Conv2d(C_in, C_tx[0], 3, 1, 1)
        self.mpbn1 = MPBN(C_tx[0], T)
        self.lif1 = LIFNeuron(C_tx[0])
        
        # Scale 2: half res
        self.pool2 = nn.AvgPool2d(2)
        self.sdsa2 = SpikeDrivenSelfAttention(C_in, n_heads)
        self.conv2 = nn.Conv2d(C_in, C_tx[1], 3, 1, 1)
        self.mpbn2 = MPBN(C_tx[1], T)
        self.lif2 = LIFNeuron(C_tx[1])
        
        # Scale 3: quarter res (semantic)
        self.pool3 = nn.AvgPool2d(4)
        self.sdsa3 = SpikeDrivenSelfAttention(C_in, n_heads)
        self.conv3 = nn.Conv2d(C_in, C_tx[2], 3, 1, 1)
        self.mpbn3 = MPBN(C_tx[2], T)
        self.lif3 = LIFNeuron(C_tx[2])

    def forward(self, feat):
        f1 = self.sdsa1(feat)
        f2 = self.sdsa2(self.pool2(feat))
        f3 = self.sdsa3(self.pool3(feat))
        
        spikes = [[], [], []]
        m1, m2, m3 = None, None, None
        for t in range(self.T):
            x1 = self.mpbn1(self.conv1(f1), t); s1, m1 = self.lif1(x1, m1); spikes[0].append(s1)
            x2 = self.mpbn2(self.conv2(f2), t); s2, m2 = self.lif2(x2, m2); spikes[1].append(s2)
            x3 = self.mpbn3(self.conv3(f3), t); s3, m3 = self.lif3(x3, m3); spikes[2].append(s3)
        return spikes


# ############################################################################
# CHANNEL + SCORER + MASKER
# ############################################################################

class BSC_Channel(nn.Module):
    def forward(self, x, ber):
        if ber <= 0: return x
        return ((x + (torch.rand_like(x.float()) < ber).float()) % 2)


class CrossScaleScorerDet(nn.Module):
    def __init__(self, C_tx=(12, 18, 24)):
        super().__init__()
        self.scale_scorers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c + 1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, True),
                nn.Conv2d(32, 1, 1), nn.Sigmoid()
            ) for c in C_tx
        ])
    def forward(self, multi_spikes, noise_param=0.0):
        imps = []
        for i, spikes in enumerate(multi_spikes):
            rate = torch.stack(spikes, dim=0).mean(0)
            B = rate.size(0)
            noise_map = torch.full((B, 1, rate.size(2), rate.size(3)),
                                   noise_param, device=rate.device)
            imps.append(self.scale_scorers[i](torch.cat([rate, noise_map], 1)))
        return imps


class MultiScaleMaskerDet(nn.Module):
    def __init__(self, target_rate=0.75, temperature=0.5, min_coarse_rate=0.90):
        super().__init__()
        self.target_rate = target_rate
        self.temperature = temperature
        self.min_coarse_rate = min_coarse_rate

    def forward(self, multi_spikes, importance_maps, training=True, target_override=None):
        target = target_override if target_override is not None else self.target_rate
        n_scales = len(multi_spikes)
        masked = []; rates = []
        for si, (spikes, imp) in enumerate(zip(multi_spikes, importance_maps)):
            B, _, H, W = imp.shape
            is_coarse = (si == n_scales - 1)
            if training:
                logits = torch.log(imp / (1 - imp + 1e-7) + 1e-7)
                if is_coarse:
                    logits = logits + 2.0
                u = torch.rand_like(logits).clamp(1e-7, 1-1e-7)
                soft = torch.sigmoid((logits - torch.log(-torch.log(u))) / self.temperature)
                hard = (soft > 0.5).float()
                mask = hard + (soft - soft.detach())
            else:
                eff_rate = max(target, self.min_coarse_rate) if is_coarse else target
                k = max(1, int(eff_rate * H * W))
                flat = imp.view(B, -1); _, idx = flat.topk(k, dim=1)
                mask = torch.zeros_like(flat).scatter_(1, idx, 1.0).view(B, 1, H, W)
            masked.append([s * mask for s in spikes])
            rates.append(mask.mean().item())
        return masked, rates


# ############################################################################
# DETECTION DECODER (continuous output)
# ############################################################################

class MultiScaleDetDecoder(nn.Module):
    """Decode multi-scale spikes back to continuous FPN features."""
    def __init__(self, C_out, C_tx=(12, 18, 24), T=8):
        super().__init__()
        self.T = T
        # Per-scale temporal aggregation + conv
        self.dec1 = nn.Sequential(nn.Conv2d(C_tx[0], C_out//2, 3, 1, 1), nn.BatchNorm2d(C_out//2), nn.SiLU())
        self.dec2 = nn.Sequential(nn.Conv2d(C_tx[1], C_out//2, 3, 1, 1), nn.BatchNorm2d(C_out//2), nn.SiLU())
        self.dec3 = nn.Sequential(nn.Conv2d(C_tx[2], C_out//2, 3, 1, 1), nn.BatchNorm2d(C_out//2), nn.SiLU())
        # Temporal weights per scale
        self.tw1 = nn.Parameter(torch.ones(T) / T)
        self.tw2 = nn.Parameter(torch.ones(T) / T)
        self.tw3 = nn.Parameter(torch.ones(T) / T)
        # Fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(C_out//2 * 3, C_out, 1), nn.BatchNorm2d(C_out), nn.SiLU())

    def _agg(self, spikes, tw, dec, target_size):
        w = torch.softmax(tw, dim=0)
        avg = sum(w[t] * spikes[t] for t in range(min(len(spikes), self.T)))
        out = dec(avg)
        if out.shape[2:] != target_size:
            out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
        return out

    def forward(self, recv_multi, target_size):
        d1 = self._agg(recv_multi[0], self.tw1, self.dec1, target_size)
        d2 = self._agg(recv_multi[1], self.tw2, self.dec2, target_size)
        d3 = self._agg(recv_multi[2], self.tw3, self.dec3, target_size)
        return self.fuse(torch.cat([d1, d2, d3], dim=1))


# ############################################################################
# FULL V6d PER-LEVEL MODULE
# ############################################################################

class SpikeAdaptSC_DetV6d(nn.Module):
    """V6d: Multi-scale progressive JSCC for one FPN level."""
    def __init__(self, C_in, C_tx=(12, 18, 24), T=8, target_rate=0.75):
        super().__init__()
        self.T = T; self.C_in = C_in
        self.encoder = MultiScaleSNNEncoderDet(C_in, C_tx, T)
        self.scorer = CrossScaleScorerDet(C_tx)
        self.masker = MultiScaleMaskerDet(target_rate, 0.5, 0.90)
        self.channel = BSC_Channel()
        self.decoder = MultiScaleDetDecoder(C_in, C_tx, T)

    def forward(self, feat, noise_param=0.0, target_rate_override=None):
        multi_spikes = self.encoder(feat)
        imp_maps = self.scorer(multi_spikes, noise_param)
        masked, rates = self.masker(multi_spikes, imp_maps, self.training, target_rate_override)
        recv = [[self.channel(s, noise_param) for s in scale] for scale in masked]
        target_size = (feat.shape[2], feat.shape[3])
        Fp = self.decoder(recv, target_size)
        return Fp, {
            'tx_rate': sum(rates) / len(rates),
            'importance': imp_maps,
            'multi_spikes': multi_spikes,
        }


class SpikeAdaptSC_DetV6d_Multi(nn.Module):
    """Wrapper for all FPN levels."""
    def __init__(self, channel_sizes, C_tx=(12, 18, 24), T=8, target_rate=0.75):
        super().__init__()
        self.levels = nn.ModuleList([
            SpikeAdaptSC_DetV6d(c, C_tx, T, target_rate) for c in channel_sizes
        ])
    def forward(self, features, ber=0.0, target_rate_override=None):
        recons, infos = [], []
        for feat, lev in zip(features, self.levels):
            r, info = lev(feat, ber, target_rate_override)
            recons.append(r); infos.append(info)
        return recons, infos


class SNN_DetV6d_Hook(nn.Module):
    """Hook to inject V6d into YOLO backbone."""
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


# ############################################################################
# PHASE 2: TRAIN
# ############################################################################

def phase2_train(args):
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data import YOLODataset
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    print("=" * 70)
    print("  SpikeAdapt-SC V6d: Multi-Scale Progressive JSCC for Detection")
    print("=" * 70)

    yolo_orig = YOLO(BASELINE)
    yolo_orig.model.to(device).eval()

    snn = SpikeAdaptSC_DetV6d_Multi(
        CHANNEL_SIZES, C_tx=(args.c_tx1, args.c_tx2, args.c_tx3),
        T=args.T, target_rate=args.target_rate
    ).to(device)
    
    n_params = sum(p.numel() for p in snn.parameters())
    print(f"  V6d SNN params: {n_params:,}")
    print(f"  C_tx=({args.c_tx1},{args.c_tx2},{args.c_tx3}), T={args.T}, target_rate={args.target_rate}")

    data_dict = check_det_dataset('DOTAv1.yaml')
    train_dataset = YOLODataset(
        img_path=data_dict['train'], imgsz=640,
        augment=True, data=data_dict, task='obb',
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=YOLODataset.collate_fn,
    )

    # Warmup + cosine LR
    optimizer = torch.optim.AdamW(snn.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup_epochs = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6)

    best_loss = float('inf')
    level_weights = [0.2, 0.3, 0.5]  # P3, P4, P5

    for epoch in range(args.epochs):
        snn.train()
        epoch_mse, epoch_rate = 0, 0
        n_batches = 0

        # BER curriculum: gentle early, aggressive later
        if epoch < 20:
            ber = np.random.uniform(0, 0.15)
        elif epoch < 60:
            ber = np.random.uniform(0.05, 0.25)
        else:
            ber = np.random.uniform(0.10, 0.40)  # aggressive

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

            mse_loss = sum(
                w * F.mse_loss(r, c.detach())
                for w, r, c in zip(level_weights, recons, clean)
            )
            rate_loss = sum(
                (info['tx_rate'] - args.target_rate) ** 2
                for info in infos
            ) / len(infos)

            loss = mse_loss + 0.1 * rate_loss

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
        print(f"  E{epoch+1:3d}/{args.epochs}  MSE={avg_mse:.6f}  Rate={avg_rate:.3f}  BER={ber:.2f}")

        if avg_mse < best_loss:
            best_loss = avg_mse
            torch.save({
                'snn_state': snn.state_dict(),
                'epoch': epoch + 1,
                'loss': best_loss,
                'config': {
                    'C_tx': (args.c_tx1, args.c_tx2, args.c_tx3),
                    'T': args.T, 'target_rate': args.target_rate,
                    'channel_sizes': CHANNEL_SIZES,
                },
            }, SNN_SAVE)

    print(f"\n✅ Best MSE: {best_loss:.6f}, saved: {SNN_SAVE}")


# ############################################################################
# PHASE 3: EVAL (mAP vs BER)
# ############################################################################

def phase3_eval(args):
    from ultralytics import YOLO

    print("=" * 70)
    print("  mAP vs BER: YOLO26 + SpikeAdaptSC V6d")
    print("=" * 70)

    ck = torch.load(SNN_SAVE, map_location=device, weights_only=False)
    cfg = ck['config']
    snn = SpikeAdaptSC_DetV6d_Multi(
        cfg['channel_sizes'], C_tx=cfg['C_tx'], T=cfg['T'],
        target_rate=cfg['target_rate']
    ).to(device)
    snn.load_state_dict(ck['snn_state'])
    snn.eval()
    print(f"  Loaded epoch {ck['epoch']}, MSE={ck['loss']:.6f}")

    # Baseline
    yolo_base = YOLO(BASELINE)
    base_res = yolo_base.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0, verbose=True, plots=False)
    base_map = float(base_res.results_dict.get('metrics/mAP50(B)', 0))
    print(f"  Baseline: mAP@50={base_map:.4f}")

    bers = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    results = {'baseline': base_map, 'snn_v6d': {}}

    for ber in bers:
        print(f"\n  BER={ber:.2f}...")
        yolo = YOLO(BASELINE)
        yolo.model.fuse = lambda *a, **kw: yolo.model
        for idx, (lid, lev) in enumerate(zip(HOOK_LAYERS, snn.levels)):
            yolo.model.model[lid] = SNN_DetV6d_Hook(
                yolo.model.model[lid], lev, ber=ber
            ).to(device)
        res = yolo.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0, verbose=False, plots=False)
        m50 = float(res.results_dict.get('metrics/mAP50(B)', 0))
        tx = np.mean([yolo.model.model[l].last_info.get('tx_rate', 0) for l in HOOK_LAYERS])
        delta = m50 - base_map
        results['snn_v6d'][str(ber)] = {'mAP50': m50, 'tx_rate': float(tx)}
        print(f"    mAP@50={m50:.4f} ({delta:+.4f}), Rate={tx:.3f}")

    print("\n" + "=" * 70)
    print(f"  {'BER':>6s}  {'mAP@50':>8s}  {'Δ':>8s}  {'Rate':>6s}")
    print(f"  {'none':>6s}  {base_map:8.4f}  {'—':>8s}  {'1.00':>6s}")
    for ber in bers:
        r = results['snn_v6d'][str(ber)]
        d = r['mAP50'] - base_map
        print(f"  {ber:6.2f}  {r['mAP50']:8.4f}  {d:+8.4f}  {r['tx_rate']:6.3f}")

    os.makedirs('eval', exist_ok=True)
    with open('eval/dota_v6d_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved: eval/dota_v6d_results.json")


# ############################################################################
# PHASE 4: VISUALIZATION
# ############################################################################

def phase4_viz(args):
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("  Detection Visualization: SpikeAdaptSC V6d (6 images)")
    print("=" * 70)

    ck = torch.load(SNN_SAVE, map_location=device, weights_only=False)
    cfg = ck['config']
    snn = SpikeAdaptSC_DetV6d_Multi(
        cfg['channel_sizes'], C_tx=cfg['C_tx'], T=cfg['T'],
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
                yolo.model.model[lid] = SNN_DetV6d_Hook(
                    yolo.model.model[lid], lev, ber=ber
                ).to(device)
            r = yolo.predict(img_path, imgsz=640, device=0, verbose=False, conf=0.25)
            ax = axes[bi + 1]
            ax.imshow(r[0].plot()[:, :, ::-1])
            nb = len(r[0].obb) if r[0].obb is not None else 0
            ax.set_title(f'V6d\nBER={ber:.2f} ({nb} det)', fontsize=10, fontweight='bold')
            ax.axis('off')

        plt.suptitle(f'YOLO26 + SpikeAdapt-SC V6d on DOTA ({name})', fontsize=13, fontweight='bold')
        plt.tight_layout()
        out = f'paper/figures/dota_v6d_viz_{img_idx+1}.png'
        plt.savefig(out, dpi=200, bbox_inches='tight')
        plt.savefig(out.replace('.png', '.pdf'), dpi=200, bbox_inches='tight')
        print(f"  Saved: {out}")
        plt.close()

    print(f"\n✅ {len(test_imgs)} visualizations saved!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, default=2, choices=[2, 3, 4])
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--c-tx1', type=int, default=12)
    parser.add_argument('--c-tx2', type=int, default=18)
    parser.add_argument('--c-tx3', type=int, default=24)
    parser.add_argument('--T', type=int, default=8)
    parser.add_argument('--target-rate', type=float, default=0.75)
    parser.add_argument('--ber-max', type=float, default=0.30)
    parser.add_argument('--max-batches', type=int, default=300)
    args = parser.parse_args()

    {2: phase2_train, 3: phase3_eval, 4: phase4_viz}[args.phase](args)
