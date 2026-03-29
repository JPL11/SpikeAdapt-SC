#!/usr/bin/env python3
"""Train and evaluate all detection baselines for paper comparison.

6 Methods (all C2=36, BSC channel, per FPN level):
  1. SpikeAdapt-SC (Ours)     — LIF+scorer+mask+BSC+ConvDec  [already trained]
  2. Conv AE (CNN-Bern)        — Conv+binarize+BSC+ConvDec    [already trained]
  3. CNN-Uni                   — Conv+8bit uniform quant+BSC   [train here]
  4. SNN-SC (No Mask)          — LIF encoder+rate=1.0+BSC      [train here]
  5. Random Mask               — LIF encoder+random mask+BSC   [train here]
  6. JPEG+Conv                 — JPEG quant+rep code+BSC       [no training]

Usage:
  python train_dota_baselines.py --method cnn_uni --phase 2     # Train CNN-Uni
  python train_dota_baselines.py --method snn_nomask --phase 2  # Train SNN-SC
  python train_dota_baselines.py --method random_mask --phase 2 # Train Random Mask
  python train_dota_baselines.py --method all --phase 3         # Eval all 6 methods
"""

import os, sys, json, argparse, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.spikeadapt_sc import Encoder
from models.snn_modules import get_channel, BSC_Channel
from train.train_dota_detv2 import (
    SpikeAdaptSC_DetV2_Multi, SNN_DetV2_Hook,
    DetectionDecoder, get_backbone_features,
    BASELINE, HOOK_LAYERS, CHANNEL_SIZES
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVES = {
    'spikeadapt': 'runs/yolo26_snn_detv2.pth',
    'conv_ae':    'runs/yolo26_ae_baseline.pth',
    'cnn_uni':    'runs/yolo26_cnn_uni.pth',
    'snn_nomask': 'runs/yolo26_snn_nomask.pth',
    'random_mask':'runs/yolo26_random_mask.pth',
}


# ===================================================================
# Baseline 3: CNN-Uni (Uniform 8-bit Quantization)
# ===================================================================
class UniformQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, n_bits=8):
        n_levels = 2 ** n_bits - 1
        x_min, x_max = x.min(), x.max()
        scale = (x_max - x_min) / n_levels
        ctx.save_for_backward(x)
        if scale == 0:
            return torch.zeros_like(x)
        x_q = torch.round((x - x_min) / scale) * scale + x_min
        return x_q

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class CNNUni_Level(nn.Module):
    """Conv encoder → 8-bit uniform quantize → per-bit BSC → dequant → Conv decoder."""
    def __init__(self, C_in, C2=36, n_bits=8):
        super().__init__()
        C1 = min(128, C_in)
        self.n_bits = n_bits
        self.encoder = nn.Sequential(
            nn.Conv2d(C_in, C1, 3, 1, 1), nn.BatchNorm2d(C1), nn.SiLU(True),
            nn.Conv2d(C1, C2, 3, 1, 1), nn.BatchNorm2d(C2),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(C2, C1, 3, 1, 1), nn.BatchNorm2d(C1), nn.SiLU(True),
            nn.Conv2d(C1, C_in, 3, 1, 1), nn.BatchNorm2d(C_in), nn.SiLU(True),
        )
        self.bsc = BSC_Channel()

    def forward(self, feat, ber=0.0):
        z = self.encoder(feat)
        # Quantize to [0, 1] range
        z_sig = torch.sigmoid(z)
        z_q = UniformQuantSTE.apply(z_sig, self.n_bits)
        # Convert to per-bit representation for BSC
        n_levels = 2 ** self.n_bits - 1
        z_int = (z_q * n_levels).clamp(0, n_levels)
        # Apply BSC noise to each bit plane
        if ber > 0:
            bits = []
            val = z_int.long()
            for b in range(self.n_bits):
                bit = ((val >> b) & 1).float()
                bit_noisy = self.bsc(bit, ber)
                bits.append(bit_noisy * (2 ** b))
            z_recv = sum(bits).float() / n_levels
        else:
            z_recv = z_q
        recon = self.decoder(z_recv)
        return recon, {'tx_rate': 1.0}


class CNNUni_Multi(nn.Module):
    def __init__(self, channel_sizes, C2=36, n_bits=8):
        super().__init__()
        self.levels = nn.ModuleList([CNNUni_Level(c, C2, n_bits) for c in channel_sizes])

    def forward(self, features, ber=0.0):
        recons, infos = [], []
        for feat, lev in zip(features, self.levels):
            r, i = lev(feat, ber=ber)
            recons.append(r); infos.append(i)
        return recons, infos


# ===================================================================
# Baseline 4: SNN-SC (No Mask, No Scorer — rate=1.0)
# ===================================================================
class SNNSC_Level(nn.Module):
    """LIF encoder (faithful) → rate=1.0 (no mask) → BSC → Conv decoder."""
    def __init__(self, C_in, C2=36, T=8):
        super().__init__()
        C1 = min(128, C_in)
        self.T = T
        self.encoder = Encoder(C_in, C1, C2, T)
        self.decoder = DetectionDecoder(C_in, C1, C2, T)
        self.channel = BSC_Channel()

    def forward(self, feat, ber=0.0):
        m1, m2 = None, None
        all_S2 = []
        for t in range(self.T):
            _, s2, m1, m2 = self.encoder(feat, m1, m2, t=t)
            all_S2.append(s2)
        # No mask — transmit everything
        mask = torch.ones(feat.shape[0], 1, all_S2[0].shape[2], all_S2[0].shape[3],
                         device=feat.device)
        recv = [self.channel(all_S2[t], ber) for t in range(self.T)]
        recon = self.decoder(recv, mask)
        return recon, {'tx_rate': 1.0}


class SNNSC_Multi(nn.Module):
    def __init__(self, channel_sizes, C2=36, T=8):
        super().__init__()
        self.levels = nn.ModuleList([SNNSC_Level(c, C2, T) for c in channel_sizes])

    def forward(self, features, ber=0.0):
        recons, infos = [], []
        for feat, lev in zip(features, self.levels):
            r, i = lev(feat, ber=ber)
            recons.append(r); infos.append(i)
        return recons, infos


# ===================================================================
# Baseline 5: Random Mask (same SNN, random spatial mask)
# ===================================================================
class RandomMask_Level(nn.Module):
    """LIF encoder → random spatial mask (75%) → BSC → Conv decoder."""
    def __init__(self, C_in, C2=36, T=8, keep_rate=0.75):
        super().__init__()
        C1 = min(128, C_in)
        self.T = T
        self.keep_rate = keep_rate
        self.encoder = Encoder(C_in, C1, C2, T)
        self.decoder = DetectionDecoder(C_in, C1, C2, T)
        self.channel = BSC_Channel()

    def forward(self, feat, ber=0.0):
        m1, m2 = None, None
        all_S2 = []
        for t in range(self.T):
            _, s2, m1, m2 = self.encoder(feat, m1, m2, t=t)
            all_S2.append(s2)
        # Random spatial mask
        h, w = all_S2[0].shape[2], all_S2[0].shape[3]
        if self.training:
            mask = (torch.rand(feat.shape[0], 1, h, w, device=feat.device) < self.keep_rate).float()
        else:
            mask = (torch.rand(feat.shape[0], 1, h, w, device=feat.device) < self.keep_rate).float()
        recv = [self.channel(all_S2[t] * mask, ber) for t in range(self.T)]
        recon = self.decoder(recv, mask)
        tx = mask.mean().item()
        return recon, {'tx_rate': tx}


class RandomMask_Multi(nn.Module):
    def __init__(self, channel_sizes, C2=36, T=8, keep_rate=0.75):
        super().__init__()
        self.levels = nn.ModuleList([
            RandomMask_Level(c, C2, T, keep_rate) for c in channel_sizes
        ])

    def forward(self, features, ber=0.0):
        recons, infos = [], []
        for feat, lev in zip(features, self.levels):
            r, i = lev(feat, ber=ber)
            recons.append(r); infos.append(i)
        return recons, infos


# ===================================================================
# Baseline 6: JPEG+Conv (no training)
# ===================================================================
class JPEGConv_Level(nn.Module):
    """Traditional: feature → quantize → JPEG-style → rep code → BSC → decode."""
    def __init__(self, C_in, jpeg_quality=50, code_rate_inv=3):
        super().__init__()
        self.C_in = C_in
        self.Q = jpeg_quality
        self.R = code_rate_inv
        self.bsc = BSC_Channel()

    def forward(self, feat, ber=0.0):
        # Normalize to [0, 1]
        f_min, f_max = feat.min(), feat.max()
        scale = f_max - f_min + 1e-8
        f_norm = (feat - f_min) / scale
        # Quantize to 8-bit
        f_q = (f_norm * 255).clamp(0, 255).round() / 255.0
        # Bit representation (simplified: take MSB as binary)
        f_bin = (f_q > 0.5).float()
        # Repetition code: repeat R times, majority vote after BSC
        if ber > 0:
            votes = []
            for _ in range(self.R):
                noisy = self.bsc(f_bin, ber)
                votes.append(noisy)
            f_recv = (sum(votes) > self.R / 2).float()
        else:
            f_recv = f_bin
        # Dequantize back
        recon = f_recv * scale + f_min
        return recon, {'tx_rate': 1.0}


class JPEGConv_Multi(nn.Module):
    def __init__(self, channel_sizes, jpeg_quality=50):
        super().__init__()
        self.levels = nn.ModuleList([JPEGConv_Level(c, jpeg_quality) for c in channel_sizes])

    def forward(self, features, ber=0.0):
        recons, infos = [], []
        for feat, lev in zip(features, self.levels):
            r, i = lev(feat, ber=ber)
            recons.append(r); infos.append(i)
        return recons, infos


# ===================================================================
# Generic Hook
# ===================================================================
class BaselineHook(nn.Module):
    def __init__(self, original_layer, baseline_level, ber=0.0):
        super().__init__()
        self.original_layer = original_layer
        self.bl = baseline_level
        self.ber = ber
        self.f = getattr(original_layer, 'f', -1)
        self.i = getattr(original_layer, 'i', 0)
        self.type = getattr(original_layer, 'type', type(original_layer).__name__)
        self.np = getattr(original_layer, 'np', 0)
        self.last_info = {}

    def forward(self, x):
        out = self.original_layer(x)
        recon, info = self.bl(out, ber=self.ber)
        self.last_info = info
        return recon


# ===================================================================
# Generic Train
# ===================================================================
def train_baseline(model_cls, model_kwargs, save_path, args):
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data import YOLODataset
    from torch.utils.data import DataLoader

    yolo_orig = YOLO(BASELINE)
    yolo_orig.model.to(device).eval()

    model = model_cls(**model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
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
    level_weights = [0.2, 0.3, 0.5]

    for epoch in range(args.epochs):
        model.train()
        epoch_mse, n_batches = 0, 0
        ber = np.random.uniform(0, args.ber_max)
        for batch_data in train_loader:
            imgs = batch_data['img'].to(device).float() / 255.0
            clean = get_backbone_features(yolo_orig.model, imgs)
            recons, infos = model([f.detach() for f in clean], ber=ber)
            mse_loss = sum(w * F.mse_loss(r, c.detach())
                          for w, r, c in zip(level_weights, recons, clean))
            optimizer.zero_grad()
            mse_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_mse += mse_loss.item()
            n_batches += 1
            if n_batches >= 200:
                break
        scheduler.step()
        avg_mse = epoch_mse / max(n_batches, 1)
        print(f"  Epoch {epoch+1:3d}/{args.epochs}  MSE={avg_mse:.6f}  BER={ber:.2f}")
        if avg_mse < best_loss:
            best_loss = avg_mse
            torch.save({'state': model.state_dict(), 'epoch': epoch+1,
                        'loss': best_loss, 'config': model_kwargs}, save_path)
    print(f"\n✅ Best MSE: {best_loss:.6f}, saved: {save_path}")


# ===================================================================
# Phase 2: Train specific baseline
# ===================================================================
def phase2(args):
    methods = {
        'cnn_uni': (CNNUni_Multi, {'channel_sizes': CHANNEL_SIZES, 'C2': args.c2, 'n_bits': 8}),
        'snn_nomask': (SNNSC_Multi, {'channel_sizes': CHANNEL_SIZES, 'C2': args.c2, 'T': args.T}),
        'random_mask': (RandomMask_Multi, {'channel_sizes': CHANNEL_SIZES, 'C2': args.c2,
                                            'T': args.T, 'keep_rate': 0.75}),
    }
    if args.method == 'all':
        for name, (cls, kwargs) in methods.items():
            print(f"\n{'='*70}\n  Training: {name}\n{'='*70}")
            train_baseline(cls, kwargs, SAVES[name], args)
    else:
        cls, kwargs = methods[args.method]
        print(f"\n{'='*70}\n  Training: {args.method}\n{'='*70}")
        train_baseline(cls, kwargs, SAVES[args.method], args)


# ===================================================================
# Phase 3: 10-seed BER sweep for all 6 methods
# ===================================================================
def phase3(args):
    from ultralytics import YOLO

    print("=" * 70)
    print(f"  Multi-Baseline BER Sweep ({args.n_seeds} seeds)")
    print("=" * 70)

    # Load all methods
    def load_method(name, cls, ck_path, extra_kwargs=None):
        if not os.path.exists(ck_path):
            print(f"  ⚠ {name}: {ck_path} not found, skipping")
            return None
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        cfg = ck.get('config', ck.get('config', {}))
        if extra_kwargs:
            cfg.update(extra_kwargs)
        model = cls(**cfg).to(device) if cfg else cls(CHANNEL_SIZES).to(device)
        sd_key = 'state' if 'state' in ck else ('snn_state' if 'snn_state' in ck else 'ae_state')
        model.load_state_dict(ck[sd_key])
        model.eval()
        mse = ck.get('loss', 0)
        print(f"  ✓ {name}: loaded (MSE={mse:.4f})")
        return model

    all_methods = {}

    # 1. SpikeAdapt-SC
    sa = load_method('SpikeAdapt-SC', SpikeAdaptSC_DetV2_Multi, SAVES['spikeadapt'],
                     {'channel_sizes': CHANNEL_SIZES, 'C1': 128, 'channel_type': 'bsc'})
    if sa: all_methods['SpikeAdapt-SC'] = sa

    # 2. Conv AE
    from train.train_dota_ae_baseline import ConvAE_Multi
    ae = load_method('Conv AE', ConvAE_Multi, SAVES['conv_ae'])
    if ae: all_methods['Conv AE'] = ae

    # 3. CNN-Uni
    uni = load_method('CNN-Uni', CNNUni_Multi, SAVES['cnn_uni'])
    if uni: all_methods['CNN-Uni'] = uni

    # 4. SNN-SC (No Mask)
    nm = load_method('SNN-SC', SNNSC_Multi, SAVES['snn_nomask'])
    if nm: all_methods['SNN-SC'] = nm

    # 5. Random Mask
    rm = load_method('Random Mask', RandomMask_Multi, SAVES['random_mask'])
    if rm: all_methods['Random Mask'] = rm

    # 6. JPEG+Conv (no checkpoint)
    jpeg = JPEGConv_Multi(CHANNEL_SIZES).to(device).eval()
    all_methods['JPEG+Conv'] = jpeg
    print(f"  ✓ JPEG+Conv: no training needed")

    # Baseline mAP
    yolo_base = YOLO(BASELINE)
    base_res = yolo_base.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0,
                              verbose=False, plots=False)
    base_map = float(base_res.results_dict.get('metrics/mAP50(B)', 0))
    print(f"\n  Baseline: mAP@50={base_map:.4f}")

    bers = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    results = {'baseline': base_map, 'methods': {}}

    for method_name, model in all_methods.items():
        print(f"\n{'='*50}")
        print(f"  {method_name}")
        print(f"{'='*50}")
        method_results = []

        for ber in bers:
            n_seeds = 1 if ber == 0.0 else args.n_seeds
            maps = []

            for seed in range(n_seeds):
                torch.manual_seed(seed * 42 + 7)
                np.random.seed(seed * 42 + 7)

                yolo = YOLO(BASELINE)
                yolo.model.fuse = lambda *a, **kw: yolo.model
                for idx, (lid, lev) in enumerate(zip(HOOK_LAYERS, model.levels)):
                    if method_name == 'SpikeAdapt-SC':
                        yolo.model.model[lid] = SNN_DetV2_Hook(
                            yolo.model.model[lid], lev, ber=ber
                        ).to(device)
                    else:
                        yolo.model.model[lid] = BaselineHook(
                            yolo.model.model[lid], lev, ber=ber
                        ).to(device)

                res = yolo.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0,
                               verbose=False, plots=False)
                m50 = float(res.results_dict.get('metrics/mAP50(B)', 0))
                maps.append(m50)
                if n_seeds > 1:
                    print(f"    BER={ber:.2f} seed={seed}: mAP={m50:.4f}")

            mean_map = np.mean(maps)
            std_map = np.std(maps) if len(maps) > 1 else 0
            method_results.append({
                'ber': ber, 'mean': mean_map, 'std': std_map,
                'maps': maps, 'n_seeds': n_seeds
            })
            print(f"  BER={ber:.2f}: {mean_map:.4f} ± {std_map:.4f}")

        results['methods'][method_name] = method_results

    # Save results
    os.makedirs('eval', exist_ok=True)
    with open('eval/multibaseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved: eval/multibaseline_results.json")

    # Print paper table
    print("\n" + "=" * 90)
    print("RESULTS TABLE (for paper)")
    print("=" * 90)
    header = f"{'Method':<20}"
    for ber in bers:
        header += f"  {'BER='+str(ber):>12}"
    print(header)
    print("-" * 90)
    print(f"{'Baseline YOLO26':<20}", end="")
    for ber in bers:
        print(f"  {base_map*100:>10.1f}%  ", end="")
    print()
    for method_name, method_results in results['methods'].items():
        print(f"{method_name:<20}", end="")
        for r in method_results:
            if r['std'] > 0:
                print(f"  {r['mean']*100:>5.1f}±{r['std']*100:>3.1f}%", end="")
            else:
                print(f"  {r['mean']*100:>10.1f}%  ", end="")
        print()

    # Generate plot
    generate_plot(results, bers)


def generate_plot(results, bers):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    styles = {
        'SpikeAdapt-SC': ('#4CAF50', 'o', '-',  2.5),
        'Conv AE':       ('#FF5722', 's', '--', 2.0),
        'CNN-Uni':       ('#2196F3', '^', '-.', 2.0),
        'SNN-SC':        ('#FF9800', 'D', '--', 2.0),
        'Random Mask':   ('#9E9E9E', 'x', '--', 1.5),
        'JPEG+Conv':     ('#9C27B0', 'v', ':',  2.0),
    }

    fig, ax = plt.subplots(1, 1, figsize=(11, 7))
    base = results['baseline']
    ax.axhline(y=base, color='#607D8B', linewidth=1.5, linestyle='--',
               label=f'Baseline YOLO26 ({base:.1%})', alpha=0.6)

    for name, method_results in results['methods'].items():
        c, mk, ls, lw = styles.get(name, ('gray', 'o', '-', 1.5))
        b = [r['ber'] for r in method_results]
        means = [r['mean'] for r in method_results]
        stds = [r['std'] for r in method_results]
        ax.plot(b, means, marker=mk, linestyle=ls, color=c, linewidth=lw,
                markersize=7, label=name, zorder=5)
        if any(s > 0 for s in stds):
            ax.fill_between(b, [m-s for m, s in zip(means, stds)],
                           [m+s for m, s in zip(means, stds)],
                           color=c, alpha=0.15)

    ax.set_xlabel('Bit Error Rate (BER)', fontsize=13)
    ax.set_ylabel('mAP@50', fontsize=13)
    ax.set_title('Detection Under Channel Noise: Multi-Baseline Comparison\n'
                 f'DOTA-v1.0, YOLO26n-OBB, C₂=36, BSC Channel, '
                 f'{results["methods"][list(results["methods"].keys())[0]][1].get("n_seeds", 10)} seeds',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(-0.01, 0.31)
    ax.set_ylim(-0.02, 0.5)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(bers)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    plt.tight_layout()

    os.makedirs('paper/figures', exist_ok=True)
    plt.savefig('paper/figures/multibaseline_comparison.png', dpi=200, bbox_inches='tight')
    plt.savefig('paper/figures/multibaseline_comparison.pdf', dpi=200, bbox_inches='tight')
    print("  Saved: paper/figures/multibaseline_comparison.{png,pdf}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, default=2, choices=[2, 3])
    parser.add_argument('--method', type=str, default='all',
                        choices=['cnn_uni', 'snn_nomask', 'random_mask', 'all'])
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--c2', type=int, default=36)
    parser.add_argument('--T', type=int, default=8)
    parser.add_argument('--ber-max', type=float, default=0.30)
    parser.add_argument('--n-seeds', type=int, default=10)
    args = parser.parse_args()

    {2: phase2, 3: phase3}[args.phase](args)
