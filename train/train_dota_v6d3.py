#!/usr/bin/env python3
"""SpikeAdapt-SC V6d-v3 — I-LIF + Distillation + Membrane for DOTA Detection.

Literature-backed improvements over V6d-v2:
  1. I-LIF integer neurons (SpikeYOLO, ECCV 2024): {0,..,4} not {0,1}
  2. Feature distillation loss (CVPR 2024): L1 + cosine + Gram
  3. Higher C_spike=64 (IBRA-LIF insight): 4× info per position
  4. Membrane shortcuts kept (EMS-YOLO, ICCV 2023): BER robustness
  5. Adaptive thresholds kept (SpikeFPN, IEEE 2024)

Usage:
  python train/train_dota_v6d3.py --phase 2   # Train (200 epochs)
  python train/train_dota_v6d3.py --phase 3   # mAP vs BER eval
  python train/train_dota_v6d3.py --phase 4   # Visualization
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
SNN_SAVE = 'runs/yolo26_snn_v6d3.pth'
HOOK_LAYERS = [4, 6, 10]     # P3, P4, P5
CHANNEL_SIZES = [128, 128, 256]


# ############################################################################
# I-LIF: INTEGER LEAKY INTEGRATE-AND-FIRE (SpikeYOLO, ECCV 2024)
# ############################################################################

class IntegerSpikeFunction(torch.autograd.Function):
    """STE for integer-valued spikes: round in forward, pass gradient backward."""
    @staticmethod
    def forward(ctx, ratio, max_level):
        ctx.save_for_backward(ratio)
        ctx.max_level = max_level
        return ratio.clamp(0, max_level).round()
    @staticmethod
    def backward(ctx, grad_output):
        ratio, = ctx.saved_tensors
        # Gradient passes through where 0 <= ratio <= max_level
        mask = (ratio >= 0) & (ratio <= ctx.max_level)
        return grad_output * mask.float(), None


class IntegerLIF(nn.Module):
    """I-LIF neuron: integer-valued training, spike-driven inference.
    
    Output ∈ {0, 1, 2, ..., max_level} instead of binary {0, 1}.
    Each spike carries log2(max_level+1) bits of information.
    
    Ref: SpikeYOLO (ECCV 2024 Best Paper Candidate)
    """
    def __init__(self, C, max_level=4):
        super().__init__()
        self.max_level = max_level
        self.threshold = nn.Parameter(torch.ones(1, C, 1, 1) * 0.5)
        self.beta_raw = nn.Parameter(torch.ones(1, C, 1, 1) * 2.2)
        # Adaptive threshold scaling (SpikeFPN)
        self.register_buffer('running_mean', torch.ones(1))
        self.momentum = 0.1

    def forward(self, x, mem=None):
        if mem is None:
            mem = torch.zeros_like(x)
        beta = torch.sigmoid(self.beta_raw)
        mem = beta * mem + x
        
        # Adaptive threshold
        if self.training:
            with torch.no_grad():
                self.running_mean.lerp_(x.abs().mean(), self.momentum)
        
        adaptive_th = self.threshold.abs() * self.running_mean.clamp(min=0.1)
        
        # Integer quantization: floor(mem / threshold), clamped
        ratio = mem / (adaptive_th + 1e-6)
        integer_val = IntegerSpikeFunction.apply(ratio, self.max_level)
        
        # Subtract fired charge
        mem_out = mem - integer_val * adaptive_th
        raw_mem = mem.clone()
        
        return integer_val, mem_out, raw_mem


class MPBN(nn.Module):
    def __init__(self, C, T):
        super().__init__()
        self.bns = nn.ModuleList([nn.BatchNorm2d(C, affine=True) for _ in range(T)])
        self.T = T
    def forward(self, x, t):
        return self.bns[min(t, self.T - 1)](x)


# ############################################################################
# BSC CHANNEL (handles integer spikes)
# ############################################################################

class BSC_IntegerChannel(nn.Module):
    """BSC channel for integer spikes — each integer level has BER probability."""
    def forward(self, x, ber):
        if ber <= 0: return x
        # For integer spikes: randomly perturb each value
        # With probability BER, flip a random bit in the integer representation
        noise_mask = (torch.rand_like(x.float()) < ber)
        # Perturb: randomly add ±1 with probability BER
        perturbation = torch.where(
            torch.rand_like(x) > 0.5,
            torch.ones_like(x),
            -torch.ones_like(x)
        )
        noisy = x + noise_mask.float() * perturbation
        return noisy.clamp(0, 4)  # Keep in valid range


# ############################################################################
# I-LIF SNN ENCODER WITH MEMBRANE SHORTCUT
# ############################################################################

class SNNEncoderILIF(nn.Module):
    """Single-scale I-LIF encoder with membrane shortcut.
    
    Each spike carries {0,1,2,3,4} ≈ 2.3 bits.
    With C_spike=64: 64×2.3 = 147 bits per position (vs 36 bits in V6d-v2).
    """
    def __init__(self, C_in, C_spike=64, T=8, max_level=4):
        super().__init__()
        self.T = T; self.C_spike = C_spike
        self.proj = nn.Sequential(
            nn.Conv2d(C_in, C_spike, 3, 1, 1),
            nn.BatchNorm2d(C_spike),
        )
        self.mpbn = MPBN(C_spike, T)
        self.lif = IntegerLIF(C_spike, max_level)
        
        # Membrane encoder
        self.mem_encoder = nn.Sequential(
            nn.Conv2d(C_spike, C_spike // 2, 1),
            nn.BatchNorm2d(C_spike // 2),
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
    def __init__(self, C_spike=64):
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
        masked_spikes = [s * mask for s in spikes]
        masked_mems = [m * mask for m in mems]
        return masked_spikes, masked_mems, mask


# ############################################################################
# DECODER WITH MEMBRANE FUSION
# ############################################################################

class DetDecoderILIF(nn.Module):
    """Decoder for integer spikes + membrane. Handles multi-level spike values."""
    def __init__(self, C_out, C_spike=64, T=8, max_level=4):
        super().__init__()
        self.T = T; self.max_level = max_level
        self.spike_tw = nn.Parameter(torch.ones(T) / T)
        self.mem_tw = nn.Parameter(torch.ones(T) / T)
        
        # Spike path: normalize integer spikes to [0,1]
        self.spike_dec = nn.Sequential(
            nn.Conv2d(C_spike, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            nn.SiLU(),
        )
        # Membrane path
        self.mem_dec = nn.Sequential(
            nn.Conv2d(C_spike // 2, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            nn.SiLU(),
        )
        # Gated fusion
        self.gate = nn.Sequential(
            nn.Conv2d(C_out * 2, C_out, 1),
            nn.Sigmoid()
        )
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
        
        # Normalize integer spikes to [0,1] before decoding
        spike_norm = spike_avg / self.max_level
        
        spike_feat = self.spike_dec(spike_norm)
        mem_feat = self.mem_dec(mem_avg)
        
        gate = self.gate(torch.cat([spike_feat, mem_feat], dim=1))
        fused = gate * spike_feat + (1 - gate) * mem_feat
        return self.blend(fused)


# ############################################################################
# DISTILLATION LOSS (CVPR 2024)
# ############################################################################

def distillation_loss(recon, target):
    """Multi-granularity feature distillation.
    
    Combines:
      - L1: preserves spatial sharpness (edges of small objects)
      - Cosine: preserves feature direction per spatial position
      - Gram: preserves inter-channel relationships
    
    Ref: ANN-to-SNN Feature Distillation (CVPR 2024)
    """
    # L1: pixel-wise sharpness
    l1 = F.l1_loss(recon, target)
    
    # Cosine similarity per spatial position
    cos = 1.0 - F.cosine_similarity(recon, target, dim=1).mean()
    
    # Gram matrix: inter-channel structure
    B, C, H, W = recon.shape
    recon_flat = recon.view(B, C, -1)
    target_flat = target.view(B, C, -1)
    gram_r = torch.bmm(recon_flat, recon_flat.transpose(1, 2)) / (H * W)
    gram_t = torch.bmm(target_flat, target_flat.transpose(1, 2)) / (H * W)
    gram = F.mse_loss(gram_r, gram_t)
    
    return l1 + 0.5 * cos + 0.1 * gram


# ############################################################################
# FULL V6d-v3 PER-LEVEL MODULE
# ############################################################################

class SpikeAdaptSC_DetV6d3(nn.Module):
    def __init__(self, C_in, C_spike=64, T=8, target_rate=0.75, max_level=4):
        super().__init__()
        self.T = T
        self.encoder = SNNEncoderILIF(C_in, C_spike, T, max_level)
        self.scorer = NoiseAwareScorerDet(C_spike)
        self.masker = DetMasker(target_rate, 0.5)
        self.channel = BSC_IntegerChannel()
        self.decoder = DetDecoderILIF(C_in, C_spike, T, max_level)

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
        
        Fp = self.decoder(recv_sp, recv_mem)
        return Fp, {
            'tx_rate': mask.mean().item(),
            'mask': mask, 'importance': importance,
        }


class SpikeAdaptSC_DetV6d3_Multi(nn.Module):
    def __init__(self, channel_sizes, C_spike=64, T=8, target_rate=0.75, max_level=4):
        super().__init__()
        self.levels = nn.ModuleList([
            SpikeAdaptSC_DetV6d3(c, C_spike, T, target_rate, max_level)
            for c in channel_sizes
        ])
    def forward(self, features, ber=0.0, target_rate_override=None):
        recons, infos = [], []
        for feat, lev in zip(features, self.levels):
            r, info = lev(feat, ber, target_rate_override)
            recons.append(r); infos.append(info)
        return recons, infos


class SNN_DetV6d3_Hook(nn.Module):
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
# BACKBONE
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
# PHASE 2: TRAIN
# ############################################################################

def phase2_train(args):
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data import YOLODataset
    from torch.utils.data import DataLoader

    print("=" * 70)
    print("  SpikeAdapt-SC V6d-v3: I-LIF + Distillation + Membrane Shortcuts")
    print("  [SpikeYOLO + CVPR 2024 KD + EMS-YOLO + SpikeFPN]")
    print("=" * 70)

    yolo_orig = YOLO(BASELINE)
    yolo_orig.model.to(device).eval()

    snn = SpikeAdaptSC_DetV6d3_Multi(
        CHANNEL_SIZES, C_spike=args.c_spike, T=args.T,
        target_rate=args.target_rate, max_level=args.max_level
    ).to(device)

    n_params = sum(p.numel() for p in snn.parameters())
    print(f"  V6d-v3 SNN params: {n_params:,}")
    print(f"  C_spike={args.c_spike}, T={args.T}, max_level={args.max_level}")
    info_bits = args.c_spike * np.log2(args.max_level + 1)
    print(f"  Info capacity: {info_bits:.0f} bits/position ({args.c_spike}ch × {np.log2(args.max_level+1):.1f}b)")
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
    level_weights = [0.2, 0.3, 0.5]

    for epoch in range(args.epochs):
        snn.train()
        epoch_loss, epoch_rate = 0, 0
        n_batches = 0

        # BER curriculum
        if epoch < 30:
            ber = np.random.uniform(0, 0.10)
        elif epoch < 80:
            ber = np.random.uniform(0.05, 0.20)
        elif epoch < 150:
            ber = np.random.uniform(0.10, 0.30)
        else:
            ber = np.random.uniform(0.15, 0.40)

        # Warmup
        if epoch < warmup_epochs:
            lr_scale = (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr * lr_scale

        pbar = tqdm(train_loader, desc=f"E{epoch+1}/{args.epochs}")
        for batch_data in pbar:
            imgs = batch_data['img'].to(device).float() / 255.0
            clean = get_backbone_features(yolo_orig.model, imgs)
            recons, infos = snn([f.detach() for f in clean], ber=ber)

            # DISTILLATION LOSS (CVPR 2024): L1 + cosine + Gram
            dist_loss = sum(
                w * distillation_loss(r, c.detach())
                for w, r, c in zip(level_weights, recons, clean)
            )
            # Rate regularization
            rate_loss = sum(
                (info['tx_rate'] - args.target_rate) ** 2
                for info in infos
            ) / len(infos)

            loss = dist_loss + 0.1 * rate_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(snn.parameters(), 1.0)
            optimizer.step()

            epoch_loss += dist_loss.item()
            epoch_rate += sum(i['tx_rate'] for i in infos) / len(infos)
            n_batches += 1
            pbar.set_postfix({
                'DL': f'{dist_loss.item():.5f}',
                'rate': f'{infos[0]["tx_rate"]:.2f}'
            })
            if n_batches >= args.max_batches:
                break

        if epoch >= warmup_epochs:
            scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_rate = epoch_rate / max(n_batches, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  E{epoch+1:3d}/{args.epochs}  DistLoss={avg_loss:.6f}  Rate={avg_rate:.3f}  BER={ber:.2f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'snn_state': snn.state_dict(),
                'epoch': epoch + 1,
                'loss': best_loss,
                'config': {
                    'C_spike': args.c_spike, 'T': args.T,
                    'target_rate': args.target_rate,
                    'max_level': args.max_level,
                    'channel_sizes': CHANNEL_SIZES,
                },
            }, SNN_SAVE)

    print(f"\n✅ Best DistLoss: {best_loss:.6f}, saved: {SNN_SAVE}")


# ############################################################################
# PHASE 3: EVAL
# ############################################################################

def phase3_eval(args):
    from ultralytics import YOLO

    print("=" * 70)
    print("  mAP vs BER: YOLO26 + SpikeAdaptSC V6d-v3")
    print("=" * 70)

    ck = torch.load(SNN_SAVE, map_location=device, weights_only=False)
    cfg = ck['config']
    snn = SpikeAdaptSC_DetV6d3_Multi(
        cfg['channel_sizes'], C_spike=cfg['C_spike'], T=cfg['T'],
        target_rate=cfg['target_rate'], max_level=cfg['max_level']
    ).to(device)
    snn.load_state_dict(ck['snn_state'])
    snn.eval()
    print(f"  Loaded epoch {ck['epoch']}, loss={ck['loss']:.6f}")

    yolo_base = YOLO(BASELINE)
    base_res = yolo_base.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0, verbose=True, plots=False)
    base_map = float(base_res.results_dict.get('metrics/mAP50(B)', 0))
    print(f"  Baseline: mAP@50={base_map:.4f}")

    bers = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    results = {'baseline': base_map, 'snn_v6d3': {}}

    for ber in bers:
        print(f"\n  BER={ber:.2f}...")
        yolo = YOLO(BASELINE)
        yolo.model.fuse = lambda *a, **kw: yolo.model
        for idx, (lid, lev) in enumerate(zip(HOOK_LAYERS, snn.levels)):
            yolo.model.model[lid] = SNN_DetV6d3_Hook(
                yolo.model.model[lid], lev, ber=ber
            ).to(device)
        res = yolo.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0, verbose=False, plots=False)
        m50 = float(res.results_dict.get('metrics/mAP50(B)', 0))
        tx = np.mean([yolo.model.model[l].last_info.get('tx_rate', 0) for l in HOOK_LAYERS])
        delta = m50 - base_map
        results['snn_v6d3'][str(ber)] = {'mAP50': m50, 'tx_rate': float(tx)}
        print(f"    mAP@50={m50:.4f} ({delta:+.4f}), Rate={tx:.3f}")

    # Comparison table
    print("\n" + "=" * 70)
    print(f"  {'BER':>6s}  {'mAP@50':>8s}  {'Δ':>8s}  {'Rate':>6s}")
    print(f"  {'none':>6s}  {base_map:8.4f}  {'—':>8s}  {'1.00':>6s}")
    for ber in bers:
        r = results['snn_v6d3'][str(ber)]
        d = r['mAP50'] - base_map
        print(f"  {ber:6.2f}  {r['mAP50']:8.4f}  {d:+8.4f}  {r['tx_rate']:6.3f}")

    # Load V6d-v2 + DetV2 for comparison
    for name, path in [('DetV2', 'archive/detection/eval/dota_detv2_results.json'),
                       ('V6d-v2', 'eval/dota_v6d2_results.json')]:
        if os.path.exists(path):
            with open(path) as f:
                prev = json.load(f)
            key = 'snn' if 'snn' in prev else 'snn_v6d2'
            print(f"\n  --- vs {name} ---")
            print(f"  {'BER':>6s}  {name:>8s}  {'V6d-v3':>8s}  {'Winner':>8s}")
            for ber in bers:
                v_prev = prev[key].get(str(ber), {}).get('mAP50', 0)
                v_new = results['snn_v6d3'][str(ber)]['mAP50']
                w = 'V6d-v3' if v_new > v_prev else name if v_prev > v_new else 'Tie'
                print(f"  {ber:6.2f}  {v_prev:8.4f}  {v_new:8.4f}  {w:>8s}")

    os.makedirs('eval', exist_ok=True)
    with open('eval/dota_v6d3_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved: eval/dota_v6d3_results.json")


# ############################################################################
# PHASE 4: VISUALIZATION
# ############################################################################

def phase4_viz(args):
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("  Detection Visualization: SpikeAdaptSC V6d-v3")
    print("=" * 70)

    ck = torch.load(SNN_SAVE, map_location=device, weights_only=False)
    cfg = ck['config']
    snn = SpikeAdaptSC_DetV6d3_Multi(
        cfg['channel_sizes'], C_spike=cfg['C_spike'], T=cfg['T'],
        target_rate=cfg['target_rate'], max_level=cfg['max_level']
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
                yolo.model.model[lid] = SNN_DetV6d3_Hook(
                    yolo.model.model[lid], lev, ber=ber
                ).to(device)
            r = yolo.predict(img_path, imgsz=640, device=0, verbose=False, conf=0.25)
            ax = axes[bi + 1]
            ax.imshow(r[0].plot()[:, :, ::-1])
            nb = len(r[0].obb) if r[0].obb is not None else 0
            ax.set_title(f'V6d-v3\nBER={ber:.2f} ({nb} det)', fontsize=10, fontweight='bold')
            ax.axis('off')

        plt.suptitle(f'V6d-v3 (I-LIF+Distill) on DOTA ({name})', fontsize=13, fontweight='bold')
        plt.tight_layout()
        out = f'paper/figures/dota_v6d3_viz_{img_idx+1}.png'
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
    parser.add_argument('--c-spike', type=int, default=64)
    parser.add_argument('--T', type=int, default=8)
    parser.add_argument('--target-rate', type=float, default=0.75)
    parser.add_argument('--max-level', type=int, default=4)
    parser.add_argument('--max-batches', type=int, default=300)
    args = parser.parse_args()

    {2: phase2_train, 3: phase3_eval, 4: phase4_viz}[args.phase](args)
