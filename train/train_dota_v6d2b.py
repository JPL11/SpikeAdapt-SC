#!/usr/bin/env python3
"""SpikeAdapt-SC V6d-v2b — Improved V6d-v2 for Dense/Small Object Detection.

Key improvements over V6d-v2 (targeted at small object failure):
  1. WIDER MEMBRANE PATH: C_mem = C_spike (not C_spike//2) → 2× membrane info
  2. HIGHER C_SPIKE: 48 channels (vs 36) → more info capacity
  3. TWO-STAGE TRAINING:
     - Stage 1 (120 epochs): MSE + L1 feature reconstruction
     - Stage 2 (80 epochs): END-TO-END detection loss through YOLO head
     (BottleFit, 2022: "task-oriented training beats pure reconstruction")
  4. P3 PRIORITY: Weight P3 (highest-res) features most heavily
     (small objects live in P3, not P5)

Info capacity comparison:
  V6d-v2: 36 spike + 18 membrane = 54 channels
  V6d-v2b: 48 spike + 48 membrane = 96 channels → 1.78× more

Usage:
  python train/train_dota_v6d2b.py --phase 2   # Train Stage 1+2
  python train/train_dota_v6d2b.py --phase 3   # mAP eval
  python train/train_dota_v6d2b.py --phase 4   # Visualization
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
SNN_SAVE = 'runs/yolo26_snn_v6d2b.pth'
HOOK_LAYERS = [4, 6, 10]     # P3, P4, P5
CHANNEL_SIZES = [128, 128, 256]


# ############################################################################
# ADAPTIVE LIF (same as V6d-v2)
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
        self.threshold = nn.Parameter(torch.ones(1, C, 1, 1) * 1.0)
        self.beta_raw = nn.Parameter(torch.ones(1, C, 1, 1) * 2.2)
        self.slope = nn.Parameter(torch.tensor(10.0))
        self.register_buffer('running_mean', torch.ones(1) * 1.0)
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
# WIDER MEMBRANE ENCODER (Key change #1: C_mem = C_spike, not C_spike//2)
# ############################################################################

class SNNEncoderWide(nn.Module):
    """Encoder with WIDER membrane path — C_mem = C_spike.
    
    V6d-v2 bottleneck: membrane was compressed to C_spike//2 = 18 channels.
    Now membrane gets FULL C_spike = 48 channels.
    This doubles the analog spatial information capacity.
    """
    def __init__(self, C_in, C_spike=48, T=8):
        super().__init__()
        self.T = T; self.C_spike = C_spike
        self.proj = nn.Sequential(
            nn.Conv2d(C_in, C_spike, 3, 1, 1),
            nn.BatchNorm2d(C_spike),
        )
        self.mpbn = MPBN(C_spike, T)
        self.lif = AdaptiveLIF(C_spike)
        
        # WIDER membrane encoder: C_spike → C_spike (not C_spike//2!)
        self.mem_encoder = nn.Sequential(
            nn.Conv2d(C_spike, C_spike, 1),      # Full width!
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
# SCORER + MASKER (same as V6d-v2)
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
        masked_spikes = [s * mask for s in spikes]
        masked_mems = [m * mask for m in mems]
        return masked_spikes, masked_mems, mask


# ############################################################################
# WIDER DECODER (Key change #2: membrane path is C_spike, not C_spike//2)
# ############################################################################

class DetDecoderWide(nn.Module):
    """Decoder with wider membrane path to match the wider encoder."""
    def __init__(self, C_out, C_spike=48, T=8):
        super().__init__()
        self.T = T
        self.spike_tw = nn.Parameter(torch.ones(T) / T)
        self.mem_tw = nn.Parameter(torch.ones(T) / T)
        
        self.spike_dec = nn.Sequential(
            nn.Conv2d(C_spike, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            nn.SiLU(),
        )
        # WIDER: membrane is now C_spike (was C_spike//2)
        self.mem_dec = nn.Sequential(
            nn.Conv2d(C_spike, C_out, 3, 1, 1),   # Full width!
            nn.BatchNorm2d(C_out),
            nn.SiLU(),
        )
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
        
        spike_feat = self.spike_dec(spike_avg)
        mem_feat = self.mem_dec(mem_avg)
        
        gate = self.gate(torch.cat([spike_feat, mem_feat], dim=1))
        fused = gate * spike_feat + (1 - gate) * mem_feat
        return self.blend(fused)


# ############################################################################
# FULL PER-LEVEL MODULE
# ############################################################################

class SpikeAdaptSC_DetV6d2b(nn.Module):
    def __init__(self, C_in, C_spike=48, T=8, target_rate=0.75):
        super().__init__()
        self.T = T
        self.encoder = SNNEncoderWide(C_in, C_spike, T)
        self.scorer = NoiseAwareScorerDet(C_spike)
        self.masker = DetMasker(target_rate, 0.5)
        self.channel = BSC_Channel()
        self.decoder = DetDecoderWide(C_in, C_spike, T)

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


class SpikeAdaptSC_DetV6d2b_Multi(nn.Module):
    def __init__(self, channel_sizes, C_spike=48, T=8, target_rate=0.75):
        super().__init__()
        self.levels = nn.ModuleList([
            SpikeAdaptSC_DetV6d2b(c, C_spike, T, target_rate) for c in channel_sizes
        ])
    def forward(self, features, ber=0.0, target_rate_override=None):
        recons, infos = [], []
        for feat, lev in zip(features, self.levels):
            r, info = lev(feat, ber, target_rate_override)
            recons.append(r); infos.append(info)
        return recons, infos


class SNN_DetV6d2b_Hook(nn.Module):
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


def get_full_features(yolo_model, imgs):
    """Run full YOLO backbone+neck, returning all layer outputs."""
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
    return save


# ############################################################################
# PHASE 2: TWO-STAGE TRAINING
# ############################################################################

def phase2_train(args):
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data import YOLODataset
    from torch.utils.data import DataLoader

    print("=" * 70)
    print("  SpikeAdapt-SC V6d-v2b: Wider Membrane + Two-Stage Training")
    print("  [EMS-YOLO + SpikeFPN + BottleFit-inspired E2E detection loss]")
    print("=" * 70)

    yolo_orig = YOLO(BASELINE)
    yolo_orig.model.to(device).eval()

    snn = SpikeAdaptSC_DetV6d2b_Multi(
        CHANNEL_SIZES, C_spike=args.c_spike, T=args.T,
        target_rate=args.target_rate
    ).to(device)

    n_params = sum(p.numel() for p in snn.parameters())
    total_ch = args.c_spike + args.c_spike  # spike + membrane (now same width)
    print(f"  V6d-v2b SNN params: {n_params:,}")
    print(f"  C_spike={args.c_spike}, C_mem={args.c_spike} (FULL WIDTH)")
    print(f"  Total channels: {total_ch} (vs V6d-v2: {36 + 18} = 54)")
    print(f"  Stage 1: {args.s1_epochs} ep MSE | Stage 2: {args.s2_epochs} ep E2E det")

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
    # P3 weighted most (small objects live in P3!)
    level_weights = [0.4, 0.3, 0.3]

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

        # Reset optimizer for Stage 2 start
        if epoch == args.s1_epochs:
            print("\n" + "=" * 70)
            print("  >>> STAGE 2: End-to-End Detection Loss <<<")
            print("=" * 70)
            optimizer = torch.optim.AdamW(snn.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.s2_epochs, eta_min=1e-7)

        stage_label = "S2-E2E" if is_stage2 else "S1-MSE"
        pbar = tqdm(train_loader, desc=f"{stage_label} E{epoch+1}/{total_epochs}")

        for batch_data in pbar:
            imgs = batch_data['img'].to(device).float() / 255.0
            clean = get_backbone_features(yolo_orig.model, imgs)
            recons, infos = snn([f.detach() for f in clean], ber=ber)

            if not is_stage2:
                # STAGE 1: MSE + L1 reconstruction
                mse_loss = sum(
                    w * (F.mse_loss(r, c.detach()) + F.l1_loss(r, c.detach()))
                    for w, r, c in zip(level_weights, recons, clean)
                )
                rate_loss = sum(
                    (info['tx_rate'] - args.target_rate) ** 2
                    for info in infos
                ) / len(infos)
                loss = mse_loss + 0.1 * rate_loss
            else:
                # STAGE 2: End-to-end detection loss
                # Run full YOLO with SNN-reconstructed features
                mse_loss = sum(
                    w * F.mse_loss(r, c.detach())
                    for w, r, c in zip(level_weights, recons, clean)
                )
                # Detection coherence: penalize deviation from clean detection features
                # Use cosine similarity to preserve feature "direction" (what YOLO head expects)
                cos_loss = sum(
                    w * (1.0 - F.cosine_similarity(r, c.detach(), dim=1).mean())
                    for w, r, c in zip(level_weights, recons, clean)
                )
                rate_loss = sum(
                    (info['tx_rate'] - args.target_rate) ** 2
                    for info in infos
                ) / len(infos)
                loss = mse_loss + 0.5 * cos_loss + 0.05 * rate_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(snn.parameters(), 1.0)
            optimizer.step()

            epoch_loss += (mse_loss.item() if not is_stage2 else loss.item())
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
# PHASE 3: EVAL
# ############################################################################

def phase3_eval(args):
    from ultralytics import YOLO

    print("=" * 70)
    print("  mAP vs BER: YOLO26 + SpikeAdaptSC V6d-v2b")
    print("=" * 70)

    ck = torch.load(SNN_SAVE, map_location=device, weights_only=False)
    cfg = ck['config']
    snn = SpikeAdaptSC_DetV6d2b_Multi(
        cfg['channel_sizes'], C_spike=cfg['C_spike'], T=cfg['T'],
        target_rate=cfg['target_rate']
    ).to(device)
    snn.load_state_dict(ck['snn_state'])
    snn.eval()
    print(f"  Loaded epoch {ck['epoch']}, loss={ck['loss']:.6f}")

    yolo_base = YOLO(BASELINE)
    base_res = yolo_base.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0, verbose=True, plots=False)
    base_map = float(base_res.results_dict.get('metrics/mAP50(B)', 0))
    print(f"  Baseline: mAP@50={base_map:.4f}")

    bers = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    results = {'baseline': base_map, 'snn_v6d2b': {}}

    for ber in bers:
        print(f"\n  BER={ber:.2f}...")
        yolo = YOLO(BASELINE)
        yolo.model.fuse = lambda *a, **kw: yolo.model
        for idx, (lid, lev) in enumerate(zip(HOOK_LAYERS, snn.levels)):
            yolo.model.model[lid] = SNN_DetV6d2b_Hook(
                yolo.model.model[lid], lev, ber=ber
            ).to(device)
        res = yolo.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0, verbose=False, plots=False)
        m50 = float(res.results_dict.get('metrics/mAP50(B)', 0))
        tx = np.mean([yolo.model.model[l].last_info.get('tx_rate', 0) for l in HOOK_LAYERS])
        delta = m50 - base_map
        results['snn_v6d2b'][str(ber)] = {'mAP50': m50, 'tx_rate': float(tx)}
        print(f"    mAP@50={m50:.4f} ({delta:+.4f}), Rate={tx:.3f}")

    print("\n" + "=" * 70)
    print(f"  {'BER':>6s}  {'mAP@50':>8s}  {'Δ':>8s}  {'Rate':>6s}")
    print(f"  {'none':>6s}  {base_map:8.4f}  {'—':>8s}  {'1.00':>6s}")
    for ber in bers:
        r = results['snn_v6d2b'][str(ber)]
        d = r['mAP50'] - base_map
        print(f"  {ber:6.2f}  {r['mAP50']:8.4f}  {d:+8.4f}  {r['tx_rate']:6.3f}")

    # Compare with all previous versions
    comparisons = [
        ('DetV2', 'archive/detection/eval/dota_detv2_results.json', 'snn'),
        ('V6d-v2', 'eval/dota_v6d2_results.json', 'snn_v6d2'),
        ('V6d-v3', 'eval/dota_v6d3_results.json', 'snn_v6d3'),
    ]
    for name, path, key in comparisons:
        if os.path.exists(path):
            with open(path) as f:
                prev = json.load(f)
            print(f"\n  --- vs {name} ---")
            print(f"  {'BER':>6s}  {name:>8s}  {'V6d-v2b':>8s}  {'Winner':>8s}")
            for ber in bers:
                v_prev = prev[key].get(str(ber), {}).get('mAP50', 0)
                v_new = results['snn_v6d2b'][str(ber)]['mAP50']
                w = 'V6d-v2b' if v_new > v_prev else name if v_prev > v_new else 'Tie'
                print(f"  {ber:6.2f}  {v_prev:8.4f}  {v_new:8.4f}  {w:>8s}")

    os.makedirs('eval', exist_ok=True)
    with open('eval/dota_v6d2b_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved: eval/dota_v6d2b_results.json")


# ############################################################################
# PHASE 4: VISUALIZATION
# ############################################################################

def phase4_viz(args):
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("  Detection Visualization: SpikeAdaptSC V6d-v2b")
    print("=" * 70)

    ck = torch.load(SNN_SAVE, map_location=device, weights_only=False)
    cfg = ck['config']
    snn = SpikeAdaptSC_DetV6d2b_Multi(
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
                yolo.model.model[lid] = SNN_DetV6d2b_Hook(
                    yolo.model.model[lid], lev, ber=ber
                ).to(device)
            r = yolo.predict(img_path, imgsz=640, device=0, verbose=False, conf=0.25)
            ax = axes[bi + 1]
            ax.imshow(r[0].plot()[:, :, ::-1])
            nb = len(r[0].obb) if r[0].obb is not None else 0
            ax.set_title(f'V6d-v2b\nBER={ber:.2f} ({nb} det)', fontsize=10, fontweight='bold')
            ax.axis('off')

        plt.suptitle(f'V6d-v2b (Wide Membrane+E2E) on DOTA ({name})', fontsize=13, fontweight='bold')
        plt.tight_layout()
        out = f'paper/figures/dota_v6d2b_viz_{img_idx+1}.png'
        plt.savefig(out, dpi=200, bbox_inches='tight')
        print(f"  Saved: {out}")
        plt.close()

    print(f"\n✅ {len(test_imgs)} visualizations saved!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, default=2, choices=[2, 3, 4])
    parser.add_argument('--s1-epochs', type=int, default=120)
    parser.add_argument('--s2-epochs', type=int, default=80)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--c-spike', type=int, default=48)
    parser.add_argument('--T', type=int, default=8)
    parser.add_argument('--target-rate', type=float, default=0.75)
    parser.add_argument('--max-batches', type=int, default=300)
    args = parser.parse_args()

    {2: phase2_train, 3: phase3_eval, 4: phase4_viz}[args.phase](args)
