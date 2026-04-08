#!/usr/bin/env python3
"""V8A-Slim: P2 Bridge only (no SE enhancers). Train + Eval + Compare."""

import os, sys, json, copy, math, argparse
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASELINE = 'runs/obb/runs/dota/yolo26n_obb_baseline/weights/best.pt'
V8A_STAGE1 = 'runs/yolo26_snn_v8A_stage1.pth'
V8A_SLIM_SAVE = 'runs/yolo26_snn_v8_slim.pth'
HOOK_LAYERS = [4, 6, 10]
CHANNEL_SIZES = [128, 128, 256]
P2_LAYER = 2
P2_CHANNELS = 64

from train.train_dota_v8 import (
    SpikeAdaptSC_Det_Multi, SpikeAdaptSC_Det,
    P2Bridge, SNN_Hook_P2, SNN_Hook_P3WithBridge,
    yolo_forward_with_hooks,
)
from train.train_dota_v6d2d import SNN_Hook


def build_slim_model(snn, ber=0.0, for_training=False):
    """V8A-Slim: P2 bridge but NO SE enhancers."""
    from ultralytics import YOLO
    yolo = YOLO(BASELINE)
    yolo.model.to(device)
    yolo.model.fuse = lambda *a, **kw: yolo.model
    extra = {}

    # P2 SNN + P2 hook (no enhancer)
    p2_snn = SpikeAdaptSC_Det(P2_CHANNELS, C_spike=36, T=8, target_rate=0.75).to(device)
    p2_hook = SNN_Hook_P2(yolo.model.model[P2_LAYER], p2_snn, ber=ber, enhancer=None).to(device)
    yolo.model.model[P2_LAYER] = p2_hook

    p2_bridge = P2Bridge(P2_CHANNELS, CHANNEL_SIZES[0]).to(device)

    # P3 hook WITH bridge (no enhancer)
    lev0 = copy.deepcopy(snn.levels[0]) if not for_training else snn.levels[0]
    if not for_training: lev0.eval()
    yolo.model.model[HOOK_LAYERS[0]] = SNN_Hook_P3WithBridge(
        yolo.model.model[HOOK_LAYERS[0]], lev0, p2_hook, p2_bridge,
        ber=ber, enhancer=None
    ).to(device)

    # P4, P5 hooks (no enhancer)
    for idx in range(1, len(HOOK_LAYERS)):
        lid = HOOK_LAYERS[idx]
        lev = copy.deepcopy(snn.levels[idx]) if not for_training else snn.levels[idx]
        if not for_training: lev.eval()
        yolo.model.model[lid] = SNN_Hook(
            yolo.model.model[lid], lev, ber=ber
        ).to(device)

    extra['p2_snn'] = p2_snn
    extra['p2_bridge'] = p2_bridge
    return yolo, extra


def train_slim(args):
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data import YOLODataset
    from torch.utils.data import DataLoader
    from types import SimpleNamespace

    print("=" * 70)
    print("  V8A-Slim: P2 Bridge Only (No SE Enhancers)")
    print("=" * 70)

    ck = torch.load(V8A_STAGE1, map_location=device, weights_only=False)
    cfg = ck['config']
    snn = SpikeAdaptSC_Det_Multi(cfg['channel_sizes'], C_spike=cfg['C_spike'],
        T=cfg['T'], target_rate=cfg['target_rate']).to(device)
    snn.load_state_dict(ck['snn_state']); snn.eval()
    for p in snn.parameters(): p.requires_grad = False
    print(f"  Loaded SNN (epoch {ck['epoch']}, MSE={ck['loss']:.6f})")

    yolo, extra = build_slim_model(snn, ber=0.0, for_training=True)
    yolo_model = yolo.model.to(device)

    trainable = []
    for i, layer in enumerate(yolo_model.model):
        if i <= max(HOOK_LAYERS) and i not in [P2_LAYER] + HOOK_LAYERS:
            for p in layer.parameters(): p.requires_grad = False
        elif i in HOOK_LAYERS or i == P2_LAYER:
            for name, p in layer.named_parameters():
                if 'snn' in name and 'p2_snn' not in name:
                    p.requires_grad = False
                elif 'bridge' in name or 'p2_snn' in name:
                    p.requires_grad = True; trainable.append(p)
                else:
                    p.requires_grad = False
        else:
            for p in layer.parameters():
                p.requires_grad = True; trainable.append(p)
    for m in extra.values():
        for p in m.parameters():
            if p.requires_grad: trainable.append(p)
    seen = set()
    trainable = [p for p in trainable if (pid := id(p)) not in seen and not seen.add(pid)]

    n_extra = sum(p.numel() for m in extra.values() for p in m.parameters())
    print(f"  Extra params (P2 only): {n_extra:,}")
    print(f"  Trainable: {sum(p.numel() for p in trainable):,}")

    ea = yolo_model.args if isinstance(yolo_model.args, dict) else vars(yolo_model.args)
    lh = {'box': 7.5, 'cls': 0.5, 'dfl': 1.5, 'angle': 1.0, 'overlap_mask': True}
    lh.update(ea)
    yolo_model.args = SimpleNamespace(**lh)
    criterion = yolo_model.init_criterion()

    data_dict = check_det_dataset('DOTAv1.yaml')
    ds = YOLODataset(img_path=data_dict['train'], imgsz=640, augment=True, data=data_dict, task='obb')
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True, collate_fn=YOLODataset.collate_fn)

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    warmup = 3
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs-warmup, eta_min=1e-6)
    best_loss = float('inf')

    for epoch in range(args.epochs):
        yolo_model.train()
        for lid in HOOK_LAYERS:
            h = yolo_model.model[lid]
            if hasattr(h, 'snn'): h.snn.eval()
        if hasattr(yolo_model.model[P2_LAYER], 'snn'):
            yolo_model.model[P2_LAYER].snn.eval()

        epoch_loss, nb = 0, 0
        if epoch < warmup:
            for pg in opt.param_groups: pg['lr'] = args.lr * (epoch+1)/warmup

        pbar = tqdm(dl, desc=f"Slim E{epoch+1}/{args.epochs}")
        for bd in pbar:
            if epoch < 20: ber = np.random.choice([0.0]*8+[0.01,0.02,0.03,0.05])
            elif epoch < 35: ber = np.random.uniform(0,0.15)
            else: ber = np.random.uniform(0,0.25)

            for lid in HOOK_LAYERS + [P2_LAYER]:
                h = yolo_model.model[lid]
                if hasattr(h, 'ber'): h.ber = ber

            imgs = bd['img'].to(device).float()/255.0
            preds = yolo_forward_with_hooks(yolo_model, imgs)
            lv, li = criterion(preds, bd); loss = lv.sum()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 10.0); opt.step()
            epoch_loss += loss.item(); nb += 1
            if li is not None and len(li) >= 3:
                pbar.set_postfix({'L': f'{loss.item():.4f}', 'box': f'{li[0]:.3f}', 'cls': f'{li[1]:.3f}', 'ber': f'{ber:.2f}'})
            if nb >= 300: break

        if epoch >= warmup: sched.step()
        avg = epoch_loss/max(nb, 1)
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"  E{epoch+1:3d}/{args.epochs}  Loss={avg:.6f}")
        if avg < best_loss:
            best_loss = avg
            es = {k: m.state_dict() for k, m in extra.items()}
            ys = {f'layer_{i}': l.state_dict() for i, l in enumerate(yolo_model.model) if i > max(HOOK_LAYERS)}
            torch.save({'snn_state': snn.state_dict(), 'yolo_head_state': ys,
                        'extra_modules': es, 'epoch': epoch+1, 'loss': avg, 'config': cfg}, V8A_SLIM_SAVE)

    print(f"\n✅ Best Loss: {best_loss:.6f}, saved: {V8A_SLIM_SAVE}")


def eval_slim(args):
    from ultralytics import YOLO

    print("=" * 70)
    print("  V8A-Slim: Evaluation")
    print("=" * 70)

    ck = torch.load(V8A_SLIM_SAVE, map_location=device, weights_only=False)
    cfg = ck['config']
    snn = SpikeAdaptSC_Det_Multi(cfg['channel_sizes'], C_spike=cfg['C_spike'],
        T=cfg['T'], target_rate=cfg['target_rate']).to(device)
    snn.load_state_dict(ck['snn_state']); snn.eval()
    print(f"  Loaded epoch {ck['epoch']}, loss={ck['loss']:.6f}")

    yolo_base = YOLO(BASELINE)
    base_res = yolo_base.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0, verbose=False, plots=False)
    base_map = float(base_res.results_dict.get('metrics/mAP50(B)', 0))

    bers = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    results = {'baseline': base_map, 'v8a_slim': {}}

    for ber in bers:
        print(f"\n  BER={ber:.2f}...")
        yolo, extra = build_slim_model(snn, ber=ber)
        for k, s in ck.get('extra_modules', {}).items():
            if k in extra:
                try: extra[k].load_state_dict(s)
                except: pass
        for lk, s in ck.get('yolo_head_state', {}).items():
            i = int(lk.split('_')[1])
            if i < len(yolo.model.model):
                try: yolo.model.model[i].load_state_dict(s)
                except: pass
        r = yolo.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0, verbose=False, plots=False)
        m50 = float(r.results_dict.get('metrics/mAP50(B)', 0))
        results['v8a_slim'][str(ber)] = {'mAP50': m50}
        print(f"    mAP@50={m50:.4f} ({m50-base_map:+.4f})")

    # Load V8A results for comparison
    v8a_results = {}
    if os.path.exists('eval/dota_v8_a_results.json'):
        with open('eval/dota_v8_a_results.json') as f:
            d = json.load(f)
            v8a_results = d.get('v8_a', {})

    print("\n" + "=" * 70)
    print(f"  {'BER':>6s}  {'Baseline':>8s}  {'V8A':>8s}  {'V8A-Slim':>9s}  {'Δ(Slim-V8A)':>12s}")
    print(f"  {'none':>6s}  {base_map:8.4f}  {'—':>8s}  {'—':>9s}  {'—':>12s}")
    for ber in bers:
        slim_m = results['v8a_slim'][str(ber)]['mAP50']
        v8a_m = v8a_results.get(str(ber), {}).get('mAP50', 0)
        delta = slim_m - v8a_m if v8a_m > 0 else 0
        print(f"  {ber:6.2f}  {base_map:8.4f}  {v8a_m:8.4f}  {slim_m:9.4f}  {delta:+12.4f}")

    with open('eval/dota_v8_slim_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved: eval/dota_v8_slim_results.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', choices=['train', 'eval', 'both'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    if args.phase in ['train', 'both']:
        train_slim(args)
    if args.phase in ['eval', 'both']:
        eval_slim(args)
