#!/usr/bin/env python3
"""SNN bottleneck (V8A structure) on DOTA-v2.0: train + BER sweep.

Recipe mirrors the proven V8A pipeline (train/train_dota_v8_slim.py):
  - YOLO base: the new DOTA-v2.0 baseline (18 classes)
  - SNN bottleneck warm-started from the v1-trained optA checkpoint
    (architecture is dataset-agnostic: hooks at backbone channels
    64/128/128/256, C_spike=36, T=8)
  - Train at 640 with BER curriculum, evaluate tiled at 1024
    (the same train/eval recipe that produced the v1 results)
  - Trainables: SNN + P2 bridge + enhancers + head (everything past the
    last hook layer); base backbone below hooks frozen.

Outputs:
  runs/yolo26_snn_dotav2.pth                     (optA-compatible format)
  eval/seed_results/detection_ber_sweep_dotav2.json

Usage:
    python train/train_dotav2_snn.py --phase both
"""

import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

V2_BASELINE = 'runs/obb/runs/obb/dotav2_baseline/weights/best.pt'
V1_OPTA = 'runs/yolo26_snn_v8_optA.pth'
SAVE = 'runs/yolo26_snn_dotav2.pth'
DATA_YAML = '/home/jpli/SemCom/datasets/DOTAv2.yaml'
OUT_JSON = 'eval/seed_results/detection_ber_sweep_dotav2.json'

import train.train_dota_v8 as v8
v8.BASELINE = V2_BASELINE          # build_v8_model reads module global

from train.train_dota_v8 import (SpikeAdaptSC_Det_Multi, build_v8_model,
                                 yolo_forward_with_hooks, HOOK_LAYERS,
                                 P2_LAYER)

BERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def load_warmstart_snn():
    ck = torch.load(V1_OPTA, map_location=device, weights_only=False)
    cfg = ck['config']
    snn = SpikeAdaptSC_Det_Multi(cfg['channel_sizes'], C_spike=cfg['C_spike'],
                                 T=cfg['T'],
                                 target_rate=cfg['target_rate']).to(device)
    snn.load_state_dict(ck['snn_state'])
    return snn, cfg, ck


def train(args):
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data import YOLODataset
    from torch.utils.data import DataLoader
    from types import SimpleNamespace

    snn, cfg, ck_v1 = load_warmstart_snn()
    print(f'warm-started SNN from {V1_OPTA}', flush=True)

    yolo, extra = build_v8_model(snn, ber=0.0, for_training=True)
    # warm-start extra modules (p2_snn, bridges, enhancers) where they match
    for k, sd in ck_v1.get('extra_modules', {}).items():
        if k in extra:
            try:
                extra[k].load_state_dict(sd)
            except Exception:
                print(f'  extra[{k}]: shape mismatch, fresh init')
    yolo_model = yolo.model.to(device)

    trainable = []
    for i, layer in enumerate(yolo_model.model):
        if i in HOOK_LAYERS + [P2_LAYER]:
            for p in layer.parameters():
                p.requires_grad = True
                trainable.append(p)
        elif i < max(HOOK_LAYERS):
            for p in layer.parameters():
                p.requires_grad = False
        else:
            for p in layer.parameters():
                p.requires_grad = True
                trainable.append(p)
    for m in extra.values():
        for p in m.parameters():
            if p.requires_grad:
                trainable.append(p)
    seen = set()
    trainable = [p for p in trainable
                 if (pid := id(p)) not in seen and not seen.add(pid)]
    print(f'trainable params: {sum(p.numel() for p in trainable):,}',
          flush=True)

    ea = yolo_model.args if isinstance(yolo_model.args, dict) \
        else vars(yolo_model.args)
    lh = {'box': 7.5, 'cls': 0.5, 'dfl': 1.5, 'angle': 1.0,
          'overlap_mask': True}
    lh.update(ea)
    yolo_model.args = SimpleNamespace(**lh)
    criterion = yolo_model.init_criterion()

    data_dict = check_det_dataset(DATA_YAML)
    ds = YOLODataset(img_path=data_dict['train'], imgsz=640, augment=True,
                     data=data_dict, task='obb')
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4,
                    pin_memory=True, collate_fn=YOLODataset.collate_fn)

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    warmup = 3
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs - warmup, eta_min=1e-6)
    best_loss = float('inf')

    for epoch in range(args.epochs):
        yolo_model.train()
        epoch_loss, nb = 0, 0
        if epoch < warmup:
            for pg in opt.param_groups:
                pg['lr'] = args.lr * (epoch + 1) / warmup

        for bd in dl:
            # BER curriculum: ramp to the full evaluated range and bias high
            # late (mirrors the classification model's [0.15,0.40] emphasis,
            # so the detector actually trains at BER=0.30, its eval extreme).
            if epoch < int(args.epochs * 0.3):
                ber = float(np.random.choice([0.0] * 8 +
                                             [0.01, 0.02, 0.03, 0.05]))
            elif epoch < int(args.epochs * 0.6):
                ber = float(np.random.uniform(0, 0.20))
            else:
                # 50% in the hard band [0.15, 0.40], else uniform [0, 0.40]
                if np.random.rand() < 0.5:
                    ber = float(np.random.uniform(0.15, 0.40))
                else:
                    ber = float(np.random.uniform(0, 0.40))
            for lid in HOOK_LAYERS + [P2_LAYER]:
                h = yolo_model.model[lid]
                if hasattr(h, 'ber'):
                    h.ber = ber

            imgs = bd['img'].to(device).float() / 255.0
            preds = yolo_forward_with_hooks(yolo_model, imgs)
            lv, _ = criterion(preds, bd)
            loss = lv.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 10.0)
            opt.step()
            epoch_loss += loss.item()
            nb += 1
            if nb >= args.batches_per_epoch:
                break

        if epoch >= warmup:
            sched.step()
        avg = epoch_loss / max(nb, 1)
        print(f'E{epoch+1:3d}/{args.epochs}  loss={avg:.5f}', flush=True)

        if avg < best_loss:
            best_loss = avg
            es = {k: m.state_dict() for k, m in extra.items()}
            ys = {f'layer_{i}': l.state_dict()
                  for i, l in enumerate(yolo_model.model)
                  if i > max(HOOK_LAYERS)}
            torch.save({'snn_state': snn.state_dict(),
                        'yolo_head_state': ys, 'extra_modules': es,
                        'epoch': epoch + 1, 'loss': avg, 'config': cfg},
                       SAVE)
    print(f'best loss {best_loss:.5f} -> {SAVE}', flush=True)


def evaluate(args):
    from ultralytics import YOLO

    results = {}
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            results = json.load(f)

    if 'baseline' not in results:
        yb = YOLO(V2_BASELINE)
        r = yb.val(data=DATA_YAML, imgsz=1024, batch=8, device=0,
                   verbose=False, plots=False)
        results['baseline'] = dict(map50=float(r.box.map50),
                                   map=float(r.box.map))
        with open(OUT_JSON, 'w') as f:
            json.dump(results, f, indent=1)
        print(f"baseline: mAP50={results['baseline']['map50']:.4f}",
              flush=True)
        del yb
        torch.cuda.empty_cache()

    ck = torch.load(SAVE, map_location=device, weights_only=False)
    cfg = ck['config']
    snn = SpikeAdaptSC_Det_Multi(cfg['channel_sizes'], C_spike=cfg['C_spike'],
                                 T=cfg['T'],
                                 target_rate=cfg['target_rate']).to(device)
    snn.load_state_dict(ck['snn_state'])
    snn.eval()

    results.setdefault('snn_v2', {})
    for ber in BERS:
        if str(ber) in results['snn_v2']:
            continue
        yolo, extra = build_v8_model(snn, ber=ber)
        for k, sd in ck.get('extra_modules', {}).items():
            if k in extra:
                try:
                    extra[k].load_state_dict(sd)
                except Exception:
                    pass
        for lk, sd in ck.get('yolo_head_state', {}).items():
            i = int(lk.split('_')[1])
            if i < len(yolo.model.model):
                try:
                    yolo.model.model[i].load_state_dict(sd)
                except Exception:
                    pass
        r = yolo.val(data=DATA_YAML, imgsz=1024, batch=8, device=0,
                     verbose=False, plots=False)
        results['snn_v2'][str(ber)] = dict(map50=float(r.box.map50),
                                           map=float(r.box.map))
        with open(OUT_JSON, 'w') as f:
            json.dump(results, f, indent=1)
        print(f'snn_v2 BER={ber}: mAP50={float(r.box.map50):.4f}',
              flush=True)
        del yolo
        torch.cuda.empty_cache()
    print('Done.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='both',
                        choices=['train', 'eval', 'both'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batches-per-epoch', type=int, default=600)
    args = parser.parse_args()
    if args.phase in ('train', 'both'):
        train(args)
    if args.phase in ('eval', 'both'):
        evaluate(args)
