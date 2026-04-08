#!/usr/bin/env python3
"""V8A Follow-up Analysis:
1. Ablation at BER=0.20 and BER=0.30 (show P2/SE value under noise)
2. SAHI with NMS deduplication
3. Small object focused analysis
"""
import os, sys, json, copy
import torch, torch.nn as nn, numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
device = torch.device("cuda")

BASELINE = 'runs/obb/runs/dota/yolo26n_obb_baseline/weights/best.pt'
V8A_SAVE = 'runs/yolo26_snn_v8_optA.pth'
HOOK_LAYERS = [4, 6, 10]
CHANNEL_SIZES = [128, 128, 256]

from train.train_dota_v8 import (SpikeAdaptSC_Det_Multi, build_v8_model,
    FeatureEnhancer, SNN_Hook_Enhanced)
from train.train_dota_v6d2d import SNN_Hook

def load_v8a():
    ck = torch.load(V8A_SAVE, map_location=device, weights_only=False)
    cfg = ck['config']
    snn = SpikeAdaptSC_Det_Multi(cfg['channel_sizes'], C_spike=cfg['C_spike'],
        T=cfg['T'], target_rate=cfg['target_rate']).to(device)
    snn.load_state_dict(ck['snn_state']); snn.eval()
    return ck, snn, cfg

def build_and_load(snn, ck, ber=0.0):
    yolo, extra = build_v8_model(snn, ber=ber)
    for k, s in ck.get('extra_modules', {}).items():
        if k in extra:
            try: extra[k].load_state_dict(s)
            except: pass
    for lk, s in ck.get('yolo_head_state', {}).items():
        i = int(lk.split('_')[1])
        if i < len(yolo.model.model):
            try: yolo.model.model[i].load_state_dict(s)
            except: pass
    return yolo, extra


# ############################################################################
# 1. ABLATION AT BER=0.20 AND BER=0.30
# ############################################################################

def ablation_noisy():
    from ultralytics import YOLO
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

    print("=" * 70)
    print("  1. Ablation Under Noise (BER=0.20 and BER=0.30)")
    print("=" * 70)

    ck, snn, cfg = load_v8a()
    hs = ck.get('yolo_head_state', {})
    results = {}

    for ber in [0.0, 0.20, 0.30]:
        print(f"\n  --- BER={ber:.2f} ---")
        
        # a) SNN only (no enhancers, no P2)
        yolo_snn = YOLO(BASELINE); yolo_snn.model.to(device)
        yolo_snn.model.fuse = lambda *a, **kw: yolo_snn.model
        for idx, (lid, lev) in enumerate(zip(HOOK_LAYERS, snn.levels)):
            lev_c = copy.deepcopy(lev); lev_c.eval()
            yolo_snn.model.model[lid] = SNN_Hook(
                yolo_snn.model.model[lid], lev_c, ber=ber).to(device)
        for lk, s in hs.items():
            i = int(lk.split('_')[1])
            if i < len(yolo_snn.model.model):
                try: yolo_snn.model.model[i].load_state_dict(s)
                except: pass
        r = yolo_snn.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0, verbose=False, plots=False)
        snn_map = float(r.results_dict.get('metrics/mAP50(B)', 0))
        print(f"    SNN only:        {snn_map:.4f}")

        # b) SNN + SE (no P2)
        yolo_se = YOLO(BASELINE); yolo_se.model.to(device)
        yolo_se.model.fuse = lambda *a, **kw: yolo_se.model
        for idx, (lid, lev) in enumerate(zip(HOOK_LAYERS, snn.levels)):
            lev_c = copy.deepcopy(lev); lev_c.eval()
            enh = FeatureEnhancer(CHANNEL_SIZES[idx]).to(device)
            yolo_se.model.model[lid] = SNN_Hook_Enhanced(
                yolo_se.model.model[lid], lev_c, ber=ber, enhancer=enh).to(device)
        for lk, s in hs.items():
            i = int(lk.split('_')[1])
            if i < len(yolo_se.model.model):
                try: yolo_se.model.model[i].load_state_dict(s)
                except: pass
        r = yolo_se.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0, verbose=False, plots=False)
        se_map = float(r.results_dict.get('metrics/mAP50(B)', 0))
        print(f"    SNN + SE:        {se_map:.4f}")

        # c) Full V8A  
        yolo_full, _ = build_and_load(snn, ck, ber=ber)
        r = yolo_full.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0, verbose=False, plots=False)
        full_map = float(r.results_dict.get('metrics/mAP50(B)', 0))
        print(f"    V8A (SNN+SE+P2): {full_map:.4f}")

        results[f'ber_{ber}'] = {
            'snn_only': snn_map, 'snn_se': se_map, 'v8a_full': full_map
        }

    # Print comparison
    print(f"\n  {'Config':<25s}  {'BER=0':>8s}  {'BER=0.20':>8s}  {'BER=0.30':>8s}  {'Degrad':>8s}")
    for key, label in [('snn_only', 'SNN only'), ('snn_se', 'SNN+SE'), ('v8a_full', 'V8A (Full)')]:
        v0 = results['ber_0.0'][key]
        v2 = results['ber_0.2'][key]
        v3 = results['ber_0.3'][key]
        deg = v3 - v0
        print(f"  {label:<25s}  {v0:8.4f}  {v2:8.4f}  {v3:8.4f}  {deg:+8.4f}")

    print(f"\n  SE contribution at BER=0.20: {results['ber_0.2']['snn_se'] - results['ber_0.2']['snn_only']:+.4f}")
    print(f"  P2 contribution at BER=0.20: {results['ber_0.2']['v8a_full'] - results['ber_0.2']['snn_se']:+.4f}")
    print(f"  SE contribution at BER=0.30: {results['ber_0.3']['snn_se'] - results['ber_0.3']['snn_only']:+.4f}")
    print(f"  P2 contribution at BER=0.30: {results['ber_0.3']['v8a_full'] - results['ber_0.3']['snn_se']:+.4f}")

    # Plot grouped bar chart
    configs = ['SNN only', 'SNN+SE', 'V8A (Full)']
    ber_labels = ['BER=0', 'BER=0.20', 'BER=0.30']
    x = np.arange(len(configs))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (ber_key, color) in enumerate(zip(['ber_0.0', 'ber_0.2', 'ber_0.3'], ['#4CAF50', '#FF9800', '#F44336'])):
        vals = [results[ber_key][k] for k in ['snn_only', 'snn_se', 'v8a_full']]
        bars = ax.bar(x + i*w, vals, w, label=ber_labels[i], color=color, alpha=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003, f'{val:.3f}',
                    ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x + w); ax.set_xticklabels(configs, fontsize=12)
    ax.set_ylabel('mAP@50', fontsize=13)
    ax.set_title('Ablation Under Noise: SE & P2 Bridge Contributions', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.3); ax.set_ylim(0, 0.45)
    plt.tight_layout()
    plt.savefig('paper/figures/v8a_ablation_noisy.png', dpi=200, bbox_inches='tight')
    plt.close()

    os.makedirs('eval', exist_ok=True)
    with open('eval/v8a_ablation_noisy.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved ablation chart and JSON")
    return results


# ############################################################################
# 2. SAHI WITH NMS DEDUPLICATION
# ############################################################################

def sahi_with_nms():
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    import cv2

    print("\n" + "=" * 70)
    print("  2. SAHI with NMS Deduplication")
    print("=" * 70)

    ck, snn, cfg = load_v8a()
    data_dict = check_det_dataset('DOTAv1.yaml')
    val_dir = Path(data_dict['val'])
    all_imgs = sorted(list(val_dir.glob('*.png')) + list(val_dir.glob('*.jpg')))

    # Dense scenes
    dense_names = ['P1384', 'P0961', 'P1821']
    dense_imgs = [str(img) for img in all_imgs if any(n in img.stem for n in dense_names)]
    if not dense_imgs:
        n = len(all_imgs)
        dense_imgs = [str(all_imgs[n//2])]

    def rotated_nms(boxes, confs, cls_ids, iou_thresh=0.3):
        """Simple center-distance based NMS for rotated boxes."""
        if len(boxes) == 0:
            return boxes, confs, cls_ids
        order = np.argsort(-confs)
        keep = []
        for i in order:
            discard = False
            for j in keep:
                if cls_ids[i] != cls_ids[j]:
                    continue
                dx = boxes[i, 0] - boxes[j, 0]
                dy = boxes[i, 1] - boxes[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                min_dim = min(boxes[i, 2], boxes[i, 3], boxes[j, 2], boxes[j, 3])
                if dist < min_dim * (1 - iou_thresh):
                    discard = True; break
            if not discard:
                keep.append(i)
        return boxes[keep], confs[keep], cls_ids[keep]

    def sahi_predict_nms(yolo_model, img_path, tile_size=320, overlap=0.25, conf=0.25):
        img = cv2.imread(img_path)
        H, W = img.shape[:2]
        stride = int(tile_size * (1 - overlap))
        all_b, all_c, all_cl = [], [], []
        for y0 in range(0, H, stride):
            for x0 in range(0, W, stride):
                y1, x1 = min(y0+tile_size, H), min(x0+tile_size, W)
                tile = img[y0:y1, x0:x1]
                r = yolo_model.predict(tile, imgsz=640, device=0, verbose=False, conf=conf)
                if r[0].obb is not None and len(r[0].obb) > 0:
                    b = r[0].obb.xywhr.cpu().numpy()
                    b[:, 0] += x0; b[:, 1] += y0
                    all_b.append(b)
                    all_c.append(r[0].obb.conf.cpu().numpy())
                    all_cl.append(r[0].obb.cls.cpu().numpy())
        if all_b:
            boxes = np.concatenate(all_b)
            confs = np.concatenate(all_c)
            cls_ids = np.concatenate(all_cl)
            # Apply NMS
            boxes_nms, confs_nms, cls_nms = rotated_nms(boxes, confs, cls_ids)
            return len(boxes), len(boxes_nms)
        return 0, 0

    results = {}
    for img_path in dense_imgs:
        name = Path(img_path).stem
        print(f"\n  {name}:")

        # Baseline
        yolo_base = YOLO(BASELINE)
        r = yolo_base.predict(img_path, imgsz=640, device=0, verbose=False, conf=0.25)
        n_base = len(r[0].obb) if r[0].obb is not None else 0

        # V8A + SAHI (raw + NMS)
        yolo_v8a, _ = build_and_load(snn, ck, ber=0.0)
        n_raw, n_nms = sahi_predict_nms(yolo_v8a, img_path)

        results[name] = {'baseline': n_base, 'sahi_raw': n_raw, 'sahi_nms': n_nms}
        print(f"    Baseline: {n_base}, SAHI raw: {n_raw}, SAHI+NMS: {n_nms}")

    with open('eval/v8a_sahi_nms.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ SAHI+NMS results saved")
    return results


# ############################################################################
# 3. SMALL OBJECT FOCUSED ANALYSIS
# ############################################################################

def small_object_analysis():
    from ultralytics import YOLO
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

    print("\n" + "=" * 70)
    print("  3. Small Object Analysis")
    print("=" * 70)

    ck, snn, cfg = load_v8a()

    # Small object categories in DOTA
    small_cats = ['small vehicle', 'ship', 'storage tank', 'swimming pool']
    large_cats = ['plane', 'baseball diamond', 'tennis court', 'ground track field']

    # Compare V8A with/without SAHI at different confidence thresholds
    from ultralytics.data.utils import check_det_dataset
    data_dict = check_det_dataset('DOTAv1.yaml')

    # V8A per-class at BER=0 vs BER=0.20
    yolo_clean, _ = build_and_load(snn, ck, ber=0.0)
    res_clean = yolo_clean.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0, verbose=False, plots=False)

    yolo_noisy, _ = build_and_load(snn, ck, ber=0.20)
    res_noisy = yolo_noisy.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0, verbose=False, plots=False)

    class_names = list(res_clean.names.values()) if hasattr(res_clean, 'names') else []
    clean_maps = res_clean.box.maps.tolist() if hasattr(res_clean.box, 'maps') else []
    noisy_maps = res_noisy.box.maps.tolist() if hasattr(res_noisy.box, 'maps') else []

    if clean_maps and noisy_maps:
        # Sort by noise resilience (noisy/clean ratio)
        ratios = []
        for i, name in enumerate(class_names):
            c = clean_maps[i] if clean_maps[i] > 0.001 else 0.001
            n = noisy_maps[i]
            ratios.append((name, clean_maps[i], noisy_maps[i], n/c))
        ratios.sort(key=lambda x: x[3], reverse=True)

        print(f"\n  {'Class':<22s}  {'Clean':>8s}  {'BER=0.20':>8s}  {'Retention':>10s}")
        for name, c, n, r in ratios:
            print(f"  {name:<22s}  {c:8.4f}  {n:8.4f}  {r*100:9.1f}%")

        # Categorize
        small_clean = np.mean([c for nm, c, n, r in ratios if nm in small_cats])
        small_noisy = np.mean([n for nm, c, n, r in ratios if nm in small_cats])
        large_clean = np.mean([c for nm, c, n, r in ratios if nm in large_cats])
        large_noisy = np.mean([n for nm, c, n, r in ratios if nm in large_cats])

        print(f"\n  Small objects avg: {small_clean:.4f} → {small_noisy:.4f} ({small_noisy/small_clean*100:.0f}% retention)")
        print(f"  Large objects avg: {large_clean:.4f} → {large_noisy:.4f} ({large_noisy/large_clean*100:.0f}% retention)")

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Per-class comparison
        names_sorted = [r[0] for r in ratios]
        clean_sorted = [r[1] for r in ratios]
        noisy_sorted = [r[2] for r in ratios]
        x = np.arange(len(names_sorted))
        ax1.barh(x, clean_sorted, 0.35, label='V8A (clean)', color='#4CAF50', alpha=0.8)
        ax1.barh(x+0.35, noisy_sorted, 0.35, label='V8A (BER=0.20)', color='#FF9800', alpha=0.8)
        ax1.set_yticks(x+0.175); ax1.set_yticklabels(names_sorted, fontsize=9)
        ax1.set_xlabel('mAP@50'); ax1.set_title('Per-Class Noise Resilience')
        ax1.legend(); ax1.invert_yaxis()

        # Right: Retention ratio
        ret_sorted = [r[3]*100 for r in ratios]
        colors = ['#4CAF50' if r > 90 else '#FF9800' if r > 70 else '#F44336' for r in ret_sorted]
        ax2.barh(x, ret_sorted, color=colors, alpha=0.8)
        ax2.set_yticks(x); ax2.set_yticklabels(names_sorted, fontsize=9)
        ax2.set_xlabel('Retention %'); ax2.set_title('mAP Retention Under BER=0.20')
        ax2.axvline(100, color='gray', linestyle='--', alpha=0.5)
        ax2.invert_yaxis()

        plt.tight_layout()
        plt.savefig('paper/figures/v8a_small_objects.png', dpi=200, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved: paper/figures/v8a_small_objects.png")

    results = {'class_names': class_names, 'clean': clean_maps, 'ber020': noisy_maps}
    with open('eval/v8a_small_objects.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Small object analysis done")


if __name__ == '__main__':
    ablation_noisy()
    sahi_with_nms()
    small_object_analysis()
    print("\n" + "=" * 70)
    print("  ALL 3 FOLLOW-UP ANALYSES COMPLETE!")
    print("=" * 70)
