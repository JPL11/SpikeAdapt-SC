#!/usr/bin/env python3
"""V8A Comprehensive Paper Analysis — 4 stages:
1. SAHI + V8A ultra-dense evaluation
2. Comparison visualizations across BER
3. Per-class analysis
4. Ablation + Latency benchmark
"""

import os, sys, json, time, copy
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
BASELINE = 'runs/obb/runs/dota/yolo26n_obb_baseline/weights/best.pt'
V8A_SAVE = 'runs/yolo26_snn_v8_optA.pth'
V8A_STAGE1 = 'runs/yolo26_snn_v8A_stage1.pth'

from train.train_dota_v8 import (
    SpikeAdaptSC_Det_Multi, build_v8_model, FeatureEnhancer,
    P2Bridge, HOOK_LAYERS, CHANNEL_SIZES, P2_LAYER, P2_CHANNELS,
    SNN_Hook_Enhanced, SpikeAdaptSC_Det,
)


def load_v8a():
    """Load V8A model components."""
    ck = torch.load(V8A_SAVE, map_location=device, weights_only=False)
    cfg = ck['config']
    snn = SpikeAdaptSC_Det_Multi(
        cfg['channel_sizes'], C_spike=cfg['C_spike'], T=cfg['T'],
        target_rate=cfg['target_rate']
    ).to(device)
    snn.load_state_dict(ck['snn_state'])
    snn.eval()
    return ck, snn, cfg


def build_and_load(snn, ck, ber=0.0):
    """Build V8A model and load weights."""
    yolo, extra = build_v8_model(snn, ber=ber)
    es = ck.get('extra_modules', {})
    for k, s in es.items():
        if k in extra:
            try: extra[k].load_state_dict(s)
            except: pass
    hs = ck.get('yolo_head_state', {})
    for lk, s in hs.items():
        i = int(lk.split('_')[1])
        if i < len(yolo.model.model):
            try: yolo.model.model[i].load_state_dict(s)
            except: pass
    return yolo, extra


# ############################################################################
# STAGE 1: SAHI + V8A on ultra-dense scenes
# ############################################################################

def stage1_sahi():
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from train.train_dota_v8 import yolo_forward_with_hooks

    print("=" * 70)
    print("  STAGE 1: SAHI + V8A Ultra-Dense Scene Evaluation")
    print("=" * 70)

    ck, snn, cfg = load_v8a()
    data_dict = check_det_dataset('DOTAv1.yaml')
    val_dir = Path(data_dict['val'])
    all_imgs = sorted(list(val_dir.glob('*.png')) + list(val_dir.glob('*.jpg')))

    # Find dense scenes (P1384 = airport)
    dense_imgs = []
    for img in all_imgs:
        if 'P1384' in img.stem or 'P0861' in img.stem or 'P2043' in img.stem:
            dense_imgs.append(str(img))
    # Also add some from spread
    n = len(all_imgs)
    for i in [0, n//6, n//3, n//2, 2*n//3, 5*n//6]:
        if i < n and str(all_imgs[i]) not in dense_imgs:
            dense_imgs.append(str(all_imgs[i]))

    # SAHI tiled inference function
    def sahi_predict(yolo_model, img_path, tile_size=320, overlap=0.25, conf=0.25):
        import cv2
        img = cv2.imread(img_path)
        H, W = img.shape[:2]
        stride = int(tile_size * (1 - overlap))
        all_boxes, all_confs, all_cls = [], [], []

        for y0 in range(0, H, stride):
            for x0 in range(0, W, stride):
                y1 = min(y0 + tile_size, H)
                x1 = min(x0 + tile_size, W)
                tile = img[y0:y1, x0:x1]
                
                r = yolo_model.predict(tile, imgsz=640, device=0, verbose=False, conf=conf)
                if r[0].obb is not None and len(r[0].obb) > 0:
                    boxes = r[0].obb.xywhr.cpu().numpy()
                    confs_t = r[0].obb.conf.cpu().numpy()
                    cls_t = r[0].obb.cls.cpu().numpy()
                    # Offset boxes to global coordinates
                    boxes[:, 0] += x0
                    boxes[:, 1] += y0
                    all_boxes.append(boxes)
                    all_confs.append(confs_t)
                    all_cls.append(cls_t)

        if all_boxes:
            return np.concatenate(all_boxes), np.concatenate(all_confs), np.concatenate(all_cls)
        return np.array([]), np.array([]), np.array([])

    os.makedirs('paper/figures', exist_ok=True)
    results = {}

    for img_path in dense_imgs[:6]:
        name = Path(img_path).stem
        print(f"\n  {name}:")

        # Baseline
        yolo_base = YOLO(BASELINE)
        r_base = yolo_base.predict(img_path, imgsz=640, device=0, verbose=False, conf=0.25)
        n_base = len(r_base[0].obb) if r_base[0].obb is not None else 0

        # V8A standard
        yolo_v8a, _ = build_and_load(snn, ck, ber=0.0)
        r_v8a = yolo_v8a.predict(img_path, imgsz=640, device=0, verbose=False, conf=0.25)
        n_v8a = len(r_v8a[0].obb) if r_v8a[0].obb is not None else 0

        # V8A + SAHI
        yolo_sahi, _ = build_and_load(snn, ck, ber=0.0)
        sahi_boxes, sahi_confs, sahi_cls = sahi_predict(yolo_sahi, img_path)
        n_sahi = len(sahi_boxes)

        results[name] = {'baseline': n_base, 'v8a': n_v8a, 'v8a_sahi': n_sahi}
        print(f"    Baseline: {n_base}, V8A: {n_v8a}, V8A+SAHI: {n_sahi}")

        # Visualization
        import cv2
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        ax = axes[0]
        ax.imshow(r_base[0].plot()[:, :, ::-1])
        ax.set_title(f'Baseline\n({n_base} det)', fontsize=11, fontweight='bold')
        ax.axis('off')

        ax = axes[1]
        ax.imshow(r_v8a[0].plot()[:, :, ::-1])
        ax.set_title(f'V8A (SNN)\n({n_v8a} det)', fontsize=11, fontweight='bold')
        ax.axis('off')

        ax = axes[2]
        img_cv = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(f'V8A+SAHI\n({n_sahi} det)', fontsize=11, fontweight='bold')
        ax.axis('off')

        plt.suptitle(f'Ultra-Dense Scene: {name}', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'paper/figures/v8a_sahi_{name}.png', dpi=200, bbox_inches='tight')
        plt.close()

    with open('eval/v8a_sahi_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Stage 1 done. Saved to eval/v8a_sahi_results.json")


# ############################################################################
# STAGE 2: Comparison visualizations across BER levels
# ############################################################################

def stage2_viz():
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("\n" + "=" * 70)
    print("  STAGE 2: V8A Detection Visualization Across BER")
    print("=" * 70)

    ck, snn, cfg = load_v8a()
    data_dict = check_det_dataset('DOTAv1.yaml')
    val_dir = Path(data_dict['val'])
    all_imgs = sorted(list(val_dir.glob('*.png')) + list(val_dir.glob('*.jpg')))
    n = len(all_imgs)
    indices = [0, n//6, n//3, n//2, 2*n//3, 5*n//6]
    test_imgs = [str(all_imgs[i]) for i in indices if i < n]

    ber_levels = [0.0, 0.10, 0.20, 0.30]

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
            yolo, _ = build_and_load(snn, ck, ber=ber)
            r = yolo.predict(img_path, imgsz=640, device=0, verbose=False, conf=0.25)
            ax = axes[bi + 1]
            ax.imshow(r[0].plot()[:, :, ::-1])
            nb = len(r[0].obb) if r[0].obb is not None else 0
            ax.set_title(f'V8A\nBER={ber:.2f} ({nb} det)', fontsize=10, fontweight='bold')
            ax.axis('off')

        plt.suptitle(f'V8A Detection: {name}', fontsize=13, fontweight='bold')
        plt.tight_layout()
        out = f'paper/figures/v8a_ber_viz_{img_idx+1}.png'
        plt.savefig(out, dpi=200, bbox_inches='tight')
        print(f"  Saved: {out}")
        plt.close()

    print(f"\n✅ Stage 2 done. {len(test_imgs)} visualizations saved.")


# ############################################################################
# STAGE 3: Per-class analysis
# ############################################################################

def stage3_perclass():
    from ultralytics import YOLO
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("\n" + "=" * 70)
    print("  STAGE 3: Per-Class mAP Analysis")
    print("=" * 70)

    ck, snn, cfg = load_v8a()

    # Baseline per-class
    yolo_base = YOLO(BASELINE)
    base_res = yolo_base.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0, verbose=False, plots=False)
    base_map = float(base_res.results_dict.get('metrics/mAP50(B)', 0))
    
    # Get class names
    class_names = list(base_res.names.values()) if hasattr(base_res, 'names') else [f'cls_{i}' for i in range(15)]
    base_per_class = base_res.box.maps.tolist() if hasattr(base_res.box, 'maps') else []

    # V8A at clean
    yolo_v8a, _ = build_and_load(snn, ck, ber=0.0)
    v8a_res = yolo_v8a.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0, verbose=False, plots=False)
    v8a_per_class = v8a_res.box.maps.tolist() if hasattr(v8a_res.box, 'maps') else []

    # V8A at BER=0.20
    yolo_v8a_noisy, _ = build_and_load(snn, ck, ber=0.20)
    v8a_noisy_res = yolo_v8a_noisy.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0, verbose=False, plots=False)
    v8a_noisy_per_class = v8a_noisy_res.box.maps.tolist() if hasattr(v8a_noisy_res.box, 'maps') else []

    results = {
        'class_names': class_names,
        'baseline': base_per_class,
        'v8a_clean': v8a_per_class,
        'v8a_ber020': v8a_noisy_per_class,
    }

    # Print table
    print(f"\n  {'Class':<20s}  {'Baseline':>8s}  {'V8A':>8s}  {'V8A@0.20':>8s}  {'Δ(clean)':>8s}")
    for i, name in enumerate(class_names):
        b = base_per_class[i] if i < len(base_per_class) else 0
        v = v8a_per_class[i] if i < len(v8a_per_class) else 0
        vn = v8a_noisy_per_class[i] if i < len(v8a_noisy_per_class) else 0
        d = v - b
        print(f"  {name:<20s}  {b:8.4f}  {v:8.4f}  {vn:8.4f}  {d:+8.4f}")

    # Plot
    if base_per_class and v8a_per_class:
        x = np.arange(len(class_names))
        w = 0.25
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(x - w, base_per_class, w, label='Baseline', color='#2196F3', alpha=0.8)
        ax.bar(x, v8a_per_class, w, label='V8A (clean)', color='#4CAF50', alpha=0.8)
        if v8a_noisy_per_class:
            ax.bar(x + w, v8a_noisy_per_class, w, label='V8A (BER=0.20)', color='#FF9800', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('mAP@50', fontsize=12)
        ax.set_title('Per-Class Detection: Baseline vs V8A', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('paper/figures/v8a_perclass.png', dpi=200, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved: paper/figures/v8a_perclass.png")

    with open('eval/v8a_perclass.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Stage 3 done.")


# ############################################################################
# STAGE 4: Ablation + Latency
# ############################################################################

def stage4_ablation():
    from ultralytics import YOLO
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("\n" + "=" * 70)
    print("  STAGE 4: Ablation Study + Latency Benchmark")
    print("=" * 70)

    ck, snn, cfg = load_v8a()

    # --- Latency Benchmark ---
    print("\n  --- Latency Benchmark ---")
    from ultralytics.data.utils import check_det_dataset
    data_dict = check_det_dataset('DOTAv1.yaml')
    val_dir = Path(data_dict['val'])
    test_img = str(sorted(val_dir.glob('*.png'))[0])

    # Baseline latency
    yolo_base = YOLO(BASELINE)
    # Warmup
    for _ in range(3):
        yolo_base.predict(test_img, imgsz=640, device=0, verbose=False)
    t0 = time.time()
    for _ in range(20):
        yolo_base.predict(test_img, imgsz=640, device=0, verbose=False)
    base_latency = (time.time() - t0) / 20 * 1000

    # V8A latency
    yolo_v8a, _ = build_and_load(snn, ck, ber=0.0)
    for _ in range(3):
        yolo_v8a.predict(test_img, imgsz=640, device=0, verbose=False)
    t0 = time.time()
    for _ in range(20):
        yolo_v8a.predict(test_img, imgsz=640, device=0, verbose=False)
    v8a_latency = (time.time() - t0) / 20 * 1000

    print(f"  Baseline: {base_latency:.1f} ms/image")
    print(f"  V8A:      {v8a_latency:.1f} ms/image")
    print(f"  Overhead: {v8a_latency - base_latency:.1f} ms ({(v8a_latency/base_latency - 1)*100:.0f}%)")

    # --- Ablation: V8A without P2 (SE enhancers only on P3/P4/P5) ---
    print("\n  --- Ablation: V8A without P2 Bridge ---")
    # Build model without P2 hook — just SE enhancers on P3/P4/P5
    from train.train_dota_v8 import SNN_Hook_Enhanced
    yolo_no_p2 = YOLO(BASELINE)
    yolo_no_p2.model.to(device)
    yolo_no_p2.model.fuse = lambda *a, **kw: yolo_no_p2.model
    
    for idx, (lid, lev) in enumerate(zip(HOOK_LAYERS, snn.levels)):
        lev_copy = copy.deepcopy(lev); lev_copy.eval()
        enhancer = FeatureEnhancer(CHANNEL_SIZES[idx]).to(device)
        yolo_no_p2.model.model[lid] = SNN_Hook_Enhanced(
            yolo_no_p2.model.model[lid], lev_copy, ber=0.0, enhancer=enhancer
        ).to(device)
    
    # Load only head weights (skip extra modules)
    hs = ck.get('yolo_head_state', {})
    for lk, s in hs.items():
        i = int(lk.split('_')[1])
        if i < len(yolo_no_p2.model.model):
            try: yolo_no_p2.model.model[i].load_state_dict(s)
            except: pass
    # Load SE enhancer weights from extra_modules
    es = ck.get('extra_modules', {})
    if 'enhancers' in es:
        enh_state = es['enhancers']
        for idx, lid in enumerate(HOOK_LAYERS):
            key = f'e{idx}'
            if key in enh_state:
                hook = yolo_no_p2.model.model[lid]
                if hasattr(hook, 'enhancer') and hook.enhancer is not None:
                    try: hook.enhancer.load_state_dict({k.replace(f'{key}.', ''): v for k, v in enh_state.items() if k.startswith(key)})
                    except: pass

    no_p2_res = yolo_no_p2.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0, verbose=False, plots=False)
    no_p2_map = float(no_p2_res.results_dict.get('metrics/mAP50(B)', 0))
    print(f"  V8A without P2: mAP@50={no_p2_map:.4f}")

    # --- Ablation: SNN only (no enhancers, no P2) ---
    print("\n  --- Ablation: SNN only (no enhancers) ---")
    from train.train_dota_v6d2d import SNN_Hook
    yolo_snn_only = YOLO(BASELINE)
    yolo_snn_only.model.to(device)
    yolo_snn_only.model.fuse = lambda *a, **kw: yolo_snn_only.model
    for idx, (lid, lev) in enumerate(zip(HOOK_LAYERS, snn.levels)):
        lev_copy = copy.deepcopy(lev); lev_copy.eval()
        yolo_snn_only.model.model[lid] = SNN_Hook(
            yolo_snn_only.model.model[lid], lev_copy, ber=0.0
        ).to(device)
    for lk, s in hs.items():
        i = int(lk.split('_')[1])
        if i < len(yolo_snn_only.model.model):
            try: yolo_snn_only.model.model[i].load_state_dict(s)
            except: pass
    
    snn_only_res = yolo_snn_only.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0, verbose=False, plots=False)
    snn_only_map = float(snn_only_res.results_dict.get('metrics/mAP50(B)', 0))
    print(f"  SNN only: mAP@50={snn_only_map:.4f}")

    # Full V8A for reference
    yolo_full, _ = build_and_load(snn, ck, ber=0.0)
    full_res = yolo_full.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0, verbose=False, plots=False)
    full_map = float(full_res.results_dict.get('metrics/mAP50(B)', 0))

    # Baseline
    base_res = yolo_base.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0, verbose=False, plots=False)
    base_map = float(base_res.results_dict.get('metrics/mAP50(B)', 0))

    ablation = {
        'baseline': {'mAP50': base_map, 'latency_ms': base_latency},
        'snn_only': {'mAP50': snn_only_map, 'desc': 'MaxFormer SNN, no enhancers'},
        'snn_se': {'mAP50': no_p2_map, 'desc': 'SNN + SE enhancers, no P2'},
        'v8a_full': {'mAP50': full_map, 'latency_ms': v8a_latency, 'desc': 'Full V8A: SNN + SE + P2'},
    }

    print(f"\n  --- Ablation Summary ---")
    print(f"  {'Config':<30s}  {'mAP@50':>8s}  {'Δ vs Base':>10s}")
    print(f"  {'Baseline (no SNN)':<30s}  {base_map:8.4f}  {'—':>10s}")
    print(f"  {'SNN only (MaxFormer)':<30s}  {snn_only_map:8.4f}  {snn_only_map-base_map:+10.4f}")
    print(f"  {'SNN + SE enhancers':<30s}  {no_p2_map:8.4f}  {no_p2_map-base_map:+10.4f}")
    print(f"  {'V8A full (SNN+SE+P2)':<30s}  {full_map:8.4f}  {full_map-base_map:+10.4f}")
    print(f"\n  P2 Bridge contribution: {full_map - no_p2_map:+.4f}")
    print(f"  SE Enhancer contribution: {no_p2_map - snn_only_map:+.4f}")

    # Save
    with open('eval/v8a_ablation.json', 'w') as f:
        json.dump(ablation, f, indent=2)

    # Plot ablation bar chart
    configs = ['Baseline', 'SNN only', 'SNN+SE', 'V8A (Full)']
    maps = [base_map, snn_only_map, no_p2_map, full_map]
    colors = ['#757575', '#2196F3', '#4CAF50', '#FF5722']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(configs, maps, color=colors, alpha=0.85, edgecolor='white', linewidth=2)
    for bar, val in zip(bars, maps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_ylabel('mAP@50', fontsize=13)
    ax.set_title('V8A Ablation Study (BER=0)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 0.5)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('paper/figures/v8a_ablation.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: paper/figures/v8a_ablation.png")
    print(f"✅ Stage 4 done.")


if __name__ == '__main__':
    stage1_sahi()
    stage2_viz()
    stage3_perclass()
    stage4_ablation()
    print("\n" + "=" * 70)
    print("  ALL 4 STAGES COMPLETE!")
    print("=" * 70)
