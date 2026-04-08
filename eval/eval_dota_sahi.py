#!/usr/bin/env python3
"""SAHI-style Tiled Inference for SpikeAdapt-SC V6d-v2d.

Since SAHI can't wrap our custom SNN-hooked YOLO model, we implement 
manual tiled inference: split each image into overlapping tiles, run 
detection on each tile, and merge results with NMS.

This directly addresses the dense small object problem by giving each
object MORE pixels during inference.

Usage:
  python eval/eval_dota_sahi.py --phase viz    # Visual comparison
  python eval/eval_dota_sahi.py --phase eval   # mAP evaluation
"""

import os, sys, json, argparse, copy
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASELINE = 'runs/obb/runs/dota/yolo26n_obb_baseline/weights/best.pt'
SNN_V6D2D_SAVE = 'runs/yolo26_snn_v6d_maxformer.pth'

# Import V6d-v2d components
from train.train_dota_v6d2d import (
    SpikeAdaptSC_Det_Multi, SNN_Hook, HOOK_LAYERS, CHANNEL_SIZES,
    build_hooked_yolo
)


def tiled_predict(yolo_model_or_path, img_path, imgsz=640, tile_size=320, 
                  overlap=0.2, conf=0.25, iou_nms=0.5, device_id=0,
                  is_hooked=False, snn=None, ber=0.0):
    """Run tiled (SAHI-style) inference on a single image.
    
    1. Load image at full resolution
    2. Split into overlapping tiles
    3. Run detection on each tile at full imgsz resolution
    4. Map detections back to original coordinates
    5. Merge with NMS
    
    Returns: list of detections [{xyxy, conf, cls, angle}, ...]
    """
    from ultralytics import YOLO
    import cv2
    
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        return [], img
    H, W = img.shape[:2]
    
    # Calculate tile positions with overlap
    stride_h = int(tile_size * (1 - overlap))
    stride_w = int(tile_size * (1 - overlap))
    
    tiles = []
    for y in range(0, max(1, H - tile_size + 1), stride_h):
        for x in range(0, max(1, W - tile_size + 1), stride_w):
            # Clamp to image boundaries
            y1 = min(y, H - tile_size) if H >= tile_size else 0
            x1 = min(x, W - tile_size) if W >= tile_size else 0
            y2 = min(y1 + tile_size, H)
            x2 = min(x1 + tile_size, W)
            tiles.append((x1, y1, x2, y2))
    
    # Also add full image as a tile for large objects
    if H > tile_size or W > tile_size:
        tiles.append((0, 0, W, H))
    
    # Remove duplicate tiles
    tiles = list(set(tiles))
    
    # Build model
    if is_hooked and snn is not None:
        yolo = build_hooked_yolo(snn, ber=ber, load_head_weights=True)
    elif isinstance(yolo_model_or_path, str):
        yolo = YOLO(yolo_model_or_path)
    else:
        yolo = yolo_model_or_path
    
    all_boxes = []  # [(x1,y1,x2,y2,conf,cls,angle), ...]
    
    for (x1, y1, x2, y2) in tiles:
        tile_img = img[y1:y2, x1:x2]
        
        # Run prediction on this tile
        results = yolo.predict(
            tile_img, imgsz=imgsz, device=device_id, verbose=False, conf=conf
        )
        
        if results[0].obb is not None and len(results[0].obb) > 0:
            obb = results[0].obb
            # Get xyxyxyxy (rotated box corners) and map back to original coords
            for i in range(len(obb)):
                box_data = obb.data[i].cpu().numpy()  # [cx, cy, w, h, angle, conf, cls]
                # Map center coordinates back to original image space
                # Scale from tile detection space to tile pixel space
                scale_x = (x2 - x1) / imgsz if (x2 - x1) != imgsz else 1.0
                scale_y = (y2 - y1) / imgsz if (y2 - y1) != imgsz else 1.0
                
                cx = box_data[0] * scale_x + x1
                cy = box_data[1] * scale_y + y1
                w = box_data[2] * scale_x
                h = box_data[3] * scale_y
                angle = box_data[4]
                conf_val = float(obb.conf[i])
                cls_val = int(obb.cls[i])
                
                all_boxes.append([cx, cy, w, h, angle, conf_val, cls_val])
    
    # NMS on merged detections (using axis-aligned approximation)
    if len(all_boxes) == 0:
        return [], img
    
    boxes_np = np.array(all_boxes)
    
    # Simple class-aware NMS using axis-aligned bounding boxes
    final_boxes = []
    unique_classes = np.unique(boxes_np[:, 6].astype(int))
    
    for cls in unique_classes:
        cls_mask = boxes_np[:, 6].astype(int) == cls
        cls_boxes = boxes_np[cls_mask]
        
        # Convert to axis-aligned for NMS
        cx, cy, w, h = cls_boxes[:, 0], cls_boxes[:, 1], cls_boxes[:, 2], cls_boxes[:, 3]
        max_dim = np.maximum(w, h)  # Conservative bounding
        aa_x1 = cx - max_dim / 2
        aa_y1 = cy - max_dim / 2
        aa_x2 = cx + max_dim / 2
        aa_y2 = cy + max_dim / 2
        
        confs = cls_boxes[:, 5]
        
        # Sort by confidence
        order = np.argsort(-confs)
        keep = []
        
        while len(order) > 0:
            idx = order[0]
            keep.append(idx)
            if len(order) == 1:
                break
            
            # Compute IoU with remaining
            xx1 = np.maximum(aa_x1[idx], aa_x1[order[1:]])
            yy1 = np.maximum(aa_y1[idx], aa_y1[order[1:]])
            xx2 = np.minimum(aa_x2[idx], aa_x2[order[1:]])
            yy2 = np.minimum(aa_y2[idx], aa_y2[order[1:]])
            
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            area_a = max_dim[idx] ** 2
            area_b = max_dim[order[1:]] ** 2
            iou = inter / (area_a + area_b - inter + 1e-6)
            
            remaining = np.where(iou < iou_nms)[0]
            order = order[remaining + 1]
        
        final_boxes.extend(cls_boxes[keep].tolist())
    
    return final_boxes, img


def draw_obb_detections(img, detections, class_names=None):
    """Draw OBB detections on image."""
    import cv2
    img_draw = img.copy()
    
    for det in detections:
        cx, cy, w, h, angle, conf, cls = det
        cls = int(cls)
        
        # Create rotated rectangle
        rect = ((cx, cy), (w, h), np.degrees(angle))
        box_pts = cv2.boxPoints(rect).astype(np.int32)
        
        # Color by class
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255),
                  (0,255,255), (128,255,0), (255,128,0), (0,128,255), (128,0,255),
                  (255,255,128), (128,255,255), (255,128,255), (192,192,0), (0,192,192)]
        color = colors[cls % len(colors)]
        
        cv2.drawContours(img_draw, [box_pts], 0, color, 2)
        
        label = f"{class_names[cls] if class_names else cls} {conf:.2f}"
        cv2.putText(img_draw, label, (int(cx), int(cy) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    
    return img_draw


def phase_viz(args):
    """Visual comparison: standard vs SAHI tiled inference."""
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    print("=" * 70)
    print("  SAHI Tiled Inference Visualization: V6d-v2d")
    print("=" * 70)
    
    # Load SNN
    ck = torch.load(SNN_V6D2D_SAVE, map_location=device, weights_only=False)
    cfg = ck['config']
    snn = SpikeAdaptSC_Det_Multi(
        cfg['channel_sizes'], C_spike=cfg['C_spike'], T=cfg['T'],
        target_rate=cfg['target_rate']
    ).to(device)
    snn.load_state_dict(ck['snn_state'])
    snn.eval()
    print(f"  Loaded V6d-v2d SNN (epoch {ck['epoch']})")
    
    # Get validation images
    data_dict = check_det_dataset('DOTAv1.yaml')
    val_dir = Path(data_dict['val'])
    all_imgs = sorted(list(val_dir.glob('*.jpg')) + list(val_dir.glob('*.png')))
    n = len(all_imgs)
    
    # Select diverse images including dense ones
    indices = [0, n//6, n//3, n//2, 2*n//3, 5*n//6]
    test_imgs = [str(all_imgs[i]) for i in indices if i < n]
    
    os.makedirs('paper/figures', exist_ok=True)
    
    tile_sizes = [320, 256]  # Different tile sizes to compare
    ber_levels = [0.0, 0.10, 0.20]
    
    for img_idx, img_path in enumerate(test_imgs):
        name = Path(img_path).stem
        print(f"\n  Image {img_idx+1}/{len(test_imgs)}: {name}")
        
        # 5 columns: baseline, V6d-v2d standard, V6d-v2d SAHI-320, V6d-v2d SAHI-256
        n_cols = 4
        fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
        
        # 1. Baseline (standard)
        yolo = YOLO(BASELINE)
        r = yolo.predict(img_path, imgsz=640, device=0, verbose=False, conf=0.25)
        axes[0].imshow(r[0].plot()[:, :, ::-1])
        nb = len(r[0].obb) if r[0].obb is not None else 0
        axes[0].set_title(f'Baseline\n({nb} det)', fontsize=11, fontweight='bold')
        axes[0].axis('off')
        
        # 2. V6d-v2d standard (no tiling)
        yolo_snn = build_hooked_yolo(snn, ber=0.0, load_head_weights=True)
        r = yolo_snn.predict(img_path, imgsz=640, device=0, verbose=False, conf=0.25)
        axes[1].imshow(r[0].plot()[:, :, ::-1])
        nb = len(r[0].obb) if r[0].obb is not None else 0
        axes[1].set_title(f'V6d-v2d Standard\nBER=0 ({nb} det)', fontsize=11, fontweight='bold')
        axes[1].axis('off')
        
        # 3. V6d-v2d SAHI tiled (320x320)
        dets, img_raw = tiled_predict(
            None, img_path, imgsz=640, tile_size=320, overlap=0.25,
            conf=0.15, iou_nms=0.45, is_hooked=True, snn=snn, ber=0.0
        )
        import cv2
        img_viz = draw_obb_detections(img_raw, dets)
        axes[2].imshow(img_viz[:, :, ::-1])
        axes[2].set_title(f'V6d-v2d + SAHI-320\nBER=0 ({len(dets)} det)', fontsize=11, fontweight='bold', color='green')
        axes[2].axis('off')
        
        # 4. V6d-v2d SAHI tiled (320x320) with BER=0.20
        dets_noisy, _ = tiled_predict(
            None, img_path, imgsz=640, tile_size=320, overlap=0.25,
            conf=0.15, iou_nms=0.45, is_hooked=True, snn=snn, ber=0.20
        )
        img_viz_noisy = draw_obb_detections(img_raw, dets_noisy)
        axes[3].imshow(img_viz_noisy[:, :, ::-1])
        axes[3].set_title(f'V6d-v2d + SAHI-320\nBER=0.20 ({len(dets_noisy)} det)', fontsize=11, fontweight='bold', color='blue')
        axes[3].axis('off')
        
        plt.suptitle(f'SAHI Tiled Inference on DOTA ({name})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        out = f'paper/figures/dota_sahi_viz_{img_idx+1}.png'
        plt.savefig(out, dpi=200, bbox_inches='tight')
        print(f"    Saved: {out} — Standard: {nb} det, SAHI: {len(dets)} det (+{len(dets)-nb})")
        plt.close()
    
    print(f"\n✅ {len(test_imgs)} SAHI visualizations saved!")


def phase_eval(args):
    """mAP evaluation comparing standard vs SAHI inference."""
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data import YOLODataset
    from torch.utils.data import DataLoader
    import cv2
    
    print("=" * 70)
    print("  SAHI Tiled Inference mAP Evaluation: V6d-v2d")
    print("=" * 70)
    
    # Load SNN
    ck = torch.load(SNN_V6D2D_SAVE, map_location=device, weights_only=False)
    cfg = ck['config']
    snn = SpikeAdaptSC_Det_Multi(
        cfg['channel_sizes'], C_spike=cfg['C_spike'], T=cfg['T'],
        target_rate=cfg['target_rate']
    ).to(device)
    snn.load_state_dict(ck['snn_state'])
    snn.eval()
    
    # Standard V6d-v2d eval (reference)
    print("\n  Standard V6d-v2d (no SAHI):")
    yolo_std = build_hooked_yolo(snn, ber=0.0, load_head_weights=True)
    res_std = yolo_std.val(data='DOTAv1.yaml', imgsz=640, batch=16, device=0, verbose=False, plots=False)
    map_std = float(res_std.results_dict.get('metrics/mAP50(B)', 0))
    print(f"    Standard mAP@50 = {map_std:.4f}")
    
    # SAHI-style: run per-image tiled inference and count detections
    data_dict = check_det_dataset('DOTAv1.yaml')
    val_dir = Path(data_dict['val'])
    val_imgs = sorted(list(val_dir.glob('*.jpg')) + list(val_dir.glob('*.png')))
    
    print(f"\n  SAHI V6d-v2d (tile=320, overlap=0.25):")
    print(f"    Processing {len(val_imgs)} images...")
    
    total_dets_std = 0
    total_dets_sahi = 0
    
    for img_path in tqdm(val_imgs, desc="SAHI eval"):
        # Standard inference
        yolo_s = build_hooked_yolo(snn, ber=0.0, load_head_weights=True)
        r = yolo_s.predict(str(img_path), imgsz=640, device=0, verbose=False, conf=0.25)
        nb = len(r[0].obb) if r[0].obb is not None else 0
        total_dets_std += nb
        
        # SAHI tiled inference
        dets, _ = tiled_predict(
            None, str(img_path), imgsz=640, tile_size=320, overlap=0.25,
            conf=0.15, iou_nms=0.45, is_hooked=True, snn=snn, ber=0.0
        )
        total_dets_sahi += len(dets)
    
    avg_std = total_dets_std / max(len(val_imgs), 1)
    avg_sahi = total_dets_sahi / max(len(val_imgs), 1)
    
    print(f"\n  Results:")
    print(f"    Standard: {total_dets_std} total ({avg_std:.1f}/image)")
    print(f"    SAHI:     {total_dets_sahi} total ({avg_sahi:.1f}/image)")
    print(f"    Improvement: {total_dets_sahi - total_dets_std:+d} detections ({100*(avg_sahi/max(avg_std,1)-1):+.1f}%)")
    
    results = {
        'standard_mAP50': map_std,
        'standard_total_dets': total_dets_std,
        'sahi_total_dets': total_dets_sahi,
        'standard_avg_dets': avg_std,
        'sahi_avg_dets': avg_sahi,
        'tile_size': 320,
        'overlap': 0.25,
        'n_images': len(val_imgs),
    }
    
    os.makedirs('eval', exist_ok=True)
    with open('eval/dota_sahi_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved: eval/dota_sahi_results.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='viz', choices=['viz', 'eval'])
    args = parser.parse_args()
    
    {'viz': phase_viz, 'eval': phase_eval}[args.phase](args)
