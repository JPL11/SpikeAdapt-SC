#!/usr/bin/env python3
"""End-to-end mAP evaluation: YOLOv8 + SNN bottleneck on DOTA.

Evaluates the full detection pipeline across BER values:
  Backbone → SNN encode → BSC(ber) → SNN decode → Neck+Head → NMS → mAP

Also evaluates baseline (no SNN) and SNN clean (BER=0) for comparison.

Output: eval/dota_detection_results.json
"""

import os, sys, json, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))

from train_dota_snn_v2 import YOLO_SNN_Wrapper, bsc_channel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_yolo_with_snn(wrapper, data_yaml, ber=0.0, imgsz=640):
    """Evaluate mAP by hooking SNN into YOLO's validation pipeline.
    
    Strategy: modify the backbone output in-place during YOLO's forward,
    injecting SNN encode→channel→decode at the feature level.
    """
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data import YOLODataset
    from torch.utils.data import DataLoader
    
    data_dict = check_det_dataset(data_yaml)
    
    val_dataset = YOLODataset(
        img_path=data_dict['val'],
        imgsz=imgsz,
        augment=False,
        data=data_dict,
        task='obb',
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=YOLODataset.collate_fn,
    )
    
    wrapper.eval()
    
    # Collect feature MSE and firing rate statistics
    total_mse = 0
    total_fr = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch_data in val_loader:
            imgs = batch_data['img'].to(device).float() / 255.0
            
            # Get original features
            features_orig, _ = wrapper.forward_backbone(imgs)
            
            # Get SNN-reconstructed features
            features_snn, info = wrapper.snn(features_orig, ber=ber)
            
            mse = sum(
                F.mse_loss(r, o) for r, o in zip(features_snn, features_orig)
            ).item()
            total_mse += mse
            total_fr += info['avg_fr']
            n_batches += 1
    
    avg_mse = total_mse / max(n_batches, 1)
    avg_fr = total_fr / max(n_batches, 1)
    
    return avg_mse, avg_fr


def evaluate_mAP_with_snn(baseline_path, snn_ckpt_path, ber=0.0, c_bot=36, T=4):
    """Evaluate actual mAP by modifying YOLO's forward pass.
    
    We monkey-patch the backbone's output to inject SNN encoding/decoding.
    """
    from ultralytics import YOLO
    
    # Load a fresh YOLO model for validation
    yolo = YOLO(baseline_path)
    
    # Create wrapper and load SNN weights
    wrapper = YOLO_SNN_Wrapper(yolo, C_bot=c_bot, T=T).to(device)
    snn_ck = torch.load(snn_ckpt_path, map_location=device, weights_only=False)
    wrapper.snn.load_state_dict(snn_ck['snn_state'])
    wrapper.eval()
    
    # Monkey-patch the YOLO model to inject SNN bottleneck
    original_model = yolo.model.model
    
    # Save references to layers we need to intercept
    layer4_orig = original_model[4]   # P3 output
    layer6_orig = original_model[6]   # P4 output  
    layer9_orig = original_model[9]   # P5 output (SPPF)
    
    # We need to intercept the outputs of layers 4, 6, 9 and pass them
    # through SNN encode → BSC → decode
    
    class SNN_Hook_Layer(nn.Module):
        """Wraps a YOLO layer to inject SNN bottleneck after it."""
        def __init__(self, original_layer, snn_encoder, snn_decoder, ber):
            super().__init__()
            self.original_layer = original_layer
            self.snn_encoder = snn_encoder
            self.snn_decoder = snn_decoder
            self.ber = ber
            self.last_fr = 0
            # Copy YOLO-required routing attributes
            self.f = getattr(original_layer, 'f', -1)
            self.i = getattr(original_layer, 'i', 0)
            self.type = getattr(original_layer, 'type', type(original_layer).__name__)
            self.np = getattr(original_layer, 'np', 0)
        
        def forward(self, x):
            # Run original layer
            out = self.original_layer(x)
            
            # SNN encode → channel → decode
            spikes = self.snn_encoder(out)
            
            with torch.no_grad():
                self.last_fr = torch.stack(spikes).mean().item()
            
            received = [bsc_channel(s, self.ber) for s in spikes]
            reconstructed = self.snn_decoder(received)
            
            return reconstructed
    
    # Replace layers 4, 6, 9 with hooked versions
    original_model[4] = SNN_Hook_Layer(
        layer4_orig, wrapper.snn.encoders[0], wrapper.snn.decoders[0], ber
    ).to(device)
    original_model[6] = SNN_Hook_Layer(
        layer6_orig, wrapper.snn.encoders[1], wrapper.snn.decoders[1], ber
    ).to(device)
    original_model[9] = SNN_Hook_Layer(
        layer9_orig, wrapper.snn.encoders[2], wrapper.snn.decoders[2], ber
    ).to(device)
    
    # Run YOLO validation with SNN-hooked model
    results = yolo.val(
        data='DOTAv1.yaml',
        imgsz=640,
        batch=16,
        device=0,
        verbose=False,
        plots=False,
    )
    
    # Extract mAP metrics
    map50 = results.results_dict.get('metrics/mAP50(B)', 0)
    map50_95 = results.results_dict.get('metrics/mAP50-95(B)', 0)
    
    # Get per-class AP
    per_class = {}
    if hasattr(results, 'ap_class_index') and hasattr(results, 'maps'):
        names = results.names
        for i, cls_idx in enumerate(results.ap_class_index):
            per_class[names[int(cls_idx)]] = float(results.maps[i])
    
    # Get firing rates
    frs = []
    for layer_idx in [4, 6, 9]:
        hook_layer = original_model[layer_idx]
        if hasattr(hook_layer, 'last_fr'):
            frs.append(hook_layer.last_fr)
    
    # Restore original layers
    original_model[4] = layer4_orig
    original_model[6] = layer6_orig
    original_model[9] = layer9_orig
    
    return {
        'mAP50': float(map50),
        'mAP50_95': float(map50_95),
        'avg_firing_rate': float(np.mean(frs)) if frs else 0,
        'per_class': per_class,
    }


def main():
    print("=" * 70)
    print("  DOTA DETECTION: mAP vs BER EVALUATION")
    print("=" * 70)
    
    baseline_path = 'runs/obb/runs/dota/baseline_yolov8n_obb/weights/best.pt'
    snn_ckpt_path = 'runs/dota_snn_bottleneck_best.pth'
    
    if not os.path.exists(baseline_path):
        print(f"ERROR: Baseline not found at {baseline_path}")
        return
    if not os.path.exists(snn_ckpt_path):
        print(f"ERROR: SNN checkpoint not found at {snn_ckpt_path}")
        return
    
    ber_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    # First: evaluate baseline (no SNN)
    print("\n  Evaluating baseline (no SNN)...")
    from ultralytics import YOLO
    baseline_model = YOLO(baseline_path)
    baseline_results = baseline_model.val(
        data='DOTAv1.yaml', imgsz=640, batch=16, device=0,
        verbose=False, plots=False,
    )
    baseline_map50 = float(baseline_results.results_dict.get('metrics/mAP50(B)', 0))
    baseline_map50_95 = float(baseline_results.results_dict.get('metrics/mAP50-95(B)', 0))
    print(f"  Baseline: mAP@50={baseline_map50:.4f}, mAP@50-95={baseline_map50_95:.4f}")
    
    # Then: evaluate SNN at each BER
    all_results = {
        'baseline': {
            'mAP50': baseline_map50,
            'mAP50_95': baseline_map50_95,
        },
        'snn_results': {}
    }
    
    for ber in ber_values:
        print(f"\n  Evaluating SNN at BER={ber:.2f}...")
        try:
            result = evaluate_mAP_with_snn(
                baseline_path, snn_ckpt_path,
                ber=ber, c_bot=36, T=4
            )
            all_results['snn_results'][str(ber)] = result
            print(f"    mAP@50={result['mAP50']:.4f}, "
                  f"mAP@50-95={result['mAP50_95']:.4f}, "
                  f"FR={result['avg_firing_rate']:.3f}")
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results['snn_results'][str(ber)] = {'error': str(e)}
    
    # Summary table
    print("\n" + "=" * 70)
    print("  SUMMARY: mAP@50 vs BER")
    print("=" * 70)
    print(f"  {'BER':>6s}  {'mAP@50':>8s}  {'mAP@50-95':>10s}  {'Δ mAP@50':>10s}")
    print(f"  {'---':>6s}  {'---':>8s}  {'---':>10s}  {'---':>10s}")
    print(f"  {'none':>6s}  {baseline_map50:8.4f}  {baseline_map50_95:10.4f}  {'(baseline)':>10s}")
    
    for ber in ber_values:
        key = str(ber)
        if key in all_results['snn_results'] and 'mAP50' in all_results['snn_results'][key]:
            r = all_results['snn_results'][key]
            delta = r['mAP50'] - baseline_map50
            print(f"  {ber:6.2f}  {r['mAP50']:8.4f}  {r['mAP50_95']:10.4f}  {delta:+10.4f}")
    
    os.makedirs('eval', exist_ok=True)
    with open('eval/dota_detection_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to eval/dota_detection_results.json")


if __name__ == '__main__':
    main()
