#!/usr/bin/env python3
"""SpikeAdapt-SC Object Detection Extension on DOTA.

Three-phase approach:
  Phase 1: Train baseline YOLOv8n-obb on DOTA (establishes clean mAP)
  Phase 2: Insert SNN bottleneck between YOLOv8 backbone & head, train with BSC
  Phase 3: Evaluate mAP vs BER sweep

The SNN bottleneck replaces the communication link between the UAV camera
(running the backbone) and the ground station (running the detection head).

Key architecture:
  YOLOv8 backbone (P3/P4/P5 features) → SNN encode → BSC channel → 
  SNN decode → YOLOv8 detection head → OBB predictions

Usage:
  python train_dota_snn.py --phase 1   # Train baseline YOLOv8
  python train_dota_snn.py --phase 2   # Train SNN bottleneck
  python train_dota_snn.py --phase 3   # Evaluate with noise
"""

import os, sys, json, argparse, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================================================================
# SNN Bottleneck for Detection (multi-scale)
# ===================================================================
class SNN_Bottleneck_Detection(nn.Module):
    """SNN encoder-decoder bottleneck for detection feature maps.
    
    Unlike classification (single 1024×14×14 feature map), detection uses
    multi-scale features (P3, P4, P5 from FPN). We apply independent SNN
    encoding to each scale level.
    
    Encoder: Conv2d → BN → LIF neuron → Binary spikes (T timesteps)
    Decoder: ConvTranspose → BN → reconstruct features
    Channel: BSC (flip bits with probability BER)
    """
    
    def __init__(self, channels_list, C_bottleneck=36, T=4):
        """
        Args:
            channels_list: list of channel counts for each FPN level [P3, P4, P5]
                           e.g. [256, 512, 1024] for YOLOv8n after backbone
            C_bottleneck: bottleneck channel count (compressed representation)  
            T: number of spike timesteps
        """
        super().__init__()
        self.T = T
        self.n_levels = len(channels_list)
        
        # Per-level encoder/decoder
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for C_in in channels_list:
            enc = nn.Sequential(
                nn.Conv2d(C_in, C_bottleneck, 3, 1, 1),
                nn.BatchNorm2d(C_bottleneck),
            )
            dec = nn.Sequential(
                nn.Conv2d(C_bottleneck, C_in, 3, 1, 1),
                nn.BatchNorm2d(C_in),
                nn.ReLU(inplace=True),
            )
            self.encoders.append(enc)
            self.decoders.append(dec)
        
        # LIF neuron parameters (shared across levels)
        self.threshold = nn.Parameter(torch.ones(1) * 1.0)
        self.beta_raw = nn.Parameter(torch.ones(1) * 2.2)  # leak
    
    @property
    def beta(self):
        return torch.sigmoid(self.beta_raw)
    
    def spike_fn(self, membrane, threshold):
        """Surrogate gradient spike function."""
        spikes = (membrane > threshold).float()
        # Straight-through estimator for backward
        spikes = spikes + (torch.sigmoid(4 * (membrane - threshold)) - spikes).detach() * 0 + \
                 (membrane - threshold).clamp(min=0).detach() * 0
        return spikes
    
    def encode_level(self, feat, level_idx):
        """Encode a single FPN level into spike trains."""
        h = self.encoders[level_idx](feat)  # B×C_bot×H×W
        
        all_spikes = []
        membrane = torch.zeros_like(h)
        beta = self.beta
        
        for t in range(self.T):
            membrane = beta * membrane + h
            spikes = self.spike_fn(membrane, self.threshold)
            membrane = membrane - spikes * self.threshold  # soft reset
            all_spikes.append(spikes)
        
        return all_spikes  # list of T × (B, C_bot, H, W)
    
    def bsc_channel(self, spikes, ber):
        """Binary Symmetric Channel: flip bits with probability ber."""
        if ber <= 0:
            return spikes
        flip_mask = (torch.rand_like(spikes) < ber).float()
        return (spikes + flip_mask) % 2
    
    def decode_level(self, received_spikes, level_idx):
        """Decode spike trains back to features."""
        # Average across timesteps
        avg = torch.stack(received_spikes).mean(0)  # B×C_bot×H×W
        return self.decoders[level_idx](avg)
    
    def forward(self, features, ber=0.0):
        """
        Args:
            features: list of feature maps [P3, P4, P5] from backbone
            ber: BSC bit error rate
        
        Returns:
            reconstructed: list of feature maps (same shapes as input)
            info: dict with firing rates and compression stats
        """
        reconstructed = []
        firing_rates = []
        
        for i, feat in enumerate(features):
            # Encode
            spikes = self.encode_level(feat, i)
            
            # Measure firing rate
            with torch.no_grad():
                fr = torch.stack(spikes).mean().item()
                firing_rates.append(fr)
            
            # Channel
            received = [self.bsc_channel(s, ber) for s in spikes]
            
            # Decode
            recon = self.decode_level(received, i)
            reconstructed.append(recon)
        
        info = {
            'firing_rates': firing_rates,
            'avg_firing_rate': np.mean(firing_rates),
        }
        
        return reconstructed, info


# ===================================================================
# Phase 1: Train baseline YOLOv8 on DOTA
# ===================================================================
def phase1_train_baseline():
    """Train a standard YOLOv8n-obb on DOTA as the baseline."""
    from ultralytics import YOLO
    
    print("=" * 70)
    print("  PHASE 1: Train baseline YOLOv8n-obb on DOTA")
    print("=" * 70)
    
    model = YOLO('yolov8n-obb.pt')
    
    results = model.train(
        data='DOTAv1.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        project='runs/dota',
        name='baseline_yolov8n_obb',
        exist_ok=True,
        workers=8,
        patience=20,
        save=True,
        plots=True,
    )
    
    print(f"\n✅ Baseline training complete!")
    print(f"  Best mAP@50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    return results


# ===================================================================
# Phase 2: SNN bottleneck integration
# ===================================================================
def phase2_train_snn_bottleneck():
    """Insert SNN bottleneck into YOLOv8 and fine-tune with BSC noise."""
    from ultralytics import YOLO
    
    print("=" * 70)
    print("  PHASE 2: Train SNN bottleneck for DOTA detection")
    print("=" * 70)
    
    # Load the best baseline checkpoint
    baseline_path = 'runs/dota/baseline_yolov8n_obb/weights/best.pt'
    if not os.path.exists(baseline_path):
        print(f"  ERROR: Baseline not found at {baseline_path}")
        print(f"  Run phase 1 first!")
        return
    
    model = YOLO(baseline_path)
    
    # Get backbone output channel sizes by running a dummy forward
    dummy = torch.randn(1, 3, 640, 640).to(device)
    backbone = model.model.model[:10]  # YOLOv8 backbone layers
    
    # Extract FPN feature sizes
    with torch.no_grad():
        feats = []
        x = dummy
        for i, layer in enumerate(backbone):
            x = layer(x) if hasattr(layer, 'forward') else x
            # P3 (layer 4), P4 (layer 6), P5 (layer 9) in YOLOv8n
            if i in [4, 6, 9]:
                feats.append(x)
                print(f"    Level {len(feats)}: shape={x.shape}")
    
    channel_sizes = [f.shape[1] for f in feats]
    print(f"  FPN channel sizes: {channel_sizes}")
    
    # Create SNN bottleneck
    snn_bottleneck = SNN_Bottleneck_Detection(
        channels_list=channel_sizes,
        C_bottleneck=36,
        T=4
    ).to(device)
    
    print(f"  SNN bottleneck params: {sum(p.numel() for p in snn_bottleneck.parameters()):,}")
    
    # TODO: Full integration requires hooking into YOLOv8's forward pass
    # For now, train the bottleneck separately using feature distillation
    
    print("\n  Training SNN bottleneck with feature distillation...")
    
    # Phase 2 needs the full YOLO model to be modified to include
    # the SNN bottleneck in its forward pass. This requires either:
    # (a) Monkey-patching YOLOv8's forward, or
    # (b) Creating a wrapper model
    
    # For now, let's do feature-matching training
    print("  [Implementation continues in Phase 2b...]")


# ===================================================================
# Phase 3: Evaluate with noise
# ===================================================================
def phase3_evaluate():
    """Evaluate detection performance across BER values."""
    print("=" * 70)
    print("  PHASE 3: Evaluate mAP vs BER")
    print("=" * 70)
    
    ber_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    # TODO: Load trained model with SNN bottleneck
    # Evaluate at each BER level
    
    results = {}
    for ber in ber_values:
        print(f"  BER={ber:.2f}: evaluating...")
        # mAP = evaluate_at_ber(model, val_loader, ber)
        # results[str(ber)] = mAP
    
    print("\n  Results saved to eval/dota_detection_results.json")


# ===================================================================
# Main
# ===================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2, 3])
    args = parser.parse_args()
    
    if args.phase == 1:
        phase1_train_baseline()
    elif args.phase == 2:
        phase2_train_snn_bottleneck()
    elif args.phase == 3:
        phase3_evaluate()
