#!/usr/bin/env python3
"""Empirical energy measurement: count actual MACs vs SynOps.

Measures the ACTUAL firing rate across the full test set and computes
empirical SynOps, rather than relying solely on the Horowitz model estimate.

Also counts exact MACs for the equivalent CNN encoder for comparison.

Output: eval/energy_empirical.json
Usage:  python eval/measure_energy.py
"""

import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from train_aid_v2 import ResNet50Front, ResNet50Back
from run_final_pipeline import AIDDataset5050, RESISC45Dataset, SpikeAdaptSC_v5c_NA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_STEPS = 8

# Horowitz model (45nm CMOS)
MAC_PJ = 4.6   # pJ per multiply-accumulate
SYNOP_PJ = 0.9  # pJ per synaptic operation (accumulate only)


def count_conv_macs(conv, H_in, W_in):
    """Count MACs for a Conv2d layer (analytical)."""
    C_in = conv.in_channels
    C_out = conv.out_channels
    K = conv.kernel_size[0] * conv.kernel_size[1]
    H_out = (H_in + 2 * conv.padding[0] - conv.kernel_size[0]) // conv.stride[0] + 1
    W_out = (W_in + 2 * conv.padding[1] - conv.kernel_size[1]) // conv.stride[1] + 1
    # One MAC per input channel per kernel element per output position
    macs = C_in * K * C_out * H_out * W_out
    return macs, H_out, W_out


def measure_snn_energy(model, front, test_loader, dataset_name):
    """Measure empirical firing rates and compute SynOps."""
    model.eval(); front.eval()
    
    # Architecture: 
    #   conv1: 1024→256, 3×3, on 14×14 grid
    #   conv2: 256→36, 3×3, on 14×14 grid
    H, W = 14, 14
    
    # Count analytical MACs for CNN equivalent (no sparsity)
    conv1_macs, _, _ = count_conv_macs(model.encoder.conv1, H, W)
    conv2_macs, _, _ = count_conv_macs(model.encoder.conv2, H, W)
    total_cnn_macs_per_step = conv1_macs + conv2_macs
    total_cnn_macs = total_cnn_macs_per_step * T_STEPS
    
    # Now measure empirical firing rates
    layer1_spikes = []
    layer2_spikes = []
    n_images = 0
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            feat = front(imgs)
            B = imgs.size(0)
            
            m1, m2 = None, None
            batch_s1, batch_s2 = [], []
            for t in range(T_STEPS):
                s1, s2, m1, m2 = model.encoder(feat, m1, m2, t=t)
                batch_s1.append(s1.cpu())
                batch_s2.append(s2.cpu())
            
            # Firing rates per layer
            s1_stack = torch.stack(batch_s1)  # T×B×C1×H×W
            s2_stack = torch.stack(batch_s2)  # T×B×C2×H×W
            
            layer1_spikes.append(s1_stack.mean().item())
            layer2_spikes.append(s2_stack.mean().item())
            n_images += B
    
    fr1 = np.mean(layer1_spikes)
    fr2 = np.mean(layer2_spikes)
    fr_overall = (fr1 + fr2) / 2  # Approximate
    
    # SynOps: each spike triggers C_out accumulations per kernel element
    # For conv1 (1024→256, 3×3): SynOps = FR1 * C_in * K * C_out * H * W * T
    # But SynOps are INPUT-sparse: only non-zero inputs contribute
    # SynOps_conv1 = fr_input_to_conv1 * MACs_conv1
    # Input to conv1 is the backbone feature (non-binary, always active) 
    # BUT in SNN the input IS the spike (binary), so:
    # - conv1 input = original feature F (not sparse, always fires)
    # - conv2 input = s1 (sparse, FR = fr1)
    
    # For a fair SNN SynOps count:
    # conv1: input is the continuous backbone feature → treated as MAC (not sparse)
    #   BUT in SNN operation, conv1 processes F once and feeds to LIF membrane
    #   The spike output s1 has fr1. The key energy metric is:
    #   conv1 is a standard MAC (input is continuous), not a SynOp
    # conv2: input IS the spike s1 (binary, sparse with rate fr1)
    #   SynOps_conv2 = fr1 * MACs_conv2_analytical
    
    # Actually the standard SNN SynOps calculation:
    # For each conv layer, SynOps = firing_rate_of_input × analytical_MACs
    # conv1 input = continuous feature F → FR = 1.0 (non-spike)
    # conv2 input = spike s1 → FR = fr1
    
    # Total SynOps = conv1_macs * 1.0 * T + conv2_macs * fr1 * T
    # Energy = SynOps_conv1_as_mac * MAC_PJ + SynOps_conv2 * SYNOP_PJ
    
    # Most conservative (and standard in SNN lit):
    # All ops are SynOps, scaled by input firing rate
    # conv1: input FR = 1.0 (backbone features are always active)  
    # conv2: input FR = fr1
    synops_conv1 = conv1_macs * 1.0 * T_STEPS  # conv1 input is continuous (FR=1)
    synops_conv2 = conv2_macs * fr1 * T_STEPS   # conv2 input is sparse spikes
    total_synops = synops_conv1 + synops_conv2
    
    # Energy comparison
    cnn_energy = total_cnn_macs * MAC_PJ
    snn_energy = synops_conv1 * SYNOP_PJ + synops_conv2 * SYNOP_PJ  
    # Note: conv1 should arguably use MAC_PJ since input isn't binary
    # Conservative version: conv1 at MAC_PJ, conv2 at SYNOP_PJ
    snn_energy_conservative = synops_conv1 * MAC_PJ + synops_conv2 * SYNOP_PJ
    
    # Standard SNN lit version: all at SYNOP_PJ (assumes neuromorphic hardware)
    snn_energy_optimistic = total_synops * SYNOP_PJ
    
    # With masking at ρ=0.75: conv2 only processes masked blocks
    # But actually masking happens AFTER encoding, so encoder cost is the same
    # The saving is in transmission, not computation
    
    results = {
        'dataset': dataset_name,
        'n_images': n_images,
        'firing_rates': {
            'layer1_mean': round(fr1, 4),
            'layer2_mean': round(fr2, 4),
            'overall_mean': round(fr_overall, 4),
            'layer1_std': round(float(np.std(layer1_spikes)), 4),
            'layer2_std': round(float(np.std(layer2_spikes)), 4),
        },
        'operation_counts': {
            'conv1_macs_per_step': conv1_macs,
            'conv2_macs_per_step': conv2_macs,
            'total_cnn_macs': total_cnn_macs,
            'synops_conv1': int(synops_conv1),
            'synops_conv2': int(synops_conv2),
            'total_synops': int(total_synops),
        },
        'energy_pj': {
            'cnn_encoder': round(cnn_energy, 1),
            'snn_conservative': round(snn_energy_conservative, 1),
            'snn_optimistic': round(snn_energy_optimistic, 1),
        },
        'energy_ratio': {
            'conservative': round(cnn_energy / snn_energy_conservative, 1),
            'optimistic': round(cnn_energy / snn_energy_optimistic, 1),
        },
    }
    
    print(f"\n  {dataset_name.upper()} Empirical Energy Analysis:")
    print(f"    Images evaluated: {n_images}")
    print(f"    Layer 1 FR: {fr1:.4f} ± {np.std(layer1_spikes):.4f}")
    print(f"    Layer 2 FR: {fr2:.4f} ± {np.std(layer2_spikes):.4f}")
    print(f"    CNN encoder MACs: {total_cnn_macs:,} ({total_cnn_macs/1e6:.1f}M)")
    print(f"    SNN SynOps (total): {int(total_synops):,} ({total_synops/1e6:.1f}M)")
    print(f"      conv1 (continuous input): {int(synops_conv1):,}")
    print(f"      conv2 (sparse input, FR={fr1:.4f}): {int(synops_conv2):,}")
    print(f"    CNN energy: {cnn_energy/1e6:.2f} µJ")
    print(f"    SNN energy (conservative): {snn_energy_conservative/1e6:.2f} µJ → {cnn_energy/snn_energy_conservative:.1f}× savings")
    print(f"    SNN energy (optimistic): {snn_energy_optimistic/1e6:.2f} µJ → {cnn_energy/snn_energy_optimistic:.1f}× savings")
    
    return results


def measure_latency(model, front, back, test_loader, n_warmup=10, n_measure=50):
    """Measure actual GPU inference latency."""
    model.eval(); front.eval(); back.eval()
    
    # Warmup
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(test_loader):
            if i >= n_warmup: break
            imgs = imgs.to(device)
            feat = front(imgs)
            Fp, _ = model(feat, noise_param=0.0)
            _ = back(Fp)
    
    torch.cuda.synchronize()
    
    # Measure
    times_front, times_snn, times_back = [], [], []
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(test_loader):
            if i >= n_measure: break
            imgs = imgs.to(device)
            
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            feat = front(imgs)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            Fp, _ = model(feat, noise_param=0.0)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            _ = back(Fp)
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            
            B = imgs.size(0)
            times_front.append((t1-t0)/B * 1000)  # ms per image
            times_snn.append((t2-t1)/B * 1000)
            times_back.append((t3-t2)/B * 1000)
    
    return {
        'backbone_ms': round(np.mean(times_front), 3),
        'snn_encoder_decoder_ms': round(np.mean(times_snn), 3),
        'classifier_ms': round(np.mean(times_back), 3),
        'total_ms': round(np.mean(times_front) + np.mean(times_snn) + np.mean(times_back), 3),
    }


def main():
    torch.manual_seed(42); np.random.seed(42)
    print(f"Device: {device}")
    
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])
    
    all_results = {}
    
    for ds_name, n_classes, ds_cls_args in [
        ('aid', 30, dict(root='./data', transform=None, split='test', seed=42)),
        ('resisc45', 45, dict(root='./data', transform=None, split='test', train_ratio=0.20, seed=42)),
    ]:
        print(f"\n{'='*60}")
        print(f"  {ds_name.upper()}")
        print(f"{'='*60}")
        
        if ds_name == 'aid':
            test_ds = AIDDataset5050('./data', tf_test, 'test', seed=42)
        else:
            test_ds = RESISC45Dataset('./data', tf_test, 'test', train_ratio=0.20, seed=42)
        test_loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)
        
        front = ResNet50Front(grid_size=14).to(device)
        bb = torch.load(f"./snapshots_{ds_name}_5050_seed42/backbone_best.pth", map_location=device)
        front.load_state_dict({k: v for k, v in bb.items()
                               if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))}, strict=False)
        front.eval()
        for p in front.parameters(): p.requires_grad = False
        
        back = ResNet50Back(n_classes).to(device)
        model = SpikeAdaptSC_v5c_NA(C_in=1024, C1=256, C2=36, T=T_STEPS,
                                     target_rate=0.75, grid_size=14).to(device)
        
        ck_name = "v5cna_best_95.42.pth" if ds_name == 'aid' else "v5cna_best_92.01.pth"
        ck = torch.load(f"./snapshots_{ds_name}_v5cna_seed42/{ck_name}", map_location=device)
        model.load_state_dict(ck['model'])
        back.load_state_dict(ck['back'])
        model.eval()
        
        energy = measure_snn_energy(model, front, test_loader, ds_name)
        latency = measure_latency(model, front, back, test_loader)
        
        print(f"\n  GPU Latency (per image):")
        print(f"    Backbone: {latency['backbone_ms']:.3f} ms")
        print(f"    SNN enc+dec: {latency['snn_encoder_decoder_ms']:.3f} ms")
        print(f"    Classifier: {latency['classifier_ms']:.3f} ms")
        print(f"    Total: {latency['total_ms']:.3f} ms")
        
        all_results[ds_name] = {'energy': energy, 'latency': latency}
        
        del model, back, front
        torch.cuda.empty_cache()
    
    with open("eval/energy_empirical.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to eval/energy_empirical.json")


if __name__ == "__main__":
    main()
