#!/usr/bin/env python3
"""BEC Cross-Baseline Evaluation: compare BEC immunity across all methods.

Evaluates Binary Erasure Channel (erasing bits to 0 with probability p) for:
  - SpikeAdapt-SC (SNN, firing rate ~0.167 → ~83% zeros → BEC-immune)
  - CNN-1bit (STE binarization, ~50% ones → moderate BEC vulnerability)
  - JSCC (continuous-valued → BEC erasure is severe)
  - CNN-Uni (8-bit, quantized → BEC erasure is severe)

Key hypothesis: BEC immunity scales with the fraction of zeros in the
transmitted representation. SNN has the most zeros (~83%) → most immune.

Output: eval/bec_comparison_results.json
"""

import os, sys, json, math
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from train_aid_v2 import ResNet50Front, ResNet50Back
from run_final_pipeline import AIDDataset5050, RESISC45Dataset, SpikeAdaptSC_v5c_NA
from train_1bit_baseline import BinaryCNN_SC
from train_jscc_baseline import JSCC_Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_STEPS = 8


# ===================================================================
# BEC Channel for different encodings
# ===================================================================
class BEC_Binary(nn.Module):
    """BEC for binary {0,1} signals: erase to 0 with probability p."""
    def forward(self, x, erasure_prob):
        if erasure_prob <= 0: return x
        mask = (torch.rand_like(x.float()) < erasure_prob).float()
        return x * (1.0 - mask)

class BEC_Continuous(nn.Module):
    """BEC for continuous signals: erase to 0 with probability p."""
    def forward(self, x, erasure_prob):
        if erasure_prob <= 0: return x
        mask = (torch.rand_like(x.float()) < erasure_prob).float()
        return x * (1.0 - mask)


# ===================================================================
# Helper: Load models and run BEC sweep
# ===================================================================
def load_snn_model(dataset_name, seed=42):
    """Load SpikeAdapt-SC using same pattern as multichannel_eval.py."""
    n_classes = 30 if dataset_name == 'aid' else 45
    bb_dir = f'snapshots_{dataset_name}_5050_seed{seed}'
    v5c_dir = f'snapshots_{dataset_name}_v5cna_seed{seed}'
    
    front = ResNet50Front(grid_size=14).to(device)
    bb = torch.load(f'./{bb_dir}/backbone_best.pth', map_location=device, weights_only=False)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))}, strict=False)
    front.eval()
    
    back = ResNet50Back(n_classes).to(device)
    model = SpikeAdaptSC_v5c_NA(C_in=1024, C1=256, C2=36, T=T_STEPS,
                                 target_rate=0.75, grid_size=14).to(device)
    
    ck_files = sorted([f for f in os.listdir(f'./{v5c_dir}') if f.startswith('v5cna_best')])
    ck = torch.load(f'./{v5c_dir}/{ck_files[-1]}', map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])
    back.load_state_dict(ck['back'])
    
    model.eval(); back.eval()
    return front, back, model


def load_cnn1bit_model(dataset_name, seed=42):
    """Load CNN-1bit (BinaryCNN_SC)."""
    n_classes = 30 if dataset_name == 'aid' else 45
    
    bb_path = f'./snapshots_{dataset_name}_5050_seed{seed}/backbone_best.pth'
    front = ResNet50Front(grid_size=14).to(device)
    bb = torch.load(bb_path, map_location=device, weights_only=False)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))}, strict=False)
    front.eval()
    
    back = ResNet50Back(n_classes).to(device)
    back_state = {k: v for k, v in bb.items()
                  if k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))}
    back.load_state_dict(back_state, strict=False)
    
    model = BinaryCNN_SC(C_in=1024, C1=256, C2=36).to(device)
    
    snap_dir = f'./snapshots_{dataset_name}_1bit_seed{seed}'
    files = sorted([f for f in os.listdir(snap_dir) if f.startswith('1bit_best_')],
                   key=lambda x: float(x.split('_')[-1].replace('.pth', '')))
    ck = torch.load(os.path.join(snap_dir, files[-1]), map_location=device, weights_only=False)
    model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
    
    front.eval(); back.eval(); model.eval()
    return front, back, model


def load_jscc_model(dataset_name, seed=42):
    """Load JSCC baseline."""
    n_classes = 30 if dataset_name == 'aid' else 45
    
    bb_path = f'./snapshots_{dataset_name}_5050_seed{seed}/backbone_best.pth'
    front = ResNet50Front(grid_size=14).to(device)
    bb = torch.load(bb_path, map_location=device, weights_only=False)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))}, strict=False)
    front.eval()
    
    back = ResNet50Back(n_classes).to(device)
    back_state = {k: v for k, v in bb.items()
                  if k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))}
    back.load_state_dict(back_state, strict=False)
    
    model = JSCC_Model(C_in=1024, C_mid=256, C_tx=36).to(device)
    
    snap_dir = f'./snapshots_{dataset_name}_jscc_seed{seed}'
    files = sorted([f for f in os.listdir(snap_dir) if f.startswith('jscc_best_')],
                   key=lambda x: float(x.split('_')[-1].replace('.pth', '')))
    ck = torch.load(os.path.join(snap_dir, files[-1]), map_location=device, weights_only=False)
    model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
    
    front.eval(); back.eval(); model.eval()
    return front, back, model


def eval_snn_bec(front, back, model, loader, erasure_rate):
    """Eval SNN under BEC using manual forward pass (matching multichannel_eval)."""
    bec = BEC_Binary().to(device)
    correct, total = 0, 0
    firing_ones, firing_total = 0, 0
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            # Manual forward pass (same as multichannel_eval.py)
            all_S2, m1, m2 = [], None, None
            for t in range(model.T):
                _, s2, m1, m2 = model.encoder(feat, m1, m2, t=t)
                all_S2.append(s2)
            importance = model.scorer(all_S2, 0.0).squeeze(1)
            mask, tx = model.block_mask(importance, training=False)
            # Apply BEC instead of BSC
            recv = [bec(all_S2[t] * mask, erasure_rate) for t in range(model.T)]
            Fp = model.decoder(recv, mask)
            
            # Count firing rate
            for s in all_S2:
                firing_ones += s.sum().item()
                firing_total += s.numel()
            
            correct += back(Fp).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    
    acc = round(100. * correct / total, 2)
    firing_rate = firing_ones / max(firing_total, 1)
    return acc, firing_rate


def eval_cnn1bit_bec(front, back, model, loader, erasure_rate):
    """Eval CNN-1bit under BEC: erase binary channels to 0."""
    bec = BEC_Binary().to(device)
    correct, total = 0, 0
    ones_frac_total, n_batches = 0, 0
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            
            # Get binary representation via encoder
            binary = model.encoder(feat)  # {0, 1}
            
            # Measure ones fraction
            ones_frac_total += binary.mean().item()
            n_batches += 1
            
            # Apply BEC (erase to 0)
            binary_bec = bec(binary, erasure_rate)
            
            # Decode
            Fp = model.decoder(binary_bec)
            
            correct += back(Fp).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    
    acc = round(100. * correct / total, 2)
    ones_frac = ones_frac_total / max(n_batches, 1)
    return acc, ones_frac


def eval_jscc_bec(front, back, model, loader, erasure_rate):
    """Eval JSCC under BEC: erase continuous symbols to 0."""
    bec = BEC_Continuous().to(device)
    correct, total = 0, 0
    zero_frac_total, n_batches = 0, 0
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            
            z_norm, power = model.encoder(feat)
            
            # Measure near-zero fraction
            zero_frac_total += (z_norm.abs() < 0.1).float().mean().item()
            n_batches += 1
            
            # Apply BEC
            z_bec = bec(z_norm, erasure_rate)
            
            Fp = model.decoder(z_bec, power)
            correct += back(Fp).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    
    acc = round(100. * correct / total, 2)
    zero_frac = zero_frac_total / max(n_batches, 1)
    return acc, zero_frac


# ===================================================================
# Main
# ===================================================================
def main():
    erasure_rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])
    
    results = {}
    
    for dataset_name in ['aid', 'resisc45']:
        n_classes = 30 if dataset_name == 'aid' else 45
        print(f"\n{'='*70}")
        print(f"  BEC CROSS-BASELINE EVALUATION — {dataset_name.upper()}")
        print(f"{'='*70}")
        
        if dataset_name == 'aid':
            test_ds = AIDDataset5050('./data', tf_test, 'test', seed=42)
        else:
            test_ds = RESISC45Dataset('./data', tf_test, 'test', train_ratio=0.20, seed=42)
        test_loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)
        print(f"  Test: {len(test_ds)} images, {n_classes} classes")
        
        results[dataset_name] = {}
        
        # --- SpikeAdapt-SC ---
        print(f"\n  Loading SpikeAdapt-SC...")
        try:
            front_s, back_s, model_s = load_snn_model(dataset_name)
            results[dataset_name]['snn'] = {'bec': {}, 'type': 'binary_sparse'}
            for er in erasure_rates:
                acc, fr = eval_snn_bec(front_s, back_s, model_s, test_loader, er)
                results[dataset_name]['snn']['bec'][str(er)] = acc
                if er == 0.0:
                    results[dataset_name]['snn']['firing_rate'] = round(fr, 4)
                    print(f"    SNN firing rate: {fr:.4f} ({(1-fr)*100:.1f}% zeros)")
                print(f"    BEC ER={er:.2f}: {acc}%")
            del front_s, back_s, model_s; torch.cuda.empty_cache()
        except Exception as e:
            print(f"    ERROR: {e}")
        
        # --- CNN-1bit ---
        print(f"\n  Loading CNN-1bit...")
        try:
            front_c, back_c, model_c = load_cnn1bit_model(dataset_name)
            results[dataset_name]['cnn1bit'] = {'bec': {}, 'type': 'binary_dense'}
            for er in erasure_rates:
                acc, ones_frac = eval_cnn1bit_bec(front_c, back_c, model_c, test_loader, er)
                results[dataset_name]['cnn1bit']['bec'][str(er)] = acc
                if er == 0.0:
                    results[dataset_name]['cnn1bit']['ones_fraction'] = round(ones_frac, 4)
                    print(f"    CNN-1bit ones fraction: {ones_frac:.4f} ({(1-ones_frac)*100:.1f}% zeros)")
                print(f"    BEC ER={er:.2f}: {acc}%")
            del front_c, back_c, model_c; torch.cuda.empty_cache()
        except Exception as e:
            print(f"    ERROR: {e}")
        
        # --- JSCC ---
        print(f"\n  Loading JSCC...")
        try:
            front_j, back_j, model_j = load_jscc_model(dataset_name)
            results[dataset_name]['jscc'] = {'bec': {}, 'type': 'continuous'}
            for er in erasure_rates:
                acc, zf = eval_jscc_bec(front_j, back_j, model_j, test_loader, er)
                results[dataset_name]['jscc']['bec'][str(er)] = acc
                if er == 0.0:
                    results[dataset_name]['jscc']['near_zero_fraction'] = round(zf, 4)
                    print(f"    JSCC near-zero fraction: {zf:.4f}")
                print(f"    BEC ER={er:.2f}: {acc}%")
            del front_j, back_j, model_j; torch.cuda.empty_cache()
        except Exception as e:
            print(f"    ERROR: {e}")
        
        print()
    
    # === Summary Table ===
    print(f"\n{'='*70}")
    print(f"  BEC COMPARISON SUMMARY (ER=0.30)")
    print(f"{'='*70}")
    print(f"  {'Method':<20} {'AID':>10} {'R45':>10}  Zero%")
    print(f"  {'-'*55}")
    
    for method in ['snn', 'cnn1bit', 'jscc']:
        aid_acc = results.get('aid', {}).get(method, {}).get('bec', {}).get('0.3', 'N/A')
        r45_acc = results.get('resisc45', {}).get(method, {}).get('bec', {}).get('0.3', 'N/A')
        mtype = results.get('aid', {}).get(method, {}).get('type', '?')
        
        if method == 'snn':
            zero_pct = f"{(1 - results.get('aid', {}).get(method, {}).get('firing_rate', 0))*100:.0f}%"
        elif method == 'cnn1bit':
            zero_pct = f"{(1 - results.get('aid', {}).get(method, {}).get('ones_fraction', 0.5))*100:.0f}%"
        else:
            zero_pct = "N/A"
        
        name = {'snn': 'SpikeAdapt-SC', 'cnn1bit': 'CNN-1bit', 'jscc': 'JSCC'}[method]
        print(f"  {name:<20} {aid_acc:>10} {r45_acc:>10}  {zero_pct}")
    
    # Save
    with open('eval/bec_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ BEC comparison saved to eval/bec_comparison_results.json")


if __name__ == '__main__':
    main()
