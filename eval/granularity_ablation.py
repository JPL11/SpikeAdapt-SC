#!/usr/bin/env python3
"""Granularity ablation: 14×14 vs 7×7 vs 4×4 masking on AID and RESISC45.

Key question: Does finer masking granularity improve accuracy under noise?

Approach (eval-only, no retraining):
  - Use the same trained model (seed-42 checkpoint)
  - At eval time, pool the 14×14 importance scores to coarser grids
  - Apply top-k masking at the coarser grid, then upsample mask back to 14×14
  - Compare: 14×14 (native) vs 7×7 (2×2 blocks) vs 4×4 (3-4 pixel blocks)
  
  Also compare learned-vs-random gap at each granularity to show that
  fine-grained scoring is what makes learned masking worthwhile.

Output: eval/granularity_ablation_results.json

Usage:
  python eval/granularity_ablation.py
"""

import os, sys, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from train_aid_v2 import ResNet50Front, ResNet50Back, BSC_Channel, LearnedBlockMask
from train_aid_v5 import EncoderV5, DecoderV5
from noise_aware_scorer import NoiseAwareScorer
from run_final_pipeline import AIDDataset5050, RESISC45Dataset, SpikeAdaptSC_v5c_NA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_STEPS = 8


def evaluate_at_granularity(model, back, front, test_loader, ber, rho, 
                             grid_size, mask_type='learned'):
    """Evaluate with masking at a specific granularity.
    
    Args:
        grid_size: Masking granularity (14, 7, or 4)
        mask_type: 'learned' (pool scores → coarse top-k → upsample),
                   'random' (random mask at coarse grid → upsample),
                   'full' (ρ=1.0 baseline, no masking)
    """
    model.eval(); back.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            B = imgs.size(0)
            
            # Forward through encoder
            all_S2, m1, m2 = [], None, None
            for t in range(T_STEPS):
                _, s2, m1, m2 = model.encoder(feat, m1, m2, t=t)
                all_S2.append(s2)
            
            H_native, W_native = all_S2[0].shape[2], all_S2[0].shape[3]  # 14×14
            
            if mask_type == 'full':
                # No masking — send everything
                mask = torch.ones(B, 1, H_native, W_native, device=device)
            elif mask_type == 'learned':
                # Get native 14×14 importance scores
                importance = model.scorer(all_S2, ber)  # B×1×14×14
                
                if grid_size == H_native:
                    # Native 14×14: standard top-k
                    n_keep = max(1, int(rho * H_native * W_native))
                    flat = importance.view(B, -1)
                    _, idx = flat.topk(n_keep, dim=1)
                    mask = torch.zeros_like(flat)
                    mask.scatter_(1, idx, 1.0)
                    mask = mask.view(B, 1, H_native, W_native)
                else:
                    # Pool to coarse grid → top-k → upsample to 14×14
                    coarse_imp = F.adaptive_avg_pool2d(importance, grid_size)  # B×1×G×G
                    n_keep_coarse = max(1, int(rho * grid_size * grid_size))
                    flat_c = coarse_imp.view(B, -1)
                    _, idx = flat_c.topk(n_keep_coarse, dim=1)
                    coarse_mask = torch.zeros_like(flat_c)
                    coarse_mask.scatter_(1, idx, 1.0)
                    coarse_mask = coarse_mask.view(B, 1, grid_size, grid_size)
                    # Upsample back to 14×14 using nearest neighbor
                    mask = F.interpolate(coarse_mask, size=(H_native, W_native), 
                                        mode='nearest')
            elif mask_type == 'random':
                if grid_size == H_native:
                    n_keep = max(1, int(rho * H_native * W_native))
                    mask = torch.zeros(B, 1, H_native, W_native, device=device)
                    for b in range(B):
                        indices = torch.randperm(H_native * W_native, device=device)[:n_keep]
                        mask[b, 0].view(-1)[indices] = 1.0
                else:
                    n_keep_coarse = max(1, int(rho * grid_size * grid_size))
                    coarse_mask = torch.zeros(B, 1, grid_size, grid_size, device=device)
                    for b in range(B):
                        indices = torch.randperm(grid_size * grid_size, device=device)[:n_keep_coarse]
                        coarse_mask[b, 0].view(-1)[indices] = 1.0
                    mask = F.interpolate(coarse_mask, size=(H_native, W_native),
                                        mode='nearest')
            
            # Channel + decode
            recv = [model.channel(all_S2[t] * mask, ber) for t in range(T_STEPS)]
            Fp = model.decoder(recv, mask)
            correct += back(Fp).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    
    return 100. * correct / total


def run_granularity_ablation(dataset_name, n_classes, model, back, front, test_loader):
    """Run full granularity ablation for one dataset."""
    
    grid_sizes = [14, 7, 4]
    rho_values = [0.50, 0.75]
    ber_values = [0.0, 0.15, 0.30]
    n_random_draws = 20  # Average over 20 random draws
    
    results = {}
    
    for gs in grid_sizes:
        gs_key = f"{gs}x{gs}"
        results[gs_key] = {}
        n_blocks = gs * gs
        
        print(f"\n{'='*60}")
        print(f"GRANULARITY {gs}×{gs} ({n_blocks} blocks) — {dataset_name.upper()}")
        print(f"{'='*60}")
        
        for rho in rho_values:
            results[gs_key][str(rho)] = {}
            
            for ber in ber_values:
                lab = "Clean" if ber == 0 else f"BER={ber}"
                
                # Learned mask
                learned = evaluate_at_granularity(
                    model, back, front, test_loader, ber, rho, gs, 'learned')
                
                # Random mask (average over draws)
                rand_accs = []
                for draw in range(n_random_draws):
                    acc = evaluate_at_granularity(
                        model, back, front, test_loader, ber, rho, gs, 'random')
                    rand_accs.append(acc)
                rand_mean = np.mean(rand_accs)
                rand_std = np.std(rand_accs)
                
                delta = learned - rand_mean
                
                results[gs_key][str(rho)][str(ber)] = {
                    'learned': round(learned, 2),
                    'random_mean': round(rand_mean, 2),
                    'random_std': round(rand_std, 2),
                    'delta_learned_vs_random': round(delta, 2),
                }
                
                n_keep = max(1, int(rho * gs * gs))
                print(f"  ρ={rho:.2f} {lab:>9} ({n_keep}/{n_blocks} blocks): "
                      f"Learned={learned:.2f}%  Random={rand_mean:.2f}±{rand_std:.2f}%  "
                      f"Δ={delta:+.2f}pp")
    
    # Also get ρ=1.0 (no masking) baseline for reference
    print(f"\n  Baseline (ρ=1.0, no masking):")
    results['baseline'] = {}
    for ber in ber_values:
        lab = "Clean" if ber == 0 else f"BER={ber}"
        acc = evaluate_at_granularity(
            model, back, front, test_loader, ber, 1.0, 14, 'full')
        results['baseline'][str(ber)] = round(acc, 2)
        print(f"    {lab}: {acc:.2f}%")
    
    return results


def main():
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    torch.backends.cudnn.deterministic = True
    print(f"Device: {device}")
    
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])
    
    all_results = {}
    
    # ========== AID ==========
    print(f"\n{'#'*60}")
    print(f"AID (50/50)")
    print(f"{'#'*60}")
    
    test_ds = AIDDataset5050("./data", tf_test, 'test', seed=42)
    test_loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)
    
    front = ResNet50Front(grid_size=14).to(device)
    bb = torch.load("./snapshots_aid_5050_seed42/backbone_best.pth", map_location=device)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))}, strict=False)
    front.eval()
    for p in front.parameters(): p.requires_grad = False
    
    back = ResNet50Back(30).to(device)
    
    model = SpikeAdaptSC_v5c_NA(C_in=1024, C1=256, C2=36, T=T_STEPS,
                                 target_rate=0.75, grid_size=14).to(device)
    
    ck = torch.load("./snapshots_aid_v5cna_seed42/v5cna_best_95.42.pth", map_location=device)
    model.load_state_dict(ck['model'])
    back.load_state_dict(ck['back'])
    
    all_results['aid'] = run_granularity_ablation('aid', 30, model, back, front, test_loader)
    
    del model, back, front
    torch.cuda.empty_cache()
    
    # ========== RESISC45 ==========
    print(f"\n{'#'*60}")
    print(f"RESISC45 (20/80)")
    print(f"{'#'*60}")
    
    test_ds = RESISC45Dataset("./data", tf_test, 'test', train_ratio=0.20, seed=42)
    test_loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)
    
    front = ResNet50Front(grid_size=14).to(device)
    bb = torch.load("./snapshots_resisc45_5050_seed42/backbone_best.pth", map_location=device)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))}, strict=False)
    front.eval()
    for p in front.parameters(): p.requires_grad = False
    
    back = ResNet50Back(45).to(device)
    
    model = SpikeAdaptSC_v5c_NA(C_in=1024, C1=256, C2=36, T=T_STEPS,
                                 target_rate=0.75, grid_size=14).to(device)
    
    ck = torch.load("./snapshots_resisc45_v5cna_seed42/v5cna_best_92.01.pth", map_location=device)
    model.load_state_dict(ck['model'])
    back.load_state_dict(ck['back'])
    
    all_results['resisc45'] = run_granularity_ablation('resisc45', 45, model, back, front, test_loader)
    
    # Save
    with open("eval/granularity_ablation_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # ========== PRINT SUMMARY TABLE ==========
    print(f"\n{'='*70}")
    print("SUMMARY: Granularity Ablation (Learned Mask Accuracy)")
    print(f"{'='*70}")
    print(f"{'Grid':>8} {'ρ':>6} {'Metric':>10} {'AID':>10} {'RESISC45':>10}")
    print("-" * 50)
    
    for gs in ['14x14', '7x7', '4x4']:
        for rho in ['0.75', '0.5']:
            for ber_key, ber_label in [('0.0', 'Clean'), ('0.3', 'BER=0.30')]:
                aid_v = all_results['aid'][gs][rho][ber_key]['learned']
                res_v = all_results['resisc45'][gs][rho][ber_key]['learned']
                print(f"{gs:>8} {rho:>6} {ber_label:>10} {aid_v:>10.2f} {res_v:>10.2f}")
        print()
    
    print(f"\n{'='*70}")
    print("SUMMARY: Learned-vs-Random Gap (Δ pp) — Does Granularity Help?")
    print(f"{'='*70}")
    print(f"{'Grid':>8} {'ρ':>6} {'BER':>8} {'AID Δ':>10} {'R45 Δ':>10}")
    print("-" * 50)
    
    for gs in ['14x14', '7x7', '4x4']:
        for rho in ['0.75', '0.5']:
            for ber_key in ['0.0', '0.3']:
                aid_d = all_results['aid'][gs][rho][ber_key]['delta_learned_vs_random']
                res_d = all_results['resisc45'][gs][rho][ber_key]['delta_learned_vs_random']
                lab = "Clean" if ber_key == '0.0' else "0.30"
                print(f"{gs:>8} {rho:>6} {lab:>8} {aid_d:>+10.2f} {res_d:>+10.2f}")
        print()
    
    print(f"\n✅ Granularity ablation complete")
    print(f"Results saved to eval/granularity_ablation_results.json")


if __name__ == "__main__":
    main()
