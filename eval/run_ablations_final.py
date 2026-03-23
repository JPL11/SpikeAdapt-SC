"""Comprehensive ablations for V5C-NA on AID (50/50) and RESISC45 (20/80).

Runs:
  1. ρ sweep: accuracy at ρ={0.10, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 1.0}
     for BER={0.0, 0.15, 0.30}
  2. Mask comparison: learned vs random (3 draws) vs uniform at ρ=0.50 and ρ=0.75
     for BER={0.0, 0.15, 0.30}
  3. Noise-awareness test: Hamming distance of masks at BER=0 vs BER=0.30

Usage:
  python eval/run_ablations_final.py
"""

import os, sys, json, random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from train_aid_v2 import ResNet50Front, ResNet50Back, BSC_Channel, LearnedBlockMask, sample_noise
from train_aid_v5 import EncoderV5, DecoderV5, SpikeFunction_Learnable, LIFNeuron, BNTT, MPBN
from noise_aware_scorer import NoiseAwareScorer
from run_final_pipeline import AIDDataset5050, RESISC45Dataset, SpikeAdaptSC_v5c_NA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_STEPS = 8


def evaluate(model, back, front, test_loader, ber=0.0, target_rate=None):
    """Evaluate model at given BER and optional rate override."""
    model.eval(); back.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            if target_rate is not None:
                Fp, _ = model(feat, noise_param=ber, target_rate_override=target_rate)
            else:
                Fp, _ = model(feat, noise_param=ber)
            correct += back(Fp).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total


def evaluate_random_mask(model, back, front, test_loader, ber, rho, n_draws=50):
    """Evaluate with random masks (average over draws)."""
    model.eval(); back.eval()
    accs = []
    for draw in range(n_draws):
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
                
                # Random mask
                H, W = all_S2[0].shape[2], all_S2[0].shape[3]
                n_keep = max(1, int(rho * H * W))
                mask = torch.zeros(B, 1, H, W, device=device)
                for b in range(B):
                    indices = torch.randperm(H * W)[:n_keep]
                    mask[b, 0].view(-1)[indices] = 1.0
                
                # Channel + decode
                recv = [model.channel(all_S2[t] * mask, ber) for t in range(T_STEPS)]
                Fp = model.decoder(recv, mask)
                correct += back(Fp).argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        accs.append(100. * correct / total)
    return np.mean(accs), np.std(accs)


def evaluate_uniform_mask(model, back, front, test_loader, ber, rho):
    """Evaluate with uniform subsampling mask."""
    model.eval(); back.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            B = imgs.size(0)
            
            all_S2, m1, m2 = [], None, None
            for t in range(T_STEPS):
                _, s2, m1, m2 = model.encoder(feat, m1, m2, t=t)
                all_S2.append(s2)
            
            # Uniform mask: keep every k-th block
            H, W = all_S2[0].shape[2], all_S2[0].shape[3]
            n_keep = max(1, int(rho * H * W))
            step = max(1, (H * W) // n_keep)
            mask = torch.zeros(B, 1, H, W, device=device)
            indices = list(range(0, H * W, step))[:n_keep]
            for idx in indices:
                r, c = idx // W, idx % W
                mask[:, 0, r, c] = 1.0
            
            recv = [model.channel(all_S2[t] * mask, ber) for t in range(T_STEPS)]
            Fp = model.decoder(recv, mask)
            correct += back(Fp).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total


def noise_awareness_test(model, front, test_loader):
    """Test: do masks actually change with BER?"""
    model.eval()
    hamming_distances = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            feat = front(imgs)
            
            all_S2, m1, m2 = [], None, None
            for t in range(T_STEPS):
                _, s2, m1, m2 = model.encoder(feat, m1, m2, t=t)
                all_S2.append(s2)
            
            stats = model.scorer.get_mask_stats(all_S2, [0.0, 0.10, 0.20, 0.30])
            for k, v in stats.items():
                if k not in hamming_distances:
                    hamming_distances[k] = []
                hamming_distances[k].append(v)
            # Collect first 10 batches
            if len(hamming_distances.get('hamming_0_vs_0.3', [])) >= 10:
                break
    
    result = {}
    for k, vals in hamming_distances.items():
        result[k] = {'mean': np.mean(vals), 'std': np.std(vals)}
    return result


def run_ablations(dataset_name, n_classes, model, back, front, test_loader):
    """Run all ablations for one dataset."""
    results = {}
    
    # 1. ρ sweep
    print(f"\n{'='*60}")
    print(f"ρ SWEEP — {dataset_name.upper()}")
    print(f"{'='*60}")
    rho_values = [0.10, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 1.0]
    ber_values = [0.0, 0.15, 0.30]
    
    rho_results = {}
    for rho in rho_values:
        rho_results[str(rho)] = {}
        for ber in ber_values:
            acc = evaluate(model, back, front, test_loader, ber=ber, target_rate=rho)
            rho_results[str(rho)][str(ber)] = acc
            lab = "Clean" if ber == 0 else f"BER={ber}"
            print(f"  ρ={rho:.3f} {lab}: {acc:.2f}%")
    results['rho_sweep'] = rho_results
    
    # 2. Mask comparison
    print(f"\n{'='*60}")
    print(f"MASK COMPARISON — {dataset_name.upper()}")
    print(f"{'='*60}")
    
    mask_results = {}
    for rho in [0.50, 0.75]:
        mask_results[str(rho)] = {}
        for ber in ber_values:
            learned = evaluate(model, back, front, test_loader, ber=ber, target_rate=rho)
            rand_mean, rand_std = evaluate_random_mask(model, back, front, test_loader, ber, rho, n_draws=50)
            uniform = evaluate_uniform_mask(model, back, front, test_loader, ber, rho)
            
            mask_results[str(rho)][str(ber)] = {
                'learned': learned,
                'random_mean': rand_mean,
                'random_std': rand_std,
                'uniform': uniform,
                'delta_random': learned - rand_mean,
            }
            lab = "Clean" if ber == 0 else f"BER={ber}"
            print(f"  ρ={rho} {lab}: Learned={learned:.2f}% Random={rand_mean:.2f}±{rand_std:.2f}% "
                  f"Uniform={uniform:.2f}% Δ={learned-rand_mean:+.2f}")
    results['mask_comparison'] = mask_results
    
    # 3. Noise awareness
    print(f"\n{'='*60}")
    print(f"NOISE AWARENESS — {dataset_name.upper()}")
    print(f"{'='*60}")
    try:
        noise_stats = noise_awareness_test(model, front, test_loader)
        for k, v in noise_stats.items():
            print(f"  {k}: {v['mean']:.6f} ± {v['std']:.6f}")
        results['noise_awareness'] = {k: v['mean'] for k, v in noise_stats.items()}
    except Exception as e:
        print(f"  Noise awareness test error: {e}")
        results['noise_awareness'] = str(e)
    
    return results


def main():
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
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
    
    all_results['aid'] = run_ablations('aid', 30, model, back, front, test_loader)
    
    # Free memory
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
    
    all_results['resisc45'] = run_ablations('resisc45', 45, model, back, front, test_loader)
    
    # Save all
    with open("eval/ablation_final_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary tables
    print(f"\n{'='*60}")
    print("SUMMARY: ρ SWEEP (BER=0.30)")
    print(f"{'='*60}")
    print(f"{'ρ':>8} {'AID':>10} {'RESISC45':>10}")
    print("-" * 30)
    for rho in ['0.10', '0.25', '0.375', '0.5', '0.625', '0.75', '0.875', '1.0']:
        aid_val = all_results['aid']['rho_sweep'].get(rho, {}).get('0.3', 0)
        res_val = all_results['resisc45']['rho_sweep'].get(rho, {}).get('0.3', 0)
        print(f"{rho:>8} {aid_val:>10.2f} {res_val:>10.2f}")
    
    print(f"\n{'='*60}")
    print("SUMMARY: MASK COMPARISON (ρ=0.50, BER=0.30)")
    print(f"{'='*60}")
    for ds in ['aid', 'resisc45']:
        mc = all_results[ds]['mask_comparison']['0.5']['0.3']
        print(f"  {ds.upper()}: Learned={mc['learned']:.2f}% Random={mc['random_mean']:.2f}% "
              f"Uniform={mc['uniform']:.2f}% Δ={mc['delta_random']:+.2f}")
    
    print(f"\n✅ ALL ABLATIONS COMPLETE")
    print(f"Results saved to eval/ablation_final_results.json")


if __name__ == "__main__":
    main()
