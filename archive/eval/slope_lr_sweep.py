"""Slope learning rate sweep for SpikeAdapt-SC V4-A.

Evaluates how the surrogate gradient slope learning rate affects:
  1. Final slope values
  2. Firing rates
  3. Accuracy (clean + BER=0.30)

Runs on AID with 20 epochs each (quick sweep).

Usage:
  python eval/slope_lr_sweep.py
"""

import os, sys, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))

from train_aid_v2 import (
    AIDDataset, ResNet50Front, ResNet50Back, BSC_Channel,
    SpikeAdaptSC_v2, sample_noise
)
from train_aid_v4 import SpikeAdaptSC_v4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_sweep():
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    
    # Data
    tf_train = T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
                           T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))])
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])
    train_ds = AIDDataset("./data", tf_train, split='train', seed=42)
    test_ds = AIDDataset("./data", tf_test, split='test', seed=42)
    train_loader = DataLoader(train_ds, 64, True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, 64, False, num_workers=4, pin_memory=True)
    
    # Front
    front = ResNet50Front(grid_size=14).to(device)
    bb = torch.load("./snapshots_aid/backbone_best.pth", map_location=device)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))}, strict=False)
    front.eval()
    for p in front.parameters(): p.requires_grad = False
    
    # Load V2 checkpoint for transfer
    v2_dir = "./snapshots_aid_v2_seed42/"
    v2_files = sorted([f for f in os.listdir(v2_dir) if f.startswith("v2_s3_")])
    v2_ck = torch.load(os.path.join(v2_dir, v2_files[-1]), map_location=device) if v2_files else None
    
    slope_lrs = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    results = []
    
    print("="*80)
    print("SLOPE LEARNING RATE SWEEP")
    print("="*80)
    
    for slope_lr in slope_lrs:
        torch.manual_seed(42); np.random.seed(42); random.seed(42)
        
        model = SpikeAdaptSC_v4(C_in=1024, C1=256, C2=36, T=8,
                                 target_rate=0.75, grid_size=14,
                                 use_mpbn=False, use_channel_gate=False).to(device)
        back = ResNet50Back(30).to(device)
        
        # Transfer from V2
        if v2_ck:
            m_state = model.state_dict()
            transferred = 0
            for k, v in v2_ck['model'].items():
                if k in m_state and m_state[k].shape == v.shape:
                    m_state[k] = v; transferred += 1
            model.load_state_dict(m_state, strict=False)
            back.load_state_dict(v2_ck['back'])
        
        # Optimizer with separate slope_lr
        slope_params = [p for n, p in model.named_parameters() if 'slope' in n]
        other_params = [p for n, p in model.named_parameters() if 'slope' not in n]
        optimizer = optim.Adam([
            {'params': other_params, 'lr': 1e-5},
            {'params': slope_params, 'lr': slope_lr},
            {'params': back.parameters(), 'lr': 1e-5}
        ])
        criterion = nn.CrossEntropyLoss()
        
        print(f"\n--- Slope LR = {slope_lr} ---")
        
        for epoch in range(1, 21):
            model.train(); back.train()
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                feat = front(imgs)
                ber = sample_noise('bsc')
                Fp, stats = model(feat, noise_param=ber)
                loss = criterion(back(Fp), labels) + 0.1 * stats.get('rate_penalty', 0)
                if 'spike_reg' in stats: loss = loss + 0.01 * stats['spike_reg']
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            
            if epoch == 20:
                # Final eval
                model.eval(); back.eval()
                correct_clean, correct_noisy, total = 0, 0, 0
                fr_sum = 0; n_batches = 0
                with torch.no_grad():
                    for imgs, labels in test_loader:
                        imgs, labels = imgs.to(device), labels.to(device)
                        feat = front(imgs)
                        Fp, stats = model(feat, noise_param=0.0)
                        correct_clean += back(Fp).argmax(1).eq(labels).sum().item()
                        Fp2, _ = model(feat, noise_param=0.30)
                        correct_noisy += back(Fp2).argmax(1).eq(labels).sum().item()
                        total += labels.size(0)
                        fr_sum += stats.get('firing_rate', 0)
                        n_batches += 1
                
                acc_clean = 100. * correct_clean / total
                acc_ber30 = 100. * correct_noisy / total
                avg_fr = fr_sum / n_batches
                
                # Get slope values
                slope_vals = []
                for n, p in model.named_parameters():
                    if 'slope' in n:
                        slope_vals.append(p.item() if p.numel() == 1 else p.mean().item())
                avg_slope = np.mean(slope_vals) if slope_vals else 0
                
                result = {
                    'slope_lr': slope_lr,
                    'clean_acc': acc_clean,
                    'ber30_acc': acc_ber30,
                    'firing_rate': avg_fr,
                    'avg_slope': avg_slope,
                }
                results.append(result)
                print(f"  Clean: {acc_clean:.2f}%, BER=0.30: {acc_ber30:.2f}%, "
                      f"FR: {avg_fr:.3f}, Slope: {avg_slope:.3f}")
    
    # Print table
    print(f"\n{'='*80}")
    print("SLOPE LR SWEEP RESULTS")
    print(f"{'='*80}")
    print(f"{'Slope LR':>10} {'Clean':>8} {'BER=0.30':>10} {'FR':>8} {'Slope':>8}")
    print("-" * 50)
    for r in results:
        print(f"{r['slope_lr']:>10.0e} {r['clean_acc']:>8.2f} {r['ber30_acc']:>10.2f} "
              f"{r['firing_rate']:>8.3f} {r['avg_slope']:>8.3f}")
    
    # Save
    with open("eval/slope_lr_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Slope LR results saved to eval/slope_lr_results.json")
    
    return results


if __name__ == "__main__":
    run_sweep()
