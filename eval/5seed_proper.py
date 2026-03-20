"""Proper 5-seed variance: retrain S3 (scorer fine-tuning) with different seeds.

For each seed:
  1. Load the best V5C-NA checkpoint (trained with seed=42)
  2. Re-initialize the scorer weights randomly with the new seed
  3. Retrain S3 (scorer + joint fine-tune) for 20 epochs
  4. Evaluate clean and BER=0.30

This gives TRUE model variance from scorer training randomness.
Uses the SAME data split (seed=42) for all runs — only model randomness varies.
"""

import os, sys, json, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from train_aid_v2 import ResNet50Front, ResNet50Back, BSC_Channel, LearnedBlockMask, sample_noise
from train_aid_v5 import EncoderV5, DecoderV5
from noise_aware_scorer import NoiseAwareScorer
from run_final_pipeline import AIDDataset5050, RESISC45Dataset, SpikeAdaptSC_v5c_NA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_STEPS = 8


def train_s3_with_seed(model, back, front, train_loader, test_loader, seed, epochs=20):
    """Retrain S3 (scorer + joint fine-tune) with a specific random seed."""
    # Set seed for this run (affects data shuffling, noise sampling, dropout)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Re-initialize scorer weights with new seed (key source of variance)
    for m in model.scorer.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    # Re-initialize block_mask slope
    if hasattr(model.block_mask, 'slope'):
        model.block_mask.slope.data.fill_(5.0)
    
    # S3 optimizer
    slope_params = [p for n, p in model.named_parameters() if 'slope' in n]
    other_params = [p for n, p in model.named_parameters() if 'slope' not in n]
    optimizer = optim.Adam([
        {'params': other_params, 'lr': 1e-5},
        {'params': slope_params, 'lr': 1e-4},
        {'params': back.parameters(), 'lr': 1e-5}
    ])
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    best_state = None
    
    for epoch in range(1, epochs + 1):
        model.train(); back.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            ber = sample_noise('bsc')
            
            Fp, stats = model(feat, noise_param=ber)
            loss = criterion(back(Fp), labels)
            loss = loss + 0.1 * stats.get('rate_penalty', 0)
            
            # Diversity loss
            diversity_loss = model.scorer.compute_diversity_loss(
                stats['all_S2'], ber_low=0.0, ber_high=0.30
            )
            loss = loss + 0.05 * diversity_loss
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        if epoch % 5 == 0 or epoch == epochs:
            model.eval(); back.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    Fp, _ = model(front(imgs), noise_param=0.0)
                    correct += back(Fp).argmax(1).eq(labels).sum().item()
                    total += labels.size(0)
            acc = 100. * correct / total
            if acc > best_acc:
                best_acc = acc
                best_state = {
                    'model': {k: v.clone() for k, v in model.state_dict().items()},
                    'back': {k: v.clone() for k, v in back.state_dict().items()},
                }
    
    # Restore best and evaluate at multiple BERs
    model.load_state_dict(best_state['model'])
    back.load_state_dict(best_state['back'])
    model.eval(); back.eval()
    
    results = {}
    for ber in [0.0, 0.15, 0.30]:
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                Fp, _ = model(front(imgs), noise_param=ber)
                correct += back(Fp).argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        results[str(ber)] = 100. * correct / total
    
    return results


def run_5seed_proper(ds_name, DSClass, n_classes, ds_args, bb_dir, v5c_dir, seeds):
    """Run proper 5-seed variance for one dataset."""
    # Same data split for all seeds (seed=42)
    tf_train = T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
                           T.ColorJitter(0.2, 0.2, 0.2),
                           T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))])
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])
    
    train_ds = DSClass(transform=tf_train, split='train', seed=42, **ds_args)
    test_ds = DSClass(transform=tf_test, split='test', seed=42, **ds_args)
    train_loader = DataLoader(train_ds, 32, True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)
    
    # Load backbone
    front = ResNet50Front(grid_size=14).to(device)
    bb = torch.load(f"./{bb_dir}/backbone_best.pth", map_location=device)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))}, strict=False)
    front.eval()
    for p in front.parameters(): p.requires_grad = False
    
    # Find best checkpoint
    ck_files = sorted([f for f in os.listdir(f"./{v5c_dir}") if f.startswith('v5cna_best')])
    best_ck = f"./{v5c_dir}/{ck_files[-1]}"
    base_ck = torch.load(best_ck, map_location=device)
    
    all_results = []
    for seed in seeds:
        t0 = time.time()
        print(f"\n  --- Seed {seed} ---")
        
        # Create fresh model and load S2 weights (encoder/decoder)
        back = ResNet50Back(n_classes).to(device)
        model = SpikeAdaptSC_v5c_NA(C_in=1024, C1=256, C2=36, T=T_STEPS,
                                     target_rate=0.75, grid_size=14).to(device)
        
        # Load encoder/decoder weights from base checkpoint (these stay fixed across seeds)
        model.load_state_dict(base_ck['model'])
        back.load_state_dict(base_ck['back'])
        
        # Retrain S3 with new seed (re-initializes scorer)
        results = train_s3_with_seed(model, back, front, train_loader, test_loader, seed, epochs=20)
        
        all_results.append(results)
        elapsed = time.time() - t0
        print(f"  Seed {seed}: Clean={results['0.0']:.2f}%, "
              f"BER=0.15={results['0.15']:.2f}%, BER=0.30={results['0.3']:.2f}% "
              f"({elapsed:.0f}s)")
        
        del model, back
        torch.cuda.empty_cache()
    
    # Compute statistics
    clean_vals = [r['0.0'] for r in all_results]
    ber15_vals = [r['0.15'] for r in all_results]
    ber30_vals = [r['0.3'] for r in all_results]
    
    stats = {
        'seeds': seeds,
        'results': all_results,
        'clean': {'mean': np.mean(clean_vals), 'std': np.std(clean_vals), 'values': clean_vals},
        'ber_0.15': {'mean': np.mean(ber15_vals), 'std': np.std(ber15_vals), 'values': ber15_vals},
        'ber_0.30': {'mean': np.mean(ber30_vals), 'std': np.std(ber30_vals), 'values': ber30_vals},
    }
    
    print(f"\n  {ds_name.upper()} 5-seed Summary:")
    print(f"    Clean:    {stats['clean']['mean']:.2f} ± {stats['clean']['std']:.2f}%")
    print(f"    BER=0.15: {stats['ber_0.15']['mean']:.2f} ± {stats['ber_0.15']['std']:.2f}%")
    print(f"    BER=0.30: {stats['ber_0.30']['mean']:.2f} ± {stats['ber_0.30']['std']:.2f}%")
    
    return stats


if __name__ == "__main__":
    print(f"Device: {device}")
    seeds = [42, 101, 123, 456, 789]
    
    all_stats = {}
    
    # AID
    print(f"\n{'='*60}")
    print(f"5-SEED PROPER VARIANCE — AID (50/50)")
    print(f"{'='*60}")
    all_stats['aid'] = run_5seed_proper(
        'aid', AIDDataset5050, 30, {"root": "./data"},
        'snapshots_aid_5050_seed42', 'snapshots_aid_v5cna_seed42', seeds
    )
    
    # Free GPU
    torch.cuda.empty_cache()
    
    # RESISC45
    print(f"\n{'='*60}")
    print(f"5-SEED PROPER VARIANCE — RESISC45 (20/80)")
    print(f"{'='*60}")
    all_stats['resisc45'] = run_5seed_proper(
        'resisc45', RESISC45Dataset, 45, {"root": "./data", "train_ratio": 0.20},
        'snapshots_resisc45_5050_seed42', 'snapshots_resisc45_v5cna_seed42', seeds
    )
    
    # Save
    # Convert numpy types to Python types for JSON
    def convert(obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj
    
    with open("eval/5seed_proper_results.json", 'w') as f:
        json.dump(all_stats, f, indent=2, default=convert)
    
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    for ds in ['aid', 'resisc45']:
        s = all_stats[ds]
        print(f"\n  {ds.upper()}:")
        print(f"    Clean:    {s['clean']['mean']:.2f} ± {s['clean']['std']:.2f}%  {s['clean']['values']}")
        print(f"    BER=0.15: {s['ber_0.15']['mean']:.2f} ± {s['ber_0.15']['std']:.2f}%")
        print(f"    BER=0.30: {s['ber_0.30']['mean']:.2f} ± {s['ber_0.30']['std']:.2f}%")
    
    print(f"\n✅ Proper 5-seed variance complete. Results: eval/5seed_proper_results.json")
