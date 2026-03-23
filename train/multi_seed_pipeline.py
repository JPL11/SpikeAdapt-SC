#!/usr/bin/env python3
"""5-seed reproducibility pipeline: full retraining + evaluation at ρ=1.0, 0.75, 0.625.

PAPER TABLES GENERATED:
  - Table I   (main.tex)  : Main BSC results (5-seed mean±std)
  - Table V   (main.tex)  : Cross-dataset generalization (5-seed mean±std)
  - Table 1   (6-page)    : Main BSC results (5-seed mean±std)

Output:  eval/seed_results/summary_5seed.json  (+ per-seed JSONs)

Trains the full pipeline (backbone → S2 → S3) for 5 seeds on both datasets,
then evaluates at ρ=1.0, 0.75, and 0.625 under BER=0.0 and BER=0.30.
Reports mean ± std and paired t-test p-values.

Usage:
    python train/multi_seed_pipeline.py          # train all seeds
    python train/multi_seed_pipeline.py --seed 123  # train one seed
"""
import torch, torch.nn as nn, torch.optim as optim
import sys, os, json, random, argparse
import numpy as np
from scipy import stats as scipy_stats
from torch.utils.data import DataLoader
from torchvision import transforms as T

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from run_final_pipeline import (
    SpikeAdaptSC_v5c_NA, AIDDataset5050, RESISC45Dataset,
    train_backbone, train_v5c_na, sample_noise, T_STEPS
)
from train_aid_v2 import ResNet50Front, ResNet50Back

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 123, 456, 789, 1024]


def evaluate(model, back, front, test_loader, ber=0.0, rho=None):
    model.eval(); back.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            kwargs = {'noise_param': ber}
            if rho is not None:
                kwargs['target_rate_override'] = rho
            Fp, _ = model(front(imgs), **kwargs)
            correct += back(Fp).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total


def train_and_eval_seed(dataset_name, n_classes, seed):
    """Train backbone + S2 + S3 for one seed, then evaluate."""
    print(f"\n{'='*60}")
    print(f"  {dataset_name.upper()} seed={seed}")
    print(f"{'='*60}", flush=True)
    
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    
    tf_train = T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
                           T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))])
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])
    
    if dataset_name == 'aid':
        train_ds = AIDDataset5050('./data', tf_train, 'train', seed=seed)
        test_ds = AIDDataset5050('./data', tf_test, 'test', seed=seed)
    else:
        train_ds = RESISC45Dataset('./data', tf_train, 'train', train_ratio=0.20, seed=seed)
        test_ds = RESISC45Dataset('./data', tf_test, 'test', train_ratio=0.20, seed=seed)
    
    train_loader = DataLoader(train_ds, 32, True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)
    
    snap_dir = f'./snapshots_{dataset_name}_5050_seed{seed}/'
    bb_path = os.path.join(snap_dir, 'backbone_best.pth')
    
    # Train backbone if not exists
    if not os.path.exists(bb_path):
        train_backbone(dataset_name, n_classes, train_loader, test_loader, seed, epochs=50)
    else:
        print(f"  Backbone exists: {bb_path}", flush=True)
    
    # Train V5C-NA if not exists
    v5cna_dir = f'./snapshots_{dataset_name}_v5cna_seed{seed}/'
    manifest = os.path.join(v5cna_dir, 'selected_checkpoint.txt')
    existing = [f for f in os.listdir(v5cna_dir) if f.startswith('v5cna_best_') and f.endswith('.pth')] if os.path.exists(v5cna_dir) else []
    
    if not existing:
        train_v5c_na(dataset_name, n_classes, train_loader, test_loader, seed, bb_path,
                     epochs_s2=60, epochs_s3=40)
        existing = [f for f in os.listdir(v5cna_dir) if f.startswith('v5cna_best_') and f.endswith('.pth')]
    
    # Checkpoint selection: use manifest if it exists, otherwise pick best and write manifest
    if os.path.exists(manifest):
        best_ck = open(manifest).read().strip()
        if not os.path.exists(os.path.join(v5cna_dir, best_ck)):
            print(f"  WARNING: manifest points to {best_ck} but file missing, re-selecting")
            best_ck = sorted(existing, key=lambda x: float(x.split('_')[-1].replace('.pth','')))[-1]
            with open(manifest, 'w') as f: f.write(best_ck)
    else:
        best_ck = sorted(existing, key=lambda x: float(x.split('_')[-1].replace('.pth','')))[-1]
        with open(manifest, 'w') as f: f.write(best_ck)
        print(f"  Wrote manifest: {manifest} -> {best_ck}", flush=True)
    ck_path = os.path.join(v5cna_dir, best_ck)
    print(f"  Using checkpoint: {ck_path} (pinned via manifest)", flush=True)
    
    # Load model
    front = ResNet50Front(grid_size=14).to(device)
    bb = torch.load(bb_path, map_location=device)
    front.load_state_dict({k:v for k,v in bb.items()
                           if not k.startswith(('layer4.','fc.','avgpool.','spatial_pool.'))}, strict=False)
    front.eval()
    for p in front.parameters(): p.requires_grad = False
    
    back = ResNet50Back(n_classes).to(device)
    model = SpikeAdaptSC_v5c_NA(C_in=1024, C1=256, C2=36, T=T_STEPS,
                                 target_rate=0.75, grid_size=14).to(device)
    ck = torch.load(ck_path, map_location=device)
    model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
    model.eval(); back.eval()
    
    # Evaluate at ρ=1.0, ρ=0.75, and ρ=0.625, BER=0.0 and BER=0.30
    results = {}
    for rho in [1.0, 0.75, 0.625]:
        results[str(rho)] = {}
        for ber in [0.0, 0.30]:
            acc = evaluate(model, back, front, test_loader, ber=ber, rho=rho)
            results[str(rho)][str(ber)] = round(acc, 2)
            print(f"  ρ={rho} BER={ber}: {acc:.2f}%", flush=True)
    
    del model, back, front
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None, help='Single seed to train')
    parser.add_argument('--eval-only', action='store_true', help='Only evaluate existing checkpoints')
    args = parser.parse_args()
    
    seeds = [args.seed] if args.seed else SEEDS
    
    os.makedirs('eval/seed_results', exist_ok=True)
    
    all_results = {}
    
    for ds_name, n_classes in [('aid', 30), ('resisc45', 45)]:
        all_results[ds_name] = {}
        for seed in seeds:
            result = train_and_eval_seed(ds_name, n_classes, seed)
            all_results[ds_name][str(seed)] = result
            
            # Save per-seed result
            with open(f'eval/seed_results/{ds_name}_seed{seed}.json', 'w') as f:
                json.dump(result, f, indent=2)
    
    # Compute statistics
    print(f"\n{'='*60}")
    print(f"  5-SEED SUMMARY")
    print(f"{'='*60}", flush=True)
    
    summary = {}
    for ds_name in ['aid', 'resisc45']:
        summary[ds_name] = {}
        for rho in ['1.0', '0.75', '0.625']:
            summary[ds_name][rho] = {}
            for ber in ['0.0', '0.3']:
                accs = [all_results[ds_name][str(s)][rho][ber] for s in seeds]
                mean_acc = np.mean(accs)
                std_acc = np.std(accs, ddof=1)  # sample std
                summary[ds_name][rho][ber] = {
                    'mean': round(mean_acc, 2),
                    'std': round(std_acc, 2),
                    'per_seed': accs
                }
                print(f"  {ds_name} ρ={rho} BER={ber}: {mean_acc:.2f} ± {std_acc:.2f}", flush=True)
        
        # Paired t-test: ρ=0.625 vs ρ=1.0 at BER=0.30
        masked = [all_results[ds_name][str(s)]['0.625']['0.3'] for s in seeds]
        full = [all_results[ds_name][str(s)]['1.0']['0.3'] for s in seeds]
        t_stat, p_val = scipy_stats.ttest_rel(masked, full)
        delta = np.mean(masked) - np.mean(full)
        summary[ds_name]['paired_test_ber030'] = {
            'delta_mean': round(delta, 2),
            't_stat': round(t_stat, 4),
            'p_value': round(p_val, 6),
            'masked_accs': masked,
            'full_accs': full
        }
        print(f"  {ds_name} PAIRED TEST (ρ=0.625 vs 1.0 @ BER=0.30):")
        print(f"    Δ = {delta:+.2f} pp, t={t_stat:.4f}, p={p_val:.6f}", flush=True)
    
    # Save summary
    with open('eval/seed_results/summary_5seed.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nAll results saved to eval/seed_results/")
    print("Done!", flush=True)


if __name__ == '__main__':
    main()
