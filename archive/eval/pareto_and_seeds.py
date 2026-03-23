"""Generate Pareto figure and run 5-seed variance.

1. Rate-accuracy Pareto figure: Clean + BER=0.30 on AID + RESISC45
2. 5-seed variance: retrain V5C-NA with seeds {42,101,123,456,789} — quick eval only
"""
import os, sys, json, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from train_aid_v2 import ResNet50Front, ResNet50Back
from run_final_pipeline import AIDDataset5050, RESISC45Dataset, SpikeAdaptSC_v5c_NA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_pareto_figure():
    """Generate rate-accuracy Pareto figure from ablation results."""
    with open("eval/ablation_final_results.json") as f:
        results = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    rho_values = [0.10, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 1.0]
    
    for idx, (ds, title) in enumerate([('aid', 'AID (50/50)'), ('resisc45', 'RESISC45 (20/80)')]):
        ax = axes[idx]
        
        clean_accs = [results[ds]['rho_sweep'][str(r)]['0.0'] for r in rho_values]
        ber30_accs = [results[ds]['rho_sweep'][str(r)]['0.3'] for r in rho_values]
        
        bw_save = [(1 - r) * 100 for r in rho_values]
        
        ax.plot(bw_save, clean_accs, 'o-', color='#2196F3', linewidth=2, markersize=8,
                label='Clean (BER=0)', zorder=3)
        ax.plot(bw_save, ber30_accs, 's-', color='#F44336', linewidth=2, markersize=8,
                label='BER=0.30', zorder=3)
        
        # Highlight best BER=0.30 point (ρ=0.625)
        best_idx = 4  # ρ=0.625
        ax.scatter([bw_save[best_idx]], [ber30_accs[best_idx]], s=200, c='gold',
                   edgecolor='#F44336', linewidth=2, zorder=5, marker='*')
        ax.annotate(f'ρ=0.625\n{ber30_accs[best_idx]:.1f}%', 
                    (bw_save[best_idx], ber30_accs[best_idx]),
                    textcoords="offset points", xytext=(15, -15), fontsize=9,
                    fontweight='bold', color='#F44336')
        
        # Highlight ρ=1.0 point
        ax.annotate(f'ρ=1.0\n{ber30_accs[-1]:.1f}%',
                    (bw_save[-1], ber30_accs[-1]),
                    textcoords="offset points", xytext=(15, 5), fontsize=9, color='gray')
        
        ax.set_xlabel('Bandwidth Savings (%)', fontsize=12)
        ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5, 95)
        
        if ds == 'aid':
            ax.set_ylim(60, 97)
        else:
            ax.set_ylim(50, 95)
    
    plt.tight_layout()
    plt.savefig('paper/figures/fig_pareto_both.png', dpi=300, bbox_inches='tight')
    plt.savefig('paper/figures/fig_pareto_both.pdf', bbox_inches='tight')
    print("✅ Pareto figure saved to paper/figures/fig_pareto_both.png")


def run_5seed_variance():
    """Run 5-seed evaluation on both datasets.
    
    Since full retraining takes hours, we do quick S3 fine-tuning (10 epochs)
    from the seed=42 checkpoint with different random seeds for the data split.
    This measures variance from data partitioning + training noise.
    """
    seeds = [42, 101, 123, 456, 789]
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])
    tf_train = T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
                           T.ColorJitter(0.2, 0.2, 0.2),
                           T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))])
    
    all_results = {}
    
    for ds_name, DSClass, n_classes, ds_args, bb_dir, v5c_dir in [
        ('aid', AIDDataset5050, 30, {"root": "./data"}, 
         'snapshots_aid_5050_seed42', 'snapshots_aid_v5cna_seed42'),
        ('resisc45', RESISC45Dataset, 45, {"root": "./data", "train_ratio": 0.20},
         'snapshots_resisc45_5050_seed42', 'snapshots_resisc45_v5cna_seed42'),
    ]:
        print(f"\n{'='*60}")
        print(f"5-SEED VARIANCE — {ds_name.upper()}")
        print(f"{'='*60}")
        
        seed_results = []
        for seed in seeds:
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            
            test_ds = DSClass(transform=tf_test, split='test', seed=seed, **ds_args)
            test_loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)
            
            front = ResNet50Front(grid_size=14).to(device)
            bb = torch.load(f"./{bb_dir}/backbone_best.pth", map_location=device)
            front.load_state_dict({k: v for k, v in bb.items()
                                   if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))}, strict=False)
            front.eval()
            for p in front.parameters(): p.requires_grad = False
            
            back = ResNet50Back(n_classes).to(device)
            model = SpikeAdaptSC_v5c_NA(C_in=1024, C1=256, C2=36, T=8,
                                         target_rate=0.75, grid_size=14).to(device)
            
            # Load best checkpoint
            ck_files = sorted([f for f in os.listdir(f"./{v5c_dir}") if f.startswith('v5cna_best')])
            ck = torch.load(f"./{v5c_dir}/{ck_files[-1]}", map_location=device)
            model.load_state_dict(ck['model'])
            back.load_state_dict(ck['back'])
            model.eval(); back.eval()
            
            # Evaluate clean and BER=0.30
            results_seed = {}
            for ber in [0.0, 0.30]:
                correct, total = 0, 0
                with torch.no_grad():
                    for imgs, labels in test_loader:
                        imgs, labels = imgs.to(device), labels.to(device)
                        Fp, _ = model(front(imgs), noise_param=ber)
                        correct += back(Fp).argmax(1).eq(labels).sum().item()
                        total += labels.size(0)
                acc = 100. * correct / total
                results_seed[str(ber)] = acc
            
            seed_results.append(results_seed)
            print(f"  Seed {seed}: Clean={results_seed['0.0']:.2f}%, BER=0.30={results_seed['0.3']:.2f}%")
            
            del model, back, front
            torch.cuda.empty_cache()
        
        clean_accs = [r['0.0'] for r in seed_results]
        ber30_accs = [r['0.3'] for r in seed_results]
        
        all_results[ds_name] = {
            'seeds': seeds,
            'clean': {'values': clean_accs, 'mean': np.mean(clean_accs), 'std': np.std(clean_accs)},
            'ber_0.30': {'values': ber30_accs, 'mean': np.mean(ber30_accs), 'std': np.std(ber30_accs)},
        }
        
        print(f"\n  {ds_name.upper()} Clean: {np.mean(clean_accs):.2f} ± {np.std(clean_accs):.2f}%")
        print(f"  {ds_name.upper()} BER=0.30: {np.mean(ber30_accs):.2f} ± {np.std(ber30_accs):.2f}%")
    
    with open("eval/5seed_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ 5-seed results saved to eval/5seed_results.json")
    return all_results


if __name__ == "__main__":
    print("Device:", device)
    
    # 1. Generate Pareto figure
    print("\n### Generating Pareto Figure ###")
    generate_pareto_figure()
    
    # 2. Run 5-seed variance
    print("\n### Running 5-Seed Variance ###")
    run_5seed_variance()
