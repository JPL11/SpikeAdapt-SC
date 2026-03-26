#!/usr/bin/env python3
"""Generate 10 individual importance score sigmoid distribution plots.

For each dataset (AID, RESISC45):
  - 10 sorted figures (raw score + bias → sigmoid, individual images)
  - 10 unsorted figures (same images, block index order)
  - 1 mean sorted figure (averaged across 10 images with ±std band)

All using the correct AID/RESISC45 pipeline:
  - ResNet50Front(grid_size=14) → 1024ch, 14×14 spatial
  - SpikeAdaptSC_v5c_NA (C_in=1024, C2=36, T=8, grid_size=14)
  - Input: Resize(256), CenterCrop(224) → 196 spatial blocks
  - Raw logit = content_branch(channel_reweighted_spikes) + spatial_bias(BER)
  - Y-axis = σ(raw logit)

Output: paper/figures/{dataset}_importance_v2/
"""

import os, sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from train_aid_v2 import ResNet50Front, ResNet50Back
from run_final_pipeline import AIDDataset5050, SpikeAdaptSC_v5c_NA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_STEPS = 8

# IEEE style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

C_BLUE = '#1976D2'; C_RED = '#D32F2F'
OUT = 'paper/figures'


# ===================================================================
# Model loading
# ===================================================================
def load_model():
    """Load SpikeAdapt-SC (v5c_NA) with AID-trained weights."""
    front = ResNet50Front(grid_size=14).to(device)
    bb = torch.load('./snapshots_aid_5050_seed42/backbone_best.pth',
                    map_location=device, weights_only=False)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))},
                          strict=False)
    front.eval()

    back = ResNet50Back(30).to(device)
    model = SpikeAdaptSC_v5c_NA(C_in=1024, C1=256, C2=36, T=T_STEPS,
                                 target_rate=0.75, grid_size=14).to(device)

    ck_files = sorted([f for f in os.listdir('./snapshots_aid_v5cna_seed42/')
                       if f.startswith('v5cna_best')])
    ck = torch.load(f'./snapshots_aid_v5cna_seed42/{ck_files[-1]}',
                    map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])
    back.load_state_dict(ck['back'])
    model.eval(); back.eval()
    return front, back, model


def extract_per_image_scores(front, model, imgs_tensor, ber=0.0):
    """Extract pre-sigmoid raw logits and post-sigmoid scores for a batch.
    
    Returns:
        raw_scores: (B, 196) array — pre-sigmoid logits (content_score + spatial_bias)
        post_scores: (B, 196) array — σ(raw_scores)
    """
    with torch.no_grad():
        feat = front(imgs_tensor)
        
        # Run encoder
        all_S2, m1, m2 = [], None, None
        for t in range(model.T):
            _, s2, m1, m2 = model.encoder(feat, m1, m2, t=t)
            all_S2.append(s2)
        
        B = all_S2[0].size(0)
        avg_spikes = torch.stack(all_S2).mean(0)  # B×C×H×W
        
        # BER-dependent channel reweighting
        ber_input = torch.full((B, 1), ber, device=device)
        channel_weights = 1.0 + torch.tanh(
            model.scorer.noise_channel_gate(ber_input))  # B×C
        reweighted = avg_spikes * channel_weights.unsqueeze(-1).unsqueeze(-1)
        
        # Content scoring (pre-sigmoid)
        raw = model.scorer.content_branch(reweighted)  # B×1×H×W
        
        # Global spatial bias
        spatial_bias = model.scorer.noise_spatial_bias(ber_input)  # B×1
        raw = raw + spatial_bias.unsqueeze(-1).unsqueeze(-1)
        
        # Post-sigmoid
        post = torch.sigmoid(raw)
        
        raw_np = raw.view(B, -1).cpu().numpy()   # (B, 196)
        post_np = post.view(B, -1).cpu().numpy()  # (B, 196)
        
    return raw_np, post_np


# ===================================================================
# Dataset setup
# ===================================================================
class RESISC45_2080:
    """RESISC45 with 20/80 train/test split."""
    def __init__(self, root, transform, split='test', seed=42):
        full = ImageFolder(os.path.join(root, 'NWPU-RESISC45'), transform=transform)
        rng = np.random.RandomState(seed)
        indices = list(range(len(full)))
        rng.shuffle(indices)
        n_train = int(0.2 * len(full))
        self.dataset = full
        self.indices = indices[n_train:] if split == 'test' else indices[:n_train]
        self.classes = full.classes
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


# ===================================================================
# Figure generation
# ===================================================================
def gen_all_plots():
    print("=" * 60)
    print("  Importance Score Sigmoid Distribution Plots")
    print("  196 blocks (14×14), raw logit = score + bias")
    print("=" * 60)
    
    front, back, model = load_model()
    
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])
    
    datasets = {
        'AID': AIDDataset5050('./data', tf_test, 'test', seed=42),
        'RESISC45': RESISC45_2080('./data', tf_test, 'test', seed=42),
    }
    
    for ds_name, dataset in datasets.items():
        ds_lower = ds_name.lower()
        out_dir = f'{OUT}/{ds_lower}_importance_v2'
        os.makedirs(out_dir, exist_ok=True)
        
        print(f"\n  === {ds_name} ({len(dataset)} images) ===")
        
        # Pick 10 diverse images
        step = len(dataset) // 10
        indices = [i * step for i in range(10)]
        
        # Extract scores for all 10 images at BER=0 and BER=0.3
        all_raw_0, all_post_0 = [], []
        all_raw_30, all_post_30 = [], []
        class_names = []
        
        for img_idx in indices:
            img, label = dataset[img_idx]
            if hasattr(dataset, 'classes'):
                class_names.append(dataset.classes[label])
            else:
                class_names.append(f'class_{label}')
            
            img_t = img.unsqueeze(0).to(device)
            
            raw_0, post_0 = extract_per_image_scores(front, model, img_t, ber=0.0)
            raw_30, post_30 = extract_per_image_scores(front, model, img_t, ber=0.3)
            
            all_raw_0.append(raw_0[0])     # (196,)
            all_post_0.append(post_0[0])
            all_raw_30.append(raw_30[0])
            all_post_30.append(post_30[0])
        
        all_raw_0 = np.array(all_raw_0)    # (10, 196)
        all_post_0 = np.array(all_post_0)
        all_raw_30 = np.array(all_raw_30)
        all_post_30 = np.array(all_post_30)
        
        # Compute global y-axis zoom range from post-sigmoid values
        all_post = np.concatenate([all_post_0.flat, all_post_30.flat])
        y_min = max(0, np.percentile(all_post, 0.5) - 0.02)
        y_max = min(1, np.percentile(all_post, 99.5) + 0.02)
        if y_max - y_min < 0.05:
            mid = (y_max + y_min) / 2
            y_min, y_max = mid - 0.03, mid + 0.03
        
        print(f"    σ(score) range: [{y_min:.3f}, {y_max:.3f}]")
        print(f"    Raw logit range: [{np.concatenate([all_raw_0.flat, all_raw_30.flat]).min():.3f}, "
              f"{np.concatenate([all_raw_0.flat, all_raw_30.flat]).max():.3f}]")
        
        # ---- 10 individual SORTED figures ----
        for i in range(10):
            fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
            
            sorted_post_0 = np.sort(all_post_0[i])[::-1]
            sorted_post_30 = np.sort(all_post_30[i])[::-1]
            
            x = np.arange(1, 197)  # 1 to 196
            ax.plot(x, sorted_post_0, '-', color=C_BLUE, linewidth=1.8,
                    label='BER=0.0', alpha=0.9)
            ax.plot(x, sorted_post_30, '--', color=C_RED, linewidth=1.5,
                    label='BER=0.3', alpha=0.9)
            
            # ρ=0.75 cutoff
            k = int(0.75 * 196)  # 147
            ax.axvline(x=k, color='green', linestyle=':', linewidth=1.0, alpha=0.6)
            ax.text(k + 2, y_max - 0.005, f'ρ=0.75', fontsize=7, color='green')
            
            ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
            
            ax.set_title(f'{ds_name} — {class_names[i]} (Sorted)',
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('Block Index (sorted by importance ↓)', fontsize=10)
            ax.set_ylabel('σ(score + bias)', fontsize=10)
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(0, 197)
            ax.legend(fontsize=9, loc='upper right')
            ax.grid(True, alpha=0.2)
            
            plt.tight_layout()
            plt.savefig(f'{out_dir}/sorted_{i+1:02d}_{class_names[i]}.png', dpi=200)
            plt.savefig(f'{out_dir}/sorted_{i+1:02d}_{class_names[i]}.pdf', dpi=200)
            plt.close()
        
        print(f"    ✅ 10 sorted figures")
        
        # ---- 10 individual UNSORTED figures ----
        for i in range(10):
            fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
            
            x = np.arange(1, 197)
            ax.plot(x, all_post_0[i], '-', color=C_BLUE, linewidth=1.8,
                    label='BER=0.0', alpha=0.9)
            ax.plot(x, all_post_30[i], '--', color=C_RED, linewidth=1.5,
                    label='BER=0.3', alpha=0.9)
            
            ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
            
            ax.set_title(f'{ds_name} — {class_names[i]} (Unsorted)',
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('Block Index (spatial order)', fontsize=10)
            ax.set_ylabel('σ(score + bias)', fontsize=10)
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(0, 197)
            ax.legend(fontsize=9, loc='upper right')
            ax.grid(True, alpha=0.2)
            
            plt.tight_layout()
            plt.savefig(f'{out_dir}/unsorted_{i+1:02d}_{class_names[i]}.png', dpi=200)
            plt.savefig(f'{out_dir}/unsorted_{i+1:02d}_{class_names[i]}.pdf', dpi=200)
            plt.close()
        
        print(f"    ✅ 10 unsorted figures")
        
        # ---- MEAN sorted figure (across 10 images) ----
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        
        sorted_0 = np.array([np.sort(p)[::-1] for p in all_post_0])   # (10, 196)
        sorted_30 = np.array([np.sort(p)[::-1] for p in all_post_30]) # (10, 196)
        
        mean_0 = sorted_0.mean(axis=0)
        std_0 = sorted_0.std(axis=0)
        mean_30 = sorted_30.mean(axis=0)
        std_30 = sorted_30.std(axis=0)
        
        x = np.arange(1, 197)
        ax.plot(x, mean_0, '-', color=C_BLUE, linewidth=2.0, label='BER=0.0 (mean)')
        ax.fill_between(x, mean_0 - std_0, mean_0 + std_0, alpha=0.15, color=C_BLUE)
        ax.plot(x, mean_30, '--', color=C_RED, linewidth=2.0, label='BER=0.3 (mean)')
        ax.fill_between(x, mean_30 - std_30, mean_30 + std_30, alpha=0.15, color=C_RED)
        
        # ρ=0.75 cutoff
        k = int(0.75 * 196)
        ax.axvline(x=k, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
                   label=f'ρ=0.75 cutoff (block {k})')
        ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        
        ax.set_title(f'{ds_name} — Mean Sorted Importance (10 images)',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Block Index (sorted by importance ↓)', fontsize=11)
        ax.set_ylabel('σ(score + bias)', fontsize=11)
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(0, 197)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(f'{out_dir}/mean_sorted.png', dpi=200)
        plt.savefig(f'{out_dir}/mean_sorted.pdf', dpi=200)
        plt.close()
        print(f"    ✅ Mean sorted figure")
    
    print("\n✅ All importance score plots generated!")


if __name__ == '__main__':
    gen_all_plots()
