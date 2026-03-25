#!/usr/bin/env python3
"""Generate the two professor-requested plots for progress report:

Plot 1: Sigmoid threshold function
  - X-axis: raw_score + b (the pre-sigmoid logit for each block)
  - Y-axis: σ(raw_score + b) (the post-sigmoid importance)
  - Show the sigmoid curve, and overlay scatter points for all 196 blocks
    from 50 images, color-coded. Mark the top-k threshold line.

Plot 2: Raw importance scores sorted by block index
  - X-axis: block index (1 to 196), sorted by importance
  - Y-axis: raw importance score (raw_score + b)
  - 50 images overlaid as semi-transparent lines

Both evaluated at BER=0.0 and BER=0.30.

Output: paper/figures/prof_score_distributions.pdf
"""

import os, sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from train_aid_v2 import ResNet50Front, ResNet50Back
from run_final_pipeline import AIDDataset5050, SpikeAdaptSC_v5c_NA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_STEPS = 8


def load_model():
    """Load SpikeAdapt-SC from v5cna checkpoint (same as multichannel_eval)."""
    front = ResNet50Front(grid_size=14).to(device)
    bb = torch.load('./snapshots_aid_5050_seed42/backbone_best.pth',
                    map_location=device, weights_only=False)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))}, strict=False)
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


def extract_raw_scores(front, model, loader, ber=0.0, n_images=50):
    """Extract pre-sigmoid and post-sigmoid scores for n_images.
    
    Returns:
        raw_scores: list of (196,) arrays — pre-sigmoid logits (raw_score + b)
        post_scores: list of (196,) arrays — post-sigmoid importance σ(raw_score + b)
    """
    raw_scores_list = []
    post_scores_list = []
    count = 0
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            
            # Run encoder
            all_S2, m1, m2 = [], None, None
            for t in range(model.T):
                _, s2, m1, m2 = model.encoder(feat, m1, m2, t=t)
                all_S2.append(s2)
            
            # Extract raw scores from scorer internals
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
            
            for i in range(B):
                if count >= n_images:
                    break
                raw_flat = raw[i].view(-1).cpu().numpy()  # 196 values
                post_flat = post[i].view(-1).cpu().numpy()
                raw_scores_list.append(raw_flat)
                post_scores_list.append(post_flat)
                count += 1
            
            if count >= n_images:
                break
    
    return raw_scores_list, post_scores_list


def plot_both(raw_scores_0, post_scores_0, raw_scores_30, post_scores_30,
              output_path, rho=0.75):
    """Generate the two-plot figure."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    k = int(rho * 196)  # 147 blocks kept
    
    # ===================================================================
    # PLOT 1: Sigmoid threshold function with scatter points
    # ===================================================================
    ax1 = axes[0]
    
    # Draw the sigmoid curve
    x_sigmoid = np.linspace(-8, 8, 500)
    y_sigmoid = 1 / (1 + np.exp(-x_sigmoid))
    ax1.plot(x_sigmoid, y_sigmoid, 'k-', linewidth=2.5, alpha=0.3, label='σ(x)', zorder=1)
    
    # Overlay scatter for 50 images at BER=0.0
    colors_0 = cm.Blues(np.linspace(0.3, 0.9, len(raw_scores_0)))
    for i, (raw, post) in enumerate(zip(raw_scores_0, post_scores_0)):
        ax1.scatter(raw, post, c=[colors_0[i]], s=6, alpha=0.25, zorder=2)
    
    # Mark the threshold: find the average k-th largest post-sigmoid score
    thresholds_0 = []
    for post in post_scores_0:
        sorted_post = np.sort(post)[::-1]
        if k < len(sorted_post):
            thresholds_0.append(sorted_post[k-1])
    avg_threshold = np.mean(thresholds_0)
    # Map threshold back to raw score: raw = log(p/(1-p))
    threshold_raw = np.log(avg_threshold / (1 - avg_threshold + 1e-8))
    
    ax1.axhline(y=avg_threshold, color='red', linestyle='--', linewidth=1.5,
                alpha=0.7, label=f'Top-k threshold (ρ={rho})')
    ax1.axvline(x=threshold_raw, color='red', linestyle=':', linewidth=1.0, alpha=0.5)
    
    # Add BER=0.30 scatter in different color
    colors_30 = cm.Reds(np.linspace(0.3, 0.9, len(raw_scores_30)))
    for i, (raw, post) in enumerate(zip(raw_scores_30, post_scores_30)):
        ax1.scatter(raw, post, c=[colors_30[i]], s=6, alpha=0.25, marker='x', zorder=3)
    
    ax1.set_xlabel('Raw Score + Bias  (pre-sigmoid logit)', fontsize=13)
    ax1.set_ylabel('σ(raw score + b)  (importance)', fontsize=13)
    ax1.set_title('(a) Importance Scores on Sigmoid Curve\n50 images, 196 blocks each',
                  fontsize=13, fontweight='bold')
    
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='k', linewidth=2, alpha=0.3, label='σ(x)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
               markersize=8, label='BER=0.0 (50 images)'),
        Line2D([0], [0], marker='x', color='w', markeredgecolor='firebrick',
               markersize=8, markeredgewidth=2, label='BER=0.30 (50 images)'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=1.5,
               label=f'Top-k threshold (ρ={rho})')
    ]
    ax1.legend(handles=legend_elements, fontsize=10, loc='upper left')
    ax1.set_xlim(-8, 8)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.2)
    
    # Shade keep/drop regions
    ax1.axhspan(avg_threshold, 1.05, alpha=0.05, color='green', label='Keep')
    ax1.axhspan(-0.05, avg_threshold, alpha=0.05, color='red', label='Drop')
    ax1.text(6, avg_threshold + 0.04, 'KEEP', fontsize=10, color='green',
             fontweight='bold', ha='center')
    ax1.text(6, avg_threshold - 0.06, 'DROP', fontsize=10, color='red',
             fontweight='bold', ha='center')
    
    # ===================================================================
    # PLOT 2: Sorted raw importance scores for 196 blocks, 50 images
    # ===================================================================
    ax2 = axes[1]
    
    # Plot sorted raw scores for BER=0.0
    for i, raw in enumerate(raw_scores_0):
        sorted_raw = np.sort(raw)[::-1]  # descending
        ax2.plot(range(1, 197), sorted_raw, color=colors_0[i],
                 alpha=0.35, linewidth=0.8)
    
    # Plot sorted raw scores for BER=0.30
    for i, raw in enumerate(raw_scores_30):
        sorted_raw = np.sort(raw)[::-1]
        ax2.plot(range(1, 197), sorted_raw, color=colors_30[i],
                 alpha=0.35, linewidth=0.8, linestyle='--')
    
    # Plot mean ± std for BER=0.0
    all_sorted_0 = np.array([np.sort(r)[::-1] for r in raw_scores_0])
    mean_0 = all_sorted_0.mean(axis=0)
    std_0 = all_sorted_0.std(axis=0)
    ax2.plot(range(1, 197), mean_0, 'b-', linewidth=2.5, label='Mean (BER=0.0)')
    ax2.fill_between(range(1, 197), mean_0 - std_0, mean_0 + std_0,
                     alpha=0.15, color='blue')
    
    # Plot mean ± std for BER=0.30
    all_sorted_30 = np.array([np.sort(r)[::-1] for r in raw_scores_30])
    mean_30 = all_sorted_30.mean(axis=0)
    std_30 = all_sorted_30.std(axis=0)
    ax2.plot(range(1, 197), mean_30, 'r-', linewidth=2.5, label='Mean (BER=0.30)')
    ax2.fill_between(range(1, 197), mean_30 - std_30, mean_30 + std_30,
                     alpha=0.15, color='red')
    
    # Mark the ρ=0.75 cutoff
    ax2.axvline(x=k, color='green', linestyle='--', linewidth=2, alpha=0.7,
                label=f'ρ={rho} cutoff (block {k})')
    ax2.text(k + 2, ax2.get_ylim()[1] * 0.9, f'←KEEP | DROP→',
             fontsize=10, color='green', fontweight='bold')
    
    ax2.set_xlabel('Block Index (sorted by importance, descending)', fontsize=13)
    ax2.set_ylabel('Raw Score + Bias  (pre-sigmoid logit)', fontsize=13)
    ax2.set_title('(b) Per-Block Raw Importance Scores\n50 images overlaid, sorted descending',
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    print(f"  Saved: {output_path.replace('.pdf', '.png')}")
    plt.close()


def main():
    print("=" * 70)
    print("  PROFESSOR'S SCORE DISTRIBUTION PLOTS — 50 IMAGES")
    print("=" * 70)
    
    front, back, model = load_model()
    
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])
    test_ds = AIDDataset5050('./data', tf_test, 'test', seed=42)
    
    # Use first 50 images
    subset = Subset(test_ds, list(range(50)))
    loader = DataLoader(subset, batch_size=10, shuffle=False, num_workers=4,
                        pin_memory=True)
    
    print("\n  Extracting scores at BER=0.0...")
    raw_0, post_0 = extract_raw_scores(front, model, loader, ber=0.0, n_images=50)
    print(f"    Got {len(raw_0)} images, {len(raw_0[0])} blocks each")
    
    print("  Extracting scores at BER=0.30...")
    raw_30, post_30 = extract_raw_scores(front, model, loader, ber=0.30, n_images=50)
    print(f"    Got {len(raw_30)} images, {len(raw_30[0])} blocks each")
    
    # Print some statistics
    all_raw_0 = np.concatenate(raw_0)
    all_raw_30 = np.concatenate(raw_30)
    print(f"\n  Raw score statistics:")
    print(f"    BER=0.0:  mean={all_raw_0.mean():.3f}, std={all_raw_0.std():.3f}, "
          f"range=[{all_raw_0.min():.3f}, {all_raw_0.max():.3f}]")
    print(f"    BER=0.30: mean={all_raw_30.mean():.3f}, std={all_raw_30.std():.3f}, "
          f"range=[{all_raw_30.min():.3f}, {all_raw_30.max():.3f}]")
    
    os.makedirs('paper/figures', exist_ok=True)
    output_path = 'paper/figures/prof_score_distributions.pdf'
    
    print(f"\n  Generating plots...")
    plot_both(raw_0, post_0, raw_30, post_30, output_path)
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()
