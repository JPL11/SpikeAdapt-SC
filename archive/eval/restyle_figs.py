"""Restyle paper figures: add 14×14 grid lines to mask overlays.

Regenerates fig2 (mask diversity) and fig4 (rate sweep) with:
  - Clear grid lines on mask overlays (like the old 8×8 style)
  - Better mask color scheme (semi-transparent red for dropped, green borders for kept)
  - Gridded mask panels in bottom row

Also regenerates the architecture diagram with an actual AID Airport image.
"""

import os, sys, torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from train_aid_v2 import ResNet50Front, ResNet50Back
from run_final_pipeline import AIDDataset5050, RESISC45Dataset, SpikeAdaptSC_v5c_NA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_STEPS = 8
OUT_DIR = 'paper/figures'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


def load_model(n_classes, bb_dir, v5c_dir):
    front = ResNet50Front(grid_size=14).to(device)
    bb = torch.load(f"./{bb_dir}/backbone_best.pth", map_location=device)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))}, strict=False)
    front.eval()
    back = ResNet50Back(n_classes).to(device)
    model = SpikeAdaptSC_v5c_NA(C_in=1024, C1=256, C2=36, T=T_STEPS,
                                 target_rate=0.75, grid_size=14).to(device)
    ck_files = sorted([f for f in os.listdir(f"./{v5c_dir}") if f.startswith('v5cna_best')])
    ck = torch.load(f"./{v5c_dir}/{ck_files[-1]}", map_location=device)
    model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
    model.eval(); back.eval()
    return front, model, back


def get_raw_image(ds, idx):
    path, label = ds.samples[idx]
    img = Image.open(path).convert('RGB')
    img = T.Resize(256)(img)
    img = T.CenterCrop(224)(img)
    return np.array(img), label


def draw_grid_overlay(ax, raw_img, mask, grid_size=14, alpha_drop=0.5):
    """Draw image with grid overlay: red blocks = dropped, clear grid lines."""
    H, W = raw_img.shape[:2]
    ax.imshow(raw_img)
    
    bh, bw = H / grid_size, W / grid_size
    
    # Draw semi-transparent red over dropped blocks
    for i in range(grid_size):
        for j in range(grid_size):
            if mask[i, j] < 0.5:
                rect = patches.Rectangle(
                    (j * bw, i * bh), bw, bh,
                    facecolor='red', alpha=alpha_drop, edgecolor='none'
                )
                ax.add_patch(rect)
    
    # Draw grid lines
    for i in range(grid_size + 1):
        ax.axhline(y=i * bh, color='white', linewidth=0.3, alpha=0.6)
        ax.axvline(x=i * bw, color='white', linewidth=0.3, alpha=0.6)
    
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')


def draw_mask_grid(ax, mask, grid_size=14):
    """Draw the 14×14 binary mask with grid lines and colored cells."""
    mask_rgb = np.zeros((grid_size, grid_size, 3))
    mask_rgb[mask > 0.5] = [0.2, 0.75, 0.2]   # Green = transmitted
    mask_rgb[mask <= 0.5] = [0.85, 0.15, 0.15]  # Red = dropped
    
    ax.imshow(mask_rgb, interpolation='nearest', extent=[0, grid_size, grid_size, 0])
    
    # Draw grid lines
    for i in range(grid_size + 1):
        ax.axhline(y=i, color='white', linewidth=0.4)
        ax.axvline(x=i, color='white', linewidth=0.4)
    
    ax.set_xlim(0, grid_size)
    ax.set_ylim(grid_size, 0)
    ax.axis('off')


def get_mask(front, model, ds, idx):
    """Get mask for a specific sample."""
    img_tensor = ds[idx][0].unsqueeze(0).to(device)
    with torch.no_grad():
        feat = front(img_tensor)
        Fp, stats = model(feat, noise_param=0.0)
    return stats['mask'][0].squeeze().cpu().numpy()


def find_class_idx(ds, class_name):
    """Find index of first sample from a specific class."""
    for i, (path, label) in enumerate(ds.samples):
        lbl_name = ds.idx_to_class[label] if hasattr(ds, 'idx_to_class') else str(label)
        if lbl_name.lower() == class_name.lower():
            return i
    return 0


# ===================================================================
# Fig 2: Mask Diversity with grid lines
# ===================================================================
def fig2_mask_diversity_gridded(ds_name, front, model, ds, n_scenes=8):
    if ds_name == 'aid':
        classes = ['Airport', 'Beach', 'Bridge', 'Desert', 'Farmland', 'Mountain', 'Stadium', 'Forest']
    else:
        classes = ['airplane', 'beach', 'bridge', 'desert', 'farmland', 'mountain', 'stadium', 'forest']
    
    indices = [find_class_idx(ds, c) for c in classes]
    
    fig, axes = plt.subplots(2, n_scenes, figsize=(7.0, 2.4),
                              gridspec_kw={'height_ratios': [3, 1.2], 'hspace': 0.08, 'wspace': 0.06})
    
    for col, idx in enumerate(indices):
        raw_img, label = get_raw_image(ds, idx)
        cls_name = ds.idx_to_class[label] if hasattr(ds, 'idx_to_class') else str(label)
        mask = get_mask(front, model, ds, idx)
        n_kept = int(mask.sum())
        
        # Top: image + grid overlay
        draw_grid_overlay(axes[0, col], raw_img, mask)
        axes[0, col].set_title(cls_name, fontsize=7, fontweight='bold', pad=2)
        
        # Bottom: 14×14 mask grid
        draw_mask_grid(axes[1, col], mask)
        axes[1, col].set_title(f'{n_kept}/196', fontsize=6, pad=1)
    
    # Row labels
    axes[0, 0].text(-0.12, 0.5, 'Image\n+ mask', transform=axes[0, 0].transAxes,
                     fontsize=7, va='center', ha='center', rotation=90, fontweight='bold')
    axes[1, 0].text(-0.12, 0.5, '14×14\nmask', transform=axes[1, 0].transAxes,
                     fontsize=7, va='center', ha='center', rotation=90, fontweight='bold')
    
    suffix = ds_name
    plt.savefig(f'{OUT_DIR}/fig2_mask_diversity_{suffix}.png', bbox_inches='tight')
    plt.savefig(f'{OUT_DIR}/fig2_mask_diversity_{suffix}.pdf', bbox_inches='tight')
    plt.close()
    print(f"✅ Fig2 mask diversity ({ds_name}) with grid lines saved")


# ===================================================================
# Fig 4: Rate Sweep with grid lines
# ===================================================================
def fig4_rate_sweep_gridded(ds_name, front, model, back, ds):
    rho_values = [1.0, 0.875, 0.75, 0.50, 0.25]
    
    target = 'Airport' if ds_name == 'aid' else 'airplane'
    idx = find_class_idx(ds, target)
    raw_img, label = get_raw_image(ds, idx)
    cls_name = ds.idx_to_class[label] if hasattr(ds, 'idx_to_class') else str(label)
    img_tensor = ds[idx][0].unsqueeze(0).to(device)
    
    fig, axes = plt.subplots(1, len(rho_values), figsize=(7.0, 1.8))
    
    orig_rate = model.block_mask.target_rate
    
    for col, rho in enumerate(rho_values):
        ax = axes[col]
        model.block_mask.target_rate = rho
        with torch.no_grad():
            feat = front(img_tensor)
            Fp, stats = model(feat, noise_param=0.0)
            logits = back(Fp)
            pred = logits.argmax(1).item()
            conf = torch.softmax(logits, dim=1).max().item() * 100
        
        mask = stats['mask'][0].squeeze().cpu().numpy()
        n_kept = int(mask.sum())
        
        draw_grid_overlay(ax, raw_img, mask)
        
        pred_name = ds.idx_to_class[pred] if hasattr(ds, 'idx_to_class') else str(pred)
        color = '#388E3C' if pred == label else '#D32F2F'
        
        ax.set_title(f'ρ={rho}  ({n_kept}/196)', fontsize=7, fontweight='bold', pad=2)
        ax.text(0.5, -0.04, f'{pred_name} ({conf:.0f}%)',
                transform=ax.transAxes, fontsize=6, ha='center', va='top',
                color=color, fontweight='bold')
    
    model.block_mask.target_rate = orig_rate
    
    suffix = ds_name
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig4_rate_sweep_{suffix}.png', bbox_inches='tight')
    plt.savefig(f'{OUT_DIR}/fig4_rate_sweep_{suffix}.pdf', bbox_inches='tight')
    plt.close()
    print(f"✅ Fig4 rate sweep ({ds_name}) with grid lines saved")


if __name__ == '__main__':
    print("Loading models...")
    
    front_aid, model_aid, back_aid = load_model(
        30, 'snapshots_aid_5050_seed42', 'snapshots_aid_v5cna_seed42')
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])
    ds_aid = AIDDataset5050("./data", tf_test, 'test', seed=42)
    
    front_res, model_res, back_res = load_model(
        45, 'snapshots_resisc45_5050_seed42', 'snapshots_resisc45_v5cna_seed42')
    ds_res = RESISC45Dataset("./data", tf_test, 'test', train_ratio=0.20, seed=42)
    
    print("\nFig2: Mask Diversity (with grid lines)...")
    fig2_mask_diversity_gridded('aid', front_aid, model_aid, ds_aid)
    fig2_mask_diversity_gridded('resisc45', front_res, model_res, ds_res)
    
    print("\nFig4: Rate Sweep (with grid lines)...")
    fig4_rate_sweep_gridded('aid', front_aid, model_aid, back_aid, ds_aid)
    fig4_rate_sweep_gridded('resisc45', front_res, model_res, back_res, ds_res)
    
    # Export a clean Airport image for the architecture diagram
    print("\nExporting AID Airport image for architecture diagram...")
    idx = find_class_idx(ds_aid, 'Airport')
    raw_img, label = get_raw_image(ds_aid, idx)
    Image.fromarray(raw_img).save(f'{OUT_DIR}/sample_airport.png')
    
    # Also get a RESISC45 sample
    idx_r = find_class_idx(ds_res, 'airplane')
    raw_r, _ = get_raw_image(ds_res, idx_r)
    Image.fromarray(raw_r).save(f'{OUT_DIR}/sample_airplane_resisc.png')
    
    print("\n✅ All restyled figures generated!")
