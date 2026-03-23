"""Generate visualization figures for V5C-NA on BOTH AID and RESISC45.

Fig2: Mask diversity — 8 scenes with 14×14 mask overlay  (AID + RESISC45)
Fig3: Feature MSE vs BER — dual panel AID + RESISC45
Fig4: Rate sweep — single scene across ρ values (AID + RESISC45)

All figures use the CORRECT 14×14 grid / C2=36 / V5C-NA architecture.
"""

import os, sys, torch, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from train_aid_v2 import ResNet50Front, ResNet50Back, BSC_Channel
from run_final_pipeline import AIDDataset5050, RESISC45Dataset, SpikeAdaptSC_v5c_NA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_STEPS = 8
OUT_DIR = 'paper/figures'

# IEEE styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


def load_model(ds_name, n_classes, bb_dir, v5c_dir):
    """Load backbone + V5C-NA model for a dataset."""
    front = ResNet50Front(grid_size=14).to(device)
    bb = torch.load(f"./{bb_dir}/backbone_best.pth", map_location=device)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))}, strict=False)
    front.eval()
    
    back = ResNet50Back(n_classes).to(device)
    model = SpikeAdaptSC_v5c_NA(C_in=1024, C1=256, C2=36, T=T_STEPS,
                                 target_rate=0.75, grid_size=14).to(device)
    
    ck_files = sorted([f for f in os.listdir(f"./{v5c_dir}") if f.startswith('v5cna_best')])
    best_ck = f"./{v5c_dir}/{ck_files[-1]}"
    ck = torch.load(best_ck, map_location=device)
    model.load_state_dict(ck['model'])
    back.load_state_dict(ck['back'])
    
    model.eval(); back.eval()
    return front, model, back


def get_test_dataset(ds_name, with_raw=False):
    """Load test dataset."""
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])
    if ds_name == 'aid':
        ds = AIDDataset5050("./data", tf_test, 'test', seed=42)
    else:
        ds = RESISC45Dataset("./data", tf_test, 'test', train_ratio=0.20, seed=42)
    return ds


def get_raw_image(ds, idx):
    """Get original image (no normalization) for visualization."""
    path, label = ds.samples[idx]
    img = Image.open(path).convert('RGB')
    img = T.Resize(256)(img)
    img = T.CenterCrop(224)(img)
    return np.array(img), label


# ===================================================================
# Fig 2: Mask Diversity — 8 scenes with 14×14 mask overlay
# ===================================================================
def fig2_mask_diversity(ds_name, front, model, ds, class_names, n_scenes=8):
    """Generate mask diversity figure for a dataset."""
    # Find one sample per class for diversity
    class_to_idx = {}
    for i, (_, label) in enumerate(ds.samples):
        if label not in class_to_idx:
            class_to_idx[label] = i
        if len(class_to_idx) >= n_scenes:
            break
    
    # Pick diverse classes
    if ds_name == 'aid':
        target_classes = ['Airport', 'Beach', 'Bridge', 'Desert', 'Farmland', 'Mountain', 'Stadium', 'Forest']
    else:
        target_classes = ['airplane', 'beach', 'bridge', 'desert', 'farmland', 'mountain', 'stadium', 'forest']
    
    indices = []
    for cls_name in target_classes:
        found = False
        for i, (path, label) in enumerate(ds.samples):
            lbl_name = ds.idx_to_class[label] if hasattr(ds, 'idx_to_class') else str(label)
            if lbl_name.lower() == cls_name.lower():
                indices.append(i)
                found = True
                break
        if not found:
            # Fallback: use any class
            for lbl, idx in class_to_idx.items():
                if idx not in indices:
                    indices.append(idx)
                    break
    
    indices = indices[:n_scenes]
    
    fig, axes = plt.subplots(2, n_scenes, figsize=(7.0, 2.0),
                              gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05, 'wspace': 0.05})
    
    for col, idx in enumerate(indices):
        raw_img, label = get_raw_image(ds, idx)
        cls_name = ds.idx_to_class[label] if hasattr(ds, 'idx_to_class') else str(label)
        
        # Get mask
        img_tensor = ds[idx][0].unsqueeze(0).to(device)
        with torch.no_grad():
            feat = front(img_tensor)
            Fp, stats = model(feat, noise_param=0.0)
        
        mask = stats['mask'][0].squeeze().cpu().numpy()  # (14, 14)
        
        # Top row: image with mask overlay
        ax = axes[0, col]
        ax.imshow(raw_img)
        
        # Create red overlay for dropped blocks
        H, W = raw_img.shape[:2]
        overlay = np.zeros((H, W, 4))
        bh, bw = H // 14, W // 14
        for i in range(14):
            for j in range(14):
                if mask[i, j] < 0.5:  # Dropped
                    overlay[i*bh:(i+1)*bh, j*bw:(j+1)*bw] = [1, 0, 0, 0.45]
        ax.imshow(overlay)
        
        # Count transmitted blocks
        n_kept = int(mask.sum())
        ax.set_title(f'{cls_name}', fontsize=7, fontweight='bold')
        ax.axis('off')
        
        # Bottom row: 14×14 mask grid
        ax2 = axes[1, col]
        mask_colored = np.zeros((14, 14, 3))
        mask_colored[mask > 0.5] = [0.2, 0.7, 0.2]  # Green = transmitted
        mask_colored[mask <= 0.5] = [0.8, 0.15, 0.15]  # Red = dropped
        ax2.imshow(mask_colored, interpolation='nearest')
        ax2.set_title(f'{n_kept}/196', fontsize=6)
        ax2.axis('off')
    
    plt.suptitle(f'{ds_name.upper()} — V5C-NA Mask Diversity (ρ=0.75, 14×14 grid)',
                 fontsize=9, fontweight='bold', y=1.02)
    
    suffix = 'aid' if ds_name == 'aid' else 'resisc45'
    plt.savefig(f'{OUT_DIR}/fig2_mask_diversity_{suffix}.png', bbox_inches='tight')
    plt.savefig(f'{OUT_DIR}/fig2_mask_diversity_{suffix}.pdf', bbox_inches='tight')
    plt.close()
    print(f"✅ Fig2 mask diversity ({ds_name}) saved")


# ===================================================================
# Fig 3: Feature MSE vs BER — dual panel
# ===================================================================
def fig3_feature_mse_dual(front_aid, model_aid, back_aid, ds_aid,
                           front_res, model_res, back_res, ds_res):
    """Generate dual-panel feature MSE vs BER figure."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    
    ber_values = np.arange(0, 0.36, 0.025)
    
    for idx, (ds_name, front, model, back, ds, n_classes) in enumerate([
        ('AID', front_aid, model_aid, back_aid, ds_aid, 30),
        ('RESISC45', front_res, model_res, back_res, ds_res, 45),
    ]):
        ax = axes[idx]
        
        # Sample 200 images
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)
        
        mse_list = []
        conf_list = []
        
        for ber in ber_values:
            batch_mses = []
            batch_confs = []
            for batch_i, (imgs, labels) in enumerate(loader):
                if batch_i >= 7:  # ~200 images
                    break
                imgs = imgs.to(device)
                with torch.no_grad():
                    feat = front(imgs)
                    # Clean
                    Fp_clean, _ = model(feat, noise_param=0.0)
                    # Noisy
                    Fp_noisy, _ = model(feat, noise_param=float(ber))
                    
                    mse = ((Fp_clean - Fp_noisy) ** 2).mean().item()
                    logits = back(Fp_noisy)
                    conf = torch.softmax(logits, dim=1).max(dim=1)[0].mean().item()
                    
                    batch_mses.append(mse)
                    batch_confs.append(conf * 100)
            
            mse_list.append(np.mean(batch_mses))
            conf_list.append(np.mean(batch_confs))
        
        # Dual y-axis
        color_mse = '#D32F2F'
        color_conf = '#388E3C'
        
        ax.plot(ber_values, mse_list, 'o-', color=color_mse, markersize=4, linewidth=1.5, label='Feature MSE')
        ax.set_xlabel('BER')
        ax.set_ylabel('Feature MSE', color=color_mse)
        ax.tick_params(axis='y', labelcolor=color_mse)
        
        ax2 = ax.twinx()
        ax2.plot(ber_values, conf_list, 's-', color=color_conf, markersize=4, linewidth=1.5, label='Confidence')
        ax2.set_ylabel('Confidence (%)', color=color_conf)
        ax2.tick_params(axis='y', labelcolor=color_conf)
        ax2.set_ylim(50, 102)
        
        ax.set_title(ds_name, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='center left', fontsize=7,
                  frameon=True, fancybox=False, edgecolor='#ccc')
    
    plt.tight_layout(w_pad=3.5)
    plt.savefig(f'{OUT_DIR}/fig3_feature_mse_dual.png')
    plt.savefig(f'{OUT_DIR}/fig3_feature_mse_dual.pdf')
    plt.close()
    print("✅ Fig3 feature MSE (dual panel) saved")


# ===================================================================
# Fig 4: Rate Sweep — single scene across ρ values
# ===================================================================
def fig4_rate_sweep(ds_name, front, model, back_model, ds):
    """Rate sweep visualization for a single scene."""
    rho_values = [1.0, 0.875, 0.75, 0.50, 0.25]
    n_panels = len(rho_values)
    
    # Pick first Airport/airplane scene
    target = 'Airport' if ds_name == 'aid' else 'airplane'
    idx = 0
    for i, (path, label) in enumerate(ds.samples):
        lbl_name = ds.idx_to_class[label] if hasattr(ds, 'idx_to_class') else str(label)
        if lbl_name.lower() == target.lower():
            idx = i
            break
    
    raw_img, label = get_raw_image(ds, idx)
    cls_name = ds.idx_to_class[label] if hasattr(ds, 'idx_to_class') else str(label)
    img_tensor = ds[idx][0].unsqueeze(0).to(device)
    
    fig, axes = plt.subplots(1, n_panels, figsize=(7.0, 1.6))
    
    # Save original target rate and restore after
    orig_rate = model.block_mask.target_rate
    
    for col, rho in enumerate(rho_values):
        ax = axes[col]
        
        model.block_mask.target_rate = rho
        with torch.no_grad():
            feat = front(img_tensor)
            Fp, stats = model(feat, noise_param=0.0)
            logits = back_model(Fp)
            pred = logits.argmax(1).item()
            conf = torch.softmax(logits, dim=1).max().item() * 100
        
        mask = stats['mask'][0].squeeze().cpu().numpy()
        n_kept = int(mask.sum())
        
        ax.imshow(raw_img)
        
        # Red overlay for dropped blocks
        H, W = raw_img.shape[:2]
        overlay = np.zeros((H, W, 4))
        bh, bw = H // 14, W // 14
        for i in range(14):
            for j in range(14):
                if mask[i, j] < 0.5:
                    overlay[i*bh:(i+1)*bh, j*bw:(j+1)*bw] = [1, 0, 0, 0.45]
        ax.imshow(overlay)
        
        pred_name = ds.idx_to_class[pred] if hasattr(ds, 'idx_to_class') else str(pred)
        color = '#388E3C' if pred == label else '#D32F2F'
        
        ax.set_title(f'ρ={rho} ({n_kept}/196)', fontsize=7, fontweight='bold')
        ax.text(0.5, -0.02, f'{pred_name} ({conf:.0f}%)', transform=ax.transAxes, 
                fontsize=6, ha='center', va='top', color=color, fontweight='bold')
        ax.axis('off')
    
    model.block_mask.target_rate = orig_rate
    
    suffix = 'aid' if ds_name == 'aid' else 'resisc45'
    plt.suptitle(f'{ds_name.upper()} — Rate Sweep ({cls_name})', fontsize=9, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig4_rate_sweep_{suffix}.png', bbox_inches='tight')
    plt.savefig(f'{OUT_DIR}/fig4_rate_sweep_{suffix}.pdf', bbox_inches='tight')
    plt.close()
    print(f"✅ Fig4 rate sweep ({ds_name}) saved")


if __name__ == '__main__':
    print("Loading models...")
    
    # AID
    front_aid, model_aid, back_aid = load_model(
        'aid', 30, 'snapshots_aid_5050_seed42', 'snapshots_aid_v5cna_seed42')
    ds_aid = get_test_dataset('aid')
    
    # RESISC45
    front_res, model_res, back_res = load_model(
        'resisc45', 45, 'snapshots_resisc45_5050_seed42', 'snapshots_resisc45_v5cna_seed42')
    ds_res = get_test_dataset('resisc45')
    
    # Store globally for rate sweep
    global back_res_global
    back_res_global = back_res
    
    print("\nGenerating Fig2: Mask Diversity...")
    fig2_mask_diversity('aid', front_aid, model_aid, ds_aid, 
                        [ds_aid.idx_to_class[i] for i in range(30)] if hasattr(ds_aid, 'idx_to_class') else [])
    fig2_mask_diversity('resisc45', front_res, model_res, ds_res,
                        [ds_res.idx_to_class[i] for i in range(45)] if hasattr(ds_res, 'idx_to_class') else [])
    
    print("\nGenerating Fig3: Feature MSE...")
    fig3_feature_mse_dual(front_aid, model_aid, back_aid, ds_aid,
                           front_res, model_res, back_res, ds_res)
    
    print("\nGenerating Fig4: Rate Sweep...")
    fig4_rate_sweep('aid', front_aid, model_aid, back_aid, ds_aid)
    
    # Need back for RESISC45 rate sweep
    fig4_rate_sweep('resisc45', front_res, model_res, back_res, ds_res)
    
    # Also generate combined mask diversity (both datasets, 4+4)
    print("\n✅ All visualization figures generated!")
