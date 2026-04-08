#!/usr/bin/env python3
"""Train SpikeAdapt-SC Image Codec V3: Stereo + Refinement + Temporal Attention.

Builds on V2 with three enhancements:
  1. Stereo cross-attention (exploit left-right correlation)
  2. Decoder refinement net (residual post-processing)
  3. Temporal attention (adaptive timestep aggregation in SNN decoder)

4-stage training:
  Stage 1: Autoencoder (no masking, no noise, stereo pairs)  - 60 epochs
  Stage 2: Add channel noise (AWGN + BSC)                     - 40 epochs
  Stage 3: Scorer training (frozen backbone)                  - 40 epochs
  Stage 4: Joint fine-tuning (all params incl stereo+refine) - 15 epochs

Usage:  python train/train_kitti_image_v3.py [--stage 0|1|2|3|4] [--eval-only]
"""

import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
import sys, os, json, random, math, time, glob
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms as T
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from models.spike_image_codec_v2 import SpikeImageCodecV2

# Pure-Python SSIM (no JIT)
def compute_ssim_torch(x, y, window_size=11, data_range=1.0):
    C = x.size(1)
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=x.device) - window_size // 2
    g = torch.exp(-coords**2 / (2 * sigma**2))
    g = g / g.sum()
    window = g.unsqueeze(1) * g.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0).expand(C, 1, window_size, window_size)
    pad = window_size // 2
    mu_x = F.conv2d(x, window, padding=pad, groups=C)
    mu_y = F.conv2d(y, window, padding=pad, groups=C)
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y
    sigma_x_sq = F.conv2d(x * x, window, padding=pad, groups=C) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=pad, groups=C) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=C) - mu_xy
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
    return ssim_map.mean()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CROP_H, CROP_W = 256, 512


# =============================================================================
# Stereo Pair Dataset
# =============================================================================

class StereoPairDataset(Dataset):
    """InStereo2K stereo pair dataset — returns (left, right) images."""
    def __init__(self, pair_dirs, transform=None):
        self.pair_dirs = pair_dirs  # list of directories with left.png, right.png
        self.transform = transform
        # Use a fixed random seed per-pair for synchronized crops
    
    def __len__(self):
        return len(self.pair_dirs)
    
    def __getitem__(self, idx):
        d = self.pair_dirs[idx]
        left = Image.open(os.path.join(d, 'left.png')).convert('RGB')
        right = Image.open(os.path.join(d, 'right.png')).convert('RGB')
        
        if self.transform:
            # Synchronized augmentation: same crop/flip for both views
            seed = torch.randint(0, 2**31, (1,)).item()
            random.seed(seed); torch.manual_seed(seed)
            left = self.transform(left)
            random.seed(seed); torch.manual_seed(seed)
            right = self.transform(right)
        
        return left, right


class SingleImageDataset(Dataset):
    """For KITTI single-image eval (no stereo pair)."""
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


def get_datasets(instereo_root='./InStereo2K', kitti_root='./data'):
    tf_train = T.Compose([
        T.RandomCrop((CROP_H, CROP_W), pad_if_needed=True, padding_mode='reflect'),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
    ])
    tf_test = T.Compose([
        T.CenterCrop((CROP_H, CROP_W)),
        T.ToTensor(),
    ])
    
    # InStereo2K stereo pairs
    instereo_train = os.path.join(instereo_root, 'train')
    instereo_test = os.path.join(instereo_root, 'test')
    
    train_dirs = sorted([
        os.path.join(instereo_train, d) for d in os.listdir(instereo_train)
        if os.path.isfile(os.path.join(instereo_train, d, 'left.png'))
           and os.path.isfile(os.path.join(instereo_train, d, 'right.png'))
    ]) if os.path.exists(instereo_train) else []
    
    test_dirs = sorted([
        os.path.join(instereo_test, d) for d in os.listdir(instereo_test)
        if os.path.isfile(os.path.join(instereo_test, d, 'left.png'))
           and os.path.isfile(os.path.join(instereo_test, d, 'right.png'))
    ]) if os.path.exists(instereo_test) else []
    
    train_ds = StereoPairDataset(train_dirs, tf_train)
    test_ds = StereoPairDataset(test_dirs, tf_test)
    
    print(f"  InStereo2K: {len(train_dirs)} train pairs, {len(test_dirs)} test pairs")
    
    # KITTI single-image eval
    kitti_test_ds = None
    kitti_dir = os.path.join(kitti_root, 'kitti_stereo_2015', 'training', 'image_2')
    if os.path.exists(kitti_dir):
        kitti_imgs = sorted(glob.glob(os.path.join(kitti_dir, '*.png')))
        if kitti_imgs:
            kitti_test_ds = SingleImageDataset(kitti_imgs, tf_test)
            print(f"  KITTI Stereo 2015: {len(kitti_imgs)} images (eval only)")
    
    return train_ds, test_ds, kitti_test_ds


# =============================================================================
# Loss
# =============================================================================

def compute_ms_ssim(x, y, weights=[0.4, 0.3, 0.3]):
    ms_ssim = 0
    for i, w in enumerate(weights):
        ms_ssim += w * compute_ssim_torch(x, y)
        if i < len(weights) - 1:
            x = F.avg_pool2d(x, 2)
            y = F.avg_pool2d(y, 2)
    return ms_ssim

def reconstruction_loss(x, x_hat):
    """MSE + MS-SSIM loss (same as V2-base production)."""
    x_hat_clamped = x_hat.clamp(0, 1)
    mse = F.mse_loss(x_hat_clamped, x)
    ssim_val = compute_ssim_torch(x, x_hat_clamped)
    ms_ssim_val = compute_ms_ssim(x, x_hat_clamped)
    loss = 0.85 * mse + 0.15 * (1 - ms_ssim_val)
    return loss, mse, ssim_val


# =============================================================================
# Evaluation
# =============================================================================

def compute_psnr(x, x_hat):
    mse = F.mse_loss(x_hat, x).item()
    if mse == 0:
        return 100.0
    return 10 * math.log10(1.0 / mse)


def evaluate(model, loader, channel='awgn', noise_param=0.0,
             use_masking=True, target_cbr=None, max_batches=None,
             stereo=True):
    """Evaluate model — handles both stereo pair and single-image loaders."""
    model.eval()
    psnr_sum, ssim_sum, cbr_sum, n = 0, 0, 0, 0
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            
            # Handle stereo pair vs single image
            if isinstance(batch, (list, tuple)) and len(batch) == 2 and stereo:
                imgs_l, imgs_r = batch[0].to(device), batch[1].to(device)
            elif isinstance(batch, (list, tuple)):
                imgs_l = batch[0].to(device)
                imgs_r = None
            else:
                imgs_l = batch.to(device)
                imgs_r = None
            
            recon, info = model(imgs_l, noise_param=noise_param, channel=channel,
                               use_masking=use_masking, target_cbr_override=target_cbr,
                               img_right=imgs_r)
            recon = recon.clamp(0, 1)
            
            psnr_sum += compute_psnr(imgs_l, recon) * imgs_l.size(0)
            cbr_sum += info['actual_cbr'] * imgs_l.size(0)
            ssim_val = compute_ssim_torch(imgs_l, recon).item()
            ssim_sum += ssim_val * imgs_l.size(0)
            n += imgs_l.size(0)
    
    return {
        'psnr': round(psnr_sum / n, 3),
        'ms_ssim': round(ssim_sum / n, 5),
        'cbr': round(cbr_sum / n, 4),
        'n_images': n,
    }


def evaluate_random_mask(model, loader, channel, noise_param, target_cbr, 
                         max_batches=None, stereo=True):
    """Evaluate with random spatial masking (baseline for gap computation)."""
    model.eval()
    psnr_sum, ssim_sum, n = 0, 0, 0
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            
            if isinstance(batch, (list, tuple)) and len(batch) == 2 and stereo:
                imgs_l, imgs_r = batch[0].to(device), batch[1].to(device)
            elif isinstance(batch, (list, tuple)):
                imgs_l = batch[0].to(device)
                imgs_r = None
            else:
                imgs_l = batch.to(device)
                imgs_r = None
            
            # Encode left view
            feat_l = model.img_encoder(imgs_l)
            spikes_l = model.snn_encoder(feat_l)
            
            B, C, H, W = spikes_l[0].shape
            k = max(1, int(target_cbr * H * W))
            
            # Random spatial mask
            flat = torch.rand(B, 1, H, W, device=device).view(B, -1)
            _, idx = flat.topk(k, dim=1)
            mask = torch.zeros_like(flat)
            mask.scatter_(1, idx, 1.0)
            mask = mask.view(B, 1, H, W)
            masked_l = [s * mask for s in spikes_l]
            
            if channel == 'awgn':
                recv_l = [model.awgn(s, noise_param) for s in masked_l]
            else:
                recv_l = [model.bsc(s, noise_param) for s in masked_l]
            
            feat_recon_l = model.snn_decoder(recv_l)
            
            # Stereo fusion (if applicable)
            if model.stereo and imgs_r is not None:
                feat_r = model.img_encoder(imgs_r)
                spikes_r = model.snn_encoder(feat_r)
                # Random mask for right view too
                flat_r = torch.rand(B, 1, H, W, device=device).view(B, -1)
                _, idx_r = flat_r.topk(k, dim=1)
                mask_r = torch.zeros_like(flat_r)
                mask_r.scatter_(1, idx_r, 1.0)
                mask_r = mask_r.view(B, 1, H, W)
                masked_r = [s * mask_r for s in spikes_r]
                if channel == 'awgn':
                    recv_r = [model.awgn(s, noise_param) for s in masked_r]
                else:
                    recv_r = [model.bsc(s, noise_param) for s in masked_r]
                feat_recon_r = model.snn_decoder(recv_r)
                feat_recon_l = model.stereo_attn(feat_recon_l, feat_recon_r)
            
            recon = model.img_decoder(feat_recon_l)
            recon = model.refinement(recon).clamp(0, 1)
            
            psnr_sum += compute_psnr(imgs_l, recon) * imgs_l.size(0)
            ssim_sum += compute_ssim_torch(imgs_l, recon).item() * imgs_l.size(0)
            n += imgs_l.size(0)
    
    return {
        'psnr': round(psnr_sum / n, 3),
        'ms_ssim': round(ssim_sum / n, 5),
    }


# =============================================================================
# Training Stages
# =============================================================================

def train_stage1(model, train_loader, test_loader, epochs=60, lr=1e-3):
    """Stage 1: Autoencoder (no masking, no noise) with stereo pairs."""
    print(f"\n{'='*60}")
    print(f"  STAGE 1: Autoencoder (stereo, no masking, no noise)")
    print(f"  Epochs: {epochs}, LR: {lr}")
    print(f"{'='*60}")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_psnr = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, epoch_n = 0, 0
        
        for batch in train_loader:
            imgs_l, imgs_r = batch[0].to(device), batch[1].to(device)
            
            recon, info = model(imgs_l, noise_param=100.0, channel='awgn',
                               use_masking=False, img_right=imgs_r)
            
            loss, mse, ssim = reconstruction_loss(imgs_l, recon)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item() * imgs_l.size(0)
            epoch_n += imgs_l.size(0)
        
        scheduler.step()
        
        if epoch % 5 == 0 or epoch == epochs:
            metrics = evaluate(model, test_loader, 'awgn', 100.0, False, max_batches=20)
            print(f"  E{epoch:02d}: loss={epoch_loss/epoch_n:.5f}, "
                  f"PSNR={metrics['psnr']:.2f}dB, SSIM={metrics['ms_ssim']:.4f}")
            
            if metrics['psnr'] > best_psnr:
                best_psnr = metrics['psnr']
                save_checkpoint(model, 'stage1', best_psnr)
    
    return best_psnr


def train_stage2(model, train_loader, test_loader, epochs=40, lr=5e-4):
    """Stage 2: Channel noise training with stereo pairs."""
    print(f"\n{'='*60}")
    print(f"  STAGE 2: Channel noise training (stereo)")
    print(f"  Epochs: {epochs}, LR: {lr}")
    print(f"{'='*60}")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_psnr = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, epoch_n = 0, 0
        
        for batch in train_loader:
            imgs_l, imgs_r = batch[0].to(device), batch[1].to(device)
            
            if random.random() < 0.5:
                snr = random.choice([1, 4, 7, 10, 13, 19])
                recon, info = model(imgs_l, noise_param=snr, channel='awgn',
                                   use_masking=False, img_right=imgs_r)
            else:
                ber = random.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.30])
                recon, info = model(imgs_l, noise_param=ber, channel='bsc',
                                   use_masking=False, img_right=imgs_r)
            
            loss, mse, ssim = reconstruction_loss(imgs_l, recon)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item() * imgs_l.size(0)
            epoch_n += imgs_l.size(0)
        
        scheduler.step()
        
        if epoch % 5 == 0 or epoch == epochs:
            m_awgn = evaluate(model, test_loader, 'awgn', 7.0, False, max_batches=20)
            m_bsc = evaluate(model, test_loader, 'bsc', 0.10, False, max_batches=20)
            print(f"  E{epoch:02d}: loss={epoch_loss/epoch_n:.5f}, "
                  f"AWGN(7dB)={m_awgn['psnr']:.2f}dB, BSC(0.10)={m_bsc['psnr']:.2f}dB")
            
            avg_psnr = (m_awgn['psnr'] + m_bsc['psnr']) / 2
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                save_checkpoint(model, 'stage2', best_psnr)
    
    return best_psnr


def train_stage3(model, train_loader, test_loader, epochs=40, lr=3e-4, target_cbr=0.5):
    """Stage 3: Scorer training (frozen backbone)."""
    print(f"\n{'='*60}")
    print(f"  STAGE 3: Scorer training (target CBR={target_cbr})")
    print(f"  Epochs: {epochs}, LR: {lr}")
    print(f"{'='*60}")
    
    # Freeze everything except scorer
    for name, p in model.named_parameters():
        if 'scorer' not in name:
            p.requires_grad = False
    
    optimizer = optim.AdamW(list(model.scorer.parameters()), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_psnr = 0
    for epoch in range(1, epochs + 1):
        model.train()
        model.img_encoder.eval()
        model.snn_encoder.eval()
        
        epoch_loss, epoch_n = 0, 0
        
        for batch in train_loader:
            imgs_l, imgs_r = batch[0].to(device), batch[1].to(device)
            
            if random.random() < 0.5:
                noise_param = random.choice([1, 4, 7, 10, 13, 19])
                channel = 'awgn'
            else:
                noise_param = random.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.30])
                channel = 'bsc'
            
            recon, info = model(imgs_l, noise_param=noise_param, channel=channel,
                               use_masking=True, img_right=imgs_r)
            
            loss_recon, mse, ssim = reconstruction_loss(imgs_l, recon)
            
            if info['importance'] is not None:
                mean_imp = info['importance'].mean()
                loss_rate = (mean_imp - target_cbr) ** 2
            else:
                loss_rate = 0.0
            
            with torch.no_grad():
                spikes_detached = [s.detach() for s in info['spikes']]
            loss_div = model.scorer.compute_diversity_loss(spikes_detached)
            
            loss = loss_recon + 10.0 * loss_rate + 0.05 * loss_div
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.scorer.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item() * imgs_l.size(0)
            epoch_n += imgs_l.size(0)
        
        scheduler.step()
        
        if epoch % 5 == 0 or epoch == epochs:
            m = evaluate(model, test_loader, 'awgn', 7.0, True, max_batches=20)
            m_rand = evaluate_random_mask(model, test_loader, 'awgn', 7.0, target_cbr, max_batches=20)
            gap = m['psnr'] - m_rand['psnr']
            print(f"  E{epoch:02d}: loss={epoch_loss/epoch_n:.5f}, "
                  f"Learned={m['psnr']:.2f}dB, Random={m_rand['psnr']:.2f}dB, "
                  f"Gap={gap:+.2f}dB, CBR={m['cbr']:.3f}")
            
            if m['psnr'] > best_psnr:
                best_psnr = m['psnr']
                save_checkpoint(model, 'stage3', best_psnr)
    
    # Unfreeze all
    for p in model.parameters():
        p.requires_grad = True
    return best_psnr


def train_stage4(model, train_loader, test_loader, epochs=15, lr=1e-5, target_cbr=0.5):
    """Stage 4: Joint fine-tuning with all components unfrozen."""
    print(f"\n{'='*60}")
    print(f"  STAGE 4: Joint fine-tuning (all params, stereo+refine)")
    print(f"  Epochs: {epochs}, LR: {lr}")
    print(f"{'='*60}")
    
    for p in model.parameters():
        p.requires_grad = True
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    
    best_psnr = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, epoch_n = 0, 0
        
        for batch in train_loader:
            imgs_l, imgs_r = batch[0].to(device), batch[1].to(device)
            
            if random.random() < 0.5:
                noise_param = random.choice([1, 4, 7, 10, 13, 19])
                channel = 'awgn'
            else:
                noise_param = random.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.30])
                channel = 'bsc'
            
            # 50% with masking, 50% without
            use_mask = random.random() < 0.5
            recon, info = model(imgs_l, noise_param=noise_param, channel=channel,
                               use_masking=use_mask, img_right=imgs_r)
            
            loss_recon, mse, ssim = reconstruction_loss(imgs_l, recon)
            
            if use_mask and info['importance'] is not None:
                mean_imp = info['importance'].mean()
                loss_rate = (mean_imp - target_cbr) ** 2
                loss = loss_recon + 5.0 * loss_rate
            else:
                loss = loss_recon
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            epoch_loss += loss.item() * imgs_l.size(0)
            epoch_n += imgs_l.size(0)
        
        scheduler.step()
        
        if epoch % 3 == 0 or epoch == epochs:
            m = evaluate(model, test_loader, 'awgn', 7.0, True, max_batches=20)
            m_full = evaluate(model, test_loader, 'awgn', 7.0, False, max_batches=20)
            m_rand = evaluate_random_mask(model, test_loader, 'awgn', 7.0, target_cbr, max_batches=20)
            gap = m['psnr'] - m_rand['psnr']
            print(f"  E{epoch:02d}: loss={epoch_loss/epoch_n:.5f}, "
                  f"Full={m_full['psnr']:.2f}dB, Masked={m['psnr']:.2f}dB, "
                  f"Random={m_rand['psnr']:.2f}dB, Gap={gap:+.2f}dB")
            
            if m_full['psnr'] > best_psnr:
                best_psnr = m_full['psnr']
                save_checkpoint(model, 'stage4', best_psnr)
    
    return best_psnr


# =============================================================================
# Utilities
# =============================================================================

SNAP_DIR = './snapshots_kitti_v3'

def save_checkpoint(model, stage, psnr):
    os.makedirs(SNAP_DIR, exist_ok=True)
    path = os.path.join(SNAP_DIR, f'{stage}_best_{psnr:.2f}.pth')
    torch.save(model.state_dict(), path)
    print(f"  ✓ Saved {path}")


def load_best_checkpoint(model, stage):
    cks = sorted(glob.glob(os.path.join(SNAP_DIR, f'{stage}_best_*.pth')),
                 key=lambda x: float(x.split('_')[-1].replace('.pth', '')))
    if cks:
        state = torch.load(cks[-1], map_location=device, weights_only=False)
        model_state = model.state_dict()
        filtered = {k: v for k, v in state.items() 
                    if k in model_state and v.shape == model_state[k].shape}
        model.load_state_dict(filtered, strict=False)
        skipped = set(state.keys()) - set(filtered.keys())
        if skipped:
            print(f"  (skipped {len(skipped)} mismatched keys)")
        print(f"  Loaded {cks[-1]}")
        return True
    return False


def load_v2_checkpoint(model):
    """Load pretrained V2 weights (skip new modules: stereo_attn, refinement, temporal_attn)."""
    v2_dir = './snapshots_kitti_v2'
    cks = sorted(glob.glob(os.path.join(v2_dir, 'stage4_best_*.pth')),
                 key=lambda x: float(x.split('_')[-1].replace('.pth', '')))
    if not cks:
        cks = sorted(glob.glob(os.path.join(v2_dir, 'stage3_best_*.pth')),
                     key=lambda x: float(x.split('_')[-1].replace('.pth', '')))
    if cks:
        state = torch.load(cks[-1], map_location=device, weights_only=False)
        model_state = model.state_dict()
        filtered = {k: v for k, v in state.items() 
                    if k in model_state and v.shape == model_state[k].shape}
        model.load_state_dict(filtered, strict=False)
        loaded = set(filtered.keys())
        new_keys = set(model_state.keys()) - loaded
        print(f"  Loaded V2 checkpoint: {cks[-1]}")
        print(f"  Transferred {len(loaded)} params, new modules: {len(new_keys)} params")
        return True
    print("  No V2 checkpoint found.")
    return False


# =============================================================================
# Full Evaluation
# =============================================================================

def full_eval(model, test_loader, target_cbr, dataset_name='instereo2k', 
              stereo=True):
    print(f"\n{'='*60}")
    print(f"  FULL EVALUATION: {dataset_name} ({'stereo' if stereo else 'single'})")
    print(f"{'='*60}")
    
    results = {'awgn': {}, 'bsc': {}}
    
    for snr in [1, 4, 7, 10, 13, 19]:
        results['awgn'][str(snr)] = {}
        for cbr in [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0]:
            use_mask = cbr < 1.0
            m_l = evaluate(model, test_loader, 'awgn', snr, use_mask, cbr,
                          stereo=stereo)
            m_r = evaluate_random_mask(model, test_loader, 'awgn', snr, cbr,
                                       stereo=stereo)
            results['awgn'][str(snr)][str(cbr)] = {
                'learned': m_l, 'random': m_r,
                'gap_psnr': round(m_l['psnr'] - m_r['psnr'], 3),
            }
            print(f"  AWGN SNR={snr:2d}dB CBR={cbr:.3f}: "
                  f"L={m_l['psnr']:.2f}dB R={m_r['psnr']:.2f}dB "
                  f"Δ={m_l['psnr']-m_r['psnr']:+.2f}dB")
    
    for ber in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
        results['bsc'][str(ber)] = {}
        for cbr in [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0]:
            use_mask = cbr < 1.0
            m_l = evaluate(model, test_loader, 'bsc', ber, use_mask, cbr,
                          stereo=stereo)
            m_r = evaluate_random_mask(model, test_loader, 'bsc', ber, cbr,
                                       stereo=stereo)
            results['bsc'][str(ber)][str(cbr)] = {
                'learned': m_l, 'random': m_r,
                'gap_psnr': round(m_l['psnr'] - m_r['psnr'], 3),
            }
            print(f"  BSC BER={ber:.2f} CBR={cbr:.3f}: "
                  f"L={m_l['psnr']:.2f}dB R={m_r['psnr']:.2f}dB "
                  f"Δ={m_l['psnr']-m_r['psnr']:+.2f}dB")
    
    os.makedirs('eval/seed_results', exist_ok=True)
    outfile = f'eval/seed_results/kitti_image_v3_{dataset_name}.json'
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved to {outfile}")
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, default=0, help='Run specific stage (0=all)')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--target-cbr', type=float, default=0.5)
    parser.add_argument('--C-tx', type=int, default=64)
    parser.add_argument('--T', type=int, default=8)
    parser.add_argument('--from-v2', action='store_true',
                        help='Initialize from V2 checkpoint (transfer learning)')
    args = parser.parse_args()
    
    print(f"Device: {device}")
    print(f"C_tx={args.C_tx}, T={args.T}, target_CBR={args.target_cbr}")
    print(f"Model: V3 (stereo + refinement + temporal attention)")
    
    # Data
    train_ds, test_ds, kitti_test_ds = get_datasets('./InStereo2K', './data')
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False,
                             num_workers=4, pin_memory=True)
    kitti_loader = None
    if kitti_test_ds is not None:
        kitti_loader = DataLoader(kitti_test_ds, batch_size=2, shuffle=False,
                                  num_workers=4, pin_memory=True)
    
    # Model — stereo=True
    model = SpikeImageCodecV2(
        C_feat=384, C_tx=args.C_tx, T=args.T,
        target_cbr=args.target_cbr, stereo=True
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")
    
    # Optional: init from V2 checkpoint
    if args.from_v2:
        load_v2_checkpoint(model)
    
    if args.eval_only:
        load_best_checkpoint(model, 'stage4') or load_best_checkpoint(model, 'stage3')
        full_eval(model, test_loader, args.target_cbr, 'instereo2k', stereo=True)
        if kitti_loader:
            # KITTI single-image eval (no stereo pair, but model can still work)
            full_eval(model, kitti_loader, args.target_cbr, 'kitti', stereo=False)
        return
    
    # Training
    if args.stage == 0 or args.stage == 1:
        if not load_best_checkpoint(model, 'stage1'):
            train_stage1(model, train_loader, test_loader, epochs=60)
        else:
            print("  Stage 1 checkpoint found, skipping.")
    
    if args.stage == 0 or args.stage == 2:
        if not load_best_checkpoint(model, 'stage1'):
            print("  WARNING: No stage 1 checkpoint.")
        if not load_best_checkpoint(model, 'stage2'):
            train_stage2(model, train_loader, test_loader, epochs=40)
        else:
            print("  Stage 2 checkpoint found, skipping.")
    
    if args.stage == 0 or args.stage == 3:
        if not load_best_checkpoint(model, 'stage2'):
            print("  WARNING: No stage 2 checkpoint.")
        if not load_best_checkpoint(model, 'stage3'):
            train_stage3(model, train_loader, test_loader, epochs=40,
                         target_cbr=args.target_cbr)
        else:
            print("  Stage 3 checkpoint found, skipping.")
    
    if args.stage == 0 or args.stage == 4:
        if not load_best_checkpoint(model, 'stage3'):
            print("  WARNING: No stage 3 checkpoint.")
        train_stage4(model, train_loader, test_loader, epochs=15,
                     target_cbr=args.target_cbr)
    
    # Final eval
    load_best_checkpoint(model, 'stage4') or load_best_checkpoint(model, 'stage3')
    full_eval(model, test_loader, args.target_cbr, 'instereo2k', stereo=True)
    if kitti_loader:
        full_eval(model, kitti_loader, args.target_cbr, 'kitti', stereo=False)


if __name__ == '__main__':
    main()
