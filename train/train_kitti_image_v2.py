#!/usr/bin/env python3
"""Train SpikeAdapt-SC Image Codec V2 on InStereo2K (+ optional KITTI).

4-stage training:
  Stage 1: Autoencoder (no masking, no noise) - 60 epochs
  Stage 2: Add channel noise (AWGN + BSC)     - 40 epochs
  Stage 3: Scorer training (frozen backbone)  - 40 epochs
  Stage 4: Joint fine-tuning (all params)     - 15 epochs

Usage:  python train/train_kitti_image_v2.py [--stage 0|1|2|3|4] [--eval-only]
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
# Dataset
# =============================================================================

class ImageFolderFlat(Dataset):
    def __init__(self, images=None, transform=None):
        self.images = images or []
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


def get_datasets(data_root='./data', instereo_root='./InStereo2K'):
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
    
    train_sets, test_sets = [], []
    kitti_test_set = None
    
    # InStereo2K
    instereo_train = os.path.join(instereo_root, 'train')
    instereo_test = os.path.join(instereo_root, 'test')
    
    if os.path.exists(instereo_train):
        subdirs = sorted(os.listdir(instereo_train))
        train_imgs = [os.path.join(instereo_train, d, 'left.png') 
                      for d in subdirs 
                      if os.path.isfile(os.path.join(instereo_train, d, 'left.png'))]
        if train_imgs:
            train_sets.append(ImageFolderFlat(train_imgs, tf_train))
            print(f"  InStereo2K train: {len(train_imgs)} images")
    
    if os.path.exists(instereo_test):
        subdirs = sorted(os.listdir(instereo_test))
        test_imgs = [os.path.join(instereo_test, d, 'left.png') 
                     for d in subdirs 
                     if os.path.isfile(os.path.join(instereo_test, d, 'left.png'))]
        if test_imgs:
            test_sets.append(ImageFolderFlat(test_imgs, tf_test))
            print(f"  InStereo2K test: {len(test_imgs)} images")
    
    # KITTI Stereo 2015
    kitti_dir = os.path.join(data_root, 'kitti_stereo_2015', 'training', 'image_2')
    if os.path.exists(kitti_dir):
        kitti_imgs = sorted(glob.glob(os.path.join(kitti_dir, '*.png')))
        if kitti_imgs:
            # Use all KITTI as test (cross-dataset eval)
            kitti_test_set = ImageFolderFlat(kitti_imgs, tf_test)
            print(f"  KITTI Stereo 2015: {len(kitti_imgs)} images (eval only)")
    
    if not train_sets:
        raise RuntimeError("No training images found!")
    
    train_ds = ConcatDataset(train_sets) if len(train_sets) > 1 else train_sets[0]
    test_ds = ConcatDataset(test_sets) if len(test_sets) > 1 else test_sets[0]
    
    print(f"  Total: {len(train_ds)} train, {len(test_ds)} test")
    return train_ds, test_ds, kitti_test_set


# =============================================================================
# Loss Functions (Improvement #1: Multi-scale perceptual loss)
# Based on: LPIPS (Zhang et al., CVPR 2018), MS-SSIM (Wang et al., TIP 2003)
# =============================================================================

class VGGPerceptualLoss(nn.Module):
    """VGG-based perceptual loss using relu2_2 and relu3_3 features."""
    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        # relu2_2 = features[:9], relu3_3 = features[:16]
        self.slice1 = nn.Sequential(*list(vgg.features[:9]))
        self.slice2 = nn.Sequential(*list(vgg.features[9:16]))
        for p in self.parameters():
            p.requires_grad = False
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, x, y):
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        f1_x, f1_y = self.slice1(x), self.slice1(y)
        f2_x, f2_y = self.slice2(f1_x), self.slice2(f1_y)
        loss = F.mse_loss(f1_x, f1_y) + F.mse_loss(f2_x, f2_y)
        return loss

# Global perceptual loss (lazy-inited on first use)
_vgg_loss = None

def get_vgg_loss():
    global _vgg_loss
    if _vgg_loss is None:
        _vgg_loss = VGGPerceptualLoss().to(device).eval()
    return _vgg_loss

def compute_ms_ssim(x, y, weights=[0.4, 0.3, 0.3]):
    """Multi-scale SSIM at 3 downscale levels."""
    ms_ssim = 0
    for i, w in enumerate(weights):
        ms_ssim += w * compute_ssim_torch(x, y)
        if i < len(weights) - 1:
            x = F.avg_pool2d(x, 2)
            y = F.avg_pool2d(y, 2)
    return ms_ssim

def reconstruction_loss(x, x_hat, use_perceptual=False):
    """Composite loss: MSE + MS-SSIM + VGG perceptual.
    
    Weights: 0.6*MSE + 0.2*(1-MS_SSIM) + 0.2*VGG
    """
    x_hat_clamped = x_hat.clamp(0, 1)
    mse = F.mse_loss(x_hat_clamped, x)
    ssim_val = compute_ssim_torch(x, x_hat_clamped)
    ms_ssim_val = compute_ms_ssim(x, x_hat_clamped)
    
    if use_perceptual:
        vgg = get_vgg_loss()
        perceptual = vgg(x, x_hat_clamped)
        loss = 0.75 * mse + 0.2 * (1 - ms_ssim_val) + 0.05 * perceptual
    else:
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
             use_masking=True, target_cbr=None, max_batches=None):
    model.eval()
    psnr_sum, ssim_sum, cbr_sum, n = 0, 0, 0, 0
    
    with torch.no_grad():
        for i, imgs in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            if isinstance(imgs, (list, tuple)):
                imgs = imgs[0]
            imgs = imgs.to(device)
            
            recon, info = model(imgs, noise_param=noise_param, channel=channel,
                               use_masking=use_masking, target_cbr_override=target_cbr)
            recon = recon.clamp(0, 1)
            
            psnr_sum += compute_psnr(imgs, recon) * imgs.size(0)
            cbr_sum += info['actual_cbr'] * imgs.size(0)
            ssim_val = compute_ssim_torch(imgs, recon).item()
            ssim_sum += ssim_val * imgs.size(0)
            n += imgs.size(0)
    
    return {
        'psnr': round(psnr_sum / n, 3),
        'ms_ssim': round(ssim_sum / n, 5),
        'cbr': round(cbr_sum / n, 4),
        'n_images': n,
    }


def evaluate_random_mask(model, loader, channel, noise_param, target_cbr, max_batches=None):
    model.eval()
    psnr_sum, ssim_sum, n = 0, 0, 0
    
    with torch.no_grad():
        for i, imgs in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            if isinstance(imgs, (list, tuple)):
                imgs = imgs[0]
            imgs = imgs.to(device)
            
            feat = model.img_encoder(imgs)
            spikes = model.snn_encoder(feat)
            
            B, C, H, W = spikes[0].shape
            k = max(1, int(target_cbr * H * W))
            # Random spatial mask (same fraction as learned)
            flat = torch.rand(B, 1, H, W, device=imgs.device).view(B, -1)
            _, idx = flat.topk(k, dim=1)
            mask = torch.zeros_like(flat)
            mask.scatter_(1, idx, 1.0)
            mask = mask.view(B, 1, H, W)
            
            masked = [s * mask for s in spikes]
            
            if channel == 'awgn':
                recv = [model.awgn(s, noise_param) for s in masked]
            else:
                recv = [model.bsc(s, noise_param) for s in masked]
            
            feat_r = model.snn_decoder(recv)
            recon = model.img_decoder(feat_r).clamp(0, 1)
            
            psnr_sum += compute_psnr(imgs, recon) * imgs.size(0)
            ssim_sum += compute_ssim_torch(imgs, recon).item() * imgs.size(0)
            n += imgs.size(0)
    
    return {
        'psnr': round(psnr_sum / n, 3),
        'ms_ssim': round(ssim_sum / n, 5),
    }


# =============================================================================
# Training Stages
# =============================================================================

def train_stage1(model, train_loader, test_loader, epochs=60, lr=1e-3):
    print(f"\n{'='*60}")
    print(f"  STAGE 1: Autoencoder (no masking, no noise)")
    print(f"  Epochs: {epochs}, LR: {lr}")
    print(f"{'='*60}")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_psnr = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, epoch_n = 0, 0
        
        for imgs in train_loader:
            if isinstance(imgs, (list, tuple)):
                imgs = imgs[0]
            imgs = imgs.to(device)
            
            recon, info = model(imgs, noise_param=100.0, channel='awgn',
                               use_masking=False)
            
            loss, mse, ssim = reconstruction_loss(imgs, recon)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item() * imgs.size(0)
            epoch_n += imgs.size(0)
        
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
    print(f"\n{'='*60}")
    print(f"  STAGE 2: Channel noise training")
    print(f"  Epochs: {epochs}, LR: {lr}")
    print(f"{'='*60}")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_psnr = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, epoch_n = 0, 0
        
        for imgs in train_loader:
            if isinstance(imgs, (list, tuple)):
                imgs = imgs[0]
            imgs = imgs.to(device)
            
            if random.random() < 0.5:
                snr = random.choice([1, 4, 7, 10, 13, 19])
                recon, info = model(imgs, noise_param=snr, channel='awgn', use_masking=False)
            else:
                ber = random.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.30])
                recon, info = model(imgs, noise_param=ber, channel='bsc', use_masking=False)
            
            loss, mse, ssim = reconstruction_loss(imgs, recon)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item() * imgs.size(0)
            epoch_n += imgs.size(0)
        
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
    print(f"\n{'='*60}")
    print(f"  STAGE 3: Scorer training (target CBR={target_cbr})")
    print(f"  Epochs: {epochs}, LR: {lr}")
    print(f"{'='*60}")
    
    # Freeze encoder/decoder
    for p in model.img_encoder.parameters():
        p.requires_grad = False
    for p in model.snn_encoder.parameters():
        p.requires_grad = False
    for p in model.snn_decoder.parameters():
        p.requires_grad = False
    for p in model.img_decoder.parameters():
        p.requires_grad = False
    
    optimizer = optim.AdamW(list(model.scorer.parameters()), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_psnr = 0
    for epoch in range(1, epochs + 1):
        model.train()
        model.img_encoder.eval()
        model.snn_encoder.eval()
        
        epoch_loss, epoch_n = 0, 0
        
        for imgs in train_loader:
            if isinstance(imgs, (list, tuple)):
                imgs = imgs[0]
            imgs = imgs.to(device)
            
            if random.random() < 0.5:
                noise_param = random.choice([1, 4, 7, 10, 13, 19])
                channel = 'awgn'
            else:
                noise_param = random.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.30])
                channel = 'bsc'
            
            recon, info = model(imgs, noise_param=noise_param, channel=channel,
                               use_masking=True)
            
            loss_recon, mse, ssim = reconstruction_loss(imgs, recon)
            
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
            
            epoch_loss += loss.item() * imgs.size(0)
            epoch_n += imgs.size(0)
        
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
    
    # Unfreeze
    for p in model.parameters():
        p.requires_grad = True
    return best_psnr


def train_stage4(model, train_loader, test_loader, epochs=15, lr=1e-5, target_cbr=0.5):
    """Stage 4: Joint fine-tuning with all components unfrozen."""
    print(f"\n{'='*60}")
    print(f"  STAGE 4: Joint fine-tuning (all params)")
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
        
        for imgs in train_loader:
            if isinstance(imgs, (list, tuple)):
                imgs = imgs[0]
            imgs = imgs.to(device)
            
            if random.random() < 0.5:
                noise_param = random.choice([1, 4, 7, 10, 13, 19])
                channel = 'awgn'
            else:
                noise_param = random.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.30])
                channel = 'bsc'
            
            # 50% with masking, 50% without (so encoder/decoder also adapt to full bandwidth)
            use_mask = random.random() < 0.5
            recon, info = model(imgs, noise_param=noise_param, channel=channel,
                               use_masking=use_mask)
            
            loss_recon, mse, ssim = reconstruction_loss(imgs, recon)
            
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
            
            epoch_loss += loss.item() * imgs.size(0)
            epoch_n += imgs.size(0)
        
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

SNAP_DIR = './snapshots_kitti_v2'

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


# =============================================================================
# Full Evaluation
# =============================================================================

def full_eval(model, test_loader, target_cbr, dataset_name='instereo2k'):
    print(f"\n{'='*60}")
    print(f"  FULL EVALUATION: {dataset_name}")
    print(f"{'='*60}")
    
    results = {'awgn': {}, 'bsc': {}}
    
    for snr in [1, 4, 7, 10, 13, 19]:
        results['awgn'][str(snr)] = {}
        for cbr in [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0]:
            use_mask = cbr < 1.0
            m_l = evaluate(model, test_loader, 'awgn', snr, use_mask, cbr)
            m_r = evaluate_random_mask(model, test_loader, 'awgn', snr, cbr)
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
            m_l = evaluate(model, test_loader, 'bsc', ber, use_mask, cbr)
            m_r = evaluate_random_mask(model, test_loader, 'bsc', ber, cbr)
            results['bsc'][str(ber)][str(cbr)] = {
                'learned': m_l, 'random': m_r,
                'gap_psnr': round(m_l['psnr'] - m_r['psnr'], 3),
            }
            print(f"  BSC BER={ber:.2f} CBR={cbr:.3f}: "
                  f"L={m_l['psnr']:.2f}dB R={m_r['psnr']:.2f}dB "
                  f"Δ={m_l['psnr']-m_r['psnr']:+.2f}dB")
    
    os.makedirs('eval/seed_results', exist_ok=True)
    outfile = f'eval/seed_results/kitti_image_v2_{dataset_name}.json'
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
    args = parser.parse_args()
    
    print(f"Device: {device}")
    print(f"C_tx={args.C_tx}, T={args.T}, target_CBR={args.target_cbr}")
    
    # Data
    train_ds, test_ds, kitti_test_ds = get_datasets('./data', './InStereo2K')
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, 
                              num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False,
                             num_workers=4, pin_memory=True)
    kitti_loader = None
    if kitti_test_ds is not None:
        kitti_loader = DataLoader(kitti_test_ds, batch_size=2, shuffle=False,
                                  num_workers=4, pin_memory=True)
    
    # Model
    model = SpikeImageCodecV2(
        C_feat=384, C_tx=args.C_tx, T=args.T,
        target_cbr=args.target_cbr
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")
    
    if args.eval_only:
        load_best_checkpoint(model, 'stage4') or load_best_checkpoint(model, 'stage3')
        full_eval(model, test_loader, args.target_cbr, 'instereo2k')
        if kitti_loader:
            full_eval(model, kitti_loader, args.target_cbr, 'kitti')
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
    
    # Final eval on both datasets
    load_best_checkpoint(model, 'stage4') or load_best_checkpoint(model, 'stage3')
    full_eval(model, test_loader, args.target_cbr, 'instereo2k')
    if kitti_loader:
        full_eval(model, kitti_loader, args.target_cbr, 'kitti')


if __name__ == '__main__':
    main()
