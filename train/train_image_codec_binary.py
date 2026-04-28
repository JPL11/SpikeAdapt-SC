#!/usr/bin/env python3
"""B1: CNN-Binary Image Codec — same V6 backbone with sign-quantized bottleneck + BSC.

Tests: "Is the PSNR gap from SNN architecture or from binary quantization?"

Architecture:
  MultiScaleEncoder (CNN, shared with V6)
  → Conv+BN+Tanh per scale (continuous [-1,1])
  → sign_quantize via STE → {-1, +1}
  → BSC channel (bit flip with prob BER)
  → Conv+BN+LeakyReLU decoder per scale
  → CrossScaleFusion → final upscale
  → Sigmoid → image reconstruction

Training: 3 stages:
  S1 (60 epochs): Clean reconstruction (BER=0)
  S2 (50 epochs): Mixed noise (BER curriculum)
  S3 (40 epochs): Joint scorer + masking fine-tune

Output: snapshots_kitti_binary/
Eval: eval/seed_results/binary_codec_results.json
"""

import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
import os, sys, json, random, math, glob, argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from models.spike_image_codec_v6 import MultiScaleEncoder, CrossScaleFusion
from models.spike_image_codec_v4 import ResBlock

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CROP_H, CROP_W = 256, 512
SNAP_DIR = './snapshots_kitti_binary'


# =============================================================================
# STE for sign quantization
# =============================================================================

class SignQuantizeSTE(torch.autograd.Function):
    """Sign quantization with straight-through estimator."""
    @staticmethod
    def forward(ctx, x):
        return torch.sign(x + 1e-8)  # {-1, +1}, avoid 0

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # STE: pass gradient through


# =============================================================================
# BSC channel for {-1, +1} symbols
# =============================================================================

class BinaryBSC(nn.Module):
    """BSC for binary {-1, +1}: flip sign with probability ber."""
    def forward(self, x, ber):
        if ber <= 0:
            return x
        flip = (torch.rand_like(x) < ber).float()
        noisy = x * (1 - 2 * flip)  # flip sign
        if self.training:
            return x + (noisy - x).detach()  # STE
        return noisy


# =============================================================================
# CNN-Binary Codec
# =============================================================================

class BinaryImageCodec(nn.Module):
    """Multi-scale CNN codec with binary quantization + BSC."""
    def __init__(self, C_tx=(16, 32, 48, 64), target_cbr=0.5):
        super().__init__()
        feat_chs = [96, 192, 384, 512]
        self.C_tx = C_tx
        self.target_cbr = target_cbr

        # Shared CNN encoder backbone (same as V6)
        self.encoder = MultiScaleEncoder()

        # Per-scale bottleneck encoders: feature → continuous → sign quantize
        self.bottleneck_encs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(fc, ct, 3, 1, 1), nn.BatchNorm2d(ct), nn.Tanh()
        ) for fc, ct in zip(feat_chs, C_tx)])

        # Per-scale decoders
        self.bottleneck_decs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(ct, fc, 3, 1, 1), nn.BatchNorm2d(fc), nn.LeakyReLU(0.2, True)
        ) for fc, ct in zip(feat_chs, C_tx)])

        # Per-scale importance scorers (for masking)
        self.scorers = nn.ModuleList([nn.Sequential(
            nn.Conv2d(ct, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 1, 1), nn.Sigmoid()
        ) for ct in C_tx])

        # Cross-scale fusion (same as V6)
        self.fuse_43 = CrossScaleFusion(512, 384)
        self.fuse_32 = CrossScaleFusion(384, 192)
        self.fuse_21 = CrossScaleFusion(192, 96)
        self.up_final = nn.Sequential(
            nn.ConvTranspose2d(96, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True),
            ResBlock(64), nn.Conv2d(64, 3, 3, 1, 1), nn.Sigmoid())

        self.bsc = BinaryBSC()

    def forward(self, img, ber=0.0, use_masking=True, target_cbr_override=None):
        target = target_cbr_override if target_cbr_override is not None else self.target_cbr

        # Multi-scale encode
        feats = self.encoder(img)
        zs = [enc(f) for enc, f in zip(self.bottleneck_encs, feats)]

        # Sign quantize: continuous [-1,1] → binary {-1, +1}
        zs_binary = [SignQuantizeSTE.apply(z) for z in zs]

        # Masking
        actual_cbrs = []
        masked = []
        for i, z in enumerate(zs_binary):
            if use_masking and target < 1.0:
                B, C, H, W = z.shape
                imp = self.scorers[i](zs[i].detach())  # Score from pre-quant
                k = max(1, int(target * H * W))
                if self.training:
                    # Gumbel-sigmoid soft mask
                    logits = torch.log(imp / (1 - imp + 1e-7) + 1e-7)
                    u = torch.rand_like(logits).clamp(1e-7, 1 - 1e-7)
                    soft = torch.sigmoid((logits - torch.log(-torch.log(u))) / 0.5)
                    hard = (soft > 0.5).float()
                    mask = hard + (soft - soft.detach())
                else:
                    flat = imp.view(B, -1)
                    _, idx = flat.topk(k, dim=1)
                    mask = torch.zeros_like(flat)
                    mask.scatter_(1, idx, 1.0)
                    mask = mask.view(B, 1, H, W)
                masked.append(z * mask)
                actual_cbrs.append(k / (H * W))
            else:
                masked.append(z)
                actual_cbrs.append(1.0)

        # BSC channel
        recv = [self.bsc(z, ber) for z in masked]

        # Multi-scale decode
        ds = [dec(z) for dec, z in zip(self.bottleneck_decs, recv)]
        f3 = self.fuse_43(ds[3], ds[2])
        f2 = self.fuse_32(f3, ds[1])
        f1 = self.fuse_21(f2, ds[0])
        img_recon = self.up_final(f1)

        return img_recon, {'actual_cbrs': actual_cbrs, 'zs': zs}


# =============================================================================
# Dataset
# =============================================================================

class ImageFolderFlat(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        return self.transform(img) if self.transform else img


def get_datasets():
    tf_train = T.Compose([T.RandomCrop((CROP_H, CROP_W), pad_if_needed=True, padding_mode='reflect'),
                          T.RandomHorizontalFlip(), T.ColorJitter(0.1, 0.1), T.ToTensor()])
    tf_test = T.Compose([T.CenterCrop((CROP_H, CROP_W)), T.ToTensor()])

    train_imgs, test_imgs = [], []
    d = './InStereo2K'
    if os.path.exists(os.path.join(d, 'train')):
        train_imgs = sorted([os.path.join(d, 'train', s, 'left.png')
                            for s in os.listdir(os.path.join(d, 'train'))
                            if os.path.isfile(os.path.join(d, 'train', s, 'left.png'))])
    if os.path.exists(os.path.join(d, 'test')):
        test_imgs = sorted([os.path.join(d, 'test', s, 'left.png')
                           for s in os.listdir(os.path.join(d, 'test'))
                           if os.path.isfile(os.path.join(d, 'test', s, 'left.png'))])

    kitti_imgs = []
    kd = './data/kitti_stereo_2015/training/image_2'
    if os.path.exists(kd):
        kitti_imgs = sorted(glob.glob(os.path.join(kd, '*.png')))

    train_ds = ImageFolderFlat(train_imgs, tf_train) if train_imgs else None
    test_ds = ImageFolderFlat(test_imgs, tf_test) if test_imgs else None
    kitti_ds = ImageFolderFlat(kitti_imgs, tf_test) if kitti_imgs else None
    return train_ds, test_ds, kitti_ds


# =============================================================================
# Loss and metrics
# =============================================================================

def compute_ssim_torch(x, y, window_size=11, data_range=1.0):
    """Gaussian-windowed SSIM."""
    C1, C2 = (0.01 * data_range)**2, (0.03 * data_range)**2
    coords = torch.arange(window_size, dtype=torch.float32, device=x.device) - window_size // 2
    g = torch.exp(-coords**2 / (2 * 1.5**2))
    g = g / g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)
    window = window.unsqueeze(0).unsqueeze(0).expand(x.size(1), -1, -1, -1)
    pad = window_size // 2
    mu_x = F.conv2d(x, window, padding=pad, groups=x.size(1))
    mu_y = F.conv2d(y, window, padding=pad, groups=y.size(1))
    mu_xx, mu_yy, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y
    s_xx = F.conv2d(x*x, window, padding=pad, groups=x.size(1)) - mu_xx
    s_yy = F.conv2d(y*y, window, padding=pad, groups=y.size(1)) - mu_yy
    s_xy = F.conv2d(x*y, window, padding=pad, groups=x.size(1)) - mu_xy
    ssim = ((2*mu_xy+C1)*(2*s_xy+C2)) / ((mu_xx+mu_yy+C1)*(s_xx+s_yy+C2))
    return ssim.mean()


def recon_loss(x, xh):
    xc = xh.clamp(0, 1)
    mse = F.mse_loss(xc, x)
    ms = 0
    a, b = x, xc
    for i, w in enumerate([0.4, 0.3, 0.3]):
        ms += w * compute_ssim_torch(a, b)
        if i < 2:
            a, b = F.avg_pool2d(a, 2), F.avg_pool2d(b, 2)
    return 0.85 * mse + 0.15 * (1 - ms), mse


def psnr(x, xh):
    m = F.mse_loss(xh, x).item()
    return 100.0 if m == 0 else 10 * math.log10(1.0 / m)


# =============================================================================
# Training
# =============================================================================

def train_stage(model, train_loader, val_loader, epochs, lr, stage_name,
                use_noise=False, use_masking=False, target_cbr=0.5):
    """Generic training stage."""
    params = model.parameters()
    optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    os.makedirs(SNAP_DIR, exist_ok=True)
    best_psnr = 0

    for epoch in range(epochs):
        model.train()
        total_loss, n = 0, 0
        for imgs in train_loader:
            if isinstance(imgs, (list, tuple)):
                imgs = imgs[0]
            imgs = imgs.to(device)

            # Sample noise
            ber = 0.0
            if use_noise:
                ber = random.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.30])

            recon, info = model(imgs, ber=ber, use_masking=use_masking,
                               target_cbr_override=target_cbr if use_masking else None)
            loss, mse = recon_loss(imgs, recon)

            # Rate penalty when masking
            if use_masking:
                rate_loss = sum((cbr - target_cbr)**2 for cbr in info['actual_cbrs']) / 4
                loss = loss + 5.0 * rate_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)

        scheduler.step()

        # Validate
        model.eval()
        val_psnr, val_n = 0, 0
        with torch.no_grad():
            for imgs in val_loader:
                if isinstance(imgs, (list, tuple)):
                    imgs = imgs[0]
                imgs = imgs.to(device)
                recon, _ = model(imgs, ber=0.0, use_masking=False)
                val_psnr += psnr(imgs, recon.clamp(0, 1)) * imgs.size(0)
                val_n += imgs.size(0)
        val_psnr /= val_n

        print(f"[{stage_name}] Epoch {epoch+1}/{epochs}: loss={total_loss/n:.4f} PSNR={val_psnr:.2f}dB lr={scheduler.get_last_lr()[0]:.2e}")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(),
                       os.path.join(SNAP_DIR, f'{stage_name}_best_{val_psnr:.2f}.pth'))

    return best_psnr


def evaluate_full(model, loader, name):
    """Full evaluation sweep: BSC BER × CBR."""
    model.eval()
    ber_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    cbr_list = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0]

    results = {'bsc': {}}
    for ber in ber_list:
        results['bsc'][str(ber)] = {}
        for cbr in cbr_list:
            total_psnr, n = 0, 0
            with torch.no_grad():
                for imgs in loader:
                    if isinstance(imgs, (list, tuple)):
                        imgs = imgs[0]
                    imgs = imgs.to(device)
                    recon, _ = model(imgs, ber=ber, use_masking=(cbr < 1.0),
                                    target_cbr_override=cbr)
                    total_psnr += psnr(imgs, recon.clamp(0, 1)) * imgs.size(0)
                    n += imgs.size(0)
            avg_psnr = total_psnr / n if n > 0 else 0
            results['bsc'][str(ber)][str(cbr)] = {'psnr': round(avg_psnr, 3), 'n': n}
            print(f"  [{name}] BER={ber} CBR={cbr}: PSNR={avg_psnr:.2f}dB")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='all', choices=['s1', 's2', 's3', 'eval', 'all'])
    parser.add_argument('--target_cbr', type=float, default=0.5)
    args = parser.parse_args()

    train_ds, test_ds, kitti_ds = get_datasets()
    if not train_ds:
        print("ERROR: No training data found")
        return

    tl = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    vl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    model = BinaryImageCodec(target_cbr=args.target_cbr).to(device)
    print(f"Device: {device}")
    print(f"Train: {len(train_ds)} images, Test: {len(test_ds)} images")
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    if args.stage in ['s1', 'all']:
        print(f"\n{'='*60}\nStage 1: Clean reconstruction\n{'='*60}")
        train_stage(model, tl, vl, epochs=60, lr=1e-3, stage_name='s1',
                   use_noise=False, use_masking=False)

    if args.stage in ['s2', 'all']:
        # Load best S1
        s1_ck = sorted(glob.glob(os.path.join(SNAP_DIR, 's1_best_*.pth')),
                       key=lambda x: float(x.split('_')[-1].replace('.pth', '')))
        if s1_ck:
            model.load_state_dict(torch.load(s1_ck[-1], map_location=device, weights_only=False))
            print(f"Loaded {s1_ck[-1]}")
        print(f"\n{'='*60}\nStage 2: Noise robustness\n{'='*60}")
        train_stage(model, tl, vl, epochs=50, lr=5e-4, stage_name='s2',
                   use_noise=True, use_masking=False)

    if args.stage in ['s3', 'all']:
        s2_ck = sorted(glob.glob(os.path.join(SNAP_DIR, 's2_best_*.pth')),
                       key=lambda x: float(x.split('_')[-1].replace('.pth', '')))
        if s2_ck:
            model.load_state_dict(torch.load(s2_ck[-1], map_location=device, weights_only=False))
            print(f"Loaded {s2_ck[-1]}")
        print(f"\n{'='*60}\nStage 3: Masking fine-tune\n{'='*60}")
        train_stage(model, tl, vl, epochs=40, lr=1e-5, stage_name='s3',
                   use_noise=True, use_masking=True, target_cbr=args.target_cbr)

    if args.stage in ['eval', 'all']:
        # Load best model
        for tag in ['s3', 's2', 's1']:
            cks = sorted(glob.glob(os.path.join(SNAP_DIR, f'{tag}_best_*.pth')),
                         key=lambda x: float(x.split('_')[-1].replace('.pth', '')))
            if cks:
                model.load_state_dict(torch.load(cks[-1], map_location=device, weights_only=False))
                print(f"Loaded {cks[-1]} for evaluation")
                break

        all_results = {}
        if test_ds:
            print(f"\n{'='*60}\nEvaluating InStereo2K\n{'='*60}")
            all_results['instereo2k'] = evaluate_full(model, vl, 'InStereo2K')
        if kitti_ds:
            kl = DataLoader(kitti_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
            print(f"\n{'='*60}\nEvaluating KITTI\n{'='*60}")
            all_results['kitti'] = evaluate_full(model, kl, 'KITTI')

        os.makedirs('eval/seed_results', exist_ok=True)
        out = 'eval/seed_results/binary_codec_results.json'
        with open(out, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved to {out}")


if __name__ == '__main__':
    main()
