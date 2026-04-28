#!/usr/bin/env python3
"""Stage A: Retrain V6 scorer + joint FT at low CBR (0.065).

Starts from existing Stage 3 checkpoint (noise-robust SNN encoder/decoder)
and retrains only the scorer (Stage 4) and joint fine-tune (Stage 5) at
target CBR=0.065 to match DHF-JSCC's compression ratio.

Key changes vs original train_kitti_image_v6.py:
  - target_cbr=0.065 (was 0.5)
  - Stage 4: stronger rate enforcement (20× vs 10×)
  - Stage 5: masking always enabled (was stochastic 20-50%)
  - Saves to snapshots_kitti_v6_lowcbr/

Usage:
    python train/train_kitti_image_v6_lowcbr.py
    python train/train_kitti_image_v6_lowcbr.py --target-cbr 0.03  # even lower
"""

import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
import sys, os, json, random, math, glob, argparse
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms as T
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.spike_image_codec_v6 import SpikeImageCodecV6

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CROP_H, CROP_W = 256, 512
SRC_SNAP = './snapshots_kitti_v6'  # Source: Stage 3 checkpoint


def compute_ssim_torch(x, y, window_size=11, data_range=1.0):
    C = x.size(1); sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=x.device) - window_size // 2
    g = torch.exp(-coords**2 / (2 * sigma**2)); g /= g.sum()
    window = (g.unsqueeze(1)*g.unsqueeze(0)).unsqueeze(0).unsqueeze(0).expand(C,1,window_size,window_size)
    pad = window_size // 2
    mu_x = F.conv2d(x, window, padding=pad, groups=C)
    mu_y = F.conv2d(y, window, padding=pad, groups=C)
    sxx = F.conv2d(x*x, window, padding=pad, groups=C) - mu_x**2
    syy = F.conv2d(y*y, window, padding=pad, groups=C) - mu_y**2
    sxy = F.conv2d(x*y, window, padding=pad, groups=C) - mu_x*mu_y
    C1, C2 = (0.01*data_range)**2, (0.03*data_range)**2
    return ((2*mu_x*mu_y+C1)*(2*sxy+C2)/((mu_x**2+mu_y**2+C1)*(sxx+syy+C2))).mean()


class ImageFolderFlat(Dataset):
    def __init__(self, images, transform=None):
        self.images = images; self.transform = transform
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        return self.transform(img) if self.transform else img


def get_datasets():
    tf_train = T.Compose([T.RandomCrop((CROP_H,CROP_W),pad_if_needed=True,padding_mode='reflect'),
        T.RandomHorizontalFlip(),T.ColorJitter(brightness=0.1,contrast=0.1),T.ToTensor()])
    tf_test = T.Compose([T.CenterCrop((CROP_H,CROP_W)),T.ToTensor()])
    train_imgs, test_imgs = [], []
    d = './InStereo2K'
    for sp, lst, tf in [('train', train_imgs, tf_train), ('test', test_imgs, tf_test)]:
        dd = os.path.join(d, sp)
        if os.path.exists(dd):
            imgs = sorted([os.path.join(dd,s,'left.png') for s in os.listdir(dd)
                          if os.path.isfile(os.path.join(dd,s,'left.png'))])
            lst.extend(imgs)
            print(f"  InStereo2K {sp}: {len(imgs)}")
    kitti_imgs = []
    kd = './data/kitti_stereo_2015/training/image_2'
    if os.path.exists(kd):
        kitti_imgs = sorted(glob.glob(os.path.join(kd,'*.png')))
        print(f"  KITTI: {len(kitti_imgs)}")
    return (ImageFolderFlat(train_imgs, tf_train) if train_imgs else None,
            ImageFolderFlat(test_imgs, tf_test) if test_imgs else None,
            ImageFolderFlat(kitti_imgs, tf_test) if kitti_imgs else None)


def recon_loss(x, xh):
    xc = xh.clamp(0,1); mse = F.mse_loss(xc,x)
    ms = 0; a, b = x, xc
    for i, w in enumerate([0.4,0.3,0.3]):
        ms += w*compute_ssim_torch(a,b)
        if i < 2: a = F.avg_pool2d(a,2); b = F.avg_pool2d(b,2)
    return 0.85*mse+0.15*(1-ms), mse, compute_ssim_torch(x,xc)


def psnr(x, xh):
    m = F.mse_loss(xh,x).item()
    return 100.0 if m==0 else 10*math.log10(1.0/m)


def evaluate(model, loader, ch='awgn', noise=0.0, masking=True, cbr=None, max_b=None):
    model.eval(); ps, n = 0, 0
    with torch.no_grad():
        for i, imgs in enumerate(loader):
            if max_b and i >= max_b: break
            if isinstance(imgs,(list,tuple)): imgs = imgs[0]
            imgs = imgs.to(device)
            recon, info = model(imgs, noise, ch, masking, cbr)
            ps += psnr(imgs, recon.clamp(0,1))*imgs.size(0); n += imgs.size(0)
    return {'psnr': round(ps/n, 3), 'n': n}


def load_stage3(model):
    """Load Stage 3 checkpoint from original V6 training."""
    cks = sorted(glob.glob(os.path.join(SRC_SNAP, 'stage3_best_*.pth')),
                 key=lambda x: float(x.split('_')[-1].replace('.pth','')))
    if cks:
        state = torch.load(cks[-1], map_location=device, weights_only=False)
        ms = model.state_dict()
        filt = {k:v for k,v in state.items() if k in ms and v.shape==ms[k].shape}
        model.load_state_dict(filt, strict=False)
        print(f"  Loaded S3 checkpoint: {cks[-1]} ({len(filt)}/{len(ms)} params)")
        return True
    print(f"  WARNING: No Stage 3 checkpoint found in {SRC_SNAP}")
    return False


def train_s4_lowcbr(model, tl, vl, epochs=40, lr=3e-4, cbr=0.065, snap_dir='./snapshots_kitti_v6_lowcbr'):
    """Stage 4: Scorer training at low CBR.

    Key differences from original:
    - Rate loss weight: 20× (was 10×) — stronger enforcement at extreme compression
    - Evaluates at target CBR (not full rate)
    """
    print(f"\n{'='*60}")
    print(f"  S4: Scorer (low CBR={cbr})")
    print(f"{'='*60}")
    # Only train scorer
    for n, p in model.named_parameters():
        p.requires_grad = 'scorer' in n
    opt = optim.AdamW(model.scorer.parameters(), lr=lr, weight_decay=1e-5)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    best = 0
    os.makedirs(snap_dir, exist_ok=True)

    for ep in range(1, epochs+1):
        model.train(); model.encoder.eval(); model.snn_encoder.eval()
        el, en = 0, 0
        for imgs in tl:
            if isinstance(imgs,(list,tuple)): imgs = imgs[0]
            imgs = imgs.to(device)
            ch = 'awgn' if random.random()<0.5 else 'bsc'
            np_ = (random.choice([1,4,7,10,13,19]) if ch=='awgn'
                   else random.choice([0.0,0.05,0.10,0.15,0.20,0.30]))
            r, info = model(imgs, np_, ch, True)
            lr_, _, _ = recon_loss(imgs, r)
            lrate = sum((im.mean()-cbr)**2 for im in info['importance'] if im is not None) / 4
            ldiv = model.scorer.compute_diversity_loss(info['multi_spikes'])
            loss = lr_ + 20*lrate + 0.05*ldiv  # 20× rate weight (was 10×)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.scorer.parameters(), 1.0); opt.step()
            el += loss.item()*imgs.size(0); en += imgs.size(0)
        sch.step()

        if ep%5==0 or ep==epochs:
            m_masked = evaluate(model, vl, 'awgn', 7.0, True, max_b=20)
            m_full = evaluate(model, vl, 'awgn', 7.0, False, max_b=20)
            # Also check actual CBR
            model.eval()
            with torch.no_grad():
                for imgs in vl:
                    if isinstance(imgs,(list,tuple)): imgs = imgs[0]
                    _, info = model(imgs.to(device), 7.0, 'awgn', True)
                    actual_cbrs = info['actual_cbrs']
                    break
            avg_cbr = sum(actual_cbrs) / len(actual_cbrs)
            print(f"  E{ep:02d}: Full={m_full['psnr']:.2f}, Masked={m_masked['psnr']:.2f}, "
                  f"AvgCBR={avg_cbr:.3f} (target={cbr})")
            if m_masked['psnr'] > best:
                best = m_masked['psnr']
                torch.save(model.state_dict(),
                           os.path.join(snap_dir, f'stage4_cbr{cbr}_best_{best:.2f}.pth'))
                print(f"    ✓ Saved")

    for p in model.parameters():
        p.requires_grad = True
    return best


def train_s5_lowcbr(model, tl, vl, epochs=30, lr=1e-5, cbr=0.065, snap_dir='./snapshots_kitti_v6_lowcbr'):
    """Stage 5: Joint FT at low CBR.

    Key differences from original:
    - Masking ALWAYS enabled (was stochastic 20-50%)
    - Rate loss weight: 10× (was 5×)
    """
    print(f"\n{'='*60}")
    print(f"  S5: Joint FT (low CBR={cbr})")
    print(f"{'='*60}")
    for p in model.parameters():
        p.requires_grad = True
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-7)
    best = 0
    os.makedirs(snap_dir, exist_ok=True)

    for ep in range(1, epochs+1):
        model.train(); el, en = 0, 0
        for imgs in tl:
            if isinstance(imgs,(list,tuple)): imgs = imgs[0]
            imgs = imgs.to(device)
            ch = 'awgn' if random.random()<0.5 else 'bsc'
            np_ = (random.choice([1,4,7,10,13,19]) if ch=='awgn'
                   else random.choice([0.0,0.05,0.10,0.15,0.20,0.30]))
            # Masking ALWAYS enabled at low CBR
            r, info = model(imgs, np_, ch, True)
            lr_, _, _ = recon_loss(imgs, r)
            lrate = sum((im.mean()-cbr)**2 for im in info['importance']
                       if im is not None) / 4
            loss = lr_ + 10*lrate  # 10× rate weight (was 5×)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()
            el += loss.item()*imgs.size(0); en += imgs.size(0)
        sch.step()

        if ep%3==0 or ep==epochs:
            mm = evaluate(model, vl, 'awgn', 7.0, True, max_b=20)
            mc = evaluate(model, vl, 'awgn', 100.0, True, max_b=20)
            print(f"  E{ep:02d}: Noisy={mm['psnr']:.2f}, Clean={mc['psnr']:.2f}")
            if mm['psnr'] > best:
                best = mm['psnr']
                torch.save(model.state_dict(),
                           os.path.join(snap_dir, f'stage5_cbr{cbr}_best_{best:.2f}.pth'))
                print(f"    ✓ Saved")

    return best


def full_eval_lowcbr(model, loader, name, cbr, snap_dir):
    """Full evaluation at target CBR and nearby CBRs."""
    print(f"\n{'='*60}")
    print(f"  EVAL: {name} (target CBR={cbr})")
    print(f"{'='*60}")
    results = {'awgn': {}, 'bsc': {}}

    # Test at target CBR and nearby
    cbr_list = [cbr, 0.03, 0.05, 0.065, 0.10, 0.125, 0.25, 0.5, 1.0]
    cbr_list = sorted(set(cbr_list))

    for snr in [0, 1, 2.5, 5, 7, 10, 100]:
        results['awgn'][str(snr)] = {}
        for c in cbr_list:
            ml = evaluate(model, loader, 'awgn', snr, c < 1.0, c)
            label = 'Clean' if snr >= 100 else f'CSNR={snr}dB'
            print(f"  AWGN {label} CBR={c:.3f}: {ml['psnr']:.2f}")
            results['awgn'][str(snr)][str(c)] = ml

    for ber in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
        results['bsc'][str(ber)] = {}
        for c in cbr_list:
            ml = evaluate(model, loader, 'bsc', ber, c < 1.0, c)
            print(f"  BSC BER={ber:.2f} CBR={c:.3f}: {ml['psnr']:.2f}")
            results['bsc'][str(ber)][str(c)] = ml

    os.makedirs('eval/seed_results', exist_ok=True)
    fn = f'eval/seed_results/v6_lowcbr_{name}.json'
    with open(fn, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved {fn}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-cbr', type=float, default=0.065)
    parser.add_argument('--eval-only', action='store_true')
    args = parser.parse_args()

    cbr = args.target_cbr
    snap_dir = f'./snapshots_kitti_v6_lowcbr'
    print(f"Device: {device}")
    print(f"Target CBR: {cbr}")
    print(f"Snapshots: {snap_dir}")

    train_ds, test_ds, kitti_ds = get_datasets()
    tl = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    vl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    kl = DataLoader(kitti_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True) if kitti_ds else None

    # Create model with target CBR
    model = SpikeImageCodecV6(target_cbr=cbr, use_sdsa=True, bw_mode='extra').to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    if args.eval_only:
        # Load best low-CBR checkpoint
        cks = sorted(glob.glob(os.path.join(snap_dir, 'stage5_cbr*_best_*.pth')),
                     key=lambda x: float(x.split('_')[-1].replace('.pth','')))
        if not cks:
            cks = sorted(glob.glob(os.path.join(snap_dir, 'stage4_cbr*_best_*.pth')),
                         key=lambda x: float(x.split('_')[-1].replace('.pth','')))
        if cks:
            model.load_state_dict(torch.load(cks[-1], map_location=device, weights_only=False))
            print(f"  Loaded {cks[-1]}")
        full_eval_lowcbr(model, vl, 'instereo2k', cbr, snap_dir)
        if kl:
            full_eval_lowcbr(model, kl, 'kitti', cbr, snap_dir)
        return

    # Load Stage 3 checkpoint from original V6
    load_stage3(model)

    # Stage 4: Scorer at low CBR
    train_s4_lowcbr(model, tl, vl, epochs=40, lr=3e-4, cbr=cbr, snap_dir=snap_dir)

    # Stage 5: Joint FT at low CBR
    # Load best S4
    cks = sorted(glob.glob(os.path.join(snap_dir, f'stage4_cbr{cbr}_best_*.pth')),
                 key=lambda x: float(x.split('_')[-1].replace('.pth','')))
    if cks:
        model.load_state_dict(torch.load(cks[-1], map_location=device, weights_only=False))
        print(f"  Loaded S4: {cks[-1]}")
    train_s5_lowcbr(model, tl, vl, epochs=30, lr=1e-5, cbr=cbr, snap_dir=snap_dir)

    # Full evaluation
    cks = sorted(glob.glob(os.path.join(snap_dir, f'stage5_cbr{cbr}_best_*.pth')),
                 key=lambda x: float(x.split('_')[-1].replace('.pth','')))
    if cks:
        model.load_state_dict(torch.load(cks[-1], map_location=device, weights_only=False))
    full_eval_lowcbr(model, vl, 'instereo2k', cbr, snap_dir)
    if kl:
        full_eval_lowcbr(model, kl, 'kitti', cbr, snap_dir)


if __name__ == '__main__':
    main()
