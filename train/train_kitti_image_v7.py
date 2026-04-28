#!/usr/bin/env python3
"""Train SpikeAdapt-SC V7: V6 + Hyperprior + optional Multi-bit encoding.

Stage B: Add hyperprior side-channel (from Stage A low-CBR checkpoint)
  B1 (30 ep): Train hyperprior enc/dec only (frozen main codec)
  B2 (30 ep): Joint fine-tune all params with noise
  B3 (20 ep): Scorer re-tuning with hyperprior bandwidth

Stage C: Multi-bit membrane encoding (from Stage B checkpoint)
  C1 (40 ep): Replace ternary with multi-bit, fine-tune encoders

The hyperprior architecture transfers to classification and detection:
  - Classification: side-channel carries "what class" hints under heavy noise
  - Detection: side-channel carries small-object location hints for DOTA

Usage:
    python train/train_kitti_image_v7.py --stage b   # Stage B only
    python train/train_kitti_image_v7.py --stage c   # Stage C only
    python train/train_kitti_image_v7.py --stage all  # Both B then C
"""

import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
import sys, os, json, random, math, glob, argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.spike_image_codec_v7 import SpikeImageCodecV7

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CROP_H, CROP_W = 256, 512
SNAP_DIR = './snapshots_kitti_v7'
STAGE_A_DIR = './snapshots_kitti_v6_lowcbr'


# =============================================================================
# Shared utilities (same as V6 training)
# =============================================================================

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

def evaluate(model, loader, ch='awgn', noise=0.0, masking=True, cbr=None,
             use_hyper=True, max_b=None):
    model.eval(); ps, n = 0, 0
    with torch.no_grad():
        for i, imgs in enumerate(loader):
            if max_b and i >= max_b: break
            if isinstance(imgs,(list,tuple)): imgs = imgs[0]
            imgs = imgs.to(device)
            recon, info = model(imgs, noise, ch, masking, cbr, use_hyperprior=use_hyper)
            ps += psnr(imgs, recon.clamp(0,1))*imgs.size(0); n += imgs.size(0)
    return {'psnr': round(ps/n, 3), 'n': n}


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
    for sp, lst in [('train', train_imgs), ('test', test_imgs)]:
        dd = os.path.join(d, sp)
        if os.path.exists(dd):
            imgs = sorted([os.path.join(dd,s,'left.png') for s in os.listdir(dd)
                          if os.path.isfile(os.path.join(dd,s,'left.png'))])
            lst.extend(imgs); print(f"  InStereo2K {sp}: {len(imgs)}")
    kitti_imgs = []
    kd = './data/kitti_stereo_2015/training/image_2'
    if os.path.exists(kd):
        kitti_imgs = sorted(glob.glob(os.path.join(kd,'*.png'))); print(f"  KITTI: {len(kitti_imgs)}")
    return (ImageFolderFlat(train_imgs, tf_train) if train_imgs else None,
            ImageFolderFlat(test_imgs, tf_test) if test_imgs else None,
            ImageFolderFlat(kitti_imgs, tf_test) if kitti_imgs else None)


# =============================================================================
# Load Stage A checkpoint into V7 model
# =============================================================================

def load_stage_a_into_v7(model):
    """Load Stage A (low-CBR V6) checkpoint into V7 model.
    V7 has all V6 params + new hyperprior params (randomly initialized).
    """
    # Try S5 first, then S4
    for tag in ['stage5_cbr0.065', 'stage4_cbr0.065']:
        cks = sorted(glob.glob(os.path.join(STAGE_A_DIR, f'{tag}_best_*.pth')),
                     key=lambda x: float(x.split('_')[-1].replace('.pth','')))
        if cks:
            state = torch.load(cks[-1], map_location=device, weights_only=False)
            ms = model.state_dict()
            # Load matching V6 params, skip new V7 params (hyperprior, etc.)
            filt = {k: v for k, v in state.items() if k in ms and v.shape == ms[k].shape}
            model.load_state_dict(filt, strict=False)
            print(f"  Loaded Stage A: {cks[-1]} ({len(filt)}/{len(ms)} params)")
            new_params = len(ms) - len(filt)
            print(f"  New V7 params (randomly init): {new_params}")
            return True
    print(f"  WARNING: No Stage A checkpoint in {STAGE_A_DIR}")
    return False


# =============================================================================
# Stage B: Hyperprior training
# =============================================================================

def train_b1_hyperprior(model, tl, vl, epochs=30, lr=1e-3, cbr=0.065):
    """B1: Train hyperprior encoder/decoder only (frozen main codec)."""
    print(f"\n{'='*60}\n  B1: Hyperprior training (frozen main codec)\n{'='*60}")

    # Freeze everything except hyperprior and hint fusion
    for n, p in model.named_parameters():
        p.requires_grad = ('hyper_' in n or 'hint_fuse' in n)

    hyper_params = [p for n, p in model.named_parameters() if p.requires_grad]
    print(f"  Trainable params: {sum(p.numel() for p in hyper_params):,}")
    opt = optim.AdamW(hyper_params, lr=lr, weight_decay=1e-5)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    best = 0; os.makedirs(SNAP_DIR, exist_ok=True)

    for ep in range(1, epochs+1):
        model.train()
        # Keep main codec components in eval mode
        model.encoder.eval(); model.snn_encoder.eval(); model.decoder.eval()
        el, en = 0, 0
        for imgs in tl:
            if isinstance(imgs,(list,tuple)): imgs = imgs[0]
            imgs = imgs.to(device)
            ch = 'awgn' if random.random()<0.5 else 'bsc'
            np_ = (random.choice([1,4,7,10,13,19]) if ch=='awgn'
                   else random.choice([0.0,0.05,0.10,0.15,0.20,0.30]))
            r, info = model(imgs, np_, ch, True, use_hyperprior=True)
            loss, _, _ = recon_loss(imgs, r)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(hyper_params, 1.0); opt.step()
            el += loss.item()*imgs.size(0); en += imgs.size(0)
        sch.step()

        if ep%5==0 or ep==epochs:
            m_hyper = evaluate(model, vl, 'awgn', 7.0, True, use_hyper=True, max_b=20)
            m_no = evaluate(model, vl, 'awgn', 7.0, True, use_hyper=False, max_b=20)
            print(f"  E{ep:02d}: +Hyper={m_hyper['psnr']:.2f}, NoHyper={m_no['psnr']:.2f}, "
                  f"Δ={m_hyper['psnr']-m_no['psnr']:+.2f}dB")
            if m_hyper['psnr'] > best:
                best = m_hyper['psnr']
                torch.save(model.state_dict(),
                           os.path.join(SNAP_DIR, f'b1_hyper_best_{best:.2f}.pth'))

    for p in model.parameters(): p.requires_grad = True
    return best


def train_b2_joint(model, tl, vl, epochs=30, lr=5e-5, cbr=0.065):
    """B2: Joint fine-tune all params with noise + hyperprior."""
    print(f"\n{'='*60}\n  B2: Joint FT with hyperprior\n{'='*60}")
    for p in model.parameters(): p.requires_grad = True
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-7)
    best = 0

    for ep in range(1, epochs+1):
        model.train(); el, en = 0, 0
        for imgs in tl:
            if isinstance(imgs,(list,tuple)): imgs = imgs[0]
            imgs = imgs.to(device)
            ch = 'awgn' if random.random()<0.5 else 'bsc'
            np_ = (random.choice([1,4,7,10,13,19]) if ch=='awgn'
                   else random.choice([0.0,0.05,0.10,0.15,0.20,0.30]))
            r, info = model(imgs, np_, ch, True, use_hyperprior=True)
            lr_, _, _ = recon_loss(imgs, r)
            lrate = sum((im.mean()-cbr)**2 for im in info['importance']
                       if im is not None) / 4
            loss = lr_ + 15*lrate
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()
            el += loss.item()*imgs.size(0); en += imgs.size(0)
        sch.step()

        if ep%5==0 or ep==epochs:
            m = evaluate(model, vl, 'awgn', 7.0, True, use_hyper=True, max_b=20)
            mc = evaluate(model, vl, 'awgn', 100.0, True, use_hyper=True, max_b=20)
            print(f"  E{ep:02d}: Noisy={m['psnr']:.2f}, Clean={mc['psnr']:.2f}")
            if m['psnr'] > best:
                best = m['psnr']
                torch.save(model.state_dict(),
                           os.path.join(SNAP_DIR, f'b2_joint_best_{best:.2f}.pth'))
    return best


def train_b3_scorer(model, tl, vl, epochs=20, lr=1e-4, cbr=0.065):
    """B3: Scorer re-tuning accounting for hyperprior bandwidth."""
    print(f"\n{'='*60}\n  B3: Scorer re-tune with hyperprior\n{'='*60}")
    for n, p in model.named_parameters():
        p.requires_grad = 'scorer' in n
    opt = optim.AdamW(model.scorer.parameters(), lr=lr, weight_decay=1e-5)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    best = 0

    for ep in range(1, epochs+1):
        model.train(); model.encoder.eval(); model.snn_encoder.eval()
        el, en = 0, 0
        for imgs in tl:
            if isinstance(imgs,(list,tuple)): imgs = imgs[0]
            imgs = imgs.to(device)
            ch = 'awgn' if random.random()<0.5 else 'bsc'
            np_ = (random.choice([1,4,7,10,13,19]) if ch=='awgn'
                   else random.choice([0.0,0.05,0.10,0.15,0.20,0.30]))
            r, info = model(imgs, np_, ch, True, use_hyperprior=True)
            lr_, _, _ = recon_loss(imgs, r)
            # Account for hyperprior BW: slightly lower target for main spikes
            effective_cbr = cbr * 0.85  # Reserve 15% for hyperprior
            lrate = sum((im.mean()-effective_cbr)**2 for im in info['importance']
                       if im is not None) / 4
            ldiv = model.scorer.compute_diversity_loss(info['multi_spikes'])
            loss = lr_ + 20*lrate + 0.05*ldiv
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.scorer.parameters(), 1.0); opt.step()
            el += loss.item()*imgs.size(0); en += imgs.size(0)
        sch.step()

        if ep%5==0 or ep==epochs:
            m = evaluate(model, vl, 'awgn', 7.0, True, use_hyper=True, max_b=20)
            print(f"  E{ep:02d}: PSNR={m['psnr']:.2f}")
            if m['psnr'] > best:
                best = m['psnr']
                torch.save(model.state_dict(),
                           os.path.join(SNAP_DIR, f'b3_scorer_best_{best:.2f}.pth'))
    for p in model.parameters(): p.requires_grad = True
    return best


# =============================================================================
# Stage C: Multi-bit encoding
# =============================================================================

def train_c1_multibit(model, tl, vl, epochs=40, lr=5e-5, cbr=0.065):
    """C1: Fine-tune with multi-bit encoding from Stage B checkpoint."""
    print(f"\n{'='*60}\n  C1: Multi-bit encoding\n{'='*60}")
    for p in model.parameters(): p.requires_grad = True
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-7)
    best = 0

    for ep in range(1, epochs+1):
        model.train(); el, en = 0, 0
        for imgs in tl:
            if isinstance(imgs,(list,tuple)): imgs = imgs[0]
            imgs = imgs.to(device)
            ch = 'awgn' if random.random()<0.5 else 'bsc'
            np_ = (random.choice([1,4,7,10,13,19]) if ch=='awgn'
                   else random.choice([0.0,0.05,0.10,0.15,0.20,0.30]))
            r, info = model(imgs, np_, ch, True, use_hyperprior=True)
            lr_, _, _ = recon_loss(imgs, r)
            lrate = sum((im.mean()-cbr)**2 for im in info['importance']
                       if im is not None) / 4
            loss = lr_ + 15*lrate
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()
            el += loss.item()*imgs.size(0); en += imgs.size(0)
        sch.step()

        if ep%5==0 or ep==epochs:
            m = evaluate(model, vl, 'awgn', 7.0, True, use_hyper=True, max_b=20)
            mc = evaluate(model, vl, 'awgn', 100.0, True, use_hyper=True, max_b=20)
            print(f"  E{ep:02d}: Noisy={m['psnr']:.2f}, Clean={mc['psnr']:.2f}")
            if m['psnr'] > best:
                best = m['psnr']
                torch.save(model.state_dict(),
                           os.path.join(SNAP_DIR, f'c1_multibit_best_{best:.2f}.pth'))
    return best


# =============================================================================
# Full evaluation
# =============================================================================

def full_eval_v7(model, loader, name, cbr):
    """Comprehensive eval: with and without hyperprior."""
    print(f"\n{'='*60}\n  EVAL: {name}\n{'='*60}")
    results = {'with_hyper': {'awgn': {}, 'bsc': {}},
               'without_hyper': {'awgn': {}, 'bsc': {}}}

    cbr_list = sorted(set([cbr, 0.03, 0.05, 0.065, 0.10, 0.125, 0.25, 0.5, 1.0]))

    for use_hyper, tag in [(True, 'with_hyper'), (False, 'without_hyper')]:
        for snr in [0, 1, 5, 7, 10, 100]:
            results[tag]['awgn'][str(snr)] = {}
            for c in cbr_list:
                ml = evaluate(model, loader, 'awgn', snr, c < 1.0, c, use_hyper)
                label = 'Clean' if snr >= 100 else f'CSNR={snr}dB'
                hyp = '+H' if use_hyper else '-H'
                print(f"  [{hyp}] AWGN {label} CBR={c:.3f}: {ml['psnr']:.2f}")
                results[tag]['awgn'][str(snr)][str(c)] = ml

        for ber in [0.0, 0.10, 0.20, 0.30]:
            results[tag]['bsc'][str(ber)] = {}
            for c in cbr_list:
                ml = evaluate(model, loader, 'bsc', ber, c < 1.0, c, use_hyper)
                hyp = '+H' if use_hyper else '-H'
                print(f"  [{hyp}] BSC BER={ber:.2f} CBR={c:.3f}: {ml['psnr']:.2f}")
                results[tag]['bsc'][str(ber)][str(c)] = ml

    os.makedirs('eval/seed_results', exist_ok=True)
    fn = f'eval/seed_results/v7_{model.encoding_mode}_{name}.json'
    with open(fn, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved {fn}")
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='all', choices=['b', 'c', 'all', 'eval'])
    parser.add_argument('--target-cbr', type=float, default=0.065)
    parser.add_argument('--n-levels', type=int, default=8, help='Multi-bit levels (Stage C)')
    parser.add_argument('--eval-only', action='store_true')
    args = parser.parse_args()

    cbr = args.target_cbr
    print(f"Device: {device}")
    print(f"Target CBR: {cbr}, Multi-bit levels: {args.n_levels}")

    train_ds, test_ds, kitti_ds = get_datasets()
    tl = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4,
                    pin_memory=True, drop_last=True)
    vl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    kl = (DataLoader(kitti_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
          if kitti_ds else None)

    encoding_mode = 'multibit' if args.stage == 'c' else 'ternary'

    model = SpikeImageCodecV7(
        target_cbr=cbr, use_sdsa=True, bw_mode='extra',
        encoding_mode=encoding_mode, n_levels=args.n_levels,
    ).to(device)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Encoding: {encoding_mode}")

    if args.eval_only:
        # Load best checkpoint
        for tag in ['c1_multibit', 'b3_scorer', 'b2_joint', 'b1_hyper']:
            cks = sorted(glob.glob(os.path.join(SNAP_DIR, f'{tag}_best_*.pth')),
                         key=lambda x: float(x.split('_')[-1].replace('.pth','')))
            if cks:
                model.load_state_dict(torch.load(cks[-1], map_location=device,
                                                 weights_only=False), strict=False)
                print(f"  Loaded {cks[-1]}")
                break
        full_eval_v7(model, vl, 'instereo2k', cbr)
        if kl: full_eval_v7(model, kl, 'kitti', cbr)
        return

    # Stage B: Hyperprior
    if args.stage in ['b', 'all']:
        # Load Stage A into V7
        load_stage_a_into_v7(model)
        train_b1_hyperprior(model, tl, vl, 30, cbr=cbr)

        # Load best B1
        cks = sorted(glob.glob(os.path.join(SNAP_DIR, 'b1_hyper_best_*.pth')),
                     key=lambda x: float(x.split('_')[-1].replace('.pth','')))
        if cks: model.load_state_dict(torch.load(cks[-1], map_location=device,
                                                  weights_only=False), strict=False)
        train_b2_joint(model, tl, vl, 30, cbr=cbr)

        cks = sorted(glob.glob(os.path.join(SNAP_DIR, 'b2_joint_best_*.pth')),
                     key=lambda x: float(x.split('_')[-1].replace('.pth','')))
        if cks: model.load_state_dict(torch.load(cks[-1], map_location=device,
                                                  weights_only=False), strict=False)
        train_b3_scorer(model, tl, vl, 20, cbr=cbr)

        # Eval Stage B
        cks = sorted(glob.glob(os.path.join(SNAP_DIR, 'b3_scorer_best_*.pth')),
                     key=lambda x: float(x.split('_')[-1].replace('.pth','')))
        if cks: model.load_state_dict(torch.load(cks[-1], map_location=device,
                                                  weights_only=False), strict=False)
        full_eval_v7(model, vl, 'instereo2k', cbr)
        if kl: full_eval_v7(model, kl, 'kitti', cbr)

    # Stage C: Multi-bit
    if args.stage in ['c', 'all']:
        # Create multi-bit model
        model_c = SpikeImageCodecV7(
            target_cbr=cbr, use_sdsa=True, bw_mode='extra',
            encoding_mode='multibit', n_levels=args.n_levels,
        ).to(device)

        # Load Stage B checkpoint into multi-bit model (shared params)
        for tag in ['b3_scorer', 'b2_joint']:
            cks = sorted(glob.glob(os.path.join(SNAP_DIR, f'{tag}_best_*.pth')),
                         key=lambda x: float(x.split('_')[-1].replace('.pth','')))
            if cks:
                state = torch.load(cks[-1], map_location=device, weights_only=False)
                ms = model_c.state_dict()
                filt = {k: v for k, v in state.items() if k in ms and v.shape == ms[k].shape}
                model_c.load_state_dict(filt, strict=False)
                print(f"  Loaded Stage B: {cks[-1]} ({len(filt)}/{len(ms)} params)")
                break

        train_c1_multibit(model_c, tl, vl, 40, cbr=cbr)

        cks = sorted(glob.glob(os.path.join(SNAP_DIR, 'c1_multibit_best_*.pth')),
                     key=lambda x: float(x.split('_')[-1].replace('.pth','')))
        if cks: model_c.load_state_dict(torch.load(cks[-1], map_location=device,
                                                    weights_only=False), strict=False)
        full_eval_v7(model_c, vl, 'instereo2k', cbr)
        if kl: full_eval_v7(model_c, kl, 'kitti', cbr)


if __name__ == '__main__':
    main()
