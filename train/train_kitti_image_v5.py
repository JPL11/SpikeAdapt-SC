#!/usr/bin/env python3
"""Train SpikeAdapt-SC V5: V4 + U-Net Skips + Wider Bottleneck.

5-stage pipeline (same as V4 but with longer Stage 5):
  S1: Float teacher (30 epochs)
  S2: SNN student + KD (60 epochs)
  S3: Channel noise (50 epochs — longer for wider model)
  S4: Scorer (40 epochs)
  S5: Joint fine-tuning (30 epochs — longer, focus on full-BW first)
"""

import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
import sys, os, json, random, math, glob
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms as T
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.spike_image_codec_v5 import SpikeImageCodecV5, FloatTeacherV5

# SSIM
def compute_ssim_torch(x, y, window_size=11, data_range=1.0):
    C = x.size(1); sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32, device=x.device) - window_size // 2
    g = torch.exp(-coords**2 / (2 * sigma**2)); g = g / g.sum()
    window = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0).expand(C, 1, window_size, window_size)
    pad = window_size // 2
    mu_x = F.conv2d(x, window, padding=pad, groups=C)
    mu_y = F.conv2d(y, window, padding=pad, groups=C)
    sigma_x_sq = F.conv2d(x*x, window, padding=pad, groups=C) - mu_x**2
    sigma_y_sq = F.conv2d(y*y, window, padding=pad, groups=C) - mu_y**2
    sigma_xy = F.conv2d(x*y, window, padding=pad, groups=C) - mu_x*mu_y
    C1, C2 = (0.01*data_range)**2, (0.03*data_range)**2
    return ((2*mu_x*mu_y+C1)*(2*sigma_xy+C2) / ((mu_x**2+mu_y**2+C1)*(sigma_x_sq+sigma_y_sq+C2))).mean()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CROP_H, CROP_W = 256, 512

class ImageFolderFlat(Dataset):
    def __init__(self, images, transform=None):
        self.images = images; self.transform = transform
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        return self.transform(img) if self.transform else img

def get_datasets(instereo_root='./InStereo2K', data_root='./data'):
    tf_train = T.Compose([T.RandomCrop((CROP_H, CROP_W), pad_if_needed=True, padding_mode='reflect'),
        T.RandomHorizontalFlip(), T.ColorJitter(brightness=0.1, contrast=0.1), T.ToTensor()])
    tf_test = T.Compose([T.CenterCrop((CROP_H, CROP_W)), T.ToTensor()])
    train_sets, test_sets = [], []
    for split, sets, tf in [('train', train_sets, tf_train), ('test', test_sets, tf_test)]:
        d = os.path.join(instereo_root, split)
        if os.path.exists(d):
            imgs = [os.path.join(d, s, 'left.png') for s in sorted(os.listdir(d))
                    if os.path.isfile(os.path.join(d, s, 'left.png'))]
            if imgs: sets.append(ImageFolderFlat(imgs, tf)); print(f"  InStereo2K {split}: {len(imgs)}")
    kitti_ds = None
    kd = os.path.join(data_root, 'kitti_stereo_2015', 'training', 'image_2')
    if os.path.exists(kd):
        ki = sorted(glob.glob(os.path.join(kd, '*.png')))
        if ki: kitti_ds = ImageFolderFlat(ki, tf_test); print(f"  KITTI: {len(ki)}")
    train_ds = ConcatDataset(train_sets) if len(train_sets) > 1 else train_sets[0]
    test_ds = ConcatDataset(test_sets) if len(test_sets) > 1 else test_sets[0]
    return train_ds, test_ds, kitti_ds

def compute_ms_ssim(x, y):
    ms = 0
    for i, w in enumerate([0.4, 0.3, 0.3]):
        ms += w * compute_ssim_torch(x, y)
        if i < 2: x = F.avg_pool2d(x, 2); y = F.avg_pool2d(y, 2)
    return ms

def recon_loss(x, x_hat):
    xc = x_hat.clamp(0, 1); mse = F.mse_loss(xc, x)
    ms = compute_ms_ssim(x, xc)
    return 0.85*mse + 0.15*(1-ms), mse, compute_ssim_torch(x, xc)

def compute_psnr(x, xh):
    mse = F.mse_loss(xh, x).item()
    return 100.0 if mse == 0 else 10*math.log10(1.0/mse)

def evaluate(model, loader, channel='awgn', noise=0.0, masking=True, cbr=None, max_b=None):
    model.eval(); ps, ss, cs, n = 0, 0, 0, 0
    with torch.no_grad():
        for i, imgs in enumerate(loader):
            if max_b and i >= max_b: break
            if isinstance(imgs, (list, tuple)): imgs = imgs[0]
            imgs = imgs.to(device)
            recon, info = model(imgs, noise, channel, masking, cbr)
            recon = recon.clamp(0, 1)
            ps += compute_psnr(imgs, recon)*imgs.size(0)
            cs += info['actual_cbr']*imgs.size(0)
            ss += compute_ssim_torch(imgs, recon).item()*imgs.size(0)
            n += imgs.size(0)
    return {'psnr': round(ps/n, 3), 'ms_ssim': round(ss/n, 5), 'cbr': round(cs/n, 4), 'n': n}

def evaluate_random_mask(model, loader, channel, noise, cbr, max_b=None):
    model.eval(); ps, ss, n = 0, 0, 0
    with torch.no_grad():
        for i, imgs in enumerate(loader):
            if max_b and i >= max_b: break
            if isinstance(imgs, (list, tuple)): imgs = imgs[0]
            imgs = imgs.to(device)
            feat, skips = model.img_encoder(imgs); spikes = model.snn_encoder(feat)
            B, C, H, W = spikes[0].shape; k = max(1, int(cbr * H * W))
            flat = torch.rand(B, 1, H, W, device=device).view(B, -1)
            _, idx = flat.topk(k, dim=1)
            mask = torch.zeros_like(flat); mask.scatter_(1, idx, 1.0); mask = mask.view(B, 1, H, W)
            masked = [s*mask for s in spikes]
            recv = [model.awgn(s, noise) if channel=='awgn' else model.bsc(s, noise) for s in masked]
            feat_r = model.snn_decoder(recv)
            recon = model.img_decoder(feat_r, skips if model.use_skips else None).clamp(0, 1)
            ps += compute_psnr(imgs, recon)*imgs.size(0)
            ss += compute_ssim_torch(imgs, recon).item()*imgs.size(0); n += imgs.size(0)
    return {'psnr': round(ps/n, 3), 'ms_ssim': round(ss/n, 5)}

SNAP_DIR = './snapshots_kitti_v5'
def save_ckpt(model, stage, psnr, prefix=''):
    os.makedirs(SNAP_DIR, exist_ok=True)
    p = os.path.join(SNAP_DIR, f'{prefix}{stage}_best_{psnr:.2f}.pth')
    torch.save(model.state_dict(), p); print(f"  ✓ Saved {p}")

def load_ckpt(model, stage, prefix=''):
    cks = sorted(glob.glob(os.path.join(SNAP_DIR, f'{prefix}{stage}_best_*.pth')),
                 key=lambda x: float(x.split('_')[-1].replace('.pth', '')))
    if cks:
        state = torch.load(cks[-1], map_location=device, weights_only=False)
        ms = model.state_dict()
        filt = {k: v for k, v in state.items() if k in ms and v.shape == ms[k].shape}
        model.load_state_dict(filt, strict=False)
        sk = set(state.keys()) - set(filt.keys())
        if sk: print(f"  (skipped {len(sk)} keys)")
        print(f"  Loaded {cks[-1]}"); return True
    return False

# S1: Teacher
def train_teacher(teacher, train_loader, test_loader, epochs=30, lr=1e-3):
    print(f"\n{'='*60}\n  S1: Float Teacher (U-Net)\n  Epochs: {epochs}\n{'='*60}")
    opt = optim.AdamW(teacher.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    best = 0
    for ep in range(1, epochs+1):
        teacher.train(); el, en = 0, 0
        for imgs in train_loader:
            if isinstance(imgs, (list, tuple)): imgs = imgs[0]
            imgs = imgs.to(device)
            recon, _ = teacher(imgs, 100.0, 'awgn')
            loss, mse, ssim = recon_loss(imgs, recon)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(teacher.parameters(), 1.0); opt.step()
            el += loss.item()*imgs.size(0); en += imgs.size(0)
        sched.step()
        if ep % 5 == 0 or ep == epochs:
            teacher.eval(); ps, n = 0, 0
            with torch.no_grad():
                for i, imgs in enumerate(test_loader):
                    if i >= 20: break
                    if isinstance(imgs, (list, tuple)): imgs = imgs[0]
                    imgs = imgs.to(device)
                    r, _ = teacher(imgs, 100.0, 'awgn')
                    ps += compute_psnr(imgs, r.clamp(0,1))*imgs.size(0); n += imgs.size(0)
            psnr = ps/n
            print(f"  E{ep:02d}: loss={el/en:.5f}, PSNR={psnr:.2f}dB")
            if psnr > best: best = psnr; save_ckpt(teacher, 'teacher', best, 'teacher_')
    return best

# S2: SNN Student + KD
def train_stage2_kd(model, teacher, train_loader, test_loader, epochs=60, lr=1e-3):
    print(f"\n{'='*60}\n  S2: SNN Student + KD\n  Epochs: {epochs}\n{'='*60}")
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    best = 0
    for ep in range(1, epochs+1):
        model.train(); el, en = 0, 0
        for imgs in train_loader:
            if isinstance(imgs, (list, tuple)): imgs = imgs[0]
            imgs = imgs.to(device)
            recon_s, info_s = model(imgs, 100.0, 'awgn', use_masking=False)
            loss_r, mse, ssim = recon_loss(imgs, recon_s)
            with torch.no_grad(): recon_t, info_t = teacher(imgs, 100.0, 'awgn')
            loss_kd_out = F.mse_loss(recon_s.clamp(0,1), recon_t.clamp(0,1))
            loss_kd_feat = F.mse_loss(info_s['feat'], info_t['feat'])
            loss_kd_dec = F.mse_loss(info_s['feat_recon'], info_t['feat_recon'])
            loss = loss_r + 0.5*loss_kd_out + 0.3*loss_kd_feat + 0.2*loss_kd_dec
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            el += loss.item()*imgs.size(0); en += imgs.size(0)
        sched.step()
        if ep % 5 == 0 or ep == epochs:
            m = evaluate(model, test_loader, 'awgn', 100.0, False, max_b=20)
            print(f"  E{ep:02d}: loss={el/en:.5f}, PSNR={m['psnr']:.2f}dB")
            if m['psnr'] > best: best = m['psnr']; save_ckpt(model, 'stage2', best)
    return best

# S3: Channel noise
def train_stage3(model, train_loader, test_loader, epochs=50, lr=5e-4):
    print(f"\n{'='*60}\n  S3: Channel noise\n  Epochs: {epochs}\n{'='*60}")
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    best = 0
    for ep in range(1, epochs+1):
        model.train(); el, en = 0, 0
        for imgs in train_loader:
            if isinstance(imgs, (list, tuple)): imgs = imgs[0]
            imgs = imgs.to(device)
            ch = 'awgn' if random.random()<0.5 else 'bsc'
            np_ = random.choice([1,4,7,10,13,19]) if ch=='awgn' else random.choice([0.0,0.05,0.10,0.15,0.20,0.30])
            recon, _ = model(imgs, np_, ch, use_masking=False)
            loss, mse, ssim = recon_loss(imgs, recon)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            el += loss.item()*imgs.size(0); en += imgs.size(0)
        sched.step()
        if ep % 5 == 0 or ep == epochs:
            ma = evaluate(model, test_loader, 'awgn', 7.0, False, max_b=20)
            mb = evaluate(model, test_loader, 'bsc', 0.10, False, max_b=20)
            avg = (ma['psnr']+mb['psnr'])/2
            print(f"  E{ep:02d}: loss={el/en:.5f}, AWGN(7)={ma['psnr']:.2f}, BSC(0.1)={mb['psnr']:.2f}")
            if avg > best: best = avg; save_ckpt(model, 'stage3', best)
    return best

# S4: Scorer
def train_stage4(model, train_loader, test_loader, epochs=40, lr=3e-4, cbr=0.5):
    print(f"\n{'='*60}\n  S4: Scorer (CBR={cbr})\n  Epochs: {epochs}\n{'='*60}")
    for n, p in model.named_parameters(): p.requires_grad = 'scorer' in n
    opt = optim.AdamW(model.scorer.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    best = 0
    for ep in range(1, epochs+1):
        model.train(); model.img_encoder.eval(); model.snn_encoder.eval()
        el, en = 0, 0
        for imgs in train_loader:
            if isinstance(imgs, (list, tuple)): imgs = imgs[0]
            imgs = imgs.to(device)
            ch = 'awgn' if random.random()<0.5 else 'bsc'
            np_ = random.choice([1,4,7,10,13,19]) if ch=='awgn' else random.choice([0.0,0.05,0.10,0.15,0.20,0.30])
            recon, info = model(imgs, np_, ch, use_masking=True)
            lr_, mse, ssim = recon_loss(imgs, recon)
            lrate = (info['importance'].mean()-cbr)**2 if info['importance'] is not None else 0
            with torch.no_grad(): sd = [s.detach() for s in info['spikes']]
            ldiv = model.scorer.compute_diversity_loss(sd)
            loss = lr_ + 10*lrate + 0.05*ldiv
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.scorer.parameters(), 1.0); opt.step()
            el += loss.item()*imgs.size(0); en += imgs.size(0)
        sched.step()
        if ep % 5 == 0 or ep == epochs:
            m = evaluate(model, test_loader, 'awgn', 7.0, True, max_b=20)
            mr = evaluate_random_mask(model, test_loader, 'awgn', 7.0, cbr, max_b=20)
            gap = m['psnr']-mr['psnr']
            print(f"  E{ep:02d}: L={m['psnr']:.2f}, R={mr['psnr']:.2f}, Gap={gap:+.2f}, CBR={m['cbr']:.3f}")
            if m['psnr'] > best: best = m['psnr']; save_ckpt(model, 'stage4', best)
    for p in model.parameters(): p.requires_grad = True
    return best

# S5: Joint fine-tuning (longer, focus full-BW first)
def train_stage5(model, train_loader, test_loader, epochs=30, lr=1e-5, cbr=0.5):
    print(f"\n{'='*60}\n  S5: Joint FT (30 epochs, start full-BW)\n  Epochs: {epochs}\n{'='*60}")
    for p in model.parameters(): p.requires_grad = True
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-7)
    best = 0
    for ep in range(1, epochs+1):
        model.train(); el, en = 0, 0
        # First 10 epochs: 80% full-BW to recover quality
        # Last 20 epochs: 50/50 masked/full to maintain masking ability
        mask_prob = 0.2 if ep <= 10 else 0.5
        for imgs in train_loader:
            if isinstance(imgs, (list, tuple)): imgs = imgs[0]
            imgs = imgs.to(device)
            ch = 'awgn' if random.random()<0.5 else 'bsc'
            np_ = random.choice([1,4,7,10,13,19]) if ch=='awgn' else random.choice([0.0,0.05,0.10,0.15,0.20,0.30])
            use_mask = random.random() < mask_prob
            recon, info = model(imgs, np_, ch, use_mask)
            lr_, mse, ssim = recon_loss(imgs, recon)
            if use_mask and info['importance'] is not None:
                lrate = (info['importance'].mean()-cbr)**2
                loss = lr_ + 5*lrate
            else:
                loss = lr_
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()
            el += loss.item()*imgs.size(0); en += imgs.size(0)
        sched.step()
        if ep % 3 == 0 or ep == epochs:
            m = evaluate(model, test_loader, 'awgn', 7.0, True, max_b=20)
            mf = evaluate(model, test_loader, 'awgn', 7.0, False, max_b=20)
            mr = evaluate_random_mask(model, test_loader, 'awgn', 7.0, cbr, max_b=20)
            gap = m['psnr']-mr['psnr']
            print(f"  E{ep:02d}: Full={mf['psnr']:.2f}, Masked={m['psnr']:.2f}, "
                  f"Rand={mr['psnr']:.2f}, Gap={gap:+.2f}")
            if mf['psnr'] > best: best = mf['psnr']; save_ckpt(model, 'stage5', best)
    return best

# Full eval
def full_eval(model, loader, cbr, name='instereo2k'):
    print(f"\n{'='*60}\n  FULL EVAL: {name}\n{'='*60}")
    results = {'awgn': {}, 'bsc': {}}
    for snr in [1,4,7,10,13,19]:
        results['awgn'][str(snr)] = {}
        for c in [0.125,0.25,0.375,0.5,0.625,0.75,1.0]:
            ml = evaluate(model, loader, 'awgn', snr, c<1.0, c)
            mr = evaluate_random_mask(model, loader, 'awgn', snr, c)
            g = round(ml['psnr']-mr['psnr'], 3)
            results['awgn'][str(snr)][str(c)] = {'learned': ml, 'random': mr, 'gap_psnr': g}
            print(f"  AWGN {snr:2d}dB CBR={c:.3f}: L={ml['psnr']:.2f} R={mr['psnr']:.2f} Δ={g:+.2f}")
    for ber in [0.0,0.05,0.10,0.15,0.20,0.30]:
        results['bsc'][str(ber)] = {}
        for c in [0.125,0.25,0.375,0.5,0.625,0.75,1.0]:
            ml = evaluate(model, loader, 'bsc', ber, c<1.0, c)
            mr = evaluate_random_mask(model, loader, 'bsc', ber, c)
            g = round(ml['psnr']-mr['psnr'], 3)
            results['bsc'][str(ber)][str(c)] = {'learned': ml, 'random': mr, 'gap_psnr': g}
            print(f"  BSC {ber:.2f} CBR={c:.3f}: L={ml['psnr']:.2f} R={mr['psnr']:.2f} Δ={g:+.2f}")
    os.makedirs('eval/seed_results', exist_ok=True)
    fn = f'eval/seed_results/kitti_image_v5_{name}.json'
    with open(fn, 'w') as f: json.dump(results, f, indent=2)
    print(f"\n✅ Saved {fn}"); return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, default=0)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--target-cbr', type=float, default=0.5)
    parser.add_argument('--C-tx', type=int, default=96)
    parser.add_argument('--T', type=int, default=16)
    parser.add_argument('--no-swin', action='store_true')
    parser.add_argument('--no-skips', action='store_true')
    args = parser.parse_args()
    use_swin = not args.no_swin
    use_skips = not args.no_skips
    print(f"Device: {device}")
    print(f"V5: Ternary+MemShortcut+KD+T={args.T}+Swin={use_swin}+UNetSkips={use_skips}+C_tx={args.C_tx}")

    train_ds, test_ds, kitti_ds = get_datasets('./InStereo2K', './data')
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    kitti_loader = DataLoader(kitti_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True) if kitti_ds else None

    model = SpikeImageCodecV5(C_feat=512, C_tx=args.C_tx, T=args.T,
        target_cbr=args.target_cbr, use_swin=use_swin, use_skips=use_skips).to(device)
    teacher = FloatTeacherV5(C_feat=512, C_tx=args.C_tx, use_swin=use_swin).to(device)

    print(f"  Student: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Teacher: {sum(p.numel() for p in teacher.parameters()):,}")

    if args.eval_only:
        load_ckpt(model, 'stage5') or load_ckpt(model, 'stage4') or load_ckpt(model, 'stage3')
        full_eval(model, test_loader, args.target_cbr, 'instereo2k')
        if kitti_loader: full_eval(model, kitti_loader, args.target_cbr, 'kitti')
        return

    if args.stage in [0, 1]:
        if not load_ckpt(teacher, 'teacher', 'teacher_'):
            train_teacher(teacher, train_loader, test_loader, epochs=30)
        else: print("  Teacher found.")
    if args.stage in [0, 2]:
        load_ckpt(teacher, 'teacher', 'teacher_')
        if not load_ckpt(model, 'stage2'):
            train_stage2_kd(model, teacher, train_loader, test_loader, epochs=60)
        else: print("  S2 found.")
    if args.stage in [0, 3]:
        if not load_ckpt(model, 'stage2'): print("  WARN: no S2")
        if not load_ckpt(model, 'stage3'):
            train_stage3(model, train_loader, test_loader, epochs=50)
        else: print("  S3 found.")
    if args.stage in [0, 4]:
        if not load_ckpt(model, 'stage3'): print("  WARN: no S3")
        if not load_ckpt(model, 'stage4'):
            train_stage4(model, train_loader, test_loader, epochs=40, cbr=args.target_cbr)
        else: print("  S4 found.")
    if args.stage in [0, 5]:
        if not load_ckpt(model, 'stage4'): print("  WARN: no S4")
        train_stage5(model, train_loader, test_loader, epochs=30, cbr=args.target_cbr)

    load_ckpt(model, 'stage5') or load_ckpt(model, 'stage4')
    full_eval(model, test_loader, args.target_cbr, 'instereo2k')
    if kitti_loader: full_eval(model, kitti_loader, args.target_cbr, 'kitti')

if __name__ == '__main__':
    main()
