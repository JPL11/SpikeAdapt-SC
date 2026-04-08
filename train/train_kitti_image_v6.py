#!/usr/bin/env python3
"""Train SpikeAdapt-SC V6: Spike-Driven Progressive JSCC.

Trains both bw_mode='extra' and bw_mode='same', compares results.
5-stage pipeline per model.
"""

import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
import sys, os, json, random, math, glob
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms as T
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.spike_image_codec_v6 import SpikeImageCodecV6, FloatTeacherV6

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CROP_H, CROP_W = 256, 512

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
    train_sets, test_sets = [], []
    for sp, sets, tf in [('train',train_sets,tf_train),('test',test_sets,tf_test)]:
        d = os.path.join('./InStereo2K', sp)
        if os.path.exists(d):
            imgs = [os.path.join(d,s,'left.png') for s in sorted(os.listdir(d))
                    if os.path.isfile(os.path.join(d,s,'left.png'))]
            if imgs: sets.append(ImageFolderFlat(imgs,tf)); print(f"  InStereo2K {sp}: {len(imgs)}")
    kitti_ds = None
    kd = './data/kitti_stereo_2015/training/image_2'
    if os.path.exists(kd):
        ki = sorted(glob.glob(os.path.join(kd,'*.png')))
        if ki: kitti_ds = ImageFolderFlat(ki, tf_test); print(f"  KITTI: {len(ki)}")
    return (ConcatDataset(train_sets) if len(train_sets)>1 else train_sets[0],
            ConcatDataset(test_sets) if len(test_sets)>1 else test_sets[0], kitti_ds)

def recon_loss(x, xh):
    xc = xh.clamp(0,1); mse = F.mse_loss(xc,x)
    ms = 0
    a, b = x, xc
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

SNAP_DIR = './snapshots_kitti_v6'
def save_ckpt(model, tag, val, prefix=''):
    os.makedirs(SNAP_DIR, exist_ok=True)
    p = os.path.join(SNAP_DIR, f'{prefix}{tag}_best_{val:.2f}.pth')
    torch.save(model.state_dict(), p); print(f"  ✓ Saved {p}")

def load_ckpt(model, tag, prefix=''):
    cks = sorted(glob.glob(os.path.join(SNAP_DIR, f'{prefix}{tag}_best_*.pth')),
                 key=lambda x: float(x.split('_')[-1].replace('.pth','')))
    if cks:
        state = torch.load(cks[-1], map_location=device, weights_only=False)
        ms = model.state_dict()
        filt = {k:v for k,v in state.items() if k in ms and v.shape==ms[k].shape}
        model.load_state_dict(filt, strict=False)
        print(f"  Loaded {cks[-1]}"); return True
    return False

def train_teacher(teacher, tl, vl, epochs=30, lr=1e-3):
    print(f"\n{'='*60}\n  S1: Float Teacher\n{'='*60}")
    opt = optim.AdamW(teacher.parameters(), lr=lr, weight_decay=1e-5)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    best = 0
    for ep in range(1, epochs+1):
        teacher.train(); el, en = 0, 0
        for imgs in tl:
            if isinstance(imgs,(list,tuple)): imgs = imgs[0]
            imgs = imgs.to(device)
            r, _ = teacher(imgs, 100.0)
            loss, _, _ = recon_loss(imgs, r)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(teacher.parameters(),1.0); opt.step()
            el += loss.item()*imgs.size(0); en += imgs.size(0)
        sch.step()
        if ep%5==0 or ep==epochs:
            teacher.eval(); ps, n = 0, 0
            with torch.no_grad():
                for i, imgs in enumerate(vl):
                    if i>=20: break
                    if isinstance(imgs,(list,tuple)): imgs = imgs[0]
                    imgs = imgs.to(device)
                    r, _ = teacher(imgs, 100.0)
                    ps += psnr(imgs,r.clamp(0,1))*imgs.size(0); n += imgs.size(0)
            p = ps/n
            print(f"  E{ep:02d}: loss={el/en:.5f}, PSNR={p:.2f}")
            if p > best: best = p; save_ckpt(teacher, 'teacher', best, 'teacher_')

def train_s2_kd(model, teacher, tl, vl, epochs=60, lr=1e-3):
    print(f"\n{'='*60}\n  S2: SNN + KD\n{'='*60}")
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    best = 0
    for ep in range(1, epochs+1):
        model.train(); el, en = 0, 0
        for imgs in tl:
            if isinstance(imgs,(list,tuple)): imgs = imgs[0]
            imgs = imgs.to(device)
            rs, infos = model(imgs, 100.0, 'awgn', False)
            lr_, mse, ssim = recon_loss(imgs, rs)
            with torch.no_grad(): rt, infot = teacher(imgs, 100.0)
            lkd = F.mse_loss(rs.clamp(0,1), rt.clamp(0,1))
            loss = lr_ + 0.5*lkd
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            el += loss.item()*imgs.size(0); en += imgs.size(0)
        sch.step()
        if ep%5==0 or ep==epochs:
            m = evaluate(model, vl, 'awgn', 100.0, False, max_b=20)
            print(f"  E{ep:02d}: loss={el/en:.5f}, PSNR={m['psnr']:.2f}")
            if m['psnr'] > best: best = m['psnr']; save_ckpt(model, 'stage2', best)

def train_s3(model, tl, vl, epochs=50, lr=5e-4):
    print(f"\n{'='*60}\n  S3: Channel noise\n{'='*60}")
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    best = 0
    for ep in range(1, epochs+1):
        model.train(); el, en = 0, 0
        for imgs in tl:
            if isinstance(imgs,(list,tuple)): imgs = imgs[0]
            imgs = imgs.to(device)
            ch = 'awgn' if random.random()<0.5 else 'bsc'
            np_ = random.choice([1,4,7,10,13,19]) if ch=='awgn' else random.choice([0.0,0.05,0.10,0.15,0.20,0.30])
            r, _ = model(imgs, np_, ch, False)
            loss, _, _ = recon_loss(imgs, r)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            el += loss.item()*imgs.size(0); en += imgs.size(0)
        sch.step()
        if ep%5==0 or ep==epochs:
            ma = evaluate(model, vl, 'awgn', 7.0, False, max_b=20)
            mb = evaluate(model, vl, 'bsc', 0.10, False, max_b=20)
            avg = (ma['psnr']+mb['psnr'])/2
            print(f"  E{ep:02d}: AWGN={ma['psnr']:.2f}, BSC={mb['psnr']:.2f}")
            if avg > best: best = avg; save_ckpt(model, 'stage3', best)

def train_s4(model, tl, vl, epochs=40, lr=3e-4, cbr=0.5):
    print(f"\n{'='*60}\n  S4: Scorer (CBR={cbr})\n{'='*60}")
    for n,p in model.named_parameters(): p.requires_grad = 'scorer' in n
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
            np_ = random.choice([1,4,7,10,13,19]) if ch=='awgn' else random.choice([0.0,0.05,0.10,0.15,0.20,0.30])
            r, info = model(imgs, np_, ch, True)
            lr_, _, _ = recon_loss(imgs, r)
            lrate = sum((im.mean()-cbr)**2 for im in info['importance'] if im is not None) / 4
            ldiv = model.scorer.compute_diversity_loss(info['multi_spikes'])
            loss = lr_ + 10*lrate + 0.05*ldiv
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.scorer.parameters(),1.0); opt.step()
            el += loss.item()*imgs.size(0); en += imgs.size(0)
        sch.step()
        if ep%5==0 or ep==epochs:
            m = evaluate(model, vl, 'awgn', 7.0, True, max_b=20)
            mf = evaluate(model, vl, 'awgn', 7.0, False, max_b=20)
            print(f"  E{ep:02d}: Full={mf['psnr']:.2f}, Masked={m['psnr']:.2f}")
            if m['psnr'] > best: best = m['psnr']; save_ckpt(model, 'stage4', best)
    for p in model.parameters(): p.requires_grad = True

def train_s5(model, tl, vl, epochs=30, lr=1e-5, cbr=0.5):
    print(f"\n{'='*60}\n  S5: Joint FT\n{'='*60}")
    for p in model.parameters(): p.requires_grad = True
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-7)
    best = 0
    for ep in range(1, epochs+1):
        model.train(); el, en = 0, 0
        mp = 0.2 if ep <= 10 else 0.5
        for imgs in tl:
            if isinstance(imgs,(list,tuple)): imgs = imgs[0]
            imgs = imgs.to(device)
            ch = 'awgn' if random.random()<0.5 else 'bsc'
            np_ = random.choice([1,4,7,10,13,19]) if ch=='awgn' else random.choice([0.0,0.05,0.10,0.15,0.20,0.30])
            um = random.random() < mp
            r, info = model(imgs, np_, ch, um)
            lr_, _, _ = recon_loss(imgs, r)
            if um and info['importance'] is not None:
                lrate = sum((im.mean()-cbr)**2 for im in info['importance']) / 4
                loss = lr_ + 5*lrate
            else: loss = lr_
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),0.5); opt.step()
            el += loss.item()*imgs.size(0); en += imgs.size(0)
        sch.step()
        if ep%3==0 or ep==epochs:
            mf = evaluate(model, vl, 'awgn', 7.0, False, max_b=20)
            mm = evaluate(model, vl, 'awgn', 7.0, True, max_b=20)
            print(f"  E{ep:02d}: Full={mf['psnr']:.2f}, Masked={mm['psnr']:.2f}")
            if mf['psnr'] > best: best = mf['psnr']; save_ckpt(model, 'stage5', best)

def full_eval(model, loader, name='instereo2k'):
    print(f"\n{'='*60}\n  FULL EVAL: {name}\n{'='*60}")
    results = {'awgn': {}, 'bsc': {}}
    for snr in [1,4,7,10,13,19]:
        results['awgn'][str(snr)] = {}
        for c in [0.125,0.25,0.375,0.5,0.625,0.75,1.0]:
            ml = evaluate(model, loader, 'awgn', snr, c<1.0, c)
            print(f"  AWGN {snr:2d}dB CBR={c:.3f}: {ml['psnr']:.2f}")
            results['awgn'][str(snr)][str(c)] = ml
    for ber in [0.0,0.05,0.10,0.15,0.20,0.30]:
        results['bsc'][str(ber)] = {}
        for c in [0.125,0.25,0.375,0.5,0.625,0.75,1.0]:
            ml = evaluate(model, loader, 'bsc', ber, c<1.0, c)
            print(f"  BSC {ber:.2f} CBR={c:.3f}: {ml['psnr']:.2f}")
            results['bsc'][str(ber)][str(c)] = ml
    os.makedirs('eval/seed_results', exist_ok=True)
    fn = f'eval/seed_results/kitti_image_v6_{name}.json'
    with open(fn,'w') as f: json.dump(results,f,indent=2)
    print(f"\n✅ Saved {fn}"); return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bw-mode', default='extra', choices=['extra','same'])
    parser.add_argument('--stage', type=int, default=0)
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--target-cbr', type=float, default=0.5)
    args = parser.parse_args()

    print(f"Device: {device}")
    print(f"V6: Progressive JSCC + SDSA, bw_mode={args.bw_mode}")

    train_ds, test_ds, kitti_ds = get_datasets()
    tl = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    vl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    kl = DataLoader(kitti_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True) if kitti_ds else None

    model = SpikeImageCodecV6(target_cbr=args.target_cbr, use_sdsa=True, bw_mode=args.bw_mode).to(device)
    teacher = FloatTeacherV6().to(device)
    print(f"  Student: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Teacher: {sum(p.numel() for p in teacher.parameters()):,}")

    if args.eval_only:
        load_ckpt(model, 'stage5') or load_ckpt(model, 'stage4') or load_ckpt(model, 'stage3')
        full_eval(model, vl, 'instereo2k')
        if kl: full_eval(model, kl, 'kitti')
        return

    if args.stage in [0,1]:
        if not load_ckpt(teacher,'teacher','teacher_'):
            train_teacher(teacher, tl, vl, 30)
    if args.stage in [0,2]:
        load_ckpt(teacher,'teacher','teacher_')
        if not load_ckpt(model,'stage2'):
            train_s2_kd(model, teacher, tl, vl, 60)
    if args.stage in [0,3]:
        load_ckpt(model,'stage2')
        if not load_ckpt(model,'stage3'):
            train_s3(model, tl, vl, 50)
    if args.stage in [0,4]:
        load_ckpt(model,'stage3')
        if not load_ckpt(model,'stage4'):
            train_s4(model, tl, vl, 40, cbr=args.target_cbr)
    if args.stage in [0,5]:
        load_ckpt(model,'stage4')
        train_s5(model, tl, vl, 30, cbr=args.target_cbr)

    load_ckpt(model,'stage5') or load_ckpt(model,'stage4')
    full_eval(model, vl, 'instereo2k')
    if kl: full_eval(model, kl, 'kitti')

if __name__ == '__main__':
    main()
