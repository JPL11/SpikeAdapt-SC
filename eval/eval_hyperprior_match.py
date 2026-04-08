#!/usr/bin/env python3
"""Evaluate SpikeAdapt-SC V6 strictly mimicking Hyperprior evaluation constraints.

Targets:
  - CBR = 0.065 to mimic their Compression Ratio = 0.065
  - CSNR (AWGN) = [0.0, 2.5, 5.0, 7.5, 10.0] dB
  - Tracks both PSNR and strictly standard MS-SSIM via pytorch_msssim.
"""

import torch, json, os, sys, glob, math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from pytorch_msssim import ms_ssim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.spike_image_codec_v6 import SpikeImageCodecV6

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CROP_H, CROP_W = 256, 512
SNAP_DIR = './snapshots_kitti_v6'

class ImageFolderFlat(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        return self.transform(img) if self.transform else img

def get_test_datasets():
    tf_test = T.Compose([T.CenterCrop((CROP_H, CROP_W)), T.ToTensor()])
    
    instereo = []
    d = os.path.join('./InStereo2K', 'test')
    if os.path.exists(d):
        imgs = [os.path.join(d, s, 'left.png') for s in sorted(os.listdir(d)) if os.path.isfile(os.path.join(d, s, 'left.png'))]
        if imgs: instereo = ImageFolderFlat(imgs, tf_test)

    kitti = []
    kd = './data/kitti_stereo_2015/training/image_2'
    if os.path.exists(kd):
        ki = sorted(glob.glob(os.path.join(kd, '*.png')))
        if ki: kitti = ImageFolderFlat(ki, tf_test)
        
    return instereo, kitti

def psnr(x, xh):
    import torch.nn.functional as F
    m = F.mse_loss(xh, x).item()
    return 100.0 if m==0 else 10*math.log10(1.0/m)

def evaluate_matched(model, loader, noise=0.0):
    model.eval()
    total_psnr, total_msssim, n = 0, 0, 0
    with torch.no_grad():
        for i, imgs in enumerate(loader):
            if i >= 40: break  # Limit to 40 batches for fast sweeping during viz
            if isinstance(imgs, (list, tuple)): imgs = imgs[0]
            imgs = imgs.to(device)
            # strictly pass target_cbr_override=0.065 to mimic CR=0.065
            recon, info = model(imgs, noise, 'awgn', True, 0.065)
            rc = recon.clamp(0, 1)
            
            p = psnr(imgs, rc)
            # Compute ms_ssim on CPU to avoid NVRTC dynamic compilation issues
            m = ms_ssim(imgs.cpu(), rc.cpu(), data_range=1.0, size_average=True).item()
            
            total_psnr += p * imgs.size(0)
            total_msssim += m * imgs.size(0)
            n += imgs.size(0)
            
    return {'psnr': round(total_psnr/n, 3), 'ms_ssim': round(total_msssim/n, 4), 'n': n}

def load_ckpt(model, tag):
    cks = sorted(glob.glob(os.path.join(SNAP_DIR, f'{tag}_best_*.pth')), key=lambda x: float(x.split('_')[-1].replace('.pth','')))
    if cks:
        state = torch.load(cks[-1], map_location=device, weights_only=False)
        ms = model.state_dict()
        filt = {k:v for k,v in state.items() if k in ms and v.shape==ms[k].shape}
        model.load_state_dict(filt, strict=False)
        print(f"Loaded {cks[-1]}")
        return True
    return False

def main():
    instereo_ds, kitti_ds = get_test_datasets()
    
    model = SpikeImageCodecV6(target_cbr=0.065, use_sdsa=True, bw_mode='extra').to(device)
    if not (load_ckpt(model, 'stage5') or load_ckpt(model, 'stage4')):
        print("ERROR: Could not find V6 model checkpoint.")
        return
        
    csnr_list = [0.0, 2.5, 5.0, 7.5, 10.0]
    results = {'kitti': {}, 'instereo2k': {}}
    
    if instereo_ds:
        il = DataLoader(instereo_ds, batch_size=1, shuffle=False, num_workers=2)
        print("\nEvaluating InStereo2K at CR=0.065...")
        for snr in csnr_list:
            res = evaluate_matched(model, il, snr)
            results['instereo2k'][str(snr)] = res
            print(f"  CSNR {snr:>4.1f} dB -> PSNR: {res['psnr']:.2f}, MS-SSIM: {res['ms_ssim']:.3f}")
            
    if kitti_ds:
        kl = DataLoader(kitti_ds, batch_size=1, shuffle=False, num_workers=2)
        print("\nEvaluating KITTI at CR=0.065...")
        for snr in csnr_list:
            res = evaluate_matched(model, kl, snr)
            results['kitti'][str(snr)] = res
            print(f"  CSNR {snr:>4.1f} dB -> PSNR: {res['psnr']:.2f}, MS-SSIM: {res['ms_ssim']:.3f}")
            
    os.makedirs('eval/seed_results', exist_ok=True)
    out_file = 'eval/seed_results/hyperprior_match_results.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nSaved structurally matched comparisons to {out_file}")

if __name__ == '__main__':
    main()
