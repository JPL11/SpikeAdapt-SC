#!/usr/bin/env python3
"""B2: Evaluate FloatTeacherV6 (continuous, no quantization) on KITTI & InStereo2K.

This is the "Continuous SpikeAdapt" ablation: same V6 multi-scale architecture
but with continuous Tanh-bounded bottleneck instead of ternary spikes.
Shows what removing quantization buys in PSNR.

Checkpoint: snapshots_kitti_v6/teacher_teacher_best_34.89.pth
Output: eval/seed_results/float_teacher_v6_results.json

Usage:
    python eval/eval_float_teacher.py
"""

import torch, json, os, sys, glob, math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))

from models.spike_image_codec_v6 import FloatTeacherV6

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
        imgs = sorted([os.path.join(d, s, 'left.png') for s in os.listdir(d)
                       if os.path.isfile(os.path.join(d, s, 'left.png'))])
        if imgs: instereo = ImageFolderFlat(imgs, tf_test)

    kitti = []
    kd = './data/kitti_stereo_2015/training/image_2'
    if os.path.exists(kd):
        ki = sorted(glob.glob(os.path.join(kd, '*.png')))
        if ki: kitti = ImageFolderFlat(ki, tf_test)

    return instereo, kitti


def psnr(x, xh):
    m = F.mse_loss(xh, x).item()
    return 100.0 if m == 0 else 10 * math.log10(1.0 / m)


def evaluate_awgn(model, loader, snr_db, max_batches=None):
    """Evaluate FloatTeacherV6 at given AWGN SNR."""
    model.eval()
    total_psnr, n = 0, 0
    with torch.no_grad():
        for i, imgs in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            if isinstance(imgs, (list, tuple)):
                imgs = imgs[0]
            imgs = imgs.to(device)
            recon, _ = model(imgs, noise_param=snr_db, channel='awgn')
            rc = recon.clamp(0, 1)
            total_psnr += psnr(imgs, rc) * imgs.size(0)
            n += imgs.size(0)
    return {'psnr': round(total_psnr / n, 3), 'n': n} if n > 0 else {'psnr': 0, 'n': 0}


def evaluate_awgn_with_masking(model, loader, snr_db, cbr, max_batches=None):
    """Evaluate with simple spatial masking: keep top-k% of bottleneck locations."""
    model.eval()
    total_psnr, n = 0, 0
    with torch.no_grad():
        for i, imgs in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            if isinstance(imgs, (list, tuple)):
                imgs = imgs[0]
            imgs = imgs.to(device)

            # Encode
            feats = model.encoder(imgs)
            zs = [enc(f) for enc, f in zip(model.float_encs, feats)]

            # Spatial masking per scale
            masked_zs = []
            for z in zs:
                B, C, H, W = z.shape
                # Importance: L1 norm across channels
                importance = z.abs().mean(dim=1, keepdim=True)  # B×1×H×W
                k = max(1, int(cbr * H * W))
                flat = importance.view(B, -1)
                _, idx = flat.topk(k, dim=1)
                mask = torch.zeros_like(flat)
                mask.scatter_(1, idx, 1.0)
                mask = mask.view(B, 1, H, W)
                masked_zs.append(z * mask)

            # Add AWGN noise
            if snr_db < 100:
                snr_lin = 10 ** (snr_db / 10.0)
                noise_std = 1.0 / math.sqrt(2 * snr_lin)
                masked_zs = [z + torch.randn_like(z) * noise_std for z in masked_zs]

            # Decode
            ds = [dec(z) for dec, z in zip(model.float_decs, masked_zs)]
            f3 = model.fuse_43(ds[3], ds[2])
            f2 = model.fuse_32(f3, ds[1])
            f1 = model.fuse_21(f2, ds[0])
            recon = model.up_final(f1)
            rc = recon.clamp(0, 1)

            total_psnr += psnr(imgs, rc) * imgs.size(0)
            n += imgs.size(0)
    return {'psnr': round(total_psnr / n, 3), 'n': n} if n > 0 else {'psnr': 0, 'n': 0}


def load_teacher(model):
    """Load best teacher checkpoint."""
    cks = sorted(glob.glob(os.path.join(SNAP_DIR, 'teacher_teacher_best_*.pth')),
                 key=lambda x: float(x.split('_')[-1].replace('.pth', '')))
    if cks:
        state = torch.load(cks[-1], map_location=device, weights_only=False)
        ms = model.state_dict()
        filt = {k: v for k, v in state.items() if k in ms and v.shape == ms[k].shape}
        model.load_state_dict(filt, strict=False)
        print(f"Loaded {cks[-1]} ({len(filt)}/{len(ms)} params)")
        return True
    return False


def main():
    instereo_ds, kitti_ds = get_test_datasets()
    print(f"Device: {device}")
    print(f"InStereo2K: {len(instereo_ds) if instereo_ds else 0} images")
    print(f"KITTI: {len(kitti_ds) if kitti_ds else 0} images")

    model = FloatTeacherV6().to(device)
    if not load_teacher(model):
        print("ERROR: Could not find FloatTeacherV6 checkpoint")
        return

    snr_list = [0, 1, 2.5, 4, 5, 7, 10, 13, 19, 100]  # 100 = noiseless
    cbr_list = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0]

    results = {}

    for ds_name, ds_obj in [('instereo2k', instereo_ds), ('kitti', kitti_ds)]:
        if not ds_obj:
            continue
        loader = DataLoader(ds_obj, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        results[ds_name] = {'awgn_full_rate': {}, 'awgn_cbr_sweep': {}}

        # 1. Full rate AWGN sweep (no masking)
        print(f"\n{'='*60}\n{ds_name.upper()} — Full rate AWGN sweep\n{'='*60}")
        for snr in snr_list:
            res = evaluate_awgn(model, loader, snr)
            results[ds_name]['awgn_full_rate'][str(snr)] = res
            label = 'Clean' if snr >= 100 else f'CSNR={snr}dB'
            print(f"  {label}: PSNR={res['psnr']:.2f} dB ({res['n']} imgs)")

        # 2. CBR sweep at selected CSNR levels
        print(f"\n{ds_name.upper()} — CBR sweep")
        for snr in [1, 4, 7, 10, 100]:
            results[ds_name]['awgn_cbr_sweep'][str(snr)] = {}
            for cbr in cbr_list:
                res = evaluate_awgn_with_masking(model, loader, snr, cbr)
                results[ds_name]['awgn_cbr_sweep'][str(snr)][str(cbr)] = res
                label = 'Clean' if snr >= 100 else f'CSNR={snr}dB'
                print(f"  {label} CBR={cbr}: PSNR={res['psnr']:.2f} dB")

    # Save
    os.makedirs('eval/seed_results', exist_ok=True)
    out = 'eval/seed_results/float_teacher_v6_results.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == '__main__':
    main()
