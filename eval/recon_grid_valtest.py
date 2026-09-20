#!/usr/bin/env python3
"""Leakage-free (CBR, T', BER) grid on the KITTI image-reconstruction codec
(SpikeImageCodecV6) — cross-modality deferred-track surface for the unified
journal paper. Mirrors eval/radial_grid_valtest.py.

CBR: spatial-rate knob (target_cbr_override), the rho analog.
T':  temporal truncation — only the first T' of each scale's spike timesteps
     are transmitted (training-free, as in the ICC/radar work); the matching
     decoder timestep count is set to T' so accumulation stays consistent.
     Per-scale native T=(4,4,4,8); T'=8 == full model.
BER: ternary-BSC bit-flip probability (channel='bsc').

Split: KITTI (400 imgs, center-cropped 256x512) is a pure TRANSFER set — the
codec was trained/selected on InStereo2K, never on KITTI — so a deterministic
even/odd split (even=val, odd=test) is fully leakage-free for the rate policy.

Incremental: each cell appended to the output JSON, keyed [seed][val|test][key].
Usage: python eval/recon_grid_valtest.py
Output: eval/seed_results/recon_grid_valtest.json
"""

import argparse
import glob
import json
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as TT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models.spike_image_codec_v6 import SpikeImageCodecV6        # noqa: E402
from train.train_kitti_image_v6_lowcbr import (CROP_H, CROP_W,   # noqa: E402
                                               compute_ssim_torch, psnr)

# v6 (target_cbr=0.5, bw_mode='same'): rate-RESPONSIVE and good quality
# (KITTI 13->24 dB, InStereo 18->35 dB across CBR). v6_lowcbr was rate-
# responsive but capped ~18 dB (trained at cbr=0.065); v5 was high quality
# but rate-INERT (flat PSNR vs CBR) so unusable for the two-axis story.
CKPT = os.path.join(ROOT, 'snapshots_kitti_v6/stage5_best_33.58.pth')
OUT = os.path.join(ROOT, 'eval/seed_results/recon_grid_valtest.json')
KITTI = os.path.join(ROOT, 'data/kitti_stereo_2015/training/image_2')
INSTEREO_TEST = os.path.join(ROOT, 'InStereo2K/test')
SEED = 'v6'
CBRS = [0.125, 0.25, 0.5, 0.75, 1.0]
TPRIMES = [1, 2, 4, 8]
BERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
C_TX = (16, 32, 48, 64)
NATIVE_T = (4, 4, 4, 8)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def payload_bits(cbr, tp):
    """Deterministic proxy: cbr * sum_scale C_tx * area * min(T', native_T)."""
    tot = 0
    for i, (c, nt) in enumerate(zip(C_TX, NATIVE_T)):
        h, w = CROP_H >> (i + 1), CROP_W >> (i + 1)
        tot += c * h * w * min(tp, nt)
    return int(cbr * tot)


def ms_ssim(x, xc):
    ms, a, b = 0.0, x, xc
    for i, w in enumerate([0.4, 0.3, 0.3]):
        ms += w * compute_ssim_torch(a, b).item()
        if i < 2:
            a, b = F.avg_pool2d(a, 2), F.avg_pool2d(b, 2)
    return ms


@torch.no_grad()
def forward_trunc(model, img, cbr, tp, ber):
    """V6 forward with CBR override, T' truncation, ternary-BSC at prob=ber."""
    feats = model.encoder(img)
    multi_spikes = model.snn_encoder(feats)
    imp_maps, sw = model.scorer(multi_spikes, ber)
    masked, _ = model.masker(multi_spikes, imp_maps, sw, False, cbr)
    recv = []
    for scale_spikes in masked:                       # truncate + channel
        trunc = scale_spikes[:tp]
        recv.append([model.bsc(s, ber) for s in trunc])
    saved = [dec.T for dec in model.decoder.snn_decoders]
    for dec in model.decoder.snn_decoders:
        dec.T = min(tp, dec.T)                          # decoder consumes T'
    img_recon, _ = model.decoder(recv)
    for dec, t in zip(model.decoder.snn_decoders, saved):
        dec.T = t
    return img_recon.clamp(0, 1)


def build_loaders(dataset='kitti'):
    tf = TT.Compose([TT.CenterCrop((CROP_H, CROP_W)), TT.ToTensor()])
    if dataset == 'kitti':
        # KITTI is a pure transfer set (never trained on) -> cleanest split.
        imgs = sorted(glob.glob(os.path.join(KITTI, '*.png')))
    else:
        # InStereo2K: in-distribution. Split the OFFICIAL TEST scenes (held out
        # from training) even/odd for the rate policy; checkpoint-selection
        # leakage on this set is disclosed (see module docstring / paper).
        imgs = sorted(glob.glob(os.path.join(INSTEREO_TEST, '*', 'left.png')))
    val = imgs[0::2]
    test = imgs[1::2]
    print(f'{dataset} split: val {len(val)} / test {len(test)}', flush=True)
    return {'val': val, 'test': test}, tf


def score(model, paths, tf, cbr, tp, ber):
    ps, ss, n = 0.0, 0.0, 0
    for p in paths:
        img = tf(Image.open(p).convert('RGB')).unsqueeze(0).to(device)
        rec = forward_trunc(model, img, cbr, tp, ber)
        ps += psnr(img, rec)
        ss += ms_ssim(img, rec)
        n += 1
    return {'psnr': round(ps / n, 4), 'ms_ssim': round(ss / n, 4),
            'payload_bits': payload_bits(cbr, tp), 'n': n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['kitti', 'instereo'], default='kitti')
    args = ap.parse_args()
    out = OUT if args.dataset == 'kitti' else OUT.replace(
        'recon_grid_valtest', 'recon_grid_valtest_instereo')
    model = SpikeImageCodecV6(C_tx=C_TX, target_cbr=0.5, use_sdsa=True,
                              bw_mode='same').to(device)
    sd = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(sd.get('model', sd) if isinstance(sd, dict)
                          and 'model' in sd else sd)
    model.eval()
    splits, tf = build_loaders(args.dataset)

    res = json.load(open(out)) if os.path.exists(out) else {}
    node = res.setdefault(SEED, {})
    for split, paths in splits.items():
        sp = node.setdefault(split, {})
        for cbr in CBRS:
            for tp in TPRIMES:
                for ber in BERS:
                    key = f'cbr{cbr}_T{tp}_ber{ber}'
                    if key in sp:
                        continue
                    sp[key] = score(model, paths, tf, cbr, tp, ber)
                    json.dump(res, open(out, 'w'), indent=1)
                    print(f"{split} {key}: psnr {sp[key]['psnr']:.3f} "
                          f"ms_ssim {sp[key]['ms_ssim']:.4f} "
                          f"pay {sp[key]['payload_bits']/1e6:.2f}M", flush=True)
    print('saved:', out)


if __name__ == '__main__':
    main()
