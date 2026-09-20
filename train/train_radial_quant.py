#!/usr/bin/env python3
"""Retrained-quantized control for the RADIal track: same frozen FPN and
same trained-decoder setup as the SNN bottleneck, but the bottleneck is a
straight-through uniform quantizer (b bits/value) with i.i.d. bit flips,
trained with the SAME BER curriculum. This is the fair digital control
(analog of the ICC paper's CNN-1bit finding: discreteness + noise training
drive robustness) — it separates "learned + noise-trained" from "spiking".

Payload/frame = 1.82M elems * b bits (b=1: 1.82 Mbit, vs SNN 2.32 Mbit).

Usage: python train/train_radial_quant.py --bits 1 [--epochs 10]
Outputs: runs/fftradnet_quant{b}_radial.pth,
         eval/seed_results/radial_quant{b}_retrained_sweep.json
"""

import argparse
import contextlib
import io
import json
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'radial_repo', 'FFTRadNet'))

from loss import pixor_loss                                       # noqa: E402
from utils.evaluation import run_FullEvaluation                   # noqa: E402
from eval.radial_feature_quant_baseline import CappedEncoder      # noqa: E402
from train.train_radial_snn import (CONFIG, SNN_LEVELS,           # noqa: E402
                                    build_loaders, load_base)

BERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ste_quant_bsc(feat, bits, ber):
    """Straight-through uniform quantization + STE bit flips."""
    scale = feat.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
    levels = 2 ** bits
    idx = torch.round((feat / scale + 1) / 2 * (levels - 1)).clamp(0, levels - 1)
    idx_i = idx.long()
    if ber > 0:
        flip = torch.zeros_like(idx_i)
        for k in range(bits):
            flip += (torch.rand_like(feat) < ber).long() << k
        idx_i = idx_i ^ flip
    deq = (idx_i.float() / (levels - 1) * 2 - 1) * scale
    return feat + (deq - feat).detach()      # STE through quant + channel


class FFTRadNetQuantTrained(nn.Module):
    def __init__(self, base, bits):
        super().__init__()
        self.FPN = base.FPN
        self.RA_decoder = base.RA_decoder
        self.detection_header = base.detection_header
        self.freespace = base.freespace
        self.bits = bits
        self.ber = 0.0
        for p in self.FPN.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.FPN.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            feats = self.FPN(x)
        f = dict(feats)
        for k in SNN_LEVELS:
            f[k] = ste_quant_bsc(feats[k], self.bits, self.ber)
        RA = self.RA_decoder(f)
        out = {'Detection': self.detection_header(RA)}
        out['Segmentation'] = self.freespace(F.interpolate(RA, (256, 224)))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bits', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--train-seed', dest='train_seed', type=int, default=None,
                    help='seed for init/training randomness (data split stays '
                         'at the official config seed)')
    ap.add_argument('--tag', default='', help='suffix for ckpt/json outputs')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    tag = f'_{args.tag}' if args.tag else ''
    save = os.path.join(ROOT, f'runs/fftradnet_quant{args.bits}_radial{tag}.pth')
    out_json = os.path.join(
        ROOT,
        f'eval/seed_results/radial_quant{args.bits}_retrained_sweep{tag}.json')

    config = json.load(open(CONFIG))
    ts = args.train_seed if args.train_seed is not None else config['seed']
    torch.manual_seed(ts)
    np.random.seed(ts)
    base = load_base(config)
    net = FFTRadNetQuantTrained(base, args.bits).to(device)
    enc, train_loader, _, test_loader = build_loaders(config, args.batch)

    seg_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
    trainable = [p for p in net.parameters() if p.requires_grad]
    print(f'bits={args.bits} trainable params: '
          f'{sum(p.numel() for p in trainable):,}', flush=True)
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    total_iters = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total_iters)
    rng = np.random.default_rng(ts)
    net.train()
    it = 0
    for epoch in range(args.epochs):
        for data in train_loader:
            inputs = data[0].to(device).float()
            label_map = data[1].to(device).float()
            seg_label = data[2].to(device).float()
            cap = 0.40 * min(1.0, it / max(1, int(total_iters * 0.6)))
            net.ber = float(rng.uniform(0.0, cap)) if cap > 0 else 0.0
            outputs = net(inputs)
            cls_loss, reg_loss = pixor_loss(outputs['Detection'], label_map,
                                            config['losses'])
            seg_loss = seg_loss_fn(outputs['Segmentation'].flatten(),
                                   seg_label.flatten()) * inputs.size(0)
            w = config['losses']['weight']
            loss = cls_loss * w[0] + reg_loss * w[1] + seg_loss * w[2]
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if it % 200 == 0:
                print(f'ep {epoch} it {it:5d}/{total_iters} '
                      f'loss {loss.item():9.2f} ber {net.ber:.3f}', flush=True)
            it += 1
            if args.smoke and it >= 30:
                break
        if args.smoke:
            break
    torch.save({'decoder_state': net.RA_decoder.state_dict(),
                'det_head_state': net.detection_header.state_dict(),
                'seg_head_state': net.freespace.state_dict(),
                'bits': args.bits}, save)
    print(f'saved: {save}', flush=True)

    # full-protocol sweep
    net.eval()
    enc_c = CappedEncoder(enc, k=200)
    elems = 160 * 128 * 64 + 192 * 64 * 32 + 224 * 32 * 16
    results = {'bits': args.bits,
               'payload_bits_per_frame': elems * args.bits}
    for ber in BERS:
        net.ber = ber
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_FullEvaluation(net, test_loader, enc_c)
        text = buf.getvalue()
        nums = {}
        for mk in ['mAP', 'mAR', 'mIoU']:
            m = re.search(rf'{mk}:?\s+([0-9.]+)', text)
            nums[mk] = float(m.group(1)) if m else None
        results[f'ber{ber}'] = nums
        json.dump(results, open(out_json, 'w'), indent=1)
        print(f"bits={args.bits} BER {ber:.2f}: mAP {nums['mAP']:.4f} "
              f"mAR {nums['mAR']:.4f} mIoU {nums['mIoU']:.2f}", flush=True)
    print(f'saved: {out_json}')


if __name__ == '__main__':
    main()
