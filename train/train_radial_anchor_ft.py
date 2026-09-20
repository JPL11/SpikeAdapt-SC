#!/usr/bin/env python3
"""E2 (review round 1): fine-tuned-anchor control.

Fine-tunes the released FFTRadNet's RA decoder + detection/segmentation heads
for the SAME 10 epochs / AdamW 1e-4 / cosine / batch 4 as the SNN system, but
on CLEAN frozen-FPN features with NO bottleneck and NO channel. This is the
fair lossless-transport anchor: any mAP/mAR/mIoU difference vs the released
checkpoint measures the fine-tuning itself, resolving the "mIoU exceeds the
lossless baseline" confound (panel finding F2).

Usage: python train/train_radial_anchor_ft.py [--epochs 10]
Outputs: runs/fftradnet_anchor_ft.pth,
         eval/seed_results/radial_anchor_ft.json
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

from loss import pixor_loss                                # noqa: E402
from utils.evaluation import run_FullEvaluation            # noqa: E402
from train.train_radial_snn import (CONFIG, build_loaders,  # noqa: E402
                                    load_base)

SAVE = os.path.join(ROOT, 'runs/fftradnet_anchor_ft.pth')
OUT = os.path.join(ROOT, 'eval/seed_results/radial_anchor_ft.json')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AnchorFT(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.FPN = base.FPN
        self.RA_decoder = base.RA_decoder
        self.detection_header = base.detection_header
        self.freespace = base.freespace
        for p in self.FPN.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.FPN.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            feats = self.FPN(x)
        RA = self.RA_decoder(feats)
        out = {'Detection': self.detection_header(RA)}
        out['Segmentation'] = self.freespace(F.interpolate(RA, (256, 224)))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    args = ap.parse_args()
    config = json.load(open(CONFIG))
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    base = load_base(config)
    net = AnchorFT(base).to(device)
    enc, train_loader, _, test_loader = build_loaders(config, args.batch)

    seg_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
    trainable = [p for p in net.parameters() if p.requires_grad]
    print(f'trainable params: {sum(p.numel() for p in trainable):,}', flush=True)
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    total_iters = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total_iters)
    net.train()
    it = 0
    for epoch in range(args.epochs):
        for data in train_loader:
            inputs = data[0].to(device).float()
            label_map = data[1].to(device).float()
            seg_label = data[2].to(device).float()
            outputs = net(inputs)
            cls_loss, reg_loss = pixor_loss(outputs['Detection'], label_map,
                                            config['losses'])
            seg_loss = seg_loss_fn(outputs['Segmentation'].flatten(),
                                   seg_label.flatten()) * inputs.size(0)
            w = config['losses']['weight']
            loss = cls_loss * w[0] + reg_loss * w[1] + seg_loss * w[2]
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if it % 500 == 0:
                print(f'ep {epoch} it {it:5d}/{total_iters} '
                      f'loss {loss.item():9.2f}', flush=True)
            it += 1
    torch.save({'decoder_state': net.RA_decoder.state_dict(),
                'det_head_state': net.detection_header.state_dict(),
                'seg_head_state': net.freespace.state_dict()}, SAVE)
    print(f'saved: {SAVE}', flush=True)

    net.eval()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_FullEvaluation(net, test_loader, enc)
    text = buf.getvalue()
    nums = {}
    for mk in ['mAP', 'mAR', 'mIoU']:
        m = re.search(rf'{mk}:?\s+([0-9.]+)', text)
        nums[mk] = float(m.group(1)) if m else None
    res = {'released_anchor': dict(mAP=0.9849, mAR=0.9170, mIoU=74.64),
           'finetuned_anchor': nums,
           'recipe': dict(epochs=args.epochs, lr=args.lr, batch=args.batch,
                          note='clean features, no bottleneck, no channel')}
    json.dump(res, open(OUT, 'w'), indent=1)
    print('fine-tuned anchor:', nums)
    print(f'saved: {OUT}')


if __name__ == '__main__':
    main()
