#!/usr/bin/env python3
"""SNN bottleneck on FFTRadNet (RADIal): train + mAP-vs-BER sweep.

Array/ISAC follow-up paper, primary track. Mirrors the DOTA recipe
(train/train_dotav2_snn.py): pretrained task net, spiking bottleneck at the
multi-scale feature interface, BER curriculum to 0.40, then a BER sweep.

Split: the (UAV-side) FFTRadNet FPN encoder is FROZEN with pretrained
weights; its x2/x3/x4 features (160/192/224 ch) pass through a
SpikeAdaptSC_Det_Multi bottleneck (C_spike=36, T=8, learned importance
masking at target_rate=0.75) over a BSC; the (ground-side) RA decoder +
detection/segmentation heads are fine-tuned on the reconstructions.
SNN levels are fresh-init (the DOTA optA warm-start has different channel
sizes and cannot be reused).

Usage (semcom conda python, repo root):
    python train/train_radial_snn.py --phase both [--epochs 10] [--batch 4]
Outputs:
    runs/fftradnet_snn_radial.pth
    eval/seed_results/radial_snn_ber_sweep.json
"""

import argparse
import copy
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'radial_repo', 'FFTRadNet'))

from model.FFTRadNet import FFTRadNet                      # noqa: E402
from dataset.dataset import RADIal                         # noqa: E402
from dataset.encoder import ra_encoder                     # noqa: E402
from dataset.dataloader import CreateDataLoaders           # noqa: E402
from loss import pixor_loss                                # noqa: E402
from utils.evaluation import run_evaluation                # noqa: E402
from train.train_dota_v6d2d import SpikeAdaptSC_Det_Multi  # noqa: E402

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONFIG = os.path.join(ROOT, 'radial_repo/FFTRadNet/config/config_local.json')
BASE_CKPT = os.path.join(
    ROOT, 'data/RADIal/FFTRadNet_RA_192_56_epoch78_loss_172.8239_AP_0.9813.pth')
SAVE = os.path.join(ROOT, 'runs/fftradnet_snn_radial.pth')
OUT_JSON = os.path.join(ROOT, 'eval/seed_results/radial_snn_ber_sweep.json')

SNN_LEVELS = ['x2', 'x3', 'x4']
CHANNEL_SIZES = [160, 192, 224]
BERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def load_base(config):
    net = FFTRadNet(blocks=config['model']['backbone_block'],
                    mimo_layer=config['model']['MIMO_output'],
                    channels=config['model']['channels'],
                    regression_layer=2,
                    detection_head=config['model']['DetectionHead'],
                    segmentation_head=config['model']['SegmentationHead'])
    ck = torch.load(BASE_CKPT, map_location='cpu', weights_only=False)
    sd = {}
    for k, v in ck['net_state_dict'].items():   # pre-refactor key names
        k = k.replace('backbone.preproc.', 'FPN.pre_enc.')
        k = k.replace('backbone.', 'FPN.')
        k = k.replace('RAmap_header.', 'RA_decoder.')
        sd[k] = v
    net.load_state_dict(sd)
    return net


class FFTRadNetSNN(nn.Module):
    """Frozen FPN encoder -> SNN bottleneck on x2/x3/x4 -> decoder + heads.

    Same call signature as FFTRadNet so run_evaluation works unchanged;
    set .ber (and optionally .rho) before calling.
    """

    def __init__(self, base, snn):
        super().__init__()
        self.FPN = base.FPN
        self.snn = snn
        self.RA_decoder = base.RA_decoder
        self.detection_header = base.detection_header
        self.freespace = base.freespace
        self.ber = 0.0
        self.rho = None
        self.last_infos = []
        for p in self.FPN.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.FPN.eval()          # frozen: keep BN running stats
        return self

    def forward(self, x):
        with torch.no_grad():
            feats = self.FPN(x)
        recons, self.last_infos = self.snn(
            [feats[k] for k in SNN_LEVELS], ber=self.ber,
            target_rate_override=self.rho)
        f = dict(feats)
        for k, r in zip(SNN_LEVELS, recons):
            f[k] = r
        RA = self.RA_decoder(f)
        out = {'Detection': self.detection_header(RA)}
        Y = F.interpolate(RA, (256, 224))
        out['Segmentation'] = self.freespace(Y)
        return out


def payload_bits(net, config):
    """Spike payload per frame at full rate (before importance masking)."""
    h, w = 512, 256
    bits = 0
    for lev, c in zip(SNN_LEVELS, CHANNEL_SIZES):
        stride = {'x2': 4, 'x3': 8, 'x4': 16}[lev]
        c_spike = net.snn.levels[0].encoder.spike_convs[-1].out_channels \
            if hasattr(net.snn.levels[0].encoder, 'spike_convs') else 36
        bits += c_spike * (h // stride) * (w // stride)
    T = net.snn.levels[0].decoder.T
    return bits * T


def build_loaders(config, batch=None):
    cfg = copy.deepcopy(config)
    if batch:
        cfg['dataloader']['train']['batch_size'] = batch
    enc = ra_encoder(geometry=cfg['dataset']['geometry'],
                     statistics=cfg['dataset']['statistics'],
                     regression_layer=2)
    dataset = RADIal(root_dir=cfg['dataset']['root_dir'],
                     statistics=cfg['dataset']['statistics'],
                     encoder=enc.encode, difficult=False)
    train_loader, val_loader, test_loader = CreateDataLoaders(
        dataset, cfg['dataloader'], cfg['seed'])
    return enc, train_loader, val_loader, test_loader


def train_phase(args, config, net, train_loader):
    seg_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
    trainable = [p for p in net.parameters() if p.requires_grad]
    print(f'trainable params: {sum(p.numel() for p in trainable):,}', flush=True)
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    total_iters = args.epochs * len(train_loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total_iters)
    rng = np.random.default_rng(getattr(args, 'rng_seed', config['seed']))
    net.train()
    it = 0
    for epoch in range(args.epochs):
        for data in train_loader:
            inputs = data[0].to(device).float()
            label_map = data[1].to(device).float()
            seg_label = data[2].to(device).float()
            # BER curriculum: cap ramps 0 -> 0.40 over first 60% of training;
            # a zero_frac share of batches trains at exactly BER=0 so the
            # clean channel stays in-distribution
            cap = 0.40 * min(1.0, it / max(1, int(total_iters * 0.6)))
            if cap <= 0 or rng.random() < args.zero_frac:
                net.ber = 0.0
            else:
                net.ber = float(rng.uniform(0.0, cap))
            outputs = net(inputs)
            cls_loss, reg_loss = pixor_loss(outputs['Detection'], label_map,
                                            config['losses'])
            seg_loss = seg_loss_fn(outputs['Segmentation'].flatten(),
                                   seg_label.flatten()) * inputs.size(0)
            w = config['losses']['weight']
            loss = cls_loss * w[0] + reg_loss * w[1] + seg_loss * w[2]
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if it % 100 == 0:
                tx = np.mean([i_['tx_rate'] for i_ in net.last_infos])
                print(f'ep {epoch} it {it:5d}/{total_iters} '
                      f'loss {loss.item():9.2f} ber {net.ber:.3f} '
                      f'tx_rate {tx:.3f}', flush=True)
            it += 1
            if args.smoke and it >= 30:
                print('smoke: stopping after 30 iters', flush=True)
                break
        if args.smoke:
            break
    os.makedirs(os.path.dirname(SAVE), exist_ok=True)
    torch.save({'snn_state': net.snn.state_dict(),
                'decoder_state': net.RA_decoder.state_dict(),
                'det_head_state': net.detection_header.state_dict(),
                'seg_head_state': net.freespace.state_dict(),
                'config': dict(channel_sizes=CHANNEL_SIZES, C_spike=36, T=8,
                               target_rate=0.75, levels=SNN_LEVELS,
                               epochs=args.epochs, lr=args.lr)}, SAVE)
    print(f'saved: {SAVE}', flush=True)


def sweep_phase(args, config, net, enc, test_loader):
    if os.path.exists(SAVE) and not args.smoke:
        ck = torch.load(SAVE, map_location=device, weights_only=False)
        net.snn.load_state_dict(ck['snn_state'])
        net.RA_decoder.load_state_dict(ck['decoder_state'])
        net.detection_header.load_state_dict(ck['det_head_state'])
        net.freespace.load_state_dict(ck['seg_head_state'])
        print(f'loaded {SAVE}', flush=True)
    net.eval()
    results = {'config': dict(channel_sizes=CHANNEL_SIZES, levels=SNN_LEVELS,
                              C_spike=36, T=8, target_rate=0.75)}
    for ber in BERS:
        net.ber = ber
        m = run_evaluation(net, test_loader, enc, check_perf=True)
        results[f'ber{ber}'] = dict(mAP=m['mAP'], mAR=m['mAR'], mIoU=m['mIoU'])
        print(f"BER {ber:.2f}: mAP {m['mAP']:.4f}  mAR {m['mAR']:.4f}  "
              f"mIoU {m['mIoU']:.4f}", flush=True)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(results, open(OUT_JSON, 'w'), indent=1)
    print(f'saved: {OUT_JSON}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase', choices=['train', 'sweep', 'both'], default='both')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--zero-frac', dest='zero_frac', type=float, default=0.0,
                    help='fraction of batches trained at exactly BER=0')
    ap.add_argument('--train-seed', dest='train_seed', type=int, default=None,
                    help='seed for init/training randomness (data split stays '
                         'at the official config seed)')
    ap.add_argument('--tag', default='', help='suffix for ckpt/json outputs')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    global SAVE, OUT_JSON
    if args.tag:
        SAVE = SAVE.replace('.pth', f'_{args.tag}.pth')
        OUT_JSON = OUT_JSON.replace('.json', f'_{args.tag}.json')

    config = json.load(open(CONFIG))
    ts = args.train_seed if args.train_seed is not None else config['seed']
    torch.manual_seed(ts)
    np.random.seed(ts)
    args.rng_seed = ts

    base = load_base(config)
    snn = SpikeAdaptSC_Det_Multi(CHANNEL_SIZES, C_spike=36, T=8,
                                 target_rate=0.75)
    net = FFTRadNetSNN(base, snn).to(device)
    enc, train_loader, _, test_loader = build_loaders(config, args.batch)
    print(f'train batches: {len(train_loader)}  test batches: {len(test_loader)}',
          flush=True)

    if args.phase in ('train', 'both'):
        train_phase(args, config, net, train_loader)
    if args.phase in ('sweep', 'both'):
        sweep_phase(args, config, net, enc, test_loader)


if __name__ == '__main__':
    main()
