#!/usr/bin/env python3
"""FAIR 1-bit control for the RADIal track: the encoder-matched, truncation-
trained binary competitor for the spiking radar codec.

Fixes the two confounds the paper discloses in its "1-bit retrained" row
(encoder-free + no temporal axis): this baseline has (a) a LEARNED per-scale
binary encoder/decoder (STE_Sign), matched to the SNN's role, (b) the SAME
BER-conditioned importance scorer + spatial masking (rho) as the SNN, and
(c) a grouped temporal axis (C2 = 8 groups x 36 = 288 binary maps) that is
TRUNCATION-TRAINED (random T' per batch), so the ANN gets its best possible
shot at the temporal axis. Payload matches the SNN exactly (C_group=36, same
scale dims -> 2.32 Mbit/frame at rho=0.75, T'=8).

If this baseline closes the +6-10 Pbar / +0.14 recall gap the paper attributes
to spiking (contribution iii), that contribution must be reframed code-agnostic
-- exactly as the visual companion found.

Trains inside the identical FFTRadNet harness (frozen FPN, pixor+BCE loss, BER
curriculum) as train_radial_snn.py / train_radial_quant.py.

Usage: /home/jpli/miniconda/envs/semcom/bin/python train/train_radial_adaptive1bit.py \
          --train-seed 42 --tag s42 [--epochs 10]
Outputs: runs/fftradnet_adaptive1bit_radial_{tag}.pth
         eval/seed_results/radial_adaptive1bit_sweep_{tag}.json   (rho=0.75,T'=8 vs BER)
         eval/seed_results/radial_adaptive1bit_temporal_{tag}.json (T' sweep)
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
sys.path.insert(0, os.path.join(ROOT, 'models'))

from loss import pixor_loss                                       # noqa: E402
from utils.evaluation import run_FullEvaluation                   # noqa: E402
from eval.radial_feature_quant_baseline import CappedEncoder      # noqa: E402
from train.train_radial_snn import (CONFIG, SNN_LEVELS, CHANNEL_SIZES,  # noqa: E402
                                    build_loaders, load_base)
from train.train_1bit_baseline import BinaryCNN_Encoder, BinaryCNN_Decoder  # noqa: E402
from noise_aware_scorer import NoiseAwareScorer                   # noqa: E402
from train.train_jscc_adaptive import LearnedBlockMask            # noqa: E402

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
N_GROUPS = 8
C_GROUP = 36
C2_TOTAL = N_GROUPS * C_GROUP           # 288
TPRIMES = [4, 6, 8]
RHOS = [0.5, 0.75, 1.0]


class Adaptive1bitDetCodec(nn.Module):
    """Per-scale grouped binary codec: learned encoder + BER-conditioned mask
    + grouped temporal truncation (payload-matched to one SNN scale)."""

    def __init__(self, C_in, C1=256, target_rate=0.75):
        super().__init__()
        self.encoder = BinaryCNN_Encoder(C_in, C1, C2_TOTAL)
        self.decoder = BinaryCNN_Decoder(C_in, C1, C2_TOTAL)
        self.scorer = NoiseAwareScorer(C_spike=C_GROUP, hidden=32)
        self.block_mask = LearnedBlockMask(target_rate, 0.5)

    def forward(self, feat, ber=0.0, tprime=N_GROUPS, target_rate_override=None,
                training=None):
        z = self.encoder(feat)                              # (B,288,H,W) binary
        B, _, H, W = z.shape
        z8 = z.view(B, N_GROUPS, C_GROUP, H, W)
        groups = [z8[:, g] for g in range(N_GROUPS)]
        importance = self.scorer(groups, ber)              # (B,1,H,W)
        train_mode = self.training if training is None else training
        if target_rate_override is not None:
            old = self.block_mask.target_rate
            self.block_mask.target_rate = target_rate_override
            mask, tx = self.block_mask(importance, training=False)
            self.block_mask.target_rate = old
        else:
            mask, tx = self.block_mask(importance, training=train_mode)
        m = mask.unsqueeze(1)                               # (B,1,1,H,W)
        send = (z8 * m).clone()
        send[:, tprime:] = 0.0
        flip = (torch.rand_like(send) < ber).float()
        recv = ((send + flip) % 2) * m
        recv[:, tprime:] = 0.0
        rec = self.decoder(recv.reshape(B, C2_TOTAL, H, W))
        return rec, {'tx_rate': tx.item(), 'kept': mask.mean().item()}


class Adaptive1bitDet_Multi(nn.Module):
    def __init__(self, channel_sizes, target_rate=0.75):
        super().__init__()
        self.levels = nn.ModuleList([
            Adaptive1bitDetCodec(c, target_rate=target_rate)
            for c in channel_sizes])

    def forward(self, feats, ber=0.0, tprime=N_GROUPS, target_rate_override=None):
        recons, infos = [], []
        for lev, feat in zip(self.levels, feats):
            r, info = lev(feat, ber=ber, tprime=tprime,
                          target_rate_override=target_rate_override)
            recons.append(r); infos.append(info)
        return recons, infos


class FFTRadNetAdaptive1bit(nn.Module):
    """Frozen FPN -> grouped-1bit bottleneck on x2/x3/x4 -> decoder + heads."""

    def __init__(self, base, codec):
        super().__init__()
        self.FPN = base.FPN
        self.codec = codec
        self.RA_decoder = base.RA_decoder
        self.detection_header = base.detection_header
        self.freespace = base.freespace
        self.ber = 0.0
        self.rho = None
        self.tprime = N_GROUPS
        self.last_infos = []
        for p in self.FPN.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.FPN.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            feats = self.FPN(x)
        recons, self.last_infos = self.codec(
            [feats[k] for k in SNN_LEVELS], ber=self.ber, tprime=self.tprime,
            target_rate_override=self.rho)
        f = dict(feats)
        for k, r in zip(SNN_LEVELS, recons):
            f[k] = r
        RA = self.RA_decoder(f)
        out = {'Detection': self.detection_header(RA)}
        out['Segmentation'] = self.freespace(F.interpolate(RA, (256, 224)))
        return out


def full_eval(net, loader, enc_c, ber, rho, tprime):
    net.eval(); net.ber = ber; net.rho = rho; net.tprime = tprime
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_FullEvaluation(net, loader, enc_c)
    text = buf.getvalue()
    out = {}
    for mk in ['mAP', 'mAR', 'mIoU']:
        m = re.search(rf'{mk}:?\s+([0-9.]+)', text)
        out[mk] = float(m.group(1)) if m else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--train-seed', dest='train_seed', type=int, default=42)
    ap.add_argument('--tag', default='s42')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    save = os.path.join(ROOT, f'runs/fftradnet_adaptive1bit_radial_{args.tag}.pth')
    sweep_json = os.path.join(
        ROOT, f'eval/seed_results/radial_adaptive1bit_sweep_{args.tag}.json')
    temporal_json = os.path.join(
        ROOT, f'eval/seed_results/radial_adaptive1bit_temporal_{args.tag}.json')

    config = json.load(open(CONFIG))
    ts = args.train_seed
    torch.manual_seed(ts); np.random.seed(ts)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(ts)

    base = load_base(config)
    codec = Adaptive1bitDet_Multi(CHANNEL_SIZES, target_rate=0.75)
    net = FFTRadNetAdaptive1bit(base, codec).to(device)
    enc, train_loader, _, test_loader = build_loaders(config, args.batch)
    print(f'train batches: {len(train_loader)}  test batches: {len(test_loader)}',
          flush=True)

    if not (os.path.exists(save) and not args.smoke):
        seg_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        trainable = [p for p in net.parameters() if p.requires_grad]
        print(f'trainable params: {sum(p.numel() for p in trainable):,}',
              flush=True)
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
                net.tprime = int(rng.choice(TPRIMES))     # truncation training
                net.rho = None                             # Gumbel mask -> ~0.75
                outputs = net(inputs)
                cls_loss, reg_loss = pixor_loss(outputs['Detection'], label_map,
                                                config['losses'])
                seg_loss = seg_loss_fn(outputs['Segmentation'].flatten(),
                                       seg_label.flatten()) * inputs.size(0)
                w = config['losses']['weight']
                tx = np.mean([i_['tx_rate'] for i_ in net.last_infos])
                rate_loss = (float(tx) - 0.75) ** 2
                loss = (cls_loss * w[0] + reg_loss * w[1] + seg_loss * w[2]
                        + 50.0 * rate_loss)
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                if it % 100 == 0:
                    print(f'ep {epoch} it {it:5d}/{total_iters} '
                          f'loss {loss.item():9.2f} ber {net.ber:.3f} '
                          f"T'{net.tprime} tx {tx:.3f}", flush=True)
                it += 1
                if args.smoke and it >= 20:
                    break
            if args.smoke:
                break
        os.makedirs(os.path.dirname(save), exist_ok=True)
        torch.save({'codec_state': net.codec.state_dict(),
                    'decoder_state': net.RA_decoder.state_dict(),
                    'det_head_state': net.detection_header.state_dict(),
                    'seg_head_state': net.freespace.state_dict()}, save)
        print(f'saved: {save}', flush=True)
    else:
        ck = torch.load(save, map_location=device, weights_only=False)
        net.codec.load_state_dict(ck['codec_state'])
        net.RA_decoder.load_state_dict(ck['decoder_state'])
        net.detection_header.load_state_dict(ck['det_head_state'])
        net.freespace.load_state_dict(ck['seg_head_state'])
        print(f'loaded {save}', flush=True)

    enc_c = CappedEncoder(enc, k=200)
    elems_per_t = sum(C_GROUP * (512 // s) * (256 // s) for s in (4, 8, 16))

    # (1) decomposition sweep: rho=0.75, T'=8 vs BER (matched 2.32 Mbit)
    res = {'payload_bits_per_frame': int(0.75 * 8 * elems_per_t),
           'rho': 0.75, 'tprime': 8}
    for ber in BERS:
        m = full_eval(net, test_loader, enc_c, ber, 0.75, 8)
        res[f'ber{ber}'] = m
        json.dump(res, open(sweep_json, 'w'), indent=1)
        print(f"[sweep] BER {ber:.2f}: mAP {m['mAP']:.4f} mAR {m['mAR']:.4f} "
              f"mIoU {m['mIoU']:.2f}", flush=True)
    print(f'saved: {sweep_json}', flush=True)

    # (2) temporal axis: T' in {4,6,8} at rho=0.75, BER in {0,0.15,0.30}
    tres = {}
    for tp in TPRIMES:
        tres[str(tp)] = {}
        for ber in [0.0, 0.15, 0.30]:
            m = full_eval(net, test_loader, enc_c, ber, 0.75, tp)
            tres[str(tp)][f'{ber:g}'] = m
            json.dump(tres, open(temporal_json, 'w'), indent=1)
            print(f"[temporal] T'{tp} BER {ber:.2f}: mAP {m['mAP']:.4f} "
                  f"mAR {m['mAR']:.4f}", flush=True)
    print(f'saved: {temporal_json}', flush=True)


if __name__ == '__main__':
    main()
