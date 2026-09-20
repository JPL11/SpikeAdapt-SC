#!/usr/bin/env python3
"""Path A stage 3: validation-protocol retrains of the Table-I baselines.

Adapted from train_cnn1bit_10seed.py / train_jscc_baseline.py /
train_jscc_adaptive.py with exactly three changes per baseline:
  1. checkpoint selection runs on a held-out VALIDATION split (same 10%
     per-seed carve as run_final_pipeline_valproto.py, test transforms);
  2. the frozen front-end loads the val-protocol backbone
     (snapshots_{ds}valp_5050_seed*, requires stage 1);
  3. the TEST split is evaluated once, at the end, over the full BER grid.
Recipes (epochs, optimizers, noise sampling, losses) are byte-identical to
the originals. Snapshots: snapshots_{ds}valp_{1bit,jscc,jscc_adaptive}_seed*.

Usage: python train/train_baselines_valproto.py --baseline all --seeds 42 ...
Outputs: eval/seed_results/valproto_{cnn1bit,jscc,jscc_adaptive}.json
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'models'))
from run_final_pipeline import AIDDataset5050, RESISC45Dataset      # noqa: E402
from train_aid_v2 import ResNet50Front, ResNet50Back                # noqa: E402
from train_1bit_baseline import BinaryCNN_SC                        # noqa: E402
from train_jscc_baseline import JSCC_Model                          # noqa: E402
from train_jscc_adaptive import AdaptiveJSCC                        # noqa: E402

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL_SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]
BERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
VAL_FRAC = 0.10

TF_TRAIN = T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
                      T.ToTensor(), T.Normalize((.485, .456, .406), (.229, .224, .225))])
TF_TEST = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                     T.Normalize((.485, .456, .406), (.229, .224, .225))])


def make_loaders(ds, seed):
    if ds == 'aid':
        tr_aug = AIDDataset5050('./data', TF_TRAIN, 'train', seed=seed)
        tr_plain = AIDDataset5050('./data', TF_TEST, 'train', seed=seed)
        te = AIDDataset5050('./data', TF_TEST, 'test', seed=seed)
        ncls = 30
    else:
        tr_aug = RESISC45Dataset('./data', TF_TRAIN, 'train', train_ratio=0.20, seed=seed)
        tr_plain = RESISC45Dataset('./data', TF_TEST, 'train', train_ratio=0.20, seed=seed)
        te = RESISC45Dataset('./data', TF_TEST, 'test', train_ratio=0.20, seed=seed)
        ncls = 45
    idx = np.random.default_rng(seed).permutation(len(tr_aug))
    n_val = int(round(VAL_FRAC * len(tr_aug)))
    train_loader = DataLoader(Subset(tr_aug, idx[:-n_val].tolist()), 32, True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(Subset(tr_plain, idx[-n_val:].tolist()), 32, False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(te, 32, False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader, ncls


def load_front_back(ds, ncls, seed):
    bb_path = f'./snapshots_{ds}valp_5050_seed{seed}/backbone_best.pth'
    if not os.path.exists(bb_path):
        return None, None
    front = ResNet50Front(grid_size=14).to(device)
    bb = torch.load(bb_path, map_location=device, weights_only=False)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.',
                                                'spatial_pool.'))}, strict=False)
    front.eval()
    for p in front.parameters():
        p.requires_grad = False
    back = ResNet50Back(ncls).to(device)
    back.load_state_dict({k: v for k, v in bb.items()
                          if k.startswith(('layer4.', 'fc.', 'avgpool.',
                                           'spatial_pool.'))}, strict=False)
    return front, back


def acc_on(loader, fwd):
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            correct += fwd(imgs).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total


def select_loop(snap_dir, prefix, epochs, cadence, train_epoch, val_fwd,
                model, back, val_loader):
    """Shared train/val-select loop; saves {prefix}_best_{valacc}.pth."""
    os.makedirs(snap_dir, exist_ok=True)
    existing = sorted([f for f in os.listdir(snap_dir) if f.startswith(prefix)],
                      key=lambda x: float(x.split('_')[-1][:-4]))
    if existing:
        print(f'  found existing {existing[-1]}', flush=True)
        return existing
    best = 0
    for epoch in range(1, epochs + 1):
        model.train(); back.train()
        train_epoch()
        if epoch % cadence == 0 or epoch == epochs:
            model.eval(); back.eval()
            acc = acc_on(val_loader, val_fwd)
            print(f'  E{epoch:02d} (val): {acc:.2f}%', flush=True)
            if acc > best:
                best = acc
                torch.save({'model': model.state_dict(),
                            'back': back.state_dict()},
                           os.path.join(snap_dir, f'{prefix}{acc:.2f}.pth'))
    return sorted([f for f in os.listdir(snap_dir) if f.startswith(prefix)],
                  key=lambda x: float(x.split('_')[-1][:-4]))


def run_cnn1bit(ds, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    train_loader, val_loader, test_loader, ncls = make_loaders(ds, seed)
    front, back = load_front_back(ds, ncls, seed)
    if front is None:
        print(f'  {ds} seed {seed}: valp backbone missing, skip'); return None
    model = BinaryCNN_SC(C_in=1024, C1=256, C2=36).to(device)
    opt = optim.Adam(list(model.parameters()) + list(back.parameters()), lr=1e-4)
    crit = nn.CrossEntropyLoss()

    def train_epoch():
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            ber = random.choice([0.0, 0.0, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
            Fp, _ = model(feat, ber=ber)
            loss = crit(back(Fp), labels)
            opt.zero_grad(); loss.backward(); opt.step()

    val_fwd = lambda imgs: back(model(front(imgs), ber=0.0)[0])
    cks = select_loop(f'./snapshots_{ds}valp_1bit_seed{seed}/', '1bit_best_',
                      60, 10, train_epoch, val_fwd, model, back, val_loader)
    ck = torch.load(os.path.join(f'./snapshots_{ds}valp_1bit_seed{seed}/',
                                 cks[-1]), map_location=device, weights_only=False)
    model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
    model.eval(); back.eval()
    res = {'selected': cks[-1]}
    for ber in BERS:
        res[str(ber)] = round(acc_on(
            test_loader, lambda im: back(model(front(im), ber=ber)[0])), 2)
        print(f'  TEST {ds} s{seed} 1bit BER {ber:.2f}: {res[str(ber)]}%', flush=True)
    del model, front, back; torch.cuda.empty_cache()
    return res


def run_jscc(ds, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    train_loader, val_loader, test_loader, ncls = make_loaders(ds, seed)
    front, back = load_front_back(ds, ncls, seed)
    if front is None:
        print(f'  {ds} seed {seed}: valp backbone missing, skip'); return None
    model = JSCC_Model(C_in=1024, C_mid=256, C_tx=36).to(device)
    opt = optim.AdamW(list(model.parameters()) + list(back.parameters()),
                      lr=1e-4, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60, eta_min=1e-6)
    crit = nn.CrossEntropyLoss()

    def train_epoch():
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            snr = random.choice([100, 20, 10, 5, 2, 0, -2, -4])
            Fp, _ = model(feat, snr_db=snr, channel='awgn')
            loss = crit(back(Fp), labels)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

    val_fwd = lambda imgs: back(model(front(imgs), snr_db=100, channel='awgn')[0])
    cks = select_loop(f'./snapshots_{ds}valp_jscc_seed{seed}/', 'jscc_best_',
                      60, 10, train_epoch, val_fwd, model, back, val_loader)
    ck = torch.load(os.path.join(f'./snapshots_{ds}valp_jscc_seed{seed}/',
                                 cks[-1]), map_location=device, weights_only=False)
    model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
    model.eval(); back.eval()
    res = {'selected': cks[-1], 'awgn': {}, 'bsc': {}}
    res['clean'] = round(acc_on(test_loader, lambda im: back(
        model(front(im), snr_db=100, channel='awgn')[0])), 2)
    for snr in [10, 5, 0, -2, -4, -6, -8]:
        res['awgn'][str(snr)] = round(acc_on(test_loader, lambda im: back(
            model(front(im), snr_db=snr, channel='awgn')[0])), 2)
    for ber in BERS:
        res['bsc'][str(ber)] = round(acc_on(test_loader, lambda im: back(
            model(front(im), channel='bsc', ber=ber, snr_db=100)[0])), 2)
    print(f'  TEST {ds} s{seed} jscc clean {res["clean"]} '
          f'bsc0.3 {res["bsc"]["0.3"]}', flush=True)
    del model, front, back; torch.cuda.empty_cache()
    return res


def run_jscc_adaptive(ds, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    train_loader, val_loader, test_loader, ncls = make_loaders(ds, seed)
    front, back = load_front_back(ds, ncls, seed)
    if front is None:
        print(f'  {ds} seed {seed}: valp backbone missing, skip'); return None
    jdir = f'./snapshots_{ds}valp_jscc_seed{seed}/'
    jcks = sorted([f for f in os.listdir(jdir) if f.startswith('jscc_best_')],
                  key=lambda x: float(x.split('_')[-1][:-4])) if os.path.isdir(jdir) else []
    if not jcks:
        print(f'  {ds} seed {seed}: valp JSCC checkpoint missing, skip'); return None
    jck = torch.load(os.path.join(jdir, jcks[-1]), map_location=device,
                     weights_only=False)
    model = AdaptiveJSCC(C_in=1024, C_mid=256, C_tx=36, target_rate=0.75).to(device)
    enc = {k.replace('encoder.', ''): v for k, v in jck['model'].items()
           if k.startswith('encoder.')}
    dec = {k.replace('decoder.', ''): v for k, v in jck['model'].items()
           if k.startswith('decoder.')}
    model.encoder.encoder.load_state_dict(enc)
    model.decoder.decoder.load_state_dict(dec)
    back.load_state_dict(jck['back'])
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.decoder.parameters():
        p.requires_grad = False
    opt = optim.AdamW(list(model.scorer.parameters())
                      + list(model.block_mask.parameters())
                      + list(back.parameters()), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30, eta_min=1e-5)
    crit = nn.CrossEntropyLoss()

    def train_epoch():
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            snr = random.choice([100, 20, 10, 5, 2, 0, -2, -4])
            Fp, info = model(feat, snr_db=snr, channel='awgn')
            loss_ce = crit(back(Fp), labels)
            loss_rate = (info['tx_rate'] - 0.75) ** 2
            with torch.no_grad():
                z_norm, _ = model.encoder(feat)
            loss_div = model.scorer.compute_diversity_loss(z_norm.detach())
            loss = loss_ce + 0.5 * loss_rate + 0.05 * loss_div
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

    val_fwd = lambda imgs: back(model(front(imgs), snr_db=100, channel='awgn')[0])
    cks = select_loop(f'./snapshots_{ds}valp_jscc_adaptive_seed{seed}/',
                      'adaptive_best_', 30, 5, train_epoch, val_fwd, model,
                      back, val_loader)
    ck = torch.load(os.path.join(f'./snapshots_{ds}valp_jscc_adaptive_seed{seed}/',
                                 cks[-1]), map_location=device, weights_only=False)
    model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
    model.eval(); back.eval()
    res = {'selected': cks[-1], 'bsc_r075': {}}
    for ber in BERS:
        res['bsc_r075'][str(ber)] = round(acc_on(test_loader, lambda im: back(
            model(front(im), channel='bsc', ber=ber, snr_db=100)[0])), 2)
    print(f'  TEST {ds} s{seed} jscc-adaptive bsc0.3 '
          f'{res["bsc_r075"]["0.3"]}', flush=True)
    del model, front, back; torch.cuda.empty_cache()
    return res


RUNNERS = {'cnn1bit': run_cnn1bit, 'jscc': run_jscc,
           'jscc_adaptive': run_jscc_adaptive}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', choices=list(RUNNERS) + ['all'], default='all')
    ap.add_argument('--seeds', type=int, nargs='+', default=ALL_SEEDS)
    ap.add_argument('--datasets', nargs='+', default=['aid', 'resisc45'])
    args = ap.parse_args()
    names = list(RUNNERS) if args.baseline == 'all' else [args.baseline]
    for name in names:                      # jscc before jscc_adaptive in dict order
        out = os.path.join(ROOT, f'eval/seed_results/valproto_{name}.json')
        res = json.load(open(out)) if os.path.exists(out) else {}
        for ds in args.datasets:
            node = res.setdefault(ds, {})
            for seed in args.seeds:
                if str(seed) in node:
                    continue
                print(f'== {name} {ds} seed {seed}', flush=True)
                r = RUNNERS[name](ds, seed)
                if r is not None:
                    node[str(seed)] = r
                    json.dump(res, open(out, 'w'), indent=1)
        print(f'{name}: saved {out}', flush=True)


if __name__ == '__main__':
    main()
