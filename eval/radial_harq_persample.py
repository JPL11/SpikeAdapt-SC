#!/usr/bin/env python3
"""Per-frame detection stats for task-aware incremental-T' HARQ (journal
deferred track). rho=1.0 fixed (HARQ rides the TEMPORAL axis). For each
frame we record, at every T' in {1..8} and BER in {0,0.1,0.2,0.3}:
  (a) RX-side decodability proxies (NO ground truth): max / mean-top3 /
      count of decoded detection confidences -- computable at the receiver;
  (b) per-confidence-threshold TP/FP/FN vs GT (IoU 0.5, range 5-100 m),
      replicating utils.metrics.GetFullMetrics exactly so mission mAP can be
      composed from per-frame decisions taken at different T'.

Leakage discipline: both val and test are dumped; the HARQ sim calibrates the
NACK threshold on VAL and scores on TEST.

Usage: python eval/radial_harq_persample.py [--seed 42]
Output: eval/seed_results/radial_harq_persample_s<seed>.npz
"""

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'radial_repo', 'FFTRadNet'))

os.environ['SPIKES_ONLY'] = '1'

from utils.metrics import (RA_to_cartesian_box, bbox_iou,          # noqa: E402
                           process_predictions_FFT)
from train.train_radial_snn import CONFIG, build_loaders           # noqa: E402
from eval.radial_grid_valtest import load_net                      # noqa: E402
import json                                                        # noqa: E402

SEED_CKPT = {'42': 'runs/fftradnet_snn_radial_so.pth',
             '123': 'runs/fftradnet_snn_radial_so_s123.pth',
             '456': 'runs/fftradnet_snn_radial_so_s456.pth'}
TPRIMES = [1, 2, 3, 4, 5, 6, 7, 8]
BERS = [0.0, 0.1, 0.2, 0.3]
THRS = np.arange(0.1, 0.96, 0.1)          # 9 confidence thresholds (as mAP)
RMIN, RMAX, IOU_TH = 5, 100, 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def frame_counts(pred, labels, thr):
    """TP,FP,FN for one frame at one confidence threshold (mirrors
    GetFullMetrics inner loop). pred: decoded [N,3]=[R,A,C]; labels:[M,>=1]."""
    tp = fp = fn = 0
    obj = process_predictions_FFT(pred, confidence_threshold=thr) \
        if len(pred) > 0 else []
    if len(obj) > 0:
        md = (obj[:, 2] + obj[:, 4]) / 2
        obj = obj[np.where((md >= RMIN) & (md <= RMAX))]
    if len(labels) > 0:
        labels = labels[np.where((labels[:, 0] >= RMIN) & (labels[:, 0] <= RMAX))]
    gt = np.asarray(RA_to_cartesian_box(labels)) if len(labels) > 0 else []
    if len(gt) > 0 and len(obj) > 0:
        used = np.zeros(len(gt))
        for p in obj:
            ids = np.where(bbox_iou(p[1:], gt) >= IOU_TH)[0]
            if len(ids) > 0:
                tp += 1
                used[ids] = 1
            else:
                fp += 1
        fn += int(np.sum(used == 0))
    elif len(gt) == 0:
        fp += len(obj)
    elif len(obj) == 0:
        fn += len(gt)
    return tp, fp, fn


def run_config(net, loader, enc, tp_steps, ber):
    net.rho, net.tprime, net.ber = 1.0, tp_steps, ber
    pmax, pmean3, pndet, TP, FP, FN = [], [], [], [], [], []
    for data in loader:
        inputs = data[0].to(device).float()
        with torch.no_grad():
            out = net(inputs)
        out_obj = out['Detection'].detach().cpu().numpy()
        labels_b = data[3]
        for pred_obj, labels in zip(out_obj, labels_b):
            dec = np.asarray(enc.decode(pred_obj, 0.05))
            conf = dec[:, 2] if dec.ndim == 2 and len(dec) else np.array([])
            s = np.sort(conf)[::-1]
            pmax.append(float(s[0]) if len(s) else 0.0)
            pmean3.append(float(s[:3].mean()) if len(s) else 0.0)
            pndet.append(int(np.sum(conf >= 0.1)))
            tps = [frame_counts(dec, np.asarray(labels), t) for t in THRS]
            TP.append([a for a, _, _ in tps])
            FP.append([b for _, b, _ in tps])
            FN.append([c for _, _, c in tps])
    return (np.array(pmax), np.array(pmean3), np.array(pndet, float),
            np.array(TP, float), np.array(FP, float), np.array(FN, float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', default='42')
    args = ap.parse_args()
    out = os.path.join(ROOT,
                       f'eval/seed_results/radial_harq_persample_s{args.seed}.npz')
    status = os.path.join(ROOT, 'eval/seed_results/harq_persample.status')
    config = json.load(open(CONFIG))
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    net = load_net(config, SEED_CKPT[args.seed])
    enc, _, val_loader, test_loader = build_loaders(config)
    loaders = {'val': val_loader, 'test': test_loader}

    store = dict(np.load(out, allow_pickle=True)) if os.path.exists(out) else {}
    store['thrs'] = THRS
    for split, loader in loaders.items():
        for ber in BERS:
            for tp in TPRIMES:
                key = f'{split}|{ber}|{tp}'
                if f'{key}|TP' in store:
                    continue
                pmax, pmean3, pndet, TP, FP, FN = run_config(
                    net, loader, enc, tp, ber)
                store[f'{key}|pmax'] = pmax
                store[f'{key}|pmean3'] = pmean3
                store[f'{key}|pndet'] = pndet
                store[f'{key}|TP'] = TP
                store[f'{key}|FP'] = FP
                store[f'{key}|FN'] = FN
                np.savez_compressed(out, **store)
                msg = (f'{split} ber{ber} T{tp}: F={len(pmax)} '
                       f'mAP@this={_map(TP, FP):.3f}')
                print(msg, flush=True)
                open(status, 'a').write(msg + '\n')
    print('saved:', out, flush=True)
    open(status, 'a').write(f'DONE {out}\n')


def _map(TP, FP):
    """Quick dataset mAP (mean-over-threshold precision) for a sanity print."""
    TP, FP = np.array(TP), np.array(FP)
    prec = []
    for j in range(TP.shape[1]):
        tp, fp = TP[:, j].sum(), FP[:, j].sum()
        prec.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    return float(np.mean(prec))


if __name__ == '__main__':
    main()
