#!/usr/bin/env python3
"""JPEG + LDPC separation baseline (modern FEC replacing repetition code).

Pipeline per image: backbone features (1024x14x14) -> per-image min/max
normalize -> 448x448 canvas -> JPEG(Q) -> LDPC encode -> BSC(p) -> BP decode
-> JPEG decode -> features -> ResNet50Back classifier.

Methodology
-----------
The LDPC codes are characterized offline (eval/ldpc.py -> FER(p), residual
BER) on random data; valid because the code is linear and the BSC symmetric.
JPEG decoding is overwhelmingly all-or-nothing under frame errors (header /
Huffman-table corruption), so accuracy is computed analytically:

    acc(p) = mean_i [ S_i(p) * correct_clean_i + (1 - S_i(p)) * acc_dead ]
    S_i(p) = (1 - FER(p)) ** n_frames_i ,  n_frames_i = ceil(bits_i / k)

acc_dead is measured by classifying the all-zero canvas. The all-or-nothing
assumption is verified by Monte Carlo spot checks at transition points
(real per-frame error sampling + real residual bit flips + real JPEG decode).

Two bandwidth regimes:
  unconstrained: JPEG Q=50 (same as the paper's JPEG+Conv baseline)
  matched:       JPEG quality binary-searched per image so that the coded
                 payload fits SpikeAdapt-SC's rho=0.75 budget (42336 channel
                 bits/image), i.e. equal spectral efficiency.

Output: eval/seed_results/jpeg_ldpc_results.json

Usage:
    python eval/eval_jpeg_ldpc.py
"""

import os, sys, json, math, random
from io import BytesIO

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train_aid_v2 import ResNet50Front, ResNet50Back
from run_final_pipeline import AIDDataset5050, RESISC45Dataset
from eval.ldpc import make_regular_ldpc, MinSumDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPIKEADAPT_BITS = 42336          # rho=0.75 payload (channel bits / image)
BER_GRID = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
            0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
MC_CHECK_POINTS = [0.06, 0.08]   # transition-region spot checks (rate 1/2)
MC_CHECK_N = 300                 # images per spot check


# --------------------------------------------------------------- JPEG helpers
def feat_to_canvas(f):
    """(1024,14,14) float tensor -> (448,448) uint8 canvas + (fmin,fmax)."""
    fmin, fmax = f.min().item(), f.max().item()
    fn = ((f - fmin) / (fmax - fmin + 1e-8) * 255).clamp(0, 255).byte()
    C, H, W = fn.shape
    nr = int(math.ceil(math.sqrt(C)))
    nc = int(math.ceil(C / nr))
    canvas = torch.zeros(nr * H, nc * W, dtype=torch.uint8)
    for c in range(C):
        r, co = divmod(c, nc)
        canvas[r*H:(r+1)*H, co*W:(co+1)*W] = fn[c].cpu()
    return canvas.numpy(), (fmin, fmax), (nr, nc, C, H, W)


def canvas_to_feat(arr, fminmax, dims):
    fmin, fmax = fminmax
    nr, nc, C, H, W = dims
    rc = torch.tensor(arr, dtype=torch.float32)
    rf = torch.zeros(C, H, W)
    for c in range(C):
        r, co = divmod(c, nc)
        rf[c] = rc[r*H:(r+1)*H, co*W:(co+1)*W]
    return rf / 255.0 * (fmax - fmin) + fmin


def jpeg_encode(canvas_np, quality):
    buf = BytesIO()
    Image.fromarray(canvas_np, 'L').save(buf, format='JPEG', quality=quality)
    return buf.getvalue()


def jpeg_decode(jpeg_bytes, shape):
    try:
        ri = Image.open(BytesIO(jpeg_bytes)).convert('L')
        arr = np.array(ri)
        if arr.shape != shape:
            return None
        return arr
    except Exception:
        return None


def quality_for_budget(canvas_np, budget_bits):
    """Largest JPEG quality in [5,95] whose size fits the bit budget."""
    lo, hi, best = 5, 95, None
    while lo <= hi:
        mid = (lo + hi) // 2
        size = len(jpeg_encode(canvas_np, mid)) * 8
        if size <= budget_bits:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best  # None if even Q=5 overflows


# ------------------------------------------------------------------ pipeline
@torch.no_grad()
def collect_clean_stats(front, back, loader, n_classes, budget_bits):
    """One pass: per-image JPEG sizes + clean correctness for both regimes,
    plus dead-canvas accuracy."""
    front.eval(); back.eval()
    stats = dict(q50=dict(bits=[], correct=[]),
                 matched=dict(bits=[], correct=[], quality=[]))
    labels_all = []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        feats = front(imgs)
        for i in range(imgs.size(0)):
            canvas, mm, dims = feat_to_canvas(feats[i])
            labels_all.append(int(labels[i]))
            # --- unconstrained Q=50
            jb = jpeg_encode(canvas, 50)
            arr = jpeg_decode(jb, canvas.shape)
            rf = canvas_to_feat(arr, mm, dims).unsqueeze(0).to(device)
            pred = back(rf).argmax(1).item()
            stats['q50']['bits'].append(len(jb) * 8)
            stats['q50']['correct'].append(int(pred == int(labels[i])))
            # --- matched bandwidth (source bits <= budget)
            q = quality_for_budget(canvas, budget_bits)
            if q is None:
                stats['matched']['bits'].append(budget_bits)
                stats['matched']['correct'].append(0)
                stats['matched']['quality'].append(0)
            else:
                jb2 = jpeg_encode(canvas, q)
                arr2 = jpeg_decode(jb2, canvas.shape)
                rf2 = canvas_to_feat(arr2, mm, dims).unsqueeze(0).to(device)
                pred2 = back(rf2).argmax(1).item()
                stats['matched']['bits'].append(len(jb2) * 8)
                stats['matched']['correct'].append(int(pred2 == int(labels[i])))
                stats['matched']['quality'].append(q)

    # dead-canvas accuracy: predict from all-zero features
    zero_feat = torch.zeros(1, 1024, 14, 14, device=device)
    dead_pred = back(zero_feat).argmax(1).item()
    acc_dead = 100.0 * np.mean([l == dead_pred for l in labels_all])
    return stats, acc_dead


def analytic_accuracy(bits, correct, fer, k_info, acc_dead):
    bits = np.asarray(bits, float)
    correct = np.asarray(correct, float)
    n_frames = np.ceil(bits / k_info)
    survive = (1.0 - fer) ** n_frames
    return float(100.0 * np.mean(survive * correct
                                 + (1.0 - survive) * acc_dead / 100.0))


@torch.no_grad()
def mc_spot_check(front, back, loader, code, dec, p, quality, n_images):
    """Real frame-error sampling + real residual flips + real JPEG decode."""
    H = code
    k_info = dec.n - dec.m
    front.eval(); back.eval()
    rng = np.random.default_rng(int(p * 1000))
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        feats = front(imgs)
        for i in range(imgs.size(0)):
            if total >= n_images:
                return 100.0 * correct / total
            canvas, mm, dims = feat_to_canvas(feats[i])
            jb = jpeg_encode(canvas, quality)
            bits = np.unpackbits(np.frombuffer(jb, dtype=np.uint8))
            nf = int(math.ceil(len(bits) / k_info))
            pad = nf * k_info - len(bits)
            bits_p = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
            frames = bits_p.reshape(nf, k_info)
            # systematic encoding unnecessary: decode all-zero-equivalent —
            # sample per-frame channel errors and BP-decode the error pattern;
            # XOR surviving error pattern onto the data (linearity).
            err = (rng.random((nf, dec.n)) < p).astype(np.uint8)
            decoded_err = dec.decode(err, max(p, 1e-3))
            frames_rx = (frames + decoded_err[:, :k_info]) % 2
            bits_rx = frames_rx.reshape(-1)[:len(bits)]
            jb_rx = np.packbits(bits_rx).tobytes()[:len(jb)]
            arr = jpeg_decode(jb_rx, canvas.shape)
            if arr is None:
                arr = np.zeros(canvas.shape, dtype=np.uint8)
            rf = canvas_to_feat(arr, mm, dims).unsqueeze(0).to(device)
            pred = back(rf).argmax(1).item()
            correct += int(pred == int(labels[i]))
            total += 1
    return 100.0 * correct / total


def main():
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    print(f'Device: {device}', flush=True)
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                         T.Normalize((.485, .456, .406), (.229, .224, .225))])

    with open('eval/seed_results/ldpc_characterization.json') as f:
        codes = json.load(f)

    results = {'spikeadapt_bits': SPIKEADAPT_BITS}
    for ds_name, ds_key, n_classes, DsCls, ds_kwargs, bb_path in [
        ('AID', 'aid', 30, AIDDataset5050, dict(seed=42),
         './snapshots_aid_5050_seed42/backbone_best.pth'),
        ('RESISC45', 'resisc45', 45, RESISC45Dataset,
         dict(train_ratio=0.20, seed=42),
         './snapshots_resisc45_5050_seed42/backbone_best.pth'),
    ]:
        print(f'\n===== {ds_name} =====', flush=True)
        test_ds = DsCls('./data', tf_test, 'test', **ds_kwargs)
        loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)

        front = ResNet50Front(grid_size=14).to(device)
        bb = torch.load(bb_path, map_location=device, weights_only=False)
        front.load_state_dict({k: v for k, v in bb.items()
                               if not k.startswith(('layer4.', 'fc.',
                                                    'avgpool.', 'spatial_pool.'))},
                              strict=False)
        back = ResNet50Back(n_classes).to(device)
        back.load_state_dict({k: v for k, v in bb.items()
                              if k.startswith(('layer4.', 'fc.', 'avgpool.'))},
                             strict=False)
        front.eval(); back.eval()

        # matched-bandwidth source budget: rate-1/2 code -> half the channel bits
        budget_r12 = SPIKEADAPT_BITS // 2
        print(f'collecting clean stats (budget={budget_r12} source bits)...',
              flush=True)
        stats, acc_dead = collect_clean_stats(front, back, loader, n_classes,
                                              budget_r12)
        q50_bits = float(np.mean(stats['q50']['bits']))
        print(f'  Q50 mean size: {q50_bits/8/1024:.1f} KiB '
              f'({q50_bits/SPIKEADAPT_BITS:.1f}x SpikeAdapt channel bits at r=1/2: '
              f'{2*q50_bits/SPIKEADAPT_BITS:.1f}x)', flush=True)
        print(f'  clean acc Q50: {100*np.mean(stats["q50"]["correct"]):.2f}%  '
              f'matched: {100*np.mean(stats["matched"]["correct"]):.2f}% '
              f'(median Q={np.median(stats["matched"]["quality"]):.0f})  '
              f'dead-canvas acc: {acc_dead:.2f}%', flush=True)

        ds_res = {'acc_dead': round(acc_dead, 2),
                  'clean_q50': round(100*float(np.mean(stats['q50']['correct'])), 2),
                  'clean_matched': round(100*float(np.mean(stats['matched']['correct'])), 2),
                  'mean_bits_q50': q50_bits,
                  'mean_bits_matched': float(np.mean(stats['matched']['bits'])),
                  'sweeps': {}, 'mc_check': {}}

        for code_name, regime in [('r12_n648_dv3dc6', 'q50'),
                                  ('r12_n648_dv3dc6', 'matched'),
                                  ('r13_n648_dv4dc6', 'q50')]:
            cs = codes[code_name]
            k_info = cs['n'] - cs['n'] * cs['dv'] // cs['dc']
            key = f'{code_name}_{regime}'
            sweep = {}
            for p in BER_GRID:
                pk = f'{p:.4f}'
                fer = cs['stats'][pk]['fer'] if pk in cs['stats'] else None
                if fer is None:
                    continue
                acc = analytic_accuracy(stats[regime]['bits'],
                                        stats[regime]['correct'],
                                        fer, k_info, acc_dead)
                sweep[str(p)] = round(acc, 2)
            ds_res['sweeps'][key] = sweep
            print(f'  {key}: ' + '  '.join(
                f'{p}:{a:.1f}' for p, a in list(sweep.items())[:10]), flush=True)

        # Monte Carlo spot checks (rate 1/2, Q50)
        H = make_regular_ldpc(648, 3, 6, seed=0)
        dec = MinSumDecoder(H, max_iter=50, method='sumprod')
        for p in MC_CHECK_POINTS:
            acc_mc = mc_spot_check(front, back, loader, H, dec, p, 50,
                                   MC_CHECK_N)
            ds_res['mc_check'][str(p)] = round(acc_mc, 2)
            ana = ds_res['sweeps']['r12_n648_dv3dc6_q50'].get(str(p))
            print(f'  MC check p={p}: {acc_mc:.2f}% (analytic {ana}%)',
                  flush=True)

        results[ds_key] = ds_res
        del front, back, loader
        torch.cuda.empty_cache()

    out = 'eval/seed_results/jpeg_ldpc_results.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=1)
    print(f'\nSaved: {out}', flush=True)


if __name__ == '__main__':
    main()
