#!/usr/bin/env python3
"""Thumbnail falsifier baseline (review-panel T1).

At BER p the fixed 42,336-use airtime budget caps reliable delivery at
C(p)*n bits (5,025 @ p=0.30; 16,520 @ p=0.15). A separation designer could
send a downscaled JPEG within that cap (granting an IDEAL capacity-achieving
code -- no real FEC simulated) and run the FULL classifier at the ground
station. This script measures that baseline's accuracy.

Protocol: per image, downscale to SxS, then pick the highest JPEG quality
whose encoded size fits the byte budget (per-image adaptive quality, like a
real rate-controlled encoder). Decode, upsample to 224, classify with the
full trained ResNet-50 (front+back from backbone_best.pth, seed 42).
Best S over {32, 48, 64, 96} is reported per budget.

Output: eval/seed_results/thumbnail_baseline.json

Usage: python eval/eval_thumbnail_baseline.py
"""

import io, os, sys, json, random
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train_aid_v2 import ResNet50Front, ResNet50Back
from run_final_pipeline import AIDDataset5050, RESISC45Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = 'eval/seed_results/thumbnail_baseline.json'
SEED = 42
BUDGETS_BITS = {'0.30': 5025, '0.15': 16520}
SIZES = [32, 48, 64, 96]
NORM = T.Normalize((.485, .456, .406), (.229, .224, .225))


def thumb_jpeg(pil_img, size, budget_bytes):
    """Downscale to size x size, encode at max quality fitting the budget."""
    small = pil_img.resize((size, size), Image.LANCZOS)
    best = None
    for q in range(95, 4, -5):
        buf = io.BytesIO()
        small.save(buf, 'JPEG', quality=q)
        if buf.tell() <= budget_bytes:
            best = buf.getvalue()
            break
    if best is None:                       # even q=5 too big
        buf = io.BytesIO()
        small.save(buf, 'JPEG', quality=5)
        best = buf.getvalue()              # counted as infeasible below
        return best, False
    return best, True


class ThumbTf:
    def __init__(self, size, budget_bytes):
        self.size, self.budget = size, budget_bytes
        self.fits = []

    def __call__(self, pil):
        data, ok = thumb_jpeg(pil.convert('RGB'), self.size, self.budget)
        self.fits.append(ok)
        rec = Image.open(io.BytesIO(data)).convert('RGB')
        rec = rec.resize((224, 224), Image.BICUBIC)
        return NORM(T.functional.to_tensor(rec))


@torch.no_grad()
def evaluate(front, back, loader):
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        correct += back(front(imgs)).argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return 100. * correct / total


def main():
    print(f'Device: {device}', flush=True)
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}

    for ds_name, ds_key, n_classes in [('AID', 'aid', 30),
                                       ('RESISC45', 'resisc45', 45)]:
        bb_path = f'./snapshots_{ds_key}_5050_seed{SEED}/backbone_best.pth'
        bb = torch.load(bb_path, map_location=device, weights_only=False)
        front = ResNet50Front(grid_size=14).to(device)
        front.load_state_dict(
            {k: v for k, v in bb.items()
             if not k.startswith(('layer4.', 'fc.', 'avgpool.',
                                  'spatial_pool.'))}, strict=False)
        back = ResNet50Back(num_classes=n_classes).to(device)
        back.load_state_dict(
            {k: v for k, v in bb.items()
             if k.startswith(('layer4.', 'fc.'))}, strict=False)
        front.eval(); back.eval()

        ds_res = results.setdefault(ds_key, {})
        for ber, bits in BUDGETS_BITS.items():
            budget_bytes = bits // 8
            b_res = ds_res.setdefault(ber, {})
            for size in SIZES:
                if str(size) in b_res:
                    continue
                torch.manual_seed(1234 + SEED); np.random.seed(1234 + SEED)
                random.seed(1234 + SEED)
                tf = ThumbTf(size, budget_bytes)
                if ds_key == 'aid':
                    ds = AIDDataset5050('./data', tf, 'test', seed=SEED)
                else:
                    ds = RESISC45Dataset('./data', tf, 'test',
                                         train_ratio=0.20, seed=SEED)
                loader = DataLoader(ds, 64, False, num_workers=6)
                acc = evaluate(front, back, loader)
                fit = 100. * np.mean(tf.fits) if tf.fits else 0.0
                b_res[str(size)] = dict(acc=acc, fit_pct=fit,
                                        budget_bits=bits)
                print(f'[{ds_name}] BER={ber} budget={bits}b size={size}: '
                      f'acc={acc:.2f} (fit {fit:.1f}%)', flush=True)
                json.dump(results, open(OUT, 'w'), indent=1)

    for ds_key, ds_res in results.items():
        for ber, b_res in ds_res.items():
            best = max(b_res.items(), key=lambda kv: kv[1]['acc'])
            print(f'{ds_key} @BER={ber}: best size={best[0]} '
                  f'acc={best[1]["acc"]:.2f}', flush=True)


if __name__ == '__main__':
    main()
