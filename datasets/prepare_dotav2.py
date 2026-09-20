#!/usr/bin/env python3
"""Assemble DOTA-v2.0 in YOLO-OBB format and tile it into 1024px patches.

Steps:
  1. Index available original images: DOTA-v1.0 originals
     (/home/jpli/MRI/datasets/DOTAv1/images/{train,val}) + newly extracted
     v2-only parts (datasets/DOTA-v2.0/extracted/{train,val}/images).
  2. For every v2.0 label file (defines v2.0 train/val membership), convert
     DOTA labelTxt -> normalized YOLO-OBB (18 classes), and link the image
     into datasets/DOTAv2_raw/images/{split}.
  3. Tile with ultralytics split_trainval (1024 crop, 200 gap) into
     datasets/DOTAv2 (final training dataset).
  4. Write DOTAv2.yaml.

Usage:
    python datasets/prepare_dotav2.py
"""

import os
import glob
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # DOTA images are huge

ROOT = Path('/home/jpli/SemCom/datasets')
V2 = ROOT / 'DOTA-v2.0/extracted'
V1_IMAGES = Path('/home/jpli/MRI/datasets/DOTAv1/images')
RAW = ROOT / 'DOTAv2_raw'
OUT = ROOT / 'DOTAv2'

CLASSES = ['plane', 'ship', 'storage-tank', 'baseball-diamond',
           'tennis-court', 'basketball-court', 'ground-track-field',
           'harbor', 'bridge', 'large-vehicle', 'small-vehicle',
           'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool',
           'container-crane', 'airport', 'helipad']
CLS_IDX = {c: i for i, c in enumerate(CLASSES)}


def index_images():
    pool = {}
    for d in [V1_IMAGES / 'train', V1_IMAGES / 'val',
              V2 / 'train/images', V2 / 'val/images']:
        if not d.exists():
            continue
        for f in d.iterdir():
            if f.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                pool.setdefault(f.stem, f)
    # extracted zips may nest a subdir (e.g. images/)
    for d in [V2 / 'train/images', V2 / 'val/images']:
        for f in d.glob('*/*'):
            if f.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                pool.setdefault(f.stem, f)
    return pool


def find_labels(split):
    base = V2 / split / 'labelTxt'
    cands = list(base.glob('*.txt')) + list(base.glob('*/*.txt'))
    return sorted(set(cands))


def convert(split, pool):
    img_out = RAW / 'images' / split
    lbl_out = RAW / 'labels' / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    labels = find_labels(split)
    n_ok, n_missing, n_badcls = 0, 0, 0
    for lf in labels:
        stem = lf.stem
        if stem not in pool:
            n_missing += 1
            continue
        src = pool[stem]
        with Image.open(src) as im:
            w, h = im.size
        lines_out = []
        with open(lf, errors='ignore') as f:
            for line in f:
                parts = line.split()
                if len(parts) < 9:
                    continue  # header lines (imagesource/gsd)
                try:
                    coords = [float(x) for x in parts[:8]]
                except ValueError:
                    continue
                cls = parts[8]
                if cls not in CLS_IDX:
                    n_badcls += 1
                    continue
                norm = []
                for k, v in enumerate(coords):
                    norm.append(min(max(v / (w if k % 2 == 0 else h), 0.0),
                                    1.0))
                lines_out.append(f"{CLS_IDX[cls]} " +
                                 ' '.join(f'{v:.6f}' for v in norm))
        with open(lbl_out / f'{stem}.txt', 'w') as f:
            f.write('\n'.join(lines_out) + ('\n' if lines_out else ''))
        dst = img_out / f'{stem}{src.suffix.lower()}'
        if not dst.exists():
            os.link(src, dst) if src.stat().st_dev == img_out.stat().st_dev \
                else dst.symlink_to(src)
        n_ok += 1
    print(f'{split}: {n_ok} images converted, {n_missing} missing images, '
          f'{n_badcls} unknown-class boxes', flush=True)
    return n_missing


def main():
    pool = index_images()
    print(f'image pool: {len(pool)} unique stems', flush=True)
    miss = 0
    for split in ['train', 'val']:
        miss += convert(split, pool)
    if miss:
        print(f'WARNING: {miss} label files had no matching image')

    print('tiling with split_trainval (1024/200)...', flush=True)
    from ultralytics.data.split_dota import split_trainval
    OUT.mkdir(parents=True, exist_ok=True)
    split_trainval(data_root=str(RAW), save_dir=str(OUT),
                   crop_size=1024, gap=200)

    yaml_path = ROOT / 'DOTAv2.yaml'
    with open(yaml_path, 'w') as f:
        f.write(f"path: {OUT}\ntrain: images/train\nval: images/val\n"
                f"names:\n" +
                ''.join(f'  {i}: {c}\n' for i, c in enumerate(CLASSES)))
    print(f'wrote {yaml_path}')
    for split in ['train', 'val']:
        n = len(glob.glob(str(OUT / 'images' / split / '*')))
        print(f'{split} patches: {n}')


if __name__ == '__main__':
    main()
