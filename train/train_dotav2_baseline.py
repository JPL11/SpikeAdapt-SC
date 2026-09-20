#!/usr/bin/env python3
"""Train the yolo26n-obb baseline on DOTA-v2.0 (1024px tiled, 18 classes).

Usage:
    python train/train_dotav2_baseline.py
"""

import os
# Mitigate intermittent NVML assert in the caching allocator (driver/library
# version mismatch on this box): keep memory pressure low and use the VMM
# allocator path. Real fix is a reboot to align kernel/userspace drivers.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

from ultralytics import YOLO

LAST = 'runs/obb/runs/obb/dotav2_baseline/weights/last.pt'


def main():
    if os.path.exists(LAST):
        # Proper resume: load OUR run's last.pt (resume=True with the
        # pretrained .pt would resume the checkpoint's own foreign args).
        model = YOLO(LAST)
        model.train(resume=True)
    else:
        model = YOLO('yolo26n-obb.pt')
        model.train(
            data='/home/jpli/SemCom/datasets/DOTAv2.yaml',
            epochs=100,
            imgsz=1024,
            batch=8,
            device=0,
            project='runs/obb',
            name='dotav2_baseline',
            exist_ok=True,
            patience=30,
        )
    metrics = model.val(data='/home/jpli/SemCom/datasets/DOTAv2.yaml',
                        imgsz=1024, batch=8)
    print(f'final: mAP50={metrics.box.map50:.4f} '
          f'mAP50-95={metrics.box.map:.4f}')


if __name__ == '__main__':
    main()
