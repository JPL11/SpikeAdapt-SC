"""JPEG+Conv baseline — separate source/channel coding comparison.

Pipeline: backbone(layer1-3) → features(1024×14×14) → JPEG compress →
          repetition code (R=3) → BSC → majority vote → JPEG decompress →
          ResNet50Back(layer4+fc) → classification

Same feature insertion point as SNN/CNN baselines. No training needed.
Known for "cliff effect" at high BER due to JPEG sensitivity.
"""
import os, sys, json, math, time
import numpy as np
from io import BytesIO
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
from train_aid_v2 import ResNet50Front, ResNet50Back
from run_final_pipeline import AIDDataset5050

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RESISC45_2080:
    def __init__(self, root, transform, split='test', seed=42):
        full = ImageFolder(os.path.join(root, 'NWPU-RESISC45'), transform=transform)
        rng = np.random.RandomState(seed)
        indices = list(range(len(full)))
        rng.shuffle(indices)
        n_train = int(0.2 * len(full))
        self.dataset = full
        self.indices = indices[n_train:] if split == 'test' else indices[:n_train]
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx): return self.dataset[self.indices[idx]]


class JPEGConv(nn.Module):
    """JPEG + repetition code baseline.

    Compresses feature maps to JPEG, applies repetition coding through BSC,
    majority-vote decodes, JPEG decompresses. Demonstrates cliff effect.
    """
    def __init__(self, jpeg_quality=50, code_rate_inv=3):
        super().__init__()
        self.Q = jpeg_quality
        self.R = code_rate_inv

    def forward(self, feat, ber=0.0):
        B, C, H, W = feat.shape
        results = []
        for b_idx in range(B):
            f = feat[b_idx]
            fmin, fmax = f.min().item(), f.max().item()
            fn = ((f - fmin) / (fmax - fmin + 1e-8) * 255).clamp(0, 255).byte()
            nr = int(math.ceil(math.sqrt(C)))
            nc = int(math.ceil(C / nr))
            canvas = torch.zeros(nr * H, nc * W, dtype=torch.uint8)
            for c in range(C):
                r, co = divmod(c, nc)
                canvas[r*H:(r+1)*H, co*W:(co+1)*W] = fn[c].cpu()
            img = Image.fromarray(canvas.numpy(), 'L')
            buf = BytesIO()
            img.save(buf, format='JPEG', quality=self.Q)
            jpeg_bytes = buf.getvalue()
            if ber > 0:
                src = np.unpackbits(np.frombuffer(jpeg_bytes, dtype=np.uint8))
                coded = np.repeat(src, self.R)
                flip = (np.random.random(len(coded)) < ber)
                coded_n = (coded + flip.astype(int)) % 2
                decoded = (coded_n.reshape(-1, self.R).sum(1) > self.R / 2).astype(np.uint8)
                pad = (8 - len(decoded) % 8) % 8
                dec_bytes = np.packbits(np.concatenate([decoded,
                                        np.zeros(pad, dtype=np.uint8)])).tobytes()[:len(jpeg_bytes)]
                try:
                    ri = Image.open(BytesIO(dec_bytes)).convert('L')
                    rc = torch.tensor(np.array(ri), dtype=torch.float32)
                    if rc.shape != (nr * H, nc * W):
                        rc = torch.zeros(nr * H, nc * W)
                except Exception:
                    rc = torch.zeros(nr * H, nc * W)
            else:
                ri = Image.open(BytesIO(jpeg_bytes)).convert('L')
                rc = torch.tensor(np.array(ri), dtype=torch.float32)
            rf = torch.zeros(C, H, W)
            for c in range(C):
                r, co = divmod(c, nc)
                try:
                    rf[c] = rc[r*H:(r+1)*H, co*W:(co+1)*W]
                except Exception:
                    pass
            rf = rf / 255.0 * (fmax - fmin) + fmin
            results.append(rf)
        return torch.stack(results).to(feat.device)


def eval_jpeg(front, back, loader, jpeg_conv, ber, n_repeat=1):
    all_accs = []
    for rep in range(n_repeat):
        correct, total = 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                feat = front(imgs)
            Fp = jpeg_conv(feat, ber=ber)
            with torch.no_grad():
                correct += back(Fp).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
        all_accs.append(100. * correct / total)
    return np.mean(all_accs), np.std(all_accs)


def run_dataset(ds_name, n_classes, backbone_path, test_ds, max_images=None):
    print(f"\n{'='*60}")
    print(f"  JPEG+Conv — {ds_name}")
    print(f"{'='*60}")

    front = ResNet50Front(grid_size=14).to(device)
    back = ResNet50Back(n_classes).to(device)

    bb = torch.load(backbone_path, map_location=device)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))},
                          strict=False)
    back.load_state_dict({k: v for k, v in bb.items()
                          if k.startswith(('layer4.', 'fc.', 'avgpool.'))},
                         strict=False)
    front.eval(); back.eval()
    print(f"  Backbone: {backbone_path}")

    # JPEG is CPU-bound and slow, use smaller batch + subsample if needed
    if max_images and len(test_ds) > max_images:
        if hasattr(test_ds, 'indices'):
            rng = np.random.RandomState(42)
            test_ds.indices = list(rng.choice(test_ds.indices, max_images, replace=False))
        print(f"  Subsampled to {len(test_ds)} images for JPEG speed")

    loader = DataLoader(test_ds, 16, False, num_workers=2, pin_memory=True)
    print(f"  Test: {len(test_ds)} images")

    jpeg_conv = JPEGConv(jpeg_quality=50, code_rate_inv=3)
    # Estimate bits per sample for rate comparison
    # JPEG Q=50 on 1024×14×14 features ≈ ~50KB × 8 = ~400K bits
    # With R=3 repetition → ~1.2M channel bits
    # vs SNN: 1024×14×14×ρ spikes (1 bit each) ≈ ~150K bits at ρ=0.75

    ber_list = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    results = {}

    for ber in ber_list:
        t0 = time.time()
        n_rep = 1 if ber == 0 else 3
        mean_acc, std_acc = eval_jpeg(front, back, loader, jpeg_conv, ber, n_repeat=n_rep)
        results[str(ber)] = {'mean': round(mean_acc, 2), 'std': round(std_acc, 2)}
        elapsed = time.time() - t0
        print(f"  BER={ber:.2f}: {mean_acc:.2f}% ±{std_acc:.2f} ({elapsed:.0f}s)")

    return results


def main():
    print(f"Device: {device}")
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])

    all_results = {}

    # AID (5000 test images)
    aid_test = AIDDataset5050(root="./data", transform=tf_test, split='test', seed=42)
    all_results['AID'] = run_dataset(
        "AID (50/50)", 30,
        "./snapshots_aid_5050_seed42/backbone_best.pth",
        aid_test
    )

    # RESISC45 (subsample to 5000 for JPEG speed)
    res_test = RESISC45_2080(root="./data", transform=tf_test, split='test', seed=42)
    all_results['RESISC45'] = run_dataset(
        "RESISC45 (20/80)", 45,
        "./snapshots_resisc45_5050_seed42/backbone_best.pth",
        res_test,
        max_images=5000
    )

    with open("eval/jpeg_conv_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY — JPEG+Conv")
    print(f"{'='*60}")
    for ds, res in all_results.items():
        print(f"\n{ds}:")
        for ber in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
            r = res[str(ber)]
            print(f"  BER={ber:.2f}: {r['mean']:.2f}% ±{r['std']:.2f}")
    print(f"\n✅ JPEG+Conv eval complete.")


if __name__ == '__main__':
    main()
