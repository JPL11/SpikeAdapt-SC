"""JPEG+Conv BER Sweep — Evaluation only, no training needed.

Runs JPEG+Conv baseline across fine-grained BER values to demonstrate
the cliff effect compared to SNN-based methods.
"""

import os, sys, json, math
import numpy as np
from io import BytesIO
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

SNAP_DIR = "./snapshots_spikeadapt/"
EVAL_DIR = "./eval_results/"
os.makedirs(EVAL_DIR, exist_ok=True)


class ResNet50Front(nn.Module):
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet50(weights=None)
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = r.bn1; self.relu = r.relu
        self.maxpool = nn.Identity()
        self.layer1 = r.layer1; self.layer2 = r.layer2
        self.layer3 = r.layer3; self.layer4 = r.layer4
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        return self.layer4(self.layer3(self.layer2(self.layer1(self.maxpool(x)))))


class ResNet50Back(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)
    def forward(self, x):
        return self.fc(torch.flatten(self.avgpool(x), 1))


class JPEGConv(nn.Module):
    """JPEG + repetition code baseline. Shows cliff effect at high BER."""
    def __init__(self, jpeg_quality=50, code_rate_inv=3):
        super().__init__()
        self.Q = jpeg_quality
        self.R = code_rate_inv

    def forward(self, feat, bit_error_rate=0.0):
        B, C, H, W = feat.shape
        results = []
        for b in range(B):
            f = feat[b]
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
            if bit_error_rate > 0:
                src = np.unpackbits(np.frombuffer(jpeg_bytes, dtype=np.uint8))
                coded = np.repeat(src, self.R)
                flip = (np.random.random(len(coded)) < bit_error_rate)
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


if __name__ == "__main__":
    print("=" * 60)
    print("JPEG+Conv BER Sweep (evaluation only)")
    print("=" * 60)

    test_tf = T.Compose([T.ToTensor(),
                         T.Normalize((0.5071, .4867, .4408), (.2675, .2565, .2761))])
    test_ds = torchvision.datasets.CIFAR100("./data", False, download=True, transform=test_tf)
    test_loader = DataLoader(test_ds, 64, shuffle=False, num_workers=4, pin_memory=True)

    front = ResNet50Front().to(device)
    back = ResNet50Back(100).to(device)
    bb_path = os.path.join(SNAP_DIR, "backbone_best.pth")
    state = torch.load(bb_path, map_location=device)
    front.load_state_dict({k: v for k, v in state.items()
                           if not k.startswith(('fc.', 'avgpool.'))}, strict=False)
    back.load_state_dict({k: v for k, v in state.items()
                          if k.startswith(('fc.', 'avgpool.'))}, strict=False)
    front.eval(); back.eval()
    print("✓ Backbone loaded")

    jpeg_conv = JPEGConv(jpeg_quality=50, code_rate_inv=3)

    # Fine-grained BER sweep
    ber_vals = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3]
    N_REPEAT = 3  # JPEG is slow

    results = []
    for ber in ber_vals:
        accs = []
        nr = N_REPEAT if ber > 0 else 1
        for rep in range(nr):
            correct, total = 0, 0
            for img, lab in tqdm(test_loader, desc=f"BER={ber:.3f} rep{rep+1}", leave=False):
                img, lab = img.to(device), lab.to(device)
                with torch.no_grad():
                    feat = front(img)
                Fp = jpeg_conv(feat, bit_error_rate=ber)
                out = back(Fp)
                correct += out.argmax(1).eq(lab).sum().item()
                total += lab.size(0)
            accs.append(100. * correct / total)
        m, s = np.mean(accs), np.std(accs)
        results.append({'ber': ber, 'mean': m, 'std': s})
        print(f"  BER={ber:.3f}: {m:.2f}% ±{s:.2f}")

    with open(os.path.join(EVAL_DIR, "jpeg_conv_sweep.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ JPEG+Conv sweep complete. Results: {EVAL_DIR}jpeg_conv_sweep.json")
