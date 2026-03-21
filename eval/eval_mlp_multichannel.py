"""Multichannel eval for MLP-FC baseline (BSC/AWGN/Rayleigh).

Runs the trained MLP-FC baseline under AWGN and Rayleigh channels
at matched-BER SNR points. Uses the same channel implementations
as the CNN multichannel eval.
"""
import os, sys, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from scipy.special import erfc, erfcinv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
from train_aid_v2 import ResNet50Front, ResNet50Back, UniformQuantizeSTE
from run_final_pipeline import AIDDataset5050

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Channel functions (same as cnn_multichannel.py)
def snr_for_ber_awgn(target_ber):
    snr_lin = (erfcinv(2 * target_ber)) ** 2
    return 10 * math.log10(max(snr_lin, 1e-12))

def snr_for_ber_rayleigh(target_ber):
    x = 1 - 2 * target_ber
    if x <= 0: return -20.0
    gamma = x**2 / (1 - x**2)
    return 10 * math.log10(max(gamma, 1e-12))

def apply_bsc(bits, ber):
    if ber <= 0: return bits
    noise = (torch.rand_like(bits) < ber).float()
    return (bits + noise) % 2

def apply_awgn(bits, snr_db):
    if snr_db > 50: return bits
    snr_lin = 10 ** (snr_db / 10)
    sigma = 1.0 / math.sqrt(2 * max(snr_lin, 1e-8))
    s = 2.0 * bits - 1.0
    y = s + sigma * torch.randn_like(s)
    return (y > 0).float()

def apply_rayleigh(bits, snr_db):
    if snr_db > 50: return bits
    snr_lin = 10 ** (snr_db / 10)
    sigma = 1.0 / math.sqrt(2 * max(snr_lin, 1e-8))
    s = 2.0 * bits - 1.0
    h_real = torch.randn_like(s)
    h_imag = torch.randn_like(s)
    h_mag = torch.sqrt(h_real**2 + h_imag**2) / math.sqrt(2)
    y = h_mag * s + sigma * torch.randn_like(s)
    return (y / (h_mag + 1e-8) > 0).float()


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


class MLPUni(nn.Module):
    def __init__(self, C_in=1024, H=14, W=14, bottleneck=7056, n_bits=8):
        super().__init__()
        self.n_bits = n_bits
        self.C_in, self.H, self.W = C_in, H, W
        self.pool = nn.AdaptiveAvgPool2d(7)
        self.unpool = nn.Upsample(size=(H, W), mode='bilinear', align_corners=False)
        feat_dim = C_in * 7 * 7
        self.enc = nn.Sequential(
            nn.Linear(feat_dim, 2048), nn.BatchNorm1d(2048), nn.ReLU(),
            nn.Linear(2048, bottleneck), nn.Sigmoid())
        self.dec = nn.Sequential(
            nn.Linear(bottleneck, 2048), nn.BatchNorm1d(2048), nn.ReLU(),
            nn.Linear(2048, feat_dim))


def eval_mlp_channel(front, model, back, loader, channel_fn, channel_param, n_bits=8):
    correct = total = 0
    n_levels = 2 ** n_bits
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            B = feat.size(0)
            x = model.pool(feat).reshape(B, -1)
            z = model.enc(x)
            z_q = UniformQuantizeSTE.apply(z, n_bits)
            bits_val = (z_q * (n_levels - 1)).round().long()
            bit_planes = [((bits_val >> b) & 1).float() for b in range(n_bits)]
            bit_tensor = torch.stack(bit_planes, dim=-1)
            shape = bit_tensor.shape
            bits_flat = bit_tensor.reshape(shape[0], -1)
            bits_noisy = channel_fn(bits_flat, channel_param)
            bit_tensor = bits_noisy.reshape(shape)
            values = sum(bit_tensor[..., b] * (2**b) for b in range(n_bits))
            z_noisy = values / (n_levels - 1)
            out = model.dec(z_noisy)
            Fp = model.unpool(out.reshape(B, model.C_in, 7, 7))
            correct += back(Fp).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total


def run_dataset(ds_name, n_classes, backbone_path, snap_dir, test_ds):
    print(f"\n{'='*60}")
    print(f"  MLP-FC multichannel — {ds_name}")
    print(f"{'='*60}")
    front = ResNet50Front(grid_size=14).to(device)
    back = ResNet50Back(n_classes).to(device)
    bb = torch.load(backbone_path, map_location=device)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))},
                          strict=False)
    back.load_state_dict({k: v for k, v in bb.items()
                          if k.startswith(('layer4.', 'fc.', 'avgpool.'))}, strict=False)
    front.eval()

    model = MLPUni().to(device)
    best_f = sorted([f for f in os.listdir(snap_dir) if f.startswith("mlp_best_")])[-1]
    ck = torch.load(f"{snap_dir}/{best_f}", map_location=device)
    model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
    model.eval(); back.eval()
    print(f"  Checkpoint: {best_f}")

    loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)
    results = {}

    for ber in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        awgn_snr = 99 if ber == 0 else snr_for_ber_awgn(ber)
        ray_snr = 99 if ber == 0 else snr_for_ber_rayleigh(ber)
        t0 = time.time()
        acc_bsc = eval_mlp_channel(front, model, back, loader, apply_bsc, ber)
        acc_awgn = eval_mlp_channel(front, model, back, loader, apply_awgn, awgn_snr)
        acc_ray = eval_mlp_channel(front, model, back, loader, apply_rayleigh, ray_snr)
        results[str(ber)] = {'bsc': acc_bsc, 'awgn': acc_awgn, 'rayleigh': acc_ray,
                             'awgn_snr': awgn_snr, 'ray_snr': ray_snr}
        print(f"  BER={ber:.2f}: BSC={acc_bsc:5.2f}  AWGN={acc_awgn:5.2f}  Ray={acc_ray:5.2f}  [{time.time()-t0:.0f}s]")
    return results


def main():
    print(f"Device: {device}")
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                     T.Normalize((.485,.456,.406),(.229,.224,.225))])
    all_results = {}

    aid_test = AIDDataset5050(root="./data", transform=tf, split='test', seed=42)
    all_results['AID'] = run_dataset("AID (50/50)", 30,
        "./snapshots_aid_5050_seed42/backbone_best.pth",
        "./snapshots_mlp_aid_5050", aid_test)

    res_test = RESISC45_2080(root="./data", transform=tf, split='test', seed=42)
    all_results['RESISC45'] = run_dataset("RESISC45 (20/80)", 45,
        "./snapshots_resisc45_5050_seed42/backbone_best.pth",
        "./snapshots_mlp_resisc45_2080", res_test)

    with open("eval/mlp_multichannel.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ MLP-FC multichannel eval complete.")


if __name__ == '__main__':
    main()
