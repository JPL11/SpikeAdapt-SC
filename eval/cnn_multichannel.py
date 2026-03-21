"""Evaluate CNN-Uni and CNN-NonUni under BSC, AWGN, Rayleigh channels.

Uses matched-BER points: for each target BER, compute the SNR that gives
that BER in AWGN/Rayleigh, then evaluate CNN through that channel.

Produces a unified multichannel table for the paper.
"""
import os, sys, json, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from scipy.special import erfc, erfcinv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
from train_aid_v2 import ResNet50Front, ResNet50Back, CNNUni
from run_final_pipeline import AIDDataset5050

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── BER/SNR mapping ───
def ber_awgn(snr_db):
    snr_lin = 10 ** (snr_db / 10)
    return 0.5 * erfc(math.sqrt(max(snr_lin, 1e-12)))

def ber_rayleigh(snr_db):
    snr_lin = 10 ** (snr_db / 10)
    return 0.5 * (1.0 - math.sqrt(snr_lin / (1.0 + snr_lin)))

def snr_for_ber_awgn(target_ber):
    snr_lin = (erfcinv(2 * target_ber)) ** 2
    return 10 * math.log10(max(snr_lin, 1e-12))

def snr_for_ber_rayleigh(target_ber):
    x = 1 - 2 * target_ber
    if x <= 0: return -20.0
    gamma = x**2 / (1 - x**2)
    return 10 * math.log10(max(gamma, 1e-12))

# ─── Channel implementations ───
def apply_bsc(bits_flat, ber):
    if ber <= 0: return bits_flat
    noise = (torch.rand_like(bits_flat) < ber).float()
    return (bits_flat + noise) % 2

def apply_awgn(bits_flat, snr_db):
    if snr_db > 50: return bits_flat
    snr_lin = 10 ** (snr_db / 10)
    sigma = 1.0 / math.sqrt(2 * max(snr_lin, 1e-8))
    s = 2.0 * bits_flat - 1.0  # BPSK: 0→-1, 1→+1
    y = s + sigma * torch.randn_like(s)
    return (y > 0).float()  # Hard decision

def apply_rayleigh(bits_flat, snr_db):
    if snr_db > 50: return bits_flat
    snr_lin = 10 ** (snr_db / 10)
    sigma = 1.0 / math.sqrt(2 * max(snr_lin, 1e-8))
    s = 2.0 * bits_flat - 1.0
    h_real = torch.randn_like(s)
    h_imag = torch.randn_like(s)
    h_mag = torch.sqrt(h_real**2 + h_imag**2) / math.sqrt(2)
    y = h_mag * s + sigma * torch.randn_like(s)
    return (y / (h_mag + 1e-8) > 0).float()

# ─── RESISC45 dataset ───
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


def eval_cnn_uni_channel(front, model, back, loader, channel_fn, channel_param):
    """Evaluate CNN-Uni with a specified channel applied to bit planes."""
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            # Encode
            z = torch.sigmoid(model.ebn2(model.enc2(F.relu(model.ebn1(model.enc1(feat))))))
            # Uniform quantize
            from train_aid_v2 import UniformQuantizeSTE
            z_q = UniformQuantizeSTE.apply(z, model.n_bits)
            n_levels = 2 ** model.n_bits
            bits = (z_q * (n_levels - 1)).round().long()
            B, C, H, W = bits.shape
            bit_planes = []
            for b in range(model.n_bits):
                bit_planes.append(((bits >> b) & 1).float())
            bit_tensor = torch.stack(bit_planes, dim=-1)
            # Apply channel to all bits
            shape = bit_tensor.shape
            bits_flat = bit_tensor.reshape(shape[0], -1)
            bits_noisy = channel_fn(bits_flat, channel_param)
            bit_tensor = bits_noisy.reshape(shape)
            # Reconstruct
            values = torch.zeros(B, C, H, W, device=device)
            for b in range(model.n_bits):
                values += bit_tensor[..., b] * (2 ** b)
            z_noisy = values / (n_levels - 1)
            # Decode
            Fp = model.dbn2(model.dec2(F.relu(model.dbn1(model.dec1(z_noisy)))))
            correct += back(Fp).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total


def eval_cnn_nonuni_channel(front, model, back, loader, channel_fn, channel_param, levels_t, n_bits=8):
    """Evaluate CNN-NonUni with a specified channel."""
    n_levels = 2**n_bits
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            z = torch.sigmoid(model.ebn2(model.enc2(F.relu(model.ebn1(model.enc1(feat))))))
            indices = (z.unsqueeze(-1) - levels_t).abs().argmin(dim=-1)
            bits = []
            for b in range(n_bits):
                bits.append(((indices >> b) & 1).float())
            bit_tensor = torch.stack(bits, dim=-1)
            shape = bit_tensor.shape
            bits_flat = bit_tensor.reshape(shape[0], -1)
            bits_noisy = channel_fn(bits_flat, channel_param)
            bit_tensor = bits_noisy.reshape(shape)
            B, C, H, W = z.shape
            idx_r = sum(bit_tensor[..., b].long() * (2**b) for b in range(n_bits))
            z_recv = levels_t[idx_r.clamp(0, n_levels-1)]
            Fp = model.dbn2(model.dec2(F.relu(model.dbn1(model.dec1(z_recv)))))
            correct += back(Fp).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total


def compute_nonuni_levels(front, model, loader, n_bits=8):
    """Compute Lloyd-Max quantization levels via k-means on activations."""
    from sklearn.cluster import MiniBatchKMeans
    n_levels = 2**n_bits
    all_acts = []
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            feat = front(imgs.to(device))
            z = torch.sigmoid(model.ebn2(model.enc2(F.relu(model.ebn1(model.enc1(feat))))))
            all_acts.append(z.cpu().numpy().flatten())
            if i >= 10: break
    flat = np.concatenate(all_acts)
    sub = flat[np.random.RandomState(42).choice(len(flat), min(200000, len(flat)), replace=False)]
    kmeans = MiniBatchKMeans(n_clusters=n_levels, n_init=3, max_iter=50,
                             random_state=42, batch_size=10000)
    kmeans.fit(sub.reshape(-1, 1))
    return torch.tensor(np.sort(kmeans.cluster_centers_.flatten()),
                        dtype=torch.float32, device=device)


def run_dataset(ds_name, n_classes, backbone_path, snap_dir, test_ds):
    print(f"\n{'='*70}")
    print(f"  {ds_name}")
    print(f"{'='*70}")

    front = ResNet50Front(grid_size=14).to(device)
    bb = torch.load(backbone_path, map_location=device)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))},
                          strict=False)
    front.eval()

    model = CNNUni(C_in=1024, C1=256, C2=36, n_bits=8).to(device)
    back = ResNet50Back(n_classes).to(device)
    
    # Load best checkpoint
    s3 = sorted([f for f in os.listdir(snap_dir) if f.startswith("cnnuni_s3_")])
    best = sorted([f for f in os.listdir(snap_dir) if f.startswith("cnnuni_best_")])
    ckf = s3[-1] if s3 else best[-1]
    ck = torch.load(f"{snap_dir}/{ckf}", map_location=device)
    model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
    model.eval(); back.eval()
    print(f"  Checkpoint: {ckf}")

    loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)
    print(f"  Test: {len(test_ds)} images")

    # Compute NonUni levels
    levels_t = compute_nonuni_levels(front, model, loader)

    ber_list = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    results = {}

    for method_name, eval_fn, extra_args in [
        ('CNN-Uni', eval_cnn_uni_channel, {}),
        ('CNN-NonUni', eval_cnn_nonuni_channel, {'levels_t': levels_t}),
    ]:
        results[method_name] = {'bsc': {}, 'awgn': {}, 'rayleigh': {}, 'matched_ber': {}}
        print(f"\n  --- {method_name} ---")

        for ber in ber_list:
            if ber == 0:
                awgn_snr, ray_snr = 99, 99
            else:
                awgn_snr = snr_for_ber_awgn(ber)
                ray_snr = snr_for_ber_rayleigh(ber)

            # BSC
            t0 = time.time()
            acc_bsc = eval_fn(front, model, back, loader, apply_bsc, ber, **extra_args)
            # AWGN at matched SNR
            acc_awgn = eval_fn(front, model, back, loader, apply_awgn, awgn_snr, **extra_args)
            # Rayleigh at matched SNR
            acc_ray = eval_fn(front, model, back, loader, apply_rayleigh, ray_snr, **extra_args)

            results[method_name]['bsc'][str(ber)] = acc_bsc
            results[method_name]['awgn'][str(ber)] = acc_awgn
            results[method_name]['rayleigh'][str(ber)] = acc_ray
            results[method_name]['matched_ber'][str(ber)] = {
                'bsc': acc_bsc, 'awgn': acc_awgn, 'rayleigh': acc_ray,
                'awgn_snr': awgn_snr, 'ray_snr': ray_snr
            }
            elapsed = time.time() - t0
            print(f"    BER={ber:.2f}: BSC={acc_bsc:5.2f}  AWGN={acc_awgn:5.2f} ({awgn_snr:+.1f}dB)  Ray={acc_ray:5.2f} ({ray_snr:+.1f}dB)  [{elapsed:.0f}s]")

    return results


def main():
    print(f"Device: {device}")

    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])

    all_results = {}

    # AID
    aid_test = AIDDataset5050(root="./data", transform=tf_test, split='test', seed=42)
    all_results['AID'] = run_dataset(
        "AID (50/50)", 30,
        "./snapshots_aid_5050_seed42/backbone_best.pth",
        "./snapshots_cnnuni_aid_5050",
        aid_test
    )

    # RESISC45
    res_test = RESISC45_2080(root="./data", transform=tf_test, split='test', seed=42)
    all_results['RESISC45'] = run_dataset(
        "RESISC45 (20/80)", 45,
        "./snapshots_resisc45_5050_seed42/backbone_best.pth",
        "./snapshots_cnnuni_resisc45_2080",
        res_test
    )

    # Save
    with open("eval/cnn_multichannel.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary table
    print(f"\n{'='*90}")
    print("MATCHED-BER TABLE (CNN baselines)")
    print(f"{'='*90}")
    for ds, res in all_results.items():
        print(f"\n{ds}:")
        for method in ['CNN-Uni', 'CNN-NonUni']:
            print(f"  {method}:")
            print(f"    {'BER':>6s}  {'BSC':>6s}  {'AWGN':>6s}  {'Rayleigh':>8s}  {'Max Δ':>6s}")
            for ber in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
                mb = res[method]['matched_ber'][str(ber)]
                vals = [mb['bsc'], mb['awgn'], mb['rayleigh']]
                delta = max(vals) - min(vals)
                print(f"    {ber:6.2f}  {mb['bsc']:6.2f}  {mb['awgn']:6.2f}  {mb['rayleigh']:8.2f}  {delta:6.2f}")

    print(f"\n✅ CNN multichannel eval complete. Results: eval/cnn_multichannel.json")


if __name__ == '__main__':
    main()
