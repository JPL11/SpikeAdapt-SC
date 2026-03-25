#!/usr/bin/env python3
"""Task-Oriented JSCC Baseline for Classification (inspired by DHF-JSCC).

Implements a continuous-valued JSCC (Joint Source-Channel Coding) baseline
that compresses ResNet-50 features, transmits through AWGN, and classifies.

Architecture:
  ResNet-50 backbone (frozen, shared with SpikeAdapt-SC)
  → JSCC Encoder: Conv1024→256→C_tx + LayerNorm + ReLU + Power Normalization
  → AWGN Channel (complex I/Q, parametric SNR in dB)
  → JSCC Decoder: Conv C_tx→256→1024 + LayerNorm + ReLU
  → ResNet-50 Back (layer4 + GAP + FC)

Key DHF-JSCC principles retained:
  1. Continuous-valued channel input (not binary)
  2. Power normalization before channel (unit average power)
  3. End-to-end training through differentiable AWGN channel

Training: 10 seeds × 2 datasets × 60 epochs, AdamW, cosine decay.
Eval: Clean + AWGN SNR sweep + BSC BER sweep (via hard quantization).
Output: eval/seed_results/jscc_10seed.json
"""

import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
import sys, os, json, random, math, numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms as T

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from run_final_pipeline import AIDDataset5050, RESISC45Dataset
from train_aid_v2 import ResNet50Front, ResNet50Back

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]


# ============================================================================
# JSCC Components
# ============================================================================

class JSCC_Encoder(nn.Module):
    """JSCC encoder: compress backbone features to channel symbols.
    
    Two conv layers reduce 1024→256→C_tx channels with LayerNorm and ReLU.
    Power normalization ensures unit average transmit power (P=1).
    
    Inspired by DHF-JSCC's encoder structure (strided convolutions with
    channel attention), but adapted for classification features (no stride,
    since we want to preserve spatial resolution for the back classifier).
    """
    def __init__(self, C_in=1024, C_mid=256, C_tx=36):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(C_in, C_mid, 3, 1, 1),
            nn.BatchNorm2d(C_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(C_mid, C_tx, 3, 1, 1),
            nn.BatchNorm2d(C_tx),
        )
    
    def forward(self, feat):
        z = self.encoder(feat)  # B×C_tx×H×W
        # Power normalization: unit average power per symbol
        power = torch.mean(z ** 2, dim=(1, 2, 3), keepdim=True)
        z_norm = z / (torch.sqrt(power) + 1e-8)
        return z_norm, power


class AWGN_Channel_Continuous(nn.Module):
    """AWGN channel for continuous-valued (analog) transmission.
    
    Unlike the BSC used for SNN (binary), this channel adds Gaussian
    noise directly to continuous-valued symbols. SNR is in dB.
    
    This is the native channel for JSCC systems like DHF-JSCC.
    """
    def forward(self, x, snr_db):
        if snr_db >= 100:
            return x  # noiseless
        snr_linear = 10 ** (snr_db / 10.0)
        # Signal power is normalized to 1 by encoder
        noise_std = 1.0 / math.sqrt(2 * snr_linear)
        noise = torch.randn_like(x) * noise_std
        return x + noise


class BSC_Channel_Quantized(nn.Module):
    """BSC channel applied to quantized (sign-binarized) continuous symbols.
    
    Steps: sign-quantize → BSC bit flip → dequantize.
    This tests how JSCC handles a binary channel it wasn't designed for.
    Uses straight-through estimator during training.
    """
    def forward(self, x, ber):
        if ber <= 0:
            return x
        # Quantize to {-1, +1}
        x_bin = torch.sign(x)
        x_bin = torch.where(x_bin == 0, torch.ones_like(x_bin), x_bin)
        # BSC: flip with probability ber
        flip = (torch.rand_like(x) < ber).float()
        x_noisy = x_bin * (1 - 2 * flip)  # flip sign
        if self.training:
            return x + (x_noisy - x).detach()  # STE
        return x_noisy


class JSCC_Decoder(nn.Module):
    """JSCC decoder: expand received symbols back to feature space.
    
    Mirrors encoder: C_tx→256→1024 with LayerNorm and ReLU.
    The decoded features are passed to the ResNet-50 back classifier.
    """
    def __init__(self, C_out=1024, C_mid=256, C_tx=36):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(C_tx, C_mid, 3, 1, 1),
            nn.BatchNorm2d(C_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(C_mid, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
        )
    
    def forward(self, z_recv, power):
        # De-normalize power
        z_denorm = z_recv * (torch.sqrt(power) + 1e-8)
        return self.decoder(z_denorm)


class JSCC_Model(nn.Module):
    """Complete Task-Oriented JSCC model for classification.
    
    Combines encoder, AWGN channel, and decoder into a single module.
    Supports both AWGN (continuous, native) and BSC (quantized) evaluation.
    """
    def __init__(self, C_in=1024, C_mid=256, C_tx=36):
        super().__init__()
        self.encoder = JSCC_Encoder(C_in, C_mid, C_tx)
        self.awgn = AWGN_Channel_Continuous()
        self.bsc = BSC_Channel_Quantized()
        self.decoder = JSCC_Decoder(C_in, C_mid, C_tx)
    
    def forward(self, feat, snr_db=20.0, channel='awgn', ber=0.0):
        z_norm, power = self.encoder(feat)
        
        if channel == 'awgn':
            z_recv = self.awgn(z_norm, snr_db)
        elif channel == 'bsc':
            z_recv = self.bsc(z_norm, ber)
        else:
            z_recv = z_norm  # noiseless
        
        Fp = self.decoder(z_recv, power)
        return Fp, {'channel': channel, 'snr_db': snr_db, 'ber': ber}


# ============================================================================
# Training & Evaluation
# ============================================================================

def train_and_eval_seed(dataset_name, n_classes, seed):
    """Train JSCC model for one seed, evaluate at clean and noisy conditions."""
    print(f"\n{'='*60}")
    print(f"  JSCC Baseline {dataset_name.upper()} seed={seed}")
    print(f"{'='*60}", flush=True)

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tf_train = T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
                           T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))])
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])

    if dataset_name == 'aid':
        train_ds = AIDDataset5050('./data', tf_train, 'train', seed=seed)
        test_ds = AIDDataset5050('./data', tf_test, 'test', seed=seed)
    else:
        train_ds = RESISC45Dataset('./data', tf_train, 'train', train_ratio=0.20, seed=seed)
        test_ds = RESISC45Dataset('./data', tf_test, 'test', train_ratio=0.20, seed=seed)

    train_loader = DataLoader(train_ds, 32, True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)

    # Load backbone (same one used for SNN and CNN-1bit)
    bb_path = f'./snapshots_{dataset_name}_5050_seed{seed}/backbone_best.pth'
    if not os.path.exists(bb_path):
        print(f"  ERROR: backbone not found: {bb_path}")
        return None

    front = ResNet50Front(grid_size=14).to(device)
    bb = torch.load(bb_path, map_location=device, weights_only=False)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))}, strict=False)
    front.eval()
    for p in front.parameters():
        p.requires_grad = False

    back = ResNet50Back(n_classes).to(device)
    back_state = {k: v for k, v in bb.items()
                  if k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))}
    back.load_state_dict(back_state, strict=False)

    # JSCC model: same bottleneck C_tx=36 as SNN for fair payload comparison
    model = JSCC_Model(C_in=1024, C_mid=256, C_tx=36).to(device)

    snap_dir = f'./snapshots_{dataset_name}_jscc_seed{seed}/'
    os.makedirs(snap_dir, exist_ok=True)

    # Check if already trained
    existing = sorted([f for f in os.listdir(snap_dir) if f.startswith('jscc_best_')],
                      key=lambda x: float(x.split('_')[-1].replace('.pth', ''))) if os.path.exists(snap_dir) else []

    if not existing:
        # Train 60 epochs with mixed SNR
        optimizer = optim.AdamW(list(model.parameters()) + list(back.parameters()), lr=1e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0
        print(f"  Training (60 epochs)...", flush=True)
        for epoch in range(1, 61):
            model.train(); back.train()
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                feat = front(imgs)
                # Random SNR during training: mix of clean and noisy
                snr = random.choice([100, 20, 10, 5, 2, 0, -2, -4])
                Fp, _ = model(feat, snr_db=snr, channel='awgn')
                loss = criterion(back(Fp), labels)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            scheduler.step()

            if epoch % 10 == 0 or epoch == 60:
                model.eval(); back.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for imgs, labels in test_loader:
                        imgs, labels = imgs.to(device), labels.to(device)
                        Fp, _ = model(front(imgs), snr_db=100, channel='awgn')  # clean
                        correct += back(Fp).argmax(1).eq(labels).sum().item()
                        total += labels.size(0)
                acc = 100. * correct / total
                print(f"  E{epoch:02d}: {acc:.2f}%", flush=True)

                if acc > best_acc:
                    best_acc = acc
                    torch.save({'model': model.state_dict(), 'back': back.state_dict()},
                               os.path.join(snap_dir, f'jscc_best_{acc:.2f}.pth'))

        existing = sorted([f for f in os.listdir(snap_dir) if f.startswith('jscc_best_')],
                          key=lambda x: float(x.split('_')[-1].replace('.pth', '')))
    else:
        print(f"  Found existing checkpoint: {existing[-1]}", flush=True)

    # Load best checkpoint
    best_ck = existing[-1]
    ck = torch.load(os.path.join(snap_dir, best_ck), map_location=device, weights_only=False)
    model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
    model.eval(); back.eval()

    # Evaluate: clean, AWGN sweep, and BSC sweep
    results = {}

    def eval_acc(model, front, back, loader, **kwargs):
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                Fp, _ = model(front(imgs), **kwargs)
                correct += back(Fp).argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        return round(100. * correct / total, 2)

    # Clean
    results['clean'] = eval_acc(model, front, back, test_loader, snr_db=100, channel='awgn')
    print(f"  Clean: {results['clean']}%", flush=True)

    # AWGN sweep
    results['awgn'] = {}
    for snr in [10, 5, 0, -2, -4, -6, -8]:
        acc = eval_acc(model, front, back, test_loader, snr_db=snr, channel='awgn')
        results['awgn'][str(snr)] = acc
    print(f"  AWGN SNR=0dB: {results['awgn']['0']}%, SNR=-8dB: {results['awgn']['-8']}%", flush=True)

    # BSC sweep (quantized): how does continuous JSCC handle binary channels?
    results['bsc'] = {}
    for ber in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        acc = eval_acc(model, front, back, test_loader, channel='bsc', ber=ber, snr_db=100)
        results['bsc'][str(ber)] = acc
    print(f"  BSC BER=0.30: {results['bsc']['0.3']}%", flush=True)

    del model, back, front; torch.cuda.empty_cache()
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--dataset', type=str, default=None, choices=['aid', 'resisc45'])
    args = parser.parse_args()

    seeds = [args.seed] if args.seed else SEEDS
    datasets = [(args.dataset, 30 if args.dataset == 'aid' else 45)] if args.dataset else [('aid', 30), ('resisc45', 45)]

    os.makedirs('eval/seed_results', exist_ok=True)
    all_results = {}

    for ds_name, n_classes in datasets:
        all_results[ds_name] = {}
        for seed in seeds:
            result = train_and_eval_seed(ds_name, n_classes, seed)
            if result:
                all_results[ds_name][str(seed)] = result

    # Compute statistics
    print(f"\n{'='*60}")
    print(f"  JSCC 10-SEED SUMMARY")
    print(f"{'='*60}", flush=True)

    summary = {}
    for ds_name in all_results:
        summary[ds_name] = {}
        # Clean
        accs = [all_results[ds_name][str(s)]['clean'] for s in seeds if str(s) in all_results[ds_name]]
        if len(accs) >= 2:
            summary[ds_name]['clean'] = {'mean': round(float(np.mean(accs)), 2), 'std': round(float(np.std(accs, ddof=1)), 2), 'per_seed': accs}
            print(f"  {ds_name} Clean: {np.mean(accs):.2f} ± {np.std(accs, ddof=1):.2f}", flush=True)

        # AWGN at key SNRs
        for snr in ['0', '-4', '-8']:
            accs = [all_results[ds_name][str(s)]['awgn'][snr] for s in seeds if str(s) in all_results[ds_name]]
            if len(accs) >= 2:
                summary[ds_name][f'awgn_snr{snr}'] = {'mean': round(float(np.mean(accs)), 2), 'std': round(float(np.std(accs, ddof=1)), 2)}
                print(f"  {ds_name} AWGN SNR={snr}dB: {np.mean(accs):.2f} ± {np.std(accs, ddof=1):.2f}", flush=True)

        # BSC at BER=0.30
        accs = [all_results[ds_name][str(s)]['bsc']['0.3'] for s in seeds if str(s) in all_results[ds_name]]
        if len(accs) >= 2:
            summary[ds_name]['bsc_ber030'] = {'mean': round(float(np.mean(accs)), 2), 'std': round(float(np.std(accs, ddof=1)), 2), 'per_seed': accs}
            print(f"  {ds_name} BSC BER=0.30: {np.mean(accs):.2f} ± {np.std(accs, ddof=1):.2f}", flush=True)

    # Save
    with open('eval/seed_results/jscc_10seed.json', 'w') as f:
        json.dump({'per_seed': all_results, 'summary': summary}, f, indent=2)

    print(f"\n✅ Results saved to eval/seed_results/jscc_10seed.json")


if __name__ == '__main__':
    main()
