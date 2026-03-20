"""Multi-channel evaluation: BSC + AWGN + Rayleigh for V5C-NA on both datasets.

Evaluates V5C-NA (trained on BSC) across all 3 channel models at matched noise levels:
  BSC:      BER = {0.00, 0.05, 0.10, 0.15, 0.25, 0.30, 0.35}
  AWGN:     SNR = {20, 10, 5, 2, 0, -2} dB
  Rayleigh: SNR = {20, 10, 5, 2, 0, -2} dB

Outputs JSON + prints formatted table for paper.
"""

import os, sys, json, math, time
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from train_aid_v2 import ResNet50Front, ResNet50Back
from run_final_pipeline import AIDDataset5050, RESISC45Dataset, SpikeAdaptSC_v5c_NA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_STEPS = 8


# ===================================================================
# Channel implementations for spike signals
# ===================================================================
class BSC_Channel_Eval(nn.Module):
    """Binary Symmetric Channel: flip each bit with probability BER."""
    def forward(self, x, param):
        if param <= 0: return x
        noise = (torch.rand_like(x.float()) < param).float()
        return ((x + noise) % 2)


class AWGN_Channel_Eval(nn.Module):
    """AWGN channel with BPSK modulation for binary spike signals.
    
    x ∈ {0,1}^{...} →  BPSK: s = 2x-1 ∈ {-1,+1}  →  y = s + n  →  x̂ = 1[y>0]
    param: SNR in dB
    """
    def forward(self, x, snr_db):
        if snr_db > 50: return x
        snr_lin = 10 ** (snr_db / 10)
        sigma = 1.0 / math.sqrt(2 * max(snr_lin, 1e-8))
        # BPSK modulate
        s = 2.0 * x.float() - 1.0
        # Add noise 
        y = s + sigma * torch.randn_like(s)
        # Hard-decision demodulate
        return (y > 0).float()


class Rayleigh_Channel_Eval(nn.Module):
    """Rayleigh fading channel with BPSK for binary spikes.
    
    x → BPSK s → y = h·s + n → x̂ = 1[y/|h| > 0] (coherent detection)
    param: SNR in dB
    """
    def forward(self, x, snr_db):
        if snr_db > 50: return x
        snr_lin = 10 ** (snr_db / 10)
        sigma = 1.0 / math.sqrt(2 * max(snr_lin, 1e-8))
        # BPSK modulate
        s = 2.0 * x.float() - 1.0
        # Rayleigh fading: h ~ Rayleigh(1)
        h_real = torch.randn_like(s)
        h_imag = torch.randn_like(s)
        h_mag = torch.sqrt(h_real**2 + h_imag**2) / math.sqrt(2)
        # Faded + noisy signal
        y = h_mag * s + sigma * torch.randn_like(s)
        # Coherent detection (equalize by |h|)
        x_hat = (y / (h_mag + 1e-8) > 0).float()
        return x_hat


def load_model(n_classes, bb_dir, v5c_dir):
    front = ResNet50Front(grid_size=14).to(device)
    bb = torch.load(f"./{bb_dir}/backbone_best.pth", map_location=device)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))}, strict=False)
    front.eval()
    
    back = ResNet50Back(n_classes).to(device)
    model = SpikeAdaptSC_v5c_NA(C_in=1024, C1=256, C2=36, T=T_STEPS,
                                 target_rate=0.75, grid_size=14).to(device)
    
    ck_files = sorted([f for f in os.listdir(f"./{v5c_dir}") if f.startswith('v5cna_best')])
    ck = torch.load(f"./{v5c_dir}/{ck_files[-1]}", map_location=device)
    model.load_state_dict(ck['model'])
    back.load_state_dict(ck['back'])
    model.eval(); back.eval()
    
    return front, model, back


def evaluate_channel(front, model, back, loader, channel, noise_param):
    """Evaluate model with a specific channel at a noise level.
    
    Uses a manual forward pass to decouple:
    - Scorer: always uses noise_param=0 (clean mask, since we're evaluating cross-channel)
    - Channel: uses the specified channel + noise_param
    """
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            
            # Encode
            all_S2, m1, m2 = [], None, None
            for t in range(model.T):
                _, s2, m1, m2 = model.encoder(feat, m1, m2, t=t)
                all_S2.append(s2)
            
            # Score + mask (use clean scoring)
            importance = model.scorer(all_S2, 0.0).squeeze(1)
            mask, tx = model.block_mask(importance, training=False)
            
            # Apply specified channel
            recv = [channel(all_S2[t] * mask, noise_param) for t in range(model.T)]
            
            # Decode
            Fp = model.decoder(recv, mask)
            correct += back(Fp).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    
    return 100. * correct / total


def run_multichannel(ds_name, n_classes, bb_dir, v5c_dir, DSClass, ds_args):
    """Run multi-channel evaluation for one dataset."""
    print(f"\n{'='*60}")
    print(f"MULTI-CHANNEL EVAL — {ds_name.upper()}")
    print(f"{'='*60}")
    
    front, model, back = load_model(n_classes, bb_dir, v5c_dir)
    
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])
    test_ds = DSClass(transform=tf_test, split='test', seed=42, **ds_args)
    loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)
    
    # BSC sweep
    bsc = BSC_Channel_Eval().to(device)
    bsc_params = [0.0, 0.05, 0.10, 0.15, 0.25, 0.30, 0.35]
    bsc_results = {}
    print(f"\n  BSC (BER sweep):")
    for ber in bsc_params:
        acc = evaluate_channel(front, model, back, loader, bsc, ber)
        bsc_results[str(ber)] = acc
        print(f"    BER={ber:.2f}: {acc:.2f}%")
    
    # AWGN sweep
    awgn = AWGN_Channel_Eval().to(device)
    awgn_params = [20, 10, 5, 2, 0, -2]
    awgn_results = {}
    print(f"\n  AWGN (SNR sweep):")
    for snr in awgn_params:
        acc = evaluate_channel(front, model, back, loader, awgn, float(snr))
        awgn_results[str(snr)] = acc
        print(f"    SNR={snr:+d} dB: {acc:.2f}%")
    
    # Rayleigh sweep
    rayleigh = Rayleigh_Channel_Eval().to(device)
    ray_params = [20, 10, 5, 2, 0, -2]
    ray_results = {}
    print(f"\n  Rayleigh (SNR sweep):")
    for snr in ray_params:
        acc = evaluate_channel(front, model, back, loader, rayleigh, float(snr))
        ray_results[str(snr)] = acc
        print(f"    SNR={snr:+d} dB: {acc:.2f}%")
    
    return {
        'bsc': {'params': bsc_params, 'results': bsc_results},
        'awgn': {'params': awgn_params, 'results': awgn_results},
        'rayleigh': {'params': ray_params, 'results': ray_results},
    }


if __name__ == '__main__':
    print(f"Device: {device}")
    all_results = {}
    
    # AID
    all_results['aid'] = run_multichannel(
        'aid', 30,
        'snapshots_aid_5050_seed42', 'snapshots_aid_v5cna_seed42',
        AIDDataset5050, {"root": "./data"}
    )
    
    torch.cuda.empty_cache()
    
    # RESISC45
    all_results['resisc45'] = run_multichannel(
        'resisc45', 45,
        'snapshots_resisc45_5050_seed42', 'snapshots_resisc45_v5cna_seed42',
        RESISC45Dataset, {"root": "./data", "train_ratio": 0.20}
    )
    
    # Save
    with open("eval/multichannel_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print paper-ready table
    print(f"\n{'='*60}")
    print("PAPER TABLE — Multi-Channel Results (V5C-NA, ρ=0.75)")
    print(f"{'='*60}")
    for ds in ['aid', 'resisc45']:
        r = all_results[ds]
        print(f"\n  {ds.upper()}:")
        print(f"    BSC:      Clean={r['bsc']['results']['0.0']:.2f}%  "
              f"BER=0.10={r['bsc']['results']['0.1']:.2f}%  "
              f"BER=0.25={r['bsc']['results']['0.25']:.2f}%  "
              f"BER=0.35={r['bsc']['results']['0.35']:.2f}%")
        print(f"    AWGN:     20dB={r['awgn']['results']['20']:.2f}%  "
              f"10dB={r['awgn']['results']['10']:.2f}%  "
              f"5dB={r['awgn']['results']['5']:.2f}%  "
              f"-2dB={r['awgn']['results']['-2']:.2f}%")
        print(f"    Rayleigh: 20dB={r['rayleigh']['results']['20']:.2f}%  "
              f"10dB={r['rayleigh']['results']['10']:.2f}%  "
              f"5dB={r['rayleigh']['results']['5']:.2f}%  "
              f"-2dB={r['rayleigh']['results']['-2']:.2f}%")
    
    print(f"\n✅ Multi-channel results saved to eval/multichannel_results.json")
