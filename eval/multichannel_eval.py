"""Multi-channel evaluation: BSC + AWGN + Rayleigh for V5C-NA on both datasets.

Two presentation modes:
  1. SNR sweep: Extended range down to -10 dB to show actual degradation
  2. Matched-BER comparison: Compute equivalent BER for each channel at each SNR,
     then compare accuracy at the SAME effective bit error rate

Theory:
  - BPSK AWGN:    BER = Q(sqrt(2·Eb/N0)) = erfc(sqrt(SNR_lin)) / 2
  - BPSK Rayleigh: BER = 0.5 · (1 - sqrt(SNR_lin / (1 + SNR_lin)))

For binary spikes, BPSK hard-decision means BSC at BER=0.13 ≈ SNR=-2dB (AWGN).
We need to go to -8...-10 dB to match BSC BER=0.30.
"""

import os, sys, json, math, time
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
from torch.utils.data import DataLoader
from scipy.special import erfc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from train_aid_v2 import ResNet50Front, ResNet50Back
from run_final_pipeline import AIDDataset5050, RESISC45Dataset, SpikeAdaptSC_v5c_NA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_STEPS = 8


# ===================================================================
# Theoretical BER for BPSK
# ===================================================================
def ber_awgn(snr_db):
    """Theoretical BER for uncoded BPSK over AWGN: Q(sqrt(2*SNR))."""
    snr_lin = 10 ** (snr_db / 10)
    return 0.5 * erfc(math.sqrt(max(snr_lin, 1e-12)))

def ber_rayleigh(snr_db):
    """Theoretical BER for uncoded BPSK over Rayleigh: 0.5*(1-sqrt(γ/(1+γ)))."""
    snr_lin = 10 ** (snr_db / 10)
    return 0.5 * (1.0 - math.sqrt(snr_lin / (1.0 + snr_lin)))

def snr_for_ber_awgn(target_ber):
    """Find SNR (dB) that gives target BER in AWGN. Inverse of Q-function."""
    from scipy.special import erfcinv
    snr_lin = (erfcinv(2 * target_ber)) ** 2
    return 10 * math.log10(max(snr_lin, 1e-12))

def snr_for_ber_rayleigh(target_ber):
    """Find SNR (dB) for target BER in Rayleigh. Inverse of BER formula."""
    # BER = 0.5*(1 - sqrt(γ/(1+γ))) → solve for γ
    x = 1 - 2 * target_ber  # x = sqrt(γ/(1+γ))
    if x <= 0: return -20.0
    gamma = x**2 / (1 - x**2)
    return 10 * math.log10(max(gamma, 1e-12))


# ===================================================================
# Channel implementations
# ===================================================================
class BSC_Channel_Eval(nn.Module):
    def forward(self, x, param):
        if param <= 0: return x
        noise = (torch.rand_like(x.float()) < param).float()
        return ((x + noise) % 2)

class AWGN_Channel_Eval(nn.Module):
    def forward(self, x, snr_db):
        if snr_db > 50: return x
        snr_lin = 10 ** (snr_db / 10)
        sigma = 1.0 / math.sqrt(2 * max(snr_lin, 1e-8))
        s = 2.0 * x.float() - 1.0
        y = s + sigma * torch.randn_like(s)
        return (y > 0).float()

class Rayleigh_Channel_Eval(nn.Module):
    def forward(self, x, snr_db):
        if snr_db > 50: return x
        snr_lin = 10 ** (snr_db / 10)
        sigma = 1.0 / math.sqrt(2 * max(snr_lin, 1e-8))
        s = 2.0 * x.float() - 1.0
        h_real = torch.randn_like(s)
        h_imag = torch.randn_like(s)
        h_mag = torch.sqrt(h_real**2 + h_imag**2) / math.sqrt(2)
        y = h_mag * s + sigma * torch.randn_like(s)
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
    model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
    model.eval(); back.eval()
    return front, model, back


def evaluate_channel(front, model, back, loader, channel, noise_param):
    """Manual forward pass decoupling scorer (BER=0) from channel noise."""
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            all_S2, m1, m2 = [], None, None
            for t in range(model.T):
                _, s2, m1, m2 = model.encoder(feat, m1, m2, t=t)
                all_S2.append(s2)
            importance = model.scorer(all_S2, 0.0).squeeze(1)
            mask, tx = model.block_mask(importance, training=False)
            recv = [channel(all_S2[t] * mask, noise_param) for t in range(model.T)]
            Fp = model.decoder(recv, mask)
            correct += back(Fp).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total


def run_multichannel(ds_name, n_classes, bb_dir, v5c_dir, DSClass, ds_args):
    print(f"\n{'='*60}")
    print(f"MULTI-CHANNEL EVAL — {ds_name.upper()}")
    print(f"{'='*60}")
    
    front, model, back = load_model(n_classes, bb_dir, v5c_dir)
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])
    test_ds = DSClass(transform=tf_test, split='test', seed=42, **ds_args)
    loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)
    
    # === Option 1: Extended SNR sweep ===
    bsc = BSC_Channel_Eval().to(device)
    bsc_params = [0.0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    bsc_results = {}
    print(f"\n  BSC (BER sweep):")
    for ber in bsc_params:
        acc = evaluate_channel(front, model, back, loader, bsc, ber)
        bsc_results[str(ber)] = acc
        print(f"    BER={ber:.2f}: {acc:.2f}%")
    
    awgn = AWGN_Channel_Eval().to(device)
    awgn_params = [20, 10, 5, 2, 0, -2, -4, -6, -8, -10]
    awgn_results = {}
    print(f"\n  AWGN (SNR sweep, extended to -10 dB):")
    for snr in awgn_params:
        eq_ber = ber_awgn(snr)
        acc = evaluate_channel(front, model, back, loader, awgn, float(snr))
        awgn_results[str(snr)] = {'acc': acc, 'eq_ber': eq_ber}
        print(f"    SNR={snr:+3d} dB (eq BER={eq_ber:.4f}): {acc:.2f}%")
    
    rayleigh = Rayleigh_Channel_Eval().to(device)
    ray_params = [20, 10, 5, 2, 0, -2, -4, -6, -8, -10]
    ray_results = {}
    print(f"\n  Rayleigh (SNR sweep, extended to -10 dB):")
    for snr in ray_params:
        eq_ber = ber_rayleigh(snr)
        acc = evaluate_channel(front, model, back, loader, rayleigh, float(snr))
        ray_results[str(snr)] = {'acc': acc, 'eq_ber': eq_ber}
        print(f"    SNR={snr:+3d} dB (eq BER={eq_ber:.4f}): {acc:.2f}%")
    
    # === Option 2: Matched-BER comparison ===
    print(f"\n  === MATCHED-BER COMPARISON ===")
    print(f"  {'BER':>8s}  {'BSC Acc':>8s}  {'AWGN SNR':>10s}  {'AWGN Acc':>8s}  {'Ray SNR':>10s}  {'Ray Acc':>8s}")
    print(f"  {'-'*60}")
    
    matched_results = {}
    target_bers = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    for tb in target_bers:
        bsc_acc = bsc_results.get(str(tb), None)
        if bsc_acc is None:
            bsc_acc = evaluate_channel(front, model, back, loader, bsc, tb)
        
        awgn_snr = snr_for_ber_awgn(tb)
        awgn_acc = evaluate_channel(front, model, back, loader, awgn, awgn_snr)
        
        ray_snr = snr_for_ber_rayleigh(tb)
        ray_acc = evaluate_channel(front, model, back, loader, rayleigh, ray_snr)
        
        matched_results[str(tb)] = {
            'bsc_acc': bsc_acc,
            'awgn_snr': awgn_snr, 'awgn_acc': awgn_acc,
            'ray_snr': ray_snr, 'ray_acc': ray_acc,
        }
        print(f"  {tb:8.2f}  {bsc_acc:7.2f}%  {awgn_snr:+8.2f}dB  {awgn_acc:7.2f}%  {ray_snr:+8.2f}dB  {ray_acc:7.2f}%")
    
    return {
        'bsc': {'params': bsc_params, 'results': bsc_results},
        'awgn': {'params': awgn_params, 'results': awgn_results},
        'rayleigh': {'params': ray_params, 'results': ray_results},
        'matched_ber': matched_results,
    }


if __name__ == '__main__':
    print(f"Device: {device}")
    
    # Print theoretical BER table
    print("\nTheoretical BER for BPSK:")
    print(f"  {'SNR (dB)':>10s}  {'AWGN BER':>10s}  {'Rayleigh BER':>12s}")
    for snr in [20, 10, 5, 2, 0, -2, -4, -6, -8, -10]:
        print(f"  {snr:+10d}  {ber_awgn(snr):10.6f}  {ber_rayleigh(snr):12.6f}")
    
    all_results = {}
    
    all_results['aid'] = run_multichannel(
        'aid', 30,
        'snapshots_aid_5050_seed42', 'snapshots_aid_v5cna_seed42',
        AIDDataset5050, {"root": "./data"}
    )
    torch.cuda.empty_cache()
    
    all_results['resisc45'] = run_multichannel(
        'resisc45', 45,
        'snapshots_resisc45_5050_seed42', 'snapshots_resisc45_v5cna_seed42',
        RESISC45Dataset, {"root": "./data", "train_ratio": 0.20}
    )
    
    # Convert numpy/scipy types for JSON
    def convert(obj):
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj
    
    with open("eval/multichannel_results_v2.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=convert)
    
    # Print summary
    print(f"\n{'='*70}")
    print("PAPER TABLE — Matched-BER Comparison (V5C-NA, ρ=0.75)")
    print(f"{'='*70}")
    for ds in ['aid', 'resisc45']:
        m = all_results[ds]['matched_ber']
        print(f"\n  {ds.upper()}:")
        print(f"  {'BER':>6s}  {'BSC':>7s}  {'AWGN':>7s}  {'Rayleigh':>8s}  {'AWGN SNR':>10s}  {'Ray SNR':>10s}")
        for tb in ['0.01', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3']:
            r = m[tb]
            print(f"  {float(tb):6.2f}  {r['bsc_acc']:6.2f}%  {r['awgn_acc']:6.2f}%  {r['ray_acc']:7.2f}%  "
                  f"{r['awgn_snr']:+8.2f}dB  {r['ray_snr']:+8.2f}dB")
    
    print(f"\n✅ Multi-channel v2 results saved to eval/multichannel_results_v2.json")
