"""Ablation experiments for SpikeAdapt-SC paper.

1. ρ sweep: accuracy at different target mask rates [0.25, 0.50, 0.625, 0.75, 0.875, 1.0]
2. Mask comparison: Learned mask vs Random mask vs Uniform mask
3. BER sweep at each ρ

Runs on V4-A (best overall) and V5C (best clean + energy) checkpoints.

Usage:
  python eval/ablation_aid.py --ablation rho    # ρ sweep
  python eval/ablation_aid.py --ablation mask   # mask comparison
  python eval/ablation_aid.py --ablation all    # both
"""

import os, sys, json, argparse
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
from train_aid_v2 import (
    AIDDataset, ResNet50Front, ResNet50Back, BSC_Channel,
    ChannelConditionedScorer, LearnedBlockMask, sample_noise
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_STEPS = 8

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ablation', type=str, default='all', help='rho, mask, or all')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def load_model_and_back(version, seed=42):
    """Load a saved model + back classifier."""
    from train_aid_v4 import SpikeAdaptSC_v4
    from train_aid_v5 import SpikeAdaptSC_v5

    front = ResNet50Front(grid_size=14).to(device)
    back = ResNet50Back(30).to(device)

    # Load backbone
    bb = torch.load("./snapshots_aid/backbone_best.pth", map_location=device)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))}, strict=False)
    front.eval()

    if version == 'v4a':
        snap_dir = f"./snapshots_v4_V4A_seed{seed}/"
        model = SpikeAdaptSC_v4(C_in=1024, C1=256, C2=36, T=T_STEPS,
                                target_rate=0.75, grid_size=14,
                                use_mpbn=False, use_channel_gate=False).to(device)
        ckf = sorted([f for f in os.listdir(snap_dir) if f.startswith("v4_V4A_")])
        if ckf:
            ck = torch.load(os.path.join(snap_dir, ckf[-1]), map_location=device)
            model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
    elif version == 'v5c':
        snap_dir = f"./snapshots_v5_V5C_seed{seed}/"
        model = SpikeAdaptSC_v5(C_in=1024, C1=256, C2=36, T=T_STEPS,
                                target_rate=0.75, grid_size=14,
                                use_mpbn=True, use_channel_gate=False,
                                use_spike_attn=False).to(device)
        ckf = sorted([f for f in os.listdir(snap_dir) if f.startswith("v5_V5C_")])
        if ckf:
            ck = torch.load(os.path.join(snap_dir, ckf[-1]), map_location=device)
            model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
    elif version == 'v2':
        snap_dir = f"./snapshots_aid_v2_seed{seed}/"
        from train_aid_v2 import SpikeAdaptSC_v2
        model = SpikeAdaptSC_v2(C_in=1024, C1=256, C2=36, T=T_STEPS,
                                target_rate=0.75, grid_size=14).to(device)
        s3f = sorted([f for f in os.listdir(snap_dir) if f.startswith("v2_s3_")])
        if s3f:
            ck = torch.load(os.path.join(snap_dir, s3f[-1]), map_location=device)
            model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
    else:
        raise ValueError(f"Unknown version: {version}")

    model.eval(); back.eval()
    return front, model, back


def evaluate(front, model, back, loader, noise_param=0.0, target_rate_override=None):
    model.eval(); back.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            if target_rate_override is not None:
                Fp, stats = model(feat, noise_param=noise_param,
                                  target_rate_override=target_rate_override)
            else:
                Fp, stats = model(feat, noise_param=noise_param)
            preds = back(Fp).argmax(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total


def evaluate_random_mask(front, model, back, loader, noise_param=0.0, target_rate=0.75):
    """Evaluate with random mask instead of learned importance."""
    model.eval(); back.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            B = feat.size(0)

            # Run encoder
            all_S2, m1, m2 = [], None, None
            for t in range(T_STEPS):
                if hasattr(model, 'encoder'):
                    if hasattr(model.encoder, 'forward'):
                        _, s2, m1, m2 = model.encoder(feat, m1, m2, t=t)
                else:
                    _, s2, m1, m2 = model.encoder(feat, m1, m2, t=t)
                all_S2.append(s2)

            # Random mask
            H, W = all_S2[0].shape[2], all_S2[0].shape[3]
            k = max(1, int(target_rate * H * W))
            mask = torch.zeros(B, 1, H, W, device=device)
            for b in range(B):
                idx = torch.randperm(H * W, device=device)[:k]
                mask_flat = torch.zeros(H * W, device=device)
                mask_flat[idx] = 1.0
                mask[b, 0] = mask_flat.view(H, W)

            # Channel + decoder
            recv = [model.channel(all_S2[t] * mask, noise_param) for t in range(T_STEPS)]
            Fp = model.decoder(recv, mask)
            preds = back(Fp).argmax(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total


def evaluate_uniform_mask(front, model, back, loader, noise_param=0.0, target_rate=0.75):
    """Evaluate with uniform grid mask (every k-th block)."""
    model.eval(); back.eval()
    correct, total = 0, 0

    # Pre-compute uniform mask
    H, W = 14, 14
    k = max(1, int(target_rate * H * W))
    mask_flat = torch.zeros(H * W, device=device)
    step = max(1, H * W // k)
    indices = torch.arange(0, H * W, step, device=device)[:k]
    mask_flat[indices] = 1.0
    uniform_mask = mask_flat.view(1, 1, H, W)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)

            all_S2, m1, m2 = [], None, None
            for t in range(T_STEPS):
                _, s2, m1, m2 = model.encoder(feat, m1, m2, t=t)
                all_S2.append(s2)

            mask = uniform_mask.expand(feat.size(0), -1, -1, -1)
            recv = [model.channel(all_S2[t] * mask, noise_param) for t in range(T_STEPS)]
            Fp = model.decoder(recv, mask)
            preds = back(Fp).argmax(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total


# ############################################################################
# ABLATION 1: ρ SWEEP
# ############################################################################

def ablation_rho_sweep(test_loader, seed=42):
    print("=" * 80)
    print("ABLATION 1: ρ (TARGET MASK RATE) SWEEP")
    print("=" * 80)

    rho_values = [0.25, 0.50, 0.625, 0.75, 0.875, 1.0]
    ber_values = [0.00, 0.10, 0.20, 0.30]
    versions = ['v2', 'v4a', 'v5c']

    results = {}
    for ver in versions:
        print(f"\n  Loading {ver}...")
        front, model, back = load_model_and_back(ver, seed)
        results[ver] = {}

        for rho in rho_values:
            results[ver][rho] = {}
            for ber in ber_values:
                acc = evaluate(front, model, back, test_loader,
                             noise_param=ber, target_rate_override=rho)
                results[ver][rho][ber] = acc
                print(f"    {ver} ρ={rho:.3f} BER={ber:.2f}: {acc:.2f}%")

    # Print table
    print(f"\n{'='*80}")
    print("ρ SWEEP RESULTS TABLE")
    print(f"{'='*80}")
    print(f"{'ρ':>6}", end="")
    for ver in versions:
        for ber in ber_values:
            print(f" | {ver.upper()} B{ber:.2f}", end="")
    print()
    print("-" * 80)
    for rho in rho_values:
        print(f"{rho:>6.3f}", end="")
        for ver in versions:
            for ber in ber_values:
                print(f" | {results[ver][rho][ber]:>10.2f}", end="")
        print()

    # Compute bandwidth savings
    print(f"\n  Bandwidth savings:")
    for rho in rho_values:
        bits_full = 14 * 14 * 36 * 8  # full transmission
        bits_masked = int(rho * 14 * 14) * 36 * 8
        saving = 100 * (1 - bits_masked / bits_full)
        print(f"    ρ={rho:.3f}: {bits_masked} bits ({saving:.1f}% saved)")

    return results


# ############################################################################
# ABLATION 2: MASK COMPARISON
# ############################################################################

def ablation_mask_comparison(test_loader, seed=42):
    print(f"\n{'='*80}")
    print("ABLATION 2: MASK COMPARISON (LEARNED vs RANDOM vs UNIFORM)")
    print(f"{'='*80}")

    rho_values = [0.50, 0.75, 1.0]
    ber_values = [0.00, 0.15, 0.30]
    versions = ['v4a', 'v5c']

    results = {}
    for ver in versions:
        print(f"\n  Loading {ver}...")
        front, model, back = load_model_and_back(ver, seed)
        results[ver] = {}

        for rho in rho_values:
            results[ver][rho] = {}
            for ber in ber_values:
                # Learned mask
                acc_learned = evaluate(front, model, back, test_loader,
                                      noise_param=ber, target_rate_override=rho)

                # Random mask (average over 3 runs for stability)
                acc_random_runs = []
                for _ in range(3):
                    acc_r = evaluate_random_mask(front, model, back, test_loader,
                                                noise_param=ber, target_rate=rho)
                    acc_random_runs.append(acc_r)
                acc_random = np.mean(acc_random_runs)

                # Uniform mask
                acc_uniform = evaluate_uniform_mask(front, model, back, test_loader,
                                                    noise_param=ber, target_rate=rho)

                results[ver][rho][ber] = {
                    'learned': acc_learned,
                    'random': acc_random,
                    'uniform': acc_uniform,
                    'delta_random': acc_learned - acc_random,
                    'delta_uniform': acc_learned - acc_uniform,
                }
                print(f"    {ver} ρ={rho:.2f} BER={ber:.2f}: "
                      f"Learned={acc_learned:.2f}% Random={acc_random:.2f}% "
                      f"Uniform={acc_uniform:.2f}% "
                      f"(Δ_rand={acc_learned-acc_random:+.2f}, Δ_uni={acc_learned-acc_uniform:+.2f})")

    # Print table
    print(f"\n{'='*80}")
    print("MASK COMPARISON TABLE")
    print(f"{'='*80}")
    for ver in versions:
        print(f"\n  {ver.upper()}:")
        print(f"  {'ρ':>5} {'BER':>5} {'Learned':>8} {'Random':>8} {'Uniform':>8} {'Δ Rand':>8} {'Δ Uni':>8}")
        print(f"  {'-'*55}")
        for rho in rho_values:
            for ber in ber_values:
                r = results[ver][rho][ber]
                print(f"  {rho:>5.2f} {ber:>5.2f} {r['learned']:>8.2f} {r['random']:>8.2f} "
                      f"{r['uniform']:>8.2f} {r['delta_random']:>+8.2f} {r['delta_uniform']:>+8.2f}")

    return results


# ############################################################################
# MAIN
# ############################################################################

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # Load test data
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])
    test_ds = AIDDataset("./data", tf_test, split='test', seed=args.seed)
    test_loader = DataLoader(test_ds, 64, False, num_workers=4)

    all_results = {}

    if args.ablation in ('rho', 'all'):
        rho_results = ablation_rho_sweep(test_loader, args.seed)
        all_results['rho_sweep'] = rho_results

    if args.ablation in ('mask', 'all'):
        mask_results = ablation_mask_comparison(test_loader, args.seed)
        all_results['mask_comparison'] = mask_results

    # Save
    with open('./eval/ablation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✅ Ablation results saved to eval/ablation_results.json")
