#!/usr/bin/env python3
"""Fair comparison: accuracy vs BER at matched rho for ALL methods.

Addresses two fairness issues:
  1. All methods evaluated at SAME rho (0.75) — equal bandwidth
  2. For rho sweep: compare learned masking vs random masking vs no masking

Methods evaluated:
  - SpikeAdapt-SC (learned mask, rho=0.75) — our best
  - SpikeAdapt-SC + Hyperprior (learned mask, rho=0.75) — our best + hyperprior
  - SNN-SC (no mask, rho=1.0 AND rho=0.75 with random mask) — ablation
  - Random mask at each rho — fair baseline for masking comparison

For alpha plots: use random masking as baseline (fair across rho).

Output: eval/seed_results/fair_comparison_results.json
        eval/figures/fair_*.pdf

Usage:
    python eval/eval_fair_comparison.py
"""

import os, sys, json, random, math, glob
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from train_aid_v2 import ResNet50Front, ResNet50Back, BSC_Channel, LearnedBlockMask
from train_aid_v5 import EncoderV5, DecoderV5, LIFNeuron, MPBN
from noise_aware_scorer import NoiseAwareScorer
from run_final_pipeline import AIDDataset5050, RESISC45Dataset, SpikeAdaptSC_v5c_NA
from models.spikeadapt_sc_hyper import SpikeAdaptSC_Hyper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_STEPS = 8
BER_SWEEP = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
RHO_SWEEP = [0.10, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 1.0]


def evaluate_learned_mask(model, back, front, loader, ber, target_rate=None):
    """Evaluate with learned noise-aware masking."""
    model.eval(); back.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            if target_rate is not None:
                Fp, _ = model(feat, noise_param=ber, target_rate_override=target_rate)
            else:
                Fp, _ = model(feat, noise_param=ber)
            correct += back(Fp).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total


def evaluate_hyper(model, back, front, loader, ber):
    """Evaluate with hyperprior (always at trained rho=0.75)."""
    model.eval(); back.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            Fp, _ = model(front(imgs), noise_param=ber, use_hyperprior=True)
            correct += back(Fp).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total


def evaluate_random_mask(model, back, front, loader, ber, rho, n_draws=10):
    """Evaluate with RANDOM spatial masking at given rho (fair baseline)."""
    model.eval(); back.eval()
    accs = []
    for draw in range(n_draws):
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                feat = front(imgs)
                B = imgs.size(0)

                # Encode spikes
                all_S2, m1, m2 = [], None, None
                for t in range(T_STEPS):
                    _, s2, m1, m2 = model.encoder(feat, m1, m2, t=t)
                    all_S2.append(s2)

                # Random mask
                H, W = all_S2[0].shape[2], all_S2[0].shape[3]
                n_keep = max(1, int(rho * H * W))
                mask = torch.zeros(B, 1, H, W, device=device)
                for b in range(B):
                    indices = torch.randperm(H * W)[:n_keep]
                    mask[b, 0].view(-1)[indices] = 1.0

                # Channel + decode
                channel = model.channel
                recv = [channel(all_S2[t] * mask, ber) for t in range(T_STEPS)]
                Fp = model.decoder(recv, mask)
                correct += back(Fp).argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        accs.append(100. * correct / total)
    return np.mean(accs)


def evaluate_no_mask(model, back, front, loader, ber):
    """Evaluate with NO masking (rho=1.0, all blocks transmitted)."""
    return evaluate_learned_mask(model, back, front, loader, ber, target_rate=1.0)


def run_dataset(ds_name, n_classes, model_v5, model_hyper, back_v5, back_hyper,
                front, test_loader):
    """Run all evaluations for one dataset."""
    results = {
        'ber_sweep_matched_rho': {},
        'rho_sweep_learned': {},
        'rho_sweep_random': {},
        'alpha_v1_learned': {},
        'alpha_v1_random': {},
        'alpha_v2_learned': {},
        'alpha_v2_random': {},
    }

    # ===== 1. BER sweep at matched rho=0.75 for all methods =====
    print(f"\n{'='*60}")
    print(f"  {ds_name}: BER sweep at matched rho=0.75")
    print(f"{'='*60}")
    print(f"{'BER':<8} {'Learned':>8} {'Random':>8} {'NoMask':>8} {'Hyper':>8}")
    print('-' * 45)

    for ber in BER_SWEEP:
        learned = evaluate_learned_mask(model_v5, back_v5, front, test_loader, ber)
        rand = evaluate_random_mask(model_v5, back_v5, front, test_loader, ber, 0.75, n_draws=5)
        nomask = evaluate_no_mask(model_v5, back_v5, front, test_loader, ber)
        hyper = evaluate_hyper(model_hyper, back_hyper, front, test_loader, ber)

        label = 'Clean' if ber == 0 else f'{ber:.2f}'
        print(f"  {label:<6} {learned:>7.2f}% {rand:>7.2f}% {nomask:>7.2f}% {hyper:>7.2f}%")

        results['ber_sweep_matched_rho'][str(ber)] = {
            'learned_075': round(learned, 2),
            'random_075': round(rand, 2),
            'no_mask_100': round(nomask, 2),
            'hyper_075': round(hyper, 2),
        }

    # ===== 2. Rho sweep: learned vs random masking =====
    print(f"\n{'='*60}")
    print(f"  {ds_name}: Rho sweep (learned vs random masking)")
    print(f"{'='*60}")

    for ber in [0.0, 0.15, 0.30, 0.40]:
        results['rho_sweep_learned'][str(ber)] = {}
        results['rho_sweep_random'][str(ber)] = {}

        label = 'Clean' if ber == 0 else f'BER={ber:.2f}'
        print(f"\n  {label}:")
        print(f"  {'rho':<8} {'Learned':>8} {'Random':>8} {'Delta':>8}")
        print(f"  {'-'*35}")

        for rho in RHO_SWEEP:
            learned = evaluate_learned_mask(model_v5, back_v5, front, test_loader, ber, rho)
            rand = evaluate_random_mask(model_v5, back_v5, front, test_loader, ber, rho, n_draws=5)

            results['rho_sweep_learned'][str(ber)][str(rho)] = round(learned, 2)
            results['rho_sweep_random'][str(ber)][str(rho)] = round(rand, 2)

            delta = learned - rand
            print(f"  {rho:<8.3f} {learned:>7.2f}% {rand:>7.2f}% {delta:>+7.2f}%")

    # ===== 3. Compute both alpha versions =====
    for mask_type, rho_data, alpha_v1_key, alpha_v2_key in [
        ('learned', results['rho_sweep_learned'], 'alpha_v1_learned', 'alpha_v2_learned'),
        ('random', results['rho_sweep_random'], 'alpha_v1_random', 'alpha_v2_random'),
    ]:
        for ber_str, rho_dict in rho_data.items():
            rhos = sorted([float(r) for r in rho_dict.keys()])
            a_full = rho_dict.get('1.0', rho_dict[str(rhos[-1])])  # A(rho=1)
            a_min = rho_dict[str(rhos[0])]  # A(rho_min)

            results[alpha_v1_key][ber_str] = {}
            results[alpha_v2_key][ber_str] = {}

            for rho in rhos:
                acc = rho_dict[str(rho)]
                alpha_v1 = (acc - a_full) / a_full if a_full > 0.1 else 0
                alpha_v2 = (acc - a_min) / a_min if a_min > 0.1 else 0
                results[alpha_v1_key][ber_str][str(rho)] = round(alpha_v1, 4)
                results[alpha_v2_key][ber_str][str(rho)] = round(alpha_v2, 4)

    return results


def main():
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    print(f"Device: {device}")

    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])

    all_results = {}

    for ds_name, n_classes, DsCls, ds_kwargs, bb_path, ck_v5_path, ck_hyper_path in [
        ('AID', 30, AIDDataset5050, dict(seed=42),
         './snapshots_aid_5050_seed42/backbone_best.pth',
         './snapshots_aid_v5cna_seed42/v5cna_best_95.42.pth',
         './snapshots_aid_hyper_seed42/h2_best_95.48.pth'),
        ('RESISC45', 45, RESISC45Dataset, dict(train_ratio=0.20, seed=42),
         './snapshots_resisc45_5050_seed42/backbone_best.pth',
         './snapshots_resisc45_v5cna_seed42/v5cna_best_92.01.pth',
         './snapshots_resisc45_hyper_seed42/h2_best_92.10.pth'),
    ]:
        print(f"\n{'#'*60}\n  {ds_name}\n{'#'*60}")

        test_ds = DsCls("./data", tf_test, 'test', **ds_kwargs)
        test_loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)

        # Load backbone
        front = ResNet50Front(grid_size=14).to(device)
        bb = torch.load(bb_path, map_location=device, weights_only=False)
        front.load_state_dict({k: v for k, v in bb.items()
                               if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))},
                              strict=False)
        front.eval()
        for p in front.parameters(): p.requires_grad = False

        # V5C-NA model
        back_v5 = ResNet50Back(n_classes).to(device)
        model_v5 = SpikeAdaptSC_v5c_NA(C_in=1024, C1=256, C2=36, T=T_STEPS,
                                         target_rate=0.75, grid_size=14).to(device)
        ck = torch.load(ck_v5_path, map_location=device, weights_only=False)
        model_v5.load_state_dict(ck['model']); back_v5.load_state_dict(ck['back'])
        print(f"  Loaded V5C-NA: {ck_v5_path}")

        # Hyperprior model
        back_hyper = ResNet50Back(n_classes).to(device)
        model_hyper = SpikeAdaptSC_Hyper(C_in=1024, C1=256, C2=36, T=T_STEPS,
                                          target_rate=0.75, grid_size=14, C_hyper=8).to(device)
        if os.path.exists(ck_hyper_path):
            ck_h = torch.load(ck_hyper_path, map_location=device, weights_only=False)
            model_hyper.load_state_dict(ck_h['model'], strict=False)
            back_hyper.load_state_dict(ck_h['back'])
            print(f"  Loaded Hyperprior: {ck_hyper_path}")
        else:
            print(f"  WARNING: No hyperprior checkpoint at {ck_hyper_path}")
            # Fallback: use V5C-NA weights
            model_hyper.load_state_dict(ck['model'], strict=False)
            back_hyper.load_state_dict(ck['back'])

        all_results[ds_name] = run_dataset(
            ds_name, n_classes, model_v5, model_hyper, back_v5, back_hyper,
            front, test_loader)

        del model_v5, model_hyper, back_v5, back_hyper, front
        torch.cuda.empty_cache()

    # Save
    os.makedirs('eval/seed_results', exist_ok=True)
    out = 'eval/seed_results/fair_comparison_results.json'
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out}")

    # ===== Generate plots =====
    plot_all(all_results)


def plot_all(all_results):
    """Generate all fair comparison plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs('eval/figures', exist_ok=True)

    # ===== Plot 1: BER sweep at matched rho=0.75 =====
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for col, ds in enumerate(['AID', 'RESISC45']):
        if ds not in all_results:
            continue
        ax = axes[col]
        d = all_results[ds]['ber_sweep_matched_rho']
        bers = sorted([float(b) for b in d.keys()])

        learned = [d[str(b)]['learned_075'] for b in bers]
        rand = [d[str(b)]['random_075'] for b in bers]
        nomask = [d[str(b)]['no_mask_100'] for b in bers]
        hyper = [d[str(b)]['hyper_075'] for b in bers]

        ax.plot(bers, hyper, 'D-', color='#7c3aed', linewidth=2.5, markersize=6,
                label='SpikeAdapt-SC + Hyper (learned, rho=0.75)', zorder=6)
        ax.plot(bers, learned, 'o-', color='#2563eb', linewidth=2.5, markersize=6,
                label='SpikeAdapt-SC (learned mask, rho=0.75)', zorder=5)
        ax.plot(bers, rand, 's--', color='#f59e0b', linewidth=1.5, markersize=5,
                label='SpikeAdapt-SC (random mask, rho=0.75)')
        ax.plot(bers, nomask, 'v:', color='#94a3b8', linewidth=1.5, markersize=5,
                label='SNN-SC (no mask, rho=1.0)')

        n_cls = 30 if ds == 'AID' else 45
        ax.axhline(y=100/n_cls, color='black', linestyle=':', linewidth=1, alpha=0.4)
        ax.set_xlabel('BER', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(f'{ds}: Fair BER Sweep (matched bandwidth)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=7.5, loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.02, 0.52)

    plt.tight_layout()
    fig.savefig('eval/figures/fair_ber_sweep_matched_rho.pdf', dpi=300, bbox_inches='tight')
    fig.savefig('eval/figures/fair_ber_sweep_matched_rho.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: eval/figures/fair_ber_sweep_matched_rho.pdf')

    # ===== Plot 2: Accuracy vs rho (learned vs random) =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for col, ds in enumerate(['AID', 'RESISC45']):
        if ds not in all_results:
            continue
        for row, ber in enumerate([0.0, 0.30]):
            ax = axes[row, col]
            d_l = all_results[ds]['rho_sweep_learned'].get(str(ber), {})
            d_r = all_results[ds]['rho_sweep_random'].get(str(ber), {})
            if not d_l:
                continue
            rhos = sorted([float(r) for r in d_l.keys()])
            learned = [d_l[str(r)] for r in rhos]
            rand = [d_r[str(r)] for r in rhos]

            ax.plot(rhos, learned, 'o-', color='#2563eb', linewidth=2.5, markersize=7,
                    label='Learned mask (noise-aware)')
            ax.plot(rhos, rand, 's--', color='#f59e0b', linewidth=2, markersize=6,
                    label='Random mask (fair baseline)')
            ax.fill_between(rhos, rand, learned, alpha=0.1, color='#2563eb',
                           where=[l >= r for l, r in zip(learned, rand)])

            label = 'Clean (BER=0)' if ber == 0 else f'BER={ber:.2f}'
            ax.set_xlabel('Transmission rate (rho)', fontsize=11)
            ax.set_ylabel('Accuracy (%)', fontsize=11)
            ax.set_title(f'{ds}: {label}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0.05, 1.05)

    fig.suptitle('Accuracy vs rho: Learned vs Random Masking (Fair Comparison)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig('eval/figures/fair_accuracy_vs_rho.pdf', dpi=300, bbox_inches='tight')
    fig.savefig('eval/figures/fair_accuracy_vs_rho.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: eval/figures/fair_accuracy_vs_rho.pdf')

    # ===== Plot 3: Alpha (both versions, learned vs random) =====
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    cmap = plt.cm.RdYlBu_r

    for col_base, ds in enumerate(['AID', 'RESISC45']):
        if ds not in all_results:
            continue
        for alpha_idx, (alpha_key, alpha_label) in enumerate([
            ('alpha_v1', 'alpha_1 = (A(rho)-A(1))/A(1)'),
            ('alpha_v2', 'alpha_2 = (A(rho)-A(min))/A(min)'),
        ]):
            for mask_idx, (mask_type, mask_label) in enumerate([
                ('learned', 'Learned mask'),
                ('random', 'Random mask'),
            ]):
                ax_col = col_base * 2 + mask_idx
                ax = axes[alpha_idx, ax_col]

                key = f'{alpha_key}_{mask_type}'
                d = all_results[ds].get(key, {})
                ber_vals = sorted([float(b) for b in d.keys()])
                norm = plt.Normalize(0, max(ber_vals) if ber_vals else 0.4)

                for ber in ber_vals:
                    rhos = sorted([float(r) for r in d[str(ber)].keys()])
                    alphas = [d[str(ber)][str(r)] for r in rhos]
                    color = cmap(norm(ber))
                    label_b = f'BER={ber:.2f}' if ber > 0 else 'Clean'
                    ax.plot(rhos, alphas, 'o-', color=color, linewidth=1.5, markersize=3.5, label=label_b)

                ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
                ax.set_xlabel('rho', fontsize=9)
                ax.set_ylabel(alpha_label.split('=')[0], fontsize=9)
                ax.set_title(f'{ds}: {mask_label}', fontsize=10, fontweight='bold')
                ax.legend(fontsize=5.5, ncol=2, loc='best')
                ax.set_xlim(0.05, 1.05)
                ax.grid(True, alpha=0.3)

    fig.suptitle('Alpha Metrics: Learned vs Random Masking (Fair Comparison)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig('eval/figures/fair_alpha_comparison.pdf', dpi=300, bbox_inches='tight')
    fig.savefig('eval/figures/fair_alpha_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: eval/figures/fair_alpha_comparison.pdf')

    print('\nAll fair comparison plots saved to eval/figures/')


if __name__ == '__main__':
    main()
