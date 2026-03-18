"""Comprehensive analysis of SpikeAdapt-SC v2 (14×14 grid).

Checks:
1. Random-mask distribution (100 draws at ρ=0.75 and ρ=0.50)
2. Per-block score statistics (variance, Gini, entropy, CoV)
3. Kept-vs-dropped margin
4. Leave-one-out ablation correlation
5. Mask diversity and stability
6. BER sweep for v2 vs baselines
7. Channel conditioning verification (does scorer respond to BER changes?)
8. Temporal early stopping analysis
"""

import os, sys, random, json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 9,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03, 'axes.linewidth': 0.6,
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import model classes
sys.path.insert(0, './train')
from train_aid_v2 import (
    AIDDataset, ResNet50Front, ResNet50Back, SpikeAdaptSC_v2, SNNSC, CNNUni,
    ChannelConditionedScorer, evaluate, evaluate_with_early_stop
)

SNAP_V1 = "./snapshots_aid/"
SNAP_V2 = "./snapshots_aid_v2_seed42/"
OUT_DIR = "./paper/figures/"
T_STEPS = 8

def load_v2():
    front = ResNet50Front(grid_size=14).to(device)
    back = ResNet50Back(30).to(device)
    bb = torch.load(os.path.join(SNAP_V1, "backbone_best.pth"), map_location=device)
    front.load_state_dict({k:v for k,v in bb.items()
                           if not k.startswith(('layer4.','fc.','avgpool.','spatial_pool.'))}, strict=False)
    front.eval()

    model = SpikeAdaptSC_v2(C_in=1024, C1=256, C2=36, T=8, target_rate=0.75,
                             channel_type='bsc', grid_size=14).to(device)
    s3f = sorted([f for f in os.listdir(SNAP_V2) if f.startswith("v2_s3_")])
    if s3f:
        ck = torch.load(os.path.join(SNAP_V2, s3f[-1]), map_location=device)
        model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
        print(f"Loaded v2: {s3f[-1]}")
    model.eval(); back.eval()
    return front, back, model

def load_snnsc(front):
    back = ResNet50Back(30).to(device)
    snnsc = SNNSC(C_in=1024, C1=256, C2=36, T=8).to(device)
    s3f = sorted([f for f in os.listdir(SNAP_V2) if f.startswith("snnsc_s3_")])
    if s3f:
        ck = torch.load(os.path.join(SNAP_V2, s3f[-1]), map_location=device)
        snnsc.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
        print(f"Loaded SNNSC: {s3f[-1]}")
    snnsc.eval(); back.eval()
    return back, snnsc

def load_cnnuni(front):
    back = ResNet50Back(30).to(device)
    cnnuni = CNNUni(C_in=1024, C1=256, C2=36, n_bits=8).to(device)
    s3f = sorted([f for f in os.listdir(SNAP_V2) if f.startswith("cnnuni_s3_")])
    if s3f:
        ck = torch.load(os.path.join(SNAP_V2, s3f[-1]), map_location=device)
        cnnuni.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
        print(f"Loaded CNNUni: {s3f[-1]}")
    cnnuni.eval(); back.eval()
    return back, cnnuni


if __name__ == "__main__":
    print(f"Device: {device}")
    front, back, model = load_v2()

    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                     T.Normalize((.485,.456,.406),(.229,.224,.225))])
    ds = AIDDataset("./data", transform=tf, split='test', seed=42)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)
    N = len(ds)
    print(f"Test set: {N} images, Grid: 14×14 = 196 blocks")

    # ============================================================
    # COLLECT ALL SCORES AND SPIKES
    # ============================================================
    print("\n=== Collecting data ===")
    all_scores = []
    all_labels = []
    all_S2_batches = []
    all_feats = []

    with torch.no_grad():
        for bi, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device)
            feat = front(imgs)
            B = imgs.shape[0]

            S2, m1, m2 = [], None, None
            for t in range(T_STEPS):
                _, s2, m1, m2 = model.encoder(feat, m1, m2)
                S2.append(s2)

            importance = model.scorer(S2, 0.0)
            all_scores.append(importance.cpu().numpy().reshape(B, -1))
            all_labels.extend(labels.numpy())
            all_S2_batches.append([s.cpu() for s in S2])
            all_feats.append(feat.cpu())

            if (bi+1) % 25 == 0: print(f"  Batch {bi+1}/{len(loader)}")

    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.array(all_labels)
    N_BLOCKS = all_scores.shape[1]  # 196
    print(f"  {N} images, {N_BLOCKS} blocks each")

    # ============================================================
    # CHECK 1: Score Statistics
    # ============================================================
    print("\n" + "="*60)
    print("CHECK 1: SCORE STATISTICS")
    print("="*60)

    per_var = np.var(all_scores, axis=1)
    per_gini = []
    for s in all_scores:
        x = np.sort(s); n = len(x)
        per_gini.append((2*np.sum(np.arange(1,n+1)*x) - (n+1)*np.sum(x)) / (n*np.sum(x)+1e-12))
    per_gini = np.array(per_gini)

    per_entropy = []
    for s in all_scores:
        p = s / (s.sum() + 1e-12)
        per_entropy.append(-np.sum(p * np.log(p + 1e-12)))
    per_entropy = np.array(per_entropy)
    max_entropy = np.log(N_BLOCKS)

    per_cov = np.std(all_scores, axis=1) / (np.mean(all_scores, axis=1) + 1e-12)

    print(f"  Variance:  {per_var.mean():.6f} ± {per_var.std():.6f}")
    print(f"  Gini:      {np.mean(per_gini):.4f} ± {np.std(per_gini):.4f}")
    print(f"  Entropy:   {np.mean(per_entropy):.4f} ± {np.std(per_entropy):.4f} (max={max_entropy:.4f})")
    print(f"  Entropy%:  {np.mean(per_entropy)/max_entropy*100:.2f}%")
    print(f"  CoV:       {per_cov.mean():.4f} ± {per_cov.std():.4f}")
    print(f"  Score range: [{all_scores.min():.4f}, {all_scores.max():.4f}]")
    mean_scores = all_scores.mean(0)
    print(f"  Blocks >0.9: {(mean_scores > 0.9).sum()}/{N_BLOCKS}")
    print(f"  Blocks <0.1: {(mean_scores < 0.1).sum()}/{N_BLOCKS}")

    # ============================================================
    # CHECK 2: Kept-Dropped Margin
    # ============================================================
    print("\n" + "="*60)
    print("CHECK 2: KEPT-DROPPED MARGIN")
    print("="*60)

    k = int(0.75 * N_BLOCKS)
    margins = []
    for i in range(N):
        si = np.argsort(all_scores[i])[::-1]
        margins.append(all_scores[i][si[:k]].mean() - all_scores[i][si[k:]].mean())
    margins = np.array(margins)
    print(f"  Margin at ρ=0.75: {margins.mean():.4f} ± {margins.std():.4f}")
    print(f"  All positive: {(margins > 0).all()}")
    print(f"  Margin > 0.05: {(margins > 0.05).sum()}/{N} ({(margins>0.05).mean()*100:.1f}%)")

    # ============================================================
    # CHECK 3: Random Mask Distribution (100 draws)
    # ============================================================
    print("\n" + "="*60)
    print("CHECK 3: RANDOM MASK DISTRIBUTION")
    print("="*60)

    # Learned accuracy
    learned_correct = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            Fp, _ = model(front(imgs), noise_param=0.0)
            learned_correct += back(Fp).argmax(1).eq(labels).sum().item()
    learned_acc = learned_correct / N * 100
    print(f"  Learned mask (ρ=0.75): {learned_acc:.2f}%")

    for rho, n_draws in [(0.75, 100), (0.50, 50)]:
        k_r = max(1, int(rho * N_BLOCKS))
        random_accs = []
        for draw in range(n_draws):
            correct = 0; total = 0
            with torch.no_grad():
                for bi in range(len(all_S2_batches)):
                    S2 = [s.to(device) for s in all_S2_batches[bi]]
                    B_cur = S2[0].shape[0]
                    labels_b = all_labels[total:total+B_cur]
                    rand_flat = torch.rand(B_cur, N_BLOCKS, device=device)
                    _, ridx = rand_flat.topk(k_r, 1)
                    mask_r = torch.zeros(B_cur, N_BLOCKS, device=device)
                    mask_r.scatter_(1, ridx, 1.0)
                    mask_r = mask_r.view(B_cur, 14, 14).unsqueeze(1)
                    recv = [S2[t] * mask_r for t in range(T_STEPS)]
                    Fp = model.decoder(recv, mask_r)
                    preds = back(Fp).argmax(1).cpu().numpy()
                    correct += (preds == labels_b).sum()
                    total += B_cur
            random_accs.append(correct / total * 100)
        ra = np.array(random_accs)

        # For ρ=0.50 with learned mask
        if rho == 0.50:
            correct_50 = 0
            with torch.no_grad():
                for imgs, labels in loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    Fp, _ = model(front(imgs), noise_param=0.0, target_rate_override=0.50)
                    correct_50 += back(Fp).argmax(1).eq(labels).sum().item()
            learned_50 = correct_50 / N * 100
            pct = (ra < learned_50).sum() / len(ra) * 100
            print(f"\n  ρ={rho}: Learned={learned_50:.2f}%, Random={ra.mean():.2f}±{ra.std():.2f}%")
            print(f"    Range: [{ra.min():.2f}, {ra.max():.2f}]")
            print(f"    Learned exceeds {pct:.0f}% of {n_draws} draws")
        else:
            pct = (ra < learned_acc).sum() / len(ra) * 100
            print(f"\n  ρ={rho}: Learned={learned_acc:.2f}%, Random={ra.mean():.2f}±{ra.std():.2f}%")
            print(f"    Range: [{ra.min():.2f}, {ra.max():.2f}]")
            print(f"    Learned exceeds {pct:.0f}% of {n_draws} draws")

    # ============================================================
    # CHECK 4: Per-Block Ablation Correlation
    # ============================================================
    print("\n" + "="*60)
    print("CHECK 4: ABLATION CORRELATION")
    print("="*60)

    N_SAMPLE = 50
    np.random.seed(42)
    sample_idx = np.random.choice(N, N_SAMPLE, replace=False)
    block_damage = np.zeros((N_SAMPLE, N_BLOCKS))
    block_scores = np.zeros((N_SAMPLE, N_BLOCKS))

    with torch.no_grad():
        for si, idx in enumerate(sample_idx):
            bi = idx // 16; li = idx % 16
            S2 = [s.to(device) for s in all_S2_batches[bi]]
            if li >= S2[0].shape[0]: continue
            S2_s = [s[li:li+1] for s in S2]

            full_mask = torch.ones(1, 1, 14, 14, device=device)
            recv_full = [S2_s[t] * full_mask for t in range(T_STEPS)]
            conf_full = F.softmax(back(model.decoder(recv_full, full_mask)), 1).max(1)[0].item()
            block_scores[si] = all_scores[idx]

            for b in range(N_BLOCKS):
                r, c = b // 14, b % 14
                mask_ab = torch.ones(1, 1, 14, 14, device=device)
                mask_ab[0, 0, r, c] = 0.0
                recv_ab = [S2_s[t] * mask_ab for t in range(T_STEPS)]
                conf_ab = F.softmax(back(model.decoder(recv_ab, mask_ab)), 1).max(1)[0].item()
                block_damage[si, b] = conf_full - conf_ab

            if (si+1) % 10 == 0: print(f"  Ablated {si+1}/{N_SAMPLE}")

    corr, p_val = scipy_stats.pearsonr(block_scores.flatten(), block_damage.flatten())
    sp_corr, sp_p = scipy_stats.spearmanr(block_scores.flatten(), block_damage.flatten())
    print(f"  Pearson r  = {corr:.4f} (p={p_val:.2e})")
    print(f"  Spearman ρ = {sp_corr:.4f} (p={sp_p:.2e})")

    # ============================================================
    # CHECK 5: Mask Diversity
    # ============================================================
    print("\n" + "="*60)
    print("CHECK 5: MASK DIVERSITY")
    print("="*60)

    k = int(0.75 * N_BLOCKS)
    masks = []
    for i in range(N):
        si = np.argsort(all_scores[i])[::-1][:k]
        masks.append(frozenset(si.tolist()))
    unique_masks = len(set(masks))
    print(f"  Unique masks: {unique_masks}/{N} ({unique_masks/N*100:.1f}%)")

    # Pairwise overlap (sample)
    ovlps = []
    for _ in range(2000):
        i, j = random.sample(range(N), 2)
        ovlps.append(len(masks[i] & masks[j]) / k)
    print(f"  Mean pairwise overlap: {np.mean(ovlps):.4f}")

    # ============================================================
    # CHECK 6: Channel Conditioning Verification
    # ============================================================
    print("\n" + "="*60)
    print("CHECK 6: CHANNEL CONDITIONING")
    print("="*60)
    print("  Does the scorer change its output when BER changes?")

    with torch.no_grad():
        sample_imgs = next(iter(loader))[0][:4].to(device)
        feat = front(sample_imgs)
        S2, m1, m2 = [], None, None
        for t in range(T_STEPS):
            _, s2, m1, m2 = model.encoder(feat, m1, m2)
            S2.append(s2)

        for ber in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
            scores = model.scorer(S2, ber).cpu().numpy()
            mean_s = scores.mean(); std_s = scores.std()
            k_test = int(0.75 * N_BLOCKS)
            n_diff = 0
            if ber > 0:
                scores_0 = model.scorer(S2, 0.0).cpu().numpy()
                for b in range(4):
                    mask_0 = set(np.argsort(scores_0[b].flatten())[-k_test:])
                    mask_b = set(np.argsort(scores[b].flatten())[-k_test:])
                    n_diff += len(mask_0 ^ mask_b) / 2
                n_diff /= 4
            print(f"    BER={ber:.2f}: mean={mean_s:.4f}, std={std_s:.4f}, "
                  f"mask_diff_from_clean={n_diff:.1f} blocks")

    # ============================================================
    # CHECK 7: BER Sweep (v2 vs baselines)
    # ============================================================
    print("\n" + "="*60)
    print("CHECK 7: BER SWEEP")
    print("="*60)

    ber_list = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    # v2 model
    v2_accs = []
    for ber in ber_list:
        acc, _ = evaluate(front, model, back, loader, noise_param=ber)
        v2_accs.append(acc)
        print(f"  v2 BER={ber:.2f}: {acc:.2f}%")

    # SNN-SC baseline
    back_bl, snnsc = load_snnsc(front)
    snnsc_accs = []
    for ber in ber_list:
        acc, _ = evaluate(front, snnsc, back_bl, loader, noise_param=ber, is_baseline=True)
        snnsc_accs.append(acc)
        print(f"  SNNSC BER={ber:.2f}: {acc:.2f}%")

    # CNN-Uni baseline
    back_cu, cnnuni = load_cnnuni(front)
    cnnuni_accs = []
    for ber in ber_list:
        acc, _ = evaluate(front, cnnuni, back_cu, loader, noise_param=ber, is_baseline=True)
        cnnuni_accs.append(acc)
        print(f"  CNNUni BER={ber:.2f}: {acc:.2f}%")

    # ============================================================
    # CHECK 8: Temporal Early Stopping
    # ============================================================
    print("\n" + "="*60)
    print("CHECK 8: TEMPORAL EARLY STOPPING")
    print("="*60)

    # Reload v2 back
    s3f = sorted([f for f in os.listdir(SNAP_V2) if f.startswith("v2_s3_")])
    if s3f:
        ck = torch.load(os.path.join(SNAP_V2, s3f[-1]), map_location=device)
        back.load_state_dict(ck['back'])

    for conf_th in [0.80, 0.85, 0.90, 0.95, 0.99]:
        for ber in [0.0, 0.15]:
            acc_es, avg_T = evaluate_with_early_stop(
                front, model, back, loader, noise_param=ber,
                conf_threshold=conf_th, min_T=2)
            ts = (1 - avg_T / T_STEPS) * 100
            print(f"  th={conf_th}, BER={ber:.2f}: {acc_es:.2f}%, "
                  f"T={avg_T:.1f}/{T_STEPS} ({ts:.0f}% saved)")

    # ============================================================
    # GENERATE FIGURES
    # ============================================================
    print("\n=== Generating figures ===")

    # Figure A: BER sweep comparison
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))
    ax.plot([b*100 for b in ber_list], v2_accs, 'o-', color='#1E88E5', lw=1.2, ms=4,
            label=f'SpikeAdapt-SC v2 (ρ=0.75)')
    ax.plot([b*100 for b in ber_list], snnsc_accs, 's--', color='#43A047', lw=1.0, ms=3.5,
            label=f'SNN-SC (ρ=1.0)')
    ax.plot([b*100 for b in ber_list], cnnuni_accs, '^:', color='#E53935', lw=1.0, ms=3.5,
            label=f'CNN-Uni 8-bit')
    ax.set_xlabel('BER (%)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('v2: BER Sweep Comparison (14×14 grid)')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(OUT_DIR, 'v2_ber_sweep.png'), facecolor='white')
    print("  Saved v2_ber_sweep.png")
    plt.close()

    # Figure B: Score distributions
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.4))
    # Sorted profiles
    ax = axes[0]
    np.random.seed(42)
    for ei, idx in enumerate(np.random.choice(N, 5, replace=False)):
        ax.plot(range(N_BLOCKS), np.sort(all_scores[idx])[::-1], '-', alpha=0.7, lw=0.8)
    ax.axhline(1/N_BLOCKS, color='gray', ls=':', lw=0.6)
    ax.set_xlabel('Block rank'); ax.set_ylabel('Score'); ax.set_title('(a) Sorted Profiles')
    # Gini
    ax = axes[1]
    ax.hist(per_gini, bins=40, color='#FF7043', edgecolor='white', lw=0.3, alpha=0.85)
    ax.axvline(per_gini.mean(), color='#333', ls='--', lw=1, label=f'Mean={per_gini.mean():.4f}')
    ax.set_xlabel('Gini'); ax.set_ylabel('Count'); ax.set_title('(b) Score Gini'); ax.legend(fontsize=6)
    # Entropy
    ax = axes[2]
    ax.hist(per_entropy, bins=40, color='#7E57C2', edgecolor='white', lw=0.3, alpha=0.85)
    ax.axvline(per_entropy.mean(), color='#333', ls='--', lw=1, label=f'Mean={per_entropy.mean():.3f}')
    ax.axvline(max_entropy, color='#E53935', ls=':', lw=1, label=f'Max={max_entropy:.3f}')
    ax.set_xlabel('Entropy'); ax.set_ylabel('Count'); ax.set_title('(c) Score Entropy'); ax.legend(fontsize=5.5)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'v2_score_distributions.png'), facecolor='white')
    print("  Saved v2_score_distributions.png")
    plt.close()

    # Figure C: Ablation correlation
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0))
    ax.scatter(block_scores.flatten(), block_damage.flatten(), s=1, alpha=0.2, c='#1E88E5')
    ax.set_xlabel('Importance Score'); ax.set_ylabel('Confidence Drop (ablated)')
    ax.set_title(f'v2 Ablation: r={corr:.4f} (p={p_val:.2e})')
    ax.grid(True, alpha=0.2)
    fig.savefig(os.path.join(OUT_DIR, 'v2_ablation_corr.png'), facecolor='white')
    print("  Saved v2_ablation_corr.png")
    plt.close()

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "="*60)
    print("COMPREHENSIVE COMPARISON: v1 (8×8) vs v2 (14×14)")
    print("="*60)
    print(f"{'Metric':<30} {'v1 (8×8)':<15} {'v2 (14×14)':<15} {'Better?':<8}")
    print("-"*68)
    print(f"{'Grid blocks':<30} {'64':<15} {'196':<15} {'v2':>8}")
    print(f"{'C2 channels':<30} {'128':<15} {'36':<15} {'-':>8}")
    print(f"{'Payload per frame':<30} {'65536':<15} {'56448':<15} {'v2':>8}")
    print(f"{'Clean accuracy':<30} {'96.10%':<15} {f'{learned_acc:.2f}%':<15} {'?':>8}")
    print(f"{'Score Gini':<30} {'0.019':<15} {f'{per_gini.mean():.4f}':<15} {'?':>8}")
    print(f"{'Score entropy ratio':<30} {'99.98%':<15} {f'{per_entropy.mean()/max_entropy*100:.2f}%':<15} {'?':>8}")
    print(f"{'Ablation Pearson r':<30} {'0.011':<15} {f'{corr:.4f}':<15} {'?':>8}")
    print(f"{'Unique masks':<30} {'1982/2000':<15} {f'{unique_masks}/{N}':<15} {'?':>8}")
    print(f"{'Margin (ρ=0.75)':<30} {'0.055':<15} {f'{margins.mean():.4f}':<15} {'?':>8}")
    print(f"{'vs SNN-SC (clean)':<30} {'-':<15} {f'{v2_accs[0]:.2f} vs {snnsc_accs[0]:.2f}':<15}")
    print(f"{'vs CNN-Uni (clean)':<30} {'-':<15} {f'{v2_accs[0]:.2f} vs {cnnuni_accs[0]:.2f}':<15}")

    # Save results
    results = {
        'v2_clean': learned_acc,
        'v2_gini': float(per_gini.mean()),
        'v2_entropy_ratio': float(per_entropy.mean() / max_entropy),
        'v2_ablation_r': float(corr),
        'v2_ablation_p': float(p_val),
        'v2_unique_masks': unique_masks,
        'v2_margin': float(margins.mean()),
        'v2_ber_sweep': {str(b): a for b, a in zip(ber_list, v2_accs)},
        'snnsc_ber_sweep': {str(b): a for b, a in zip(ber_list, snnsc_accs)},
        'cnnuni_ber_sweep': {str(b): a for b, a in zip(ber_list, cnnuni_accs)},
    }
    with open(os.path.join(OUT_DIR, 'v2_analysis_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ All results saved")
