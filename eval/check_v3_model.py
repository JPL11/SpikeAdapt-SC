"""Quick randomization check on the v3 enhanced scorer model.
Runs the key checks: random distribution, ablation correlation, score stats.
"""

import os, sys, random, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy import stats as scipy_stats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SNAP_DIR = "./snapshots_aid/"

# Import model classes from retrain script
sys.path.insert(0, './train')
from retrain_enhanced_scorer import (
    AIDDataset, ResNet50Front, ResNet50Back, SpikeAdaptSC_v2,
    gini_coefficient_batch
)

T_STEPS = 8

def decode_classify(dec, back, S2, mask):
    recv = [S2[t] * mask for t in range(T_STEPS)]
    Fp = dec(recv, mask)
    return back(Fp), Fp

if __name__ == "__main__":
    print(f"Device: {device}")

    # Load models
    front = ResNet50Front(224).to(device)
    back = ResNet50Back(30).to(device)
    bb = torch.load(os.path.join(SNAP_DIR, "backbone_best.pth"), map_location=device)
    front.load_state_dict({k:v for k,v in bb.items() if not k.startswith(('layer4.','fc.','avgpool.'))}, strict=False)
    front.eval()

    model = SpikeAdaptSC_v2(C_in=1024, C1=256, C2=128, T=8, target_rate=0.75).to(device)

    # Load v3 checkpoint
    v3f = sorted([f for f in os.listdir(SNAP_DIR) if f.startswith("bsc_v3_s3_")])
    assert v3f, "No v3 checkpoint found!"
    ck = torch.load(os.path.join(SNAP_DIR, v3f[-1]), map_location=device)
    model.load_state_dict(ck['model'])
    back.load_state_dict(ck['back'])
    model.eval(); back.eval()
    print(f"Loaded: {v3f[-1]}")

    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                     T.Normalize((.485,.456,.406),(.229,.224,.225))])
    ds = AIDDataset("./data", transform=tf, split='test')
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)
    print(f"Test set: {len(ds)} images")

    # ============================================================
    # PHASE 1: Collect scores and spikes
    # ============================================================
    print("\n=== Collecting scores ===")
    all_S2 = []
    all_feats = []
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for bi, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device)
            feat = front(imgs)
            # Run through model to get importance scores
            Fp, stats = model(feat, noise_param=0.0)
            importance = stats['importance']  # (B, 8, 8)
            B = imgs.shape[0]
            all_scores.append(importance.cpu().numpy().reshape(B, -1))
            all_labels.extend(labels.numpy())

            # Store spikes for later
            m1, m2 = None, None
            S2_batch = []
            for t in range(T_STEPS):
                _, s2, m1, m2 = model.encoder(feat, m1, m2)
                S2_batch.append(s2)
            all_S2.append([s.cpu() for s in S2_batch])
            all_feats.append(feat.cpu())

            if (bi+1) % 20 == 0:
                print(f"  Batch {bi+1}/{len(loader)}")

    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.array(all_labels)
    N = len(all_labels)
    print(f"  {N} images collected")

    # ============================================================
    # CHECK 1: Score statistics
    # ============================================================
    print("\n=== Score Statistics ===")
    per_image_var = np.var(all_scores, axis=1)
    per_image_gini = []
    for s in all_scores:
        x = np.sort(s)
        n = len(x)
        idx = np.arange(1, n+1)
        g = (2*np.sum(idx*x) - (n+1)*np.sum(x)) / (n*np.sum(x)+1e-12)
        per_image_gini.append(g)
    per_image_gini = np.array(per_image_gini)

    print(f"  Per-image variance: {per_image_var.mean():.6f} ± {per_image_var.std():.6f}")
    print(f"  Per-image Gini:     {np.mean(per_image_gini):.4f} ± {np.std(per_image_gini):.4f}")
    print(f"  Score range: [{all_scores.min():.4f}, {all_scores.max():.4f}]")
    print(f"  Blocks with mean>0.9: {(all_scores.mean(0) > 0.9).sum()}/64")
    print(f"  Blocks with mean<0.1: {(all_scores.mean(0) < 0.1).sum()}/64")

    # Kept-dropped margin
    k = int(0.75 * 64)
    margins = []
    for i in range(N):
        si = np.argsort(all_scores[i])[::-1]
        margins.append(all_scores[i][si[:k]].mean() - all_scores[i][si[k:]].mean())
    margins = np.array(margins)
    print(f"  Kept-dropped margin: {margins.mean():.4f} ± {margins.std():.4f}")

    # ============================================================
    # CHECK 2: Learned accuracy + random distribution (50 draws)
    # ============================================================
    print("\n=== Random Distribution (50 draws, ρ=0.75) ===")

    # Learned accuracy
    learned_correct = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            Fp, _ = model(feat, noise_param=0.0)
            learned_correct += back(Fp).argmax(1).eq(labels).sum().item()
    learned_acc = learned_correct / N * 100
    print(f"  Learned mask: {learned_acc:.2f}%")

    # Random draws
    random_accs = []
    for draw in range(50):
        correct = 0; total = 0
        with torch.no_grad():
            for bi in range(len(all_S2)):
                S2_batch = [s.to(device) for s in all_S2[bi]]
                feat_batch = all_feats[bi].to(device)
                B_cur = S2_batch[0].shape[0]
                labels_batch = all_labels[total:total+B_cur]
                k = int(0.75 * 64)
                rand_flat = torch.rand(B_cur, 64, device=device)
                _, ridx = rand_flat.topk(k, 1)
                mask_r = torch.zeros(B_cur, 64, device=device)
                mask_r.scatter_(1, ridx, 1.0)
                mask_r = mask_r.view(B_cur, 8, 8).unsqueeze(1)
                recv = [S2_batch[t] * mask_r for t in range(T_STEPS)]
                Fp = model.decoder(recv, mask_r)
                preds = back(Fp).argmax(1).cpu().numpy()
                correct += (preds == labels_batch).sum()
                total += B_cur
        random_accs.append(correct / total * 100)
        if (draw+1) % 10 == 0:
            print(f"    Draw {draw+1}/50: {random_accs[-1]:.2f}%")

    ra = np.array(random_accs)
    pct = (ra < learned_acc).sum() / len(ra) * 100
    print(f"\n  Random: {ra.mean():.2f} ± {ra.std():.2f}% [{ra.min():.2f}, {ra.max():.2f}]")
    print(f"  Learned ({learned_acc:.2f}%) exceeds {pct:.0f}% of random draws")

    # ============================================================
    # CHECK 3: Per-block ablation correlation (50 images)
    # ============================================================
    print("\n=== Per-block Ablation (50 images) ===")
    N_SAMPLE = 50
    np.random.seed(42)
    sample_idx = np.random.choice(N, N_SAMPLE, replace=False)
    block_damage = np.zeros((N_SAMPLE, 64))
    block_scores_sample = np.zeros((N_SAMPLE, 64))

    with torch.no_grad():
        for si, img_idx in enumerate(sample_idx):
            bi = img_idx // 16
            local_idx = img_idx % 16
            S2_batch = [s.to(device) for s in all_S2[bi]]
            if local_idx >= S2_batch[0].shape[0]:
                continue
            S2_single = [s[local_idx:local_idx+1] for s in S2_batch]

            full_mask = torch.ones(1, 1, 8, 8, device=device)
            recv_full = [S2_single[t] * full_mask for t in range(T_STEPS)]
            conf_full = F.softmax(back(model.decoder(recv_full, full_mask)), 1).max(1)[0].item()
            block_scores_sample[si] = all_scores[img_idx]

            for b in range(64):
                r, c = b // 8, b % 8
                mask_ab = torch.ones(1, 1, 8, 8, device=device)
                mask_ab[0, 0, r, c] = 0.0
                recv_ab = [S2_single[t] * mask_ab for t in range(T_STEPS)]
                conf_ab = F.softmax(back(model.decoder(recv_ab, mask_ab)), 1).max(1)[0].item()
                block_damage[si, b] = conf_full - conf_ab

            if (si+1) % 10 == 0:
                print(f"  Ablated {si+1}/{N_SAMPLE}")

    scores_flat = block_scores_sample.flatten()
    damage_flat = block_damage.flatten()
    corr, p_val = scipy_stats.pearsonr(scores_flat, damage_flat)
    sp_corr, sp_p = scipy_stats.spearmanr(scores_flat, damage_flat)

    print(f"\n  Pearson r  = {corr:.4f} (p={p_val:.2e})")
    print(f"  Spearman ρ = {sp_corr:.4f} (p={sp_p:.2e})")

    # Quartile analysis
    quartiles = np.percentile(scores_flat, [25, 50, 75])
    q_bins = np.digitize(scores_flat, quartiles)
    for qi, ql in enumerate(['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']):
        mask_q = q_bins == qi
        if mask_q.sum() > 0:
            print(f"    {ql}: damage={damage_flat[mask_q].mean():.6f} "
                  f"(score={scores_flat[mask_q].mean():.4f}, n={mask_q.sum()})")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "="*60)
    print("COMPARISON: Original vs Enhanced Scorer")
    print("="*60)
    print(f"{'Metric':<25} {'Original':<15} {'Enhanced (v3)':<15}")
    print(f"{'-'*55}")
    print(f"{'Accuracy':<25} {'96.10%':<15} {f'{learned_acc:.2f}%':<15}")
    print(f"{'Gini':<25} {'0.019':<15} {f'{np.mean(per_image_gini):.4f}':<15}")
    print(f"{'Per-image variance':<25} {'0.001':<15} {f'{per_image_var.mean():.4f}':<15}")
    print(f"{'Kept-dropped margin':<25} {'0.055':<15} {f'{margins.mean():.4f}':<15}")
    print(f"{'Ablation Pearson r':<25} {'0.011':<15} {f'{corr:.4f}':<15}")
    print(f"{'Random pct rank':<25} {'98%':<15} {f'{pct:.0f}%':<15}")
    print(f"{'Random mean±std':<25} {'95.72±0.16':<15} {f'{ra.mean():.2f}±{ra.std():.2f}':<15}")
