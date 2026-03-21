"""50-draw random mask comparison for Table V.

Evaluates random masks at ρ={0.50, 0.75} × BER={0.0, 0.30} with 50 independent
random draws each. Reports mean ± std for proper statistical comparison.
Also evaluates uniform (deterministic) and learned masks.
"""

import os, sys, json, time
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from train_aid_v2 import ResNet50Front, ResNet50Back, BSC_Channel
from train_aid_v5 import EncoderV5, DecoderV5
from noise_aware_scorer import NoiseAwareScorer
from run_final_pipeline import AIDDataset5050, SpikeAdaptSC_v5c_NA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_STEPS = 8


def evaluate_with_mask(front, model, back, loader, mask_type, rho, ber, rng=None):
    """Evaluate with a specific mask type.
    
    mask_type: 'learned', 'random', 'uniform'
    """
    correct, total = 0, 0
    k = int(rho * 14 * 14)  # number of blocks to keep
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            B = imgs.size(0)
            feat = front(imgs)
            
            # Encode
            all_S2, m1, m2 = [], None, None
            for t in range(model.T):
                _, s2, m1, m2 = model.encoder(feat, m1, m2, t=t)
                all_S2.append(s2)
            
            if mask_type == 'learned':
                importance = model.scorer(all_S2, ber).squeeze(1)
                mask, _ = model.block_mask(importance, training=False)
            elif mask_type == 'random':
                # Random mask: sample k positions per image independently
                mask = torch.zeros(B, 1, 14, 14, device=device)
                for b in range(B):
                    perm = rng.permutation(196)[:k]
                    rows, cols = perm // 14, perm % 14
                    mask[b, 0, rows, cols] = 1.0
            elif mask_type == 'uniform':
                # Uniform spacing: deterministic grid
                mask = torch.zeros(B, 1, 14, 14, device=device)
                # Select k blocks in a fixed pattern
                flat_idx = np.linspace(0, 195, k, dtype=int)
                rows, cols = flat_idx // 14, flat_idx % 14
                mask[:, 0, rows, cols] = 1.0
            
            # Apply channel
            bsc = BSC_Channel()
            recv = []
            for t in range(model.T):
                masked = all_S2[t] * mask
                if ber > 0:
                    noise = (torch.rand_like(masked) < ber).float()
                    noisy = ((masked + noise) % 2)
                    recv.append(noisy)
                else:
                    recv.append(masked)
            
            Fp = model.decoder(recv, mask)
            correct += back(Fp).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    
    return 100. * correct / total


def main():
    print(f"Device: {device}")
    
    # Load model
    front = ResNet50Front(grid_size=14).to(device)
    bb = torch.load("./snapshots_aid_5050_seed42/backbone_best.pth", map_location=device)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))}, strict=False)
    front.eval()
    
    back = ResNet50Back(30).to(device)
    model = SpikeAdaptSC_v5c_NA(C_in=1024, C1=256, C2=36, T=T_STEPS,
                                 target_rate=0.75, grid_size=14).to(device)
    
    ck_files = sorted([f for f in os.listdir("./snapshots_aid_v5cna_seed42") if f.startswith('v5cna_best')])
    ck = torch.load(f"./snapshots_aid_v5cna_seed42/{ck_files[-1]}", map_location=device)
    model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
    model.eval(); back.eval()
    
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])
    test_ds = AIDDataset5050(root="./data", transform=tf_test, split='test', seed=42)
    loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)
    print(f"  AID test: {len(test_ds)} images")
    
    N_DRAWS = 50
    results = {}
    
    for rho in [0.50, 0.75]:
        for ber in [0.0, 0.30]:
            key = f"rho={rho}_ber={ber}"
            print(f"\n=== {key} ===")
            
            # Learned mask (deterministic)
            learned_acc = evaluate_with_mask(front, model, back, loader, 'learned', rho, ber)
            print(f"  Learned: {learned_acc:.2f}%")
            
            # Uniform mask (deterministic)
            uniform_acc = evaluate_with_mask(front, model, back, loader, 'uniform', rho, ber)
            print(f"  Uniform: {uniform_acc:.2f}%")
            
            # Random masks (50 draws)
            random_accs = []
            for draw in range(N_DRAWS):
                rng = np.random.RandomState(draw)
                acc = evaluate_with_mask(front, model, back, loader, 'random', rho, ber, rng)
                random_accs.append(acc)
                if (draw + 1) % 10 == 0:
                    print(f"  Random draw {draw+1}/{N_DRAWS}: {acc:.2f}% (running mean={np.mean(random_accs):.2f}±{np.std(random_accs):.2f}%)")
            
            results[key] = {
                'learned': learned_acc,
                'uniform': uniform_acc,
                'random_mean': float(np.mean(random_accs)),
                'random_std': float(np.std(random_accs)),
                'random_all': [float(x) for x in random_accs],
                'delta_rand': learned_acc - float(np.mean(random_accs)),
            }
            print(f"  Random (50 draws): {np.mean(random_accs):.2f} ± {np.std(random_accs):.2f}%")
            print(f"  Δ_rand = {learned_acc - np.mean(random_accs):+.2f} pp")
    
    # Save
    with open("eval/mask_compare_50draws.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print paper table
    print(f"\n{'='*60}")
    print("TABLE V — Mask Comparison (50 random draws)")
    print(f"{'='*60}")
    print(f"{'ρ':>4s} {'BER':>5s}  {'Learned':>8s}  {'Random (50)':>14s}  {'Uniform':>8s}  {'Δ_rand':>8s}")
    for rho in [0.50, 0.75]:
        for ber in [0.0, 0.30]:
            r = results[f"rho={rho}_ber={ber}"]
            print(f"{rho:4.2f} {ber:5.2f}  {r['learned']:7.2f}%  "
                  f"{r['random_mean']:6.2f}±{r['random_std']:.2f}%  "
                  f"{r['uniform']:7.2f}%  {r['delta_rand']:+7.2f}")
    
    print(f"\n✅ 50-draw mask comparison complete. Results: eval/mask_compare_50draws.json")


if __name__ == '__main__':
    main()
