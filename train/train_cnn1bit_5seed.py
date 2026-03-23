#!/usr/bin/env python3
"""5-seed CNN-1bit baseline: BinaryCNN_SC (STE sign, T=1, C2=36).

Uses the EXACT same model as train_1bit_baseline.py, which produced the
paper's CNN-1bit results (AID: 95.32%/87.56%, R45: 91.48%/85.19% at seed-42).

Trains for 5 seeds on both datasets using the SAME backbone per seed.
Output: eval/seed_results/cnn1bit_5seed.json
"""

import torch, torch.nn as nn, torch.optim as optim
import sys, os, json, random, numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms as T

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from run_final_pipeline import AIDDataset5050, RESISC45Dataset
from train_aid_v2 import ResNet50Front, ResNet50Back
# Import the exact CNN-1bit model class from the original script
from train_1bit_baseline import BinaryCNN_SC

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 123, 456, 789, 1024]


def train_and_eval_seed(dataset_name, n_classes, seed):
    """Train BinaryCNN_SC for one seed, then evaluate at BER=0.0 and BER=0.30."""
    print(f"\n{'='*60}")
    print(f"  CNN-1bit (BinaryCNN_SC) {dataset_name.upper()} seed={seed}")
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

    # Load backbone (same one used for SNN)
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

    # BinaryCNN_SC: exactly matching the original train_1bit_baseline.py
    model = BinaryCNN_SC(C_in=1024, C1=256, C2=36).to(device)

    snap_dir = f'./snapshots_{dataset_name}_1bit_seed{seed}/'
    os.makedirs(snap_dir, exist_ok=True)

    # Check if already trained
    existing = sorted([f for f in os.listdir(snap_dir) if f.startswith('1bit_best_')],
                      key=lambda x: float(x.split('_')[-1].replace('.pth', ''))) if os.path.exists(snap_dir) else []

    if not existing:
        # Train 60 epochs (same as original)
        optimizer = optim.Adam(list(model.parameters()) + list(back.parameters()), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0
        print(f"  Training (60 epochs)...", flush=True)
        for epoch in range(1, 61):
            model.train(); back.train()
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                feat = front(imgs)
                ber = random.choice([0.0, 0.0, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
                Fp, _ = model(feat, ber=ber)
                loss = criterion(back(Fp), labels)
                optimizer.zero_grad(); loss.backward(); optimizer.step()

            if epoch % 10 == 0 or epoch == 60:
                model.eval(); back.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for imgs, labels in test_loader:
                        imgs, labels = imgs.to(device), labels.to(device)
                        Fp, _ = model(front(imgs), ber=0.0)
                        correct += back(Fp).argmax(1).eq(labels).sum().item()
                        total += labels.size(0)
                acc = 100. * correct / total
                print(f"  E{epoch:02d}: {acc:.2f}%", flush=True)

                if acc > best_acc:
                    best_acc = acc
                    torch.save({'model': model.state_dict(), 'back': back.state_dict()},
                               os.path.join(snap_dir, f'1bit_best_{acc:.2f}.pth'))

        existing = sorted([f for f in os.listdir(snap_dir) if f.startswith('1bit_best_')],
                          key=lambda x: float(x.split('_')[-1].replace('.pth', '')))
    else:
        print(f"  Found existing checkpoint: {existing[-1]}", flush=True)

    # Load best checkpoint
    best_ck = existing[-1]
    ck = torch.load(os.path.join(snap_dir, best_ck), map_location=device, weights_only=False)
    model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
    model.eval(); back.eval()

    # Evaluate at BER=0.0 and BER=0.30
    results = {}
    for ber in [0.0, 0.30]:
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                Fp, _ = model(front(imgs), ber=ber)
                correct += back(Fp).argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        acc = round(100. * correct / total, 2)
        results[str(ber)] = acc
        print(f"  BER={ber:.2f}: {acc}%", flush=True)

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
    from scipy import stats as scipy_stats

    print(f"\n{'='*60}")
    print(f"  CNN-1BIT 5-SEED SUMMARY")
    print(f"{'='*60}", flush=True)

    summary = {}
    for ds_name in all_results:
        summary[ds_name] = {}
        for ber in ['0.0', '0.3']:
            accs = [all_results[ds_name][str(s)][ber] for s in seeds if str(s) in all_results[ds_name]]
            if len(accs) >= 2:
                mean_acc = np.mean(accs)
                std_acc = np.std(accs, ddof=1)
                summary[ds_name][f'ber_{ber}'] = {
                    'mean': round(float(mean_acc), 2),
                    'std': round(float(std_acc), 2),
                    'per_seed': accs
                }
                print(f"  {ds_name} BER={ber}: {mean_acc:.2f} ± {std_acc:.2f}", flush=True)

    # Save
    with open('eval/seed_results/cnn1bit_5seed.json', 'w') as f:
        json.dump({'per_seed': all_results, 'summary': summary}, f, indent=2)

    print(f"\n✅ Results saved to eval/seed_results/cnn1bit_5seed.json")


if __name__ == '__main__':
    main()
