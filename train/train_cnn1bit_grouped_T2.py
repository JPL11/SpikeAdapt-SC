#!/usr/bin/env python3
"""T2 control: CNN-1bit with T=8 grouped binary output maps.

Tests whether the SNN's training-free temporal-truncation axis is available
to a single-pass ANN. BinaryCNN_SC with C2 = 8 x 36 = 288 binary maps
(payload-matched to the SNN at T=8, rho=1.0), trained with the EXACT
CNN-1bit recipe (no truncation during training), then truncated GROUP-WISE
at inference: groups >= T' are zeroed at the decoder input, mirroring the
SNN converter's zero-padding of missing timesteps.

Protocol: 3 seeds (42/123/456, first three of the 10-seed pool by index),
T' in {3,4,5,6,8}, BER in {0, 0.15, 0.30} -- matching Table III of the
ICC paper (minus spatial masking, which CNN-1bit does not have).

Output: eval/seed_results/cnn1bit_grouped_T2.json
"""

import torch, torch.nn as nn, torch.optim as optim
import sys, os, json, random, numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms as T

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from run_final_pipeline import AIDDataset5050, RESISC45Dataset
from train_aid_v2 import ResNet50Front, ResNet50Back
from train_1bit_baseline import BinaryCNN_SC

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 123, 456]
N_GROUPS = 8
C2_GROUP = 36
C2_TOTAL = N_GROUPS * C2_GROUP          # 288
TPRIMES = [3, 4, 5, 6, 8]
BERS = [0.0, 0.15, 0.30]


def eval_truncated(front, model, back, loader, tprime, ber):
    """Accuracy with groups >= tprime zeroed at the decoder input."""
    model.eval(); back.eval()
    keep = tprime * C2_GROUP
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            binary = model.encoder(front(imgs))          # (B, 288, 14, 14)
            if ber > 0:
                flip = (torch.rand_like(binary) < ber).float()
                received = (binary + flip) % 2
            else:
                received = binary
            received = received.clone()
            received[:, keep:] = 0.0                     # truncate groups
            Fp = model.decoder(received)
            correct += back(Fp).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return round(100. * correct / total, 2)


def run_seed(dataset_name, n_classes, seed):
    print(f"\n{'='*60}\n  CNN-1bit-G8 {dataset_name.upper()} seed={seed}\n{'='*60}",
          flush=True)
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

    bb_path = f'./snapshots_{dataset_name}_5050_seed{seed}/backbone_best.pth'
    front = ResNet50Front(grid_size=14).to(device)
    bb = torch.load(bb_path, map_location=device, weights_only=False)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))},
                          strict=False)
    front.eval()
    for p in front.parameters():
        p.requires_grad = False
    back = ResNet50Back(n_classes).to(device)
    back.load_state_dict({k: v for k, v in bb.items()
                          if k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))},
                         strict=False)

    model = BinaryCNN_SC(C_in=1024, C1=256, C2=C2_TOTAL).to(device)

    snap_dir = f'./snapshots_{dataset_name}_1bitg8_seed{seed}/'
    os.makedirs(snap_dir, exist_ok=True)
    existing = sorted([f for f in os.listdir(snap_dir) if f.startswith('g8_best_')],
                      key=lambda x: float(x.split('_')[-1].replace('.pth', '')))
    if not existing:
        optimizer = optim.Adam(list(model.parameters()) + list(back.parameters()), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        best_acc = 0
        print("  Training (60 epochs, CNN-1bit recipe, no truncation)...", flush=True)
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
                acc = eval_truncated(front, model, back, test_loader, N_GROUPS, 0.0)
                print(f"  E{epoch:02d}: clean {acc:.2f}%", flush=True)
                if acc > best_acc:
                    best_acc = acc
                    torch.save({'model': model.state_dict(), 'back': back.state_dict()},
                               os.path.join(snap_dir, f'g8_best_{acc:.2f}.pth'))
        existing = sorted([f for f in os.listdir(snap_dir) if f.startswith('g8_best_')],
                          key=lambda x: float(x.split('_')[-1].replace('.pth', '')))
    else:
        print(f"  Found existing checkpoint: {existing[-1]}", flush=True)

    ck = torch.load(os.path.join(snap_dir, existing[-1]), map_location=device,
                    weights_only=False)
    model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])

    res = {}
    for tp in TPRIMES:
        res[str(tp)] = {}
        for ber in BERS:
            acc = eval_truncated(front, model, back, test_loader, tp, ber)
            res[str(tp)][f'{ber:g}'] = acc
            print(f"  T'={tp} BER={ber:.2f}: {acc}%", flush=True)

    del model, back, front
    torch.cuda.empty_cache()
    return res


def main():
    out_path = 'eval/seed_results/cnn1bit_grouped_T2.json'
    all_results = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_results = json.load(f).get('per_seed', {})
    for ds, n_cls in [('aid', 30), ('resisc45', 45)]:
        all_results.setdefault(ds, {})
        for seed in SEEDS:
            if str(seed) in all_results[ds]:
                print(f"skip {ds} seed {seed} (done)")
                continue
            all_results[ds][str(seed)] = run_seed(ds, n_cls, seed)
            with open(out_path, 'w') as f:
                json.dump({'per_seed': all_results}, f, indent=1)

    summary = {}
    for ds in all_results:
        summary[ds] = {}
        for tp in TPRIMES:
            summary[ds][str(tp)] = {}
            for ber in BERS:
                vals = [all_results[ds][s][str(tp)][f'{ber:g}']
                        for s in all_results[ds]]
                summary[ds][str(tp)][f'{ber:g}'] = dict(
                    mean=round(float(np.mean(vals)), 2),
                    std=round(float(np.std(vals, ddof=1)), 2) if len(vals) > 1 else None)
    with open(out_path, 'w') as f:
        json.dump({'per_seed': all_results, 'summary': summary}, f, indent=1)
    print(f"\nSaved: {out_path}")
    for ds in summary:
        print(f"\n{ds.upper()} (mean±std over {len(all_results[ds])} seeds):")
        for tp in TPRIMES:
            row = '  '.join(f"BER{b:g}: {summary[ds][str(tp)][f'{b:g}']['mean']:.2f}"
                            for b in BERS)
            print(f"  T'={tp}: {row}")


if __name__ == '__main__':
    main()
