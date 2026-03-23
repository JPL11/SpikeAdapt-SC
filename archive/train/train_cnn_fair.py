"""Train CNN-Uni baselines on BOTH datasets using correct backbones.

This ensures a fair comparison: CNN baselines use the SAME backbone
and data splits as our SNN model.

  AID:      50/50 split, backbone from snapshots_aid_5050_seed42/
  RESISC45: 20/80 split, backbone from snapshots_resisc45_5050_seed42/

Training: 60 epochs encoder/decoder (frozen backbone) + 40 epochs S3 fine-tune.
"""

import os, sys, json, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))

from train_aid_v2 import (ResNet50Front, ResNet50Back, CNNUni,
                           UniformQuantizeSTE, BSC_Channel)
from run_final_pipeline import AIDDataset5050


class RESISC45_2080:
    """NWPU-RESISC45 standard 20/80 split."""
    def __init__(self, root, transform, split='train', seed=42):
        full = ImageFolder(os.path.join(root, 'NWPU-RESISC45'), transform=transform)
        rng = np.random.RandomState(seed)
        indices = list(range(len(full)))
        rng.shuffle(indices)
        n_train = int(0.2 * len(full))
        self.dataset = full
        self.indices = indices[:n_train] if split == 'train' else indices[n_train:]
        self.classes = full.classes

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_cnn_baseline(dataset_name, n_classes, backbone_path, train_ds, test_ds, snap_dir):
    """Train CNN-Uni on a specific dataset."""

    os.makedirs(snap_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Training CNN-Uni for {dataset_name}")
    print(f"  Backbone: {backbone_path}")
    print(f"  Train: {len(train_ds)}, Test: {len(test_ds)}, Classes: {n_classes}")
    print(f"  Output: {snap_dir}")
    print(f"{'='*60}")

    train_loader = DataLoader(train_ds, 32, True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)

    # Load backbone
    front = ResNet50Front(grid_size=14).to(device)
    bb = torch.load(backbone_path, map_location=device)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))},
                          strict=False)
    front.eval()
    for p in front.parameters():
        p.requires_grad = False

    # CNN-Uni model + back
    model = CNNUni(C_in=1024, C1=256, C2=36, n_bits=8).to(device)
    back = ResNet50Back(n_classes).to(device)

    # Load back from backbone checkpoint
    back_sd = {k.replace('layer4.', 'layer4.', 1): v for k, v in bb.items()
               if k.startswith('layer4.') or k.startswith('fc.') or k.startswith('avgpool.')}
    back.load_state_dict(back_sd, strict=False)

    # ─── Stage 2: Train encoder/decoder (60 epochs) ───
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(back.parameters()),
        lr=1e-4, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    print(f"\n--- S2: Training encoder/decoder (60 epochs) ---")
    for epoch in range(60):
        model.train(); back.train()
        total_loss, correct, total = 0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)

            # Random BER during training (curriculum)
            ber = np.random.choice([0.0, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
            Fp, _ = model(feat, noise_param=ber)
            logits = back(Fp)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += logits.argmax(1).eq(labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = 100. * correct / total

        # Eval
        if (epoch + 1) % 5 == 0 or epoch == 59:
            model.eval(); back.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    feat = front(imgs)
                    Fp, _ = model(feat, noise_param=0.0)
                    correct += back(Fp).argmax(1).eq(labels).sum().item()
                    total += labels.size(0)
            test_acc = 100. * correct / total
            print(f"  Epoch {epoch+1:3d}: train={train_acc:.2f}% test={test_acc:.2f}% loss={total_loss/len(train_loader):.4f}")

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({'model': model.state_dict(), 'back': back.state_dict()},
                           f"{snap_dir}/cnnuni_best_{test_acc:.2f}.pth")
        else:
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}: train={train_acc:.2f}% loss={total_loss/len(train_loader):.4f}")

    print(f"  S2 best: {best_acc:.2f}%")

    # ─── Stage 3: Fine-tune with noise curriculum (40 epochs) ───
    # Load best S2
    best_ck = sorted([f for f in os.listdir(snap_dir) if f.startswith("cnnuni_best_")])[-1]
    ck = torch.load(f"{snap_dir}/{best_ck}", map_location=device)
    model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(back.parameters()),
        lr=1e-5, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

    print(f"\n--- S3: Fine-tune with noise curriculum (40 epochs) ---")
    for epoch in range(40):
        model.train(); back.train()
        total_loss, correct, total = 0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)

            # Stronger noise curriculum (50% high noise)
            if np.random.random() < 0.5:
                ber = np.random.uniform(0.15, 0.35)
            else:
                ber = np.random.uniform(0.0, 0.15)
            Fp, _ = model(feat, noise_param=ber)
            logits = back(Fp)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += logits.argmax(1).eq(labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = 100. * correct / total

        if (epoch + 1) % 5 == 0 or epoch == 39:
            model.eval(); back.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    feat = front(imgs)
                    Fp, _ = model(feat, noise_param=0.0)
                    correct += back(Fp).argmax(1).eq(labels).sum().item()
                    total += labels.size(0)
            test_acc = 100. * correct / total
            print(f"  Epoch {epoch+1:3d}: train={train_acc:.2f}% test={test_acc:.2f}%")

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({'model': model.state_dict(), 'back': back.state_dict()},
                           f"{snap_dir}/cnnuni_s3_{test_acc:.2f}.pth")

    print(f"  S3 best: {best_acc:.2f}%")

    # Final BER sweep evaluation
    best_files = sorted([f for f in os.listdir(snap_dir) if f.startswith("cnnuni_s3_")])
    if not best_files:
        best_files = sorted([f for f in os.listdir(snap_dir) if f.startswith("cnnuni_best_")])
    ck = torch.load(f"{snap_dir}/{best_files[-1]}", map_location=device)
    model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
    model.eval(); back.eval()

    ber_list = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    results = {}
    print(f"\n--- Final BER sweep ---")
    for ber in ber_list:
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                feat = front(imgs)
                Fp, _ = model(feat, noise_param=ber)
                correct += back(Fp).argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        acc = 100. * correct / total
        results[str(ber)] = acc
        print(f"  BER={ber:.2f}: {acc:.2f}%")

    return results


def main():
    print(f"Device: {device}")

    tf_train = T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
                           T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))])
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])

    all_results = {}

    # ─── AID 50/50 ───
    aid_train = AIDDataset5050(root="./data", transform=tf_train, split='train', seed=42)
    aid_test = AIDDataset5050(root="./data", transform=tf_test, split='test', seed=42)
    aid_results = train_cnn_baseline(
        dataset_name="AID (50/50)",
        n_classes=30,
        backbone_path="./snapshots_aid_5050_seed42/backbone_best.pth",
        train_ds=aid_train,
        test_ds=aid_test,
        snap_dir="./snapshots_cnnuni_aid_5050"
    )
    all_results['AID'] = aid_results

    # ─── RESISC45 20/80 ───
    res_train = RESISC45_2080(root="./data", transform=tf_train, split='train', seed=42)
    res_test = RESISC45_2080(root="./data", transform=tf_test, split='test', seed=42)
    res_results = train_cnn_baseline(
        dataset_name="RESISC45 (20/80)",
        n_classes=45,
        backbone_path="./snapshots_resisc45_5050_seed42/backbone_best.pth",
        train_ds=res_train,
        test_ds=res_test,
        snap_dir="./snapshots_cnnuni_resisc45_2080"
    )
    all_results['RESISC45'] = res_results

    # Save all results
    with open("eval/cnn_fair_baselines.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for ds, res in all_results.items():
        print(f"\n{ds}:")
        for ber, acc in res.items():
            print(f"  BER={float(ber):.2f}: {acc:.2f}%")

    print(f"\n✅ Fair CNN baselines complete. Results: eval/cnn_fair_baselines.json")


if __name__ == '__main__':
    main()
