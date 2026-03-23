"""Unified training script for SpikeAdapt-SC on UCM (UC Merced Land Use).

Supports all stages:
  backbone: ResNet50 fine-tuning on UCM (21 classes)
  v2: IF baseline with learned spatial masking
  v4a: SNN-native cherry-pick (LIF + BNTT + learnable slope + spike reg + STDP)
  v5c: MPBN (membrane potential batch norm, no KD)

Reuses architecture code from AID training scripts with UCM-specific dataset.

Usage:
  python train/train_ucm.py --stage backbone --epochs 50
  python train/train_ucm.py --stage v2 --epochs_s2 60 --epochs_s3 40
  python train/train_ucm.py --stage v4a --epochs_s3 40
  python train/train_ucm.py --stage v5c --epochs_s3 40
"""

import os, sys, argparse, random, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ############################################################################
# UCM DATASET
# ############################################################################

class UCMDataset(Dataset):
    """UC Merced Land Use dataset — 21 classes × 100 images = 2100 total."""
    def __init__(self, root, transform=None, split='train', train_ratio=0.8, seed=42):
        self.transform = transform; self.samples = []
        ucm_dir = os.path.join(root, 'UCMerced_LandUse', 'Images')
        if not os.path.exists(ucm_dir):
            raise FileNotFoundError(f"UCM not found at {ucm_dir}")
        classes = sorted([d for d in os.listdir(ucm_dir) if os.path.isdir(os.path.join(ucm_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        all_samples = []
        for cls in classes:
            cd = os.path.join(ucm_dir, cls)
            for f in sorted(os.listdir(cd)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
                    all_samples.append((os.path.join(cd, f), self.class_to_idx[cls]))
        rng = random.Random(seed); rng.shuffle(all_samples)
        n = int(len(all_samples) * train_ratio)
        self.samples = all_samples[:n] if split == 'train' else all_samples[n:]
        print(f"  UCM {split}: {len(self.samples)} images, {len(classes)} classes")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, l = self.samples[idx]; img = Image.open(p).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, l


# ############################################################################
# IMPORTS FROM EXISTING SCRIPTS
# ############################################################################

from train_aid_v2 import (
    ResNet50Front, ResNet50Back, BSC_Channel,
    ChannelConditionedScorer, LearnedBlockMask, sample_noise
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', type=str, required=True, choices=['backbone', 'v2', 'v4a', 'v5c'])
    p.add_argument('--epochs', type=int, default=50, help='Epochs for backbone')
    p.add_argument('--epochs_s2', type=int, default=60)
    p.add_argument('--epochs_s3', type=int, default=40)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_classes', type=int, default=21)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--batch_size', type=int, default=32)
    return p.parse_args()


def get_ucm_loaders(seed=42, batch_size=32):
    tf_train = T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
                           T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))])
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])
    train_ds = UCMDataset("./data", tf_train, split='train', seed=seed)
    test_ds = UCMDataset("./data", tf_test, split='test', seed=seed)
    train_loader = DataLoader(train_ds, batch_size, True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size, False, num_workers=4, pin_memory=True)
    return train_loader, test_loader


# ############################################################################
# STAGE: BACKBONE FINE-TUNING
# ############################################################################

def train_backbone(args):
    print(f"\n{'='*60}")
    print(f"STAGE 1: ResNet50 backbone fine-tuning on UCM")
    print(f"Classes: {args.num_classes}, Epochs: {args.epochs}")
    print(f"{'='*60}")

    train_loader, test_loader = get_ucm_loaders(args.seed, args.batch_size)

    model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(2048, args.num_classes)
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss()

    snap_dir = f"./snapshots_ucm_seed{args.seed}/"
    os.makedirs(snap_dir, exist_ok=True)
    best_acc = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward(); optimizer.step()
        scheduler.step()

        if epoch % 5 == 0 or epoch == args.epochs:
            model.eval(); correct, total = 0, 0
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    correct += model(imgs).argmax(1).eq(labels).sum().item()
                    total += labels.size(0)
            acc = 100. * correct / total
            print(f"  S1 E{epoch}: {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), os.path.join(snap_dir, "backbone_best.pth"))
                print(f"  ✓ Best: {best_acc:.2f}%")

    print(f"\nBackbone done. Best: {best_acc:.2f}%")
    return best_acc


# ############################################################################
# STAGE: V2 (IF baseline)
# ############################################################################

def train_v2(args):
    from train_aid_v2 import SpikeAdaptSC_v2

    print(f"\n{'='*60}")
    print(f"STAGE 2+3: V2 baseline on UCM")
    print(f"{'='*60}")

    train_loader, test_loader = get_ucm_loaders(args.seed, args.batch_size)

    snap_dir = f"./snapshots_ucm_seed{args.seed}/"
    front = ResNet50Front(grid_size=14).to(device)
    back = ResNet50Back(args.num_classes).to(device)

    # Load backbone
    bb = torch.load(os.path.join(snap_dir, "backbone_best.pth"), map_location=device)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.'))}, strict=False)
    back_state = {k: v for k, v in bb.items() if k.startswith(('layer4.', 'avgpool.'))}
    back_state['fc.weight'] = bb['fc.weight']; back_state['fc.bias'] = bb['fc.bias']
    back.load_state_dict(back_state, strict=False)

    model = SpikeAdaptSC_v2(C_in=1024, C1=256, C2=36, T=8, target_rate=0.75, grid_size=14).to(device)

    # S2: encoder/decoder training (front frozen)
    front.eval()
    for p in front.parameters(): p.requires_grad = False

    optimizer = optim.Adam(list(model.parameters()), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"\n--- S2 Training ({args.epochs_s2} epochs) ---")
    best_acc = 0
    for epoch in range(1, args.epochs_s2 + 1):
        model.train(); back.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            ber = sample_noise('bsc')
            Fp, stats = model(feat, noise_param=ber)
            loss = criterion(back(Fp), labels) + 0.1 * stats.get('rate_penalty', 0)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        if epoch % 5 == 0 or epoch == args.epochs_s2:
            model.eval(); back.eval(); correct, total = 0, 0
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    correct += back(model(front(imgs), 0.0)[0]).argmax(1).eq(labels).sum().item()
                    total += labels.size(0)
            acc = 100. * correct / total
            print(f"  S2 E{epoch}: {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                torch.save({'model': model.state_dict(), 'back': back.state_dict()},
                           os.path.join(snap_dir, f"v2_s2_{acc:.2f}.pth"))
                print(f"  ✓ Best: {best_acc:.2f}%")

    # S3: joint fine-tuning
    print(f"\n--- S3 Training ({args.epochs_s3} epochs) ---")
    optimizer = optim.Adam(list(model.parameters()) + list(back.parameters()), lr=1e-5)

    for epoch in range(1, args.epochs_s3 + 1):
        model.train(); back.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            ber = sample_noise('bsc')
            Fp, stats = model(feat, noise_param=ber)
            loss = criterion(back(Fp), labels) + 0.1 * stats.get('rate_penalty', 0)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        if epoch % 5 == 0 or epoch == args.epochs_s3:
            model.eval(); back.eval(); correct, total = 0, 0
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    correct += back(model(front(imgs), 0.0)[0]).argmax(1).eq(labels).sum().item()
                    total += labels.size(0)
            acc = 100. * correct / total
            print(f"  S3 E{epoch}: {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                torch.save({'model': model.state_dict(), 'back': back.state_dict()},
                           os.path.join(snap_dir, f"v2_s3_{acc:.2f}.pth"))
                print(f"  ✓ Best: {best_acc:.2f}%")

    # Final BER evaluation
    print(f"\nFINAL EVALUATION — V2 on UCM")
    model.eval(); back.eval()
    for ber in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                correct += back(model(front(imgs), ber)[0]).argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        label = "Clean" if ber == 0 else f"BER={ber:.2f}"
        print(f"  {label}: {100.*correct/total:.2f}%")

    return best_acc


# ############################################################################
# STAGE: V4-A (SNN-native cherry-pick)
# ############################################################################

def train_v4a(args):
    from train_aid_v4 import SpikeAdaptSC_v4

    print(f"\n{'='*60}")
    print(f"V4-A: SNN-native cherry-pick on UCM")
    print(f"{'='*60}")

    train_loader, test_loader = get_ucm_loaders(args.seed, args.batch_size)

    snap_dir = "./snapshots_ucm_seed42/"  # Always use seed=42 backbone
    v4_snap = f"./snapshots_ucm_v4a_seed{args.seed}/"
    os.makedirs(v4_snap, exist_ok=True)

    front = ResNet50Front(grid_size=14).to(device)
    back = ResNet50Back(args.num_classes).to(device)

    # Load backbone
    bb = torch.load(os.path.join(snap_dir, "backbone_best.pth"), map_location=device)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.'))}, strict=False)
    front.eval()
    for p in front.parameters(): p.requires_grad = False

    model = SpikeAdaptSC_v4(C_in=1024, C1=256, C2=36, T=8,
                             target_rate=0.75, grid_size=14,
                             use_mpbn=False, use_channel_gate=False).to(device)

    # Transfer V2 weights if available
    v4_files = sorted([f for f in os.listdir(v4_snap) if f.startswith("v4a_")]) if os.path.exists(v4_snap) else []
    if not v4_files:
        # Try V2 weights from seed=42 as fallback
        v2_files = sorted([f for f in os.listdir(snap_dir) if f.startswith("v2_s3_")]) if os.path.exists(snap_dir) else []
        if v2_files:
            ck = torch.load(os.path.join(snap_dir, v2_files[-1]), map_location=device)
            transferred = 0
            m_state = model.state_dict()
            for k, v in ck['model'].items():
                if k in m_state and m_state[k].shape == v.shape:
                    m_state[k] = v; transferred += 1
            model.load_state_dict(m_state, strict=False)
            back.load_state_dict(ck['back'])
            print(f"  ✓ Transferred {transferred} params from V2 checkpoint")
    else:
        ck = torch.load(os.path.join(v4_snap, v4_files[-1]), map_location=device)
        transferred = 0
        m_state = model.state_dict()
        for k, v in ck['model'].items():
            if k in m_state and m_state[k].shape == v.shape:
                m_state[k] = v; transferred += 1
        model.load_state_dict(m_state, strict=False)
        back.load_state_dict(ck['back'])
        print(f"  ✓ Transferred {transferred} params from V4-A best checkpoint")

    # Param groups with 10× lr for slope
    slope_params = [p for n, p in model.named_parameters() if 'slope' in n]
    other_params = [p for n, p in model.named_parameters() if 'slope' not in n]
    optimizer = optim.Adam([
        {'params': other_params, 'lr': 1e-5},
        {'params': slope_params, 'lr': 1e-4},
        {'params': back.parameters(), 'lr': 1e-5}
    ])
    criterion = nn.CrossEntropyLoss()
    best_acc = 0

    for epoch in range(1, args.epochs_s3 + 1):
        model.train(); back.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            ber = sample_noise('bsc')
            Fp, stats = model(feat, noise_param=ber)
            loss = criterion(back(Fp), labels) + 0.1 * stats.get('rate_penalty', 0)
            if 'spike_reg' in stats: loss = loss + 0.01 * stats['spike_reg']
            if 'stdp_loss' in stats: loss = loss + 0.005 * stats['stdp_loss']
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        if epoch % 5 == 0 or epoch == args.epochs_s3:
            model.eval(); back.eval(); correct, total = 0, 0
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    correct += back(model(front(imgs), 0.0)[0]).argmax(1).eq(labels).sum().item()
                    total += labels.size(0)
            acc = 100. * correct / total
            fr = stats.get('firing_rate', 0)
            beta = stats.get('beta_mean', 0)
            slope = stats.get('slope_mean', 0)
            print(f"  S3 E{epoch}: {acc:.2f}%, FR={fr:.3f}, β={beta:.3f}, slope={slope:.1f}")
            if acc > best_acc:
                best_acc = acc
                torch.save({'model': model.state_dict(), 'back': back.state_dict()},
                           os.path.join(v4_snap, f"v4a_{acc:.2f}.pth"))
                print(f"  ✓ Best: {best_acc:.2f}%")

    # Final BER evaluation
    print(f"\nFINAL EVALUATION — V4-A on UCM [seed={args.seed}]")
    results = {}
    model.eval(); back.eval()
    for ber in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                correct += back(model(front(imgs), ber)[0]).argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        acc = 100.*correct/total
        label = "Clean" if ber == 0 else f"BER={ber:.2f}"
        print(f"  {label}: {acc:.2f}%")
        results[str(ber)] = acc

    with open(os.path.join(v4_snap, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    return best_acc


# ############################################################################
# STAGE: V5-C (MPBN)
# ############################################################################

def train_v5c(args):
    from train_aid_v5 import SpikeAdaptSC_v5

    print(f"\n{'='*60}")
    print(f"V5-C: MPBN on UCM")
    print(f"{'='*60}")

    train_loader, test_loader = get_ucm_loaders(args.seed, args.batch_size)

    snap_dir = "./snapshots_ucm_seed42/"  # Always use seed=42 backbone
    v4_snap = f"./snapshots_ucm_v4a_seed42/"  # Transfer from seed=42 V4-A
    v5_snap = f"./snapshots_ucm_v5c_seed{args.seed}/"
    os.makedirs(v5_snap, exist_ok=True)

    front = ResNet50Front(grid_size=14).to(device)
    back = ResNet50Back(args.num_classes).to(device)

    # Load backbone
    bb = torch.load(os.path.join(snap_dir, "backbone_best.pth"), map_location=device)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.'))}, strict=False)
    front.eval()
    for p in front.parameters(): p.requires_grad = False

    model = SpikeAdaptSC_v5(C_in=1024, C1=256, C2=36, T=8,
                             target_rate=0.75, grid_size=14,
                             use_mpbn=True, use_channel_gate=False,
                             use_spike_attn=False).to(device)

    # Transfer V4-A weights if available
    v4_files = sorted([f for f in os.listdir(v4_snap) if f.startswith("v4a_")]) if os.path.exists(v4_snap) else []
    if v4_files:
        ck = torch.load(os.path.join(v4_snap, v4_files[-1]), map_location=device)
        transferred = 0
        m_state = model.state_dict()
        for k, v in ck['model'].items():
            if k in m_state and m_state[k].shape == v.shape:
                m_state[k] = v; transferred += 1
        model.load_state_dict(m_state, strict=False)
        back.load_state_dict(ck['back'])
        print(f"  ✓ Transferred {transferred} params from V4-A checkpoint")

    slope_params = [p for n, p in model.named_parameters() if 'slope' in n]
    other_params = [p for n, p in model.named_parameters() if 'slope' not in n]
    optimizer = optim.Adam([
        {'params': other_params, 'lr': 1e-5},
        {'params': slope_params, 'lr': 1e-4},
        {'params': back.parameters(), 'lr': 1e-5}
    ])
    criterion = nn.CrossEntropyLoss()
    best_acc = 0

    for epoch in range(1, args.epochs_s3 + 1):
        model.train(); back.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            ber = sample_noise('bsc')
            Fp, stats = model(feat, noise_param=ber)
            loss = criterion(back(Fp), labels) + 0.1 * stats.get('rate_penalty', 0)
            if 'spike_reg' in stats: loss = loss + 0.01 * stats['spike_reg']
            if 'stdp_loss' in stats: loss = loss + 0.005 * stats['stdp_loss']
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        if epoch % 5 == 0 or epoch == args.epochs_s3:
            model.eval(); back.eval(); correct, total = 0, 0
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    correct += back(model(front(imgs), 0.0)[0]).argmax(1).eq(labels).sum().item()
                    total += labels.size(0)
            acc = 100. * correct / total
            fr = stats.get('firing_rate', 0)
            print(f"  S3 E{epoch}: {acc:.2f}%, FR={fr:.3f}")
            if acc > best_acc:
                best_acc = acc
                torch.save({'model': model.state_dict(), 'back': back.state_dict()},
                           os.path.join(v5_snap, f"v5c_{acc:.2f}.pth"))
                print(f"  ✓ Best: {best_acc:.2f}%")

    # Final BER evaluation
    print(f"\nFINAL EVALUATION — V5C on UCM [seed={args.seed}]")
    results = {}
    model.eval(); back.eval()
    for ber in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                correct += back(model(front(imgs), ber)[0]).argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        acc = 100.*correct/total
        label = "Clean" if ber == 0 else f"BER={ber:.2f}"
        print(f"  {label}: {acc:.2f}%")
        results[str(ber)] = acc

    with open(os.path.join(v5_snap, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    return best_acc


# ############################################################################
# MAIN
# ############################################################################

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    print(f"Device: {device}")
    print(f"Stage: {args.stage}, Seed: {args.seed}, Classes: {args.num_classes}")

    if args.stage == 'backbone':
        train_backbone(args)
    elif args.stage == 'v2':
        train_v2(args)
    elif args.stage == 'v4a':
        train_v4a(args)
    elif args.stage == 'v5c':
        train_v5c(args)
