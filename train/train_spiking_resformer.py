#!/usr/bin/env python3
"""Train fully-spiking SpikeAdapt-SC with SpikingResformer backbone.

Replaces ResNet-50 ANN frontend with SpikingResformer (CVPR 2024).
Three-stage training, same structure as run_final_pipeline.py:
    S1: Train SpikingResformer backbone on target dataset
    S2: Train SNN encoder/decoder (freeze backbone)
    S3: Train noise-aware scorer + diversity loss

Usage:
    python train/train_spiking_resformer.py --stage s1 --dataset aid
    python train/train_spiking_resformer.py --stage s2 --dataset aid
    python train/train_spiking_resformer.py --stage s3 --dataset aid
    python train/train_spiking_resformer.py --stage all --dataset aid
    python train/train_spiking_resformer.py --stage eval --dataset aid
"""

import os, sys, argparse, random, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from run_final_pipeline import AIDDataset5050, RESISC45Dataset
from train_aid_v2 import BSC_Channel, LearnedBlockMask, sample_noise
from train_aid_v5 import EncoderV5, DecoderV5
from noise_aware_scorer import NoiseAwareScorer
from models.spiking_resformer_backbone import (
    SpikingResformerFront, SpikingResformerBack, SpikeAdaptSC_FullSpiking,
    count_params
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_STEPS = 8
BER_SWEEP = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

DATASET_CONFIGS = {
    'aid': {
        'n_classes': 30,
        'ds_cls': AIDDataset5050,
        'ds_kwargs': dict(seed=42),
    },
    'resisc45': {
        'n_classes': 45,
        'ds_cls': RESISC45Dataset,
        'ds_kwargs': dict(train_ratio=0.20, seed=42),
    },
}


def get_snap_dir(dataset, variant, seed):
    return f'./snapshots_{dataset}_spikingresformer_{variant}_seed{seed}'


# ======================================================================
# Stage 1: Train SpikingResformer backbone (classification only)
# ======================================================================

def train_s1_backbone(dataset, variant, n_classes, train_loader, test_loader,
                       seed, epochs=80, lr=0.01):
    """Train SpikingResformer backbone on target dataset."""
    snap_dir = get_snap_dir(dataset, variant, seed)
    os.makedirs(snap_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  S1: SpikingResformer-{variant.upper()} backbone ({epochs} epochs)")
    print(f"{'='*60}")

    front = SpikingResformerFront(variant=variant, T=4).to(device)
    back = SpikingResformerBack(in_channels=1024, n_classes=n_classes).to(device)

    total, trainable = count_params(front)
    print(f"  Frontend: {total/1e6:.1f}M params")
    total_b, _ = count_params(back)
    print(f"  Backend: {total_b/1e6:.1f}M params")

    all_params = list(front.parameters()) + list(back.parameters())
    optimizer = optim.AdamW(all_params, lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0

    for epoch in range(1, epochs + 1):
        front.train(); back.train()
        ep_loss, ep_correct, ep_total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            logits = back(feat)
            loss = criterion(logits, labels)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            ep_loss += loss.item()
            ep_correct += logits.argmax(1).eq(labels).sum().item()
            ep_total += labels.size(0)
        scheduler.step()

        if epoch % 10 == 0 or epoch == epochs:
            front.eval(); back.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    correct += back(front(imgs)).argmax(1).eq(labels).sum().item()
                    total += labels.size(0)
            acc = 100. * correct / total
            train_acc = 100. * ep_correct / ep_total
            print(f"  E{epoch:03d}: train={train_acc:.2f}%, test={acc:.2f}%, "
                  f"loss={ep_loss/len(train_loader):.4f}")
            if acc > best_acc:
                best_acc = acc
                torch.save({
                    'front': front.state_dict(),
                    'back': back.state_dict(),
                    'epoch': epoch, 'acc': best_acc,
                }, os.path.join(snap_dir, f's1_best_{best_acc:.2f}.pth'))
                print(f"  -> Best: {best_acc:.2f}%")

    print(f"\n  S1 complete. Best: {best_acc:.2f}%")
    return best_acc


# ======================================================================
# Stage 2: Train SNN encoder/decoder (freeze backbone)
# ======================================================================

def train_s2_codec(dataset, variant, n_classes, train_loader, test_loader,
                    seed, epochs=60, lr=1e-4):
    """Train SNN encoder/decoder with frozen SpikingResformer backbone."""
    snap_dir = get_snap_dir(dataset, variant, seed)

    print(f"\n{'='*60}")
    print(f"  S2: SNN encoder/decoder ({epochs} epochs)")
    print(f"{'='*60}")

    # Load S1 backbone
    import glob
    s1_cks = sorted(glob.glob(os.path.join(snap_dir, 's1_best_*.pth')),
                    key=lambda x: float(x.split('_')[-1].replace('.pth', '')))
    if not s1_cks:
        raise FileNotFoundError(f"No S1 checkpoint in {snap_dir}")
    s1_ck = torch.load(s1_cks[-1], map_location=device, weights_only=False)
    print(f"  Loaded S1: {s1_cks[-1]} (acc={s1_ck['acc']:.2f}%)")

    front = SpikingResformerFront(variant=variant, T=4).to(device)
    front.load_state_dict(s1_ck['front'])
    front.eval()
    for p in front.parameters():
        p.requires_grad = False

    back = SpikingResformerBack(in_channels=1024, n_classes=n_classes).to(device)
    back.load_state_dict(s1_ck['back'])

    # Create SNN codec
    encoder = EncoderV5(1024, 256, 36, T_STEPS, use_mpbn=True).to(device)
    decoder = DecoderV5(1024, 256, 36, T_STEPS, use_mpbn=True).to(device)
    channel = BSC_Channel()

    codec_params = list(encoder.parameters()) + list(decoder.parameters()) + list(back.parameters())
    optimizer = optim.Adam(codec_params, lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0

    for epoch in range(1, epochs + 1):
        encoder.train(); decoder.train(); back.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            ber = sample_noise('bsc')

            # Encode
            all_S2, m1, m2 = [], None, None
            for t in range(T_STEPS):
                _, s2, m1, m2 = encoder(feat, m1, m2, t=t)
                all_S2.append(s2)

            # Channel (no masking in S2)
            mask = torch.ones(imgs.size(0), 1, 14, 14, device=device)
            recv = [channel(all_S2[t], ber) for t in range(T_STEPS)]
            Fp = decoder(recv, mask)

            loss = criterion(back(Fp), labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()

        if epoch % 10 == 0 or epoch == epochs:
            encoder.eval(); decoder.eval(); back.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    feat = front(imgs)
                    all_S2, m1, m2 = [], None, None
                    for t in range(T_STEPS):
                        _, s2, m1, m2 = encoder(feat, m1, m2, t=t)
                        all_S2.append(s2)
                    mask = torch.ones(imgs.size(0), 1, 14, 14, device=device)
                    recv = [all_S2[t] for t in range(T_STEPS)]
                    Fp = decoder(recv, mask)
                    correct += back(Fp).argmax(1).eq(labels).sum().item()
                    total += labels.size(0)
            acc = 100. * correct / total
            print(f"  S2 E{epoch:03d}: test={acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                torch.save({
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'back': back.state_dict(),
                    'epoch': epoch, 'acc': best_acc,
                }, os.path.join(snap_dir, f's2_best_{best_acc:.2f}.pth'))

    print(f"\n  S2 complete. Best: {best_acc:.2f}%")
    return best_acc


# ======================================================================
# Stage 3: Train noise-aware scorer + diversity loss
# ======================================================================

def train_s3_scorer(dataset, variant, n_classes, train_loader, test_loader,
                     seed, epochs=40, lr=1e-5):
    """Train noise-aware scorer with frozen backbone, fine-tune codec."""
    snap_dir = get_snap_dir(dataset, variant, seed)

    print(f"\n{'='*60}")
    print(f"  S3: Noise-aware scorer ({epochs} epochs)")
    print(f"{'='*60}")

    import glob

    # Load backbone
    s1_cks = sorted(glob.glob(os.path.join(snap_dir, 's1_best_*.pth')),
                    key=lambda x: float(x.split('_')[-1].replace('.pth', '')))
    s1_ck = torch.load(s1_cks[-1], map_location=device, weights_only=False)
    front = SpikingResformerFront(variant=variant, T=4).to(device)
    front.load_state_dict(s1_ck['front'])
    front.eval()
    for p in front.parameters():
        p.requires_grad = False

    # Load codec
    s2_cks = sorted(glob.glob(os.path.join(snap_dir, 's2_best_*.pth')),
                    key=lambda x: float(x.split('_')[-1].replace('.pth', '')))
    s2_ck = torch.load(s2_cks[-1], map_location=device, weights_only=False)

    encoder = EncoderV5(1024, 256, 36, T_STEPS, use_mpbn=True).to(device)
    decoder = DecoderV5(1024, 256, 36, T_STEPS, use_mpbn=True).to(device)
    scorer = NoiseAwareScorer(C_spike=36, hidden=32).to(device)
    block_mask = LearnedBlockMask(0.75, 0.5).to(device)
    channel = BSC_Channel()
    back = SpikingResformerBack(in_channels=1024, n_classes=n_classes).to(device)

    encoder.load_state_dict(s2_ck['encoder'])
    decoder.load_state_dict(s2_ck['decoder'])
    back.load_state_dict(s2_ck['back'])
    print(f"  Loaded S2: {s2_cks[-1]}")

    # Fine-tune: scorer + encoder + decoder + back
    ft_params = (list(scorer.parameters()) + list(encoder.parameters()) +
                 list(decoder.parameters()) + list(back.parameters()))
    optimizer = optim.Adam(ft_params, lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0

    for epoch in range(1, epochs + 1):
        encoder.train(); decoder.train(); scorer.train(); back.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            ber = sample_noise('bsc')

            # Encode
            all_S2, m1, m2 = [], None, None
            for t in range(T_STEPS):
                _, s2, m1, m2 = encoder(feat, m1, m2, t=t)
                all_S2.append(s2)

            # Score & mask
            importance = scorer(all_S2, ber).squeeze(1)
            mask, tx = block_mask(importance, training=True)

            # Channel + decode
            recv = [channel(all_S2[t] * mask, ber) for t in range(T_STEPS)]
            Fp = decoder(recv, mask)

            loss = criterion(back(Fp), labels)

            # Rate penalty
            rate_loss = (importance.mean() - 0.75) ** 2
            loss = loss + 10.0 * rate_loss

            # Diversity loss
            div = scorer.compute_diversity_loss(all_S2, 0.0, 0.30)
            loss = loss + 0.05 * div

            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(ft_params, 1.0)
            optimizer.step()
        scheduler.step()

        if epoch % 5 == 0 or epoch == epochs:
            encoder.eval(); decoder.eval(); scorer.eval(); back.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    feat = front(imgs)
                    all_S2, m1, m2 = [], None, None
                    for t in range(T_STEPS):
                        _, s2, m1, m2 = encoder(feat, m1, m2, t=t)
                        all_S2.append(s2)
                    importance = scorer(all_S2, 0.0).squeeze(1)
                    mask, tx = block_mask(importance, training=False)
                    recv = [all_S2[t] * mask for t in range(T_STEPS)]
                    Fp = decoder(recv, mask)
                    correct += back(Fp).argmax(1).eq(labels).sum().item()
                    total += labels.size(0)
            acc = 100. * correct / total
            print(f"  S3 E{epoch:03d}: test={acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                torch.save({
                    'front': front.state_dict(),
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'scorer': scorer.state_dict(),
                    'back': back.state_dict(),
                    'block_mask': block_mask.state_dict(),
                    'epoch': epoch, 'acc': best_acc,
                }, os.path.join(snap_dir, f's3_best_{best_acc:.2f}.pth'))

    # Final BER sweep evaluation
    print(f"\n  Final BER sweep ({dataset.upper()}):")
    for ber in BER_SWEEP:
        encoder.eval(); decoder.eval(); scorer.eval(); back.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                feat = front(imgs)
                all_S2, m1, m2 = [], None, None
                for t in range(T_STEPS):
                    _, s2, m1, m2 = encoder(feat, m1, m2, t=t)
                    all_S2.append(s2)
                importance = scorer(all_S2, ber).squeeze(1)
                mask, tx = block_mask(importance, training=False)
                recv = [channel(all_S2[t] * mask, ber) for t in range(T_STEPS)]
                Fp = decoder(recv, mask)
                correct += back(Fp).argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        acc = 100. * correct / total
        label = "Clean" if ber == 0 else f"BER={ber:.2f}"
        print(f"    {label}: {acc:.2f}%")

    print(f"\n  S3 complete. Best: {best_acc:.2f}%")
    return best_acc


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True,
                        choices=['s1', 's2', 's3', 'all', 'eval', 'test_arch'])
    parser.add_argument('--dataset', default='aid', choices=['aid', 'resisc45'])
    parser.add_argument('--variant', default='ti', choices=['ti', 's', 'm'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs_s1', type=int, default=80)
    parser.add_argument('--epochs_s2', type=int, default=60)
    parser.add_argument('--epochs_s3', type=int, default=40)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}, Variant: SpikingResformer-{args.variant.upper()}")

    if args.stage == 'test_arch':
        # Quick architecture test (no data needed)
        print("\nArchitecture test...")
        from models.spiking_resformer_backbone import SpikingResformerFront
        front = SpikingResformerFront(variant=args.variant, T=4)
        total, _ = count_params(front)
        print(f"SpikingResformer-{args.variant.upper()}: {total/1e6:.1f}M params")
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            y = front(x)
        print(f"Input: {x.shape} → Output: {y.shape}")
        print("Architecture test passed!")
        return

    cfg = DATASET_CONFIGS[args.dataset]

    tf_train = T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
                           T.ColorJitter(0.2, 0.2, 0.2),
                           T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))])
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])

    train_ds = cfg['ds_cls']("./data", tf_train, 'train', **cfg['ds_kwargs'])
    test_ds = cfg['ds_cls']("./data", tf_test, 'test', **cfg['ds_kwargs'])
    train_loader = DataLoader(train_ds, 32, True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)

    stages = ['s1', 's2', 's3'] if args.stage == 'all' else [args.stage]

    for stage in stages:
        if stage == 's1':
            train_s1_backbone(args.dataset, args.variant, cfg['n_classes'],
                              train_loader, test_loader, args.seed, args.epochs_s1)
        elif stage == 's2':
            train_s2_codec(args.dataset, args.variant, cfg['n_classes'],
                           train_loader, test_loader, args.seed, args.epochs_s2)
        elif stage == 's3':
            train_s3_scorer(args.dataset, args.variant, cfg['n_classes'],
                            train_loader, test_loader, args.seed, args.epochs_s3)


if __name__ == '__main__':
    main()
