"""Multi-exit temporal training for SpikeAdapt-SC v2.

Instead of ACT-style halting (which fails for near-saturated accuracy),
this trains the v2 decoder to produce good outputs at MULTIPLE exit
points (T=2, 4, 6, 8). At inference, the system uses classifier
confidence to select the earliest sufficient exit.

Loss = Σ_t w_t * CE(back(decoder(recv[:t], mask)), label)
     + λ_rate * (tx_rate - target)^2

Where w_t is a per-exit weight (earlier exits get stronger weight to
encourage good representations at lower T).

This is more like BranchyNet/early-exit than Adaptive Computation Time.
"""

import os, sys, random, json, math, argparse
import numpy as np
from tqdm import tqdm

os.environ['PYTHONUNBUFFERED'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, './train')
from train_aid_v2 import (
    AIDDataset, ResNet50Front, ResNet50Back,
    SpikeAdaptSC_v2, sample_noise, evaluate
)


EXIT_POINTS = [2, 4, 6, 8]  # Timesteps at which we evaluate
EXIT_WEIGHTS = [2.0, 1.5, 1.0, 1.0]  # Higher weight for earlier exits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--target_rate', type=float, default=0.75)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    SNAP_DIR = f"./snapshots_aid_v3_seed{args.seed}/"
    os.makedirs(SNAP_DIR, exist_ok=True)
    print(f"Device: {device} | Seed: {args.seed}")
    print(f"Exit points: {EXIT_POINTS}")
    print(f"Exit weights: {EXIT_WEIGHTS}")

    # Dataset
    train_tf = T.Compose([
        T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
        T.RandomRotation(15), T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))
    ])
    test_tf = T.Compose([
        T.Resize(256), T.CenterCrop(224),
        T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))
    ])
    train_ds = AIDDataset("./data", transform=train_tf, split='train', seed=42)
    test_ds = AIDDataset("./data", transform=test_tf, split='test', seed=42)
    train_loader = DataLoader(train_ds, 32, shuffle=True, num_workers=4,
                               pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, 64, shuffle=False, num_workers=4,
                              pin_memory=True)

    # Backbone
    front = ResNet50Front(grid_size=14).to(device)
    back = ResNet50Back(30).to(device)
    bb_path = "./snapshots_aid/backbone_best.pth"
    state = torch.load(bb_path, map_location=device)
    front.load_state_dict({k: v for k, v in state.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))},
                          strict=False)
    front.eval()
    for p in front.parameters(): p.requires_grad = False

    # Load v2 model as starting point
    model = SpikeAdaptSC_v2(C_in=1024, C1=256, C2=36, T=8,
                             target_rate=args.target_rate, channel_type='bsc',
                             grid_size=14).to(device)
    v2_path = "./snapshots_aid_v2_seed42/"
    s3f = sorted([f for f in os.listdir(v2_path) if f.startswith("v2_s3_")])
    if s3f:
        ck = torch.load(os.path.join(v2_path, s3f[-1]), map_location=device)
        model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
        print(f"✓ Loaded v2: {s3f[-1]}")

    criterion = nn.CrossEntropyLoss()

    # ============================================================
    # MULTI-EXIT TRAINING
    # ============================================================
    print(f"\n{'='*60}")
    print(f"MULTI-EXIT FINE-TUNING ({args.epochs} epochs)")
    print(f"  Training decoder to produce good outputs at T={EXIT_POINTS}")
    print(f"{'='*60}")

    # Only train decoder + back (encoder + scorer + mask are frozen from v2)
    for p in model.encoder.parameters(): p.requires_grad = False
    for p in model.scorer.parameters(): p.requires_grad = False
    params = list(model.decoder.parameters()) + list(back.parameters())
    opt = optim.Adam(params, lr=5e-5, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, 1e-6)

    best_score = 0.0
    for ep in range(args.epochs):
        model.train(); back.train()
        # Keep encoder frozen
        model.encoder.eval()
        model.scorer.eval()

        pbar = tqdm(train_loader, desc=f"ME E{ep+1}/{args.epochs}")
        opt.zero_grad()
        for step, (img, lab) in enumerate(pbar):
            img, lab = img.to(device), lab.to(device)
            noise = sample_noise('bsc')
            with torch.no_grad():
                feat = front(img)
                # Encode all timesteps
                all_S2, m1, m2 = [], None, None
                for t in range(model.T):
                    _, s2, m1, m2 = model.encoder(feat, m1, m2)
                    all_S2.append(s2)
                # Score and mask
                importance = model.scorer(all_S2, noise)
                mask, tx = model.block_mask(importance, training=False)
                # Channel
                recv = [model.channel(model.block_mask.apply_mask(all_S2[t], mask), noise)
                        for t in range(model.T)]

            # Multi-exit loss: decode at each exit point
            total_loss = 0.0
            for exit_t, exit_w in zip(EXIT_POINTS, EXIT_WEIGHTS):
                Fp = model.decoder(recv[:exit_t], mask, T_actual=exit_t)
                ce = criterion(back(Fp), lab)
                total_loss = total_loss + exit_w * ce

            rate_loss = 2.0 * (tx.item() - args.target_rate) ** 2
            loss = (total_loss / sum(EXIT_WEIGHTS) + rate_loss) / 2
            loss.backward()

            if (step + 1) % 2 == 0:
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad()

            pbar.set_postfix({'L': f'{loss.item():.3f}', 'tx': f'{tx.item():.2f}'})
        sched.step()

        if (ep + 1) % 5 == 0:
            model.eval(); back.eval()
            print(f"\n  --- Epoch {ep+1} Multi-Exit Results ---")

            for exit_t in EXIT_POINTS:
                correct, total = 0, 0
                with torch.no_grad():
                    for imgs, labs in test_loader:
                        imgs, labs = imgs.to(device), labs.to(device)
                        feat = front(imgs)
                        all_S2, m1, m2 = [], None, None
                        for t in range(model.T):
                            _, s2, m1, m2 = model.encoder(feat, m1, m2)
                            all_S2.append(s2)
                        importance = model.scorer(all_S2, 0.0)
                        mask, _ = model.block_mask(importance, training=False)
                        recv = [model.channel(model.block_mask.apply_mask(all_S2[t], mask), 0.0)
                                for t in range(model.T)]
                        Fp = model.decoder(recv[:exit_t], mask, T_actual=exit_t)
                        correct += back(Fp).argmax(1).eq(labs).sum().item()
                        total += labs.size(0)
                acc = 100. * correct / total
                ts = (1 - exit_t / 8) * 100
                print(f"    T={exit_t}: {acc:.2f}% (temporal savings: {ts:.0f}%)")

            # Confidence-based early exit evaluation
            correct_ee, total_ee, T_sum, n_b = 0, 0, 0, 0
            conf_threshold = 0.95
            with torch.no_grad():
                for imgs, labs in test_loader:
                    imgs, labs = imgs.to(device), labs.to(device)
                    feat = front(imgs)
                    all_S2, m1, m2 = [], None, None
                    for t in range(model.T):
                        _, s2, m1, m2 = model.encoder(feat, m1, m2)
                        all_S2.append(s2)
                    importance = model.scorer(all_S2, 0.0)
                    mask, _ = model.block_mask(importance, training=False)
                    recv = [model.channel(model.block_mask.apply_mask(all_S2[t], mask), 0.0)
                            for t in range(model.T)]

                    T_used = 8
                    for exit_t in EXIT_POINTS:
                        Fp = model.decoder(recv[:exit_t], mask, T_actual=exit_t)
                        logits = back(Fp)
                        conf = F.softmax(logits, 1).max(1)[0].mean().item()
                        if conf >= conf_threshold:
                            T_used = exit_t
                            break

                    Fp = model.decoder(recv[:T_used], mask, T_actual=T_used)
                    preds = back(Fp).argmax(1)
                    correct_ee += preds.eq(labs).sum().item()
                    total_ee += labs.size(0)
                    T_sum += T_used; n_b += 1

            acc_ee = 100. * correct_ee / total_ee
            avg_T_ee = T_sum / n_b
            ts_ee = (1 - avg_T_ee / 8) * 100
            score = acc_ee * (1 + ts_ee / 200)  # Reward temporal savings
            print(f"    Early-Exit (conf≥{conf_threshold}): {acc_ee:.2f}%, "
                  f"avg T={avg_T_ee:.1f}, saving={ts_ee:.0f}%")

            if score > best_score:
                best_score = score
                torch.save({'model': model.state_dict(), 'back': back.state_dict()},
                           os.path.join(SNAP_DIR, f"v3_me_{acc_ee:.2f}_T{avg_T_ee:.1f}.pth"))
                print(f"    ✓ Best score: {best_score:.1f}")

    # ============================================================
    # FINAL EVALUATION
    # ============================================================
    best_f = sorted([f for f in os.listdir(SNAP_DIR) if f.startswith("v3_me_")])
    if best_f:
        ck = torch.load(os.path.join(SNAP_DIR, best_f[-1]), map_location=device)
        model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])

    model.eval(); back.eval()
    print(f"\n{'='*60}")
    print("FINAL EVALUATION — SpikeAdapt-SC v3 (Multi-Exit)")
    print(f"{'='*60}")

    # Per-exit accuracy at multiple BERs
    for ber in [0.0, 0.15, 0.30]:
        print(f"\n  BER={ber:.2f}:")
        for exit_t in EXIT_POINTS:
            correct, total = 0, 0
            with torch.no_grad():
                for imgs, labs in test_loader:
                    imgs, labs = imgs.to(device), labs.to(device)
                    feat = front(imgs)
                    all_S2, m1, m2 = [], None, None
                    for t in range(model.T):
                        _, s2, m1, m2 = model.encoder(feat, m1, m2)
                        all_S2.append(s2)
                    importance = model.scorer(all_S2, ber)
                    mask, _ = model.block_mask(importance, training=False)
                    recv = [model.channel(model.block_mask.apply_mask(all_S2[t], mask), ber)
                            for t in range(model.T)]
                    Fp = model.decoder(recv[:exit_t], mask, T_actual=exit_t)
                    correct += back(Fp).argmax(1).eq(labs).sum().item()
                    total += labs.size(0)
            acc = 100. * correct / total
            print(f"    T={exit_t}: {acc:.2f}%")

    # Confidence-based early exit at multiple BERs
    print(f"\n  Confidence-based early exit (threshold=0.95):")
    for ber in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        correct_ee, total_ee, T_sum, n_b = 0, 0, 0, 0
        with torch.no_grad():
            for imgs, labs in test_loader:
                imgs, labs = imgs.to(device), labs.to(device)
                feat = front(imgs)
                all_S2, m1, m2 = [], None, None
                for t in range(model.T):
                    _, s2, m1, m2 = model.encoder(feat, m1, m2)
                    all_S2.append(s2)
                importance = model.scorer(all_S2, ber)
                mask, _ = model.block_mask(importance, training=False)
                recv = [model.channel(model.block_mask.apply_mask(all_S2[t], mask), ber)
                        for t in range(model.T)]
                T_used = 8
                for exit_t in EXIT_POINTS:
                    Fp = model.decoder(recv[:exit_t], mask, T_actual=exit_t)
                    logits = back(Fp)
                    conf = F.softmax(logits, 1).max(1)[0].mean().item()
                    if conf >= 0.95:
                        T_used = exit_t; break
                Fp = model.decoder(recv[:T_used], mask, T_actual=T_used)
                correct_ee += back(Fp).argmax(1).eq(labs).sum().item()
                total_ee += labs.size(0); T_sum += T_used; n_b += 1
        acc_ee = 100. * correct_ee / total_ee
        avg_T = T_sum / n_b
        ts = (1 - avg_T / 8) * 100
        print(f"    BER={ber:.2f}: {acc_ee:.2f}%, avg T={avg_T:.1f}, saving={ts:.0f}%")

    results = {
        'model': 'SpikeAdapt-SC_v3_multi_exit',
        'seed': args.seed,
        'exit_points': EXIT_POINTS,
    }
    with open(os.path.join(SNAP_DIR, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved to {SNAP_DIR}")
