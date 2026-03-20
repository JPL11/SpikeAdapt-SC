"""Master pipeline for paper-ready experiments.

Runs the full training pipeline for the FINAL model (V5C with noise-aware scorer)
on AID (50/50 split) and RESISC45 (20/80 split), with 5-seed variance.

Usage:
    python train/run_final_pipeline.py --stage backbone_aid
    python train/run_final_pipeline.py --stage v5c_aid
    python train/run_final_pipeline.py --stage backbone_resisc
    python train/run_final_pipeline.py --stage v5c_resisc
    python train/run_final_pipeline.py --stage all

Each stage:
    1. Backbone: finetune ResNet50 on target dataset
    2. V2: train base SNN encoder/decoder
    3. V5C-NA: train V5C with noise-aware scorer + MPBN + diversity loss
"""

import os, sys, argparse, random, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from train_aid_v2 import (
    ResNet50Front, ResNet50Back, BSC_Channel,
    LearnedBlockMask, sample_noise
)
from train_aid_v5 import (
    SpikeFunction_Learnable, LIFNeuron, BNTT, MPBN,
    EncoderV5, DecoderV5
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
from noise_aware_scorer import NoiseAwareScorer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_STEPS = 8


# ############################################################################
# DATASETS
# ############################################################################

class RESISC45Dataset(torch.utils.data.Dataset):
    """NWPU-RESISC45: 31,500 images, 45 classes, 256×256."""
    
    def __init__(self, root, transform=None, split='train', train_ratio=0.20, seed=42):
        self.transform = transform; self.samples = []
        resisc_dir = os.path.join(root, 'NWPU-RESISC45')
        if not os.path.exists(resisc_dir):
            raise FileNotFoundError(f"RESISC45 not found at {resisc_dir}")
        classes = sorted([d for d in os.listdir(resisc_dir) 
                         if os.path.isdir(os.path.join(resisc_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        all_samples = []
        for cls in classes:
            cd = os.path.join(resisc_dir, cls)
            for f in sorted(os.listdir(cd)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
                    all_samples.append((os.path.join(cd, f), self.class_to_idx[cls]))
        rng = random.Random(seed); rng.shuffle(all_samples)
        n = int(len(all_samples) * train_ratio)
        self.samples = all_samples[:n] if split == 'train' else all_samples[n:]
        print(f"  RESISC45 {split}: {len(self.samples)} images, {len(classes)} classes")
    
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, label


class AIDDataset5050(torch.utils.data.Dataset):
    """AID with standard 50/50 benchmark split."""
    
    def __init__(self, root, transform=None, split='train', seed=42):
        self.transform = transform; self.samples = []
        aid_dir = os.path.join(root, 'AID')
        classes = sorted([d for d in os.listdir(aid_dir)
                         if os.path.isdir(os.path.join(aid_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        all_samples = []
        for cls in classes:
            cd = os.path.join(aid_dir, cls)
            for f in sorted(os.listdir(cd)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
                    all_samples.append((os.path.join(cd, f), self.class_to_idx[cls]))
        rng = random.Random(seed); rng.shuffle(all_samples)
        n = int(len(all_samples) * 0.50)  # Standard 50/50 split
        self.samples = all_samples[:n] if split == 'train' else all_samples[n:]
        print(f"  AID-50/50 {split}: {len(self.samples)} images, {len(classes)} classes")
    
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, label


# ############################################################################
# V5C-NA: V5C + Noise-Aware Scorer
# ############################################################################

class SpikeAdaptSC_v5c_NA(nn.Module):
    """V5C with noise-aware scorer (final model for paper)."""
    
    def __init__(self, C_in=1024, C1=256, C2=36, T=8,
                 target_rate=0.75, grid_size=14):
        super().__init__()
        self.T = T; self.C2 = C2; self.grid_size = grid_size
        
        self.encoder = EncoderV5(C_in, C1, C2, T, use_mpbn=True)
        self.scorer = NoiseAwareScorer(C_spike=C2, hidden=32)
        self.block_mask = LearnedBlockMask(target_rate, 0.5)
        self.decoder = DecoderV5(C_in, C1, C2, T, use_mpbn=True)
        self.channel = BSC_Channel()
    
    def forward(self, feat, noise_param=0.0, target_rate_override=None):
        all_S2, m1, m2 = [], None, None
        for t in range(self.T):
            _, s2, m1, m2 = self.encoder(feat, m1, m2, t=t)
            all_S2.append(s2)
        
        importance = self.scorer(all_S2, noise_param).squeeze(1)  # (B,1,H,W) → (B,H,W)
        
        if target_rate_override is not None:
            old = self.block_mask.target_rate
            self.block_mask.target_rate = target_rate_override
            mask, tx = self.block_mask(importance, training=False)
            self.block_mask.target_rate = old
        else:
            mask, tx = self.block_mask(importance, training=self.training)
        
        recv = [self.channel(all_S2[t] * mask, noise_param) for t in range(self.T)]
        Fp = self.decoder(recv, mask)
        
        with torch.no_grad():
            fr = torch.stack(all_S2).mean().item()
        
        return Fp, {
            'tx_rate': tx.item(), 'mask': mask, 'importance': importance,
            'firing_rate': fr, 'all_S2': all_S2,
        }


# ############################################################################
# TRAINING FUNCTIONS
# ############################################################################

def train_backbone(dataset_name, n_classes, train_loader, test_loader, seed, epochs=50):
    """Train ResNet50 backbone on target dataset."""
    snap_dir = f"./snapshots_{dataset_name}_5050_seed{seed}/"
    os.makedirs(snap_dir, exist_ok=True)
    
    import torchvision.models as models
    model = models.resnet50(weights='DEFAULT').to(device)
    model.fc = nn.Linear(2048, n_classes).to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            loss = criterion(model(imgs), labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == epochs:
            model.eval(); correct, total = 0, 0
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    correct += model(imgs).argmax(1).eq(labels).sum().item()
                    total += labels.size(0)
            acc = 100. * correct / total
            print(f"  Backbone E{epoch}: {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), os.path.join(snap_dir, "backbone_best.pth"))
                print(f"  ✓ Best: {best_acc:.2f}%")
    
    return best_acc


def train_v5c_na(dataset_name, n_classes, train_loader, test_loader, seed, 
                  bb_path, epochs_s2=60, epochs_s3=40):
    """Train V5C with noise-aware scorer."""
    snap_dir = f"./snapshots_{dataset_name}_v5cna_seed{seed}/"
    os.makedirs(snap_dir, exist_ok=True)
    
    # Front/Back
    front = ResNet50Front(grid_size=14).to(device)
    bb = torch.load(bb_path, map_location=device)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))}, strict=False)
    front.eval()
    for p in front.parameters(): p.requires_grad = False
    
    back = ResNet50Back(n_classes).to(device)
    back_state = {k: v for k, v in bb.items() if k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))}
    back.load_state_dict(back_state, strict=False)
    
    model = SpikeAdaptSC_v5c_NA(C_in=1024, C1=256, C2=36, T=T_STEPS,
                                 target_rate=0.75, grid_size=14).to(device)
    
    # Stage 2: Train SNN encoder/decoder (no scorer)
    print(f"\n--- S2: SNN encoder/decoder ({epochs_s2} epochs) ---")
    optimizer = optim.Adam(list(model.encoder.parameters()) + 
                          list(model.decoder.parameters()) +
                          list(back.parameters()), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, epochs_s2 + 1):
        model.train(); back.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            ber = sample_noise('bsc')
            Fp, _ = model(feat, noise_param=ber)
            loss = criterion(back(Fp), labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        if epoch % 10 == 0:
            model.eval(); back.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    Fp, _ = model(front(imgs), noise_param=0.0)
                    correct += back(Fp).argmax(1).eq(labels).sum().item()
                    total += labels.size(0)
            print(f"  S2 E{epoch}: {100.*correct/total:.2f}%")
    
    # Stage 3: Fine-tune with NOISE-AWARE scorer + diversity loss
    print(f"\n--- S3: Noise-aware scorer ({epochs_s3} epochs) ---")
    slope_params = [p for n, p in model.named_parameters() if 'slope' in n]
    other_params = [p for n, p in model.named_parameters() if 'slope' not in n]
    optimizer = optim.Adam([
        {'params': other_params, 'lr': 1e-5},
        {'params': slope_params, 'lr': 1e-4},
        {'params': back.parameters(), 'lr': 1e-5}
    ])
    
    best_acc = 0
    for epoch in range(1, epochs_s3 + 1):
        model.train(); back.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            ber = sample_noise('bsc')
            
            Fp, stats = model(feat, noise_param=ber)
            loss = criterion(back(Fp), labels)
            loss = loss + 0.1 * stats.get('rate_penalty', 0)
            
            # Diversity loss: force masks to change with BER
            diversity_loss = model.scorer.compute_diversity_loss(
                stats['all_S2'], ber_low=0.0, ber_high=0.30
            )
            loss = loss + 0.05 * diversity_loss  # λ_div = 0.05
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        if epoch % 5 == 0 or epoch == epochs_s3:
            model.eval(); back.eval()
            correct, total, fr_sum, n_b = 0, 0, 0, 0
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    Fp, stats = model(front(imgs), noise_param=0.0)
                    correct += back(Fp).argmax(1).eq(labels).sum().item()
                    total += labels.size(0)
                    fr_sum += stats['firing_rate']; n_b += 1
            acc = 100. * correct / total
            fr = fr_sum / n_b
            
            # Measure noise awareness
            noise_stats = {}
            with torch.no_grad():
                for imgs, labels in test_loader:
                    feat = front(imgs.to(device))
                    all_S2, m1, m2 = [], None, None
                    for t in range(T_STEPS):
                        _, s2, m1, m2 = model.encoder(feat, m1, m2, t=t)
                        all_S2.append(s2)
                    noise_stats = model.scorer.get_mask_stats(all_S2)
                    break  # one batch is enough for diagnosis
            
            hamming = noise_stats.get('hamming_0_vs_0.3', 0)
            print(f"  S3 E{epoch}: {acc:.2f}%, FR={fr:.3f}, "
                  f"mask_diff(0→0.30)={hamming:.4f}")
            
            if acc > best_acc:
                best_acc = acc
                torch.save({'model': model.state_dict(), 'back': back.state_dict(),
                           'noise_stats': noise_stats},
                           os.path.join(snap_dir, f"v5cna_best_{acc:.2f}.pth"))
                print(f"  ✓ Best: {best_acc:.2f}%")
    
    # Final evaluation
    print(f"\nFINAL EVALUATION — V5C-NA on {dataset_name.upper()}")
    results = {}
    for ber in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        model.eval(); back.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                Fp, _ = model(front(imgs), noise_param=ber)
                correct += back(Fp).argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        acc = 100. * correct / total
        label = "Clean" if ber == 0 else f"BER={ber:.2f}"
        print(f"  {label}: {acc:.2f}%")
        results[str(ber)] = acc
    
    with open(os.path.join(snap_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    return results


# ############################################################################
# MAIN
# ############################################################################

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', type=str, required=True,
                   choices=['backbone_aid', 'v5c_aid', 
                            'backbone_resisc', 'v5c_resisc', 'all'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs_bb', type=int, default=50)
    p.add_argument('--epochs_s2', type=int, default=60)
    p.add_argument('--epochs_s3', type=int, default=40)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    print(f"Device: {device}")
    
    tf_train = T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
                           T.ColorJitter(0.2, 0.2, 0.2),
                           T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))])
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])
    
    stages = [args.stage] if args.stage != 'all' else [
        'backbone_aid', 'v5c_aid', 'backbone_resisc', 'v5c_resisc'
    ]
    
    for stage in stages:
        print(f"\n{'='*60}")
        print(f"STAGE: {stage}")
        print(f"{'='*60}")
        
        if stage == 'backbone_aid':
            train_ds = AIDDataset5050("./data", tf_train, 'train', seed=args.seed)
            test_ds = AIDDataset5050("./data", tf_test, 'test', seed=args.seed)
            train_loader = DataLoader(train_ds, 32, True, num_workers=4, pin_memory=True)
            test_loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)
            train_backbone('aid', 30, train_loader, test_loader, args.seed, args.epochs_bb)
        
        elif stage == 'v5c_aid':
            train_ds = AIDDataset5050("./data", tf_train, 'train', seed=args.seed)
            test_ds = AIDDataset5050("./data", tf_test, 'test', seed=args.seed)
            train_loader = DataLoader(train_ds, 32, True, num_workers=4, pin_memory=True)
            test_loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)
            bb_path = f"./snapshots_aid_5050_seed{args.seed}/backbone_best.pth"
            train_v5c_na('aid', 30, train_loader, test_loader, args.seed, bb_path,
                         args.epochs_s2, args.epochs_s3)
        
        elif stage == 'backbone_resisc':
            train_ds = RESISC45Dataset("./data", tf_train, 'train', train_ratio=0.20, seed=args.seed)
            test_ds = RESISC45Dataset("./data", tf_test, 'test', train_ratio=0.20, seed=args.seed)
            train_loader = DataLoader(train_ds, 32, True, num_workers=4, pin_memory=True)
            test_loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)
            train_backbone('resisc45', 45, train_loader, test_loader, args.seed, args.epochs_bb)
        
        elif stage == 'v5c_resisc':
            train_ds = RESISC45Dataset("./data", tf_train, 'train', train_ratio=0.20, seed=args.seed)
            test_ds = RESISC45Dataset("./data", tf_test, 'test', train_ratio=0.20, seed=args.seed)
            train_loader = DataLoader(train_ds, 32, True, num_workers=4, pin_memory=True)
            test_loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)
            bb_path = f"./snapshots_resisc45_5050_seed{args.seed}/backbone_best.pth"
            train_v5c_na('resisc45', 45, train_loader, test_loader, args.seed, bb_path,
                         args.epochs_s2, args.epochs_s3)
    
    print(f"\n✅ ALL STAGES COMPLETE")
