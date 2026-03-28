#!/usr/bin/env python3
"""CIFAR-100: T=4 + 14×14 training and comprehensive multi-graph evaluation.

Creates:
1. T=4 SpikeAdapt-SC model (4×4 spatial, same as T=8 but half timesteps)
2. 14×14 backbone (ResNet50 ImageNet-style, 224px input from CIFAR upscale)
3. 14×14 SpikeAdapt-SC model
4. All BER+BEC sweep figures with CNN baselines

Run: python train/train_cifar_extended.py --step [1|2|3|4|5|6]
  step 1: Train T=4 SNN on 4×4 features
  step 2: Train 14×14 backbone
  step 3: Train 14×14 SNN
  step 4: Evaluate all models (BSC + BEC sweeps)
  step 5: Generate comparison figures
"""

import os, sys, json, random, argparse
import numpy as np
from tqdm import tqdm

os.environ['PYTHONUNBUFFERED'] = '1'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.spikeadapt_sc import SpikeAdaptSC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

OUT = './paper/figures/'
os.makedirs(OUT, exist_ok=True)

# ============================================================================
# MODELS
# ============================================================================
class ResNet50Front32(nn.Module):
    """CIFAR-style (32px → 4×4 output)."""
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet50(weights=None)
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = r.bn1; self.relu = r.relu
        self.maxpool = nn.Identity()
        self.layer1 = r.layer1; self.layer2 = r.layer2
        self.layer3 = r.layer3; self.layer4 = r.layer4
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))

class ResNet50Front224(nn.Module):
    """ImageNet-style (224px → 14×14 output via layer3)."""
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet50(weights=None)
        self.conv1 = r.conv1; self.bn1 = r.bn1
        self.relu = r.relu; self.maxpool = r.maxpool
        self.layer1 = r.layer1; self.layer2 = r.layer2
        self.layer3 = r.layer3
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.layer3(self.layer2(self.layer1(x)))

class ResNet50Back(nn.Module):
    def __init__(self, n=100, c_in=2048):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c_in, n)
    def forward(self, x):
        return self.fc(torch.flatten(self.avgpool(x), 1))


# ============================================================================
# DATA
# ============================================================================
mean_c, std_c = (0.5071, .4867, .4408), (.2675, .2565, .2761)

train_tf_32 = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                         T.ToTensor(), T.Normalize(mean_c, std_c)])
test_tf_32 = T.Compose([T.ToTensor(), T.Normalize(mean_c, std_c)])

train_tf_224 = T.Compose([T.Resize(224), T.RandomCrop(224, padding=16),
                          T.RandomHorizontalFlip(), T.ToTensor(),
                          T.Normalize(mean_c, std_c)])
test_tf_224 = T.Compose([T.Resize(224), T.ToTensor(), T.Normalize(mean_c, std_c)])


# ============================================================================
# BEC Channel
# ============================================================================
class BEC_Channel:
    def __call__(self, x, erasure_rate):
        if erasure_rate <= 0: return x
        keep = (torch.rand_like(x.float()) >= erasure_rate).float()
        random_bits = (torch.rand_like(x.float()) > 0.5).float()
        return x * keep + random_bits * (1 - keep)


# ============================================================================
# TRAINING HELPERS
# ============================================================================
def train_snn(front, snn, back, train_loader, test_loader, epochs, lr, snap_dir, tag,
              diversity_w=0.1):
    """Train SNN module (freeze front, train SNN+back)."""
    print(f"\n  Training {tag}...")
    os.makedirs(snap_dir, exist_ok=True)
    
    for p in front.parameters():
        p.requires_grad = False
    optimizer = torch.optim.Adam(list(snn.parameters()) + list(back.parameters()),
                                 lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    ce = nn.CrossEntropyLoss()
    
    best_acc = 0
    for epoch in range(1, epochs + 1):
        snn.train(); back.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                feat = front(images)
            ber = random.uniform(0, 0.15)
            Fp, info = snn(feat, noise_param=ber)
            out = back(Fp)
            loss = ce(out, labels) + 0.01 * info.get('rate_loss', 0)
            if 'diversity_loss' in info:
                loss = loss + diversity_w * info['diversity_loss']
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(snn.parameters()) + list(back.parameters()), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        
        if epoch % 5 == 0 or epoch == epochs:
            snn.eval(); back.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    feat = front(images)
                    Fp, _ = snn(feat, noise_param=0.0)
                    out = back(Fp)
                    correct += out.argmax(1).eq(labels).sum().item()
                    total += labels.size(0)
            acc = 100. * correct / total
            print(f"    Epoch {epoch:3d}: loss={total_loss/len(train_loader):.4f}, "
                  f"acc={acc:.2f}%  {'★' if acc > best_acc else ''}")
            if acc > best_acc:
                best_acc = acc
                torch.save({'model': snn.state_dict(), 'back': back.state_dict(),
                           'epoch': epoch, 'acc': acc},
                          f'{snap_dir}/{tag}_best_{acc:.2f}.pth')
    
    print(f"  ✅ {tag}: best = {best_acc:.2f}%")
    return best_acc


def joint_finetune(front, snn, back, train_loader, test_loader, epochs, lr, snap_dir, tag,
                   diversity_w=0.1):
    """Joint finetune (front + SNN + back)."""
    print(f"\n  Joint finetuning {tag}...")
    
    for p in front.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(
        [{'params': front.parameters(), 'lr': lr * 0.1},
         {'params': snn.parameters(), 'lr': lr},
         {'params': back.parameters(), 'lr': lr}],
        weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    ce = nn.CrossEntropyLoss()
    
    best_acc = 0
    for epoch in range(1, epochs + 1):
        front.train(); snn.train(); back.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            feat = front(images)
            ber = random.uniform(0, 0.15)
            Fp, info = snn(feat, noise_param=ber)
            out = back(Fp)
            loss = ce(out, labels) + 0.01 * info.get('rate_loss', 0)
            if 'diversity_loss' in info:
                loss = loss + diversity_w * info['diversity_loss']
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(front.parameters()) + list(snn.parameters()) +
                                     list(back.parameters()), 1.0)
            optimizer.step()
        scheduler.step()
        
        if epoch % 5 == 0 or epoch == epochs:
            front.eval(); snn.eval(); back.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    feat = front(images)
                    Fp, _ = snn(feat, noise_param=0.0)
                    out = back(Fp)
                    correct += out.argmax(1).eq(labels).sum().item()
                    total += labels.size(0)
            acc = 100. * correct / total
            print(f"    Epoch {epoch:3d}: acc={acc:.2f}%  {'★' if acc > best_acc else ''}")
            if acc > best_acc:
                best_acc = acc
                torch.save({'model': snn.state_dict(), 'back': back.state_dict(),
                           'front': front.state_dict(), 'epoch': epoch, 'acc': acc},
                          f'{snap_dir}/{tag}_best_{acc:.2f}.pth')
    
    print(f"  ✅ {tag}: best = {best_acc:.2f}%")
    return best_acc


@torch.no_grad()
def eval_channel(front, snn, back, loader, channel_fn, noise_param, n_repeat=5):
    """Eval with arbitrary channel function."""
    nr = n_repeat if noise_param > 0 else 1
    accs = []
    for _ in range(nr):
        correct, total = 0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            feat = front(images)
            all_S2, m1, m2 = [], None, None
            for t in range(snn.T):
                _, s2, m1, m2 = snn.encoder(feat, m1, m2, t=t)
                all_S2.append(s2)
            importance = snn.scorer(all_S2, ber=0.0)
            mask, tx = snn.block_mask(importance, training=False)
            recv = [channel_fn(all_S2[t] * mask, noise_param) for t in range(snn.T)]
            Fp, _, _ = snn.decoder(recv, mask)
            out = back(Fp)
            correct += out.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
        accs.append(100. * correct / total)
    return np.mean(accs), np.std(accs)


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=0,
                        help='1=T4 train, 2=14x14 backbone, 3=14x14 SNN, 4=eval, 5=figures')
    args = parser.parse_args()
    
    # ================================================================
    # STEP 1: Train T=4 SNN on 4×4 features
    # ================================================================
    if args.step in [0, 1]:
        print("\n" + "=" * 60)
        print("  Step 1: Training T=4 SpikeAdapt-SC (4×4)")
        print("=" * 60)
        
        train_ds = torchvision.datasets.CIFAR100("./data", True, download=True,
                                                  transform=train_tf_32)
        test_ds = torchvision.datasets.CIFAR100("./data", False, download=True,
                                                 transform=test_tf_32)
        train_loader = DataLoader(train_ds, 128, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_ds, 128, shuffle=False, num_workers=4, pin_memory=True)
        
        front = ResNet50Front32().to(device)
        bb = torch.load('./snapshots_spikeadapt/backbone_best.pth',
                        map_location=device, weights_only=False)
        f_st = {k: v for k, v in bb.items() if not k.startswith(('fc.', 'avgpool.'))}
        front.load_state_dict(f_st, strict=False)
        
        snn_t4 = SpikeAdaptSC(C_in=2048, C1=256, C2=128, T=4,
                               target_rate=0.75, channel_type='bsc').to(device)
        back_t4 = ResNet50Back(100, 2048).to(device)
        back_t4.fc.load_state_dict(
            {k.replace('fc.', ''): v for k, v in bb.items() if k.startswith('fc.')})
        
        snap = 'snapshots_cifar_t4'
        os.makedirs(snap, exist_ok=True)
        
        # Phase 2: SNN training (frozen backbone)
        train_snn(front, snn_t4, back_t4, train_loader, test_loader,
                  epochs=50, lr=1e-3, snap_dir=snap, tag='s2_t4')
        
        # Phase 3: Joint finetune
        # Reload best from phase 2
        s2_files = sorted([f for f in os.listdir(snap) if f.startswith('s2_t4_best')])
        if s2_files:
            ck = torch.load(f'{snap}/{s2_files[-1]}', map_location=device, weights_only=False)
            snn_t4.load_state_dict(ck['model'])
            back_t4.load_state_dict(ck['back'])
        
        joint_finetune(front, snn_t4, back_t4, train_loader, test_loader,
                       epochs=50, lr=5e-5, snap_dir=snap, tag='s3_t4')
    
    # ================================================================
    # STEP 2: Train 14×14 backbone
    # ================================================================
    if args.step in [0, 2]:
        print("\n" + "=" * 60)
        print("  Step 2: Training 14×14 backbone (224px input)")
        print("=" * 60)
        
        train_ds = torchvision.datasets.CIFAR100("./data", True, download=True,
                                                  transform=train_tf_224)
        test_ds = torchvision.datasets.CIFAR100("./data", False, download=True,
                                                 transform=test_tf_224)
        train_loader = DataLoader(train_ds, 64, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_ds, 64, shuffle=False, num_workers=4, pin_memory=True)
        
        front14 = ResNet50Front224().to(device)
        back14 = ResNet50Back(100, 1024).to(device)
        
        # Full backbone training
        optimizer = torch.optim.SGD(list(front14.parameters()) + list(back14.parameters()),
                                    lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
        ce = nn.CrossEntropyLoss()
        
        snap = 'snapshots_cifar_14x14'
        os.makedirs(snap, exist_ok=True)
        best_acc = 0
        
        for epoch in range(1, 101):
            front14.train(); back14.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                feat = front14(images)
                out = back14(feat)
                loss = ce(out, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            if epoch % 5 == 0 or epoch == 1:
                front14.eval(); back14.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for images, labels in test_loader:
                        images, labels = images.to(device), labels.to(device)
                        out = back14(front14(images))
                        correct += out.argmax(1).eq(labels).sum().item()
                        total += labels.size(0)
                acc = 100. * correct / total
                print(f"    Epoch {epoch:3d}: acc={acc:.2f}%  "
                      f"{'★' if acc > best_acc else ''}")
                if acc > best_acc:
                    best_acc = acc
                    torch.save({**{f'front.{k}': v for k, v in front14.state_dict().items()},
                               **{f'back.{k}': v for k, v in back14.state_dict().items()},
                               'epoch': epoch, 'acc': acc},
                              f'{snap}/backbone_best_{acc:.2f}.pth')
        
        print(f"  ✅ 14×14 backbone: best = {best_acc:.2f}%")
    
    # ================================================================
    # STEP 3: Train 14×14 SNN
    # ================================================================
    if args.step in [0, 3]:
        print("\n" + "=" * 60)
        print("  Step 3: Training 14×14 SpikeAdapt-SC")
        print("=" * 60)
        
        train_ds = torchvision.datasets.CIFAR100("./data", True, download=True,
                                                  transform=train_tf_224)
        test_ds = torchvision.datasets.CIFAR100("./data", False, download=True,
                                                 transform=test_tf_224)
        train_loader = DataLoader(train_ds, 64, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_ds, 64, shuffle=False, num_workers=4, pin_memory=True)
        
        snap14 = 'snapshots_cifar_14x14'
        
        # Load backbone
        front14 = ResNet50Front224().to(device)
        back14 = ResNet50Back(100, 1024).to(device)
        bb_files = sorted([f for f in os.listdir(snap14) if f.startswith('backbone_best')])
        if not bb_files:
            print("  ⚠ No 14×14 backbone found! Run --step 2 first.")
            return
        ck = torch.load(f'{snap14}/{bb_files[-1]}', map_location=device, weights_only=False)
        front14.load_state_dict({k.replace('front.', ''): v for k, v in ck.items()
                                 if k.startswith('front.')})
        back14.load_state_dict({k.replace('back.', ''): v for k, v in ck.items()
                                if k.startswith('back.')})
        
        # 14×14 SNN: C_in=1024 (layer3 output), 14×14 spatial = 196 blocks
        snn14 = SpikeAdaptSC(C_in=1024, C1=256, C2=36, T=8,
                              target_rate=0.75, channel_type='bsc').to(device)
        
        # Phase 2: SNN training (frozen backbone)
        train_snn(front14, snn14, back14, train_loader, test_loader,
                  epochs=50, lr=1e-3, snap_dir=snap14, tag='s2_snn14')
        
        # Reload best
        s2_files = sorted([f for f in os.listdir(snap14) if f.startswith('s2_snn14_best')])
        if s2_files:
            ck = torch.load(f'{snap14}/{s2_files[-1]}', map_location=device, weights_only=False)
            snn14.load_state_dict(ck['model'])
            back14.load_state_dict(ck['back'])
        
        # Phase 3: Joint finetune
        joint_finetune(front14, snn14, back14, train_loader, test_loader,
                       epochs=50, lr=5e-5, snap_dir=snap14, tag='s3_snn14')
    
    # ================================================================
    # STEP 4: Evaluate all models (BSC + BEC)
    # ================================================================
    if args.step in [0, 4]:
        print("\n" + "=" * 60)
        print("  Step 4: Evaluating all models")
        print("=" * 60)
        
        from models.snn_modules import BSC_Channel
        bsc = BSC_Channel()
        bec = BEC_Channel()
        
        test_ds_32 = torchvision.datasets.CIFAR100("./data", False, download=True,
                                                    transform=test_tf_32)
        test_loader_32 = DataLoader(test_ds_32, 128, shuffle=False, num_workers=4,
                                     pin_memory=True)
        test_ds_224 = torchvision.datasets.CIFAR100("./data", False, download=True,
                                                     transform=test_tf_224)
        test_loader_224 = DataLoader(test_ds_224, 64, shuffle=False, num_workers=4,
                                      pin_memory=True)
        
        noise_vals = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        all_results = {}
        
        # Load T=8 (4×4) model
        front32 = ResNet50Front32().to(device)
        bb = torch.load('./snapshots_spikeadapt/backbone_best.pth',
                        map_location=device, weights_only=False)
        front32.load_state_dict({k: v for k, v in bb.items()
                                 if not k.startswith(('fc.', 'avgpool.'))}, strict=False)
        front32.eval()
        
        snn_t8 = SpikeAdaptSC(C_in=2048, C1=256, C2=128, T=8,
                               target_rate=0.75, channel_type='bsc').to(device)
        back_t8 = ResNet50Back(100, 2048).to(device)
        s3_files = sorted([f for f in os.listdir('snapshots_cifar_v5cna/')
                           if f.startswith('s3_best')])
        ck = torch.load(f'snapshots_cifar_v5cna/{s3_files[-1]}',
                        map_location=device, weights_only=False)
        snn_t8.load_state_dict(ck['model'])
        back_t8.load_state_dict(ck['back'])
        snn_t8.eval(); back_t8.eval()
        
        # T=8 BSC sweep
        print("\n  T=8 (4×4) BSC sweep:")
        t8_bsc = []
        for nv in noise_vals:
            m, s = eval_channel(front32, snn_t8, back_t8, test_loader_32, bsc, nv)
            t8_bsc.append({'rate': nv, 'mean': m, 'std': s})
            print(f"    BSC={nv:.3f}: {m:.2f}±{s:.2f}")
        all_results['T8_4x4_BSC'] = t8_bsc
        
        # T=8 BEC sweep
        print("\n  T=8 (4×4) BEC sweep:")
        t8_bec = []
        for nv in noise_vals + [0.4, 0.5]:
            m, s = eval_channel(front32, snn_t8, back_t8, test_loader_32, bec, nv)
            t8_bec.append({'rate': nv, 'mean': m, 'std': s})
            print(f"    BEC={nv:.3f}: {m:.2f}±{s:.2f}")
        all_results['T8_4x4_BEC'] = t8_bec
        
        # T=4 model
        snap_t4 = 'snapshots_cifar_t4'
        s3_t4 = sorted([f for f in os.listdir(snap_t4) if f.startswith('s3_t4_best')])
        if s3_t4:
            snn_t4 = SpikeAdaptSC(C_in=2048, C1=256, C2=128, T=4,
                                   target_rate=0.75, channel_type='bsc').to(device)
            back_t4 = ResNet50Back(100, 2048).to(device)
            ck4 = torch.load(f'{snap_t4}/{s3_t4[-1]}', map_location=device, weights_only=False)
            snn_t4.load_state_dict(ck4['model'])
            back_t4.load_state_dict(ck4['back'])
            snn_t4.eval(); back_t4.eval()
            
            if 'front' in ck4:
                front32.load_state_dict(ck4['front'])
            
            print("\n  T=4 (4×4) BSC sweep:")
            t4_bsc = []
            for nv in noise_vals:
                m, s = eval_channel(front32, snn_t4, back_t4, test_loader_32, bsc, nv)
                t4_bsc.append({'rate': nv, 'mean': m, 'std': s})
                print(f"    BSC={nv:.3f}: {m:.2f}±{s:.2f}")
            all_results['T4_4x4_BSC'] = t4_bsc
            
            print("\n  T=4 (4×4) BEC sweep:")
            t4_bec = []
            for nv in noise_vals + [0.4, 0.5]:
                m, s = eval_channel(front32, snn_t4, back_t4, test_loader_32, bec, nv)
                t4_bec.append({'rate': nv, 'mean': m, 'std': s})
                print(f"    BEC={nv:.3f}: {m:.2f}±{s:.2f}")
            all_results['T4_4x4_BEC'] = t4_bec
        
        # 14×14 model
        snap14 = 'snapshots_cifar_14x14'
        if os.path.exists(snap14):
            s3_14 = sorted([f for f in os.listdir(snap14)
                           if f.startswith('s3_snn14_best')])
            if s3_14:
                front14 = ResNet50Front224().to(device)
                snn14 = SpikeAdaptSC(C_in=1024, C1=256, C2=36, T=8,
                                      target_rate=0.75, channel_type='bsc').to(device)
                back14 = ResNet50Back(100, 1024).to(device)
                ck14 = torch.load(f'{snap14}/{s3_14[-1]}', map_location=device,
                                  weights_only=False)
                snn14.load_state_dict(ck14['model'])
                back14.load_state_dict(ck14['back'])
                if 'front' in ck14:
                    front14.load_state_dict(ck14['front'])
                snn14.eval(); back14.eval(); front14.eval()
                
                print("\n  14×14 BSC sweep:")
                r14_bsc = []
                for nv in noise_vals:
                    m, s = eval_channel(front14, snn14, back14, test_loader_224, bsc, nv)
                    r14_bsc.append({'rate': nv, 'mean': m, 'std': s})
                    print(f"    BSC={nv:.3f}: {m:.2f}±{s:.2f}")
                all_results['T8_14x14_BSC'] = r14_bsc
                
                print("\n  14×14 BEC sweep:")
                r14_bec = []
                for nv in noise_vals + [0.4, 0.5]:
                    m, s = eval_channel(front14, snn14, back14, test_loader_224, bec, nv)
                    r14_bec.append({'rate': nv, 'mean': m, 'std': s})
                    print(f"    BEC={nv:.3f}: {m:.2f}±{s:.2f}")
                all_results['T8_14x14_BEC'] = r14_bec
        
        with open(f'{OUT}/cifar_extended_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  ✅ Results saved to {OUT}cifar_extended_results.json")
    
    # ================================================================
    # STEP 5: Generate comparison figures
    # ================================================================
    if args.step in [0, 5]:
        print("\n" + "=" * 60)
        print("  Step 5: Generating comparison figures")
        print("=" * 60)
        
        with open(f'{OUT}/cifar_extended_results.json') as f:
            all_results = json.load(f)
        
        # Figure: 4×4 BSC + BEC (T=8 vs T=4)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        colors = {'T8': '#D32F2F', 'T4': '#1976D2'}
        
        # BSC panel
        for key, label, c, mk in [('T8_4x4_BSC', 'T=8 (4×4)', colors['T8'], 's'),
                                    ('T4_4x4_BSC', 'T=4 (4×4)', colors['T4'], 'o')]:
            if key in all_results:
                res = all_results[key]
                ax1.plot([r['rate'] for r in res], [r['mean'] for r in res],
                        f'{mk}-', color=c, lw=2, ms=5, label=label)
        
        ax1.set_xlabel('Bit Error Rate (BER)')
        ax1.set_ylabel('Classification Accuracy (%)')
        ax1.set_title('(a) BSC Channel (4×4)', fontweight='bold')
        ax1.legend(loc='lower left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # BEC panel
        for key, label, c, mk in [('T8_4x4_BEC', 'T=8 (4×4)', colors['T8'], 's'),
                                    ('T4_4x4_BEC', 'T=4 (4×4)', colors['T4'], 'o')]:
            if key in all_results:
                res = all_results[key]
                ax2.plot([r['rate'] for r in res], [r['mean'] for r in res],
                        f'{mk}--', color=c, lw=2, ms=5, label=label)
        
        ax2.set_xlabel('Bit Erasure Rate')
        ax2.set_ylabel('Classification Accuracy (%)')
        ax2.set_title('(b) BEC Channel (4×4)', fontweight='bold')
        ax2.legend(loc='lower left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{OUT}/cifar_t4_vs_t8.png', dpi=200)
        plt.savefig(f'{OUT}/cifar_t4_vs_t8.pdf', dpi=200)
        plt.close()
        print(f"  ✅ {OUT}cifar_t4_vs_t8.png")
        
        # Figure: 14×14 vs 4×4
        if 'T8_14x14_BSC' in all_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            
            for key, label, c, mk in [('T8_4x4_BSC', '4×4 (16 blocks)', '#D32F2F', 's'),
                                        ('T8_14x14_BSC', '14×14 (196 blocks)', '#1976D2', 'o')]:
                if key in all_results:
                    res = all_results[key]
                    ax1.plot([r['rate'] for r in res], [r['mean'] for r in res],
                            f'{mk}-', color=c, lw=2, ms=5, label=label)
            ax1.set_xlabel('Bit Error Rate')
            ax1.set_ylabel('Classification Accuracy (%)')
            ax1.set_title('(a) BSC Channel', fontweight='bold')
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            for key, label, c, mk in [('T8_4x4_BEC', '4×4 (16 blocks)', '#D32F2F', 's'),
                                        ('T8_14x14_BEC', '14×14 (196 blocks)', '#1976D2', 'o')]:
                if key in all_results:
                    res = all_results[key]
                    ax2.plot([r['rate'] for r in res], [r['mean'] for r in res],
                            f'{mk}--', color=c, lw=2, ms=5, label=label)
            ax2.set_xlabel('Bit Erasure Rate')
            ax2.set_ylabel('Classification Accuracy (%)')
            ax2.set_title('(b) BEC Channel', fontweight='bold')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{OUT}/cifar_4x4_vs_14x14.png', dpi=200)
            plt.savefig(f'{OUT}/cifar_4x4_vs_14x14.pdf', dpi=200)
            plt.close()
            print(f"  ✅ {OUT}cifar_4x4_vs_14x14.png")
        
        print("\n✅ All figures generated!")


if __name__ == '__main__':
    main()
