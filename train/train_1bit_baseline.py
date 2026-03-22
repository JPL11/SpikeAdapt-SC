#!/usr/bin/env python3
"""Non-spiking 1-bit baseline: matched-payload binary CNN for fair comparison.

Answers: "Is the robustness uniquely spiking, or just binary transmission?"

Architecture:
  ResNet-50 backbone → CNN encoder (Conv→BN→sign()) → BSC → CNN decoder → classifier
  Same spatial grid (14×14), same C2=36 channels, T=1 (no temporal coding).
  Payload: 14×14×36 = 7056 bits (same as SNN at ρ=1.0 per timestep).

Uses straight-through estimator (STE) for sign() during backprop.
"""
import torch, torch.nn as nn, torch.optim as optim
import sys, os, json, random, numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms as T

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from run_final_pipeline import AIDDataset5050, RESISC45Dataset
from train_aid_v2 import ResNet50Front, ResNet50Back

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class STE_Sign(torch.autograd.Function):
    """Straight-through estimator for sign()."""
    @staticmethod
    def forward(ctx, x):
        return (x > 0).float()  # Binary: 0 or 1 (like spikes)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # Pass gradient straight through


class BinaryCNN_Encoder(nn.Module):
    """CNN encoder that outputs binary (0/1) features."""
    def __init__(self, C_in=1024, C1=256, C2=36):
        super().__init__()
        self.conv1 = nn.Conv2d(C_in, C1, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(C1)
        self.conv2 = nn.Conv2d(C1, C2, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(C2)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return STE_Sign.apply(x)  # Binary output: 0/1


class BinaryCNN_Decoder(nn.Module):
    """CNN decoder that reconstructs features from binary input."""
    def __init__(self, C_in=1024, C1=256, C2=36):
        super().__init__()
        self.conv1 = nn.Conv2d(C2, C1, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(C1)
        self.conv2 = nn.Conv2d(C1, C_in, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(C_in)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        return torch.relu(self.bn2(self.conv2(x)))


class BinaryCNN_SC(nn.Module):
    """Non-spiking 1-bit semantic communication baseline.
    
    No temporal coding (T=1), no LIF neurons, no membrane potential.
    Binary output via sign(), same payload as SNN at ρ=1.0 per timestep.
    """
    def __init__(self, C_in=1024, C1=256, C2=36):
        super().__init__()
        self.encoder = BinaryCNN_Encoder(C_in, C1, C2)
        self.decoder = BinaryCNN_Decoder(C_in, C1, C2)
    
    def forward(self, feat, ber=0.0):
        # Encode to binary
        binary = self.encoder(feat)  # (B, C2, 14, 14), values in {0, 1}
        
        # BSC channel: flip bits with probability ber
        if ber > 0 and self.training or (ber > 0 and not self.training):
            flip_mask = (torch.rand_like(binary) < ber).float()
            received = (binary + flip_mask) % 2  # XOR: flip selected bits
        else:
            received = binary
        
        # Decode
        Fp = self.decoder(received)
        
        with torch.no_grad():
            fr = binary.mean().item()  # "firing rate" for comparison
        
        return Fp, {'firing_rate': fr, 'binary': binary}


def train_1bit(dataset_name, n_classes, train_loader, test_loader, seed, bb_path, epochs=60):
    """Train 1-bit CNN baseline."""
    snap_dir = f'./snapshots_{dataset_name}_1bit_seed{seed}/'
    os.makedirs(snap_dir, exist_ok=True)
    
    # Load backbone
    front = ResNet50Front(grid_size=14).to(device)
    bb = torch.load(bb_path, map_location=device)
    front.load_state_dict({k:v for k,v in bb.items()
                           if not k.startswith(('layer4.','fc.','avgpool.','spatial_pool.'))}, strict=False)
    front.eval()
    for p in front.parameters(): p.requires_grad = False
    
    back = ResNet50Back(n_classes).to(device)
    back_state = {k:v for k,v in bb.items()
                  if k.startswith(('layer4.','fc.','avgpool.','spatial_pool.'))}
    back.load_state_dict(back_state, strict=False)
    
    model = BinaryCNN_SC(C_in=1024, C1=256, C2=36).to(device)
    
    optimizer = optim.Adam(list(model.parameters()) + list(back.parameters()), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    print(f"\n--- Training 1-bit CNN baseline ({dataset_name}, seed={seed}, {epochs} epochs) ---", flush=True)
    
    for epoch in range(1, epochs + 1):
        model.train(); back.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            # Train with random BER (same as SNN training)
            ber = random.choice([0.0, 0.0, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
            Fp, _ = model(feat, ber=ber)
            loss = criterion(back(Fp), labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        if epoch % 10 == 0 or epoch == epochs:
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
    
    # Full BER sweep evaluation
    results = {}
    # Reload best
    best_ck = sorted([f for f in os.listdir(snap_dir) if f.startswith('1bit_best_')],
                     key=lambda x: float(x.split('_')[-1].replace('.pth','')))[-1]
    ck = torch.load(os.path.join(snap_dir, best_ck), map_location=device)
    model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
    model.eval(); back.eval()
    
    for ber in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                Fp, _ = model(front(imgs), ber=ber)
                correct += back(Fp).argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        acc = round(100. * correct / total, 2)
        results[str(ber)] = acc
        print(f"  BER={ber}: {acc}%", flush=True)
    
    with open(os.path.join(snap_dir, 'ber_sweep.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    del model, back, front; torch.cuda.empty_cache()
    return results


def main():
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    
    tf_train = T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
                           T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))])
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])
    
    all_results = {}
    
    # AID
    ds_train = AIDDataset5050('./data', tf_train, 'train', seed=42)
    ds_test = AIDDataset5050('./data', tf_test, 'test', seed=42)
    train_loader = DataLoader(ds_train, 32, True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(ds_test, 32, False, num_workers=4, pin_memory=True)
    all_results['aid'] = train_1bit('aid', 30, train_loader, test_loader, 42,
                                     './snapshots_aid_5050_seed42/backbone_best.pth')
    
    # RESISC45
    ds_train = RESISC45Dataset('./data', tf_train, 'train', train_ratio=0.20, seed=42)
    ds_test = RESISC45Dataset('./data', tf_test, 'test', train_ratio=0.20, seed=42)
    train_loader = DataLoader(ds_train, 32, True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(ds_test, 32, False, num_workers=4, pin_memory=True)
    all_results['resisc45'] = train_1bit('resisc45', 45, train_loader, test_loader, 42,
                                          './snapshots_resisc45_5050_seed42/backbone_best.pth')
    
    # Save combined results
    with open('eval/1bit_baseline_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n=== 1-BIT BASELINE COMPLETE ===")
    for ds in ['aid', 'resisc45']:
        print(f"{ds}: clean={all_results[ds]['0.0']}% BER=0.30={all_results[ds]['0.3']}%")
    print("Done!", flush=True)


if __name__ == '__main__':
    main()
