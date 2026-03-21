"""Train and evaluate MLP-FC baseline (fully-connected encoder/decoder).

MLP-FC: features → FC flatten → FC bottleneck → 8-bit uniform quantize → BSC → FC decode → classify
Same bottleneck size as CNN-Uni, but uses FC layers instead of convolutions.
Trains with same backbone and noise curriculum for fair comparison.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
from train_aid_v2 import ResNet50Front, ResNet50Back, UniformQuantizeSTE
from run_final_pipeline import AIDDataset5050

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RESISC45_2080:
    def __init__(self, root, transform, split='test', seed=42):
        full = ImageFolder(os.path.join(root, 'NWPU-RESISC45'), transform=transform)
        rng = np.random.RandomState(seed)
        indices = list(range(len(full)))
        rng.shuffle(indices)
        n_train = int(0.2 * len(full))
        self.dataset = full
        self.indices = indices[n_train:] if split == 'test' else indices[:n_train]
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx): return self.dataset[self.indices[idx]]


class MLPUni(nn.Module):
    """MLP encoder + uniform 8-bit quantization → BSC → MLP decoder.

    Uses FC layers instead of convolutions for the bottleneck.
    Pools features to 7×7 before FC to keep memory manageable.
    """
    def __init__(self, C_in=1024, H=14, W=14, bottleneck=7056, n_bits=8):
        super().__init__()
        self.n_bits = n_bits
        self.C_in, self.H, self.W = C_in, H, W
        # Pool 14×14 → 7×7 to reduce memory
        self.pool = nn.AdaptiveAvgPool2d(7)
        self.unpool = nn.Upsample(size=(H, W), mode='bilinear', align_corners=False)
        feat_dim = C_in * 7 * 7  # 1024×7×7 = 50,176

        # Encoder: flatten → FC → bottleneck
        self.enc = nn.Sequential(
            nn.Linear(feat_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, bottleneck),
            nn.Sigmoid()  # Map to [0,1] for quantization
        )
        # Decoder: bottleneck → FC → reshape
        self.dec = nn.Sequential(
            nn.Linear(bottleneck, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, feat_dim),
        )

    def forward(self, feat, noise_param=0.0):
        B = feat.size(0)
        x = self.pool(feat)  # Pool 14×14 → 7×7
        x = x.reshape(B, -1)  # Flatten

        # Encode + quantize
        z = self.enc(x)
        z_q = UniformQuantizeSTE.apply(z, self.n_bits)

        # BSC on bit planes
        n_levels = 2 ** self.n_bits
        bits = (z_q * (n_levels - 1)).round().long()
        bit_planes = []
        for b in range(self.n_bits):
            bit_planes.append(((bits >> b) & 1).float())
        bit_tensor = torch.stack(bit_planes, dim=-1)

        if noise_param > 0:
            flip = (torch.rand_like(bit_tensor) < noise_param).float()
            bit_tensor = (bit_tensor + flip) % 2

        values = torch.zeros_like(z_q)
        for b in range(self.n_bits):
            values += bit_tensor[..., b] * (2 ** b)
        z_noisy = values / (n_levels - 1)

        # Decode
        out = self.dec(z_noisy)
        out = out.reshape(B, self.C_in, 7, 7)
        Fp = self.unpool(out)  # 7×7 → 14×14
        total_bits = B * z.shape[1] * self.n_bits
        return Fp, {'total_bits': total_bits}


def train_mlp(ds_name, n_classes, backbone_path, train_ds, test_ds, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  MLP-FC — Training on {ds_name}")
    print(f"{'='*60}")

    front = ResNet50Front(grid_size=14).to(device)
    back = ResNet50Back(n_classes).to(device)
    bb = torch.load(backbone_path, map_location=device)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))},
                          strict=False)
    back.load_state_dict({k: v for k, v in bb.items()
                          if k.startswith(('layer4.', 'fc.', 'avgpool.'))},
                         strict=False)
    front.eval()

    model = MLPUni(C_in=1024, H=14, W=14, bottleneck=7056, n_bits=8).to(device)

    train_loader = DataLoader(train_ds, 32, True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)

    # Freeze backbone
    for p in front.parameters(): p.requires_grad = False
    for p in back.parameters(): p.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    ce = nn.CrossEntropyLoss()
    best = 0

    # Stage 1: 60 epochs with noise curriculum
    print("\n--- S2: Training MLP encoder/decoder (60 epochs) ---")
    for epoch in range(1, 61):
        model.train()
        correct = total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                feat = front(imgs)
            ber = np.random.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.25],
                                   p=[0.2, 0.2, 0.2, 0.15, 0.15, 0.10])
            Fp, _ = model(feat, noise_param=ber)
            out = back(Fp)
            loss = ce(out, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            correct += out.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
        scheduler.step()

        if epoch % 5 == 0:
            model.eval()
            tc = tt = 0
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    feat = front(imgs)
                    Fp, _ = model(feat, noise_param=0.0)
                    tc += back(Fp).argmax(1).eq(labels).sum().item()
                    tt += labels.size(0)
            test_acc = 100. * tc / tt
            print(f"  Epoch {epoch:3d}: train={100.*correct/total:.2f}% test={test_acc:.2f}%")
            if test_acc > best:
                best = test_acc
                torch.save({'model': model.state_dict(), 'back': back.state_dict(),
                            'epoch': epoch, 'test_acc': test_acc},
                           f"{out_dir}/mlp_best_{best:.2f}.pth")

    print(f"  Best: {best:.2f}%")
    return best


def eval_mlp(ds_name, n_classes, backbone_path, snap_dir, test_ds):
    print(f"\n--- Evaluating MLP-FC — {ds_name} ---")
    front = ResNet50Front(grid_size=14).to(device)
    back = ResNet50Back(n_classes).to(device)
    bb = torch.load(backbone_path, map_location=device)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))},
                          strict=False)
    front.eval()

    model = MLPUni(C_in=1024, H=14, W=14, bottleneck=7056, n_bits=8).to(device)
    best_files = sorted([f for f in os.listdir(snap_dir) if f.startswith("mlp_best_")])
    ck = torch.load(f"{snap_dir}/{best_files[-1]}", map_location=device)
    model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
    model.eval(); back.eval()
    print(f"  Checkpoint: {best_files[-1]}")

    loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)
    results = {}

    for ber in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in loader:
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
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                     T.Normalize((.485,.456,.406),(.229,.224,.225))])

    all_results = {}

    # AID
    aid_train = AIDDataset5050(root="./data", transform=tf, split='train', seed=42)
    aid_test = AIDDataset5050(root="./data", transform=tf, split='test', seed=42)
    train_mlp("AID (50/50)", 30, "./snapshots_aid_5050_seed42/backbone_best.pth",
              aid_train, aid_test, "./snapshots_mlp_aid_5050")
    all_results['AID'] = eval_mlp("AID (50/50)", 30,
                                   "./snapshots_aid_5050_seed42/backbone_best.pth",
                                   "./snapshots_mlp_aid_5050", aid_test)

    # RESISC45
    res_train = RESISC45_2080(root="./data", transform=tf, split='train', seed=42)
    res_test = RESISC45_2080(root="./data", transform=tf, split='test', seed=42)
    train_mlp("RESISC45 (20/80)", 45, "./snapshots_resisc45_5050_seed42/backbone_best.pth",
              res_train, res_test, "./snapshots_mlp_resisc45_2080")
    all_results['RESISC45'] = eval_mlp("RESISC45 (20/80)", 45,
                                        "./snapshots_resisc45_5050_seed42/backbone_best.pth",
                                        "./snapshots_mlp_resisc45_2080", res_test)

    with open("eval/mlp_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY — MLP-FC")
    print(f"{'='*60}")
    for ds, res in all_results.items():
        print(f"\n{ds}:")
        for ber in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
            print(f"  BER={ber:.2f}: {res[str(ber)]:.2f}%")
    print(f"\n✅ MLP-FC baseline complete.")


if __name__ == '__main__':
    main()
