#!/usr/bin/env python3
"""Extended ρ sweep: evaluate all 8 ρ values at 7 BER levels.
BER = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
ρ   = [0.10, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 1.0]
"""

import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, '/home/jpli/SemCom')
from models.spikeadapt_sc import SpikeAdaptSC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

BER_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
RHO_VALUES = [0.10, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 1.0]
N_REPEAT = 5  # repeats per noisy BER

# ============================================================================
# Model definitions
# ============================================================================
class ResNet50Front14(nn.Module):
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet50(weights=None)
        self.conv1 = r.conv1; self.bn1 = r.bn1
        self.relu = r.relu; self.maxpool = r.maxpool
        self.layer1 = r.layer1; self.layer2 = r.layer2
        self.layer3 = r.layer3
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        return self.layer3(self.layer2(self.layer1(x)))


class ResNet50Back(nn.Module):
    def __init__(self, C_in=1024, n_classes=30):
        super().__init__()
        r = torchvision.models.resnet50(weights=None)
        self.layer4 = r.layer4
        # Adjust layer4 input if needed
        if C_in != 1024:
            self.layer4[0].conv1 = nn.Conv2d(C_in, 256, 1, bias=False)
            self.layer4[0].downsample[0] = nn.Conv2d(C_in, 2048, 1, stride=2, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, n_classes)
    def forward(self, x):
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))


def bsc_channel(x, ber):
    if ber <= 0: return x
    flip = (torch.rand_like(x.float()) < ber).float()
    return (x.float() * (1 - flip) + (1 - x.float()) * flip)


@torch.no_grad()
def eval_rho_ber(front, snn, back, loader, rho, ber, n_repeat=5):
    """Evaluate at specific rho and BER."""
    # Temporarily override target rate
    old_rate = snn.block_mask.target_rate
    snn.block_mask.target_rate = rho
    
    nr = n_repeat if ber > 0 else 1
    accs = []
    for _ in range(nr):
        correct, total = 0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            feat = front(images)
            
            # SNN encode
            all_S2, m1, m2 = [], None, None
            for t in range(snn.T):
                _, s2, m1, m2 = snn.encoder(feat, m1, m2, t=t)
                all_S2.append(s2)
            
            # Score + mask at given rho
            importance = snn.scorer(all_S2, ber=ber)
            mask, tx = snn.block_mask(importance, training=False)
            
            # BSC channel
            recv = [bsc_channel(all_S2[t] * mask, ber) for t in range(snn.T)]
            
            # Decode
            Fp, _, _ = snn.decoder(recv, mask)
            out = back(Fp)
            correct += out.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
        accs.append(100. * correct / total)
    
    snn.block_mask.target_rate = old_rate
    return float(np.mean(accs)), float(np.std(accs))


def load_dataset(ds_name):
    test_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    if ds_name == 'aid':
        ds = torchvision.datasets.ImageFolder('./data/AID', transform=test_tf)
        n_classes = 30
        # 50/50 split — use second half as test
        n = len(ds)
        test_idx = list(range(n // 2, n))
        test_ds = Subset(ds, test_idx)
    else:
        ds = torchvision.datasets.ImageFolder('./data/NWPU-RESISC45', transform=test_tf)
        n_classes = 45
        # 20/80 split — use last 80% as test
        n = len(ds)
        test_idx = list(range(int(0.2 * n), n))
        test_ds = Subset(ds, test_idx)
    return test_ds, n_classes


def load_models(ds_name, n_classes, bb_dir, v5c_dir):
    front = ResNet50Front14().to(device)
    # Load backbone
    bb_files = sorted([f for f in os.listdir(bb_dir) if 'best' in f and f.endswith('.pth')])
    bb = torch.load(f'{bb_dir}/{bb_files[-1]}', map_location=device, weights_only=False)
    bb_st = {k: v for k, v in bb.items() if not k.startswith(('fc.', 'avgpool.', 'layer4.'))}
    front.load_state_dict(bb_st, strict=False)
    front.eval()
    
    snn = SpikeAdaptSC(C_in=1024, C1=256, C2=36, T=8,
                       target_rate=0.75, channel_type='bsc').to(device)
    back = ResNet50Back(C_in=1024, n_classes=n_classes).to(device)
    
    v5c_files = sorted([f for f in os.listdir(v5c_dir) if 'best' in f and f.endswith('.pth')])
    ck = torch.load(f'{v5c_dir}/{v5c_files[-1]}', map_location=device, weights_only=False)
    if 'model' in ck:
        snn.load_state_dict(ck['model'], strict=False)
        back.load_state_dict(ck['back'], strict=False)
    else:
        snn.load_state_dict(ck, strict=False)
    snn.eval(); back.eval()
    
    for p in list(front.parameters()) + list(snn.parameters()) + list(back.parameters()):
        p.requires_grad = False
    
    print(f"  ✓ Loaded {ds_name} models from {bb_files[-1]}, {v5c_files[-1]}")
    return front, snn, back


# ============================================================================
# MAIN
# ============================================================================
results = {}
out_file = 'paper/figures/rho_sweep_extended.json'

for ds_name, bb_dir, v5c_dir in [
    ('aid', 'snapshots_aid_5050_seed42', 'snapshots_aid_v5cna_seed42'),
    ('resisc45', 'snapshots_resisc45_5050_seed42', 'snapshots_resisc45_v5cna_seed42'),
]:
    print(f"\n{'='*60}")
    print(f"  {ds_name.upper()}")
    print(f"{'='*60}")
    
    test_ds, n_classes = load_dataset(ds_name)
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    front, snn, back = load_models(ds_name, n_classes, bb_dir, v5c_dir)
    
    ds_results = {}
    for rho in RHO_VALUES:
        ds_results[str(rho)] = {}
        for ber in BER_LEVELS:
            # Check if we already have this data point
            mean, std = eval_rho_ber(front, snn, back, loader, rho, ber, N_REPEAT)
            ds_results[str(rho)][str(ber)] = {'mean': mean, 'std': std}
            print(f"  ρ={rho:.3f}  BER={ber:.2f}: {mean:.2f}% ±{std:.2f}")
        
        # Save after each rho to avoid data loss
        results[ds_name] = ds_results
        with open(out_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    print(f"\n  ✅ {ds_name} complete")

print(f"\n✅ All results saved to {out_file}")
print(f"   {len(RHO_VALUES)} ρ × {len(BER_LEVELS)} BER × 2 datasets = {len(RHO_VALUES)*len(BER_LEVELS)*2} evaluations")
