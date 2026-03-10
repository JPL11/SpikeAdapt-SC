# ============================================================================
# STEP 2 OF 2: Retrain SpikeAdapt-SC with Layer3 Split
# ============================================================================
# Run: python train_layer3_split.py
#
# WHY: Layer4 output is (2048, 4, 4) = 16 spatial blocks.
#      Every image drops the same 4 blocks → not content-adaptive.
#
#      Layer3 output is (1024, 8, 8) = 64 spatial blocks.
#      Different images WILL get different masks → real adaptation.
#      Also: less compute on edge device (no layer4!) = better UAV story.
#
# CHANGES from original:
#   - Front = layers 1-3 (output: 1024×8×8), Back = layer4 + avgpool + fc
#   - C_in = 1024 (was 2048)
#   - Spatial: 8×8 = 64 blocks (was 4×4 = 16)
#   - Mask has 64 positions → real per-image variance possible
#   - Backbone reused (same checkpoint, just split differently)
#
# Output: ./snapshots_layer3/
# ============================================================================

import os, sys, math, random, json
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
print(f"Device: {device}")

SNAP_DIR = "./snapshots_layer3/"
BACKBONE_DIR = "./snapshots_spikeadapt/"
os.makedirs(SNAP_DIR, exist_ok=True)


# ############################################################################
# BUG-FIXED CORE (same as train_baselines.py)
# ############################################################################

class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane, threshold):
        th_needs_grad = isinstance(threshold, torch.Tensor) and threshold.requires_grad
        if isinstance(threshold, (int, float)):
            threshold = torch.tensor(float(threshold), device=membrane.device)
        ctx.save_for_backward(membrane, threshold)
        ctx.th_needs_grad = th_needs_grad
        return (membrane > threshold).float()
    @staticmethod
    def backward(ctx, grad_output):
        membrane, threshold = ctx.saved_tensors
        scale = 10.0
        sig = torch.sigmoid(scale * (membrane - threshold))
        sg = sig * (1 - sig) * scale
        grad_th = -(grad_output * sg).sum() if ctx.th_needs_grad else None
        return grad_output * sg, grad_th

class IFNeuron(nn.Module):
    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = threshold
    def forward(self, x, mem=None):
        if mem is None: mem = torch.zeros_like(x)
        mem = mem + x
        sp = SpikeFunction.apply(mem, self.threshold)
        mem = mem - sp * self.threshold
        return sp, mem

class IHFNeuron(nn.Module):
    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(float(threshold)))
    def forward(self, x, mem=None):
        if mem is None: mem = torch.zeros_like(x)
        mem = mem + x
        sp = SpikeFunction.apply(mem, self.threshold)
        mem = mem - sp * self.threshold
        return sp, mem

class BSC_Channel(nn.Module):
    def forward(self, x, ber):
        if ber <= 0: return x
        flip = (torch.rand_like(x.float()) < ber).float()
        x_n = (x + flip) % 2
        return x + (x_n - x).detach() if self.training else x_n

class BEC_Channel(nn.Module):
    def forward(self, x, er):
        if er <= 0: return x
        erase = (torch.rand_like(x.float()) < er).float()
        rb = (torch.rand_like(x.float()) > 0.5).float()
        x_n = x * (1 - erase) + rb * erase
        return x + (x_n - x).detach() if self.training else x_n


# ############################################################################
# NEW BACKBONE SPLIT: Front = layer1-3, Back = layer4 + head
# ############################################################################

class ResNet50Front_L3(nn.Module):
    """
    Layers 1-3 of ResNet50. Output: (B, 1024, 8, 8) for CIFAR-100.

    Computation path:
      conv1(3→64, 3×3, s1) → BN → ReLU → (no maxpool for CIFAR)
      layer1: (64→256, 3 blocks)     → 32×32 → 32×32
      layer2: (256→512, 4 blocks, s2) → 32×32 → 16×16
      layer3: (512→1024, 6 blocks, s2) → 16×16 → 8×8
      Output: (B, 1024, 8, 8)

    This runs on the UAV/edge device.
    Less compute than layer4 split → better for constrained edge.
    """
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet50(weights=None)
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)  # CIFAR mod
        self.bn1 = r.bn1
        self.relu = r.relu
        self.maxpool = nn.Identity()  # CIFAR mod
        self.layer1 = r.layer1
        self.layer2 = r.layer2
        self.layer3 = r.layer3
        # NO layer4 — that's on the cloud now

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x  # (B, 1024, 8, 8)


class ResNet50Back_L3(nn.Module):
    """
    Layer4 + classification head. Runs on the cloud.

    Input: reconstructed features (B, 1024, 8, 8)
    Output: logits (B, num_classes)
    """
    def __init__(self, num_classes=100):
        super().__init__()
        r = torchvision.models.resnet50(weights=None)
        self.layer4 = r.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.layer4(x)    # (B, 1024, 8, 8) → (B, 2048, 4, 4)
        x = self.avgpool(x)   # (B, 2048, 1, 1)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ############################################################################
# SPIKEADAPT-SC (same architecture, C_in=1024, spatial=8×8)
# ############################################################################

class SpikeRateEntropyEstimator(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
    def forward(self, all_S2):
        stacked = torch.stack(all_S2, dim=0)
        fr = stacked.mean(dim=0).mean(dim=1)
        f = fr.clamp(self.eps, 1.0 - self.eps)
        return -f * torch.log2(f) - (1 - f) * torch.log2(1 - f), fr

class BlockMask(nn.Module):
    def __init__(self, eta=0.5, temperature=0.1):
        super().__init__()
        self.eta = eta
        self.temperature = temperature
    def forward(self, ent, training=True):
        if training:
            soft = torch.sigmoid((ent - self.eta) / self.temperature)
            hard = (ent >= self.eta).float()
            mask = hard + (soft - soft.detach())
        else:
            mask = (ent >= self.eta).float()
        mask = mask.unsqueeze(1)
        return mask, mask.mean()
    def apply_mask(self, x, mask):
        return x * mask

class Encoder(nn.Module):
    def __init__(self, C_in=1024, C1=256, C2=128):
        super().__init__()
        self.conv1 = nn.Conv2d(C_in, C1, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(C1)
        self.if1 = IFNeuron()
        self.conv2 = nn.Conv2d(C1, C2, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(C2)
        self.if2 = IFNeuron()
    def forward(self, F, m1=None, m2=None):
        x = self.bn1(self.conv1(F))
        s1, m1 = self.if1(x, m1)
        x = self.bn2(self.conv2(s1))
        s2, m2 = self.if2(x, m2)
        return s2, m1, m2

class Decoder(nn.Module):
    def __init__(self, C_out=1024, C1=256, C2=128, T=8):
        super().__init__()
        self.T = T
        self.conv3 = nn.Conv2d(C2, C1, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(C1)
        self.if3 = IFNeuron()
        self.conv4 = nn.Conv2d(C1, C_out, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(C_out)
        self.ihf = IHFNeuron()
        self.converter_fc = nn.Linear(2 * T, 2 * T)

    def forward(self, recv_all, mask):
        m3, m4 = None, None
        Fs, Fm = [], []
        for t in range(self.T):
            x = self.bn3(self.conv3(recv_all[t] * mask))
            s3, m3 = self.if3(x, m3)
            x = self.bn4(self.conv4(s3))
            sp, m4 = self.ihf(x, m4)
            Fs.append(sp); Fm.append(m4.clone())
        il = []
        for t in range(self.T):
            il.append(Fs[t]); il.append(Fm[t])
        stk = torch.stack(il, dim=1)
        x = stk.permute(0, 2, 3, 4, 1)
        w = torch.sigmoid(self.converter_fc(x))
        return (x * w).sum(dim=-1)

class SpikeAdaptSC(nn.Module):
    def __init__(self, C_in=1024, C1=256, C2=128, T=8,
                 eta=0.5, temperature=0.1, channel_type='bsc'):
        super().__init__()
        self.T = T
        self.encoder = Encoder(C_in, C1, C2)
        self.entropy_est = SpikeRateEntropyEstimator()
        self.block_mask = BlockMask(eta, temperature)
        self.decoder = Decoder(C_in, C1, C2, T)
        self.channel = BSC_Channel() if channel_type == 'bsc' else BEC_Channel()

    def forward(self, feat, bit_error_rate=0.0, eta_override=None):
        all_S2, m1, m2 = [], None, None
        for t in range(self.T):
            s2, m1, m2 = self.encoder(feat, m1, m2)
            all_S2.append(s2)
        ent, fr = self.entropy_est(all_S2)
        if eta_override is not None:
            old = self.block_mask.eta
            self.block_mask.eta = eta_override
            mask, tx = self.block_mask(ent, training=False)
            self.block_mask.eta = old
        else:
            mask, tx = self.block_mask(ent, training=self.training)
        recv = []
        for t in range(self.T):
            recv.append(self.channel(self.block_mask.apply_mask(all_S2[t], mask),
                                     bit_error_rate))
        Fp = self.decoder(recv, mask)
        B, C2, H2, W2 = all_S2[0].shape
        ob = feat.shape[1] * feat.shape[2] * feat.shape[3] * 32
        tb = self.T * (mask.sum() / B).item() * C2 + H2 * W2
        stats = {
            'tx_rate': tx.item(),
            'compression_ratio': ob / max(tb, 1),
            'mask_density': mask.mean().item(),
            'avg_entropy': ent.mean().item(),
        }
        return Fp, all_S2, tx, stats


# ############################################################################
# LOSSES
# ############################################################################

def entropy_loss(all_S2, alpha=1.0):
    bits = torch.cat([s.flatten() for s in all_S2])
    p1 = bits.mean(); p0 = 1 - p1; eps = 1e-7
    H = -(torch.clamp(p0,eps,1-eps)*torch.log2(torch.clamp(p0,eps,1-eps)) +
          torch.clamp(p1,eps,1-eps)*torch.log2(torch.clamp(p1,eps,1-eps)))
    return (alpha - H) ** 2

def rate_loss(tx_rate, target=0.75):
    return (tx_rate - target) ** 2


# ############################################################################
# EVALUATION
# ############################################################################

def evaluate(front, model, back, loader, ber=0.0):
    front.eval(); model.eval(); back.eval()
    correct, total, tx_sum, n = 0, 0, 0.0, 0
    with torch.no_grad():
        for img, lab in loader:
            img, lab = img.to(device), lab.to(device)
            feat = front(img)
            Fp, _, _, stats = model(feat, bit_error_rate=ber)
            out = back(Fp)
            correct += out.argmax(1).eq(lab).sum().item()
            total += lab.size(0)
            tx_sum += stats['tx_rate']; n += 1
    return 100.*correct/total, tx_sum/max(n,1)


# ############################################################################
# CONFIG
# ############################################################################

class Config:
    # Architecture
    C_in = 1024   # ← CHANGED from 2048
    C1 = 256
    C2 = 128
    T = 8
    eta = 0.5
    temperature = 0.1
    target_rate = 0.75
    channel_type = 'bsc'

    # Loss weights
    lambda_entropy = 1.0
    lambda_rate = 0.5

    # Step 2
    step2_epochs = 60     # Slightly more — new split needs convergence
    step2_lr = 1e-4
    step2_batch = 64
    ber_max = 0.3

    # Step 3
    step3_epochs = 40
    step3_lr = 1e-5
    step3_batch = 32
    step3_accum = 2

    early_stop = 20

args = Config()


# ############################################################################
# DATASET
# ############################################################################

train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                      T.ToTensor(),
                      T.Normalize((0.5071,.4867,.4408),(.2675,.2565,.2761))])
test_tf = T.Compose([T.ToTensor(),
                     T.Normalize((0.5071,.4867,.4408),(.2675,.2565,.2761))])
train_ds = torchvision.datasets.CIFAR100("./data", True, download=True, transform=train_tf)
test_ds  = torchvision.datasets.CIFAR100("./data", False, download=True, transform=test_tf)
train_loader = DataLoader(train_ds, args.step2_batch, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
test_loader = DataLoader(test_ds, 128, shuffle=False, num_workers=4, pin_memory=True)


# ############################################################################
# LOAD BACKBONE (reuse existing, split at layer3)
# ############################################################################

print("\nLoading backbone and splitting at layer3...")
front = ResNet50Front_L3().to(device)
back = ResNet50Back_L3(100).to(device)

bb_path = os.path.join(BACKBONE_DIR, "backbone_best.pth")
state = torch.load(bb_path, map_location=device)

# Map state dict to new split
front_keys = ['conv1.', 'bn1.', 'layer1.', 'layer2.', 'layer3.']
back_keys = ['layer4.', 'fc.']

f_st = {k: v for k, v in state.items()
        if any(k.startswith(p) for p in front_keys)}
b_st = {k: v for k, v in state.items()
        if any(k.startswith(p) for p in back_keys)}

front.load_state_dict(f_st, strict=False)
back.load_state_dict(b_st, strict=False)
front.eval()
print("✓ Backbone loaded (layer3 split)")

# Verify dimensions
with torch.no_grad():
    d = front(torch.randn(1, 3, 32, 32).to(device))
    print(f"★ Feature shape: {d.shape}")  # Should be (1, 1024, 8, 8)
    o = back(d)
    print(f"  Classifier out: {o.shape}")  # Should be (1, 100)
assert d.shape == (1, 1024, 8, 8), f"Expected (1,1024,8,8) got {d.shape}"

# Verify backbone accuracy at this split
back.eval()
correct, total = 0, 0
with torch.no_grad():
    for img, lab in test_loader:
        img, lab = img.to(device), lab.to(device)
        out = back(front(img))
        correct += out.argmax(1).eq(lab).sum().item()
        total += lab.size(0)
print(f"★ Layer3 backbone accuracy: {100.*correct/total:.2f}%")

# Print bandwidth comparison
print(f"\n  Layer4 split: (2048, 4, 4)  → 16 spatial blocks → mask has 16 positions")
print(f"  Layer3 split: (1024, 8, 8)  → 64 spatial blocks → mask has 64 positions")
print(f"  At η=0.5, 75% rate: Layer4 drops 4/16 = same 4 blocks for every image")
print(f"                      Layer3 drops 16/64 = different blocks per image!")
print(f"\n  Layer3 original bits: 1024×8×8×32 = 2,097,152")
print(f"  Layer4 original bits: 2048×4×4×32 = 1,048,576")
print(f"  Layer3 transmitted (T=8, 75%): 8×48×128+64 = 49,216 bits (CR≈42×)")
print(f"  Layer4 transmitted (T=8, 75%): 8×12×128+16 = 12,304 bits (CR≈85×)")
print(f"  Layer3 sends more bits but compresses a bigger feature.\n")


# ############################################################################
# CREATE MODEL
# ############################################################################

spikeadapt = SpikeAdaptSC(
    C_in=args.C_in, C1=args.C1, C2=args.C2, T=args.T,
    eta=args.eta, temperature=args.temperature,
    channel_type=args.channel_type
).to(device)

params = sum(p.numel() for p in spikeadapt.parameters())
print(f"SpikeAdapt-SC (L3 split) params: {params:,}")
print(f"  C_in={args.C_in}, C1={args.C1}, C2={args.C2}, T={args.T}")
print(f"  Spatial: 8×8 = 64 blocks")


# ############################################################################
# STEP 2: TRAIN SPIKEADAPT-SC (frozen backbone)
# ############################################################################

print("\n" + "="*60)
print("STEP 2: Train SpikeAdapt-SC (Layer3, frozen backbone)")
print("="*60)

for p in front.parameters(): p.requires_grad = False
for p in back.parameters():  p.requires_grad = False
front.eval(); back.eval()

criterion = nn.CrossEntropyLoss()
opt_s2 = optim.Adam(spikeadapt.parameters(), lr=args.step2_lr, weight_decay=1e-4)
sched_s2 = optim.lr_scheduler.CosineAnnealingLR(opt_s2, T_max=args.step2_epochs, eta_min=1e-6)

best_s2, no_improve = 0.0, 0

for ep in range(args.step2_epochs):
    spikeadapt.train()
    correct, total, ep_loss, ep_tx = 0, 0, 0.0, 0.0

    pbar = tqdm(train_loader, desc=f"S2 E{ep+1}/{args.step2_epochs}")
    for img, lab in pbar:
        img, lab = img.to(device), lab.to(device)
        ber = random.uniform(0, args.ber_max)
        with torch.no_grad():
            feat = front(img)
        Fp, all_S2, tx, stats = spikeadapt(feat, bit_error_rate=ber)
        out = back(Fp)

        L_CE = criterion(out, lab)
        L_ent = entropy_loss(all_S2, alpha=1.0)
        L_rt = rate_loss(tx, args.target_rate)
        loss = L_CE + args.lambda_entropy * L_ent + args.lambda_rate * L_rt

        opt_s2.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(spikeadapt.parameters(), max_norm=1.0)
        opt_s2.step()

        _, pred = out.max(1)
        total += lab.size(0)
        correct += pred.eq(lab).sum().item()
        ep_loss += loss.item()
        ep_tx += stats['tx_rate']
        pbar.set_postfix({'L': f'{loss.item():.3f}',
                          'A': f'{100.*pred.eq(lab).sum().item()/lab.size(0):.0f}%',
                          'tx': f'{stats["tx_rate"]:.2f}'})

    sched_s2.step()
    print(f"  S2 E{ep+1}: Loss={ep_loss/len(train_loader):.4f}, "
          f"Acc={100.*correct/total:.2f}%, Tx={ep_tx/len(train_loader):.3f}")

    if (ep + 1) % 5 == 0:
        acc, tx = evaluate(front, spikeadapt, back, test_loader, ber=0.0)
        acc_n, _ = evaluate(front, spikeadapt, back, test_loader, ber=0.1)
        print(f"  Test: {acc:.2f}% (BER=0), {acc_n:.2f}% (BER=0.1), Tx={tx:.3f}")
        if acc > best_s2:
            best_s2 = acc; no_improve = 0
            torch.save(spikeadapt.state_dict(),
                       os.path.join(SNAP_DIR, f"s2_best_{acc:.2f}.pth"))
            print(f"  ✓ Best S2: {acc:.2f}%")
        else:
            no_improve += 5
        if no_improve >= args.early_stop:
            print("  ⚠️ Early stopping"); break

# Reload best
s2f = sorted([f for f in os.listdir(SNAP_DIR) if f.startswith("s2_best_")])
if s2f:
    spikeadapt.load_state_dict(torch.load(os.path.join(SNAP_DIR, s2f[-1]),
                                           map_location=device))
    print(f"✓ Loaded best S2: {s2f[-1]}")


# ############################################################################
# STEP 3: FINE-TUNE (head + SpikeAdapt, backbone frozen)
# ############################################################################

print("\n" + "="*60)
print("STEP 3: Fine-tune (Layer3 split)")
print("="*60)

for p in back.parameters(): p.requires_grad = True
params_s3 = list(back.parameters()) + list(spikeadapt.parameters())
opt_s3 = optim.Adam(params_s3, lr=args.step3_lr, weight_decay=1e-4)
sched_s3 = optim.lr_scheduler.CosineAnnealingLR(opt_s3, T_max=args.step3_epochs, eta_min=1e-7)

train_loader_s3 = DataLoader(train_ds, args.step3_batch, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
best_s3, no_improve = 0.0, 0

for ep in range(args.step3_epochs):
    spikeadapt.train(); back.train()
    correct, total, ep_loss = 0, 0, 0.0

    pbar = tqdm(train_loader_s3, desc=f"S3 E{ep+1}/{args.step3_epochs}")
    opt_s3.zero_grad()
    for step, (img, lab) in enumerate(pbar):
        img, lab = img.to(device), lab.to(device)
        ber = random.uniform(0, args.ber_max)
        with torch.no_grad():
            feat = front(img)
        Fp, all_S2, tx, stats = spikeadapt(feat, bit_error_rate=ber)
        out = back(Fp)

        L_CE = criterion(out, lab)
        L_ent = entropy_loss(all_S2, alpha=1.0)
        L_rt = rate_loss(tx, args.target_rate)
        loss = (L_CE + args.lambda_entropy * L_ent + args.lambda_rate * L_rt) / args.step3_accum
        loss.backward()

        if (step + 1) % args.step3_accum == 0:
            nn.utils.clip_grad_norm_(params_s3, max_norm=1.0)
            opt_s3.step(); opt_s3.zero_grad()

        _, pred = out.max(1)
        total += lab.size(0)
        correct += pred.eq(lab).sum().item()
        ep_loss += loss.item() * args.step3_accum
        pbar.set_postfix({'L': f'{loss.item()*args.step3_accum:.3f}',
                          'A': f'{100.*pred.eq(lab).sum().item()/lab.size(0):.0f}%'})

    sched_s3.step()

    if (ep + 1) % 5 == 0:
        acc, tx = evaluate(front, spikeadapt, back, test_loader, ber=0.0)
        acc_n, _ = evaluate(front, spikeadapt, back, test_loader, ber=0.1)
        print(f"  S3 Test: {acc:.2f}% (BER=0), {acc_n:.2f}% (BER=0.1), Tx={tx:.3f}")

        # Check mask variance (THE KEY METRIC)
        tx_rates = []
        with torch.no_grad():
            spikeadapt.eval()
            for img, _ in test_loader:
                feat = front(img.to(device))
                _, _, _, st = spikeadapt(feat, bit_error_rate=0.0)
                tx_rates.append(st['tx_rate'])
        tx_std = np.std(tx_rates)
        print(f"  ★ Tx rate std across batches: {tx_std:.4f} "
              f"(was ~0.001 on 4×4, want >>0.01 on 8×8)")

        if acc > best_s3:
            best_s3 = acc; no_improve = 0
            torch.save({'spikeadapt': spikeadapt.state_dict(),
                        'back': back.state_dict()},
                       os.path.join(SNAP_DIR, f"s3_best_{acc:.2f}.pth"))
            print(f"  ✓ Best S3: {acc:.2f}%")
        else:
            no_improve += 5
        if no_improve >= args.early_stop:
            print("  ⚠️ Early stopping"); break


# ############################################################################
# FINAL EVALUATION
# ############################################################################

print("\n" + "="*60)
print("FINAL EVALUATION (Layer3 Split)")
print("="*60)

# Reload best
s3f = sorted([f for f in os.listdir(SNAP_DIR) if f.startswith("s3_best_")])
if s3f:
    ckpt = torch.load(os.path.join(SNAP_DIR, s3f[-1]), map_location=device)
    spikeadapt.load_state_dict(ckpt['spikeadapt'])
    back.load_state_dict(ckpt['back'])

# BER sweep
print("\nBER Sweep:")
for ber in [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
    accs = []
    for _ in range(10 if ber > 0 else 1):
        a, _ = evaluate(front, spikeadapt, back, test_loader, ber=ber)
        accs.append(a)
    print(f"  BER={ber:.3f}: {np.mean(accs):.2f}% ±{np.std(accs):.2f}")

# η sweep
print("\nη Sweep:")
for eta in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    spikeadapt.eval()
    correct, total, tx_sum, n = 0, 0, 0.0, 0
    tx_per_batch = []
    with torch.no_grad():
        for img, lab in test_loader:
            img, lab = img.to(device), lab.to(device)
            Fp, _, _, st = spikeadapt(front(img), bit_error_rate=0.0, eta_override=eta)
            out = back(Fp)
            correct += out.argmax(1).eq(lab).sum().item()
            total += lab.size(0)
            tx_sum += st['tx_rate']; n += 1
            tx_per_batch.append(st['tx_rate'])
    acc = 100.*correct/total
    tx_std = np.std(tx_per_batch)
    ob = 1024 * 8 * 8 * 32
    avg_tx = tx_sum / n
    tb = args.T * avg_tx * 64 * 128 + 64
    cr = ob / max(tb, 1)
    print(f"  η={eta:.1f}: Acc={acc:.2f}%, Tx={avg_tx:.3f} (std={tx_std:.4f}), CR={cr:.1f}×")

# IHF threshold
print(f"\nIHF learned threshold: {spikeadapt.decoder.ihf.threshold.item():.4f}")
if spikeadapt.decoder.ihf.threshold.grad is not None:
    print(f"IHF threshold grad: {spikeadapt.decoder.ihf.threshold.grad.item():.6f}")

print(f"\n{'='*60}")
print(f"✅ LAYER3 TRAINING COMPLETE!")
print(f"  S2 Best: {best_s2:.2f}%")
print(f"  S3 Best: {best_s3:.2f}%")
print(f"  Key: Check tx_rate std — should be much larger than 4×4 version")
print(f"{'='*60}")
