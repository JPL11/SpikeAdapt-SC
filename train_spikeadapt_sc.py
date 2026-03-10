# ============================================================================
# SpikeAdapt-SC: Adaptive Spiking Semantic Communication for CIFAR-100
# Based on architecture doc: spikeadapt_sc_architecture.md
#
# Extends SNN-SC with:
#   - Spike Rate Entropy Estimator (content-adaptive masking)
#   - Block Masking with Straight-Through Estimator
#   - BSC/BEC Digital Channel Simulation
#   - Rate Regularization Loss
#
# 3-Step Training:
#   Step 1: Train ResNet50 backbone on CIFAR-100 (or load pretrained)
#   Step 2: Train SpikeAdapt-SC (freeze backbone) - 50 epochs, lr=1e-4
#   Step 3: Joint fine-tune everything - 50 epochs, lr=1e-5
#
# Loss: L_total = L_CE + λ₁·L_entropy + λ₂·L_rate
# ============================================================================
import os
import sys
import math
import random
import numpy as np
from tqdm import tqdm

os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

print("="*70, flush=True)
print("SpikeAdapt-SC Training for CIFAR-100", flush=True)
print("="*70, flush=True)


# ############################################################################
# 1. SURROGATE GRADIENT
# ############################################################################
class SpikeFunction(torch.autograd.Function):
    """Heaviside forward, sigmoid surrogate backward.
    FIXED: conditional gradient for threshold (only when requires_grad)."""
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
        grad_m = grad_output * sg
        grad_th = -(grad_output * sg).sum() if ctx.th_needs_grad else None
        return grad_m, grad_th


# ############################################################################
# 2. SPIKING NEURONS
# ############################################################################
class IFNeuron(nn.Module):
    """Integrate-and-Fire with soft reset (Eq. 1-3)."""
    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, x, membrane=None):
        if membrane is None:
            membrane = torch.zeros_like(x)
        membrane = membrane + x
        spike = SpikeFunction.apply(membrane, self.threshold)
        membrane = membrane - spike * self.threshold
        return spike, membrane


class IHFNeuron(nn.Module):
    """Integrate-Hybrid-Fire — outputs spike AND membrane (Eq. 11).
    FIXED: no .item() — threshold stays in computation graph."""
    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(float(threshold)))  # Learnable

    def forward(self, x, membrane=None):
        if membrane is None:
            membrane = torch.zeros_like(x)
        membrane = membrane + x
        spike = SpikeFunction.apply(membrane, self.threshold)  # NO .item()!
        membrane = membrane - spike * self.threshold
        return spike, membrane  # Both used by Converter


# ############################################################################
# 3. ENCODER (Table I - Classification)
# ############################################################################
class Encoder(nn.Module):
    """
    Input:  (B, C_in, H, W)   e.g. (B, 2048, 4, 4)
    Output: (B, C2, H, W)     e.g. (B, 128, 4, 4) — binary spikes
    """
    def __init__(self, C_in=2048, C1=256, C2=128):
        super().__init__()
        self.conv1 = nn.Conv2d(C_in, C1, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(C1)
        self.if1 = IFNeuron()

        self.conv2 = nn.Conv2d(C1, C2, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(C2)
        self.if2 = IFNeuron()

    def forward(self, F, mem_if1=None, mem_if2=None):
        """Single timestep. Returns spikes + membrane states."""
        x = self.bn1(self.conv1(F))
        s1, mem_if1 = self.if1(x, mem_if1)

        x = self.bn2(self.conv2(s1))
        s2, mem_if2 = self.if2(x, mem_if2)

        return s2, mem_if1, mem_if2  # s2 is binary (B, C2, H, W)


# ############################################################################
# 4. SPIKE RATE ENTROPY ESTIMATOR (Section 2.2 - NEW)
# ############################################################################
class SpikeRateEntropyEstimator(nn.Module):
    """
    Computes per-spatial-block entropy from spike firing rates.
    No learnable parameters — purely statistical.

    Input: list of T tensors, each (B, C2, H2, W2) ∈ {0,1}
    Output: entropy_map (B, H2, W2) ∈ [0, 1]
            firing_rate (B, H2, W2) ∈ [0, 1]
    """
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, all_S2):
        # Stack: (T, B, C2, H2, W2)
        stacked = torch.stack(all_S2, dim=0)

        # Average over time (dim=0) and channels (dim=2) → (B, H2, W2)
        firing_rate = stacked.mean(dim=0).mean(dim=1)

        # Binary entropy: H(f) = -f·log₂(f) - (1-f)·log₂(1-f)
        f = firing_rate.clamp(self.eps, 1.0 - self.eps)
        entropy_map = -f * torch.log2(f) - (1 - f) * torch.log2(1 - f)

        return entropy_map, firing_rate


# ############################################################################
# 5. BLOCK MASKING WITH STE (Section 2.3 - NEW)
# ############################################################################
class BlockMask(nn.Module):
    """
    Generates binary spatial mask based on entropy threshold.
    Uses straight-through estimator (STE) for training.

    High entropy → high information → KEEP (mask=1)
    Low entropy  → low information  → DROP (mask=0)
    """
    def __init__(self, eta=0.5, temperature=0.1):
        super().__init__()
        self.eta = eta
        self.temperature = temperature

    def forward(self, entropy_map, training=True):
        """
        Args:
            entropy_map: (B, H2, W2) ∈ [0, 1]
        Returns:
            mask: (B, 1, H2, W2) — broadcastable over channels
            tx_rate: scalar — fraction of blocks transmitted
        """
        if training:
            # Soft mask (gradient flows through sigmoid)
            soft_mask = torch.sigmoid((entropy_map - self.eta) / self.temperature)
            # Hard mask (binary)
            hard_mask = (entropy_map >= self.eta).float()
            # STE: forward=hard, backward=soft
            mask = hard_mask + (soft_mask - soft_mask.detach())
        else:
            mask = (entropy_map >= self.eta).float()

        mask = mask.unsqueeze(1)  # (B, 1, H2, W2)
        tx_rate = mask.mean()

        return mask, tx_rate

    def apply_mask(self, S2_t, mask):
        """Zero out blocks below entropy threshold."""
        return S2_t * mask  # broadcasts over C2


# ############################################################################
# 6. DIGITAL CHANNEL SIMULATION (Section 2.4 - NEW)
# ############################################################################
class BSC_Channel(nn.Module):
    """Binary Symmetric Channel — flips bits with probability p.
    FIXED: applies noise in BOTH train and eval modes."""
    def forward(self, x, bit_error_rate):
        if bit_error_rate <= 0:
            return x
        flip_mask = (torch.rand_like(x.float()) < bit_error_rate).float()
        x_noisy = (x + flip_mask) % 2
        if self.training:
            return x + (x_noisy - x).detach()  # STE
        else:
            return x_noisy  # No gradient needed in eval


class BEC_Channel(nn.Module):
    """Binary Erasure Channel — erases bits with probability p.
    FIXED: applies noise in BOTH train and eval modes."""
    def forward(self, x, erasure_rate):
        if erasure_rate <= 0:
            return x
        erase_mask = (torch.rand_like(x.float()) < erasure_rate).float()
        random_bits = (torch.rand_like(x.float()) > 0.5).float()
        x_noisy = x * (1 - erase_mask) + random_bits * erase_mask
        if self.training:
            return x + (x_noisy - x).detach()  # STE
        else:
            return x_noisy


# ############################################################################
# 7. DECODER: RECONSTRUCTOR + CONVERTER (Section 2.5-2.6)
# ############################################################################
class Decoder(nn.Module):
    """
    Reconstructor: Conv(C2→C1)-BN-IF → Conv(C1→C_out)-BN-IHF
    Converter:     Stack 2T outputs, FC+Sigmoid weights, weighted sum
    """
    def __init__(self, C_out=2048, C1=256, C2=128, T=8):
        super().__init__()
        self.T = T

        # Reconstructor Stage 3
        self.conv3 = nn.Conv2d(C2, C1, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(C1)
        self.if3 = IFNeuron()

        # Reconstructor Stage 4 with IHF
        self.conv4 = nn.Conv2d(C1, C_out, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(C_out)
        self.ihf = IHFNeuron()

        # Converter: learnable weights for 2T outputs
        self.converter_fc = nn.Linear(2 * T, 2 * T)

    def reconstruct_step(self, S2_t, mem_if3, mem_ihf):
        """Single timestep reconstruction."""
        x = self.bn3(self.conv3(S2_t))
        s3, mem_if3 = self.if3(x, mem_if3)

        x = self.bn4(self.conv4(s3))
        spike, mem_ihf = self.ihf(x, mem_ihf)

        return spike, mem_ihf.clone(), mem_if3, mem_ihf

    def convert(self, all_F_s, all_F_m):
        """Fuse 2T outputs into final feature. Paper Eq. 10."""
        # Interleave: [F_s^1, F_m^1, F_s^2, F_m^2, ...]
        interleaved = []
        for t in range(self.T):
            interleaved.append(all_F_s[t])
            interleaved.append(all_F_m[t])

        # Stack: (B, 2T, C, H, W)
        stacked = torch.stack(interleaved, dim=1)
        B, K, C, H, W = stacked.shape  # K = 2T

        # Apply FC across 2T dimension with Sigmoid
        x = stacked.permute(0, 2, 3, 4, 1)        # (B, C, H, W, 2T)
        weights = torch.sigmoid(self.converter_fc(x))  # (B, C, H, W, 2T)
        x_weighted = x * weights
        F_prime = x_weighted.sum(dim=-1)            # (B, C, H, W)

        return F_prime

    def forward(self, received_all_S2, mask):
        """Full decoder: zero-fill → reconstruct T steps → convert."""
        mem_if3, mem_ihf = None, None
        all_F_s = []
        all_F_m = []

        for t in range(self.T):
            # Zero-fill: already applied by mask during transmission
            S2_t = received_all_S2[t] * mask  # Re-apply mask for safety
            spike, mem_out, mem_if3, mem_ihf = self.reconstruct_step(
                S2_t, mem_if3, mem_ihf)
            all_F_s.append(spike)
            all_F_m.append(mem_out)

        F_prime = self.convert(all_F_s, all_F_m)
        return F_prime


# ############################################################################
# 8. FULL SpikeAdapt-SC MODEL
# ############################################################################
class SpikeAdaptSC(nn.Module):
    """
    Complete SpikeAdapt-SC module.
    Accepts backbone features, returns reconstructed features + metadata.
    """
    def __init__(self, C_in=2048, C1=256, C2=128, T=8,
                 eta=0.5, temperature=0.1, channel_type='bsc'):
        super().__init__()
        self.T = T
        self.encoder = Encoder(C_in, C1, C2)
        self.entropy_est = SpikeRateEntropyEstimator()
        self.block_mask = BlockMask(eta=eta, temperature=temperature)
        self.decoder = Decoder(C_in, C1, C2, T)

        if channel_type == 'bsc':
            self.channel = BSC_Channel()
        else:
            self.channel = BEC_Channel()

    def forward(self, backbone_features, bit_error_rate=0.0):
        """
        Args:
            backbone_features: (B, C_in, H, W) from frozen backbone
            bit_error_rate: channel BER (randomized during training)
        Returns:
            F_prime: reconstructed features (B, C_in, H, W)
            latent_spikes: list of T spike tensors (for entropy loss)
            tx_rate: transmission rate scalar (for rate loss)
            stats: dict with compression info
        """
        # Phase 1: Encode T timesteps
        all_S2 = []
        mem_if1, mem_if2 = None, None
        for t in range(self.T):
            s2, mem_if1, mem_if2 = self.encoder(backbone_features, mem_if1, mem_if2)
            all_S2.append(s2)

        # Phase 2: Compute entropy & generate mask
        entropy_map, firing_rate = self.entropy_est(all_S2)
        mask, tx_rate = self.block_mask(entropy_map, training=self.training)

        # Phase 3: Apply mask & transmit through channel
        received_all_S2 = []
        for t in range(self.T):
            masked = self.block_mask.apply_mask(all_S2[t], mask)
            noisy = self.channel(masked, bit_error_rate)
            received_all_S2.append(noisy)

        # Phase 4: Decode
        F_prime = self.decoder(received_all_S2, mask)

        # Compute compression stats
        B, C2, H2, W2 = all_S2[0].shape
        C_in = backbone_features.shape[1]
        original_bits = C_in * backbone_features.shape[2] * backbone_features.shape[3] * 32
        nnz = mask.sum() / B  # Average non-zero blocks
        transmitted_bits = self.T * nnz.item() * C2 + H2 * W2
        compression_ratio = original_bits / max(transmitted_bits, 1)

        stats = {
            'tx_rate': tx_rate.item(),
            'compression_ratio': compression_ratio,
            'mask_density': mask.mean().item(),
            'avg_entropy': entropy_map.mean().item(),
        }

        return F_prime, all_S2, tx_rate, stats


# ############################################################################
# 9. LOSS FUNCTIONS
# ############################################################################
def compute_entropy_loss(all_S2, alpha=1.0):
    """
    Entropy maximization loss (Paper Eq. 14).
    L_entropy = (alpha - H(p))^2
    """
    all_bits = torch.cat([s.flatten() for s in all_S2])
    p1 = all_bits.mean()
    p0 = 1.0 - p1
    eps = 1e-7
    p0 = torch.clamp(p0, eps, 1 - eps)
    p1 = torch.clamp(p1, eps, 1 - eps)
    H = -(p0 * torch.log2(p0) + p1 * torch.log2(p1))
    return (alpha - H) ** 2


def compute_rate_loss(tx_rate, target_rate=0.75):
    """
    Rate regularization loss.
    Encourages transmission rate to match target (e.g., 75% = 25% savings).
    """
    return (tx_rate - target_rate) ** 2


# ############################################################################
# 10. RESNET50 BACKBONE (split at layer4 output)
# ############################################################################
class ResNet50Front(nn.Module):
    """Layers 1-14 of ResNet50 (everything up to and including layer4)."""
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # (B, 2048, H/32, W/32) → (B, 2048, 1, 1) for CIFAR 32x32


class ResNet50Back(nn.Module):
    """Classification head: avgpool + fc."""
    def __init__(self, num_classes=100, pretrained=True):
        super().__init__()
        resnet = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None)
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ############################################################################
# 11. CONFIG
# ############################################################################
class Config:
    # Dataset
    data_root = "./data"
    num_classes = 100
    input_size = 32  # CIFAR-100

    # Architecture
    C_in = 2048
    C1 = 256
    C2 = 128
    T = 8           # timesteps

    # Masking
    eta = 0.5           # entropy threshold
    temperature = 0.1   # mask softness
    target_rate = 0.75  # target transmission rate
    channel_type = 'bsc'

    # Loss weights
    lambda_entropy = 1.0
    lambda_rate = 0.5

    # Step 1: Backbone
    step1_epochs = 100
    step1_lr = 0.1
    step1_batch = 128

    # Step 2: Train SpikeAdapt-SC (frozen backbone)
    step2_epochs = 50
    step2_lr = 1e-4
    step2_batch = 64
    ber_max = 0.3  # max BER during training

    # Step 3: Joint fine-tuning
    step3_epochs = 50
    step3_lr = 1e-5
    step3_batch = 32
    step3_accum = 2

    # Training
    early_stop_patience = 15

    # Save
    save_dir = "./snapshots_spikeadapt/"
    backbone_path = "./snapshots_spikeadapt/backbone_best.pth"

args = Config()
os.makedirs(args.save_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)


# ############################################################################
# 12. DATASET
# ############################################################################
print("\nLoading CIFAR-100...", flush=True)

train_transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])
test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

train_dataset = torchvision.datasets.CIFAR100(
    root=args.data_root, train=True, download=True, transform=train_transform)
test_dataset = torchvision.datasets.CIFAR100(
    root=args.data_root, train=False, download=True, transform=test_transform)

print(f"✓ Train: {len(train_dataset)}, Test: {len(test_dataset)}", flush=True)


# ############################################################################
# 13. STEP 1: TRAIN BACKBONE (or load pretrained)
# ############################################################################
def train_backbone():
    """Train ResNet50 on CIFAR-100 from scratch."""
    print("\n" + "="*70, flush=True)
    print("STEP 1: Train ResNet50 Backbone", flush=True)
    print("="*70, flush=True)

    model = torchvision.models.resnet50(weights=None)
    model.fc = nn.Linear(2048, args.num_classes)
    # Adjust conv1 for CIFAR-100 (32x32 instead of 224x224)
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = nn.Identity()
    model = model.to(device)

    loader = DataLoader(train_dataset, batch_size=args.step1_batch,
                        shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=256,
                             shuffle=False, num_workers=4, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.step1_lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.step1_epochs)

    best_acc = 0.0
    for epoch in range(args.step1_epochs):
        model.train()
        correct, total = 0, 0
        pbar = tqdm(loader, desc=f"S1 Epoch {epoch+1}/{args.step1_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            pbar.set_postfix({'acc': f'{100.*correct/total:.1f}%',
                              'loss': f'{loss.item():.4f}'})
        scheduler.step()

        # Evaluate
        if (epoch + 1) % 5 == 0:
            model.eval()
            test_correct, test_total = 0, 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    test_total += labels.size(0)
                    test_correct += predicted.eq(labels).sum().item()
            test_acc = 100. * test_correct / test_total
            print(f"  Test Acc: {test_acc:.2f}%", flush=True)

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), args.backbone_path)
                print(f"  ✓ Best backbone: {best_acc:.2f}%", flush=True)

    print(f"\n✅ Step 1 Complete! Best Acc: {best_acc:.2f}%", flush=True)
    return model


# ############################################################################
# 14. LOAD/BUILD BACKBONE PARTS
# ############################################################################
def build_backbone_parts():
    """Load trained backbone and split into front (feature) + back (classifier)."""
    front = ResNet50Front(pretrained=False)
    back = ResNet50Back(num_classes=args.num_classes, pretrained=False)

    # Adjust conv1 for CIFAR (same as training)
    front.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    front.maxpool = nn.Identity()

    if os.path.exists(args.backbone_path):
        print(f"\nLoading backbone from {args.backbone_path}", flush=True)
        state = torch.load(args.backbone_path, map_location=device)
        # Map full ResNet state into front/back
        front_state = {}
        back_state = {}
        for k, v in state.items():
            if k.startswith('fc.'):
                back_state[k] = v
            elif k.startswith('avgpool.'):
                back_state[k] = v
            else:
                front_state[k] = v

        front.load_state_dict(front_state, strict=False)
        back.load_state_dict(back_state, strict=False)
        print("✓ Backbone loaded", flush=True)
    else:
        print("\n⚠️ No backbone checkpoint found. Training from scratch...", flush=True)
        model = train_backbone()
        # Re-extract parts
        return build_backbone_parts()

    return front.to(device), back.to(device)


# Build backbone
front, back = build_backbone_parts()

# Quick test to verify backbone dimensions
with torch.no_grad():
    dummy = torch.randn(2, 3, 32, 32).to(device)
    feat = front(dummy)
    print(f"✓ Backbone output: {feat.shape}", flush=True)  # Should be (2, 2048, 1, 1)
    out = back(feat)
    print(f"✓ Classifier output: {out.shape}", flush=True)  # Should be (2, 100)


# ############################################################################
# 15. CREATE SpikeAdapt-SC
# ############################################################################
# Determine spatial dims from backbone
with torch.no_grad():
    sample_feat = front(torch.randn(1, 3, 32, 32).to(device))
    spatial_h, spatial_w = sample_feat.shape[2], sample_feat.shape[3]

print(f"\nCreating SpikeAdapt-SC...", flush=True)
print(f"  Feature dims: ({args.C_in}, {spatial_h}, {spatial_w})", flush=True)

spikeadapt = SpikeAdaptSC(
    C_in=args.C_in, C1=args.C1, C2=args.C2, T=args.T,
    eta=args.eta, temperature=args.temperature,
    channel_type=args.channel_type
).to(device)

total_params = sum(p.numel() for p in spikeadapt.parameters())
print(f"  SpikeAdapt-SC params: {total_params:,}", flush=True)
print(f"  T={args.T}, C2={args.C2}, η={args.eta}, target_rate={args.target_rate}", flush=True)


# ############################################################################
# 16. METRICS
# ############################################################################
criterion = nn.CrossEntropyLoss()

def evaluate(front, spikeadapt, back, loader, ber=0.0):
    """Evaluate accuracy with given channel BER."""
    front.eval()
    spikeadapt.eval()
    back.eval()
    correct, total = 0, 0
    avg_tx_rate = 0.0
    n_batches = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Eval (BER={ber})", leave=False):
            images, labels = images.to(device), labels.to(device)

            feat = front(images)
            F_prime, _, tx_rate, stats = spikeadapt(feat, bit_error_rate=ber)
            outputs = back(F_prime)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            avg_tx_rate += stats['tx_rate']
            n_batches += 1

    acc = 100. * correct / total
    avg_tx = avg_tx_rate / max(n_batches, 1)
    return acc, avg_tx


# ############################################################################
# 17. STEP 2: TRAIN SpikeAdapt-SC (Frozen backbone)
# ############################################################################
print("\n" + "="*70, flush=True)
print("STEP 2: Train SpikeAdapt-SC (Freeze Backbone)", flush=True)
print("="*70, flush=True)

# Freeze backbone
for p in front.parameters():
    p.requires_grad = False
for p in back.parameters():
    p.requires_grad = False
front.eval()
back.eval()

train_loader_s2 = DataLoader(train_dataset, batch_size=args.step2_batch,
                              shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=128,
                          shuffle=False, num_workers=4, pin_memory=True)

optimizer_s2 = optim.Adam(spikeadapt.parameters(), lr=args.step2_lr, weight_decay=1e-4)
scheduler_s2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_s2, T_max=args.step2_epochs, eta_min=1e-6)

best_s2 = 0.0
no_improve = 0

for epoch in range(args.step2_epochs):
    spikeadapt.train()
    epoch_loss = 0.0
    epoch_correct, epoch_total = 0, 0
    epoch_tx = 0.0

    pbar = tqdm(train_loader_s2, desc=f"S2 Epoch {epoch+1}/{args.step2_epochs}")

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Random BER for robustness
        ber = random.uniform(0, args.ber_max)

        with torch.no_grad():
            feat = front(images)

        F_prime, all_S2, tx_rate, stats = spikeadapt(feat, bit_error_rate=ber)
        outputs = back(F_prime)

        # Losses
        L_CE = criterion(outputs, labels)
        L_ent = compute_entropy_loss(all_S2, alpha=1.0)
        L_rate = compute_rate_loss(tx_rate, args.target_rate)

        total_loss = L_CE + args.lambda_entropy * L_ent + args.lambda_rate * L_rate

        optimizer_s2.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(spikeadapt.parameters(), max_norm=1.0)
        optimizer_s2.step()

        _, predicted = outputs.max(1)
        epoch_total += labels.size(0)
        epoch_correct += predicted.eq(labels).sum().item()
        epoch_loss += total_loss.item()
        epoch_tx += stats['tx_rate']

        pbar.set_postfix({
            'loss': f'{total_loss.item():.4f}',
            'acc': f'{100.*predicted.eq(labels).sum().item()/labels.size(0):.0f}%',
            'tx': f'{stats["tx_rate"]:.2f}',
            'ber': f'{ber:.2f}'
        })

        del feat, F_prime, all_S2, total_loss
    torch.cuda.empty_cache()

    scheduler_s2.step()

    train_acc = 100. * epoch_correct / epoch_total
    avg_loss = epoch_loss / len(train_loader_s2)
    avg_tx = epoch_tx / len(train_loader_s2)

    print(f"\nS2 Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={train_acc:.2f}%, "
          f"TxRate={avg_tx:.3f}, LR={scheduler_s2.get_last_lr()[0]:.2e}", flush=True)

    # Validate every 5 epochs
    if (epoch + 1) % 5 == 0:
        test_acc, test_tx = evaluate(front, spikeadapt, back, test_loader, ber=0.0)
        print(f"  Test Acc: {test_acc:.2f}% (BER=0), TxRate={test_tx:.3f}", flush=True)

        # Also test with channel noise
        test_acc_noisy, _ = evaluate(front, spikeadapt, back, test_loader, ber=0.1)
        print(f"  Test Acc: {test_acc_noisy:.2f}% (BER=0.1)", flush=True)

        if test_acc > best_s2:
            best_s2 = test_acc
            no_improve = 0
            torch.save(spikeadapt.state_dict(),
                       os.path.join(args.save_dir, f"spikeadapt_best_{test_acc:.2f}.pth"))
            print(f"  ✓ Best S2: {test_acc:.2f}%", flush=True)
        else:
            no_improve += 5
            print(f"  No improvement for {no_improve} epochs", flush=True)

        if no_improve >= args.early_stop_patience:
            print(f"\n⚠️ Early stopping", flush=True)
            break

print(f"\n✅ Step 2 Complete! Best: {best_s2:.2f}%", flush=True)

# Load best
best_s2_files = [f for f in os.listdir(args.save_dir) if f.startswith("spikeadapt_best_")]
if best_s2_files:
    best_path = os.path.join(args.save_dir, sorted(best_s2_files)[-1])
    spikeadapt.load_state_dict(torch.load(best_path, map_location=device))
    print(f"✓ Loaded best S2: {best_path}", flush=True)


# ############################################################################
# 18. STEP 3: JOINT FINE-TUNING (head only + SpikeAdapt-SC, backbone frozen)
# ############################################################################
print("\n" + "="*70, flush=True)
print("STEP 3: Fine-tune Classifier Head + SpikeAdapt-SC", flush=True)
print("="*70, flush=True)

# Unfreeze only the classification head, keep backbone frozen
for p in front.parameters():
    p.requires_grad = False
for p in back.parameters():
    p.requires_grad = True

trainable_params = list(back.parameters()) + list(spikeadapt.parameters())
optimizer_s3 = optim.Adam(trainable_params, lr=args.step3_lr, weight_decay=1e-4)
scheduler_s3 = optim.lr_scheduler.CosineAnnealingLR(optimizer_s3, T_max=args.step3_epochs, eta_min=1e-7)

train_loader_s3 = DataLoader(train_dataset, batch_size=args.step3_batch,
                              shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

best_s3 = 0.0
no_improve_s3 = 0

for epoch in range(args.step3_epochs):
    spikeadapt.train()
    back.train()

    epoch_loss = 0.0
    epoch_correct, epoch_total = 0, 0

    pbar = tqdm(train_loader_s3, desc=f"S3 Epoch {epoch+1}/{args.step3_epochs}")
    optimizer_s3.zero_grad()

    for step, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        ber = random.uniform(0, args.ber_max)

        with torch.no_grad():
            feat = front(images)

        F_prime, all_S2, tx_rate, stats = spikeadapt(feat, bit_error_rate=ber)
        outputs = back(F_prime)

        L_CE = criterion(outputs, labels)
        L_ent = compute_entropy_loss(all_S2, alpha=1.0)
        L_rate = compute_rate_loss(tx_rate, args.target_rate)
        total_loss = (L_CE + args.lambda_entropy * L_ent + args.lambda_rate * L_rate) / args.step3_accum

        total_loss.backward()

        if (step + 1) % args.step3_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer_s3.step()
            optimizer_s3.zero_grad()

        _, predicted = outputs.max(1)
        epoch_total += labels.size(0)
        epoch_correct += predicted.eq(labels).sum().item()
        epoch_loss += total_loss.item() * args.step3_accum

        pbar.set_postfix({
            'loss': f'{total_loss.item() * args.step3_accum:.4f}',
            'acc': f'{100.*predicted.eq(labels).sum().item()/labels.size(0):.0f}%',
            'tx': f'{stats["tx_rate"]:.2f}'
        })

        del feat, F_prime, all_S2, total_loss
    torch.cuda.empty_cache()

    scheduler_s3.step()

    train_acc = 100. * epoch_correct / epoch_total
    avg_loss = epoch_loss / len(train_loader_s3)
    print(f"\nS3 Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={train_acc:.2f}%, "
          f"LR={scheduler_s3.get_last_lr()[0]:.2e}", flush=True)

    if (epoch + 1) % 5 == 0:
        test_acc, test_tx = evaluate(front, spikeadapt, back, test_loader, ber=0.0)
        print(f"  Test Acc: {test_acc:.2f}% (BER=0), TxRate={test_tx:.3f}", flush=True)

        test_acc_noisy, _ = evaluate(front, spikeadapt, back, test_loader, ber=0.1)
        print(f"  Test Acc: {test_acc_noisy:.2f}% (BER=0.1)", flush=True)

        if test_acc > best_s3:
            best_s3 = test_acc
            no_improve_s3 = 0
            torch.save({
                'spikeadapt': spikeadapt.state_dict(),
                'back': back.state_dict(),
            }, os.path.join(args.save_dir, f"step3_best_{test_acc:.2f}.pth"))
            print(f"  ✓ Best S3: {test_acc:.2f}%", flush=True)
        else:
            no_improve_s3 += 5
            print(f"  No improvement for {no_improve_s3} epochs", flush=True)

        if no_improve_s3 >= args.early_stop_patience:
            print(f"\n⚠️ Early stopping", flush=True)
            break


# ############################################################################
# 19. FINAL EVALUATION
# ############################################################################
print("\n" + "="*70, flush=True)
print("FINAL EVALUATION", flush=True)
print("="*70, flush=True)

for ber in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
    acc, tx = evaluate(front, spikeadapt, back, test_loader, ber=ber)
    print(f"  BER={ber:.2f}: Acc={acc:.2f}%, TxRate={tx:.3f}", flush=True)

print("\n" + "="*70, flush=True)
print("✅ ALL STEPS COMPLETE!", flush=True)
print(f"Step 2 Best (SpikeAdapt-SC): {best_s2:.2f}%", flush=True)
print(f"Step 3 Best (Fine-tuned):    {best_s3:.2f}%", flush=True)
print(f"Target transmission rate:    {args.target_rate}", flush=True)
print("="*70, flush=True)
