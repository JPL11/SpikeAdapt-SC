# ============================================================================
# LEARNED IMPORTANCE NETWORK — Layer3 SpikeAdapt-SC
# ============================================================================
# Replace spike-rate entropy with a learned lightweight importance scorer.
# The model LEARNS which spatial blocks to keep vs drop.
#
# Key difference from entropy-based:
#   Entropy: H(firing_rate) → hardcoded criterion, can't adapt
#   Learned: MLP(features) → importance score, trained end-to-end
#
# Loss: CE + λ_rate * |tx_rate - target|  (NO entropy loss)
# ============================================================================

import os, sys, random, json
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

SNAP_DIR = "./snapshots_learned_imp/"
BACKBONE_DIR = "./snapshots_spikeadapt/"
os.makedirs(SNAP_DIR, exist_ok=True)


# ############################################################################
# CORE (same bug-fixed versions)
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
        return grad_output * sg, -(grad_output * sg).sum() if ctx.th_needs_grad else None

class IFNeuron(nn.Module):
    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = threshold
    def forward(self, x, mem=None):
        if mem is None: mem = torch.zeros_like(x)
        mem = mem + x
        sp = SpikeFunction.apply(mem, self.threshold)
        return sp, mem - sp * self.threshold

class IHFNeuron(nn.Module):
    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(float(threshold)))
    def forward(self, x, mem=None):
        if mem is None: mem = torch.zeros_like(x)
        mem = mem + x
        sp = SpikeFunction.apply(mem, self.threshold)
        return sp, mem - sp * self.threshold

class BSC_Channel(nn.Module):
    def forward(self, x, ber):
        if ber <= 0: return x
        flip = (torch.rand_like(x.float()) < ber).float()
        x_n = (x + flip) % 2
        return x + (x_n - x).detach() if self.training else x_n


# ############################################################################
# BACKBONE (Layer3 split)
# ############################################################################

class ResNet50Front_L3(nn.Module):
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet50(weights=None)
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = r.bn1; self.relu = r.relu; self.maxpool = nn.Identity()
        self.layer1 = r.layer1; self.layer2 = r.layer2; self.layer3 = r.layer3
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        return self.layer3(self.layer2(self.layer1(self.maxpool(x))))

class ResNet50Back_L3(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        r = torchvision.models.resnet50(weights=None)
        self.layer4 = r.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)
    def forward(self, x):
        return self.fc(torch.flatten(self.avgpool(self.layer4(x)), 1))


# ############################################################################
# LEARNED IMPORTANCE SCORER (NEW — replaces SpikeRateEntropyEstimator)
# ############################################################################

class LearnedImportanceScorer(nn.Module):
    """Lightweight importance scorer: features → per-block score.
    
    Architecture: 
      1×1 conv to reduce channels → global channel pool → sigmoid
      Produces (B, H, W) importance scores
      
    Key: trained end-to-end with task loss + rate regularization.
    The model learns WHAT to keep, not just high-entropy blocks.
    """
    def __init__(self, C_in=128, hidden=32):
        super().__init__()
        # Channel reduction + spatial importance
        self.scorer = nn.Sequential(
            nn.Conv2d(C_in, hidden, 1),       # (B, 32, H, W)
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),           # (B, 1, H, W) 
            nn.Sigmoid()                        # scores in [0, 1]
        )
    
    def forward(self, all_S2):
        """Takes list of spike tensors, returns per-block importance scores."""
        # Average across timesteps: (T, B, C, H, W) → (B, C, H, W)
        stacked = torch.stack(all_S2, dim=0)
        avg_spikes = stacked.mean(dim=0)  # (B, C2, H, W)
        importance = self.scorer(avg_spikes).squeeze(1)  # (B, H, W)
        return importance


class LearnedBlockMask(nn.Module):
    """Differentiable masking using learned importance scores.
    
    Training: Gumbel-sigmoid for differentiable hard sampling
    Eval: threshold at target_rate percentile for exact rate control
    """
    def __init__(self, target_rate=0.75, temperature=0.5):
        super().__init__()
        self.target_rate = target_rate
        self.temperature = temperature
    
    def forward(self, importance, training=True):
        B, H, W = importance.shape
        
        if training:
            # Gumbel-sigmoid: differentiable binary sampling
            # Higher importance → higher probability of being kept
            logits = torch.log(importance / (1 - importance + 1e-7) + 1e-7)
            # Add Gumbel noise for exploration
            u = torch.rand_like(logits).clamp(1e-7, 1 - 1e-7)
            gumbel = -torch.log(-torch.log(u))
            soft = torch.sigmoid((logits + gumbel) / self.temperature)
            hard = (soft > 0.5).float()
            mask = hard + (soft - soft.detach())  # STE
        else:
            # At eval: keep top-k by importance score
            k = max(1, int(self.target_rate * H * W))
            flat = importance.view(B, -1)
            _, idx = flat.topk(k, dim=1)
            mask = torch.zeros_like(flat)
            mask.scatter_(1, idx, 1.0)
            mask = mask.view(B, H, W)
        
        mask = mask.unsqueeze(1)  # (B, 1, H, W)
        tx_rate = mask.mean()
        return mask, tx_rate
    
    def apply_mask(self, x, mask):
        return x * mask


# ############################################################################
# ENCODER / DECODER (same as before)
# ############################################################################

class Encoder(nn.Module):
    def __init__(self, C_in=1024, C1=256, C2=128):
        super().__init__()
        self.conv1 = nn.Conv2d(C_in, C1, 3, 1, 1); self.bn1 = nn.BatchNorm2d(C1); self.if1 = IFNeuron()
        self.conv2 = nn.Conv2d(C1, C2, 3, 1, 1); self.bn2 = nn.BatchNorm2d(C2); self.if2 = IFNeuron()
    def forward(self, F, m1=None, m2=None):
        s1, m1 = self.if1(self.bn1(self.conv1(F)), m1)
        s2, m2 = self.if2(self.bn2(self.conv2(s1)), m2)
        return s2, m1, m2

class Decoder(nn.Module):
    def __init__(self, C_out=1024, C1=256, C2=128, T=8):
        super().__init__()
        self.T = T
        self.conv3 = nn.Conv2d(C2, C1, 3, 1, 1); self.bn3 = nn.BatchNorm2d(C1); self.if3 = IFNeuron()
        self.conv4 = nn.Conv2d(C1, C_out, 3, 1, 1); self.bn4 = nn.BatchNorm2d(C_out); self.ihf = IHFNeuron()
        self.converter_fc = nn.Linear(2*T, 2*T)
    def forward(self, recv_all, mask):
        m3, m4 = None, None; Fs, Fm = [], []
        for t in range(self.T):
            s3, m3 = self.if3(self.bn3(self.conv3(recv_all[t] * mask)), m3)
            sp, m4 = self.ihf(self.bn4(self.conv4(s3)), m4)
            Fs.append(sp); Fm.append(m4.clone())
        il = []
        for t in range(self.T): il.append(Fs[t]); il.append(Fm[t])
        stk = torch.stack(il, dim=1)
        x = stk.permute(0, 2, 3, 4, 1)
        return (x * torch.sigmoid(self.converter_fc(x))).sum(dim=-1)


# ############################################################################
# FULL MODEL WITH LEARNED IMPORTANCE
# ############################################################################

class SpikeAdaptSC_LearnedImp(nn.Module):
    """SpikeAdapt-SC with learned importance scorer instead of entropy."""
    def __init__(self, C_in=1024, C1=256, C2=128, T=8,
                 target_rate=0.75, temperature=0.5):
        super().__init__()
        self.T = T
        self.encoder = Encoder(C_in, C1, C2)
        self.importance_scorer = LearnedImportanceScorer(C_in=C2, hidden=32)
        self.block_mask = LearnedBlockMask(target_rate, temperature)
        self.decoder = Decoder(C_in, C1, C2, T)
        self.channel = BSC_Channel()
    
    def forward(self, feat, bit_error_rate=0.0, target_rate_override=None):
        all_S2, m1, m2 = [], None, None
        for t in range(self.T):
            s2, m1, m2 = self.encoder(feat, m1, m2)
            all_S2.append(s2)
        
        # Learned importance instead of entropy
        importance = self.importance_scorer(all_S2)  # (B, H, W)
        
        if target_rate_override is not None:
            old_rate = self.block_mask.target_rate
            self.block_mask.target_rate = target_rate_override
            mask, tx = self.block_mask(importance, training=False)
            self.block_mask.target_rate = old_rate
        else:
            mask, tx = self.block_mask(importance, training=self.training)
        
        recv = [self.channel(self.block_mask.apply_mask(all_S2[t], mask), bit_error_rate)
                for t in range(self.T)]
        Fp = self.decoder(recv, mask)
        return Fp, all_S2, tx, {'tx_rate': tx.item(), 'importance': importance, 'mask': mask}


# ############################################################################
# EVALUATION
# ############################################################################

def evaluate(front, model, back, loader, ber=0.0):
    front.eval(); model.eval(); back.eval()
    correct, total, tx_sum, n = 0, 0, 0.0, 0
    with torch.no_grad():
        for img, lab in loader:
            img, lab = img.to(device), lab.to(device)
            Fp, _, _, stats = model(front(img), bit_error_rate=ber)
            correct += back(Fp).argmax(1).eq(lab).sum().item()
            total += lab.size(0); tx_sum += stats['tx_rate']; n += 1
    return 100.*correct/total, tx_sum/max(n,1)


# ############################################################################
# MAIN
# ############################################################################

if __name__ == "__main__":
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize((0.5071,.4867,.4408),(.2675,.2565,.2761))])
    test_tf = T.Compose([T.ToTensor(), T.Normalize((0.5071,.4867,.4408),(.2675,.2565,.2761))])
    train_ds = torchvision.datasets.CIFAR100("./data", True, download=True, transform=train_tf)
    test_ds = torchvision.datasets.CIFAR100("./data", False, download=True, transform=test_tf)
    train_loader = DataLoader(train_ds, 64, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, 128, shuffle=False, num_workers=4, pin_memory=True)

    # Load backbone
    front = ResNet50Front_L3().to(device)
    back = ResNet50Back_L3(100).to(device)
    bb_state = torch.load(os.path.join(BACKBONE_DIR, "backbone_best.pth"), map_location=device)
    front_keys = ['conv1.', 'bn1.', 'layer1.', 'layer2.', 'layer3.']
    back_keys = ['layer4.', 'fc.']
    front.load_state_dict({k: v for k, v in bb_state.items()
                           if any(k.startswith(p) for p in front_keys)}, strict=False)
    back.load_state_dict({k: v for k, v in bb_state.items()
                          if any(k.startswith(p) for p in back_keys)}, strict=False)
    front.eval()
    print("✓ Backbone loaded (Layer3)")

    TARGET_RATE = 0.75
    model = SpikeAdaptSC_LearnedImp(
        C_in=1024, C1=256, C2=128, T=8,
        target_rate=TARGET_RATE, temperature=0.5
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    n_scorer = sum(p.numel() for p in model.importance_scorer.parameters())
    print(f"  Total params: {n_params:,}")
    print(f"  Scorer params: {n_scorer:,} ({100*n_scorer/n_params:.1f}%)")

    # ==================================================================
    # STEP 2: Train with CE + rate loss (NO entropy loss)
    # ==================================================================
    print("\n" + "="*60)
    print(f"LEARNED IMPORTANCE — S2 (target_rate={TARGET_RATE})")
    print("="*60)

    for p in front.parameters(): p.requires_grad = False
    for p in back.parameters(): p.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60, eta_min=1e-6)

    LAMBDA_RATE = 2.0  # rate regularization weight

    best_s2, no_improve = 0.0, 0
    for ep in range(60):
        model.train()
        correct, total, ep_loss, ep_tx = 0, 0, 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"LI S2 E{ep+1}/60")
        for img, lab in pbar:
            img, lab = img.to(device), lab.to(device)
            ber = random.uniform(0, 0.3)
            with torch.no_grad(): feat = front(img)
            Fp, all_S2, tx, stats = model(feat, bit_error_rate=ber)
            out = back(Fp)

            # CE + rate loss only (no entropy loss!)
            ce_loss = criterion(out, lab)
            rate_loss = (tx - TARGET_RATE).abs()
            loss = ce_loss + LAMBDA_RATE * rate_loss

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            _, pred = out.max(1)
            total += lab.size(0); correct += pred.eq(lab).sum().item()
            ep_loss += loss.item(); ep_tx += stats['tx_rate']
            pbar.set_postfix({'L': f'{ce_loss.item():.3f}',
                              'tx': f'{stats["tx_rate"]:.2f}',
                              'A': f'{100.*pred.eq(lab).sum().item()/lab.size(0):.0f}%'})
        sched.step()
        avg_tx = ep_tx / len(train_loader)
        if (ep + 1) % 5 == 0:
            acc, tx = evaluate(front, model, back, test_loader, ber=0.0)
            acc_n, _ = evaluate(front, model, back, test_loader, ber=0.1)
            print(f"  Test: {acc:.2f}% (BER=0), {acc_n:.2f}% (BER=0.1), Tx={tx:.3f}")
            if acc > best_s2:
                best_s2 = acc; no_improve = 0
                torch.save(model.state_dict(), os.path.join(SNAP_DIR, f"li_s2_best_{acc:.2f}.pth"))
                print(f"  ✓ Best: {acc:.2f}%")
            else:
                no_improve += 5
            if no_improve >= 25: print("  ⚠ Early stop"); break

    s2f = sorted([f for f in os.listdir(SNAP_DIR) if f.startswith("li_s2_best_")])
    if s2f:
        model.load_state_dict(torch.load(os.path.join(SNAP_DIR, s2f[-1]), map_location=device))

    # ==================================================================
    # STEP 3: Fine-tune head
    # ==================================================================
    print("\n" + "="*60)
    print("STEP 3: Fine-tune (Learned Importance)")
    print("="*60)

    for p in back.parameters(): p.requires_grad = True
    params_s3 = list(back.parameters()) + list(model.parameters())
    opt_s3 = optim.Adam(params_s3, lr=1e-5, weight_decay=1e-4)
    sched_s3 = optim.lr_scheduler.CosineAnnealingLR(opt_s3, T_max=40, eta_min=1e-7)
    train_loader_s3 = DataLoader(train_ds, 32, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    best_s3, no_improve = 0.0, 0
    for ep in range(40):
        model.train(); back.train()
        pbar = tqdm(train_loader_s3, desc=f"LI S3 E{ep+1}/40")
        opt_s3.zero_grad()
        for step, (img, lab) in enumerate(pbar):
            img, lab = img.to(device), lab.to(device)
            ber = random.uniform(0, 0.3)
            with torch.no_grad(): feat = front(img)
            Fp, _, tx, stats = model(feat, bit_error_rate=ber)
            ce_loss = criterion(back(Fp), lab)
            rate_loss = (tx - TARGET_RATE).abs()
            loss = (ce_loss + LAMBDA_RATE * rate_loss) / 2
            loss.backward()
            if (step + 1) % 2 == 0:
                nn.utils.clip_grad_norm_(params_s3, max_norm=1.0)
                opt_s3.step(); opt_s3.zero_grad()
            pbar.set_postfix({'L': f'{ce_loss.item():.3f}', 'tx': f'{stats["tx_rate"]:.2f}'})
        sched_s3.step()
        if (ep + 1) % 5 == 0:
            acc, tx = evaluate(front, model, back, test_loader, ber=0.0)
            acc_n, _ = evaluate(front, model, back, test_loader, ber=0.1)
            print(f"  S3 E{ep+1}: {acc:.2f}% (BER=0), {acc_n:.2f}% (BER=0.1), Tx={tx:.3f}")
            if acc > best_s3:
                best_s3 = acc; no_improve = 0
                torch.save({'model': model.state_dict(), 'back': back.state_dict()},
                           os.path.join(SNAP_DIR, f"li_s3_best_{acc:.2f}.pth"))
                print(f"  ✓ Best: {acc:.2f}%")
            else:
                no_improve += 5
            if no_improve >= 25: print("  ⚠ Early stop"); break

    # ==================================================================
    # FINAL EVALUATION
    # ==================================================================
    print("\n" + "="*60)
    print("LEARNED IMPORTANCE — FINAL RESULTS")
    print("="*60)

    s3f = sorted([f for f in os.listdir(SNAP_DIR) if f.startswith("li_s3_best_")])
    if s3f:
        ckpt = torch.load(os.path.join(SNAP_DIR, s3f[-1]), map_location=device)
        model.load_state_dict(ckpt['model']); back.load_state_dict(ckpt['back'])

    # BER sweep
    print("\nBER Sweep:")
    li_results = []
    for ber in [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        accs = [evaluate(front, model, back, test_loader, ber=ber)[0]
                for _ in range(10 if ber > 0 else 1)]
        m, s = np.mean(accs), np.std(accs)
        li_results.append({'ber': ber, 'mean': m, 'std': s})
        print(f"  BER={ber:.3f}: {m:.2f}% ±{s:.2f}")

    # Rate sweep
    print("\nRate Sweep:")
    for rate in [1.0, 0.9, 0.8, 0.75, 0.6, 0.5, 0.4, 0.25]:
        model.eval()
        correct, total, tx_sum, n = 0, 0, 0.0, 0
        with torch.no_grad():
            for img, lab in test_loader:
                img, lab = img.to(device), lab.to(device)
                Fp, _, _, st = model(front(img), bit_error_rate=0.0,
                                      target_rate_override=rate)
                correct += back(Fp).argmax(1).eq(lab).sum().item()
                total += lab.size(0); tx_sum += st['tx_rate']; n += 1
        print(f"  Rate={rate:.2f}: {100.*correct/total:.2f}%, Tx={tx_sum/n:.3f}")

    # Mask uniqueness
    print("\nMask uniqueness:")
    hashes = []
    with torch.no_grad():
        for img, lab in test_loader:
            img = img.to(device)
            _, _, _, st = model(front(img), bit_error_rate=0.0)
            mask = st['mask'].squeeze(1)
            for i in range(mask.shape[0]):
                hashes.append(mask[i].cpu().numpy().tobytes())
    unique = len(set(hashes))
    print(f"  {unique} unique masks / {len(hashes)} images")

    # Comparison
    L3_full = {0.0: 74.50, 0.05: 74.60, 0.1: 74.66, 0.15: 74.50, 0.2: 73.97, 0.25: 72.72, 0.3: 69.32}
    print("\n" + "="*60)
    print("COMPARISON TABLE")
    print("="*60)
    print(f"{'BER':<8} {'Learned':<12} {'Entropy':<12} {'Δ':<8}")
    print("-"*40)
    for r in li_results:
        full = L3_full.get(r['ber'], 0)
        if full > 0:
            print(f"{r['ber']:<8.3f} {r['mean']:<12.2f} {full:<12.2f} {r['mean']-full:+.2f}")

    print(f"\nLearned Imp S2 Best: {best_s2:.2f}%")
    print(f"Learned Imp S3 Best: {best_s3:.2f}%")
    print(f"Entropy L3 S3 Best:  74.50%")
    print(f"Unique masks:        {unique}")
    print(f"Scorer params:       {n_scorer:,}")

    with open(os.path.join(SNAP_DIR, "learned_imp_results.json"), 'w') as f:
        json.dump({'li_results': li_results, 'best_s2': best_s2, 'best_s3': best_s3,
                   'unique_masks': unique, 'scorer_params': n_scorer}, f, indent=2)
    print(f"\n✅ LEARNED IMPORTANCE TRAINING COMPLETE!")
