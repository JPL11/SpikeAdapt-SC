# ============================================================================
# STEP 1 OF 2: Train Baselines + BER Comparison
# ============================================================================
# Run: python train_baselines.py
#
# Prereqs: trained backbone at ./snapshots_spikeadapt/backbone_best.pth
#          trained SpikeAdapt-SC at ./snapshots_spikeadapt/step3_best_*.pth
#
# Trains:  CNN-Uni, CNN-Bern, SNN-SC(T=8), SNN-SC(T=6), RandomMask
# Evals:   All above + SpikeAdapt-SC + JPEG+Conv on BER sweep
# Output:  ./eval_results/ber_comparison_all.png (the money plot)
#          ./eval_results/baseline_results.json
# ============================================================================

import os, sys, math, random, json
import numpy as np
from io import BytesIO
from PIL import Image
from tqdm import tqdm

os.environ['PYTHONUNBUFFERED'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

SNAP_DIR = "./snapshots_spikeadapt/"
BASE_DIR = "./snapshots_baselines/"
EVAL_DIR = "./eval_results/"
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)


# ############################################################################
# BUG-FIXED CORE COMPONENTS
# ############################################################################

class SpikeFunction(torch.autograd.Function):
    """Heaviside fwd, sigmoid surrogate bwd. FIXED: threshold gradient."""
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


class IFNeuron(nn.Module):
    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = threshold
    def forward(self, x, mem=None):
        if mem is None:
            mem = torch.zeros_like(x)
        mem = mem + x
        spike = SpikeFunction.apply(mem, self.threshold)
        mem = mem - spike * self.threshold
        return spike, mem


class IHFNeuron(nn.Module):
    """FIXED: no .item() — threshold stays in computation graph."""
    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(float(threshold)))
    def forward(self, x, mem=None):
        if mem is None:
            mem = torch.zeros_like(x)
        mem = mem + x
        spike = SpikeFunction.apply(mem, self.threshold)
        mem = mem - spike * self.threshold
        return spike, mem


class BSC_Channel(nn.Module):
    """FIXED: always apply noise when BER>0, even in eval mode."""
    def forward(self, x, ber):
        if ber <= 0:
            return x
        flip = (torch.rand_like(x.float()) < ber).float()
        x_noisy = (x + flip) % 2
        if self.training:
            return x + (x_noisy - x).detach()  # STE
        else:
            return x_noisy


class BEC_Channel(nn.Module):
    """FIXED: always apply noise when erasure_rate>0."""
    def forward(self, x, er):
        if er <= 0:
            return x
        erase = (torch.rand_like(x.float()) < er).float()
        rand_bits = (torch.rand_like(x.float()) > 0.5).float()
        x_noisy = x * (1 - erase) + rand_bits * erase
        if self.training:
            return x + (x_noisy - x).detach()
        else:
            return x_noisy


# ############################################################################
# BACKBONE (same as your train_spikeadapt_sc.py)
# ############################################################################

class ResNet50Front(nn.Module):
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet50(weights=None)
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)  # CIFAR
        self.bn1 = r.bn1
        self.relu = r.relu
        self.maxpool = nn.Identity()  # CIFAR
        self.layer1 = r.layer1
        self.layer2 = r.layer2
        self.layer3 = r.layer3
        self.layer4 = r.layer4
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


class ResNet50Back(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)
    def forward(self, x):
        return self.fc(torch.flatten(self.avgpool(x), 1))


# ############################################################################
# YOUR SPIKEADAPT-SC (bug-fixed version for loading checkpoint)
# ############################################################################

class SpikeRateEntropyEstimator(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
    def forward(self, all_S2):
        stacked = torch.stack(all_S2, dim=0)
        fr = stacked.mean(dim=0).mean(dim=1)
        f = fr.clamp(self.eps, 1.0 - self.eps)
        ent = -f * torch.log2(f) - (1 - f) * torch.log2(1 - f)
        return ent, fr

class BlockMask(nn.Module):
    def __init__(self, eta=0.5, temperature=0.1):
        super().__init__()
        self.eta = eta
        self.temperature = temperature
    def forward(self, entropy_map, training=True):
        if training:
            soft = torch.sigmoid((entropy_map - self.eta) / self.temperature)
            hard = (entropy_map >= self.eta).float()
            mask = hard + (soft - soft.detach())
        else:
            mask = (entropy_map >= self.eta).float()
        mask = mask.unsqueeze(1)
        return mask, mask.mean()
    def apply_mask(self, x, mask):
        return x * mask

class Encoder(nn.Module):
    def __init__(self, C_in=2048, C1=256, C2=128):
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
    def __init__(self, C_out=2048, C1=256, C2=128, T=8):
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
    def __init__(self, C_in=2048, C1=256, C2=128, T=8,
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
        return Fp, all_S2, tx, {'tx_rate': tx.item(),
                                 'compression_ratio': ob / max(tb, 1)}


# ############################################################################
# BASELINE 1: SNN-SC (Fixed Rate) — your direct predecessor
# ############################################################################

class SNNSC(nn.Module):
    """Original SNN-SC: same arch as SpikeAdapt but NO masking."""
    def __init__(self, C_in=2048, C1=256, C2=128, T=8):
        super().__init__()
        self.T = T
        self.encoder = Encoder(C_in, C1, C2)
        self.decoder = Decoder(C_in, C1, C2, T)
        self.channel = BSC_Channel()

    def forward(self, feat, bit_error_rate=0.0):
        all_S2, m1, m2 = [], None, None
        for t in range(self.T):
            s2, m1, m2 = self.encoder(feat, m1, m2)
            all_S2.append(s2)
        recv = []
        for t in range(self.T):
            recv.append(self.channel(all_S2[t], bit_error_rate))
        # Decoder with all-ones mask (no masking)
        ones_mask = torch.ones(feat.shape[0], 1,
                               all_S2[0].shape[2], all_S2[0].shape[3],
                               device=feat.device)
        Fp = self.decoder(recv, ones_mask)
        return Fp, all_S2


# ############################################################################
# BASELINE 2: CNN-Uni — Uniform Quantization [ref 17]
# ############################################################################

class UniformQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, n_bits):
        x_min, x_max = x.min(), x.max()
        x_norm = (x - x_min) / (x_max - x_min + 1e-8)
        n_levels = 2 ** n_bits
        x_q = torch.round(x_norm * (n_levels - 1)) / (n_levels - 1)
        ctx.save_for_backward(x)
        ctx.x_min, ctx.x_max = x_min.item(), x_max.item()
        return x_q * (x_max - x_min) + x_min
    @staticmethod
    def backward(ctx, g):
        return g, None

class CNNUni(nn.Module):
    """DNN encoder/decoder + uniform quantization → BSC → dequantize."""
    def __init__(self, C_in=2048, C1=256, C2=128, n_bits=8):
        super().__init__()
        self.n_bits = n_bits
        # Encoder
        self.enc1 = nn.Conv2d(C_in, C1, 3, 1, 1)
        self.ebn1 = nn.BatchNorm2d(C1)
        self.enc2 = nn.Conv2d(C1, C2, 3, 1, 1)
        self.ebn2 = nn.BatchNorm2d(C2)
        # Decoder
        self.dec1 = nn.Conv2d(C2, C1, 3, 1, 1)
        self.dbn1 = nn.BatchNorm2d(C1)
        self.dec2 = nn.Conv2d(C1, C_in, 3, 1, 1)
        self.dbn2 = nn.BatchNorm2d(C_in)
        self.channel = BSC_Channel()

    def forward(self, feat, bit_error_rate=0.0):
        # Encode (float)
        x = F.relu(self.ebn1(self.enc1(feat)))
        x = F.relu(self.ebn2(self.enc2(x)))
        # Quantize
        x_q = UniformQuantizeSTE.apply(x, self.n_bits)
        # To bits
        x_min, x_max = x_q.min(), x_q.max()
        x_norm = (x_q - x_min) / (x_max - x_min + 1e-8)
        x_int = torch.round(x_norm * (2**self.n_bits - 1)).long()
        bits = []
        for b in range(self.n_bits):
            bits.append(((x_int >> b) & 1).float())
        bits_stk = torch.stack(bits, dim=-1)  # (B,C2,H,W,n_bits)
        # Channel on each bit plane
        shape = bits_stk.shape
        bits_flat = bits_stk.reshape(shape[0], -1)
        bits_noisy = self.channel(bits_flat, bit_error_rate)
        bits_recv = bits_noisy.reshape(shape)
        # Dequantize
        x_int_r = torch.zeros_like(bits_recv[..., 0])
        for b in range(self.n_bits):
            x_int_r = x_int_r + bits_recv[..., b] * (2 ** b)
        x_r = x_int_r / (2**self.n_bits - 1) * (x_max - x_min) + x_min
        # Decode
        x_r = F.relu(self.dbn1(self.dec1(x_r)))
        Fp = self.dbn2(self.dec2(x_r))
        return Fp


# ############################################################################
# BASELINE 3: CNN-Bern — Bernoulli Sampling [ref 22]
# ############################################################################

class BernoulliSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p):
        return torch.bernoulli(p)
    @staticmethod
    def backward(ctx, g):
        return g

class CNNBern(nn.Module):
    """DNN encoder → sigmoid → Bernoulli sample → BSC → decoder.
    Single-sample version: sends ONE Bernoulli sample through BSC.
    This is the fair comparison matching CNN-Uni bandwidth.
    """
    def __init__(self, C_in=2048, C1=256, C2=128):
        super().__init__()
        self.enc1 = nn.Conv2d(C_in, C1, 3, 1, 1)
        self.ebn1 = nn.BatchNorm2d(C1)
        self.enc2 = nn.Conv2d(C1, C2, 3, 1, 1)
        self.ebn2 = nn.BatchNorm2d(C2)
        self.dec1 = nn.Conv2d(C2, C1, 3, 1, 1)
        self.dbn1 = nn.BatchNorm2d(C1)
        self.dec2 = nn.Conv2d(C1, C_in, 3, 1, 1)
        self.dbn2 = nn.BatchNorm2d(C_in)
        self.channel = BSC_Channel()

    def forward(self, feat, bit_error_rate=0.0):
        x = F.relu(self.ebn1(self.enc1(feat)))
        prob = torch.sigmoid(self.ebn2(self.enc2(x)))
        bits = BernoulliSTE.apply(prob)
        bits_noisy = self.channel(bits, bit_error_rate)
        x = F.relu(self.dbn1(self.dec1(bits_noisy)))
        Fp = self.dbn2(self.dec2(x))
        return Fp


# ############################################################################
# BASELINE 4: JPEG + Convolutional Code
# ############################################################################

class JPEGConv(nn.Module):
    """
    Traditional: quantize → JPEG → repetition code → BSC → decode.
    Shows cliff effect at high BER.
    """
    def __init__(self, jpeg_quality=50, code_rate_inv=3):
        super().__init__()
        self.Q = jpeg_quality
        self.R = code_rate_inv

    def forward(self, feat, bit_error_rate=0.0):
        B, C, H, W = feat.shape
        results = []
        for b in range(B):
            f = feat[b]
            fmin, fmax = f.min().item(), f.max().item()
            fn = ((f - fmin) / (fmax - fmin + 1e-8) * 255).clamp(0, 255).byte()
            # Tile to 2D
            nr = int(math.ceil(math.sqrt(C)))
            nc = int(math.ceil(C / nr))
            canvas = torch.zeros(nr * H, nc * W, dtype=torch.uint8)
            for c in range(C):
                r, co = divmod(c, nc)
                canvas[r*H:(r+1)*H, co*W:(co+1)*W] = fn[c].cpu()
            # JPEG
            img = Image.fromarray(canvas.numpy(), 'L')
            buf = BytesIO()
            img.save(buf, format='JPEG', quality=self.Q)
            jpeg_bytes = buf.getvalue()
            # Channel coding + BSC
            if bit_error_rate > 0:
                src = np.unpackbits(np.frombuffer(jpeg_bytes, dtype=np.uint8))
                coded = np.repeat(src, self.R)
                flip = (np.random.random(len(coded)) < bit_error_rate)
                coded_n = (coded + flip.astype(int)) % 2
                decoded = (coded_n.reshape(-1, self.R).sum(1) > self.R / 2).astype(np.uint8)
                pad = (8 - len(decoded) % 8) % 8
                dec_bytes = np.packbits(np.concatenate([decoded,
                                        np.zeros(pad, dtype=np.uint8)])).tobytes()[:len(jpeg_bytes)]
                try:
                    ri = Image.open(BytesIO(dec_bytes)).convert('L')
                    rc = torch.tensor(np.array(ri), dtype=torch.float32)
                    # Check decoded image has correct dimensions
                    if rc.shape != (nr * H, nc * W):
                        rc = torch.zeros(nr * H, nc * W)
                except Exception:
                    rc = torch.zeros(nr * H, nc * W)
            else:
                ri = Image.open(BytesIO(jpeg_bytes)).convert('L')
                rc = torch.tensor(np.array(ri), dtype=torch.float32)
            # Untile
            rf = torch.zeros(C, H, W)
            for c in range(C):
                r, co = divmod(c, nc)
                try:
                    rf[c] = rc[r*H:(r+1)*H, co*W:(co+1)*W]
                except Exception:
                    pass  # Leave as zeros
            rf = rf / 255.0 * (fmax - fmin) + fmin
            results.append(rf)
        return torch.stack(results).to(feat.device)


# ############################################################################
# BASELINE 5: Random Mask Ablation
# ############################################################################

class SpikeAdaptSC_RandomMask(nn.Module):
    """Same as SpikeAdapt but drops random blocks instead of entropy-guided."""
    def __init__(self, C_in=2048, C1=256, C2=128, T=8, keep_rate=0.75):
        super().__init__()
        self.T = T
        self.keep_rate = keep_rate
        self.encoder = Encoder(C_in, C1, C2)
        self.decoder = Decoder(C_in, C1, C2, T)
        self.channel = BSC_Channel()

    def forward(self, feat, bit_error_rate=0.0):
        all_S2, m1, m2 = [], None, None
        for t in range(self.T):
            s2, m1, m2 = self.encoder(feat, m1, m2)
            all_S2.append(s2)
        B, C2, H2, W2 = all_S2[0].shape
        mask = (torch.rand(B, 1, H2, W2, device=feat.device) < self.keep_rate).float()
        recv = []
        for t in range(self.T):
            recv.append(self.channel(all_S2[t] * mask, bit_error_rate))
        Fp = self.decoder(recv, mask)
        return Fp, all_S2


# ############################################################################
# LOSS FUNCTIONS
# ############################################################################

def entropy_loss(all_S2, alpha=1.0):
    """SNN-SC Eq 14: maximize entropy of semantic info."""
    bits = torch.cat([s.flatten() for s in all_S2])
    p1 = bits.mean()
    p0 = 1.0 - p1
    eps = 1e-7
    H = -(torch.clamp(p0, eps, 1-eps) * torch.log2(torch.clamp(p0, eps, 1-eps)) +
          torch.clamp(p1, eps, 1-eps) * torch.log2(torch.clamp(p1, eps, 1-eps)))
    return (alpha - H) ** 2


# ############################################################################
# GENERIC TRAINING LOOP
# ############################################################################

def train_model(model, front, back, train_loader, test_loader, name,
                epochs=50, lr=1e-4, ber_max=0.3, use_entropy_loss=False,
                finetune_head=True, ft_epochs=20, ft_lr=1e-5):
    """
    Train any SC baseline with optional Step 3 head fine-tuning.
    use_entropy_loss=True for SNN-SC variants (per SNN-SC paper Eq 16).
    use_entropy_loss=False for CNN variants (can't differentiate through quant).
    finetune_head=True to add Step 3 fine-tuning of classifier head.
    """
    print(f"\n{'='*60}\nTraining: {name} (Step 2: SC module, ber_max={ber_max})\n{'='*60}")
    front.eval()
    for p in front.parameters():
        p.requires_grad = False
    for p in back.parameters():
        p.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    best = 0.0
    for ep in range(epochs):
        model.train()
        correct, total, ep_loss = 0, 0, 0.0
        pbar = tqdm(train_loader, desc=f"{name} E{ep+1}/{epochs}")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            ber = random.uniform(0, ber_max)
            with torch.no_grad():
                feat = front(images)

            # Forward — handle different return types
            if isinstance(model, (SNNSC, SpikeAdaptSC_RandomMask)):
                Fp, all_S2 = model(feat, bit_error_rate=ber)
            elif isinstance(model, (CNNUni, CNNBern)):
                Fp = model(feat, bit_error_rate=ber)
                all_S2 = None
            else:
                raise ValueError(f"Unknown: {type(model)}")

            out = back(Fp)
            loss = criterion(out, labels)

            # SNN-SC uses entropy loss; CNN baselines don't
            if use_entropy_loss and all_S2 is not None:
                loss = loss + entropy_loss(all_S2, alpha=1.0)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            _, pred = out.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
            ep_loss += loss.item()
            pbar.set_postfix({'L': f'{loss.item():.3f}',
                              'A': f'{100.*pred.eq(labels).sum().item()/labels.size(0):.0f}%'})

        sched.step()
        if (ep + 1) % 10 == 0:
            acc = eval_model(model, front, back, test_loader, ber=0.0, name=name)
            print(f"  {name} E{ep+1}: TestAcc={acc:.2f}% (BER=0)")
            if acc > best:
                best = acc
                torch.save(model.state_dict(), os.path.join(BASE_DIR, f"{name}.pth"))
                print(f"  ✓ Best: {best:.2f}%")

    # Reload best SC module
    ckpt_path = os.path.join(BASE_DIR, f"{name}.pth")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # ---- Step 3: Fine-tune classifier head ----
    if finetune_head:
        print(f"  → Step 3: Fine-tuning head for {name} ({ft_epochs} epochs)")
        for p in back.parameters():
            p.requires_grad = True
        ft_params = list(model.parameters()) + list(back.parameters())
        ft_opt = optim.Adam(ft_params, lr=ft_lr, weight_decay=1e-4)
        ft_sched = optim.lr_scheduler.CosineAnnealingLR(ft_opt, T_max=ft_epochs, eta_min=1e-7)

        best_ft = best
        for ep in range(ft_epochs):
            model.train(); back.train()
            pbar = tqdm(train_loader, desc=f"{name} FT{ep+1}/{ft_epochs}", leave=False)
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                ber = random.uniform(0, ber_max)
                with torch.no_grad():
                    feat = front(images)

                if isinstance(model, (SNNSC, SpikeAdaptSC_RandomMask)):
                    Fp, all_S2 = model(feat, bit_error_rate=ber)
                elif isinstance(model, (CNNUni, CNNBern)):
                    Fp = model(feat, bit_error_rate=ber)
                    all_S2 = None
                else:
                    Fp = model(feat, bit_error_rate=ber)
                    all_S2 = None

                out = back(Fp)
                loss = criterion(out, labels)
                if use_entropy_loss and all_S2 is not None:
                    loss = loss + entropy_loss(all_S2, alpha=1.0)

                ft_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ft_params, max_norm=1.0)
                ft_opt.step()
            ft_sched.step()

            if (ep + 1) % 5 == 0:
                acc = eval_model(model, front, back, test_loader, ber=0.0, name=name)
                print(f"    {name} FT{ep+1}: {acc:.2f}%")
                if acc > best_ft:
                    best_ft = acc
                    torch.save({'model': model.state_dict(), 'back': back.state_dict()},
                               os.path.join(BASE_DIR, f"{name}_ft.pth"))

        # Reload best fine-tuned
        ft_path = os.path.join(BASE_DIR, f"{name}_ft.pth")
        if os.path.exists(ft_path):
            ckpt = torch.load(ft_path, map_location=device)
            model.load_state_dict(ckpt['model'])
            back.load_state_dict(ckpt['back'])
        best = max(best, best_ft)
        for p in back.parameters():
            p.requires_grad = False

    print(f"✅ {name} done. Best={best:.2f}%\n")
    return best


# ############################################################################
# EVALUATION
# ############################################################################

def eval_model(model, front, back, loader, ber=0.0, name=""):
    """Evaluate any model at given BER."""
    model.eval()
    front.eval()
    back.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            feat = front(images)
            if isinstance(model, SpikeAdaptSC):
                Fp, _, _, _ = model(feat, bit_error_rate=ber)
            elif isinstance(model, (SNNSC, SpikeAdaptSC_RandomMask)):
                Fp, _ = model(feat, bit_error_rate=ber)
            elif isinstance(model, (CNNUni, CNNBern)):
                Fp = model(feat, bit_error_rate=ber)
            elif isinstance(model, JPEGConv):
                Fp = model(feat, bit_error_rate=ber)
            else:
                raise ValueError(f"Unknown: {type(model)}")
            out = back(Fp)
            _, pred = out.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    return 100. * correct / total


# ############################################################################
# MAIN
# ############################################################################

if __name__ == "__main__":

    # ---- Dataset ----
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                          T.ToTensor(),
                          T.Normalize((0.5071,.4867,.4408),(.2675,.2565,.2761))])
    test_tf = T.Compose([T.ToTensor(),
                         T.Normalize((0.5071,.4867,.4408),(.2675,.2565,.2761))])
    train_ds = torchvision.datasets.CIFAR100("./data", True, download=True, transform=train_tf)
    test_ds = torchvision.datasets.CIFAR100("./data", False, download=True, transform=test_tf)
    train_loader = DataLoader(train_ds, 64, shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, 128, shuffle=False, num_workers=4, pin_memory=True)

    # ---- Load backbone ----
    front = ResNet50Front().to(device)
    back = ResNet50Back(100).to(device)
    bb_path = os.path.join(SNAP_DIR, "backbone_best.pth")
    state = torch.load(bb_path, map_location=device)
    f_st = {k: v for k, v in state.items() if not k.startswith(('fc.', 'avgpool.'))}
    b_st = {k: v for k, v in state.items() if k.startswith(('fc.', 'avgpool.'))}
    front.load_state_dict(f_st, strict=False)
    back.load_state_dict(b_st, strict=False)
    front.eval()
    print("✓ Backbone loaded")

    # ---- Verify feature dims ----
    with torch.no_grad():
        d = front(torch.randn(1, 3, 32, 32).to(device))
        print(f"★ Feature shape: {d.shape}")  # Should be (1, 2048, 4, 4)
    C_in, H_feat, W_feat = d.shape[1], d.shape[2], d.shape[3]

    # ---- STEP 0: Backbone accuracy ----
    print("\n" + "="*60)
    print("STEP 0: Backbone accuracy (no SC)")
    back.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for img, lab in test_loader:
            img, lab = img.to(device), lab.to(device)
            out = back(front(img))
            correct += out.argmax(1).eq(lab).sum().item()
            total += lab.size(0)
    bb_acc = 100. * correct / total
    print(f"★ BACKBONE ACCURACY: {bb_acc:.2f}%")
    print(f"  SNN-SC paper: 77.81%")
    print(f"  Your SpikeAdapt-SC: 72.85%")
    print(f"  Gap: {bb_acc - 72.85:+.2f}%")

    # ---- Load your trained SpikeAdapt-SC ----
    spikeadapt = SpikeAdaptSC(C_in=C_in, C1=256, C2=128, T=8,
                               eta=0.5, temperature=0.1).to(device)
    s3_files = sorted([f for f in os.listdir(SNAP_DIR) if f.startswith("step3_best_")])
    if s3_files:
        ckpt = torch.load(os.path.join(SNAP_DIR, s3_files[-1]), map_location=device)
        spikeadapt.load_state_dict(ckpt['spikeadapt'])
        back.load_state_dict(ckpt['back'])
        print(f"✓ Loaded SpikeAdapt-SC: {s3_files[-1]}")
    else:
        s2_files = sorted([f for f in os.listdir(SNAP_DIR) if f.startswith("spikeadapt_best_")])
        if s2_files:
            spikeadapt.load_state_dict(torch.load(os.path.join(SNAP_DIR, s2_files[-1]),
                                                    map_location=device))
            print(f"✓ Loaded SpikeAdapt-SC (S2): {s2_files[-1]}")
    spikeadapt.eval()
    # Save a reference to the fine-tuned back
    back_sa = ResNet50Back(100).to(device)
    back_sa.load_state_dict(back.state_dict())

    # ---- TRAIN BASELINES ----
    # Each gets its own back (reset from backbone weights)
    def fresh_back():
        b = ResNet50Back(100).to(device)
        b.load_state_dict(b_st, strict=False)
        return b

    # 1. SNN-SC T=8 (direct predecessor, uses entropy loss, trained with noise)
    snnsc_t8 = SNNSC(C_in, 256, 128, T=8).to(device)
    back_snn8 = fresh_back()
    train_model(snnsc_t8, front, back_snn8, train_loader, test_loader,
                "SNNSC_T8", epochs=50, ber_max=0.3, use_entropy_loss=True)

    # 2. SNN-SC T=6 (bandwidth-fair: 6×16×128=12288 ≈ your 12304)
    snnsc_t6 = SNNSC(C_in, 256, 128, T=6).to(device)
    back_snn6 = fresh_back()
    train_model(snnsc_t6, front, back_snn6, train_loader, test_loader,
                "SNNSC_T6", epochs=50, ber_max=0.3, use_entropy_loss=True)

    # 3. CNN-Uni (uniform quantization, NO entropy loss)
    # FIXED: trained with BER=0 per SNN-SC paper — CNN can't handle channel
    # noise during training with multi-bit quantization
    cnn_uni = CNNUni(C_in, 256, 128, n_bits=8).to(device)
    back_uni = fresh_back()
    train_model(cnn_uni, front, back_uni, train_loader, test_loader,
                "CNN_Uni", epochs=50, ber_max=0.0, use_entropy_loss=False)

    # 4. CNN-Bern (single Bernoulli sample, NO entropy loss)
    # FIXED: single sample (not T=8 averaged), trained with BER=0
    cnn_bern = CNNBern(C_in, 256, 128).to(device)
    back_bern = fresh_back()
    train_model(cnn_bern, front, back_bern, train_loader, test_loader,
                "CNN_Bern", epochs=50, ber_max=0.0, use_entropy_loss=False)

    # 5. Random Mask ablation (uses entropy loss for SNN training)
    rand_mask = SpikeAdaptSC_RandomMask(C_in, 256, 128, T=8, keep_rate=0.75).to(device)
    back_rand = fresh_back()
    train_model(rand_mask, front, back_rand, train_loader, test_loader,
                "RandomMask", epochs=50, ber_max=0.3, use_entropy_loss=True)

    # 6. JPEG+Conv (no training)
    jpeg_conv = JPEGConv(jpeg_quality=50, code_rate_inv=3)

    # ---- BER SWEEP ----
    print("\n" + "="*60)
    print("BER SWEEP — ALL MODELS")
    print("="*60)

    ber_vals = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    N_REPEAT = 10  # per SNN-SC paper: "repeated 10 times"

    all_models = {
        'SpikeAdapt-SC':   (spikeadapt, back_sa),
        'SNN-SC (T=8)':    (snnsc_t8, back_snn8),
        'SNN-SC (T=6)':    (snnsc_t6, back_snn6),
        'CNN-Uni':         (cnn_uni, back_uni),
        'CNN-Bern':        (cnn_bern, back_bern),
        'Random Mask':     (rand_mask, back_rand),
        'JPEG+Conv':       (jpeg_conv, back_sa),  # Uses same back as SpikeAdapt
    }

    results = {}
    for name, (model, bk) in all_models.items():
        print(f"\n  {name}:")
        res = []
        bk.eval()
        for ber in ber_vals:
            accs = []
            nr = N_REPEAT if ber > 0 else 1
            if name == 'JPEG+Conv':
                nr = min(nr, 3)  # JPEG is slow
            for _ in range(nr):
                a = eval_model(model, front, bk, test_loader, ber=ber, name=name)
                accs.append(a)
            m, s = np.mean(accs), np.std(accs)
            res.append({'ber': ber, 'mean': m, 'std': s})
            print(f"    BER={ber:.3f}: {m:.2f}% ±{s:.2f}")
        results[name] = res

    # Save
    with open(os.path.join(EVAL_DIR, "baseline_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved: {EVAL_DIR}baseline_results.json")

    # ---- THE MONEY PLOT ----
    fig, ax = plt.subplots(figsize=(12, 7))
    styles = {
        'SpikeAdapt-SC':  ('#D32F2F', 's', '-',  2.5),
        'SNN-SC (T=8)':   ('#1976D2', 'o', '-',  2.0),
        'SNN-SC (T=6)':   ('#42A5F5', 'o', '--', 2.0),
        'CNN-Uni':         ('#388E3C', '^', '-.', 2.0),
        'CNN-Bern':        ('#F57C00', 'D', '--', 2.0),
        'JPEG+Conv':       ('#7B1FA2', 'v', ':',  2.0),
        'Random Mask':     ('#9E9E9E', 'x', '--', 1.5),
    }
    for name, res in results.items():
        c, mk, ls, lw = styles.get(name, ('gray', 'o', '-', 1.5))
        bers = [r['ber'] for r in res]
        means = [r['mean'] for r in res]
        stds = [r['std'] for r in res]
        ax.plot(bers, means, marker=mk, ls=ls, color=c, lw=lw, ms=7, label=name)
        if max(stds) > 0.1:
            ax.fill_between(bers, [m-s for m,s in zip(means, stds)],
                           [m+s for m,s in zip(means, stds)], alpha=0.08, color=c)

    ax.set_xlabel('Bit Error Rate (BER)', fontsize=14)
    ax.set_ylabel('Classification Accuracy (%)', fontsize=14)
    ax.set_title('SpikeAdapt-SC vs Baselines on BSC (CIFAR-100, ResNet50)', fontsize=14)
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.01, 0.31)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "ber_comparison_all.png"), dpi=200)
    print(f"✓ Saved: {EVAL_DIR}ber_comparison_all.png")
    plt.close()

    # ---- RESULTS TABLE ----
    print("\n" + "="*60)
    print("RESULTS TABLE (for paper)")
    print("="*60)
    print(f"{'Method':<22} {'BW(bits)':>10} {'CR':>6}", end="")
    for ber in [0.0, 0.1, 0.2, 0.3]:
        print(f" {'BER='+str(ber):>10}", end="")
    print()
    print("-"*75)

    bw_info = {
        'SpikeAdapt-SC':  (f'{8*12*128+16}', '85×'),
        'SNN-SC (T=8)':   (f'{8*16*128}', '64×'),
        'SNN-SC (T=6)':   (f'{6*16*128}', '85×'),
        'CNN-Uni':         (f'{8*128*H_feat*W_feat}', '64×'),
        'CNN-Bern':        (f'{128*H_feat*W_feat}', '512×'),
        'JPEG+Conv':       ('~50K', '~21×'),
        'Random Mask':     (f'~{8*12*128}', '85×'),
    }
    for name in all_models:
        bw, cr = bw_info.get(name, ('?', '?'))
        print(f"{name:<22} {bw:>10} {cr:>6}", end="")
        for ber in [0.0, 0.1, 0.2, 0.3]:
            for r in results[name]:
                if abs(r['ber'] - ber) < 0.001:
                    print(f" {r['mean']:>9.2f}%", end="")
                    break
        print()

    print("\n✅ BASELINES COMPLETE!")
    print("Next: python train_layer3_split.py")
