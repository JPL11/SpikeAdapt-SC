"""Visualize AID images through SpikeAdapt-SC pipeline.

Shows: original image, importance map, block mask, and prediction
for multiple AID test images across different scene classes.
"""

import os, sys, random, json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SNAP_DIR = "./snapshots_aid/"
OUT_DIR = "./figures/"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Copy model classes from train_aid.py ----

class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane, threshold):
        if isinstance(threshold, (int, float)):
            threshold = torch.tensor(float(threshold), device=membrane.device)
        ctx.save_for_backward(membrane, threshold)
        ctx.th_needs_grad = isinstance(threshold, torch.Tensor) and threshold.requires_grad
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
        return x_n

class AWGN_Channel(nn.Module):
    def forward(self, x, snr_db):
        if snr_db >= 100: return x
        bpsk = 2.0 * x - 1.0
        snr_linear = 10 ** (snr_db / 10.0)
        noise_std = 1.0 / math.sqrt(2 * snr_linear)
        decoded = (bpsk + torch.randn_like(bpsk) * noise_std > 0).float()
        return decoded

class Rayleigh_Channel(nn.Module):
    def forward(self, x, snr_db):
        if snr_db >= 100: return x
        bpsk = 2.0 * x - 1.0
        snr_linear = 10 ** (snr_db / 10.0)
        h_real = torch.randn_like(bpsk) / math.sqrt(2)
        h_imag = torch.randn_like(bpsk) / math.sqrt(2)
        h_mag = torch.sqrt(h_real**2 + h_imag**2)
        noise_std = 1.0 / math.sqrt(2 * snr_linear)
        received = h_mag * bpsk + torch.randn_like(bpsk) * noise_std
        decoded = (received / (h_mag + 1e-8) > 0).float()
        return decoded

class LearnedImportanceScorer(nn.Module):
    def __init__(self, C_in=128, hidden=32):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Conv2d(C_in, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1), nn.Sigmoid()
        )
    def forward(self, all_S2):
        return self.scorer(torch.stack(all_S2, dim=0).mean(dim=0)).squeeze(1)

class LearnedBlockMask(nn.Module):
    def __init__(self, target_rate=0.75, temperature=0.5):
        super().__init__()
        self.target_rate = target_rate; self.temperature = temperature
    def forward(self, importance, training=True):
        B, H, W = importance.shape
        if training:
            logits = torch.log(importance / (1 - importance + 1e-7) + 1e-7)
            u = torch.rand_like(logits).clamp(1e-7, 1-1e-7)
            soft = torch.sigmoid((logits - torch.log(-torch.log(u))) / self.temperature)
            hard = (soft > 0.5).float()
            mask = hard + (soft - soft.detach())
        else:
            k = max(1, int(self.target_rate * H * W))
            flat = importance.view(B, -1)
            _, idx = flat.topk(k, dim=1)
            mask = torch.zeros_like(flat); mask.scatter_(1, idx, 1.0)
            mask = mask.view(B, H, W)
        return mask.unsqueeze(1), mask.mean()
    def apply_mask(self, x, mask): return x * mask

class Encoder(nn.Module):
    def __init__(self, C_in=1024, C1=256, C2=128):
        super().__init__()
        self.conv1 = nn.Conv2d(C_in, C1, 3, 1, 1); self.bn1 = nn.BatchNorm2d(C1); self.if1 = IFNeuron()
        self.conv2 = nn.Conv2d(C1, C2, 3, 1, 1); self.bn2 = nn.BatchNorm2d(C2); self.if2 = IFNeuron()
    def forward(self, F, m1=None, m2=None):
        s1, m1 = self.if1(self.bn1(self.conv1(F)), m1)
        s2, m2 = self.if2(self.bn2(self.conv2(s1)), m2)
        return s1, s2, m1, m2

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

class SpikeAdaptSC(nn.Module):
    def __init__(self, C_in=1024, C1=256, C2=128, T=8,
                 target_rate=0.75, channel_type='bsc'):
        super().__init__()
        self.T = T
        self.encoder = Encoder(C_in, C1, C2)
        self.importance_scorer = LearnedImportanceScorer(C_in=C2, hidden=32)
        self.block_mask = LearnedBlockMask(target_rate, 0.5)
        self.decoder = Decoder(C_in, C1, C2, T)
        if channel_type == 'awgn': self.channel = AWGN_Channel()
        elif channel_type == 'rayleigh': self.channel = Rayleigh_Channel()
        else: self.channel = BSC_Channel()
        self.channel_type = channel_type

    def forward(self, feat, noise_param=0.0, target_rate_override=None):
        all_S1, all_S2, m1, m2 = [], [], None, None
        for t in range(self.T):
            s1, s2, m1, m2 = self.encoder(feat, m1, m2)
            all_S1.append(s1); all_S2.append(s2)
        importance = self.importance_scorer(all_S2)
        if target_rate_override is not None:
            old = self.block_mask.target_rate
            self.block_mask.target_rate = target_rate_override
            mask, tx = self.block_mask(importance, training=False)
            self.block_mask.target_rate = old
        else:
            mask, tx = self.block_mask(importance, training=self.training)
        recv = [self.channel(self.block_mask.apply_mask(all_S2[t], mask), noise_param)
                for t in range(self.T)]
        Fp = self.decoder(recv, mask)
        return Fp, {'tx_rate': tx.item(), 'mask': mask, 'importance': importance}

class ResNet50Front(nn.Module):
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        self.conv1 = r.conv1; self.bn1 = r.bn1; self.relu = r.relu
        self.maxpool = r.maxpool
        self.layer1 = r.layer1; self.layer2 = r.layer2; self.layer3 = r.layer3
        self.spatial_pool = nn.AdaptiveAvgPool2d(8)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(self.maxpool(x))))
        return self.spatial_pool(x)

class ResNet50Back(nn.Module):
    def __init__(self, num_classes=30):
        super().__init__()
        r = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        self.layer4 = r.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)
    def forward(self, x):
        return self.fc(torch.flatten(self.avgpool(self.layer4(x)), 1))


class AIDDataset(Dataset):
    def __init__(self, root, transform=None, split='test', train_ratio=0.8):
        self.root = root; self.transform = transform
        aid_dir = os.path.join(root, 'AID')
        classes = sorted([d for d in os.listdir(aid_dir)
                         if os.path.isdir(os.path.join(aid_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        self.classes = classes
        all_samples = []
        for cls_name in classes:
            cls_dir = os.path.join(aid_dir, cls_name)
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
                    all_samples.append((os.path.join(cls_dir, fname),
                                       self.class_to_idx[cls_name]))
        random.seed(42); random.shuffle(all_samples)
        split_idx = int(len(all_samples) * train_ratio)
        self.samples = all_samples[split_idx:] if split == 'test' else all_samples[:split_idx]

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, label, path


# ---- Denormalize for display ----
def denorm(img_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    img = img_tensor.cpu().clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


if __name__ == "__main__":
    print(f"Device: {device}")

    # Load models
    front = ResNet50Front().to(device)
    back = ResNet50Back(30).to(device)

    # Load backbone
    bb_state = torch.load(os.path.join(SNAP_DIR, "backbone_best.pth"), map_location=device)
    front.load_state_dict({k: v for k, v in bb_state.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.'))}, strict=False)
    back.load_state_dict({k: v for k, v in bb_state.items()
                          if k.startswith(('layer4.', 'fc.', 'avgpool.'))}, strict=False)
    front.eval()

    # Load best BSC model (best results)
    model = SpikeAdaptSC(C_in=1024, C1=256, C2=128, T=8,
                          target_rate=0.75, channel_type='bsc').to(device)
    s3_files = sorted([f for f in os.listdir(SNAP_DIR) if f.startswith("bsc_s3_")])
    if s3_files:
        ckpt = torch.load(os.path.join(SNAP_DIR, s3_files[-1]), map_location=device)
        model.load_state_dict(ckpt['model'])
        back.load_state_dict(ckpt['back'])
        print(f"✓ Loaded BSC model: {s3_files[-1]}")
    model.eval(); back.eval()

    # Dataset (no augmentation for visualization)
    test_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                         T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    test_ds = AIDDataset("./data", transform=test_tf, split='test')

    # Pick diverse images — one per class for variety
    class_samples = {}
    for idx in range(len(test_ds)):
        img, lab, path = test_ds[idx]
        cls_name = test_ds.idx_to_class[lab]
        if cls_name not in class_samples:
            class_samples[cls_name] = (img, lab, path)
        if len(class_samples) >= 30:
            break

    # Select 12 interesting aerial classes
    target_classes = ['Airport', 'BaseballField', 'Beach', 'Bridge', 'Desert',
                      'Farmland', 'Forest', 'Harbor', 'Mountain', 'Parking',
                      'River', 'Stadium']
    selected = []
    for c in target_classes:
        if c in class_samples:
            selected.append(class_samples[c])
    # Fill from what we have if needed
    if len(selected) < 12:
        for c, v in class_samples.items():
            if c not in target_classes:
                selected.append(v)
            if len(selected) >= 12:
                break

    print(f"Selected {len(selected)} images from {len(class_samples)} classes")

    # ================================================================
    # FIGURE 1: Pipeline visualization (6 images × 4 columns)
    # ================================================================
    fig = plt.figure(figsize=(20, 18))
    fig.suptitle('SpikeAdapt-SC on AID Aerial Scenes — Pipeline Visualization',
                 fontsize=16, fontweight='bold', y=0.98)

    n_rows = 6
    gs = gridspec.GridSpec(n_rows, 4, hspace=0.35, wspace=0.15,
                           left=0.05, right=0.95, top=0.94, bottom=0.02)

    col_titles = ['Original Image', 'Importance Map', 'Block Mask (75%)', 'Prediction']

    for row_idx in range(min(n_rows, len(selected))):
        img_t, label, path = selected[row_idx]
        cls_name = test_ds.idx_to_class[label]
        img_batch = img_t.unsqueeze(0).to(device)

        with torch.no_grad():
            feat = front(img_batch)
            Fp, stats = model(feat, noise_param=0.0)
            logits = back(Fp)
            probs = F.softmax(logits, dim=1)
            conf, pred_idx = probs.max(1)
            pred_name = test_ds.idx_to_class[pred_idx.item()]

        importance = stats['importance'][0].cpu().numpy()  # (8, 8)
        mask = stats['mask'][0, 0].cpu().numpy()  # (8, 8)

        # Col 0: Original image
        ax = fig.add_subplot(gs[row_idx, 0])
        ax.imshow(denorm(img_t))
        ax.set_title(f'{cls_name}', fontsize=11, fontweight='bold')
        ax.axis('off')

        # Col 1: Importance map
        ax = fig.add_subplot(gs[row_idx, 1])
        im = ax.imshow(importance, cmap='hot', vmin=0, vmax=1,
                       interpolation='nearest')
        ax.set_title(f'Score range: [{importance.min():.2f}, {importance.max():.2f}]',
                     fontsize=9)
        ax.axis('off')

        # Col 2: Block mask overlay on image
        ax = fig.add_subplot(gs[row_idx, 2])
        img_np = denorm(img_t)
        # Upsample mask to image size for overlay
        mask_up = np.kron(mask, np.ones((28, 28)))[:224, :224]
        overlay = img_np.copy()
        overlay[mask_up == 0] = overlay[mask_up == 0] * 0.3 + np.array([1, 0, 0]) * 0.7
        ax.imshow(overlay)
        tx_rate = mask.mean() * 100
        ax.set_title(f'Tx: {tx_rate:.0f}% ({int(mask.sum())}/64 blocks)',
                     fontsize=9)
        ax.axis('off')

        # Col 3: Prediction
        ax = fig.add_subplot(gs[row_idx, 3])
        ax.imshow(denorm(img_t))
        correct = pred_name == cls_name
        color = '#2ecc71' if correct else '#e74c3c'
        symbol = '✓' if correct else '✗'
        ax.set_title(f'{symbol} {pred_name} ({conf.item()*100:.1f}%)',
                     fontsize=11, fontweight='bold', color=color)
        ax.axis('off')

        # Column headers on first row
        if row_idx == 0:
            for col, title in enumerate(col_titles):
                fig.text(0.05 + col * 0.225 + 0.1, 0.95, title,
                         ha='center', fontsize=12, fontweight='bold',
                         fontstyle='italic')

    plt.savefig(os.path.join(OUT_DIR, 'aid_pipeline_viz.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {OUT_DIR}aid_pipeline_viz.png")

    # ================================================================
    # FIGURE 2: BER robustness comparison (same image at different BER)
    # ================================================================
    fig2, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig2.suptitle('SpikeAdapt-SC Robustness: Same Image at Different BER Levels',
                  fontsize=16, fontweight='bold')

    ber_vals = [0.0, 0.05, 0.1, 0.2, 0.3]
    # Pick 3 representative images
    demo_imgs = selected[:3]

    for row, (img_t, label, path) in enumerate(demo_imgs):
        cls_name = test_ds.idx_to_class[label]
        img_batch = img_t.unsqueeze(0).to(device)

        for col, ber in enumerate(ber_vals):
            with torch.no_grad():
                feat = front(img_batch)
                Fp, stats = model(feat, noise_param=ber)
                logits = back(Fp)
                probs = F.softmax(logits, dim=1)
                conf, pred_idx = probs.max(1)
                pred_name = test_ds.idx_to_class[pred_idx.item()]

            mask = stats['mask'][0, 0].cpu().numpy()
            img_np = denorm(img_t)
            mask_up = np.kron(mask, np.ones((28, 28)))[:224, :224]
            overlay = img_np.copy()
            overlay[mask_up == 0] = overlay[mask_up == 0] * 0.3 + np.array([0.5, 0, 0]) * 0.5

            ax = axes[row, col]
            ax.imshow(overlay)
            correct = pred_name == cls_name
            color = '#2ecc71' if correct else '#e74c3c'
            symbol = '✓' if correct else '✗'
            ax.set_title(f'{symbol} {pred_name}\n{conf.item()*100:.1f}%',
                        fontsize=10, color=color, fontweight='bold')
            ax.axis('off')

            if row == 0:
                ax.text(0.5, 1.25, f'BER={ber}', transform=ax.transAxes,
                       ha='center', fontsize=12, fontweight='bold')
        # Row label
        axes[row, 0].text(-0.15, 0.5, cls_name, transform=axes[row, 0].transAxes,
                          ha='center', va='center', fontsize=12, fontweight='bold',
                          rotation=90)

    plt.tight_layout(rect=[0.03, 0, 1, 0.95])
    plt.savefig(os.path.join(OUT_DIR, 'aid_ber_robustness_viz.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {OUT_DIR}aid_ber_robustness_viz.png")

    # ================================================================
    # FIGURE 3: Rate sweep — same image at different tx rates
    # ================================================================
    fig3, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig3.suptitle('Adaptive Bandwidth: Same Image at Different Transmission Rates',
                  fontsize=16, fontweight='bold')

    rates = [1.0, 0.9, 0.75, 0.5, 0.25]
    demo2 = selected[:2]

    for row, (img_t, label, path) in enumerate(demo2):
        cls_name = test_ds.idx_to_class[label]
        img_batch = img_t.unsqueeze(0).to(device)

        for col, rate in enumerate(rates):
            with torch.no_grad():
                feat = front(img_batch)
                Fp, stats = model(feat, noise_param=0.0, target_rate_override=rate)
                logits = back(Fp)
                probs = F.softmax(logits, dim=1)
                conf, pred_idx = probs.max(1)
                pred_name = test_ds.idx_to_class[pred_idx.item()]

            mask = stats['mask'][0, 0].cpu().numpy()
            img_np = denorm(img_t)
            mask_up = np.kron(mask, np.ones((28, 28)))[:224, :224]
            overlay = img_np.copy()
            overlay[mask_up == 0] = overlay[mask_up == 0] * 0.2 + np.array([1, 0.2, 0.2]) * 0.8

            ax = axes[row, col]
            ax.imshow(overlay)
            correct = pred_name == cls_name
            color = '#2ecc71' if correct else '#e74c3c'
            symbol = '✓' if correct else '✗'
            ax.set_title(f'{symbol} {pred_name} ({conf.item()*100:.1f}%)',
                        fontsize=10, color=color, fontweight='bold')
            ax.axis('off')

            if row == 0:
                ax.text(0.5, 1.2, f'Rate={rate*100:.0f}%\n({int(mask.sum())}/64 blocks)',
                       transform=ax.transAxes, ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(OUT_DIR, 'aid_rate_sweep_viz.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {OUT_DIR}aid_rate_sweep_viz.png")

    # ================================================================
    # FIGURE 4: Mask diversity — 8 different images, show masks are unique
    # ================================================================
    fig4, axes = plt.subplots(2, 8, figsize=(24, 6))
    fig4.suptitle('Content-Adaptive Masking: Each Image Gets a Unique Block Mask',
                  fontsize=16, fontweight='bold')

    for idx in range(min(8, len(selected))):
        img_t, label, path = selected[idx]
        cls_name = test_ds.idx_to_class[label]
        img_batch = img_t.unsqueeze(0).to(device)

        with torch.no_grad():
            feat = front(img_batch)
            Fp, stats = model(feat, noise_param=0.0)

        importance = stats['importance'][0].cpu().numpy()
        mask = stats['mask'][0, 0].cpu().numpy()

        # Top row: image
        axes[0, idx].imshow(denorm(img_t))
        axes[0, idx].set_title(cls_name, fontsize=10, fontweight='bold')
        axes[0, idx].axis('off')

        # Bottom row: mask
        axes[1, idx].imshow(mask, cmap='RdYlGn', vmin=0, vmax=1,
                           interpolation='nearest')
        axes[1, idx].set_title(f'{int(mask.sum())}/64', fontsize=10)
        axes[1, idx].axis('off')

    axes[0, 0].text(-0.2, 0.5, 'Image', transform=axes[0, 0].transAxes,
                    ha='center', va='center', fontsize=12, fontweight='bold', rotation=90)
    axes[1, 0].text(-0.2, 0.5, 'Mask', transform=axes[1, 0].transAxes,
                    ha='center', va='center', fontsize=12, fontweight='bold', rotation=90)

    plt.tight_layout(rect=[0.02, 0, 1, 0.93])
    plt.savefig(os.path.join(OUT_DIR, 'aid_mask_diversity.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {OUT_DIR}aid_mask_diversity.png")

    print(f"\n✅ All visualizations saved to {OUT_DIR}")
