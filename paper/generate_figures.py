"""Generate publication-quality figures for the GlobeCom paper.

Creates 4 figures:
1. System architecture block diagram (IEEE style)
2. Mask diversity across aerial scene classes
3. Feature MSE vs confidence (semantic communication argument)
4. Rate-accuracy tradeoff
"""

import os, sys, random, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image

# IEEE style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SNAP_DIR = "./snapshots_aid/"
OUT_DIR = "./paper/figures/"
os.makedirs(OUT_DIR, exist_ok=True)


# ========================= MODEL CLASSES =========================
# (Same as visualize_features.py — needed to load checkpoints)

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
        if ber <= 0: return x, torch.zeros_like(x)
        flip = (torch.rand_like(x.float()) < ber).float()
        return (x + flip) % 2, flip

class LearnedImportanceScorer(nn.Module):
    def __init__(self, C_in=128, hidden=32):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Conv2d(C_in, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1), nn.Sigmoid())
    def forward(self, all_S2):
        return self.scorer(torch.stack(all_S2, dim=0).mean(dim=0)).squeeze(1)

class LearnedBlockMask(nn.Module):
    def __init__(self, target_rate=0.75):
        super().__init__()
        self.target_rate = target_rate
    def forward(self, importance):
        B, H, W = importance.shape
        k = max(1, int(self.target_rate * H * W))
        flat = importance.view(B, -1)
        _, idx = flat.topk(k, dim=1)
        mask = torch.zeros_like(flat); mask.scatter_(1, idx, 1.0)
        return mask.view(B, H, W).unsqueeze(1), mask.view(B,H,W).mean()
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
        stk = torch.stack(il, dim=1); x = stk.permute(0, 2, 3, 4, 1)
        return (x * torch.sigmoid(self.converter_fc(x))).sum(dim=-1)

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
        return self.spatial_pool(self.layer3(self.layer2(self.layer1(self.maxpool(x)))))

class ResNet50Back(nn.Module):
    def __init__(self, num_classes=30):
        super().__init__()
        r = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        self.layer4 = r.layer4; self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)
    def forward(self, x):
        return self.fc(torch.flatten(self.avgpool(self.layer4(x)), 1))

class AIDDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        aid_dir = os.path.join(root, 'AID')
        classes = sorted([d for d in os.listdir(aid_dir) if os.path.isdir(os.path.join(aid_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        all_samples = []
        for cls_name in classes:
            cls_dir = os.path.join(aid_dir, cls_name)
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
                    all_samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls_name]))
        random.seed(42); random.shuffle(all_samples)
        self.samples = all_samples[int(len(all_samples)*0.8):]
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, label, path

def denorm(t, m=(0.485,0.456,0.406), s=(0.229,0.224,0.225)):
    t = t.cpu().clone()
    for c in range(3): t[c] = t[c]*s[c]+m[c]
    return t.clamp(0,1).permute(1,2,0).numpy()


def load_models():
    front = ResNet50Front().to(device)
    back = ResNet50Back(30).to(device)
    bb = torch.load(os.path.join(SNAP_DIR, "backbone_best.pth"), map_location=device)
    front.load_state_dict({k:v for k,v in bb.items() if not k.startswith(('layer4.','fc.','avgpool.'))}, strict=False)
    back.load_state_dict({k:v for k,v in bb.items() if k.startswith(('layer4.','fc.','avgpool.'))}, strict=False)
    front.eval()

    encoder = Encoder(1024,256,128).to(device)
    imp_scorer = LearnedImportanceScorer(128,32).to(device)
    block_mask = LearnedBlockMask(0.75)
    channel = BSC_Channel().to(device)
    decoder = Decoder(1024,256,128,8).to(device)

    s3 = sorted([f for f in os.listdir(SNAP_DIR) if f.startswith("bsc_s3_")])
    if s3:
        ckpt = torch.load(os.path.join(SNAP_DIR, s3[-1]), map_location=device)
        ms = ckpt['model']
        encoder.load_state_dict({k.replace('encoder.',''):v for k,v in ms.items() if k.startswith('encoder.')})
        imp_scorer.load_state_dict({k.replace('importance_scorer.',''):v for k,v in ms.items() if k.startswith('importance_scorer.')})
        decoder.load_state_dict({k.replace('decoder.',''):v for k,v in ms.items() if k.startswith('decoder.')})
        back.load_state_dict(ckpt['back'])
        print(f"Loaded: {s3[-1]}")
    encoder.eval(); imp_scorer.eval(); decoder.eval(); back.eval()
    return front, back, encoder, imp_scorer, block_mask, channel, decoder


def run_pipeline(front, encoder, imp_scorer, block_mask, channel, decoder, back, img, ber, T=8, rate=None):
    with torch.no_grad():
        feat = front(img)
        all_S2, m1, m2 = [], None, None
        for t in range(T):
            _, s2, m1, m2 = encoder(feat, m1, m2)
            all_S2.append(s2)
        importance = imp_scorer(all_S2)
        if rate is not None:
            old = block_mask.target_rate; block_mask.target_rate = rate
            mask, tx = block_mask(importance)
            block_mask.target_rate = old
        else:
            mask, tx = block_mask(importance)
        masked = [block_mask.apply_mask(all_S2[t], mask) for t in range(T)]
        recv, flips = [], []
        for t in range(T):
            r, f = channel(masked[t], ber); recv.append(r); flips.append(f)
        Fp = decoder(recv, mask)
        logits = back(Fp)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(1)
    return {'feat': feat, 'spikes': torch.stack(all_S2,0), 'importance': importance,
            'mask': mask, 'flips': torch.stack(flips,0), 'received': torch.stack(recv,0),
            'reconstructed': Fp, 'pred': pred, 'conf': conf, 'tx': tx}


if __name__ == "__main__":
    print(f"Device: {device}")
    front, back, encoder, imp_scorer, block_mask, channel, decoder = load_models()

    test_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                         T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
    ds = AIDDataset("./data", transform=test_tf)

    # Collect one sample per class
    class_samples = {}
    for idx in range(len(ds)):
        img, lab, path = ds[idx]
        cls = ds.idx_to_class[lab]
        if cls not in class_samples:
            class_samples[cls] = (img, lab, path)
        if len(class_samples) >= 30: break

    # ================================================================
    # FIGURE 1: System Architecture (drawn programmatically)
    # ================================================================
    fig, ax = plt.subplots(figsize=(7.16, 2.2))  # IEEE column width
    ax.set_xlim(0, 14); ax.set_ylim(0, 3.2)
    ax.axis('off')

    colors = {
        'uav': '#4A90D9', 'resnet': '#5DADE2', 'snn': '#2ECC71',
        'mask': '#E67E22', 'channel': '#E74C3C', 'decoder': '#9B59B6',
        'back': '#1ABC9C'
    }

    boxes = [
        (0.2, 1.2, 1.5, 0.9, 'UAV\nCamera', colors['uav']),
        (2.0, 1.2, 1.8, 0.9, 'ResNet50\nL1-L3\n$\\mathbf{F}\\in\\mathbb{R}^{1024\\times8\\times8}$', colors['resnet']),
        (4.1, 1.2, 1.8, 0.9, 'SNN Encoder\nIF Neurons\n$T{=}8$ steps', colors['snn']),
        (6.2, 1.8, 1.8, 0.65, 'Importance\nScorer', colors['mask']),
        (6.2, 0.7, 1.8, 0.65, 'Block Mask\n$\\rho{=}0.75$', colors['mask']),
        (8.3, 1.2, 1.5, 0.9, 'A2G\nChannel\nBSC/AWGN\n/Rayleigh', colors['channel']),
        (10.1, 1.2, 1.8, 0.9, 'SNN Decoder\nIHF Neurons\n$T{=}8$ steps', colors['decoder']),
        (12.2, 1.2, 1.5, 0.9, 'ResNet50\nL4+FC\n30 classes', colors['back']),
    ]

    for x, y, w, h, txt, col in boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                              facecolor=col, edgecolor='black', linewidth=0.8, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x+w/2, y+h/2, txt, ha='center', va='center', fontsize=6.5,
                fontweight='bold', color='white')

    # Arrows
    arrow_style = dict(arrowstyle='->', color='black', lw=1.0)
    connections = [
        (1.7, 1.65, 2.0, 1.65), (3.8, 1.65, 4.1, 1.65),
        (5.9, 1.65, 6.2, 2.1), (5.9, 1.65, 6.2, 1.0),
        (8.0, 1.0, 8.3, 1.5), (9.8, 1.65, 10.1, 1.65),
        (11.9, 1.65, 12.2, 1.65),
    ]
    for x1,y1,x2,y2 in connections:
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1), arrowprops=arrow_style)

    # Channel arrow (noisy)
    ax.annotate('', xy=(10.1, 1.65), xytext=(9.8, 1.65),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.2, linestyle='--'))

    # Labels
    ax.text(7.1, 0.35, 'Masked spikes', ha='center', fontsize=7, fontstyle='italic')
    ax.text(9.05, 2.3, 'Noisy channel', ha='center', fontsize=7, color='red', fontstyle='italic')

    # UAV and Ground Station labels
    ax.text(3.0, 2.9, '— UAV Onboard —', ha='center', fontsize=8, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.text(11.2, 2.9, '— Ground Station —', ha='center', fontsize=8, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.axvline(x=8.15, color='gray', linestyle=':', linewidth=0.8, ymin=0.05, ymax=0.95)

    plt.savefig(os.path.join(OUT_DIR, 'fig1_architecture.pdf'), format='pdf')
    plt.savefig(os.path.join(OUT_DIR, 'fig1_architecture.png'), format='png')
    print("✓ Fig 1: Architecture")
    plt.close()

    # ================================================================
    # FIGURE 2: Mask Diversity (8 aerial scenes with unique masks)
    # ================================================================
    target_classes = ['Airport', 'Beach', 'Bridge', 'Desert',
                      'Farmland', 'Harbor', 'Mountain', 'Stadium']
    fig, axes = plt.subplots(2, 8, figsize=(7.16, 2.0))
    fig.subplots_adjust(hspace=0.15, wspace=0.08)

    for idx, cls in enumerate(target_classes):
        if cls not in class_samples: continue
        img_t, lab, _ = class_samples[cls]
        img_batch = img_t.unsqueeze(0).to(device)
        result = run_pipeline(front, encoder, imp_scorer, block_mask, channel, decoder, back, img_batch, 0.0)

        mask = result['mask'][0,0].cpu().numpy()
        imp = result['importance'][0].cpu().numpy()

        # Top: image with mask overlay
        img_np = denorm(img_t)
        mask_up = np.kron(mask, np.ones((28,28)))[:224,:224]
        overlay = img_np.copy()
        overlay[mask_up==0] = overlay[mask_up==0]*0.25 + np.array([0.8,0.15,0.15])*0.75

        axes[0, idx].imshow(overlay)
        axes[0, idx].set_title(cls, fontsize=6.5, fontweight='bold', pad=2)
        axes[0, idx].axis('off')

        # Bottom: mask grid
        axes[1, idx].imshow(mask, cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')
        axes[1, idx].set_title(f'{int(mask.sum())}/64', fontsize=6, pad=1)
        axes[1, idx].axis('off')

    axes[0,0].text(-0.35, 0.5, 'Image +\nmask', transform=axes[0,0].transAxes,
                   ha='center', va='center', fontsize=7, fontweight='bold', rotation=90)
    axes[1,0].text(-0.35, 0.5, '8×8\nmask', transform=axes[1,0].transAxes,
                   ha='center', va='center', fontsize=7, fontweight='bold', rotation=90)

    plt.savefig(os.path.join(OUT_DIR, 'fig2_mask_diversity.pdf'), format='pdf')
    plt.savefig(os.path.join(OUT_DIR, 'fig2_mask_diversity.png'), format='png')
    print("✓ Fig 2: Mask diversity")
    plt.close()

    # ================================================================
    # FIGURE 3: Feature MSE vs Confidence vs BER
    # ================================================================
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.16, 2.2))
    fig.subplots_adjust(wspace=0.55)

    ber_fine = [0.0, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    plot_classes = ['Airport', 'Beach', 'Mountain']
    plot_axes = [ax1, ax2, ax3]

    for ax, cls in zip(plot_axes, plot_classes):
        if cls not in class_samples: continue
        img_t, lab, _ = class_samples[cls]
        img_batch = img_t.unsqueeze(0).to(device)

        clean = run_pipeline(front, encoder, imp_scorer, block_mask, channel, decoder, back, img_batch, 0.0)
        clean_feat = clean['reconstructed'][0].cpu()

        mses, confs = [], []
        for ber in ber_fine:
            r = run_pipeline(front, encoder, imp_scorer, block_mask, channel, decoder, back, img_batch, ber)
            mses.append(F.mse_loss(r['reconstructed'][0].cpu(), clean_feat).item())
            confs.append(r['conf'].item() * 100)

        ax2r = ax.twinx()
        l1, = ax.plot(ber_fine, mses, 'o-', color='#c0392b', markersize=3, linewidth=1.0, label='Feature MSE')
        l2, = ax2r.plot(ber_fine, confs, 's-', color='#27ae60', markersize=3, linewidth=1.0, label='Confidence')

        ax.set_xlabel('BER')
        ax.set_ylabel('Feature MSE', color='#c0392b')
        ax2r.set_ylabel('Confidence (%)', color='#27ae60')
        ax.set_title(f'({chr(97+plot_classes.index(cls))}) {cls}', fontsize=9, fontweight='bold')
        ax.tick_params(axis='y', labelcolor='#c0392b')
        ax2r.tick_params(axis='y', labelcolor='#27ae60')
        ax2r.set_ylim(50, 105)
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.legend([l1, l2], ['Feature MSE', 'Confidence'], loc='center right', fontsize=6.5,
                  framealpha=0.8)

    plt.savefig(os.path.join(OUT_DIR, 'fig3_feature_mse.pdf'), format='pdf')
    plt.savefig(os.path.join(OUT_DIR, 'fig3_feature_mse.png'), format='png')
    print("✓ Fig 3: Feature MSE vs Confidence")
    plt.close()

    # ================================================================
    # FIGURE 4: Rate-Accuracy with image overlays
    # ================================================================
    rates = [1.0, 0.90, 0.75, 0.50, 0.25]
    cls = 'Airport'
    img_t, lab, _ = class_samples[cls]
    img_batch = img_t.unsqueeze(0).to(device)

    fig, axes = plt.subplots(1, 5, figsize=(7.16, 1.8))
    fig.subplots_adjust(wspace=0.1)

    for col, rate in enumerate(rates):
        r = run_pipeline(front, encoder, imp_scorer, block_mask, channel, decoder, back, img_batch, 0.0, rate=rate)
        mask = r['mask'][0,0].cpu().numpy()
        pred_name = ds.idx_to_class[r['pred'].item()]
        correct = pred_name == cls

        img_np = denorm(img_t)
        mask_up = np.kron(mask, np.ones((28,28)))[:224,:224]
        overlay = img_np.copy()
        overlay[mask_up==0] = overlay[mask_up==0]*0.2 + np.array([0.9,0.2,0.2])*0.8

        ax = axes[col]
        ax.imshow(overlay)
        color = '#27ae60' if correct else '#c0392b'
        symbol = '✓' if correct else '✗'
        ax.set_title(f'$\\rho$={rate:.0%}\n{symbol} {pred_name} ({r["conf"].item()*100:.0f}%)',
                    fontsize=7, color=color, fontweight='bold')
        ax.set_xlabel(f'{int(mask.sum())}/64 blocks', fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

    plt.savefig(os.path.join(OUT_DIR, 'fig4_rate_sweep.pdf'), format='pdf')
    plt.savefig(os.path.join(OUT_DIR, 'fig4_rate_sweep.png'), format='png')
    print("✓ Fig 4: Rate sweep")
    plt.close()

    print(f"\n✅ All paper figures saved to {OUT_DIR}")
    print("   PDF versions ready for LaTeX \\includegraphics")
