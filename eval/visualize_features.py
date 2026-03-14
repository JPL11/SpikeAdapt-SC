"""Visualize SpikeAdapt-SC internals: features, spikes, channel corruption.

Shows what actually happens at each pipeline stage, making the channel
noise visible in feature space rather than pixel space.
"""

import os, sys, random, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SNAP_DIR = "./snapshots_aid/"
OUT_DIR = "./figures/"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Model classes (same as visualize_aid.py) ----

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
        x_n = (x + flip) % 2
        return x_n, flip  # Return flip mask for visualization

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
    def __init__(self, root, transform=None):
        self.transform = transform
        aid_dir = os.path.join(root, 'AID')
        classes = sorted([d for d in os.listdir(aid_dir)
                         if os.path.isdir(os.path.join(aid_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        all_samples = []
        for cls_name in classes:
            cls_dir = os.path.join(aid_dir, cls_name)
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
                    all_samples.append((os.path.join(cls_dir, fname),
                                       self.class_to_idx[cls_name]))
        random.seed(42); random.shuffle(all_samples)
        split_idx = int(len(all_samples) * 0.8)
        self.samples = all_samples[split_idx:]

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, label, path


def denorm(img_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    img = img_tensor.cpu().clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def run_pipeline_with_internals(front, encoder, importance_scorer, block_mask,
                                channel, decoder, back, img_batch, ber, T=8):
    """Run through pipeline saving all intermediate representations."""
    with torch.no_grad():
        feat = front(img_batch)                  # (B, 1024, 8, 8)

        # Encode — collect spikes over T timesteps
        all_S1, all_S2, m1, m2 = [], [], None, None
        for t in range(T):
            s1, s2, m1, m2 = encoder(feat, m1, m2)
            all_S1.append(s1); all_S2.append(s2)

        # Importance + mask
        importance = importance_scorer(all_S2)
        mask, tx = block_mask(importance, training=False)

        # Apply mask to spikes
        masked_spikes = [block_mask.apply_mask(all_S2[t], mask) for t in range(T)]

        # Channel — get corruption details
        recv = []
        all_flips = []
        for t in range(T):
            noisy, flips = channel(masked_spikes[t], ber)
            recv.append(noisy)
            all_flips.append(flips)

        # Decode
        Fp = decoder(recv, mask)

        # Classify
        logits = back(Fp)
        probs = F.softmax(logits, dim=1)
        conf, pred_idx = probs.max(1)

    return {
        'feat': feat,                                    # (B, 1024, 8, 8)
        'spikes': torch.stack(all_S2, dim=0),            # (T, B, 128, 8, 8)
        'importance': importance,                         # (B, 8, 8)
        'mask': mask,                                     # (B, 1, 8, 8)
        'masked_spikes': torch.stack(masked_spikes, dim=0),  # (T, B, 128, 8, 8)
        'flips': torch.stack(all_flips, dim=0),           # (T, B, 128, 8, 8)
        'received': torch.stack(recv, dim=0),             # (T, B, 128, 8, 8)
        'reconstructed': Fp,                              # (B, 1024, 8, 8)
        'pred_idx': pred_idx,
        'confidence': conf,
        'tx_rate': tx.item(),
    }


if __name__ == "__main__":
    print(f"Device: {device}")

    # Load models
    front = ResNet50Front().to(device)
    back = ResNet50Back(30).to(device)
    bb_state = torch.load(os.path.join(SNAP_DIR, "backbone_best.pth"), map_location=device)
    front.load_state_dict({k: v for k, v in bb_state.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.'))}, strict=False)
    back.load_state_dict({k: v for k, v in bb_state.items()
                          if k.startswith(('layer4.', 'fc.', 'avgpool.'))}, strict=False)
    front.eval()

    # Build SpikeAdapt-SC components individually for access to internals
    encoder = Encoder(1024, 256, 128).to(device)
    importance_scorer = LearnedImportanceScorer(128, 32).to(device)
    block_mask_module = LearnedBlockMask(0.75, 0.5)
    channel = BSC_Channel().to(device)
    decoder = Decoder(1024, 256, 128, 8).to(device)

    # Load from checkpoint
    s3_files = sorted([f for f in os.listdir(SNAP_DIR) if f.startswith("bsc_s3_")])
    if s3_files:
        ckpt = torch.load(os.path.join(SNAP_DIR, s3_files[-1]), map_location=device)
        model_state = ckpt['model']
        # Map keys
        enc_state = {k.replace('encoder.', ''): v for k, v in model_state.items() if k.startswith('encoder.')}
        imp_state = {k.replace('importance_scorer.', ''): v for k, v in model_state.items() if k.startswith('importance_scorer.')}
        dec_state = {k.replace('decoder.', ''): v for k, v in model_state.items() if k.startswith('decoder.')}
        blk_state = {k.replace('block_mask.', ''): v for k, v in model_state.items() if k.startswith('block_mask.')}
        encoder.load_state_dict(enc_state)
        importance_scorer.load_state_dict(imp_state)
        decoder.load_state_dict(dec_state)
        if blk_state:
            block_mask_module.load_state_dict(blk_state, strict=False)
        back.load_state_dict(ckpt['back'])
        print(f"✓ Loaded: {s3_files[-1]}")
    encoder.eval(); importance_scorer.eval(); decoder.eval(); back.eval()

    # Dataset
    test_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                         T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    test_ds = AIDDataset("./data", transform=test_tf)

    # Pick 3 images from different classes
    picked = {}
    for idx in range(len(test_ds)):
        img, lab, path = test_ds[idx]
        cls = test_ds.idx_to_class[lab]
        if cls not in picked and cls in ['Airport', 'Beach', 'Forest', 'Harbor',
                                          'Mountain', 'Stadium', 'Bridge', 'Farmland']:
            picked[cls] = (img, lab, path)
        if len(picked) >= 3:
            break
    images = list(picked.values())
    print(f"Selected: {list(picked.keys())}")

    # ================================================================
    # FIGURE: Feature-level pipeline for ONE image across BER levels
    # ================================================================
    ber_vals = [0.0, 0.05, 0.15, 0.3]
    n_bers = len(ber_vals)

    for img_idx, (img_t, label, path) in enumerate(images):
        cls_name = test_ds.idx_to_class[label]
        img_batch = img_t.unsqueeze(0).to(device)

        fig = plt.figure(figsize=(24, 16))
        fig.suptitle(f'SpikeAdapt-SC Internal Pipeline: "{cls_name}" at Different BER Levels',
                     fontsize=18, fontweight='bold', y=0.99)

        # 7 rows: Original | ResNet Features | Spike Rates | Importance+Mask |
        #          Channel Bit Flips | Received Spikes | Reconstructed Features
        gs = gridspec.GridSpec(7, n_bers + 1, hspace=0.4, wspace=0.2,
                               width_ratios=[0.12] + [1]*n_bers,
                               left=0.02, right=0.98, top=0.95, bottom=0.02)

        row_labels = [
            'Original\nImage',
            'ResNet L3\nFeatures',
            'SNN Encoder\nSpike Rates',
            'Importance\n& Mask',
            'Channel\nBit Flips',
            'Received\nSpikes',
            'Reconstructed\nFeatures'
        ]

        for row_idx, label_text in enumerate(row_labels):
            ax_label = fig.add_subplot(gs[row_idx, 0])
            ax_label.text(0.5, 0.5, label_text, ha='center', va='center',
                         fontsize=10, fontweight='bold', transform=ax_label.transAxes)
            ax_label.axis('off')

        for col_idx, ber in enumerate(ber_vals):
            result = run_pipeline_with_internals(
                front, encoder, importance_scorer, block_mask_module,
                channel, decoder, back, img_batch, ber, T=8
            )

            pred_name = test_ds.idx_to_class[result['pred_idx'].item()]
            correct = pred_name == cls_name
            color = '#2ecc71' if correct else '#e74c3c'
            symbol = '✓' if correct else '✗'

            col = col_idx + 1

            # Row 0: Original image
            ax = fig.add_subplot(gs[0, col])
            ax.imshow(denorm(img_t))
            if col_idx == 0:
                ax.set_title(f'BER={ber}\n(Clean)', fontsize=12, fontweight='bold')
            else:
                ax.set_title(f'BER={ber}', fontsize=12, fontweight='bold')
            ax.axis('off')

            # Row 1: ResNet features (mean over channels → 8×8 heatmap)
            ax = fig.add_subplot(gs[1, col])
            feat_map = result['feat'][0].mean(dim=0).cpu().numpy()  # (8, 8)
            ax.imshow(feat_map, cmap='viridis', interpolation='nearest')
            ax.set_title(f'μ={feat_map.mean():.2f}', fontsize=9)
            ax.axis('off')

            # Row 2: Spike rates (mean firing rate per spatial block)
            ax = fig.add_subplot(gs[2, col])
            spike_rates = result['spikes'][:, 0].mean(dim=(0, 1)).cpu().numpy()  # (8, 8)
            ax.imshow(spike_rates, cmap='hot', vmin=0, vmax=1, interpolation='nearest')
            ax.set_title(f'Rate: {spike_rates.mean():.2f}', fontsize=9)
            ax.axis('off')

            # Row 3: Importance map + mask overlay
            ax = fig.add_subplot(gs[3, col])
            imp = result['importance'][0].cpu().numpy()
            mask = result['mask'][0, 0].cpu().numpy()
            # Show importance with mask borders
            display = np.stack([imp, imp * mask, imp * mask], axis=-1)
            # Red where masked out
            display[mask == 0, 0] = 0.8
            display[mask == 0, 1] = 0.1
            display[mask == 0, 2] = 0.1
            ax.imshow(display, interpolation='nearest')
            ax.set_title(f'{int(mask.sum())}/64 sent', fontsize=9)
            ax.axis('off')

            # Row 4: Bit flips from channel (very visible!)
            ax = fig.add_subplot(gs[4, col])
            if ber > 0:
                # Show flip density per spatial block
                flip_density = result['flips'][:, 0].mean(dim=(0, 1)).cpu().numpy()  # (8, 8)
                total_flips = result['flips'].sum().item()
                total_bits = result['flips'].numel()
                actual_ber = total_flips / total_bits
                im = ax.imshow(flip_density, cmap='Reds', vmin=0,
                              vmax=max(0.4, flip_density.max()),
                              interpolation='nearest')
                ax.set_title(f'Flipped: {actual_ber*100:.1f}%', fontsize=9,
                            color='red', fontweight='bold')
            else:
                ax.imshow(np.zeros((8, 8)), cmap='Reds', vmin=0, vmax=0.4,
                         interpolation='nearest')
                ax.set_title('No flips', fontsize=9, color='green')
            ax.axis('off')

            # Row 5: Received spikes (after channel)
            ax = fig.add_subplot(gs[5, col])
            recv_rates = result['received'][:, 0].mean(dim=(0, 1)).cpu().numpy()
            ax.imshow(recv_rates, cmap='hot', vmin=0, vmax=1, interpolation='nearest')
            # Show difference from clean
            if ber > 0:
                clean_result = run_pipeline_with_internals(
                    front, encoder, importance_scorer, block_mask_module,
                    channel, decoder, back, img_batch, 0.0, T=8)
                clean_recv = clean_result['received'][:, 0].mean(dim=(0, 1)).cpu().numpy()
                diff = np.abs(recv_rates - clean_recv).mean()
                ax.set_title(f'Δ from clean: {diff:.3f}', fontsize=9)
            else:
                ax.set_title(f'Rate: {recv_rates.mean():.2f}', fontsize=9)
            ax.axis('off')

            # Row 6: Reconstructed features + prediction
            ax = fig.add_subplot(gs[6, col])
            recon = result['reconstructed'][0].mean(dim=0).cpu().numpy()
            ax.imshow(recon, cmap='viridis', interpolation='nearest')
            ax.set_title(f'{symbol} {pred_name}\n{result["confidence"].item()*100:.1f}%',
                        fontsize=11, color=color, fontweight='bold')
            ax.axis('off')

        fname = f'aid_feature_pipeline_{cls_name.lower()}.png'
        plt.savefig(os.path.join(OUT_DIR, fname), dpi=150,
                    bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {OUT_DIR}{fname}")
        plt.close()

    # ================================================================
    # FIGURE: Side-by-side spike raster at BER=0 vs BER=0.3
    # ================================================================
    img_t, label, path = images[0]
    cls_name = test_ds.idx_to_class[label]
    img_batch = img_t.unsqueeze(0).to(device)

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle(f'Spike Trains: Clean vs BER=0.3 — "{cls_name}"',
                 fontsize=16, fontweight='bold')

    for row, ber in enumerate([0.0, 0.3]):
        result = run_pipeline_with_internals(
            front, encoder, importance_scorer, block_mask_module,
            channel, decoder, back, img_batch, ber, T=8
        )

        # Show 4 timesteps of spike patterns (channel 0:16 mean)
        for t_idx, t in enumerate([0, 2, 5, 7]):
            ax = axes[row, t_idx]
            spk = result['received'][t, 0, :16].mean(dim=0).cpu().numpy()  # (8, 8)
            ax.imshow(spk, cmap='binary', vmin=0, vmax=1, interpolation='nearest')
            if row == 0:
                ax.set_title(f't={t}', fontsize=12, fontweight='bold')
            ax.axis('off')

        axes[row, 0].set_ylabel(f'BER={ber}', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'aid_spike_raster.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {OUT_DIR}aid_spike_raster.png")

    # ================================================================
    # FIGURE: Feature reconstruction error vs BER
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Feature Reconstruction Quality vs Channel Noise',
                 fontsize=16, fontweight='bold')

    ber_fine = [0.0, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]

    for img_idx, (img_t, label, path) in enumerate(images):
        cls_name = test_ds.idx_to_class[label]
        img_batch = img_t.unsqueeze(0).to(device)

        # Get clean reconstruction
        clean = run_pipeline_with_internals(
            front, encoder, importance_scorer, block_mask_module,
            channel, decoder, back, img_batch, 0.0, T=8
        )
        clean_feat = clean['reconstructed'][0].cpu()
        clean_conf = clean['confidence'].item()

        mses = []
        confs = []
        for ber in ber_fine:
            result = run_pipeline_with_internals(
                front, encoder, importance_scorer, block_mask_module,
                channel, decoder, back, img_batch, ber, T=8
            )
            noisy_feat = result['reconstructed'][0].cpu()
            mse = F.mse_loss(noisy_feat, clean_feat).item()
            mses.append(mse)
            confs.append(result['confidence'].item() * 100)

        ax = axes[img_idx]
        ax2 = ax.twinx()
        l1 = ax.plot(ber_fine, mses, 'o-', color='#e74c3c', linewidth=2,
                     markersize=6, label='Feature MSE')
        l2 = ax2.plot(ber_fine, confs, 's-', color='#2ecc71', linewidth=2,
                      markersize=6, label='Confidence %')
        ax.set_xlabel('BER', fontsize=12)
        ax.set_ylabel('Feature MSE', fontsize=12, color='#e74c3c')
        ax2.set_ylabel('Confidence %', fontsize=12, color='#2ecc71')
        ax.set_title(cls_name, fontsize=13, fontweight='bold')
        ax.tick_params(axis='y', labelcolor='#e74c3c')
        ax2.tick_params(axis='y', labelcolor='#2ecc71')
        ax2.set_ylim(0, 105)

        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='center right', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'aid_feature_mse_vs_ber.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {OUT_DIR}aid_feature_mse_vs_ber.png")

    print(f"\n✅ All feature-level visualizations saved to {OUT_DIR}")
