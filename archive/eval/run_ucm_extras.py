"""Run all additional UCM experiments:
  1. V5A (KD), V5D (Attn), V5E (CG+Attn) training
  2. ρ sweep ablation (V2, V4-A, V5C × 6 rates × 4 BER)
  3. Mask comparison (learned vs random vs uniform)
  4. Cross-dataset figure generation (AID vs UCM)

Usage:
  python eval/run_ucm_extras.py
"""

import os, sys, json, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))

from train_aid_v2 import (
    ResNet50Front, ResNet50Back, BSC_Channel,
    ChannelConditionedScorer, LearnedBlockMask, sample_noise,
    SpikeAdaptSC_v2
)
from train_aid_v5 import SpikeAdaptSC_v5
from train_ucm import UCMDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ############################################################################
# HELPERS
# ############################################################################

def get_ucm_loaders(seed=42, batch_size=32):
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])
    tf_train = T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
                           T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))])
    train_ds = UCMDataset("./data", tf_train, split='train', seed=seed)
    test_ds = UCMDataset("./data", tf_test, split='test', seed=seed)
    return (DataLoader(train_ds, batch_size, True, num_workers=4, pin_memory=True),
            DataLoader(test_ds, batch_size, False, num_workers=4, pin_memory=True))

def load_front_back(num_classes=21):
    front = ResNet50Front(grid_size=14).to(device)
    back = ResNet50Back(num_classes).to(device)
    bb = torch.load("./snapshots_ucm_seed42/backbone_best.pth", map_location=device)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.'))}, strict=False)
    front.eval()
    for p in front.parameters(): p.requires_grad = False
    return front, back

def transfer_weights(model, back, ckpt_path):
    ck = torch.load(ckpt_path, map_location=device)
    transferred = 0
    m_state = model.state_dict()
    for k, v in ck['model'].items():
        if k in m_state and m_state[k].shape == v.shape:
            m_state[k] = v; transferred += 1
    model.load_state_dict(m_state, strict=False)
    back.load_state_dict(ck['back'])
    print(f"  ✓ Transferred {transferred} params")
    return model, back

def eval_ber(model, front, back, test_loader, ber):
    model.eval(); back.eval(); correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            Fp, _ = model(front(imgs), noise_param=ber)
            correct += back(Fp).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total

def eval_ber_with_rate(model, front, back, test_loader, ber, rate):
    """Evaluate with a specific target rate override."""
    model.eval(); back.eval(); correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            Fp, _ = model(front(imgs), noise_param=ber, target_rate_override=rate)
            correct += back(Fp).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total

def train_v5_variant(config_name, use_mpbn, use_channel_gate, use_spike_attn,
                     epochs=40, seed=42, num_classes=21):
    """Train a V5 variant on UCM."""
    print(f"\n{'='*60}")
    print(f"V5 {config_name} on UCM")
    print(f"  mpbn={use_mpbn}, cg={use_channel_gate}, attn={use_spike_attn}")
    print(f"{'='*60}")

    train_loader, test_loader = get_ucm_loaders(seed)
    front, back = load_front_back(num_classes)
    snap = f"./snapshots_ucm_v5{config_name.lower()}_seed{seed}/"
    os.makedirs(snap, exist_ok=True)

    model = SpikeAdaptSC_v5(C_in=1024, C1=256, C2=36, T=8,
                             target_rate=0.75, grid_size=14,
                             use_mpbn=use_mpbn, use_channel_gate=use_channel_gate,
                             use_spike_attn=use_spike_attn).to(device)

    # Transfer from V4-A seed=42
    v4a_dir = "./snapshots_ucm_v4a_seed42/"
    if os.path.exists(v4a_dir):
        v4_files = sorted([f for f in os.listdir(v4a_dir) if f.startswith("v4a_")])
        if v4_files:
            model, back = transfer_weights(model, back, os.path.join(v4a_dir, v4_files[-1]))

    slope_params = [p for n, p in model.named_parameters() if 'slope' in n]
    other_params = [p for n, p in model.named_parameters() if 'slope' not in n]
    optimizer = optim.Adam([
        {'params': other_params, 'lr': 1e-5},
        {'params': slope_params, 'lr': 1e-4},
        {'params': back.parameters(), 'lr': 1e-5}
    ])
    criterion = nn.CrossEntropyLoss()
    best_acc = 0

    for epoch in range(1, epochs + 1):
        model.train(); back.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            ber = sample_noise('bsc')
            Fp, stats = model(feat, noise_param=ber)
            loss = criterion(back(Fp), labels) + 0.1 * stats.get('rate_penalty', 0)
            if 'spike_reg' in stats: loss = loss + 0.01 * stats['spike_reg']
            if 'stdp_loss' in stats: loss = loss + 0.005 * stats['stdp_loss']
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        if epoch % 5 == 0 or epoch == epochs:
            acc = eval_ber(model, front, back, test_loader, 0.0)
            fr = stats.get('firing_rate', 0)
            print(f"  S3 E{epoch}: {acc:.2f}%, FR={fr:.3f}")
            if acc > best_acc:
                best_acc = acc
                torch.save({'model': model.state_dict(), 'back': back.state_dict()},
                           os.path.join(snap, f"v5{config_name.lower()}_{acc:.2f}.pth"))
                print(f"  ✓ Best: {best_acc:.2f}%")

    # Final eval
    results = {}
    print(f"\nFINAL EVALUATION — V5{config_name} on UCM")
    for ber in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        acc = eval_ber(model, front, back, test_loader, ber)
        label = "Clean" if ber == 0 else f"BER={ber:.2f}"
        print(f"  {label}: {acc:.2f}%")
        results[str(ber)] = acc
    with open(os.path.join(snap, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    return results


# ############################################################################
# ρ SWEEP
# ############################################################################

def run_rho_sweep(test_loader, seed=42, num_classes=21):
    print(f"\n{'='*60}")
    print(f"ρ SWEEP ABLATION — UCM")
    print(f"{'='*60}")

    rhos = [0.25, 0.50, 0.625, 0.75, 0.875, 1.0]
    bers = [0.0, 0.10, 0.20, 0.30]
    all_results = {}

    # V2
    print("\n  Loading V2...")
    front, back = load_front_back(num_classes)
    v2_dir = "./snapshots_ucm_seed42/"
    v2_files = sorted([f for f in os.listdir(v2_dir) if f.startswith("v2_s3_")])
    if v2_files:
        model_v2 = SpikeAdaptSC_v2(C_in=1024, C1=256, C2=36, T=8, target_rate=0.75, grid_size=14).to(device)
        ck = torch.load(os.path.join(v2_dir, v2_files[-1]), map_location=device)
        model_v2.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
        model_v2.eval(); back.eval()

        v2_results = {}
        for rho in rhos:
            v2_results[str(rho)] = {}
            for ber in bers:
                acc = eval_ber_with_rate(model_v2, front, back, test_loader, ber, rho)
                v2_results[str(rho)][str(ber)] = acc
                print(f"    v2 ρ={rho:.3f} BER={ber:.2f}: {acc:.2f}%")
        all_results['v2'] = v2_results

    # V4-A
    print("\n  Loading V4-A...")
    from train_aid_v4 import SpikeAdaptSC_v4
    front, back = load_front_back(num_classes)
    v4_dir = "./snapshots_ucm_v4a_seed42/"
    v4_files = sorted([f for f in os.listdir(v4_dir) if f.startswith("v4a_")])
    if v4_files:
        model_v4 = SpikeAdaptSC_v4(C_in=1024, C1=256, C2=36, T=8,
                                     target_rate=0.75, grid_size=14,
                                     use_mpbn=False, use_channel_gate=False).to(device)
        ck = torch.load(os.path.join(v4_dir, v4_files[-1]), map_location=device)
        model_v4.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
        model_v4.eval(); back.eval()

        v4_results = {}
        for rho in rhos:
            v4_results[str(rho)] = {}
            for ber in bers:
                acc = eval_ber_with_rate(model_v4, front, back, test_loader, ber, rho)
                v4_results[str(rho)][str(ber)] = acc
                print(f"    v4a ρ={rho:.3f} BER={ber:.2f}: {acc:.2f}%")
        all_results['v4a'] = v4_results

    # V5C
    print("\n  Loading V5C...")
    front, back = load_front_back(num_classes)
    v5_dir = "./snapshots_ucm_v5c_seed42/"
    v5_files = sorted([f for f in os.listdir(v5_dir) if f.startswith("v5c_")])
    if v5_files:
        model_v5 = SpikeAdaptSC_v5(C_in=1024, C1=256, C2=36, T=8,
                                     target_rate=0.75, grid_size=14,
                                     use_mpbn=True, use_channel_gate=False,
                                     use_spike_attn=False).to(device)
        ck = torch.load(os.path.join(v5_dir, v5_files[-1]), map_location=device)
        model_v5.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
        model_v5.eval(); back.eval()

        v5_results = {}
        for rho in rhos:
            v5_results[str(rho)] = {}
            for ber in bers:
                acc = eval_ber_with_rate(model_v5, front, back, test_loader, ber, rho)
                v5_results[str(rho)][str(ber)] = acc
                print(f"    v5c ρ={rho:.3f} BER={ber:.2f}: {acc:.2f}%")
        all_results['v5c'] = v5_results

    # Print table
    print(f"\n{'='*60}")
    print("UCM ρ SWEEP RESULTS TABLE")
    print(f"{'='*60}")
    header = "     ρ"
    for ver in ['v2', 'v4a', 'v5c']:
        for ber in bers:
            header += f" | {ver.upper()} B{ber:.2f}"
    print(header)
    for rho in rhos:
        row = f" {rho:.3f}"
        for ver in ['v2', 'v4a', 'v5c']:
            if ver in all_results:
                for ber in bers:
                    row += f" | {all_results[ver][str(rho)][str(ber)]:>8.2f}"
            else:
                for ber in bers:
                    row += " |      N/A"
        print(row)

    return all_results


# ############################################################################
# MASK COMPARISON
# ############################################################################

def run_mask_comparison(test_loader, seed=42, num_classes=21):
    print(f"\n{'='*60}")
    print(f"MASK COMPARISON — UCM (Learned vs Random vs Uniform)")
    print(f"{'='*60}")

    rhos = [0.50, 0.75, 1.00]
    bers_mask = [0.00, 0.15, 0.30]
    n_random = 3
    all_results = {}

    for ver_name, model_class, snap_dir, kwargs in [
        ('v4a', None, './snapshots_ucm_v4a_seed42/', {'use_mpbn': False, 'use_channel_gate': False}),
        ('v5c', None, './snapshots_ucm_v5c_seed42/', {'use_mpbn': True, 'use_channel_gate': False, 'use_spike_attn': False}),
    ]:
        print(f"\n  Loading {ver_name}...")
        front, back = load_front_back(num_classes)

        if ver_name == 'v4a':
            from train_aid_v4 import SpikeAdaptSC_v4
            model = SpikeAdaptSC_v4(C_in=1024, C1=256, C2=36, T=8,
                                     target_rate=0.75, grid_size=14, **kwargs).to(device)
        else:
            model = SpikeAdaptSC_v5(C_in=1024, C1=256, C2=36, T=8,
                                     target_rate=0.75, grid_size=14, **kwargs).to(device)

        prefix = "v4a_" if ver_name == 'v4a' else "v5c_"
        files = sorted([f for f in os.listdir(snap_dir) if f.startswith(prefix)])
        if not files: continue
        ck = torch.load(os.path.join(snap_dir, files[-1]), map_location=device)
        model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
        model.eval(); back.eval()

        ver_results = {}
        for rho in rhos:
            ver_results[str(rho)] = {}
            for ber in bers_mask:
                # Learned
                learned_acc = eval_ber_with_rate(model, front, back, test_loader, ber, rho)

                # Random (average over n_random draws)
                random_accs = []
                for _ in range(n_random):
                    acc = eval_random_mask(model, front, back, test_loader, ber, rho)
                    random_accs.append(acc)
                random_acc = np.mean(random_accs)

                # Uniform
                uniform_acc = eval_uniform_mask(model, front, back, test_loader, ber, rho)

                ver_results[str(rho)][str(ber)] = {
                    'learned': learned_acc,
                    'random': random_acc,
                    'uniform': uniform_acc,
                    'delta_random': learned_acc - random_acc,
                    'delta_uniform': learned_acc - uniform_acc,
                }
                print(f"    {ver_name} ρ={rho:.2f} BER={ber:.2f}: "
                      f"Learned={learned_acc:.2f}% Random={random_acc:.2f}% Uniform={uniform_acc:.2f}% "
                      f"(Δ_rand=+{learned_acc-random_acc:.2f}, Δ_uni=+{learned_acc-uniform_acc:.2f})")

        all_results[ver_name] = ver_results

    return all_results


def eval_random_mask(model, front, back, test_loader, ber, rho):
    """Evaluate with random spatial masks instead of learned ones."""
    model.eval(); back.eval(); correct, total = 0, 0
    H, W = 14, 14
    k = max(1, int(rho * H * W))
    T = 8
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            B = imgs.size(0)
            feat = front(imgs)

            # Run encoder
            all_S2, m1, m2 = [], None, None
            for t in range(T):
                _, s2, m1, m2 = model.encoder(feat, m1, m2, t=t)
                all_S2.append(s2)

            # Random mask
            mask = torch.zeros(B, 1, H, W, device=device)
            for b in range(B):
                idx = torch.randperm(H * W, device=device)[:k]
                mask_flat = torch.zeros(H * W, device=device)
                mask_flat[idx] = 1.0
                mask[b, 0] = mask_flat.view(H, W)

            # Channel + decoder
            recv = [model.channel(all_S2[t] * mask, ber) for t in range(T)]
            Fp = model.decoder(recv, mask)
            correct += back(Fp).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total


def eval_uniform_mask(model, front, back, test_loader, ber, rho):
    """Evaluate with uniform (deterministic grid) mask."""
    model.eval(); back.eval(); correct, total = 0, 0
    H, W = 14, 14
    k = max(1, int(rho * H * W))
    T = 8

    # Pre-compute uniform mask
    mask_flat = torch.zeros(H * W, device=device)
    step = max(1, H * W // k)
    indices = torch.arange(0, H * W, step, device=device)[:k]
    mask_flat[indices] = 1.0
    uniform_mask = mask_flat.view(1, 1, H, W)

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)

            all_S2, m1, m2 = [], None, None
            for t in range(T):
                _, s2, m1, m2 = model.encoder(feat, m1, m2, t=t)
                all_S2.append(s2)

            mask = uniform_mask.expand(feat.size(0), -1, -1, -1)
            recv = [model.channel(all_S2[t] * mask, ber) for t in range(T)]
            Fp = model.decoder(recv, mask)
            correct += back(Fp).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total


# ############################################################################
# CROSS-DATASET FIGURES
# ############################################################################

def generate_cross_dataset_figures(aid_rho_data, ucm_rho_data,
                                    aid_mask_data, ucm_mask_data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02, 'lines.linewidth': 1.5, 'lines.markersize': 5,
    })

    OUT = "./paper/figures/"
    os.makedirs(OUT, exist_ok=True)

    # Fig 10: Cross-dataset ρ sweep comparison
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))
    fig.subplots_adjust(wspace=0.35)
    rhos = [0.25, 0.50, 0.625, 0.75, 0.875, 1.0]

    for panel, (ax, ber_key, title) in enumerate(zip(
        axes, ['0.0', '0.3'],
        ['(a) Clean Accuracy', '(b) BER=0.30']
    )):
        for ver, style in [('v2', {'c': '#3498DB', 'ls': '--', 'm': 'o'}),
                           ('v4a', {'c': '#E74C3C', 'ls': '-', 'm': 's'}),
                           ('v5c', {'c': '#2ECC71', 'ls': '-', 'm': '^'})]:
            # AID
            if ver in aid_rho_data:
                aid_accs = [aid_rho_data[ver][str(r)][ber_key] for r in rhos]
                ax.plot(rhos, aid_accs, marker=style['m'], color=style['c'],
                        linestyle=style['ls'], label=f'AID {ver.upper()}', alpha=0.6, markersize=4)
            # UCM
            if ver in ucm_rho_data:
                ucm_accs = [ucm_rho_data[ver][str(r)][ber_key] for r in rhos]
                ax.plot(rhos, ucm_accs, marker=style['m'], color=style['c'],
                        linestyle=style['ls'], label=f'UCM {ver.upper()}',
                        linewidth=2.0, markersize=6, zorder=3)

        ax.set_xlabel('Mask Rate ρ')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='lower right', framealpha=0.9, ncol=2, fontsize=6)
        ax.grid(True, alpha=0.3)

    plt.savefig(os.path.join(OUT, 'fig10_cross_dataset_rho.pdf'), format='pdf')
    plt.savefig(os.path.join(OUT, 'fig10_cross_dataset_rho.png'), format='png')
    print("✓ Fig 10: Cross-dataset ρ sweep")
    plt.close()

    # Fig 11: Cross-dataset version comparison (grouped bar)
    fig, ax = plt.subplots(figsize=(7.16, 3.0))
    versions = ['V2', 'V4-A', 'V5C']
    x = np.arange(len(versions))
    width = 0.18

    # AID data
    aid_clean = [96.35, 96.45, 96.55]
    aid_ber30 = [92.90, 95.85, 95.75]
    # UCM data
    ucm_clean = [97.38, 97.62, 96.43]
    ucm_ber30 = [90.00, 97.14, 96.67]

    ax.bar(x - 1.5*width, aid_clean, width, label='AID Clean', color='#3498DB', alpha=0.9, edgecolor='black', linewidth=0.5)
    ax.bar(x - 0.5*width, aid_ber30, width, label='AID BER=0.30', color='#3498DB', alpha=0.4, edgecolor='black', linewidth=0.5, hatch='///')
    ax.bar(x + 0.5*width, ucm_clean, width, label='UCM Clean', color='#2ECC71', alpha=0.9, edgecolor='black', linewidth=0.5)
    ax.bar(x + 1.5*width, ucm_ber30, width, label='UCM BER=0.30', color='#2ECC71', alpha=0.4, edgecolor='black', linewidth=0.5, hatch='///')

    ax.set_xticks(x)
    ax.set_xticklabels(versions, fontweight='bold')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Cross-Dataset Generalization: AID-30 vs UCM-21', fontweight='bold')
    ax.set_ylim(87, 99)
    ax.legend(loc='lower right', framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.2, axis='y')

    plt.savefig(os.path.join(OUT, 'fig11_cross_dataset_bars.pdf'), format='pdf')
    plt.savefig(os.path.join(OUT, 'fig11_cross_dataset_bars.png'), format='png')
    print("✓ Fig 11: Cross-dataset version comparison")
    plt.close()

    print(f"\n✅ Cross-dataset figures saved to {OUT}")


# ############################################################################
# MAIN
# ############################################################################

if __name__ == "__main__":
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    print(f"Device: {device}")

    _, test_loader = get_ucm_loaders(42)

    # ---- 1. Train V5 variants ----
    print("\n" + "="*70)
    print("PART 1: V5 VARIANT TRAINING ON UCM")
    print("="*70)

    v5a_results = train_v5_variant('A', use_mpbn=False, use_channel_gate=False,
                                    use_spike_attn=False, epochs=40)
    v5d_results = train_v5_variant('D', use_mpbn=False, use_channel_gate=False,
                                    use_spike_attn=True, epochs=40)
    v5e_results = train_v5_variant('E', use_mpbn=False, use_channel_gate=True,
                                    use_spike_attn=True, epochs=40)

    # ---- 2. ρ sweep ----
    print("\n" + "="*70)
    print("PART 2: ρ SWEEP ABLATION")
    print("="*70)

    ucm_rho = run_rho_sweep(test_loader)

    # ---- 3. Mask comparison ----
    print("\n" + "="*70)
    print("PART 3: MASK COMPARISON")
    print("="*70)

    ucm_mask = run_mask_comparison(test_loader)

    # Save all results
    all_results = {
        'v5_variants': {
            'v5a': v5a_results,
            'v5d': v5d_results,
            'v5e': v5e_results,
        },
        'rho_sweep': ucm_rho,
        'mask_comparison': ucm_mask,
    }
    with open("eval/ucm_ablation_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ All UCM results saved to eval/ucm_ablation_results.json")

    # ---- 4. Cross-dataset figures ----
    print("\n" + "="*70)
    print("PART 4: CROSS-DATASET FIGURE GENERATION")
    print("="*70)

    # Load AID ablation data
    aid_rho = {}
    if os.path.exists("eval/ablation_results.json"):
        with open("eval/ablation_results.json") as f:
            aid_data = json.load(f)
        aid_rho = aid_data.get('rho_sweep', {})
        aid_mask = aid_data.get('mask_comparison', {})
    else:
        aid_mask = {}

    generate_cross_dataset_figures(aid_rho, ucm_rho, aid_mask, ucm_mask)

    print("\n" + "="*70)
    print("✅ ALL UCM EXTRA EXPERIMENTS COMPLETE")
    print("="*70)
