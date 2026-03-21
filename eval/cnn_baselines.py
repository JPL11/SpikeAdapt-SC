"""Evaluate CNN-NonUni and CNN-Bern on freshly trained fair CNN-Uni models.

Run AFTER train_cnn_fair.py completes. Uses the same trained encoder/decoder
weights but applies different quantization at inference time.

Also generates merged BER sweep figure for the paper.
"""

import os, sys, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
from train_aid_v2 import ResNet50Front, ResNet50Back, CNNUni, BSC_Channel
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


def eval_cnn_uni(front, model, back, loader, ber):
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            Fp, _ = model(front(imgs), noise_param=ber)
            correct += back(Fp).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total


def eval_cnn_nonuni(front, model, back, loader, ber, n_bits=8):
    from sklearn.cluster import MiniBatchKMeans
    n_levels = 2**n_bits

    # Collect activations for k-means fit
    all_acts = []
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            feat = front(imgs.to(device))
            z = torch.sigmoid(model.ebn2(model.enc2(F.relu(model.ebn1(model.enc1(feat))))))
            all_acts.append(z.cpu().numpy().flatten())
            if i >= 10: break

    flat = np.concatenate(all_acts)
    sub = flat[np.random.RandomState(42).choice(len(flat), min(200000, len(flat)), replace=False)]
    kmeans = MiniBatchKMeans(n_clusters=n_levels, n_init=3, max_iter=50,
                             random_state=42, batch_size=10000)
    kmeans.fit(sub.reshape(-1, 1))
    levels_t = torch.tensor(np.sort(kmeans.cluster_centers_.flatten()),
                            dtype=torch.float32, device=device)

    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            z = torch.sigmoid(model.ebn2(model.enc2(F.relu(model.ebn1(model.enc1(front(imgs)))))))
            indices = (z.unsqueeze(-1) - levels_t).abs().argmin(dim=-1)

            bits = []
            for b in range(n_bits):
                bits.append(((indices >> b) & 1).float())
            bit_tensor = torch.stack(bits, dim=-1)

            if ber > 0:
                bit_tensor = (bit_tensor + (torch.rand_like(bit_tensor) < ber).float()) % 2

            idx_r = sum(bit_tensor[..., b].long() * (2**b) for b in range(n_bits))
            z_recv = levels_t[idx_r.clamp(0, n_levels-1)]

            Fp = model.dbn2(model.dec2(F.relu(model.dbn1(model.dec1(z_recv)))))
            correct += back(Fp).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total


def eval_cnn_bern(front, model, back, loader, ber):
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            z = torch.sigmoid(model.ebn2(model.enc2(F.relu(model.ebn1(model.enc1(front(imgs)))))))
            bits = (torch.rand_like(z) < z).float()
            if ber > 0:
                bits = (bits + (torch.rand_like(bits) < ber).float()) % 2
            Fp = model.dbn2(model.dec2(F.relu(model.dbn1(model.dec1(bits)))))
            correct += back(Fp).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total


def load_best_checkpoint(snap_dir, model, back):
    """Load best CNN-Uni checkpoint (prefer S3)."""
    s3 = sorted([f for f in os.listdir(snap_dir) if f.startswith("cnnuni_s3_")])
    best = sorted([f for f in os.listdir(snap_dir) if f.startswith("cnnuni_best_")])
    ckf = s3[-1] if s3 else best[-1]
    ck = torch.load(f"{snap_dir}/{ckf}", map_location=device)
    model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
    model.eval(); back.eval()
    return ckf


def run_dataset(dataset_name, n_classes, backbone_path, snap_dir, test_ds):
    """Run all 3 CNN baselines on one dataset."""
    print(f"\n{'='*60}")
    print(f"Evaluating CNN baselines on {dataset_name}")
    print(f"{'='*60}")

    front = ResNet50Front(grid_size=14).to(device)
    bb = torch.load(backbone_path, map_location=device)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))},
                          strict=False)
    front.eval()

    model = CNNUni(C_in=1024, C1=256, C2=36, n_bits=8).to(device)
    back = ResNet50Back(n_classes).to(device)
    ckf = load_best_checkpoint(snap_dir, model, back)
    print(f"  Checkpoint: {ckf}")

    loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)
    print(f"  Test: {len(test_ds)} images")

    ber_list = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    results = {}

    for method_name, eval_fn in [('CNN-Uni', eval_cnn_uni),
                                  ('CNN-NonUni', eval_cnn_nonuni),
                                  ('CNN-Bern', eval_cnn_bern)]:
        print(f"\n  {method_name}:")
        accs = []
        for ber in ber_list:
            acc = eval_fn(front, model, back, loader, ber)
            accs.append(acc)
            print(f"    BER={ber:.2f}: {acc:.2f}%")
        results[method_name] = {str(b): a for b, a in zip(ber_list, accs)}

    return results


def generate_merged_figure(all_results, snn_results):
    """Generate merged BER sweep figure: SNN vs CNN baselines, both datasets."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 9, 'figure.dpi': 300,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })

    ber_list = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=False)

    colors = {'V5C-NA': '#1B5E20', 'SNN-SC': '#4CAF50',
              'CNN-Uni': '#E53935', 'CNN-NonUni': '#FF9800', 'CNN-Bern': '#9C27B0'}
    markers = {'V5C-NA': 'o', 'SNN-SC': 's',
               'CNN-Uni': '^', 'CNN-NonUni': 'D', 'CNN-Bern': 'v'}

    for ax, ds_name in [(ax1, 'AID'), (ax2, 'RESISC45')]:
        # SNN results
        for method in ['V5C-NA', 'SNN-SC']:
            if method in snn_results.get(ds_name, {}):
                accs = [snn_results[ds_name][method][str(b)] for b in ber_list]
                ax.plot(ber_list, accs, f'-{markers[method]}', color=colors[method],
                        lw=2.0, ms=5, label=method, zorder=3)

        # CNN results
        if ds_name in all_results:
            for method in ['CNN-Uni', 'CNN-NonUni', 'CNN-Bern']:
                if method in all_results[ds_name]:
                    accs = [all_results[ds_name][method][str(b)] for b in ber_list]
                    ax.plot(ber_list, accs, f'--{markers[method]}', color=colors[method],
                            lw=1.5, ms=4, label=method, zorder=2)

        ax.set_xlabel('Bit Error Rate')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'({chr(97 + [ax1,ax2].index(ax))}) {ds_name}', fontweight='bold')
        ax.legend(fontsize=7, loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.01, 0.31)

    plt.tight_layout()
    plt.savefig('paper/figures/fig_ber_sweep_all_baselines.pdf')
    plt.savefig('paper/figures/fig_ber_sweep_all_baselines.png')
    plt.close()
    print("  ✅ Merged BER sweep figure saved")


def main():
    print(f"Device: {device}")

    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])

    all_results = {}

    # AID
    aid_test = AIDDataset5050(root="./data", transform=tf_test, split='test', seed=42)
    all_results['AID'] = run_dataset(
        "AID (50/50)", 30,
        "./snapshots_aid_5050_seed42/backbone_best.pth",
        "./snapshots_cnnuni_aid_5050",
        aid_test
    )

    # RESISC45
    res_test = RESISC45_2080(root="./data", transform=tf_test, split='test', seed=42)
    all_results['RESISC45'] = run_dataset(
        "RESISC45 (20/80)", 45,
        "./snapshots_resisc45_5050_seed42/backbone_best.pth",
        "./snapshots_cnnuni_resisc45_2080",
        res_test
    )

    # Save
    with open("eval/cnn_fair_eval.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # Load SNN results for the merged figure
    snn_results = {}
    if os.path.exists("eval/multichannel_results_v2.json"):
        with open("eval/multichannel_results_v2.json") as f:
            mc = json.load(f)
        for ds in ['aid', 'resisc45']:
            ds_key = 'AID' if ds == 'aid' else 'RESISC45'
            bsc = mc[ds]['bsc']['results']
            snn_results[ds_key] = {
                'V5C-NA': bsc,
                'SNN-SC': {str(b): bsc[str(b)] for b in [0.0,0.05,0.1,0.15,0.2,0.25,0.3]}
            }

    generate_merged_figure(all_results, snn_results)

    # Print paper table format
    print(f"\n{'='*80}")
    print("PAPER TABLE — CNN Baselines (both datasets)")
    print(f"{'='*80}")
    for ds, res in all_results.items():
        print(f"\n{ds}:")
        print(f"  {'Method':<16s}  {'Clean':>6s}  {'0.15':>6s}  {'0.30':>6s}")
        for name, accs in res.items():
            print(f"  {name:<16s}  {accs['0.0']:6.2f}  {accs['0.15']:6.2f}  {accs['0.3']:6.2f}")

    print(f"\n✅ CNN eval complete. Results: eval/cnn_fair_eval.json")


if __name__ == '__main__':
    main()
