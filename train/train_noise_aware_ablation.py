"""Noise-Aware Ablation: train S3 variants to isolate BER-branch and diversity-loss effects.

Four variants (all share same S2-pretrained encoder/decoder):
  (a) Full          — BER branch ON,  diversity loss ON   (already trained, skip)
  (b) No-BER        — BER branch OFF, diversity loss ON   (scorer always sees ber=0)
  (c) No-Diversity   — BER branch ON,  diversity loss OFF  (λ_div=0)
  (d) No-Noise-Aware — BER branch OFF, diversity loss OFF  (masking only)

Usage:
  python train/train_noise_aware_ablation.py --dataset aid
  python train/train_noise_aware_ablation.py --dataset resisc45
"""

import os, sys, json, argparse, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from run_final_pipeline import (
    SpikeAdaptSC_v5c_NA, AIDDataset5050, RESISC45Dataset
)
from train_aid_v2 import ResNet50Front, ResNet50Back

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_STEPS = 8


def sample_noise():
    """50% clean, 50% uniform [0.01, 0.30] — same as main pipeline."""
    if random.random() < 0.5:
        return 0.0
    return random.uniform(0.01, 0.30)


def train_s3_variant(variant_name, model, back, front, train_loader, test_loader,
                     use_ber_branch=True, use_diversity_loss=True,
                     epochs=40, save_dir='snapshots_ablation_na'):
    """Train one S3 variant.
    
    Args:
        use_ber_branch: if False, always pass ber=0.0 to scorer (zeroes BER branch)
        use_diversity_loss: if False, skip diversity loss term
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Training: {variant_name}")
    print(f"  BER branch: {'ON' if use_ber_branch else 'OFF'}")
    print(f"  Diversity loss: {'ON' if use_diversity_loss else 'OFF'}")
    print(f"  Epochs: {epochs}")
    print(f"{'='*60}")
    
    criterion = nn.CrossEntropyLoss()
    
    slope_params = [p for n, p in model.named_parameters() if 'slope' in n]
    other_params = [p for n, p in model.named_parameters() if 'slope' not in n]
    optimizer = optim.Adam([
        {'params': other_params, 'lr': 1e-5},
        {'params': slope_params, 'lr': 1e-4},
        {'params': back.parameters(), 'lr': 1e-5}
    ])
    
    best_acc = 0
    results = {'variant': variant_name, 'epochs': [], 'best_acc': 0}
    
    for epoch in range(1, epochs + 1):
        model.train(); back.train()
        t0 = time.time()
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            
            actual_ber = sample_noise()
            
            # Key ablation: if BER branch disabled, scorer always sees 0.0
            scorer_ber = actual_ber if use_ber_branch else 0.0
            
            # Forward: use actual_ber for channel noise, scorer_ber for mask generation
            all_S2, m1, m2 = [], None, None
            for t in range(T_STEPS):
                _, s2, m1, m2 = model.encoder(feat, m1, m2, t=t)
                all_S2.append(s2)
            
            importance = model.scorer(all_S2, scorer_ber).squeeze(1)
            mask, tx = model.block_mask(importance, training=True)
            recv = [model.channel(all_S2[t] * mask, actual_ber) for t in range(T_STEPS)]
            Fp = model.decoder(recv, mask)
            
            loss = criterion(back(Fp), labels)
            loss = loss + 0.1 * (tx - model.block_mask.target_rate).abs()
            
            # Diversity loss (only if enabled)
            if use_diversity_loss:
                diversity_loss = model.scorer.compute_diversity_loss(
                    all_S2, ber_low=0.0, ber_high=0.30
                )
                loss = loss + 0.05 * diversity_loss
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        elapsed = time.time() - t0
        
        # Evaluate every 5 epochs
        if epoch % 5 == 0 or epoch == epochs:
            model.eval(); back.eval()
            accs = {}
            for eval_ber in [0.0, 0.15, 0.30]:
                correct, total = 0, 0
                with torch.no_grad():
                    for imgs, labels in test_loader:
                        imgs, labels = imgs.to(device), labels.to(device)
                        feat = front(imgs)
                        Fp, _ = model(feat, noise_param=eval_ber)
                        correct += back(Fp).argmax(1).eq(labels).sum().item()
                        total += labels.size(0)
                accs[str(eval_ber)] = round(100. * correct / total, 2)
            
            results['epochs'].append({
                'epoch': epoch, 'accs': accs, 'time': round(elapsed, 1)
            })
            
            print(f"  E{epoch:02d}: clean={accs['0.0']:.2f}% "
                  f"BER=0.15={accs['0.15']:.2f}% "
                  f"BER=0.30={accs['0.3']:.2f}% ({elapsed:.0f}s)")
            
            if accs['0.0'] > best_acc:
                best_acc = accs['0.0']
                results['best_acc'] = best_acc
                results['best_accs'] = accs
                torch.save({
                    'model': model.state_dict(),
                    'back': back.state_dict(),
                    'variant': variant_name,
                    'accs': accs,
                }, os.path.join(save_dir, f'{variant_name}_best.pth'))
    
    results['final_accs'] = accs
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='aid', choices=['aid', 'resisc45'])
    parser.add_argument('--epochs', type=int, default=40)
    args = parser.parse_args()
    
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                     T.Normalize((.485,.456,.406), (.229,.224,.225))])
    
    if args.dataset == 'aid':
        n_classes = 30
        train_ds = AIDDataset5050("./data", tf, 'train', seed=42)
        test_ds = AIDDataset5050("./data", tf, 'test', seed=42)
        ckpt_path = "./snapshots_aid_v5cna_seed42/v5cna_best_95.42.pth"
        bb_path = "./snapshots_aid_5050_seed42/backbone_best.pth"
    else:
        n_classes = 45
        train_ds = RESISC45Dataset("./data", tf, 'train', train_ratio=0.20, seed=42)
        test_ds = RESISC45Dataset("./data", tf, 'test', train_ratio=0.20, seed=42)
        ckpt_path = "./snapshots_resisc45_v5cna_seed42/v5cna_best_92.01.pth"
        bb_path = "./snapshots_resisc45_5050_seed42/backbone_best.pth"
    
    train_loader = DataLoader(train_ds, 32, True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)
    
    # Load backbone (frozen)
    front = ResNet50Front(grid_size=14).to(device)
    bb = torch.load(bb_path, map_location=device)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))},
                          strict=False)
    front.eval()
    for p in front.parameters():
        p.requires_grad = False
    
    save_dir = f"snapshots_ablation_na_{args.dataset}"
    all_results = {}
    
    # Variant configs: (name, use_ber_branch, use_diversity_loss)
    variants = [
        ('no_ber', False, True),
        ('no_div', True, False),
        ('no_both', False, False),
    ]
    
    for var_name, use_ber, use_div in variants:
        torch.manual_seed(42); np.random.seed(42); random.seed(42)
        
        # Load fresh model from S2+S3 checkpoint for each variant
        # We reload to start from the SAME S2-pretrained encoder weights
        model = SpikeAdaptSC_v5c_NA(C_in=1024, C1=256, C2=36, T=T_STEPS,
                                     target_rate=0.75, grid_size=14).to(device)
        back = ResNet50Back(n_classes).to(device)
        
        ck = torch.load(ckpt_path, map_location=device)
        # Load S2-pretrained encoder/decoder weights, but reinit scorer
        state = ck['model']
        # Only load encoder and decoder weights (not scorer)
        encoder_decoder_state = {k: v for k, v in state.items()
                                  if k.startswith(('encoder.', 'decoder.', 'block_mask.', 'channel.'))}
        model.load_state_dict(encoder_decoder_state, strict=False)
        back.load_state_dict(ck['back'])
        
        result = train_s3_variant(
            var_name, model, back, front, train_loader, test_loader,
            use_ber_branch=use_ber, use_diversity_loss=use_div,
            epochs=args.epochs, save_dir=save_dir
        )
        all_results[var_name] = result
        
        del model, back
        torch.cuda.empty_cache()
    
    # Also evaluate the full (already trained) model
    print(f"\n{'='*60}")
    print("Evaluating: full (already trained)")
    print(f"{'='*60}")
    model = SpikeAdaptSC_v5c_NA(C_in=1024, C1=256, C2=36, T=T_STEPS,
                                 target_rate=0.75, grid_size=14).to(device)
    back = ResNet50Back(n_classes).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ck['model'])
    back.load_state_dict(ck['back'])
    model.eval(); back.eval()
    
    full_accs = {}
    for eval_ber in [0.0, 0.15, 0.30]:
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                Fp, _ = model(front(imgs), noise_param=eval_ber)
                correct += back(Fp).argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        full_accs[str(eval_ber)] = round(100. * correct / total, 2)
    
    all_results['full'] = {'variant': 'full', 'best_accs': full_accs, 'final_accs': full_accs}
    print(f"  Full: clean={full_accs['0.0']:.2f}% "
          f"BER=0.15={full_accs['0.15']:.2f}% "
          f"BER=0.30={full_accs['0.3']:.2f}%")
    
    # Measure firing rate
    fr_sum, n_b = 0, 0
    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.to(device)
            Fp, stats = model(front(imgs), noise_param=0.0)
            fr_sum += stats['firing_rate']; n_b += 1
    all_results['measured_firing_rate'] = round(fr_sum / n_b, 4)
    print(f"\nMeasured firing rate: {fr_sum/n_b:.4f}")
    
    # Summary table
    print(f"\n{'='*60}")
    print("NOISE-AWARE ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Variant':<20} {'BER':>5} {'Div':>5} {'Clean':>8} {'BER=0.15':>10} {'BER=0.30':>10}")
    print('-' * 60)
    for var_name in ['full', 'no_ber', 'no_div', 'no_both']:
        r = all_results[var_name]
        accs = r.get('final_accs', r.get('best_accs', {}))
        ber_on = '✓' if var_name in ('full', 'no_div') else '✗'
        div_on = '✓' if var_name in ('full', 'no_ber') else '✗'
        print(f"{var_name:<20} {ber_on:>5} {div_on:>5} "
              f"{accs.get('0.0', 0):>8.2f} {accs.get('0.15', 0):>10.2f} {accs.get('0.3', 0):>10.2f}")
    
    # Save
    out_path = f"eval/noise_aware_ablation_{args.dataset}.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved to {out_path}")


if __name__ == '__main__':
    main()
