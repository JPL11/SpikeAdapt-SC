#!/usr/bin/env python3
"""Retrain RESISC45 scorer (S3) with higher LR and more epochs.

The original S3 training used scorer LR=1e-5 (same as encoder/decoder),
which was too low for RESISC45's 45-class spatial complexity. This script
retrains S3 from the S2-pretrained encoder/decoder with:
  - Scorer LR: 1e-4 (10x higher than original)
  - Epochs: 80 (2x original)
  - Block mask LR: 1e-3 (separate, to learn threshold faster)
  - Evaluate mask quality every 10 epochs via 10-draw random comparison
"""
import torch, torch.nn as nn, torch.optim as optim
import sys, os, json, random, numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms as T

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from run_final_pipeline import (SpikeAdaptSC_v5c_NA, RESISC45Dataset,
                                 sample_noise, T_STEPS)
from train_aid_v2 import ResNet50Front, ResNet50Back

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model, back, front, test_loader, ber=0.0, rho=None):
    model.eval(); back.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            kwargs = {'noise_param': ber}
            if rho is not None:
                kwargs['target_rate_override'] = rho
            Fp, _ = model(front(imgs), **kwargs)
            correct += back(Fp).argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total

def evaluate_random_mask(model, front, back, test_loader, ber, rho, n_draws=10):
    """Quick random mask evaluation."""
    model.eval(); back.eval()
    draw_accs = []
    for d in range(n_draws):
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                feat = front(imgs); B = feat.size(0)
                all_S2, m1, m2 = [], None, None
                for t in range(T_STEPS):
                    _, s2, m1, m2 = model.encoder(feat, m1, m2, t=t)
                    all_S2.append(s2)
                n_keep = int(rho * 196)
                mask = torch.zeros(B, 1, 14, 14, device=device)
                for b in range(B):
                    idx = torch.randperm(196, device=device)[:n_keep]
                    mask[b, 0].view(-1)[idx] = 1.0
                recv = [model.channel(all_S2[t] * mask, ber) for t in range(T_STEPS)]
                Fp = model.decoder(recv, mask)
                correct += back(Fp).argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        draw_accs.append(100. * correct / total)
    return np.mean(draw_accs), np.std(draw_accs)


def main():
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    
    tf_train = T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
                           T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))])
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])
    
    train_ds = RESISC45Dataset('./data', tf_train, 'train', train_ratio=0.20, seed=42)
    test_ds = RESISC45Dataset('./data', tf_test, 'test', train_ratio=0.20, seed=42)
    train_loader = DataLoader(train_ds, 32, True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)
    
    # Load backbone
    front = ResNet50Front(grid_size=14).to(device)
    bb_path = './snapshots_resisc45_5050_seed42/backbone_best.pth'
    bb = torch.load(bb_path, map_location=device)
    front.load_state_dict({k:v for k,v in bb.items() 
                           if not k.startswith(('layer4.','fc.','avgpool.','spatial_pool.'))}, strict=False)
    front.eval()
    for p in front.parameters(): p.requires_grad = False
    
    # Load model from existing best checkpoint (has S2+S3 trained weights)
    back = ResNet50Back(45).to(device)
    model = SpikeAdaptSC_v5c_NA(C_in=1024, C1=256, C2=36, T=T_STEPS,
                                 target_rate=0.75, grid_size=14).to(device)
    
    ck_path = './snapshots_resisc45_v5cna_seed42/v5cna_best_92.01.pth'
    ck = torch.load(ck_path, map_location=device)
    model.load_state_dict(ck['model'])
    back.load_state_dict(ck['back'])
    
    # Re-initialize scorer from scratch using the class from run_final_pipeline
    # (NoiseAwareScorer is imported via run_final_pipeline's own imports)
    scorer_cls = type(model.scorer)
    model.scorer = scorer_cls(C_spike=36, hidden=32).to(device)
    
    # Re-initialize block_mask using its own class
    mask_cls = type(model.block_mask)
    model.block_mask = mask_cls(0.75, 0.5).to(device)
    
    print(f"Retraining RESISC45 S3 scorer")
    print(f"  Starting from: {ck_path}")
    print(f"  Scorer: fresh init, LR=1e-4")
    print(f"  Block mask: fresh init, LR=1e-3")
    print(f"  Encoder/decoder: LR=1e-5 (fine-tune)")
    print(f"  Back: LR=1e-5 (fine-tune)")
    print(f"  Epochs: 80, diversity λ=0.05")
    
    # Optimizer with higher LR for scorer
    scorer_params = list(model.scorer.parameters())
    mask_params = list(model.block_mask.parameters())
    encoder_params = list(model.encoder.parameters())
    decoder_params = list(model.decoder.parameters())
    
    optimizer = optim.Adam([
        {'params': scorer_params, 'lr': 1e-4},      # 10x higher for scorer
        {'params': mask_params, 'lr': 1e-3},         # high LR for threshold
        {'params': encoder_params, 'lr': 1e-5},      # gentle fine-tune
        {'params': decoder_params, 'lr': 1e-5},      # gentle fine-tune
        {'params': back.parameters(), 'lr': 1e-5}    # gentle fine-tune
    ])
    
    criterion = nn.CrossEntropyLoss()
    snap_dir = './snapshots_resisc45_v5cna_seed42_retrained/'
    os.makedirs(snap_dir, exist_ok=True)
    
    best_acc = 0
    best_ckpt = None
    
    for epoch in range(1, 81):
        model.train(); back.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            ber = sample_noise('bsc')
            
            Fp, stats = model(feat, noise_param=ber)
            loss = criterion(back(Fp), labels)
            loss = loss + 0.1 * stats.get('rate_penalty', 0)
            
            diversity_loss = model.scorer.compute_diversity_loss(
                stats['all_S2'], ber_low=0.0, ber_high=0.30
            )
            loss = loss + 0.05 * diversity_loss
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        if epoch % 5 == 0 or epoch == 80:
            clean = evaluate(model, back, front, test_loader, ber=0.0)
            ber15 = evaluate(model, back, front, test_loader, ber=0.15)
            ber30 = evaluate(model, back, front, test_loader, ber=0.30)
            
            status = f"  E{epoch:02d}: clean={clean:.2f}% BER=0.15={ber15:.2f}% BER=0.30={ber30:.2f}%"
            
            # Quick mask quality check every 20 epochs
            if epoch % 20 == 0 or epoch == 80:
                learned_75 = evaluate(model, back, front, test_loader, ber=0.30, rho=0.75)
                rand_mean, rand_std = evaluate_random_mask(model, front, back, test_loader, 0.30, 0.75, n_draws=10)
                delta = learned_75 - rand_mean
                status += f" | mask@0.75: L={learned_75:.2f} R={rand_mean:.2f}+-{rand_std:.2f} Δ={delta:+.2f}"
            
            print(status, flush=True)
            
            if clean > best_acc:
                best_acc = clean
                best_ckpt = os.path.join(snap_dir, f'v5cna_best_{clean:.2f}.pth')
                torch.save({'model': model.state_dict(), 'back': back.state_dict()}, best_ckpt)
                print(f"    → saved {best_ckpt}", flush=True)
    
    # Final 50-draw evaluation at both rates
    print(f"\n=== FINAL 50-DRAW EVALUATION ===", flush=True)
    if best_ckpt:
        ck = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
    
    results = {}
    for rho in [0.50, 0.75]:
        results[str(rho)] = {}
        for ber in [0.0, 0.30]:
            learned = evaluate(model, back, front, test_loader, ber=ber, rho=rho)
            rand_mean, rand_std = evaluate_random_mask(model, front, back, test_loader, ber, rho, n_draws=50)
            delta = round(learned - rand_mean, 2)
            results[str(rho)][str(ber)] = {
                'learned': round(learned, 2), 'random_mean': round(rand_mean, 2),
                'random_std': round(rand_std, 2), 'delta_random': delta
            }
            print(f"  rho={rho} ber={ber}: L={learned:.2f} R={rand_mean:.2f}+-{rand_std:.2f} Δ={delta:+.2f}", flush=True)
    
    with open(os.path.join(snap_dir, 'mask_comparison_50draws.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBest checkpoint: {best_ckpt} ({best_acc:.2f}%)")
    print("Done!", flush=True)


if __name__ == '__main__':
    main()
