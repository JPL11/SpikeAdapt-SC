#!/usr/bin/env python3
"""Adaptive-Rate JSCC Baseline: JSCC with learned spatial masking.

Adds an SNR-conditioned scorer + spatial masking to the existing JSCC model.
This creates a fair comparison: both SpikeAdapt-SC and Adaptive-JSCC have
learned spatial masking, but one uses binary spikes and the other continuous.

Architecture:
  ResNet-50 front → JSCC_Encoder → Scorer → Mask → AWGN/BSC → Decoder → Back

Training: AWGN-only (JSCC's native channel), 30 epochs, scorer only.
Eval: AWGN sweep + BSC sweep at ρ=0.75

Output: eval/seed_results/jscc_adaptive_results.json
Usage:  python train/train_jscc_adaptive.py
"""

import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
import sys, os, json, random, math, numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms as T

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from run_final_pipeline import AIDDataset5050, RESISC45Dataset
from train_aid_v2 import ResNet50Front, ResNet50Back
from train_jscc_baseline import JSCC_Encoder, JSCC_Decoder, AWGN_Channel_Continuous, BSC_Channel_Quantized

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SNRAwareScorer(nn.Module):
    """Importance scorer conditioned on SNR (continuous channel equivalent of NoiseAwareScorer).
    
    Same architecture as NoiseAwareScorer but conditioned on SNR_dB instead of BER.
    """
    def __init__(self, C_tx=36, hidden=32):
        super().__init__()
        self.content_branch = nn.Sequential(
            nn.Conv2d(C_tx, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1)
        )
        self.noise_channel_gate = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, C_tx)
        )
        nn.init.zeros_(self.noise_channel_gate[-1].weight)
        nn.init.zeros_(self.noise_channel_gate[-1].bias)
        
        self.noise_spatial_bias = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )
        nn.init.zeros_(self.noise_spatial_bias[-1].weight)
        nn.init.zeros_(self.noise_spatial_bias[-1].bias)
    
    def forward(self, z, snr_db):
        """Score spatial blocks based on encoded features + SNR."""
        B = z.size(0)
        # Normalize SNR to [0, 1] range for MLP: SNR_norm = (snr_db + 10) / 30
        snr_norm = (snr_db + 10.0) / 30.0
        snr_input = torch.full((B, 1), snr_norm, device=z.device)
        
        channel_weights = 1.0 + torch.tanh(self.noise_channel_gate(snr_input))
        reweighted = z * channel_weights.unsqueeze(-1).unsqueeze(-1)
        importance = self.content_branch(reweighted)
        spatial_bias = self.noise_spatial_bias(snr_input)
        importance = importance + spatial_bias.unsqueeze(-1).unsqueeze(-1)
        return torch.sigmoid(importance)
    
    def compute_diversity_loss(self, z, snr_high=20.0, snr_low=-4.0):
        mask_high = self.forward(z, snr_high)
        mask_low = self.forward(z, snr_low)
        return -torch.mean((mask_high - mask_low).pow(2))


class LearnedBlockMask(nn.Module):
    """Same masking module as SpikeAdapt-SC."""
    def __init__(self, target_rate=0.75, temperature=0.5):
        super().__init__()
        self.target_rate = target_rate
        self.temperature = temperature

    def forward(self, importance, training=True):
        B, _, H, W = importance.shape
        if training:
            logits = torch.log(importance / (1 - importance + 1e-7) + 1e-7).squeeze(1)
            u = torch.rand_like(logits).clamp(1e-7, 1 - 1e-7)
            soft = torch.sigmoid((logits - torch.log(-torch.log(u))) / self.temperature)
            hard = (soft > 0.5).float()
            mask = hard + (soft - soft.detach())
        else:
            flat = importance.view(B, -1)
            k = max(1, int(self.target_rate * H * W))
            _, idx = flat.topk(k, dim=1)
            mask = torch.zeros_like(flat)
            mask.scatter_(1, idx, 1.0)
            mask = mask.view(B, H, W)
        return mask.unsqueeze(1), mask.mean()


class AdaptiveJSCC(nn.Module):
    """JSCC model with adaptive spatial masking."""
    def __init__(self, C_in=1024, C_mid=256, C_tx=36, target_rate=0.75):
        super().__init__()
        self.encoder = JSCC_Encoder(C_in, C_mid, C_tx)
        self.scorer = SNRAwareScorer(C_tx, hidden=32)
        self.block_mask = LearnedBlockMask(target_rate, 0.5)
        self.awgn = AWGN_Channel_Continuous()
        self.bsc = BSC_Channel_Quantized()
        self.decoder = JSCC_Decoder(C_in, C_mid, C_tx)
    
    def forward(self, feat, snr_db=20.0, channel='awgn', ber=0.0, 
                target_rate_override=None):
        z_norm, power = self.encoder(feat)
        
        # Score and mask
        importance = self.scorer(z_norm, snr_db)
        
        if target_rate_override is not None:
            old = self.block_mask.target_rate
            self.block_mask.target_rate = target_rate_override
            mask, tx = self.block_mask(importance, training=False)
            self.block_mask.target_rate = old
        else:
            mask, tx = self.block_mask(importance, training=self.training)
        
        z_masked = z_norm * mask
        
        if channel == 'awgn':
            z_recv = self.awgn(z_masked, snr_db)
        elif channel == 'bsc':
            z_recv = self.bsc(z_masked, ber)
        else:
            z_recv = z_masked
        
        Fp = self.decoder(z_recv, power)
        return Fp, {'tx_rate': tx.item(), 'mask': mask, 'importance': importance}


def train_adaptive_jscc(dataset_name, n_classes, seed):
    """Train adaptive JSCC for one seed."""
    print(f"\n{'='*60}")
    print(f"  Adaptive JSCC {dataset_name.upper()} seed={seed}")
    print(f"{'='*60}", flush=True)
    
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    tf_train = T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
                           T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))])
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])
    
    if dataset_name == 'aid':
        train_ds = AIDDataset5050('./data', tf_train, 'train', seed=seed)
        test_ds = AIDDataset5050('./data', tf_test, 'test', seed=seed)
    else:
        train_ds = RESISC45Dataset('./data', tf_train, 'train', train_ratio=0.20, seed=seed)
        test_ds = RESISC45Dataset('./data', tf_test, 'test', train_ratio=0.20, seed=seed)
    
    train_loader = DataLoader(train_ds, 32, True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)
    
    # Load backbone
    bb_path = f'./snapshots_{dataset_name}_5050_seed{seed}/backbone_best.pth'
    if not os.path.exists(bb_path):
        print(f"  ERROR: backbone not found: {bb_path}")
        return None
    
    front = ResNet50Front(grid_size=14).to(device)
    bb = torch.load(bb_path, map_location=device, weights_only=False)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))}, strict=False)
    front.eval()
    for p in front.parameters(): p.requires_grad = False
    
    back = ResNet50Back(n_classes).to(device)
    
    # Load pretrained fixed JSCC encoder/decoder weights
    jscc_snap_dir = f'./snapshots_{dataset_name}_jscc_seed{seed}/'
    jscc_cks = sorted([f for f in os.listdir(jscc_snap_dir) if f.startswith('jscc_best_')],
                      key=lambda x: float(x.split('_')[-1].replace('.pth', '')))
    if not jscc_cks:
        print(f"  ERROR: no JSCC checkpoint found in {jscc_snap_dir}")
        return None
    
    jscc_ck = torch.load(os.path.join(jscc_snap_dir, jscc_cks[-1]), 
                         map_location=device, weights_only=False)
    
    # Create adaptive model and load JSCC weights
    model = AdaptiveJSCC(C_in=1024, C_mid=256, C_tx=36, target_rate=0.75).to(device)
    
    # Load encoder/decoder from pretrained JSCC
    enc_state = {k.replace('encoder.', ''): v for k, v in jscc_ck['model'].items() 
                 if k.startswith('encoder.')}
    dec_state = {k.replace('decoder.', ''): v for k, v in jscc_ck['model'].items() 
                 if k.startswith('decoder.')}
    model.encoder.encoder.load_state_dict(enc_state)
    model.decoder.decoder.load_state_dict(dec_state)
    back.load_state_dict(jscc_ck['back'])
    
    # Freeze encoder/decoder, only train scorer
    for p in model.encoder.parameters(): p.requires_grad = False
    for p in model.decoder.parameters(): p.requires_grad = False
    
    snap_dir = f'./snapshots_{dataset_name}_jscc_adaptive_seed{seed}/'
    os.makedirs(snap_dir, exist_ok=True)
    
    existing = sorted([f for f in os.listdir(snap_dir) if f.startswith('adaptive_best_')],
                      key=lambda x: float(x.split('_')[-1].replace('.pth', ''))) if os.path.exists(snap_dir) else []
    
    if not existing:
        # Train scorer only, 30 epochs
        optimizer = optim.AdamW(
            list(model.scorer.parameters()) + list(model.block_mask.parameters()) + list(back.parameters()),
            lr=1e-3, weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0
        print(f"  Training scorer (30 epochs)...", flush=True)
        for epoch in range(1, 31):
            model.train(); back.train()
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                feat = front(imgs)
                snr = random.choice([100, 20, 10, 5, 2, 0, -2, -4])
                Fp, info = model(feat, snr_db=snr, channel='awgn')
                
                loss_ce = criterion(back(Fp), labels)
                loss_rate = (info['tx_rate'] - 0.75) ** 2
                
                # Diversity loss: masks should differ at high vs low SNR
                with torch.no_grad():
                    z_norm, _ = model.encoder(feat)
                loss_div = model.scorer.compute_diversity_loss(z_norm.detach())
                
                loss = loss_ce + 0.5 * loss_rate + 0.05 * loss_div
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            scheduler.step()
            
            if epoch % 5 == 0 or epoch == 30:
                model.eval(); back.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for imgs, labels in test_loader:
                        imgs, labels = imgs.to(device), labels.to(device)
                        Fp, _ = model(front(imgs), snr_db=100, channel='awgn')
                        correct += back(Fp).argmax(1).eq(labels).sum().item()
                        total += labels.size(0)
                acc = 100. * correct / total
                print(f"  E{epoch:02d}: {acc:.2f}%", flush=True)
                if acc > best_acc:
                    best_acc = acc
                    torch.save({'model': model.state_dict(), 'back': back.state_dict()},
                               os.path.join(snap_dir, f'adaptive_best_{acc:.2f}.pth'))
        
        existing = sorted([f for f in os.listdir(snap_dir) if f.startswith('adaptive_best_')],
                          key=lambda x: float(x.split('_')[-1].replace('.pth', '')))
    else:
        print(f"  Found existing: {existing[-1]}", flush=True)
    
    # Load best
    ck = torch.load(os.path.join(snap_dir, existing[-1]), map_location=device, weights_only=False)
    model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
    model.eval(); back.eval()
    
    # Evaluate
    def eval_acc(rho, channel='awgn', snr_db=100, ber=0.0):
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                Fp, _ = model(front(imgs), snr_db=snr_db, channel=channel, ber=ber,
                             target_rate_override=rho)
                correct += back(Fp).argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        return round(100. * correct / total, 2)
    
    results = {}
    
    # Clean
    results['clean_r075'] = eval_acc(0.75)
    results['clean_r100'] = eval_acc(1.0)
    print(f"  Clean ρ=0.75: {results['clean_r075']}%, ρ=1.0: {results['clean_r100']}%")
    
    # AWGN sweep at ρ=0.75
    results['awgn_r075'] = {}
    for snr in [10, 5, 0, -2, -4, -6, -8]:
        acc = eval_acc(0.75, 'awgn', snr)
        results['awgn_r075'][str(snr)] = acc
    print(f"  AWGN ρ=0.75 SNR=-8dB: {results['awgn_r075']['-8']}%")
    
    # BSC sweep at ρ=0.75
    results['bsc_r075'] = {}
    for ber in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        acc = eval_acc(0.75, 'bsc', ber=ber)
        results['bsc_r075'][str(ber)] = acc
    print(f"  BSC ρ=0.75 BER=0.30: {results['bsc_r075']['0.3']}%")
    
    # BSC sweep at ρ=1.0 (no masking, for comparison with fixed JSCC)
    results['bsc_r100'] = {}
    for ber in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        acc = eval_acc(1.0, 'bsc', ber=ber)
        results['bsc_r100'][str(ber)] = acc
    print(f"  BSC ρ=1.0  BER=0.30: {results['bsc_r100']['0.3']}%")
    
    del model, back, front; torch.cuda.empty_cache()
    return results


def main():
    SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]
    os.makedirs('eval/seed_results', exist_ok=True)
    
    all_results = {}
    for ds_name, n_classes in [('aid', 30), ('resisc45', 45)]:
        all_results[ds_name] = {}
        for seed in SEEDS:
            result = train_adaptive_jscc(ds_name, n_classes, seed)
            if result:
                all_results[ds_name][str(seed)] = result
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  ADAPTIVE JSCC SUMMARY")
    print(f"{'='*60}")
    
    summary = {}
    for ds in all_results:
        seeds_done = list(all_results[ds].keys())
        if not seeds_done:
            continue
        
        clean_075 = [all_results[ds][s]['clean_r075'] for s in seeds_done]
        bsc_030_075 = [all_results[ds][s]['bsc_r075']['0.3'] for s in seeds_done]
        bsc_030_100 = [all_results[ds][s]['bsc_r100']['0.3'] for s in seeds_done]
        awgn_m8_075 = [all_results[ds][s]['awgn_r075']['-8'] for s in seeds_done]
        
        summary[ds] = {
            'clean_r075': {'mean': round(np.mean(clean_075), 2), 'std': round(np.std(clean_075), 2)},
            'bsc_030_r075': {'mean': round(np.mean(bsc_030_075), 2), 'std': round(np.std(bsc_030_075), 2)},
            'bsc_030_r100': {'mean': round(np.mean(bsc_030_100), 2), 'std': round(np.std(bsc_030_100), 2)},
            'awgn_m8_r075': {'mean': round(np.mean(awgn_m8_075), 2), 'std': round(np.std(awgn_m8_075), 2)},
        }
        
        print(f"  {ds.upper()}:")
        print(f"    Clean ρ=0.75: {np.mean(clean_075):.2f} ± {np.std(clean_075):.2f}")
        print(f"    BSC BER=0.30 ρ=0.75: {np.mean(bsc_030_075):.2f} ± {np.std(bsc_030_075):.2f}")
        print(f"    BSC BER=0.30 ρ=1.0 : {np.mean(bsc_030_100):.2f} ± {np.std(bsc_030_100):.2f}")
        print(f"    AWGN SNR=-8 ρ=0.75 : {np.mean(awgn_m8_075):.2f} ± {np.std(awgn_m8_075):.2f}")
    
    with open('eval/seed_results/jscc_adaptive_results.json', 'w') as f:
        json.dump({'per_seed': all_results, 'summary': summary}, f, indent=2)
    
    print(f"\n✅ Saved to eval/seed_results/jscc_adaptive_results.json")


if __name__ == '__main__':
    main()
