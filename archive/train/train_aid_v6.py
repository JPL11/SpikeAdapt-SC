"""SpikeAdapt-SC v6 — Membrane Shortcut Architecture.

Based on literature review (SEW-ResNet, MS-ResNet):
  - V6 = V5C (MPBN) + encoder→decoder membrane shortcut
  - Skip: enc.L1 membrane (256ch) → 1×1 conv → decoder L2 input
  - Bypasses noisy channel for structural features
  - Helps decoder reconstruct at low ρ where spatial info is lost

Architecture: V5C-MPBN base + membrane projection shortcut.

Usage:
  python train/train_aid_v6.py --config v6a  # membrane shortcut only
  python train/train_aid_v6.py --config v6b  # membrane shortcut + MPBN
"""

import os, sys, argparse, random, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from train_aid_v2 import (
    AIDDataset, ResNet50Front, ResNet50Back, BSC_Channel,
    ChannelConditionedScorer, LearnedBlockMask, sample_noise
)
from train_aid_v5 import (
    SpikeFunction_Learnable, LIFNeuron, BNTT, MPBN, EncoderV5
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_STEPS = 8


# ############################################################################
# V6 DECODER with Membrane Shortcut
# ############################################################################

class DecoderV6(nn.Module):
    """Decoder with membrane shortcut from encoder L1.
    
    The encoder's L1 membrane potential (C1=256, 14×14) captures fine-grained
    spatial features that are lost through the bottleneck (C2=36) and noisy 
    channel. We project encoder_m1 (256ch) to decoder input and add it to the
    decoder's L1 output, providing a "structural highway" that bypasses noise.
    
    The shortcut is gated by a learnable scalar (initialized to 0) so the 
    network can gradually learn to use it without disrupting pretrained weights.
    """
    
    def __init__(self, C_in=1024, C1=256, C2=36, T=8, use_mpbn=True, shortcut_from=256):
        super().__init__()
        self.T = T; self.use_mpbn = use_mpbn
        
        # Standard decoder layers
        self.conv3 = nn.Conv2d(C2, C1, 1)
        self.conv4 = nn.Conv2d(C1, C_in, 1)  # C1→C_in (256→1024)
        
        # LIF neurons
        self.if3 = LIFNeuron(C1)
        self.ihf = LIFNeuron(C_in)
        
        # BN layers
        if use_mpbn:
            self.bn3 = nn.BatchNorm2d(C1)
            self.bn4 = nn.BatchNorm2d(C_in)
            self.mpbn3 = MPBN(C1, T)
            self.mpbn4 = MPBN(C_in, T)
        else:
            self.bn3 = BNTT(C1, T)
            self.bn4 = BNTT(C_in, T)
        
        # Membrane shortcut: project enc_m1 (C1=256) → dec_L3 input (C_in=1024)
        # Using a simple 1×1 conv + learnable gate initialized to 0
        self.shortcut_proj = nn.Sequential(
            nn.Conv2d(shortcut_from, C_in, 1, bias=False),
            nn.BatchNorm2d(C_in)
        )
        # Gate starts at 0 (sigmoid(-5) ≈ 0.007) - minimal initial contribution
        self.shortcut_gate = nn.Parameter(torch.tensor(-5.0))
        
        # Output converter
        self.converter_fc = nn.Linear(2 * T, 1, bias=True)
    
    def forward(self, recv_all, mask, enc_m1_history=None):
        """Forward pass with optional membrane shortcut.
        
        Args:
            recv_all: List of T received spike tensors
            mask: Spatial mask
            enc_m1_history: List of T encoder L1 membrane potentials (for shortcut)
        """
        T_use = len(recv_all)
        m3, m4 = None, None
        Fs, Fm = [], []
        
        gate = torch.sigmoid(self.shortcut_gate)
        
        for t in range(T_use):
            inp = recv_all[t] * mask
            
            # Decoder L1
            if self.use_mpbn:
                h3 = self.bn3(self.conv3(inp))
                if m3 is None: m3 = torch.zeros_like(h3)
                beta3 = torch.sigmoid(self.if3.beta_raw)
                m3 = beta3 * m3 + h3
                m3 = self.mpbn3(m3, t)
                s3 = SpikeFunction_Learnable.apply(m3, self.if3.threshold, self.if3.slope)
                m3 = m3 - s3 * self.if3.threshold
            else:
                h3 = self.bn3(self.conv3(inp), t)
                s3, m3 = self.if3(h3, m3)
            
            # Decoder L2 (with membrane shortcut)
            if self.use_mpbn:
                h4 = self.bn4(self.conv4(s3))
                if m4 is None: m4 = torch.zeros_like(h4)
                beta4 = torch.sigmoid(self.ihf.beta_raw)
                m4 = beta4 * m4 + h4
                
                # Add membrane shortcut: encoder m1 → projected to 1024ch
                if enc_m1_history is not None and t < len(enc_m1_history):
                    shortcut = self.shortcut_proj(enc_m1_history[t])
                    m4 = m4 + gate * shortcut
                
                m4 = self.mpbn4(m4, t)
                sp = SpikeFunction_Learnable.apply(m4, self.ihf.threshold, self.ihf.slope)
                m4 = m4 - sp * self.ihf.threshold
            else:
                h4 = self.bn4(self.conv4(s3), t)
                sp, m4 = self.ihf(h4, m4)
                # Add shortcut after neuron update (for non-MPBN)
                if enc_m1_history is not None and t < len(enc_m1_history):
                    shortcut = self.shortcut_proj(enc_m1_history[t])
                    m4 = m4 + gate * shortcut
            
            Fs.append(sp); Fm.append(m4.clone())
        
        while len(Fs) < self.T:
            Fs.append(torch.zeros_like(Fs[0]))
            Fm.append(torch.zeros_like(Fm[0]))
        
        il = []
        for t in range(self.T):
            il.append(Fs[t]); il.append(Fm[t])
        x = torch.stack(il, 1).permute(0, 2, 3, 4, 1)
        return (x * torch.sigmoid(self.converter_fc(x))).sum(-1)


# ############################################################################
# V6 MODEL
# ############################################################################

class SpikeAdaptSC_v6(nn.Module):
    """V6: V5C (MPBN) + membrane shortcut from encoder to decoder."""
    
    def __init__(self, C_in=1024, C1=256, C2=36, T=8,
                 target_rate=0.75, grid_size=14, use_mpbn=True):
        super().__init__()
        self.T = T; self.C2 = C2; self.grid_size = grid_size
        
        self.encoder = EncoderV5(C_in, C1, C2, T, use_mpbn=use_mpbn)
        self.scorer = ChannelConditionedScorer(C_spike=C2, hidden=32)
        self.block_mask = LearnedBlockMask(target_rate, 0.5)
        self.decoder = DecoderV6(C_in, C1, C2, T, use_mpbn=use_mpbn, shortcut_from=C1)
        self.channel = BSC_Channel()
    
    def forward(self, feat, noise_param=0.0, target_rate_override=None):
        # Encoder: collect both spikes and membrane potentials
        all_S2, all_m1 = [], []
        m1, m2 = None, None
        for t in range(self.T):
            s1, s2, m1, m2 = self.encoder(feat, m1, m2, t=t)
            all_S2.append(s2)
            all_m1.append(m1.detach().clone())  # detach shortcut from encoder grad
        
        # Importance scoring + masking
        importance = self.scorer(all_S2, noise_param)
        
        if target_rate_override is not None:
            old = self.block_mask.target_rate
            self.block_mask.target_rate = target_rate_override
            mask, tx = self.block_mask(importance, training=False)
            self.block_mask.target_rate = old
        else:
            mask, tx = self.block_mask(importance, training=self.training)
        
        # Channel transmission
        recv = [self.channel(all_S2[t] * mask, noise_param) for t in range(self.T)]
        
        # Decoder with membrane shortcut
        Fp = self.decoder(recv, mask, enc_m1_history=all_m1)
        
        # Compute firing rate
        with torch.no_grad():
            all_spikes = torch.stack(all_S2)
            fr = all_spikes.mean().item()
        
        gate_val = torch.sigmoid(self.decoder.shortcut_gate).item()
        
        return Fp, {
            'tx_rate': tx.item(), 'mask': mask, 'importance': importance,
            'firing_rate': fr, 'shortcut_gate': gate_val,
        }


# ############################################################################
# TRAINING
# ############################################################################

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='v6b', choices=['v6a', 'v6b'],
                   help='v6a: shortcut only, v6b: shortcut + MPBN')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--dataset', type=str, default='aid', choices=['aid', 'ucm'])
    return p.parse_args()


def train_v6(args):
    use_mpbn = (args.config == 'v6b')
    n_classes = 30 if args.dataset == 'aid' else 21
    
    print(f"\n{'='*60}")
    print(f"V6 {args.config.upper()} on {args.dataset.upper()}")
    print(f"  MPBN={use_mpbn}, membrane shortcut=True")
    print(f"{'='*60}")
    
    # Data
    tf_train = T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
                           T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))])
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])
    
    if args.dataset == 'aid':
        train_ds = AIDDataset("./data", tf_train, split='train', seed=args.seed)
        test_ds = AIDDataset("./data", tf_test, split='test', seed=args.seed)
        bb_path = "./snapshots_aid/backbone_best.pth"
        v4_dir = f"./snapshots_v4_V4A_seed{args.seed}/"
    else:
        sys.path.insert(0, os.path.dirname(__file__))
        from train_ucm import UCMDataset
        train_ds = UCMDataset("./data", tf_train, split='train', seed=args.seed)
        test_ds = UCMDataset("./data", tf_test, split='test', seed=args.seed)
        bb_path = "./snapshots_ucm_seed42/backbone_best.pth"
        v4_dir = f"./snapshots_ucm_v4a_seed{args.seed}/"
    
    train_loader = DataLoader(train_ds, 32, True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)
    
    # Front
    front = ResNet50Front(grid_size=14).to(device)
    bb = torch.load(bb_path, map_location=device)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))}, strict=False)
    front.eval()
    for p in front.parameters(): p.requires_grad = False
    
    back = ResNet50Back(n_classes).to(device)
    
    # Create V6 model
    model = SpikeAdaptSC_v6(C_in=1024, C1=256, C2=36, T=T_STEPS,
                             target_rate=0.75, grid_size=14,
                             use_mpbn=use_mpbn).to(device)
    
    # Transfer from V4-A (or V5C if available)
    v4_files = sorted([f for f in os.listdir(v4_dir) if f.startswith("v4")]) if os.path.exists(v4_dir) else []
    if v4_files:
        ck = torch.load(os.path.join(v4_dir, v4_files[-1]), map_location=device)
        m_state = model.state_dict()
        transferred = 0
        for k, v in ck['model'].items():
            if k in m_state and m_state[k].shape == v.shape:
                m_state[k] = v; transferred += 1
        model.load_state_dict(m_state, strict=False)
        back.load_state_dict(ck['back'])
        print(f"  ✓ Transferred {transferred} params from V4-A")
    
    snap_dir = f"./snapshots_v6_{args.config}_seed{args.seed}/"
    os.makedirs(snap_dir, exist_ok=True)
    
    # Optimizer
    shortcut_params = [p for n, p in model.named_parameters() 
                       if 'shortcut' in n]
    slope_params = [p for n, p in model.named_parameters() 
                    if 'slope' in n and 'shortcut' not in n]
    other_params = [p for n, p in model.named_parameters() 
                    if 'slope' not in n and 'shortcut' not in n]
    
    optimizer = optim.Adam([
        {'params': other_params, 'lr': args.lr},
        {'params': slope_params, 'lr': 1e-4},
        {'params': shortcut_params, 'lr': 1e-4},  # separate lr for shortcut
        {'params': back.parameters(), 'lr': args.lr}
    ])
    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    
    for epoch in range(1, args.epochs + 1):
        model.train(); back.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            ber = sample_noise('bsc')
            Fp, stats = model(feat, noise_param=ber)
            loss = criterion(back(Fp), labels)
            loss = loss + 0.1 * stats.get('rate_penalty', 0)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        if epoch % 5 == 0 or epoch == args.epochs:
            model.eval(); back.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    Fp, stats = model(front(imgs), noise_param=0.0)
                    correct += back(Fp).argmax(1).eq(labels).sum().item()
                    total += labels.size(0)
            acc = 100. * correct / total
            gate_val = stats.get('shortcut_gate', 0)
            fr = stats.get('firing_rate', 0)
            print(f"  E{epoch}: {acc:.2f}%, FR={fr:.3f}, gate={gate_val:.4f}")
            if acc > best_acc:
                best_acc = acc
                torch.save({'model': model.state_dict(), 'back': back.state_dict()},
                           os.path.join(snap_dir, f"v6_{args.config}_{acc:.2f}.pth"))
                print(f"  ✓ Best: {best_acc:.2f}%")
    
    # Final evaluation
    results = {}
    print(f"\nFINAL EVALUATION — V6 {args.config.upper()} on {args.dataset.upper()}")
    for ber in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        model.eval(); back.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                Fp, _ = model(front(imgs), noise_param=ber)
                correct += back(Fp).argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        acc = 100. * correct / total
        label = "Clean" if ber == 0 else f"BER={ber:.2f}"
        print(f"  {label}: {acc:.2f}%")
        results[str(ber)] = acc
    
    with open(os.path.join(snap_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Gate value: {torch.sigmoid(model.decoder.shortcut_gate).item():.4f}")
    return results


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    print(f"Device: {device}")
    train_v6(args)
