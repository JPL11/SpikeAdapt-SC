"""AID Ablation Study — run all variants for the GlobeCom paper.

Evaluates:
1. Full model (ρ=0.75, T=8, learned mask, CE+rate loss) — baseline
2. Random masking (ρ=0.75) — proves learned masking matters
3. No masking (ρ=1.0) — proves masking doesn't hurt
4. T=4 timesteps — proves 8 timesteps is needed
5. Rate sweep (ρ=0.25..1.0) — graceful degradation
6. BER sweep (0..0.35) — channel robustness curve
7. Per-channel comparison (BSC vs AWGN vs Rayleigh)
"""

import os, sys, random, math, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SNAP_DIR = "./snapshots_aid/"

# ========================= MODEL CLASSES =========================

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
        scale = 10.0; sig = torch.sigmoid(scale * (membrane - threshold))
        sg = sig * (1-sig) * scale
        return grad_output * sg, -(grad_output * sg).sum() if ctx.th_needs_grad else None

class IFNeuron(nn.Module):
    def __init__(self, th=1.0):
        super().__init__(); self.threshold = th
    def forward(self, x, mem=None):
        if mem is None: mem = torch.zeros_like(x)
        mem = mem + x; sp = SpikeFunction.apply(mem, self.threshold)
        return sp, mem - sp * self.threshold

class IHFNeuron(nn.Module):
    def __init__(self, th=1.0):
        super().__init__(); self.threshold = nn.Parameter(torch.tensor(float(th)))
    def forward(self, x, mem=None):
        if mem is None: mem = torch.zeros_like(x)
        mem = mem + x; sp = SpikeFunction.apply(mem, self.threshold)
        return sp, mem - sp * self.threshold

class BSC_Channel(nn.Module):
    def forward(self, x, ber):
        if ber <= 0: return x
        return ((x + (torch.rand_like(x.float()) < ber).float()) % 2)

class AWGN_Channel(nn.Module):
    def forward(self, x, snr_db):
        if snr_db >= 100: return x
        bpsk = 2.0*x - 1.0; snr = 10**(snr_db/10.0)
        return (bpsk + torch.randn_like(bpsk) / math.sqrt(2*snr) > 0).float()

class Rayleigh_Channel(nn.Module):
    def forward(self, x, snr_db):
        if snr_db >= 100: return x
        bpsk = 2.0*x - 1.0; snr = 10**(snr_db/10.0)
        h_r = torch.randn_like(bpsk)/math.sqrt(2); h_i = torch.randn_like(bpsk)/math.sqrt(2)
        h = torch.sqrt(h_r**2 + h_i**2)
        return ((h*bpsk + torch.randn_like(bpsk)/math.sqrt(2*snr)) / (h+1e-8) > 0).float()

class LearnedImportanceScorer(nn.Module):
    def __init__(self, C_in=128, h=32):
        super().__init__()
        self.scorer = nn.Sequential(nn.Conv2d(C_in,h,1), nn.ReLU(True), nn.Conv2d(h,1,1), nn.Sigmoid())
    def forward(self, S): return self.scorer(torch.stack(S,0).mean(0)).squeeze(1)

class Encoder(nn.Module):
    def __init__(self, Ci=1024, C1=256, C2=128):
        super().__init__()
        self.conv1=nn.Conv2d(Ci,C1,3,1,1); self.bn1=nn.BatchNorm2d(C1); self.if1=IFNeuron()
        self.conv2=nn.Conv2d(C1,C2,3,1,1); self.bn2=nn.BatchNorm2d(C2); self.if2=IFNeuron()
    def forward(self,F,m1=None,m2=None):
        s1,m1=self.if1(self.bn1(self.conv1(F)),m1); s2,m2=self.if2(self.bn2(self.conv2(s1)),m2)
        return s1,s2,m1,m2

class Decoder(nn.Module):
    def __init__(self, Co=1024, C1=256, C2=128, T=8):
        super().__init__(); self.T=T
        self.conv3=nn.Conv2d(C2,C1,3,1,1); self.bn3=nn.BatchNorm2d(C1); self.if3=IFNeuron()
        self.conv4=nn.Conv2d(C1,Co,3,1,1); self.bn4=nn.BatchNorm2d(Co); self.ihf=IHFNeuron()
        self.converter_fc=nn.Linear(2*T,2*T)
    def forward(self,recv,mask):
        m3,m4=None,None; Fs,Fm=[],[]
        for t in range(self.T):
            s3,m3=self.if3(self.bn3(self.conv3(recv[t]*mask)),m3)
            sp,m4=self.ihf(self.bn4(self.conv4(s3)),m4)
            Fs.append(sp); Fm.append(m4.clone())
        il=[]
        for t in range(self.T): il.append(Fs[t]); il.append(Fm[t])
        x=torch.stack(il,1).permute(0,2,3,4,1)
        return (x*torch.sigmoid(self.converter_fc(x))).sum(-1)

class SpikeAdaptSC(nn.Module):
    def __init__(self, C_in=1024, C1=256, C2=128, T=8, target_rate=0.75, channel_type='bsc'):
        super().__init__(); self.T=T
        self.encoder=Encoder(C_in,C1,C2)
        self.importance_scorer=LearnedImportanceScorer(C2,32)
        self.target_rate=target_rate
        self.decoder=Decoder(C_in,C1,C2,T)
        if channel_type=='awgn': self.channel=AWGN_Channel()
        elif channel_type=='rayleigh': self.channel=Rayleigh_Channel()
        else: self.channel=BSC_Channel()
        self.channel_type=channel_type

    def forward(self, feat, noise_param=0.0, rate_override=None, random_mask=False):
        T = self.T
        all_S2, m1, m2 = [], None, None
        for t in range(T):
            _,s2,m1,m2 = self.encoder(feat, m1, m2)
            all_S2.append(s2)

        importance = self.importance_scorer(all_S2)
        rate = rate_override if rate_override is not None else self.target_rate
        B,H,W = importance.shape; k = max(1,int(rate*H*W))

        if random_mask:
            # Random mask baseline
            flat = torch.rand(B, H*W, device=importance.device)
            _,idx = flat.topk(k,1)
            mask = torch.zeros_like(flat); mask.scatter_(1,idx,1.0)
            mask = mask.view(B,H,W).unsqueeze(1)
        else:
            flat = importance.view(B,-1)
            _,idx = flat.topk(k,1)
            mask = torch.zeros_like(flat); mask.scatter_(1,idx,1.0)
            mask = mask.view(B,H,W).unsqueeze(1)

        recv = [self.channel(all_S2[t]*mask, noise_param) for t in range(T)]
        Fp = self.decoder(recv, mask)
        return Fp, mask.mean().item()

class ResNet50Front(nn.Module):
    def __init__(self):
        super().__init__()
        r=torchvision.models.resnet50(weights='IMAGENET1K_V1')
        self.conv1=r.conv1;self.bn1=r.bn1;self.relu=r.relu;self.maxpool=r.maxpool
        self.layer1=r.layer1;self.layer2=r.layer2;self.layer3=r.layer3
        self.pool=nn.AdaptiveAvgPool2d(8)
    def forward(self,x):
        return self.pool(self.layer3(self.layer2(self.layer1(self.maxpool(self.relu(self.bn1(self.conv1(x))))))))

class ResNet50Back(nn.Module):
    def __init__(self, nc=30):
        super().__init__()
        r=torchvision.models.resnet50(weights='IMAGENET1K_V1')
        self.layer4=r.layer4; self.avgpool=nn.AdaptiveAvgPool2d(1); self.fc=nn.Linear(2048,nc)
    def forward(self,x): return self.fc(torch.flatten(self.avgpool(self.layer4(x)),1))

class AIDDataset(Dataset):
    def __init__(self, root, transform=None, split='test'):
        self.transform=transform; aid=os.path.join(root,'AID')
        classes=sorted([d for d in os.listdir(aid) if os.path.isdir(os.path.join(aid,d))])
        self.c2i={c:i for i,c in enumerate(classes)}; self.i2c={i:c for c,i in self.c2i.items()}
        all_s=[]
        for c in classes:
            cd=os.path.join(aid,c)
            for f in sorted(os.listdir(cd)):
                if f.lower().endswith(('.jpg','.jpeg','.png','.tif')): all_s.append((os.path.join(cd,f),self.c2i[c]))
        random.seed(42); random.shuffle(all_s)
        sp=int(len(all_s)*0.8)
        self.samples = all_s[sp:] if split=='test' else all_s[:sp]
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        p,l=self.samples[i]; img=Image.open(p).convert('RGB')
        if self.transform: img=self.transform(img)
        return img,l


def evaluate(front, model, back, loader, noise_param=0.0, rate_override=None, random_mask=False):
    """Evaluate accuracy over the full test set."""
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feat = front(imgs)
            Fp, tx = model(feat, noise_param, rate_override, random_mask)
            pred = back(Fp).argmax(1)
            correct += (pred==labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def load_model(channel_type='bsc'):
    front=ResNet50Front().to(device); back=ResNet50Back(30).to(device)
    bb=torch.load(os.path.join(SNAP_DIR,"backbone_best.pth"),map_location=device)
    front.load_state_dict({k:v for k,v in bb.items() if not k.startswith(('layer4.','fc.','avgpool.'))},strict=False)
    back.load_state_dict({k:v for k,v in bb.items() if k.startswith(('layer4.','fc.','avgpool.'))},strict=False)
    front.eval()

    model = SpikeAdaptSC(1024,256,128,8,0.75,channel_type).to(device)
    prefix = f"{channel_type}_s3_"
    s3 = sorted([f for f in os.listdir(SNAP_DIR) if f.startswith(prefix)])
    if not s3:
        prefix = f"{channel_type}_s2_"
        s3 = sorted([f for f in os.listdir(SNAP_DIR) if f.startswith(prefix)])
    if s3:
        ck = torch.load(os.path.join(SNAP_DIR,s3[-1]),map_location=device)
        model.load_state_dict(ck['model']); back.load_state_dict(ck['back'])
        print(f"  Loaded: {s3[-1]}")
    model.eval(); back.eval()
    return front, model, back


if __name__ == "__main__":
    print(f"Device: {device}")
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize((.485,.456,.406),(.229,.224,.225))])
    test_ds = AIDDataset("./data", transform=tf, split='test')
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)
    print(f"Test set: {len(test_ds)} images")

    results = {}

    # =============================================
    # 1. ABLATION: Masking strategy comparison
    # =============================================
    print("\n=== ABLATION: Masking Strategy ===")
    front, model, back = load_model('bsc')

    # Full model (learned mask, ρ=0.75)
    acc = evaluate(front, model, back, test_loader, 0.0, rate_override=0.75)
    print(f"  Learned mask ρ=0.75: {acc:.2f}%")
    results['learned_075'] = acc

    # No masking (ρ=1.0)
    acc = evaluate(front, model, back, test_loader, 0.0, rate_override=1.0)
    print(f"  No mask ρ=1.0:       {acc:.2f}%")
    results['no_mask'] = acc

    # Random masking (ρ=0.75)
    acc = evaluate(front, model, back, test_loader, 0.0, rate_override=0.75, random_mask=True)
    print(f"  Random mask ρ=0.75:  {acc:.2f}%")
    results['random_075'] = acc

    # =============================================
    # 2. ABLATION: Rate sweep (clean channel)
    # =============================================
    print("\n=== ABLATION: Rate Sweep (clean BSC) ===")
    rates = [1.0, 0.9, 0.75, 0.5, 0.25]
    results['rate_sweep'] = {}
    for r in rates:
        acc = evaluate(front, model, back, test_loader, 0.0, rate_override=r)
        print(f"  ρ={r:.2f}: {acc:.2f}%")
        results['rate_sweep'][str(r)] = acc

    # =============================================
    # 3. ABLATION: BER sweep (fine-grained)
    # =============================================
    print("\n=== ABLATION: BER Sweep (ρ=0.75) ===")
    bers = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    results['ber_sweep'] = {}
    for ber in bers:
        acc = evaluate(front, model, back, test_loader, ber, rate_override=0.75)
        print(f"  BER={ber:.2f}: {acc:.2f}%")
        results['ber_sweep'][str(ber)] = acc

    # =============================================
    # 4. Cross-channel comparison
    # =============================================
    print("\n=== Cross-Channel Comparison ===")
    results['channels'] = {}

    for ch_type in ['bsc', 'awgn', 'rayleigh']:
        print(f"\n  --- {ch_type.upper()} ---")
        front_c, model_c, back_c = load_model(ch_type)

        if ch_type == 'bsc':
            params = [(0.0, 'clean'), (0.1, 'mid'), (0.3, 'severe')]
        else:
            params = [(20, 'clean'), (5, 'mid'), (-2, 'severe')]

        ch_results = {}
        for noise, label in params:
            acc = evaluate(front_c, model_c, back_c, test_loader, noise)
            print(f"    {label} ({noise}): {acc:.2f}%")
            ch_results[label] = acc

        # Also test at ρ=0.75
        acc_75 = evaluate(front_c, model_c, back_c, test_loader,
                          params[0][0], rate_override=0.75)
        print(f"    ρ=0.75: {acc_75:.2f}%")
        ch_results['rate_075'] = acc_75

        results['channels'][ch_type] = ch_results

    # =============================================
    # 5. Random mask under noise (BER=0.15)
    # =============================================
    print("\n=== Random Mask Under Noise ===")
    front, model, back = load_model('bsc')
    acc_learned_noisy = evaluate(front, model, back, test_loader, 0.15, rate_override=0.75)
    acc_random_noisy = evaluate(front, model, back, test_loader, 0.15, rate_override=0.75, random_mask=True)
    print(f"  Learned mask + BER=0.15: {acc_learned_noisy:.2f}%")
    print(f"  Random mask + BER=0.15:  {acc_random_noisy:.2f}%")
    results['noisy_learned'] = acc_learned_noisy
    results['noisy_random'] = acc_random_noisy

    # Save results
    with open('ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # =============================================
    # Print summary table
    # =============================================
    print("\n" + "="*60)
    print("ABLATION SUMMARY FOR PAPER")
    print("="*60)

    print("\nTable: Component Ablation (BSC, clean channel)")
    print(f"{'Variant':<40} {'Acc %':>8}")
    print("-"*50)
    print(f"{'SpikeAdapt-SC (full, ρ=0.75)':<40} {results['learned_075']:>8.2f}")
    print(f"{'w/o masking (ρ=1.0)':<40} {results['no_mask']:>8.2f}")
    print(f"{'Random mask (ρ=0.75)':<40} {results['random_075']:>8.2f}")

    print("\nTable: Rate Sweep")
    for r in rates:
        print(f"  ρ={r:.2f}: {results['rate_sweep'][str(r)]:.2f}%")

    print("\nTable: BER Robustness")
    for ber in bers:
        print(f"  BER={ber:.2f}: {results['ber_sweep'][str(ber)]:.2f}%")

    print("\nTable: Cross-Channel")
    for ch in ['bsc', 'awgn', 'rayleigh']:
        d = results['channels'][ch]
        print(f"  {ch.upper()}: clean={d['clean']:.2f}%, mid={d['mid']:.2f}%, severe={d['severe']:.2f}%, ρ=0.75={d['rate_075']:.2f}%")

    print(f"\nNoisy comparison:")
    print(f"  Learned + BER=0.15: {results['noisy_learned']:.2f}%")
    print(f"  Random  + BER=0.15: {results['noisy_random']:.2f}%")

    print(f"\n✅ All results saved to ablation_results.json")
