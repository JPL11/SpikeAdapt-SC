"""Restyle Figs 2 and 4 with REAL AID data — publication quality.

Uses actual model predictions and masks from the trained checkpoint.
Styled for IEEE conference paper with serif fonts, clean layout.
"""

import os, sys, random, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

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
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
    'axes.linewidth': 0.6,
    'lines.linewidth': 1.0,
    'lines.markersize': 3.5,
    'grid.linewidth': 0.4,
    'grid.alpha': 0.3,
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SNAP_DIR = "./snapshots_aid/"
OUT_DIR = "./paper/figures/"
os.makedirs(OUT_DIR, exist_ok=True)

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
        return grad_output * sig * (1-sig) * scale, -(grad_output * sig * (1-sig) * scale).sum() if ctx.th_needs_grad else None

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

class LearnedImportanceScorer(nn.Module):
    def __init__(self, C_in=128, h=32):
        super().__init__()
        self.scorer = nn.Sequential(nn.Conv2d(C_in,h,1), nn.ReLU(True), nn.Conv2d(h,1,1), nn.Sigmoid())
    def forward(self, S): return self.scorer(torch.stack(S,0).mean(0)).squeeze(1)

class LearnedBlockMask(nn.Module):
    def __init__(self, rate=0.75):
        super().__init__(); self.target_rate = rate
    def forward(self, imp):
        B,H,W = imp.shape; k = max(1,int(self.target_rate*H*W))
        flat = imp.view(B,-1); _,idx = flat.topk(k,1)
        m = torch.zeros_like(flat); m.scatter_(1,idx,1.0)
        return m.view(B,H,W).unsqueeze(1), m.view(B,H,W).mean()

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
    def __init__(self, root, transform=None):
        self.transform=transform; aid=os.path.join(root,'AID')
        classes=sorted([d for d in os.listdir(aid) if os.path.isdir(os.path.join(aid,d))])
        self.c2i={c:i for i,c in enumerate(classes)}; self.i2c={i:c for c,i in self.c2i.items()}
        all_s=[]
        for c in classes:
            cd=os.path.join(aid,c)
            for f in sorted(os.listdir(cd)):
                if f.lower().endswith(('.jpg','.jpeg','.png','.tif')): all_s.append((os.path.join(cd,f),self.c2i[c]))
        random.seed(42); random.shuffle(all_s)
        self.samples=all_s[int(len(all_s)*0.8):]
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        p,l=self.samples[i]; img=Image.open(p).convert('RGB')
        if self.transform: img=self.transform(img)
        return img,l,p

def denorm(t):
    t=t.cpu().clone()
    for c,(m,s) in enumerate(zip([.485,.456,.406],[.229,.224,.225])): t[c]=t[c]*s+m
    return t.clamp(0,1).permute(1,2,0).numpy()

def load_all():
    front=ResNet50Front().to(device); back=ResNet50Back(30).to(device)
    bb=torch.load(os.path.join(SNAP_DIR,"backbone_best.pth"),map_location=device)
    front.load_state_dict({k:v for k,v in bb.items() if not k.startswith(('layer4.','fc.','avgpool.'))},strict=False)
    back.load_state_dict({k:v for k,v in bb.items() if k.startswith(('layer4.','fc.','avgpool.'))},strict=False)
    front.eval()
    enc=Encoder(1024,256,128).to(device); imp=LearnedImportanceScorer(128,32).to(device)
    bm=LearnedBlockMask(0.75); dec=Decoder(1024,256,128,8).to(device); ch=BSC_Channel().to(device)
    s3=sorted([f for f in os.listdir(SNAP_DIR) if f.startswith("bsc_s3_")])
    if s3:
        ck=torch.load(os.path.join(SNAP_DIR,s3[-1]),map_location=device)
        enc.load_state_dict({k.replace('encoder.',''):v for k,v in ck['model'].items() if 'encoder.' in k})
        imp.load_state_dict({k.replace('importance_scorer.',''):v for k,v in ck['model'].items() if 'importance_scorer.' in k})
        dec.load_state_dict({k.replace('decoder.',''):v for k,v in ck['model'].items() if 'decoder.' in k})
        back.load_state_dict(ck['back']); print(f"Loaded: {s3[-1]}")
    enc.eval();imp.eval();dec.eval();back.eval()
    return front,back,enc,imp,bm,ch,dec

def run(front,enc,imp_s,bm,ch,dec,back,img,ber=0.0,rate=None,T=8):
    with torch.no_grad():
        feat=front(img); S2,m1,m2=[],None,None
        for t in range(T): _,s2,m1,m2=enc(feat,m1,m2); S2.append(s2)
        importance=imp_s(S2)
        if rate: old=bm.target_rate; bm.target_rate=rate
        mask,tx=bm(importance)
        if rate: bm.target_rate=old
        recv=[ch(S2[t]*mask,ber) for t in range(T)]
        Fp=dec(recv,mask); logits=back(Fp); probs=F.softmax(logits,1); conf,pred=probs.max(1)
    return {'importance':importance,'mask':mask,'pred':pred,'conf':conf}


if __name__=="__main__":
    print(f"Device: {device}")
    front,back,enc,imp_s,bm,ch,dec = load_all()
    tf=T.Compose([T.Resize(256),T.CenterCrop(224),T.ToTensor(),T.Normalize((.485,.456,.406),(.229,.224,.225))])
    ds=AIDDataset("./data",transform=tf)

    # Collect samples per class
    cs={}
    for i in range(len(ds)):
        img,lab,p=ds[i]; c=ds.i2c[lab]
        if c not in cs: cs[c]=(img,lab,p)
        if len(cs)>=30: break

    # =================================================================
    # FIG 2: Mask Diversity — 8 classes, publication quality
    # =================================================================
    targets=['Airport','Beach','Bridge','Desert','Farmland','Harbor','Mountain','Stadium']
    fig = plt.figure(figsize=(7.16, 2.3))  # IEEE double-column width
    gs = gridspec.GridSpec(2, 8, hspace=0.08, wspace=0.06,
                           left=0.06, right=0.99, top=0.88, bottom=0.02)

    for idx, cls in enumerate(targets):
        if cls not in cs: continue
        img_t, lab, _ = cs[cls]
        r = run(front,enc,imp_s,bm,ch,dec,back, img_t.unsqueeze(0).to(device))
        mask = r['mask'][0,0].cpu().numpy()
        imp_map = r['importance'][0].cpu().numpy()

        # Top: image with mask overlay
        ax = fig.add_subplot(gs[0, idx])
        img_np = denorm(img_t)
        mask_up = np.kron(mask, np.ones((28,28)))[:224,:224]
        overlay = img_np.copy()
        # Red tint for masked blocks, but semi-transparent
        alpha_mask = 0.65
        overlay[mask_up==0, 0] = img_np[mask_up==0, 0] * (1-alpha_mask) + 0.85 * alpha_mask
        overlay[mask_up==0, 1] = img_np[mask_up==0, 1] * (1-alpha_mask) + 0.12 * alpha_mask
        overlay[mask_up==0, 2] = img_np[mask_up==0, 2] * (1-alpha_mask) + 0.12 * alpha_mask
        # Add thin grid lines
        ax.imshow(overlay)
        for g in range(1, 8):
            ax.axhline(g*28, color='white', linewidth=0.3, alpha=0.5)
            ax.axvline(g*28, color='white', linewidth=0.3, alpha=0.5)
        ax.set_title(cls, fontsize=7, fontweight='bold', pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_linewidth(0.4)

        # Bottom: mask as clean grid
        ax2 = fig.add_subplot(gs[1, idx])
        # Custom colormap: red=0, green=1
        cmap_mask = plt.cm.colors.ListedColormap(['#d63031', '#27ae60'])
        ax2.imshow(mask, cmap=cmap_mask, vmin=0, vmax=1, interpolation='nearest')
        # Grid lines
        for g in range(1, 8):
            ax2.axhline(g-0.5, color='white', linewidth=0.5)
            ax2.axvline(g-0.5, color='white', linewidth=0.5)
        n_sent = int(mask.sum())
        ax2.set_title(f'{n_sent}/64', fontsize=6.5, pad=1)
        ax2.set_xticks([]); ax2.set_yticks([])
        for spine in ax2.spines.values(): spine.set_linewidth(0.4)

    # Row labels
    fig.text(0.025, 0.68, 'Image +\nmask', ha='center', va='center', fontsize=7,
             fontweight='bold', rotation=90, fontstyle='italic')
    fig.text(0.025, 0.23, '8×8\nmask', ha='center', va='center', fontsize=7,
             fontweight='bold', rotation=90, fontstyle='italic')

    plt.savefig(os.path.join(OUT_DIR, 'fig2_mask_diversity.png'), facecolor='white')
    print("✓ Fig 2: Mask diversity (restyled with real data)")
    plt.close()

    # =================================================================
    # FIG 4: Rate-Accuracy Tradeoff — 5 rates, single row
    # =================================================================
    rates = [1.0, 0.90, 0.75, 0.50, 0.25]
    cls = 'Airport'
    img_t, lab, _ = cs[cls]
    img_batch = img_t.unsqueeze(0).to(device)

    fig = plt.figure(figsize=(7.16, 1.9))
    gs = gridspec.GridSpec(1, 5, wspace=0.08, left=0.01, right=0.99, top=0.82, bottom=0.01)

    for col, rate in enumerate(rates):
        r = run(front,enc,imp_s,bm,ch,dec,back, img_batch, rate=rate)
        mask = r['mask'][0,0].cpu().numpy()
        pred_name = ds.i2c[r['pred'].item()]
        conf_val = r['conf'].item() * 100
        correct = pred_name == cls

        ax = fig.add_subplot(gs[0, col])
        img_np = denorm(img_t)
        mask_up = np.kron(mask, np.ones((28,28)))[:224,:224]
        overlay = img_np.copy()

        # Progressive red intensity for masked blocks
        alpha = 0.7
        overlay[mask_up==0, 0] = img_np[mask_up==0, 0]*(1-alpha) + 0.9*alpha
        overlay[mask_up==0, 1] = img_np[mask_up==0, 1]*(1-alpha) + 0.15*alpha
        overlay[mask_up==0, 2] = img_np[mask_up==0, 2]*(1-alpha) + 0.15*alpha

        ax.imshow(overlay)
        # Grid
        for g in range(1, 8):
            ax.axhline(g*28, color='white', linewidth=0.3, alpha=0.4)
            ax.axvline(g*28, color='white', linewidth=0.3, alpha=0.4)

        n_sent = int(mask.sum())
        color = '#27ae60' if correct else '#c0392b'
        symbol = '✓' if correct else '✗'

        ax.set_title(f'$\\rho$={rate:.0%}  ({n_sent}/64 blocks)\n'
                     f'{symbol} {pred_name} ({conf_val:.0f}%)',
                     fontsize=7, color=color, fontweight='bold', pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
            if not correct: spine.set_color('#c0392b'); spine.set_linewidth(1.5)

    plt.savefig(os.path.join(OUT_DIR, 'fig4_rate_sweep.png'), facecolor='white')
    print("✓ Fig 4: Rate sweep (restyled with real data)")
    plt.close()

    print(f"\n✅ Figs 2 & 4 restyled. Saved to {OUT_DIR}")
