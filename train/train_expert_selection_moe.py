#!/usr/bin/env python3
"""Expert Selection MoE for adaptive rho — journal-extension experiment.

Architecture:
    Pre-trained experts (FROZEN): one full (encoder, scorer, decoder, back)
    pipeline per target rho, from per-rho scorer training.

    Trainable: A tiny gate MLP that maps BER → expert index.

    At inference: gate picks the expert, that expert's pipeline runs end-to-end.

Key design choice: the gate is conditioned ONLY on BER (scalar). Since the
optimal rho*(BER) mapping is deterministic, we can:
  - Train the gate with supervised cross-entropy against oracle expert indices
    derived from per_rho_scorer_results.json (no expensive forward passes)
  - Validate by evaluating the full expert pipeline at inference

This avoids the collapse failure mode of the earlier Opt2/Opt3 attempts:
the experts are frozen at their per-rho-optimal states, so the gate can't
destroy them by joint fine-tuning.

Usage:
    # Train gate only (fast, ~5 min)
    python train/train_expert_selection_moe.py --dataset aid
    python train/train_expert_selection_moe.py --dataset resisc45

    # Evaluate
    python train/train_expert_selection_moe.py --dataset both --eval-only
"""

import os, sys, json, glob, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

from train_aid_v2 import ResNet50Front, ResNet50Back
from run_final_pipeline import AIDDataset5050, RESISC45Dataset, SpikeAdaptSC_v5c_NA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_STEPS = 8
BER_SWEEP = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

# Experts correspond to the pre-trained per-rho scorers we trust most
EXPERT_RHOS = [0.500, 0.625, 0.750, 0.875]

DATASET_CONFIGS = {
    'aid': {
        'n_classes': 30,
        'ds_cls': AIDDataset5050,
        'ds_kwargs': dict(seed=42),
        'bb_path': './snapshots_aid_5050_seed42/backbone_best.pth',
        'per_rho_dir': './snapshots_aid_per_rho',
        'snap': './snapshots_aid_expert_moe',
    },
    'resisc45': {
        'n_classes': 45,
        'ds_cls': RESISC45Dataset,
        'ds_kwargs': dict(train_ratio=0.20, seed=42),
        'bb_path': './snapshots_resisc45_5050_seed42/backbone_best.pth',
        'per_rho_dir': './snapshots_resisc45_per_rho',
        'snap': './snapshots_resisc45_expert_moe',
    },
}


# ======================================================================
# Gate MLP (the only trainable component)
# ======================================================================

class ExpertGate(nn.Module):
    """Tiny MLP: BER (scalar) → softmax over experts.

    Since BER is 1-D, this is a very small network. Params: ~500.
    """
    def __init__(self, n_experts=4, hidden=64):
        super().__init__()
        self.n_experts = n_experts
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_experts),
        )

    def forward(self, ber, tau=1.0):
        """
        Args:
            ber: float or tensor
            tau: temperature for softmax
        Returns:
            weights: [n_experts] soft weights
        """
        if isinstance(ber, (int, float)):
            ber_t = torch.tensor([[ber]], dtype=torch.float32,
                                 device=next(self.parameters()).device)
        else:
            ber_t = ber.view(-1, 1)
        logits = self.mlp(ber_t) / tau
        return F.softmax(logits, dim=-1)

    def get_expert_idx(self, ber):
        """Hard argmax for inference."""
        with torch.no_grad():
            w = self.forward(ber).squeeze()  # [n_experts]
            return int(w.argmax().item())


# ======================================================================
# Load frozen experts
# ======================================================================

def load_expert(rho, per_rho_dir, n_classes):
    """Load a per-rho expert (full pipeline: encoder, scorer, decoder, back)."""
    rho_tag = f'rho{rho:.3f}'

    # Prefer s5 (joint fine-tuned) checkpoints over s4
    cks = sorted(glob.glob(os.path.join(per_rho_dir, f's5_{rho_tag}_best_*.pth')),
                 key=lambda x: float(x.split('_')[-1].replace('.pth', '')))
    if not cks:
        cks = sorted(glob.glob(os.path.join(per_rho_dir, f's4_{rho_tag}_best_*.pth')),
                     key=lambda x: float(x.split('_')[-1].replace('.pth', '')))

    if not cks:
        raise FileNotFoundError(f"No checkpoint for rho={rho} in {per_rho_dir}")

    ck_path = cks[-1]  # Best (highest accuracy)
    ck = torch.load(ck_path, map_location=device, weights_only=False)

    model = SpikeAdaptSC_v5c_NA(C_in=1024, C1=256, C2=36, T=T_STEPS,
                                 target_rate=rho, grid_size=14).to(device)
    back = ResNet50Back(n_classes).to(device)
    model.load_state_dict(ck['model'], strict=False)
    back.load_state_dict(ck['back'])
    model.eval(); back.eval()

    # Freeze everything
    for p in model.parameters(): p.requires_grad = False
    for p in back.parameters(): p.requires_grad = False

    return model, back, ck_path


# ======================================================================
# Oracle from per-rho scorer results
# ======================================================================

def compute_oracle_expert(ds_name, expert_rhos=EXPERT_RHOS):
    """For each BER, find which expert (index into expert_rhos) is optimal.

    Returns: dict {ber: expert_idx}
    """
    with open('eval/seed_results/per_rho_scorer_results.json') as f:
        per_rho = json.load(f)

    oracle = {}
    d = per_rho[ds_name]
    for ber in BER_SWEEP:
        best_idx, best_acc = 0, 0
        for idx, rho in enumerate(expert_rhos):
            acc = d.get(str(rho), {}).get(str(ber), 0)
            if acc > best_acc:
                best_acc = acc
                best_idx = idx
        oracle[ber] = best_idx
    return oracle


# ======================================================================
# Training (gate only, fast)
# ======================================================================

def train_gate(ds_name, cfg, epochs=300):
    """Train the gate MLP via supervised oracle matching.

    Since the gate depends only on BER (scalar), we don't need forward
    passes through the experts. We train directly on oracle labels.
    """
    print(f'\n{"="*60}')
    print(f'  Training Expert Gate — {ds_name.upper()}')
    print(f'{"="*60}')

    os.makedirs(cfg['snap'], exist_ok=True)

    oracle = compute_oracle_expert(ds_name)
    print(f'  Oracle mapping (BER → expert_idx):')
    for ber, idx in oracle.items():
        label = 'Clean' if ber == 0 else f'BER={ber:.2f}'
        print(f'    {label:<10}: expert {idx} (rho={EXPERT_RHOS[idx]:.3f})')

    # Build supervised dataset: (BER, expert_idx) pairs
    bers = torch.tensor([[b] for b in oracle.keys()], dtype=torch.float32, device=device)
    labels = torch.tensor(list(oracle.values()), dtype=torch.long, device=device)

    gate = ExpertGate(n_experts=len(EXPERT_RHOS), hidden=64).to(device)
    opt = optim.Adam(gate.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Augmentation: sample random BERs near oracle points for smoother interpolation
    best_acc = 0
    for ep in range(1, epochs + 1):
        gate.train()

        # Core oracle loss
        logits = gate.mlp(bers)
        loss_oracle = criterion(logits, labels)

        # Smooth interpolation: sample random BERs and regularize
        n_rand = 32
        ber_rand = torch.rand(n_rand, 1, device=device) * 0.5  # [0, 0.5]
        logits_rand = gate.mlp(ber_rand)
        # Encourage soft predictions (entropy regularization, not too peaky)
        probs_rand = F.softmax(logits_rand, dim=-1)
        entropy_reg = -(probs_rand * torch.log(probs_rand + 1e-8)).sum(-1).mean()

        loss = loss_oracle - 0.05 * entropy_reg  # encourage some uncertainty

        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep % 50 == 0 or ep == epochs:
            gate.eval()
            with torch.no_grad():
                pred = gate.mlp(bers).argmax(-1)
                acc = (pred == labels).float().mean().item() * 100
            print(f'    Ep{ep:03d}: oracle_match={acc:.1f}%, loss={loss.item():.4f}')

    # Print final gate decisions
    print(f'\n  Learned gate decisions:')
    gate.eval()
    for ber in BER_SWEEP:
        w = gate(ber).detach().cpu().numpy().squeeze()  # [n_experts]
        idx = int(np.argmax(w))
        label = 'Clean' if ber == 0 else f'BER={ber:.2f}'
        weights_str = ', '.join(f'{float(w[i]):.2f}' for i in range(len(EXPERT_RHOS)))
        oracle_idx = oracle[ber]
        match = 'match' if idx == oracle_idx else f'(oracle: {oracle_idx})'
        print(f'    {label:<10}: rho={EXPERT_RHOS[idx]:.3f} [{weights_str}] {match}')

    torch.save({'gate': gate.state_dict(),
                'expert_rhos': EXPERT_RHOS,
                'oracle': oracle},
               os.path.join(cfg['snap'], 'gate.pth'))
    print(f'\n  Saved gate to {cfg["snap"]}/gate.pth')
    return gate


# ======================================================================
# Evaluation: full expert pipeline with gate routing
# ======================================================================

def evaluate_expert_moe(ds_name, cfg):
    """Evaluate by routing each batch through gate-selected expert."""
    print(f'\n{"="*60}')
    print(f'  Evaluating Expert MoE — {ds_name.upper()}')
    print(f'{"="*60}')

    # Load gate
    gate_path = os.path.join(cfg['snap'], 'gate.pth')
    if not os.path.exists(gate_path):
        raise FileNotFoundError(f"No gate at {gate_path}. Run training first.")
    gate_ck = torch.load(gate_path, map_location=device, weights_only=False)
    gate = ExpertGate(n_experts=len(EXPERT_RHOS), hidden=64).to(device)
    gate.load_state_dict(gate_ck['gate'])
    gate.eval()

    # Load all experts
    experts = []
    print(f'  Loading {len(EXPERT_RHOS)} experts...')
    for rho in EXPERT_RHOS:
        model, back, path = load_expert(rho, cfg['per_rho_dir'], cfg['n_classes'])
        experts.append((rho, model, back))
        print(f'    rho={rho:.3f}: {os.path.basename(path)}')

    # Load backbone and test data
    tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((.485,.456,.406),(.229,.224,.225))])
    test_ds = cfg['ds_cls']("./data", tf_test, 'test', **cfg['ds_kwargs'])
    test_loader = DataLoader(test_ds, 32, False, num_workers=4, pin_memory=True)

    front = ResNet50Front(grid_size=14).to(device)
    bb = torch.load(cfg['bb_path'], map_location=device, weights_only=False)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.', 'spatial_pool.'))},
                          strict=False)
    front.eval()
    for p in front.parameters(): p.requires_grad = False

    # Evaluate across BER sweep
    results = {}
    print(f'\n  BER sweep:')
    print(f'  {"BER":<8} {"Expert":<12} {"rho":>8} {"Acc":>8}')
    print(f'  {"-"*40}')

    for ber in BER_SWEEP:
        expert_idx = gate.get_expert_idx(ber)
        rho, model, back = experts[expert_idx]

        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                Fp, _ = model(front(imgs), noise_param=ber)
                correct += back(Fp).argmax(1).eq(labels).sum().item()
                total += labels.size(0)

        acc = 100. * correct / total
        results[str(ber)] = {
            'acc': round(acc, 2),
            'expert_idx': int(expert_idx),
            'rho': float(rho),
        }

        label = 'Clean' if ber == 0 else f'{ber:.2f}'
        print(f'  {label:<8} expert{expert_idx:<6} {rho:>8.3f} {acc:>7.2f}%')

    return results


# ======================================================================
# Comparison
# ======================================================================

def compare_with_baselines(ds_name, moe_results):
    """Compare MoE results with fixed rho=0.75, rho=0.625, and oracle per-rho."""
    with open('eval/seed_results/per_rho_scorer_results.json') as f:
        per_rho = json.load(f)

    d = per_rho[ds_name]

    print(f'\n  COMPARISON: MoE vs Fixed rhos vs Oracle')
    print(f'  {"BER":<8} {"MoE":>10} {"Fix0.75":>10} {"Fix0.625":>10} {"Oracle":>10}')
    print(f'  {"-"*48}')

    moe_vals, f075, f0625, oracle_vals = [], [], [], []
    for ber in BER_SWEEP:
        moe = moe_results[str(ber)]['acc']
        f075_v = d.get('0.75', {}).get(str(ber), 0)
        f0625_v = d.get('0.625', {}).get(str(ber), 0)
        oracle_v = max(d[str(r)].get(str(ber), 0) for r in [0.1, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])
        label = 'Clean' if ber == 0 else f'{ber:.2f}'
        print(f'  {label:<8} {moe:>10.2f} {f075_v:>10.2f} {f0625_v:>10.2f} {oracle_v:>10.2f}')
        moe_vals.append(moe); f075.append(f075_v); f0625.append(f0625_v); oracle_vals.append(oracle_v)

    # Averages over [0, 0.30]
    idx_op = [i for i, b in enumerate(BER_SWEEP) if b <= 0.30]
    print(f'\n  Average over BER [0, 0.30]:')
    print(f'    MoE:              {np.mean([moe_vals[i] for i in idx_op]):.2f}%')
    print(f'    Fixed rho=0.75:   {np.mean([f075[i] for i in idx_op]):.2f}%')
    print(f'    Fixed rho=0.625:  {np.mean([f0625[i] for i in idx_op]):.2f}%')
    print(f'    Oracle per-rho:   {np.mean([oracle_vals[i] for i in idx_op]):.2f}%')

    return {'moe_avg': np.mean([moe_vals[i] for i in idx_op]),
            'f075_avg': np.mean([f075[i] for i in idx_op]),
            'f0625_avg': np.mean([f0625[i] for i in idx_op]),
            'oracle_avg': np.mean([oracle_vals[i] for i in idx_op])}


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='both', choices=['aid', 'resisc45', 'both'])
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    ds_list = ['aid', 'resisc45'] if args.dataset == 'both' else [args.dataset]
    all_results = {}

    for ds_name in ds_list:
        cfg = DATASET_CONFIGS[ds_name]

        if not args.eval_only:
            train_gate(ds_name, cfg, epochs=args.epochs)

        results = evaluate_expert_moe(ds_name, cfg)
        summary = compare_with_baselines(ds_name, results)

        all_results[ds_name] = {
            'results': results,
            'summary': summary,
        }

    # Save
    os.makedirs('eval/seed_results', exist_ok=True)
    out = 'eval/seed_results/expert_moe_results.json'
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f'\nSaved results to {out}')


if __name__ == '__main__':
    main()
