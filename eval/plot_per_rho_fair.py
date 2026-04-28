#!/usr/bin/env python3
"""Fair analysis with per-ρ trained scorers.

Generates:
  1. Accuracy vs BER at each ρ (per-ρ trained, fair)
  2. Accuracy vs ρ at each BER (fair comparison)
  3. Alpha v1 and v2 (fair, with per-ρ scorers)
  4. Rate-accuracy Pareto frontier
  5. Stochastic resonance analysis
  6. Comparison with old (unfair) results

Uses eval/seed_results/per_rho_scorer_results.json
Output: eval/figures/fair_per_rho_*.pdf
"""

import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('eval/figures', exist_ok=True)

with open('eval/seed_results/per_rho_scorer_results.json') as f:
    data = json.load(f)

datasets = [('aid', 'AID', 30), ('resisc45', 'RESISC45', 45)]
BER_SWEEP = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


# ====== Plot 1: Accuracy vs BER at each ρ ======
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
rho_colors = {'0.1': '#e63946', '0.25': '#f4a261', '0.375': '#e9c46a',
              '0.5': '#2a9d8f', '0.625': '#264653', '0.75': '#457b9d', '0.875': '#6d6875'}

for col, (ds_key, ds_name, n_cls) in enumerate(datasets):
    ax = axes[col]
    d = data[ds_key]
    for rho_str in sorted(d.keys(), key=float):
        rho = float(rho_str)
        accs = [d[rho_str].get(str(b), 0) for b in BER_SWEEP]
        ax.plot(BER_SWEEP, accs, 'o-', color=rho_colors.get(rho_str, 'gray'),
                linewidth=2, markersize=5, label=f'rho={rho} (own scorer)')
    ax.axhline(y=100/n_cls, color='black', linestyle=':', linewidth=1, alpha=0.4)
    ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.4)
    ax.set_xlabel('BER', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'{ds_name}: Accuracy vs BER (per-rho trained, FAIR)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 0.52)

plt.tight_layout()
fig.savefig('eval/figures/fair_per_rho_acc_vs_ber.pdf', dpi=300, bbox_inches='tight')
fig.savefig('eval/figures/fair_per_rho_acc_vs_ber.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: eval/figures/fair_per_rho_acc_vs_ber.pdf')


# ====== Plot 2: Accuracy vs ρ at each BER ======
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
cmap = plt.cm.RdYlBu_r

for col, (ds_key, ds_name, n_cls) in enumerate(datasets):
    ax = axes[col]
    d = data[ds_key]
    rhos = sorted([float(r) for r in d.keys()])
    ber_plot = [b for b in BER_SWEEP if b <= 0.40]
    norm = plt.Normalize(0, 0.4)

    for ber in ber_plot:
        accs = [d[str(r)].get(str(ber), 0) for r in rhos]
        color = cmap(norm(ber))
        label = 'Clean' if ber == 0 else f'BER={ber:.2f}'
        ax.plot(rhos, accs, 'o-', color=color, linewidth=2, markersize=5, label=label)

    ax.set_xlabel('Transmission rate (rho)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'{ds_name}: Accuracy vs rho (FAIR, per-rho trained)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, ncol=3, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.05, 0.92)

plt.tight_layout()
fig.savefig('eval/figures/fair_per_rho_acc_vs_rho.pdf', dpi=300, bbox_inches='tight')
fig.savefig('eval/figures/fair_per_rho_acc_vs_rho.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: eval/figures/fair_per_rho_acc_vs_rho.pdf')


# ====== Plot 3: Alpha v1 and v2 (FAIR) ======
fig, axes = plt.subplots(2, 2, figsize=(15, 11))

for col, (ds_key, ds_name, n_cls) in enumerate(datasets):
    d = data[ds_key]
    rhos = sorted([float(r) for r in d.keys()])
    ber_plot = [b for b in BER_SWEEP if b <= 0.40]
    norm = plt.Normalize(0, 0.4)

    # A(rho=max) and A(rho=min) for each BER
    rho_max = str(rhos[-1])  # 0.875
    rho_min = str(rhos[0])   # 0.1

    # Alpha v1: (A(rho) - A(rho_max)) / A(rho_max)
    ax = axes[0, col]
    for ber in ber_plot:
        a_max = d[rho_max].get(str(ber), 0)
        alphas = [(d[str(r)].get(str(ber), 0) - a_max) / a_max if a_max > 0.1 else 0 for r in rhos]
        color = cmap(norm(ber))
        label = 'Clean' if ber == 0 else f'BER={ber:.2f}'
        ax.plot(rhos, alphas, 'o-', color=color, linewidth=1.8, markersize=4, label=label)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_xlabel('rho', fontsize=11)
    ax.set_ylabel('alpha_1', fontsize=11)
    ax.set_title(f'{ds_name}: alpha_1 = (A(rho)-A(max))/A(max)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=6, ncol=3)
    ax.set_xlim(0.05, 0.92)
    ax.grid(True, alpha=0.3)

    # Alpha v2: (A(rho) - A(rho_min)) / A(rho_min)
    ax = axes[1, col]
    for ber in ber_plot:
        a_min = d[rho_min].get(str(ber), 0)
        alphas = [(d[str(r)].get(str(ber), 0) - a_min) / a_min if a_min > 0.1 else 0 for r in rhos]
        color = cmap(norm(ber))
        label = 'Clean' if ber == 0 else f'BER={ber:.2f}'
        ax.plot(rhos, alphas, 'o-', color=color, linewidth=1.8, markersize=4, label=label)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_xlabel('rho', fontsize=11)
    ax.set_ylabel('alpha_2', fontsize=11)
    ax.set_title(f'{ds_name}: alpha_2 = (A(rho)-A(min))/A(min)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=6, ncol=3)
    ax.set_xlim(0.05, 0.92)
    ax.grid(True, alpha=0.3)

fig.suptitle('Fair Alpha Metrics (per-rho trained scorers)', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig('eval/figures/fair_per_rho_alpha.pdf', dpi=300, bbox_inches='tight')
fig.savefig('eval/figures/fair_per_rho_alpha.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: eval/figures/fair_per_rho_alpha.pdf')


# ====== Plot 4: Rate-Accuracy Pareto frontier ======
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for col, (ds_key, ds_name, n_cls) in enumerate(datasets):
    ax = axes[col]
    d = data[ds_key]
    rhos = sorted([float(r) for r in d.keys()])

    for ber in [0.0, 0.10, 0.20, 0.30, 0.35]:
        accs = [d[str(r)].get(str(ber), 0) for r in rhos]
        # Bandwidth savings = (1 - rho) * 100
        bw_savings = [(1 - r) * 100 for r in rhos]
        color = cmap(plt.Normalize(0, 0.35)(ber))
        label = 'Clean' if ber == 0 else f'BER={ber:.2f}'
        ax.plot(bw_savings, accs, 'o-', color=color, linewidth=2.5, markersize=7, label=label)

        # Mark Pareto-optimal points
        pareto = []
        best_acc = 0
        for i in range(len(accs)-1, -1, -1):  # scan from low savings to high
            if accs[i] > best_acc:
                best_acc = accs[i]
                pareto.append(i)

    ax.set_xlabel('Bandwidth Savings (%)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'{ds_name}: Rate-Accuracy Pareto (per-rho trained)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 95)
    ax.invert_xaxis()  # Higher savings on right

plt.tight_layout()
fig.savefig('eval/figures/fair_per_rho_pareto.pdf', dpi=300, bbox_inches='tight')
fig.savefig('eval/figures/fair_per_rho_pareto.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: eval/figures/fair_per_rho_pareto.pdf')


# ====== Plot 5: Stochastic resonance ======
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for col, (ds_key, ds_name, n_cls) in enumerate(datasets):
    ax = axes[col]
    d = data[ds_key]
    rhos = sorted([float(r) for r in d.keys()])

    # For each rho: accuracy at BER=0.05 - BER=0.0
    deltas_005 = [d[str(r)]['0.05'] - d[str(r)]['0.0'] for r in rhos]
    deltas_010 = [d[str(r)]['0.1'] - d[str(r)]['0.0'] for r in rhos]
    deltas_015 = [d[str(r)]['0.15'] - d[str(r)]['0.0'] for r in rhos]

    ax.bar([r - 0.02 for r in rhos], deltas_005, 0.02, label='BER=0.05 - Clean', color='#2a9d8f', alpha=0.8)
    ax.bar(rhos, deltas_010, 0.02, label='BER=0.10 - Clean', color='#e9c46a', alpha=0.8)
    ax.bar([r + 0.02 for r in rhos], deltas_015, 0.02, label='BER=0.15 - Clean', color='#e76f51', alpha=0.8)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Transmission rate (rho)', fontsize=12)
    ax.set_ylabel('Accuracy gain over clean (%)', fontsize=12)
    ax.set_title(f'{ds_name}: Stochastic Resonance\n(positive = noise HELPS)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig.savefig('eval/figures/fair_per_rho_resonance.pdf', dpi=300, bbox_inches='tight')
fig.savefig('eval/figures/fair_per_rho_resonance.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: eval/figures/fair_per_rho_resonance.pdf')


# ====== Print analysis ======
for ds_key, ds_name, n_cls in datasets:
    d = data[ds_key]
    rhos = sorted([float(r) for r in d.keys()])

    print(f'\n{"="*80}')
    print(f'{ds_name} — KEY FINDINGS (per-rho trained, fair)')
    print(f'{"="*80}')

    # Optimal rho per BER
    print(f'\nOptimal rho* per BER:')
    for ber in BER_SWEEP:
        best_rho, best_acc = None, 0
        for rho in rhos:
            acc = d[str(rho)].get(str(ber), 0)
            if acc > best_acc:
                best_acc = acc
                best_rho = rho
        label = 'Clean' if ber == 0 else f'BER={ber:.2f}'
        print(f'  {label}: rho*={best_rho:.3f} acc={best_acc:.2f}%')

    # Stochastic resonance
    print(f'\nStochastic resonance (BER=0.05 vs Clean):')
    for rho in rhos:
        delta = d[str(rho)]['0.05'] - d[str(rho)]['0.0']
        print(f'  rho={rho:.3f}: {delta:+.2f}%{"  ← resonance" if delta > 0.3 else ""}')

    # Bandwidth savings at 90% accuracy
    print(f'\nMax bandwidth savings at 90%+ accuracy:')
    for ber in [0.0, 0.15, 0.30]:
        for rho in rhos:
            if d[str(rho)].get(str(ber), 0) >= 90:
                savings = (1 - rho) * 100
                label = 'Clean' if ber == 0 else f'BER={ber:.2f}'
                print(f'  {label}: rho={rho:.3f} -> {savings:.0f}% savings, acc={d[str(rho)][str(ber)]:.2f}%')
                break
        else:
            label = 'Clean' if ber == 0 else f'BER={ber:.2f}'
            print(f'  {label}: cannot reach 90%')

print('\nAll plots saved to eval/figures/')
