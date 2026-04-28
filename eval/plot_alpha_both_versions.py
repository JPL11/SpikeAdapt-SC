#!/usr/bin/env python3
"""Plot both α versions for classification:

  α₁ = (A(ρ) - A(ρ=1)) / A(ρ=1)    — relative to FULL transmission
  α₂ = (A(ρ) - A(ρ_min)) / A(ρ_min) — relative to MINIMAL transmission

α₁ answers: "does masking help vs sending everything?"
α₂ answers: "how much does increasing ρ help vs sending almost nothing?"

Uses eval/extended_ber_rho_results.json (BER 0.0 to 0.6).
Output: eval/figures/alpha_v1_vs_v2_*.pdf
"""

import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('eval/figures', exist_ok=True)

with open('eval/extended_ber_rho_results.json') as f:
    data = json.load(f)

datasets = [('aid', 'AID', 30), ('resisc45', 'RESISC45', 45)]


def get_rho_key(ds, rho):
    return [k for k in ds.keys() if abs(float(k) - rho) < 0.002][0]


# ====== Combined 2×3: Both α versions + difference ======
fig, axes = plt.subplots(2, 3, figsize=(20, 11))

for row, (ds_key, ds_name, n_cls) in enumerate(datasets):
    ds = data[ds_key]
    rho_vals = sorted([float(r) for r in ds.keys()])
    ber_vals = sorted([float(b) for b in ds[list(ds.keys())[0]].keys()])
    # Limit to BER ≤ 0.4 for readability
    ber_plot = [b for b in ber_vals if b <= 0.4]

    rho_max_key = get_rho_key(ds, 1.0)
    rho_min_key = get_rho_key(ds, rho_vals[0])  # smallest ρ (0.1)

    a_full = {ber: ds[rho_max_key][str(ber)] for ber in ber_vals}  # A(ρ=1)
    a_min = {ber: ds[rho_min_key][str(ber)] for ber in ber_vals}   # A(ρ_min)
    chance = 100.0 / n_cls

    cmap = plt.cm.RdYlBu_r
    norm = plt.Normalize(0, 0.4)

    # --- Column 0: α₁ = (A(ρ) - A(ρ=1)) / A(ρ=1) ---
    ax = axes[row, 0]
    for ber in ber_plot:
        alphas = []
        for r in rho_vals:
            acc = ds[get_rho_key(ds, r)][str(ber)]
            a = (acc - a_full[ber]) / a_full[ber] if a_full[ber] > 0.1 else 0
            alphas.append(a)
        color = cmap(norm(ber))
        label = f'BER={ber:.2f}' if ber > 0 else 'Clean'
        ax.plot(rho_vals, alphas, 'o-', color=color, linewidth=1.8, markersize=4, label=label)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_xlabel('Transmission rate (rho)', fontsize=11)
    ax.set_ylabel('alpha_1', fontsize=12)
    title = 'alpha_1 = (A(rho) - A(1)) / A(1)' if row == 0 else ''
    ax.set_title(f'{ds_name}: ' + title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=6.5, ncol=3, loc='best')
    ax.set_xlim(0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # --- Column 1: α₂ = (A(ρ) - A(ρ_min)) / A(ρ_min) ---
    ax = axes[row, 1]
    for ber in ber_plot:
        alphas = []
        for r in rho_vals:
            acc = ds[get_rho_key(ds, r)][str(ber)]
            a = (acc - a_min[ber]) / a_min[ber] if a_min[ber] > 0.1 else 0
            alphas.append(a)
        color = cmap(norm(ber))
        label = f'BER={ber:.2f}' if ber > 0 else 'Clean'
        ax.plot(rho_vals, alphas, 'o-', color=color, linewidth=1.8, markersize=4, label=label)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_xlabel('Transmission rate (rho)', fontsize=11)
    ax.set_ylabel('alpha_2', fontsize=12)
    title = 'alpha_2 = (A(rho) - A(rho_min)) / A(rho_min)' if row == 0 else ''
    ax.set_title(f'{ds_name}: ' + title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=6.5, ncol=3, loc='best')
    ax.set_xlim(0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # --- Column 2: Accuracy vs BER at key ρ values (context) ---
    ax = axes[row, 2]
    rho_subset = [0.1, 0.25, 0.5, 0.75, 1.0]
    colors_rho = ['#e63946', '#f4a261', '#2a9d8f', '#264653', '#6d6875']
    for i, rho in enumerate(rho_subset):
        rkey = get_rho_key(ds, rho)
        accs = [ds[rkey][str(ber)] for ber in ber_plot]
        ax.plot(ber_plot, accs, 'o-', color=colors_rho[i], linewidth=2, markersize=5,
                label=f'rho={rho}')
    ax.axhline(y=chance, color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('BER', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title(f'{ds_name}: Accuracy vs BER', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.suptitle('Classification: Two Alpha Metrics Compared (BER 0.0 - 0.4)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig('eval/figures/alpha_v1_vs_v2_both.pdf', dpi=300, bbox_inches='tight')
fig.savefig('eval/figures/alpha_v1_vs_v2_both.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: eval/figures/alpha_v1_vs_v2_both.pdf')


# ====== Print comparison tables ======
for ds_key, ds_name, n_cls in datasets:
    ds = data[ds_key]
    rho_vals = sorted([float(r) for r in ds.keys()])
    ber_vals = [b for b in sorted([float(b) for b in ds[list(ds.keys())[0]].keys()]) if b <= 0.4]

    rho_max_key = get_rho_key(ds, 1.0)
    rho_min_key = get_rho_key(ds, rho_vals[0])
    a_full = {ber: ds[rho_max_key][str(ber)] for ber in ber_vals}
    a_min = {ber: ds[rho_min_key][str(ber)] for ber in ber_vals}

    print(f'\n{"="*90}')
    print(f'{ds_name} — alpha_1: (A(rho) - A(1)) / A(1)  [relative to FULL transmission]')
    print(f'{"="*90}')
    header = f"{'rho':>8}" + ''.join(f' {b:.2f}' for b in ber_vals)
    print(header); print('-' * len(header))
    for rho in rho_vals:
        rkey = get_rho_key(ds, rho)
        row = f'{rho:>8.3f}'
        for ber in ber_vals:
            acc = ds[rkey][str(ber)]
            a = (acc - a_full[ber]) / a_full[ber] if a_full[ber] > 0.1 else 0
            row += f' {a:+5.2f}'
        print(row)

    print(f'\n{ds_name} — alpha_2: (A(rho) - A(rho_min)) / A(rho_min)  [relative to MIN transmission]')
    print(f'rho_min = {rho_vals[0]:.3f}')
    print(header); print('-' * len(header))
    for rho in rho_vals:
        rkey = get_rho_key(ds, rho)
        row = f'{rho:>8.3f}'
        for ber in ber_vals:
            acc = ds[rkey][str(ber)]
            a = (acc - a_min[ber]) / a_min[ber] if a_min[ber] > 0.1 else 0
            row += f' {a:+5.2f}'
        print(row)

    print(f'\nKey difference:')
    print(f'  alpha_1 at BER=0.35, rho=0.375: masking HELPS vs full tx '
          f'(alpha_1 = {(ds[get_rho_key(ds,0.375)]["0.35"] - a_full[0.35]) / max(a_full[0.35],0.1):+.3f})')
    print(f'  alpha_2 at BER=0.35, rho=0.375: big gain vs min tx '
          f'(alpha_2 = {(ds[get_rho_key(ds,0.375)]["0.35"] - a_min[0.35]) / max(a_min[0.35],0.1):+.3f})')
