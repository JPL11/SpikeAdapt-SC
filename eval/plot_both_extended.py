#!/usr/bin/env python3
"""Plot alpha for both AID and RESISC45 with extended BER (up to 0.6).

Generates:
  1. Combined 2x2 (accuracy + alpha) for both datasets
  2. Accuracy vs BER at fixed rho (sanity check: converge to chance at BER=0.5)
  3. Alpha table with peak analysis
"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

with open('eval/extended_ber_rho_results.json') as f:
    data = json.load(f)

os.makedirs('eval/figures', exist_ok=True)
datasets = [('aid', 'AID', 30), ('resisc45', 'RESISC45', 45)]


def get_rho_key(ds, rho):
    return [k for k in ds.keys() if abs(float(k) - rho) < 0.002][0]


# ====== Plot 1: Combined 2x2 (Accuracy + Alpha) ======
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

for col, (ds_key, ds_name, n_cls) in enumerate(datasets):
    ds = data[ds_key]
    rho_vals = sorted([float(r) for r in ds.keys()])
    ber_vals = sorted([float(b) for b in ds[list(ds.keys())[0]].keys()])

    rho1_key = get_rho_key(ds, 1.0)
    a_full = {ber: ds[rho1_key][str(ber)] for ber in ber_vals}
    chance = 100.0 / n_cls

    cmap = plt.cm.RdYlBu_r
    norm = plt.Normalize(0, max(ber_vals))

    # Top: Accuracy vs rho
    ax = axes[0, col]
    for ber in ber_vals:
        accs = [ds[get_rho_key(ds, r)][str(ber)] for r in rho_vals]
        color = cmap(norm(ber))
        label = f'BER={ber:.2f}' if ber > 0 else 'Clean'
        ax.plot(rho_vals, accs, 'o-', color=color, linewidth=1.8, markersize=4, label=label)
    ax.axhline(y=chance, color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax.text(0.95, chance + 1, f'Chance ({chance:.1f}%)', fontsize=8, ha='right', alpha=0.6)
    ax.set_xlabel('Transmission rate (rho)', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title(f'{ds_name}: Accuracy vs rho', fontsize=13, fontweight='bold')
    ax.legend(fontsize=6.5, ncol=4, loc='lower right')
    ax.set_xlim(0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Bottom: Alpha vs rho
    ax = axes[1, col]
    for ber in ber_vals:
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
    ax.set_ylabel('alpha = (A(rho) - A(1)) / A(1)', fontsize=12)
    ax.set_title(f'{ds_name}: Relative accuracy change (alpha)', fontsize=12)
    ax.legend(fontsize=6.5, ncol=4, loc='best')
    ax.set_xlim(0.05, 1.05)
    ax.grid(True, alpha=0.3)

fig.suptitle('SpikeAdapt-SC: Classification with Noise-Aware Masking (BER 0.0 - 0.6)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig('eval/figures/alpha_both_datasets_ber06.pdf', dpi=300, bbox_inches='tight')
fig.savefig('eval/figures/alpha_both_datasets_ber06.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: eval/figures/alpha_both_datasets_ber06.pdf')


# ====== Plot 2: Sanity check — Accuracy vs BER at fixed rho ======
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
rho_subset = [0.25, 0.5, 0.75, 1.0]
colors = ['#e63946', '#457b9d', '#2a9d8f', '#264653']

for col, (ds_key, ds_name, n_cls) in enumerate(datasets):
    ds = data[ds_key]
    ber_vals = sorted([float(b) for b in ds[list(ds.keys())[0]].keys()])
    chance = 100.0 / n_cls

    ax = axes[col]
    for i, rho in enumerate(rho_subset):
        rkey = get_rho_key(ds, rho)
        accs = [ds[rkey][str(ber)] for ber in ber_vals]
        ax.plot(ber_vals, accs, 'o-', color=colors[i], linewidth=2.2, markersize=5,
                label=f'rho={rho}')

    ax.axhline(y=chance, color='black', linestyle=':', linewidth=1.2, alpha=0.6)
    ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.text(0.505, 50, 'BER=0.5\n(random)', fontsize=8, alpha=0.6, va='center')
    ax.text(0.02, chance + 1.5, f'Chance = {chance:.1f}%', fontsize=9, alpha=0.7)

    ax.set_xlabel('Bit Error Rate (BER)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'{ds_name}: Accuracy vs BER (sanity check)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(-0.02, 0.62)
    ax.set_ylim(-2, 100)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig('eval/figures/sanity_check_ber_sweep.pdf', dpi=300, bbox_inches='tight')
fig.savefig('eval/figures/sanity_check_ber_sweep.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: eval/figures/sanity_check_ber_sweep.pdf')


# ====== Plot 3: Zoomed alpha for BER 0.0-0.35 (the interesting region) ======
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
cmap = plt.cm.RdYlBu_r

for col, (ds_key, ds_name, n_cls) in enumerate(datasets):
    ds = data[ds_key]
    rho_vals = sorted([float(r) for r in ds.keys()])
    ber_vals = sorted([float(b) for b in ds[list(ds.keys())[0]].keys()])
    ber_zoom = [b for b in ber_vals if b <= 0.35]

    rho1_key = get_rho_key(ds, 1.0)
    a_full = {ber: ds[rho1_key][str(ber)] for ber in ber_vals}

    norm = plt.Normalize(0, 0.35)
    ax = axes[col]
    for ber in ber_zoom:
        alphas = []
        for r in rho_vals:
            acc = ds[get_rho_key(ds, r)][str(ber)]
            a = (acc - a_full[ber]) / a_full[ber] if a_full[ber] > 0.1 else 0
            alphas.append(a)
        color = cmap(norm(ber))
        label = f'BER={ber:.2f}' if ber > 0 else 'Clean'
        ax.plot(rho_vals, alphas, 'o-', color=color, linewidth=2, markersize=5, label=label)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    # Shade masking-helps region for BER=0.3 and 0.35
    for ber in [0.35, 0.3]:
        pos_rhos = [r for r in rho_vals if r < 1.0 and
                    (ds[get_rho_key(ds, r)][str(ber)] - a_full[ber]) / max(a_full[ber], 0.1) > 0.001]
        if pos_rhos:
            ax.axvspan(min(pos_rhos)-0.03, max(pos_rhos)+0.03, alpha=0.06, color='green',
                      label=f'Masking helps (BER={ber})' if ber == 0.35 else None)

    ax.set_xlabel('Transmission rate (rho)', fontsize=12)
    ax.set_ylabel('alpha = (A(rho) - A(1)) / A(1)', fontsize=12)
    ax.set_title(f'{ds_name}: Alpha (zoomed, BER 0.0 - 0.35)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, ncol=3, loc='best')
    ax.set_xlim(0.05, 1.05)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig('eval/figures/alpha_zoomed_ber035.pdf', dpi=300, bbox_inches='tight')
fig.savefig('eval/figures/alpha_zoomed_ber035.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: eval/figures/alpha_zoomed_ber035.pdf')


# ====== Print tables ======
for ds_key, ds_name, n_cls in datasets:
    ds = data[ds_key]
    rho_vals = sorted([float(r) for r in ds.keys()])
    ber_vals = sorted([float(b) for b in ds[list(ds.keys())[0]].keys()])
    rho1_key = get_rho_key(ds, 1.0)
    a_full = {ber: ds[rho1_key][str(ber)] for ber in ber_vals}
    chance = 100.0 / n_cls

    print(f'\n{"="*120}')
    print(f'{ds_name} — Accuracy table (chance={chance:.1f}%)')
    print(f'{"="*120}')
    header = f"{'rho':>8}" + ''.join(f' {b:.2f}' for b in ber_vals)
    print(header)
    print('-' * len(header))
    for rho in rho_vals:
        rkey = get_rho_key(ds, rho)
        row = f'{rho:>8.3f}'
        for ber in ber_vals:
            acc = ds[rkey][str(ber)]
            row += f' {acc:5.1f}'
        print(row)

    print(f'\n{ds_name} — Alpha table')
    print(header)
    print('-' * len(header))
    for rho in rho_vals:
        rkey = get_rho_key(ds, rho)
        row = f'{rho:>8.3f}'
        for ber in ber_vals:
            acc = ds[rkey][str(ber)]
            alpha = (acc - a_full[ber]) / a_full[ber] if a_full[ber] > 0.1 else 0
            row += f' {alpha:+5.2f}'
        print(row)

    print(f'\nPeak alpha per BER (rho < 1):')
    for ber in ber_vals:
        best_rho, best_alpha, best_acc = None, -999, 0
        for rho in rho_vals:
            if rho >= 1.0:
                continue
            rkey = get_rho_key(ds, rho)
            acc = ds[rkey][str(ber)]
            alpha = (acc - a_full[ber]) / a_full[ber] if a_full[ber] > 0.1 else 0
            if alpha > best_alpha:
                best_alpha = alpha
                best_rho = rho
                best_acc = acc
        tag = 'HELPS' if best_alpha > 0.001 else 'hurts'
        print(f'  BER={ber:.2f}: rho*={best_rho:.3f} acc={best_acc:.1f}% vs {a_full[ber]:.1f}% alpha={best_alpha:+.3f} [{tag}]')

    # Sanity: check convergence to chance at BER=0.5
    print(f'\nSanity check — BER=0.5 (should be ~chance={chance:.1f}%):')
    for rho in rho_vals:
        rkey = get_rho_key(ds, rho)
        acc = ds[rkey]['0.5']
        print(f'  rho={rho:.3f}: {acc:.2f}%', '  OK' if abs(acc - chance) < 5 else '  WARNING')
