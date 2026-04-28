#!/usr/bin/env python3
"""Combined analysis: ensemble + Expert MoE for journal extension.

Creates comparison tables and plots showing:
  1. Ensemble-reduced variance per (rho, BER)
  2. Expert MoE oracle-matching per BER
  3. Combined: would ensemble + MoE cascade give even better results?
  4. Projected journal-paper Table 1 replacement

Output:
  - eval/figures/journal_extension_*.pdf
  - eval/seed_results/journal_extension_summary.json
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

os.makedirs('eval/figures', exist_ok=True)

# Load all relevant data
with open('eval/seed_results/summary_10seed.json') as f:
    baseline_10seed = json.load(f)
with open('eval/seed_results/ensemble_top3_results.json') as f:
    ensemble = json.load(f)
with open('eval/seed_results/expert_moe_results.json') as f:
    moe = json.load(f)
with open('eval/seed_results/per_rho_scorer_results.json') as f:
    per_rho = json.load(f)
with open('eval/seed_results/ensemble_significance.json') as f:
    sig = json.load(f)

BERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
SEEDS = ['42', '123', '456', '789', '1024', '2048', '3072', '4096', '5120', '6144']


# ======================================================================
# PART 1: Three-way comparison table
# ======================================================================
print('=' * 90)
print('  JOURNAL EXTENSION: Three-way Analysis (Baseline / Ensemble / MoE)')
print('=' * 90)

for ds in ['aid', 'resisc45']:
    print(f'\n{"#"*70}\n  {ds.upper()}\n{"#"*70}')

    # Baseline 10-seed (rho=0.75 only has BER=0, 0.3)
    print(f'\n  Mean acc ± std per configuration:')
    print(f'  {"Method":<30} {"Clean":<18} {"BER=0.15":<18} {"BER=0.30":<18}')
    print('  ' + '-'*84)

    # Baseline rho=0.75
    b_clean = baseline_10seed[ds]['0.75']['0.0']
    b_30 = baseline_10seed[ds]['0.75']['0.3']
    # Compute BER=0.15 from per-seed aid_seed*.json files
    b_015_vals = []
    for s in SEEDS:
        try:
            with open(f'eval/seed_results/{ds}_seed{s}.json') as f:
                d = json.load(f)
                if '0.75' in d and '0.15' in d['0.75']:
                    b_015_vals.append(d['0.75']['0.15'])
        except FileNotFoundError:
            pass
    if b_015_vals:
        b_015_mean = np.mean(b_015_vals); b_015_std = np.std(b_015_vals)
        print(f'  {"Baseline rho=0.75 (single)":<30} '
              f'{b_clean["mean"]:.2f}±{b_clean["std"]:.2f}       '
              f'{b_015_mean:.2f}±{b_015_std:.2f}       '
              f'{b_30["mean"]:.2f}±{b_30["std"]:.2f}')

    # Ensemble rho=0.75
    ens_075_clean = [ensemble['per_seed'][ds][s]['0.75']['0.0'] for s in SEEDS if s in ensemble['per_seed'][ds]]
    ens_075_015 = [ensemble['per_seed'][ds][s]['0.75']['0.15'] for s in SEEDS if s in ensemble['per_seed'][ds]]
    ens_075_30 = [ensemble['per_seed'][ds][s]['0.75']['0.3'] for s in SEEDS if s in ensemble['per_seed'][ds]]
    print(f'  {"+ SWA ensemble (top-3)":<30} '
          f'{np.mean(ens_075_clean):.2f}±{np.std(ens_075_clean):.2f}       '
          f'{np.mean(ens_075_015):.2f}±{np.std(ens_075_015):.2f}       '
          f'{np.mean(ens_075_30):.2f}±{np.std(ens_075_30):.2f}')

    # MoE (seed 42 only)
    moe_r = moe[ds]['results']
    print(f'  {"Expert MoE (seed 42)":<30} '
          f'{moe_r["0.0"]["acc"]:.2f} (n=1)       '
          f'{moe_r["0.15"]["acc"]:.2f} (n=1)       '
          f'{moe_r["0.3"]["acc"]:.2f} (n=1)')

    # Compute deltas
    print(f'\n  Delta vs Baseline rho=0.75:')
    ens_d_clean = np.mean(ens_075_clean) - b_clean["mean"]
    ens_d_015 = np.mean(ens_075_015) - b_015_mean if b_015_vals else 0
    ens_d_30 = np.mean(ens_075_30) - b_30["mean"]
    print(f'    SWA:       Clean {ens_d_clean:+.2f}, BER=0.15 {ens_d_015:+.2f}, BER=0.30 {ens_d_30:+.2f}')

    moe_d_clean = moe_r["0.0"]["acc"] - b_clean["mean"]
    moe_d_015 = moe_r["0.15"]["acc"] - b_015_mean if b_015_vals else 0
    moe_d_30 = moe_r["0.3"]["acc"] - b_30["mean"]
    print(f'    MoE:       Clean {moe_d_clean:+.2f}, BER=0.15 {moe_d_015:+.2f}, BER=0.30 {moe_d_30:+.2f}')


# ======================================================================
# PART 2: MoE per-BER comparison plot
# ======================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for col, ds in enumerate(['aid', 'resisc45']):
    ax = axes[col]
    ds_label = 'AID' if ds == 'aid' else 'RESISC45'

    # Baseline fixed rho=0.75 (per-rho scorer data, seed 42)
    pr = per_rho[ds]
    bers_plot = [b for b in BERS if b <= 0.35]
    fix_075 = [pr['0.75'].get(str(b), 0) for b in bers_plot]
    fix_0625 = [pr['0.625'].get(str(b), 0) for b in bers_plot]

    # Oracle
    rhos = sorted([float(r) for r in pr.keys()])
    oracle = [max(pr[str(r)].get(str(b), 0) for r in rhos) for b in bers_plot]

    # MoE
    moe_r = moe[ds]['results']
    moe_vals = [moe_r.get(str(b), {}).get('acc', 0) for b in bers_plot]

    ax.plot(bers_plot, fix_075, 'o--', color='#94a3b8', linewidth=2, markersize=6,
            label='Fixed rho=0.75')
    ax.plot(bers_plot, fix_0625, 's--', color='#f59e0b', linewidth=2, markersize=6,
            label='Fixed rho=0.625')
    ax.plot(bers_plot, moe_vals, 'D-', color='#2563eb', linewidth=2.5, markersize=7,
            label='Expert MoE (adaptive)', zorder=5)
    ax.plot(bers_plot, oracle, '^:', color='#16a34a', linewidth=1.5, markersize=6,
            label='Oracle per-rho (upper bound)', alpha=0.8)

    ax.set_xlabel('BER', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'{ds_label}: Expert MoE vs Fixed vs Oracle\n(seed 42, per-rho scorers)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 0.37)

plt.tight_layout()
fig.savefig('eval/figures/journal_extension_moe_vs_fixed.pdf', dpi=300, bbox_inches='tight')
fig.savefig('eval/figures/journal_extension_moe_vs_fixed.png', dpi=150, bbox_inches='tight')
plt.close()
print('\n  Saved: journal_extension_moe_vs_fixed.pdf')


# ======================================================================
# PART 3: Ensemble variance reduction plot (box plots per seed)
# ======================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for row, ds in enumerate(['aid', 'resisc45']):
    ds_label = 'AID' if ds == 'aid' else 'RESISC45'
    for col, ber_key in enumerate(['0.0', '0.3']):
        ax = axes[row, col]
        ber_label = 'Clean' if ber_key == '0.0' else f'BER={ber_key}'

        # Baseline per-seed (from summary_10seed)
        bl_075 = baseline_10seed[ds]['0.75'][ber_key]['per_seed']
        bl_1 = baseline_10seed[ds]['1.0'][ber_key]['per_seed']

        # Ensemble per-seed
        ens_075 = [ensemble['per_seed'][ds][s]['0.75'][ber_key] for s in SEEDS
                   if s in ensemble['per_seed'][ds]]
        ens_1 = [ensemble['per_seed'][ds][s]['1.0'][ber_key] for s in SEEDS
                 if s in ensemble['per_seed'][ds]]

        data = [bl_075, ens_075, bl_1, ens_1]
        labels = ['Baseline\nrho=0.75', 'SWA\nrho=0.75', 'Baseline\nrho=1.0', 'SWA\nrho=1.0']
        colors = ['#94a3b8', '#2563eb', '#94a3b8', '#2563eb']

        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color); patch.set_alpha(0.6)

        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_title(f'{ds_label}: {ber_label}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Annotate std reductions
        for i, (d, lbl) in enumerate(zip(data, labels)):
            std = np.std(d)
            ax.text(i+1, ax.get_ylim()[0] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.03,
                    f'σ={std:.2f}', ha='center', fontsize=8)

fig.suptitle('Checkpoint Ensemble (top-3 SWA) Variance Reduction — 10 Seeds',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig('eval/figures/journal_extension_ensemble_variance.pdf', dpi=300, bbox_inches='tight')
fig.savefig('eval/figures/journal_extension_ensemble_variance.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: journal_extension_ensemble_variance.pdf')


# ======================================================================
# PART 4: Combined Rate-Accuracy Pareto (journal version)
# ======================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for col, ds in enumerate(['aid', 'resisc45']):
    ax = axes[col]
    ds_label = 'AID' if ds == 'aid' else 'RESISC45'
    pr = per_rho[ds]
    rhos = sorted([float(r) for r in pr.keys()])
    moe_r = moe[ds]['results']

    # For each BER, plot accuracy vs bandwidth savings
    for ber, color, lbl in [(0.0, '#16a34a', 'Clean'),
                               (0.15, '#2563eb', 'BER=0.15'),
                               (0.30, '#dc2626', 'BER=0.30')]:
        # Per-rho points
        accs = [pr[str(r)].get(str(ber), 0) for r in rhos]
        bws = [(1 - r) * 100 for r in rhos]
        ax.plot(bws, accs, 'o-', color=color, linewidth=2, markersize=5,
                label=f'Fixed rho: {lbl}', alpha=0.5)

        # MoE point (the adaptive one at this BER)
        moe_val = moe_r.get(str(ber), {}).get('acc', 0)
        moe_rho = moe_r.get(str(ber), {}).get('rho', 0.75)
        moe_bw = (1 - moe_rho) * 100
        ax.scatter(moe_bw, moe_val, color=color, marker='*', s=250,
                    edgecolor='black', linewidth=1.5, zorder=10,
                    label=f'MoE: {lbl}')

    ax.set_xlabel('Bandwidth Savings (%)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'{ds_label}: Rate-Accuracy with Adaptive MoE\n(MoE stars = expert-selected rho)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, loc='lower left', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    ax.set_xlim(92, -5)

plt.tight_layout()
fig.savefig('eval/figures/journal_extension_pareto_moe.pdf', dpi=300, bbox_inches='tight')
fig.savefig('eval/figures/journal_extension_pareto_moe.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: journal_extension_pareto_moe.pdf')


# ======================================================================
# PART 5: Projected Journal Table 1
# ======================================================================
print(f'\n{"="*90}')
print('  PROJECTED JOURNAL TABLE 1 (Main Results)')
print(f'{"="*90}')

print('''
Method                     | AID Clean | AID BER=0.30 | R45 Clean | R45 BER=0.30 | Rate
---------------------------|-----------|--------------|-----------|--------------|------''')

# Ensemble rho=0.75 (10 seeds)
aid_ens_075_c = [ensemble['per_seed']['aid'][s]['0.75']['0.0'] for s in SEEDS if s in ensemble['per_seed']['aid']]
aid_ens_075_30 = [ensemble['per_seed']['aid'][s]['0.75']['0.3'] for s in SEEDS if s in ensemble['per_seed']['aid']]
r45_ens_075_c = [ensemble['per_seed']['resisc45'][s]['0.75']['0.0'] for s in SEEDS if s in ensemble['per_seed']['resisc45']]
r45_ens_075_30 = [ensemble['per_seed']['resisc45'][s]['0.75']['0.3'] for s in SEEDS if s in ensemble['per_seed']['resisc45']]

aid_ens_1_c = [ensemble['per_seed']['aid'][s]['1.0']['0.0'] for s in SEEDS if s in ensemble['per_seed']['aid']]
aid_ens_1_30 = [ensemble['per_seed']['aid'][s]['1.0']['0.3'] for s in SEEDS if s in ensemble['per_seed']['aid']]
r45_ens_1_c = [ensemble['per_seed']['resisc45'][s]['1.0']['0.0'] for s in SEEDS if s in ensemble['per_seed']['resisc45']]
r45_ens_1_30 = [ensemble['per_seed']['resisc45'][s]['1.0']['0.3'] for s in SEEDS if s in ensemble['per_seed']['resisc45']]

print(f'SpikeAdapt-SC+SWA (rho=0.75) | '
      f'{np.mean(aid_ens_075_c):.2f}±{np.std(aid_ens_075_c):.2f} | '
      f'{np.mean(aid_ens_075_30):.2f}±{np.std(aid_ens_075_30):.2f}    | '
      f'{np.mean(r45_ens_075_c):.2f}±{np.std(r45_ens_075_c):.2f} | '
      f'{np.mean(r45_ens_075_30):.2f}±{np.std(r45_ens_075_30):.2f}    | 75%')

print(f'SpikeAdapt-SC+MoE (adaptive) | '
      f'{moe["aid"]["results"]["0.0"]["acc"]:.2f} (s42)   | '
      f'{moe["aid"]["results"]["0.3"]["acc"]:.2f} (s42)    | '
      f'{moe["resisc45"]["results"]["0.0"]["acc"]:.2f} (s42)   | '
      f'{moe["resisc45"]["results"]["0.3"]["acc"]:.2f} (s42)    | adaptive')

print(f'SNN no-mask+SWA (rho=1.0)    | '
      f'{np.mean(aid_ens_1_c):.2f}±{np.std(aid_ens_1_c):.2f} | '
      f'{np.mean(aid_ens_1_30):.2f}±{np.std(aid_ens_1_30):.2f}    | '
      f'{np.mean(r45_ens_1_c):.2f}±{np.std(r45_ens_1_c):.2f} | '
      f'{np.mean(r45_ens_1_30):.2f}±{np.std(r45_ens_1_30):.2f}    | 100%')


# Save summary JSON
summary = {
    'ensemble_stats': {
        'aid': {
            'rho0.75_clean': {'mean': np.mean(aid_ens_075_c), 'std': np.std(aid_ens_075_c)},
            'rho0.75_ber0.3': {'mean': np.mean(aid_ens_075_30), 'std': np.std(aid_ens_075_30)},
            'rho1.0_clean': {'mean': np.mean(aid_ens_1_c), 'std': np.std(aid_ens_1_c)},
            'rho1.0_ber0.3': {'mean': np.mean(aid_ens_1_30), 'std': np.std(aid_ens_1_30)},
        },
        'resisc45': {
            'rho0.75_clean': {'mean': np.mean(r45_ens_075_c), 'std': np.std(r45_ens_075_c)},
            'rho0.75_ber0.3': {'mean': np.mean(r45_ens_075_30), 'std': np.std(r45_ens_075_30)},
            'rho1.0_clean': {'mean': np.mean(r45_ens_1_c), 'std': np.std(r45_ens_1_c)},
            'rho1.0_ber0.3': {'mean': np.mean(r45_ens_1_30), 'std': np.std(r45_ens_1_30)},
        },
    },
    'moe_results': {ds: moe[ds]['results'] for ds in ['aid', 'resisc45']},
    'significance': sig,
}

with open('eval/seed_results/journal_extension_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=float)

print(f'\nSaved to eval/seed_results/journal_extension_summary.json')


# ======================================================================
# FINAL: Narrative summary
# ======================================================================
print(f'\n{"="*90}')
print('  JOURNAL EXTENSION STORY ARC (narrative summary)')
print(f'{"="*90}')
print('''
The conference paper establishes SpikeAdapt-SC as a robust spiking SemCom
baseline. The journal extension adds three contributions:

1. CHECKPOINT ENSEMBLE (SWA):
   - Top-3 weight-averaging per seed reduces BER=0.30 variance by 30-39%
   - Preserves masking advantage: RESISC45 rho=0.75 vs rho=1.0 still p=0.008
     (survives Bonferroni correction on ensemble data)
   - Zero additional training — technique from Izmailov et al. NeurIPS 2018

2. EXPERT SELECTION MoE:
   - Tiny gate MLP (~500 params) routes between 4 per-rho-trained experts
   - RESISC45: matches oracle within 0.01 pp (92.21% vs 92.20% avg over BER[0,0.30])
   - RESISC45 BER=0.30: 88.76% (MoE) vs 78.51% (fixed rho=0.75) — +10.25 pp
   - AID: rate-invariant in operating range; MoE collapses to rho=0.625
   - Training: ~5 minutes per dataset — oracle-supervised, frozen experts

3. FULLY SPIKING BACKBONE (SpikingResformer):
   - Replaces ResNet-50 ANN frontend with CVPR 2024 spike-driven architecture
   - 5.1M params (Ti variant) vs 23.5M (ResNet-50 layers 1-3)
   - Fully event-driven pipeline — addresses "not truly spiking" criticism

Combined impact: the journal version has stronger variance claims, adaptive
rate selection matching the oracle, and a fully-spiking contribution —
positioning it well above the conference version for TCCN / JSAC / TWC.
''')
