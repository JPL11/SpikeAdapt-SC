#!/usr/bin/env python3
"""Paired significance tests on ensemble results.

Tests key claims from the paper using the top-3 checkpoint-averaged per-seed
values instead of single-checkpoint values.

Outputs:
  - Paired t-test p-values
  - 95% BCa bootstrap CIs
  - Bonferroni correction
  - Comparison to original (single-checkpoint) claims
"""

import json
import numpy as np
from scipy import stats

SEEDS = ['42', '123', '456', '789', '1024', '2048', '3072', '4096', '5120', '6144']

with open('eval/seed_results/ensemble_top3_results.json') as f:
    ens = json.load(f)

with open('eval/seed_results/summary_10seed.json') as f:
    baseline = json.load(f)


def bootstrap_ci_bca(deltas, n_boot=10000, alpha=0.05):
    """Bias-corrected and accelerated bootstrap CI."""
    deltas = np.array(deltas)
    n = len(deltas)
    theta_hat = deltas.mean()

    # Bootstrap
    rng = np.random.default_rng(42)
    boot = np.array([rng.choice(deltas, n, replace=True).mean() for _ in range(n_boot)])

    # Bias correction
    z0 = stats.norm.ppf((boot < theta_hat).mean())

    # Acceleration (jackknife)
    jack = np.array([np.delete(deltas, i).mean() for i in range(n)])
    jack_mean = jack.mean()
    num = ((jack_mean - jack) ** 3).sum()
    den = 6 * ((jack_mean - jack) ** 2).sum() ** 1.5
    a = num / den if den != 0 else 0

    # Adjusted quantiles
    zl = stats.norm.ppf(alpha / 2)
    zu = stats.norm.ppf(1 - alpha / 2)
    pl = stats.norm.cdf(z0 + (z0 + zl) / (1 - a * (z0 + zl)))
    pu = stats.norm.cdf(z0 + (z0 + zu) / (1 - a * (z0 + zu)))

    return np.quantile(boot, pl), np.quantile(boot, pu)


def paired_test(A, B, name_A, name_B, ber=0.3):
    """Paired t-test: A - B."""
    deltas = [A[s] - B[s] for s in SEEDS if s in A and s in B]
    deltas = np.array(deltas)

    mean_d = deltas.mean()
    std_d = deltas.std(ddof=1)
    t, p = stats.ttest_rel(
        [A[s] for s in SEEDS if s in A and s in B],
        [B[s] for s in SEEDS if s in A and s in B]
    )
    ci_low, ci_high = bootstrap_ci_bca(deltas)

    print(f'\n  {name_A} vs {name_B} at BER={ber}:')
    print(f'    Delta: {mean_d:+.2f} pp (std {std_d:.2f})')
    print(f'    Paired t: t={t:.3f}, p={p:.4f}')
    print(f'    95% BCa CI: [{ci_low:+.2f}, {ci_high:+.2f}]')
    print(f'    CI excludes 0: {ci_low > 0 or ci_high < 0}')
    return {
        'mean_delta': float(mean_d), 'std_delta': float(std_d),
        't': float(t), 'p_value': float(p),
        'ci_low': float(ci_low), 'ci_high': float(ci_high),
        'ci_excludes_zero': bool(ci_low > 0 or ci_high < 0),
    }


def get_ensemble_dict(ds, rho, ber):
    """Extract per-seed ensemble values: {seed: acc}."""
    d = {}
    for s in SEEDS:
        if s in ens['per_seed'][ds]:
            d[s] = ens['per_seed'][ds][s][str(rho)][str(ber)]
    return d


def get_baseline_dict(ds, rho, ber):
    """Extract per-seed baseline values from summary_10seed.json."""
    rho_k = '1.0' if rho == 1.0 else str(rho)
    ber_k = '0.0' if ber == 0.0 else f'{ber}' if ber != 0.3 else '0.3'
    per_seed = baseline[ds][rho_k][ber_k]['per_seed']
    return {s: per_seed[i] for i, s in enumerate(SEEDS)}


print('=' * 80)
print('  SIGNIFICANCE TESTS ON ENSEMBLE (top-3 SWA)')
print('  Paired t-test + 95% BCa bootstrap CI, n=10 seeds')
print('=' * 80)

results = {'ensemble': {}, 'baseline': {}}

# === Key comparisons at BER=0.30 ===
for ds in ['aid', 'resisc45']:
    print(f'\n{"#"*60}')
    print(f'  {ds.upper()} — BER=0.30')
    print(f'{"#"*60}')

    # Ensemble data
    rho1 = get_ensemble_dict(ds, 1.0, 0.3)
    rho075 = get_ensemble_dict(ds, 0.75, 0.3)
    rho0625 = get_ensemble_dict(ds, 0.625, 0.3)

    print(f'\n  === ENSEMBLE (top-3 SWA) ===')
    r1 = paired_test(rho075, rho1, 'rho=0.75', 'rho=1.0')
    r2 = paired_test(rho0625, rho1, 'rho=0.625', 'rho=1.0')
    results['ensemble'][ds] = {'rho075_vs_rho1': r1, 'rho0625_vs_rho1': r2}

    # Baseline (existing paper)
    rho1_b = get_baseline_dict(ds, 1.0, 0.3)
    rho075_b = get_baseline_dict(ds, 0.75, 0.3)
    rho0625_b = get_baseline_dict(ds, 0.625, 0.3)

    print(f'\n  === BASELINE (single checkpoint, for reference) ===')
    rb1 = paired_test(rho075_b, rho1_b, 'rho=0.75', 'rho=1.0')
    rb2 = paired_test(rho0625_b, rho1_b, 'rho=0.625', 'rho=1.0')
    results['baseline'][ds] = {'rho075_vs_rho1': rb1, 'rho0625_vs_rho1': rb2}


# === Bonferroni correction ===
print(f'\n{"="*80}')
print('  BONFERRONI CORRECTION (4 tests: 2 comparisons × 2 datasets)')
print(f'{"="*80}')

p_raw = []
for ds in ['aid', 'resisc45']:
    for key in ['rho075_vs_rho1', 'rho0625_vs_rho1']:
        p_raw.append(results['ensemble'][ds][key]['p_value'])

p_corrected = np.minimum(np.array(p_raw) * 4, 1.0)
labels = [f'{ds} {k}' for ds in ['aid', 'resisc45'] for k in ['rho075_vs_rho1', 'rho0625_vs_rho1']]

print(f'\n  Ensemble results:')
print(f'  {"Comparison":<35} {"Raw p":<12} {"Bonferroni p":<15} {"Surv (α=0.05)"}')
print(f'  {"-"*75}')
for lbl, pr, pc in zip(labels, p_raw, p_corrected):
    surv = 'YES' if pc < 0.05 else 'no'
    print(f'  {lbl:<35} {pr:.4f}       {pc:.4f}          {surv}')


# === Side-by-side summary ===
print(f'\n{"="*80}')
print('  SIDE-BY-SIDE: Ensemble vs Baseline deltas at BER=0.30')
print(f'{"="*80}')

for ds in ['aid', 'resisc45']:
    print(f'\n  {ds.upper()}:')
    print(f'    {"Test":<25} {"Ensemble Δ (p)":<25} {"Baseline Δ (p)":<25}')
    print(f'    {"-"*75}')
    for key, name in [('rho075_vs_rho1', 'rho=0.75 vs rho=1.0'),
                        ('rho0625_vs_rho1', 'rho=0.625 vs rho=1.0')]:
        e = results['ensemble'][ds][key]
        b = results['baseline'][ds][key]
        print(f'    {name:<25} {e["mean_delta"]:+.2f} (p={e["p_value"]:.4f})       '
              f'{b["mean_delta"]:+.2f} (p={b["p_value"]:.4f})')


# Save
with open('eval/seed_results/ensemble_significance.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved to eval/seed_results/ensemble_significance.json')
