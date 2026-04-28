#!/usr/bin/env python3
"""Compare adaptive rho options (1, 2, 3) against fixed-rho baselines.

Produces side-by-side comparison tables and failure analysis.
"""

import json
import numpy as np

with open('eval/seed_results/adaptive_rho_opt1_results.json') as f:
    opt1 = json.load(f)
with open('eval/seed_results/adaptive_rho_opt2_results.json') as f:
    opt2 = json.load(f)
with open('eval/seed_results/adaptive_rho_opt3_results.json') as f:
    opt3 = json.load(f)
with open('eval/seed_results/per_rho_scorer_results.json') as f:
    per_rho = json.load(f)

BERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

print('=' * 100)
print('ADAPTIVE RHO COMPARISON — seed 42')
print('=' * 100)

for ds in ['aid', 'resisc45']:
    print(f'\n{"#"*80}\n  {ds.upper()}\n{"#"*80}')
    print(f'\n{"BER":<8} {"Fix0.75":>10} {"Fix0.625":>10} {"Oracle":>10} {"Opt1(MLP)":>12} {"Opt2(MoE)":>12} {"Opt3(Ent)":>12}')
    print('-' * 80)

    for ber in BERS:
        ber_s = str(ber)

        # Baselines (per-rho scorers)
        fix_075 = per_rho[ds].get('0.75', {}).get(ber_s, None)
        fix_0625 = per_rho[ds].get('0.625', {}).get(ber_s, None)

        # Oracle = best rho at this BER
        rhos = sorted([float(r) for r in per_rho[ds].keys()])
        oracle_acc = max((per_rho[ds][str(r)].get(ber_s, 0), r) for r in rhos)
        oracle_val = oracle_acc[0]
        oracle_rho = oracle_acc[1]

        # Adaptive options
        o1 = opt1[ds].get(ber_s, {})
        o2 = opt2[ds].get(ber_s, {})
        o3 = opt3[ds].get(ber_s, {})

        o1_acc = o1.get('acc', 0); o1_rho = o1.get('rho', 0)
        o2_acc = o2.get('acc', 0); o2_rho = o2.get('rho', 0)
        o3_acc = o3.get('acc', 0); o3_rho = o3.get('rho', 0)

        print(f'{ber:<8.2f} '
              f'{fix_075:>10.2f} {fix_0625:>10.2f} '
              f'{oracle_val:>6.2f}@{oracle_rho:.2f} '
              f'{o1_acc:>6.2f}@{o1_rho:.2f} '
              f'{o2_acc:>6.2f}@{o2_rho:.2f} '
              f'{o3_acc:>6.2f}@{o3_rho:.2f}')


    # Compute deltas vs best baseline (oracle per-rho)
    print(f'\n  DELTAS vs ORACLE (per-rho scorer best at each BER):')
    print(f'  {"BER":<8} {"Opt1":>10} {"Opt2":>10} {"Opt3":>10}')
    print('  ' + '-'*40)
    for ber in [0.0, 0.15, 0.30, 0.35]:
        ber_s = str(ber)
        rhos = sorted([float(r) for r in per_rho[ds].keys()])
        oracle_val = max(per_rho[ds][str(r)].get(ber_s, 0) for r in rhos)
        d1 = opt1[ds].get(ber_s, {}).get('acc', 0) - oracle_val
        d2 = opt2[ds].get(ber_s, {}).get('acc', 0) - oracle_val
        d3 = opt3[ds].get(ber_s, {}).get('acc', 0) - oracle_val
        print(f'  {ber:<8.2f} {d1:>+10.2f} {d2:>+10.2f} {d3:>+10.2f}')

    # Check if rho is actually adapting
    print(f'\n  RHO ADAPTATION CHECK (variance of rho across BER):')
    rhos_o1 = [opt1[ds].get(str(b), {}).get('rho', 0) for b in BERS]
    rhos_o2 = [opt2[ds].get(str(b), {}).get('rho', 0) for b in BERS]
    rhos_o3 = [opt3[ds].get(str(b), {}).get('rho', 0) for b in BERS]
    print(f'    Opt1: min={min(rhos_o1):.3f}, max={max(rhos_o1):.3f}, std={np.std(rhos_o1):.4f}')
    print(f'    Opt2: min={min(rhos_o2):.3f}, max={max(rhos_o2):.3f}, std={np.std(rhos_o2):.4f}')
    print(f'    Opt3: min={min(rhos_o3):.3f}, max={max(rhos_o3):.3f}, std={np.std(rhos_o3):.4f}')

# Summary
print(f'\n{"="*100}')
print('SUMMARY')
print(f'{"="*100}')

for ds in ['aid', 'resisc45']:
    print(f'\n{ds.upper()}:')
    # Average accuracy over BER [0, 0.30]
    oracle_avg = np.mean([max(per_rho[ds][str(r)].get(str(b), 0)
                              for r in [0.1, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])
                          for b in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]])
    fix075_avg = np.mean([per_rho[ds]['0.75'].get(str(b), 0) for b in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]])
    fix0625_avg = np.mean([per_rho[ds]['0.625'].get(str(b), 0) for b in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]])
    o1_avg = np.mean([opt1[ds].get(str(b), {}).get('acc', 0) for b in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]])
    o2_avg = np.mean([opt2[ds].get(str(b), {}).get('acc', 0) for b in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]])
    o3_avg = np.mean([opt3[ds].get(str(b), {}).get('acc', 0) for b in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]])

    print(f'  Avg acc over BER [0, 0.30]:')
    print(f'    Oracle per-rho:   {oracle_avg:.2f}%')
    print(f'    Fixed rho=0.625:  {fix0625_avg:.2f}%')
    print(f'    Fixed rho=0.75:   {fix075_avg:.2f}%')
    print(f'    Option 1 (MLP):   {o1_avg:.2f}%   ({o1_avg - fix0625_avg:+.2f} vs fix-0.625)')
    print(f'    Option 2 (MoE):   {o2_avg:.2f}%   ({o2_avg - fix0625_avg:+.2f} vs fix-0.625)')
    print(f'    Option 3 (Ent):   {o3_avg:.2f}%   ({o3_avg - fix0625_avg:+.2f} vs fix-0.625)')
