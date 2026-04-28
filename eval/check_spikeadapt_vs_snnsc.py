#!/usr/bin/env python3
"""Check: does SpikeAdapt-SC beat SNN-SC at every (rho, BER) combination?"""

import json

with open('eval/seed_results/per_rho_scorer_results.json') as f:
    per_rho = json.load(f)

# SNN-SC at rho=1.0 (from paper Table 2)
SNN_SC = {
    'aid': {
        0.0: 95.40, 0.05: 95.39, 0.10: 95.21, 0.15: 94.86,
        0.20: 94.29, 0.25: 92.22, 0.30: 82.42,
    },
    'resisc45': {
        0.0: 92.39, 0.05: 92.43, 0.10: 92.35, 0.15: 92.12,
        0.20: 91.51, 0.25: 89.60, 0.30: 80.55,
    },
}

BERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

for ds in ['aid', 'resisc45']:
    print(f'\n{"="*90}')
    print(f'  {ds.upper()}: SpikeAdapt-SC (per-rho trained) vs SNN-SC (rho=1.0)')
    print(f'  (+) = SpikeAdapt WINS, (-) = SNN-SC WINS')
    print(f'{"="*90}')

    rhos = sorted([float(r) for r in per_rho[ds].keys()])

    # Header
    header = f'{"BER":<8}{"SNN-SC":>10}'
    for rho in rhos:
        header += f'{"rho=" + str(rho):>11}'
    print(header)
    print('-' * len(header))

    total_cells = 0
    wins = 0
    losses = []

    for ber in BERS:
        snn = SNN_SC[ds].get(ber, None)
        if snn is None: continue
        row = f'{ber:<8.2f}{snn:>10.2f}'
        for rho in rhos:
            our = per_rho[ds][str(rho)].get(str(ber), 0)
            delta = our - snn
            total_cells += 1
            if delta > 0:
                wins += 1
                marker = '+'
            elif delta < 0:
                losses.append((rho, ber, our, snn, delta))
                marker = '-'
            else:
                marker = '='
            row += f'{our:>7.2f}{marker:>1} '
        print(row)

    print(f'\n  Win rate: {wins}/{total_cells} = {100*wins/total_cells:.1f}%')
    if losses:
        print(f'\n  LOSSES (where SNN-SC beats SpikeAdapt-SC):')
        for rho, ber, our, snn, d in losses:
            print(f'    rho={rho:.3f} BER={ber:.2f}: us={our:.2f}% vs SNN-SC={snn:.2f}% ({d:+.2f} pp)')
    else:
        print(f'\n  -> SpikeAdapt-SC DOMINATES SNN-SC at every cell.')

# Also compute Pareto domination: does SpikeAdapt beat SNN-SC with LESS bandwidth?
print(f'\n{"="*90}')
print(f'  PARETO ANALYSIS: SpikeAdapt-SC with LESS bandwidth (rho<1.0)')
print(f'  vs SNN-SC at FULL bandwidth (rho=1.0)')
print(f'{"="*90}')

for ds in ['aid', 'resisc45']:
    print(f'\n  {ds.upper()}:')
    for ber in BERS:
        snn = SNN_SC[ds][ber]
        # Find the smallest rho (most savings) where SpikeAdapt still beats SNN-SC
        rhos = sorted([float(r) for r in per_rho[ds].keys()])
        best_rho_wins = None
        for rho in rhos:
            our = per_rho[ds][str(rho)].get(str(ber), 0)
            if our >= snn:
                if best_rho_wins is None or rho < best_rho_wins[0]:
                    best_rho_wins = (rho, our)
        if best_rho_wins:
            rho_w, our_w = best_rho_wins
            savings = (1 - rho_w) * 100
            label = 'Clean' if ber == 0 else f'BER={ber:.2f}'
            print(f'    {label}: SpikeAdapt matches SNN-SC ({snn:.2f}%) at rho={rho_w:.3f} '
                  f'({savings:.0f}% savings) with {our_w:.2f}%  [+{our_w-snn:.2f} pp]')
        else:
            label = 'Clean' if ber == 0 else f'BER={ber:.2f}'
            print(f'    {label}: no rho beats SNN-SC (SNN-SC = {snn:.2f}%)')
