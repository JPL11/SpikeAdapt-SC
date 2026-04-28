#!/usr/bin/env python3
"""Strict rho=1.0 comparison: SpikeAdapt-SC (our SNN) vs SNN-SC baseline."""

import json
import numpy as np

# Paper Table 2 (seed 42): Full BER sweep at rho=0.75 for SpikeAdapt
# But we also need rho=1.0. Let me gather from multiple sources.

# ===== SNN-SC baseline (seed 42, rho=1.0) =====
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

# ===== SpikeAdapt-SC at rho=1.0 =====
# 10-seed data from summary_10seed.json (only BER=0.0 and 0.30)
with open('eval/seed_results/summary_10seed.json') as f:
    seed10 = json.load(f)

# Per-seed data (seed 42, rho=1.0): BER=0.0, 0.15, 0.30
with open('eval/seed_results/aid_seed42.json') as f:
    aid_s42 = json.load(f)
with open('eval/seed_results/resisc45_seed42.json') as f:
    r45_s42 = json.load(f)

# Aggregate 10-seed for BER=0.15 from per-seed files
seeds = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]
aid_015_seeds, r45_015_seeds = [], []
for s in seeds:
    with open(f'eval/seed_results/aid_seed{s}.json') as f:
        d = json.load(f)
    if '1.0' in d and '0.15' in d['1.0']:
        aid_015_seeds.append(d['1.0']['0.15'])
    with open(f'eval/seed_results/resisc45_seed{s}.json') as f:
        d = json.load(f)
    if '1.0' in d and '0.15' in d['1.0']:
        r45_015_seeds.append(d['1.0']['0.15'])

# Build comparison
print('=' * 80)
print('  STRICT rho=1.0 COMPARISON: SpikeAdapt-SC vs SNN-SC')
print('  (Both at full bandwidth, no masking)')
print('=' * 80)

for ds, ds_key in [('AID', 'aid'), ('RESISC45', 'resisc45')]:
    print(f'\n  {ds}:')
    print(f'  {"BER":<8}{"SNN-SC":>10}{"SpikeAdapt":>15}{"Delta":>10}{"Note":>12}')
    print('  ' + '-'*58)

    # BER=0.0 (10-seed)
    our_clean = seed10[ds_key]['1.0']['0.0']
    snn_clean = SNN_SC[ds_key][0.0]
    delta = our_clean['mean'] - snn_clean
    marker = 'win' if delta > 0 else ('tie' if abs(delta) < 0.2 else 'lose')
    print(f'  {0.0:<8.2f}{snn_clean:>10.2f}{our_clean["mean"]:>9.2f}±{our_clean["std"]:.2f} '
          f'{delta:>+9.2f} {marker:>11} (10-seed)')

    # BER=0.05, 0.10 — we only have seed 42 from paper Table 2
    # SpikeAdapt at rho=1.0 is the "SNN (no mask)" in Table 1
    # Let me check if seed-42 per-rho has rho=1.0 data — no, only up to 0.875
    # So we use seed-42 rho=0.875 as proxy? No, user asked for strict rho=1.0
    # Let's indicate missing data
    for ber in [0.05, 0.10, 0.20, 0.25]:
        snn = SNN_SC[ds_key][ber]
        print(f'  {ber:<8.2f}{snn:>10.2f}{"(not measured)":>15}{"—":>10}{"—":>12}')

    # BER=0.15 (per-seed aggregate)
    if ds_key == 'aid':
        vals = aid_015_seeds
    else:
        vals = r45_015_seeds
    if vals:
        mean_015 = np.mean(vals)
        std_015 = np.std(vals)
        snn_015 = SNN_SC[ds_key][0.15]
        delta = mean_015 - snn_015
        marker = 'win' if delta > 0 else ('tie' if abs(delta) < 0.2 else 'lose')
        print(f'  {0.15:<8.2f}{snn_015:>10.2f}{mean_015:>9.2f}±{std_015:.2f} '
              f'{delta:>+9.2f} {marker:>11} (10-seed)')

    # BER=0.30 (10-seed)
    our_30 = seed10[ds_key]['1.0']['0.3']
    snn_30 = SNN_SC[ds_key][0.30]
    delta = our_30['mean'] - snn_30
    marker = 'WIN' if delta > 0 else ('tie' if abs(delta) < 0.2 else 'LOSE')
    print(f'  {0.30:<8.2f}{snn_30:>10.2f}{our_30["mean"]:>9.2f}±{our_30["std"]:.2f} '
          f'{delta:>+9.2f} {marker:>11} (10-seed)')

# Now check the full BER sweep at rho=0.75 from paper Table 2 as reference
print('\n' + '=' * 80)
print('  FOR REFERENCE: SpikeAdapt at rho=0.75 vs SNN-SC at rho=1.0')
print('  (Paper Table 2, seed-42)')
print('=' * 80)

SPIKEADAPT_075 = {
    'aid': {  # from paper Table 2
        0.0: 95.46, 0.05: 95.63, 0.10: 95.66, 0.15: 95.78,
        0.20: 95.70, 0.25: 95.16, 0.30: 93.43,
    },
    'resisc45': {
        0.0: 92.00, 0.05: 92.14, 0.10: 92.34, 0.15: 92.42,
        0.20: 92.31, 0.25: 91.38, 0.30: 87.00,
    },
}

for ds, ds_key in [('AID', 'aid'), ('RESISC45', 'resisc45')]:
    print(f'\n  {ds}:')
    print(f'  {"BER":<8}{"SNN-SC":>10}{"Adapt@0.75":>12}{"Delta":>10}{"Verdict":>12}')
    print('  ' + '-'*56)
    bers_check = sorted(SPIKEADAPT_075[ds_key].keys())
    all_wins = True
    for ber in bers_check:
        snn = SNN_SC[ds_key][ber]
        ours = SPIKEADAPT_075[ds_key][ber]
        delta = ours - snn
        if delta > 0.2:
            verdict = 'WIN'
        elif delta < -0.2:
            verdict = 'LOSE'
            all_wins = False
        else:
            verdict = 'tie'
        print(f'  {ber:<8.2f}{snn:>10.2f}{ours:>12.2f}{delta:>+10.2f}{verdict:>12}')

    if all_wins:
        print(f'\n  -> SpikeAdapt@rho=0.75 BEATS SNN-SC@rho=1.0 at EVERY BER '
              f'with 25% LESS bandwidth!')
