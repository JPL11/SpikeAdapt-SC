#!/usr/bin/env python3
"""Radar decomposition head-to-head: SNN vs FAIR 1-bit vs confounded 1-bit.

Tests whether the +6-10 Pbar / +0.14 recall gap the array paper (contribution
iii) attributes to spiking survives an encoder-matched, truncation-trained 1-bit
baseline. 3-seed means (42/123/456), RADIal full protocol, matched ~2.32 Mbit
payload (rho=0.75, T'=8).

Usage: python eval/compare_radar_baseline.py
"""
import json
import os

import numpy as np

SR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  'eval/seed_results')
BSHOW = ['0.0', '0.15', '0.3']


def load3(paths, ber):
    P, R = [], []
    for p in paths:
        d = json.load(open(os.path.join(SR, p)))
        cell = d.get(f'ber{ber}')
        if cell and cell.get('mAP') is not None:
            P.append(cell['mAP']); R.append(cell['mAR'])
    return P, R


def ms(xs):
    return f'{np.mean(xs)*100:5.1f}±{np.std(xs)*100:.1f}' if xs else '  n/a'


METHODS = {
    'SNN spikes-only': ['radial_snn_full_sweep_so.json',
                        'radial_snn_full_sweep_s123_so.json',
                        'radial_snn_full_sweep_s456_so.json'],
    'Fair 1-bit (NEW)': ['radial_adaptive1bit_sweep_s42.json',
                         'radial_adaptive1bit_sweep_s123.json',
                         'radial_adaptive1bit_sweep_s456.json'],
    'Confounded 1-bit': ['radial_quant1_retrained_sweep.json',
                         'radial_quant1_retrained_sweep_s123.json',
                         'radial_quant1_retrained_sweep_s456.json'],
}


def main():
    print('=' * 72)
    print('RADAR DECOMPOSITION (3-seed mean, RADIal full protocol, '
          'rho=0.75 T\'=8)')
    print('=' * 72)
    print(f"{'method':20}" + ''.join(f'  BER{b:>4} Pbar/Rbar' for b in BSHOW))
    snn = {}
    for name, paths in METHODS.items():
        row = f'{name:20}'
        for b in BSHOW:
            P, R = load3(paths, b)
            row += f'   {ms(P)}/{ms(R)}'
            if name == 'SNN spikes-only':
                snn[b] = (np.mean(P) if P else None, np.mean(R) if R else None)
        print(row)
    # gap vs SNN at BER0.30
    print('-' * 72)
    fair = METHODS['Fair 1-bit (NEW)']
    for b in ['0.3']:
        P, R = load3(fair, b)
        if P and snn.get(b) and snn[b][0] is not None:
            dP = (snn[b][0] - np.mean(P)) * 100
            dR = (snn[b][1] - np.mean(R)) * 100
            print(f'SNN - Fair1bit @ BER{b}:  dPbar={dP:+.1f}pp  dRbar={dR:+.1f}pp')
            print('  (paper claims spiking adds +6-10 Pbar / +14 recall vs the '
                  'CONFOUNDED 1-bit;')
            print('   if these deltas are ~0, contribution (iii) must go '
                  'code-agnostic.)')


if __name__ == '__main__':
    main()
