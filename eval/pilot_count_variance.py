#!/usr/bin/env python3
"""Pilot-count dispersion check (panel-3 follow-up): does the 256-bit pilot's
error count match the Binomial(256, p) model the closed loop assumes?

Reuses run_bpsk from waveform_channel_validation.py (block-fading Rician BPSK,
256-symbol coherence blocks, K_dB = 9.44 as modeled). A pilot CONFINED to one
coherence block shares one fading gain -> over-dispersed vs binomial; pilot
bits SPREAD across k coherence blocks approach binomial as k grows. The frame
spans ~66 coherence times, so frame-spread pilots (k ~ 64) are realizable.

Usage: python eval/pilot_count_variance.py
Output: eval/seed_results/pilot_count_variance.json
"""
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location(
    'wcv', os.path.join(ROOT, 'eval/waveform_channel_validation.py'))
wcv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wcv)

PILOT, BLK, K_DB = 256, 256, 9.44
SNRS = [-6, -4, -2]
SPREADS = [1, 4, 16, 64]          # coherence blocks the pilot is spread over


def main():
    rng = np.random.default_rng(7)
    out = {'params': dict(pilot=PILOT, blk_len=BLK, k_db=K_DB,
                          n_blocks=40960, spreads=SPREADS)}
    for snr in SNRS:
        err, pbar = wcv.run_bpsk(snr, K_DB, 40960, BLK, rng)
        v_binom = PILOT * pbar * (1 - pbar)
        row = {'pbar': round(pbar, 4)}
        for k in SPREADS:
            per = PILOT // k
            cnt = err.reshape(-1, k, BLK)[:, :, :per].reshape(-1, k * per).sum(1)
            row[f'var_ratio_spread{k}'] = round(float(cnt.var() / v_binom), 3)
        out[f'snr{snr}'] = row
        print(f"SNR {snr:+d} dB (pbar {pbar:.3f}): " + '  '.join(
            f"k={k}: {row[f'var_ratio_spread{k}']}" for k in SPREADS))
    path = os.path.join(ROOT, 'eval/seed_results/pilot_count_variance.json')
    json.dump(out, open(path, 'w'), indent=1)
    print('saved', path)


if __name__ == '__main__':
    main()
