#!/usr/bin/env python3
"""Run V6c on AID and RESISC45 with 10 seeds for statistical significance."""

import subprocess, sys, os, json, glob
import numpy as np

SEEDS = [42, 123, 456, 789, 1024, 2048, 3141, 4096, 5555, 7777]
DATASETS = ['aid', 'resisc45']
SCRIPT = 'train/train_aid_v6c.py'

def run_one(dataset, seed):
    print(f"\n{'='*60}")
    print(f"  V6c — {dataset.upper()} — Seed {seed}")
    print(f"{'='*60}")
    
    snap_dir = f"./snapshots_v6c_{dataset}_seed{seed}/"
    result_file = os.path.join(snap_dir, 'results.json')
    
    # Skip if already done
    if os.path.exists(result_file):
        with open(result_file) as f:
            r = json.load(f)
        print(f"  ✓ Already done: {r['best_s3']:.2f}%")
        return r
    
    cmd = [sys.executable, SCRIPT, '--dataset', dataset, '--seed', str(seed)]
    proc = subprocess.run(cmd, capture_output=False)
    
    if os.path.exists(result_file):
        with open(result_file) as f:
            return json.load(f)
    return None

def main():
    all_results = {}
    
    for dataset in DATASETS:
        print(f"\n\n{'#'*60}")
        print(f"  DATASET: {dataset.upper()}")
        print(f"{'#'*60}")
        
        results = []
        for seed in SEEDS:
            r = run_one(dataset, seed)
            if r:
                results.append(r)
        
        if results:
            cleans = [r['ber_results']['clean'] for r in results]
            ber30s = [r['ber_results'].get('ber_0.30', 0) for r in results]
            
            print(f"\n{'='*60}")
            print(f"  {dataset.upper()} — {len(results)} seeds")
            print(f"{'='*60}")
            print(f"  Clean: {np.mean(cleans):.2f}% ± {np.std(cleans):.2f}%")
            print(f"  BER=0.30: {np.mean(ber30s):.2f}% ± {np.std(ber30s):.2f}%")
            print(f"  Range: {min(cleans):.2f}% — {max(cleans):.2f}%")
            
            all_results[dataset] = {
                'seeds': SEEDS[:len(results)],
                'clean_mean': round(np.mean(cleans), 2),
                'clean_std': round(np.std(cleans), 2),
                'ber30_mean': round(np.mean(ber30s), 2),
                'ber30_std': round(np.std(ber30s), 2),
                'per_seed': results,
            }
    
    # Save summary
    os.makedirs('eval/seed_results', exist_ok=True)
    with open('eval/seed_results/v6c_10seed_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\n{'='*60}")
    print("FINAL SUMMARY — V6c 10-Seed")
    print(f"{'='*60}")
    for ds, r in all_results.items():
        print(f"  {ds.upper():12s}: {r['clean_mean']:.2f} ± {r['clean_std']:.2f}% (BER=0.30: {r['ber30_mean']:.2f} ± {r['ber30_std']:.2f}%)")
    print(f"\n✅ Saved eval/seed_results/v6c_10seed_summary.json")

if __name__ == '__main__':
    main()
