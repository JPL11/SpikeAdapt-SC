#!/bin/bash
# run_uav_experiments.sh — Automated experiment pipeline
# Run with: nohup bash run_uav_experiments.sh > uav_experiments.log 2>&1 &
#
# This script runs all UAV/satellite-related experiments sequentially:
# 1. JPEG+Conv BER sweep (eval only, ~30 min)
# 2. AID aerial dataset: backbone + SpikeAdapt-SC × 3 channels + dynamic rate (~12-18h)
#
# Each step checks for existing checkpoints and skips if already done.
set -e

cd /home/jpli/SemCom
CONDA_BASE=$(conda info --base 2>/dev/null || echo "/home/jpli/anaconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate semcom

echo "=================================================="
echo "UAV/Satellite Enhancement Pipeline"
echo "Started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "=================================================="

# Step 1: JPEG+Conv BER Sweep
echo ""
echo "[1/2] JPEG+Conv BER Sweep..."
if [ -f "eval_results/jpeg_conv_sweep.json" ]; then
    echo "  ✓ Already done, skipping."
else
    python eval/eval_jpeg_sweep.py
    echo "  ✓ JPEG sweep complete."
fi

# Step 2: AID Aerial Dataset Training
echo ""
echo "[2/2] AID Aerial Dataset Training..."
if [ -f "snapshots_aid/aid_results.json" ]; then
    echo "  ✓ Already done, skipping."
else
    pip install huggingface_hub -q 2>/dev/null || true
    python train/train_aid.py
    echo "  ✓ AID training complete."
fi

echo ""
echo "=================================================="
echo "ALL EXPERIMENTS COMPLETE: $(date)"
echo "=================================================="
echo "Results:"
echo "  - JPEG sweep: eval_results/jpeg_conv_sweep.json"
echo "  - AID results: snapshots_aid/aid_results.json"
echo ""
echo "Next: update docs, README, and push to GitHub"
