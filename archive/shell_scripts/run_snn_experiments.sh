#!/bin/bash
# Run all SNN-native experiments sequentially
set -e
cd /home/jpli/SemCom
source ~/miniconda/etc/profile.d/conda.sh
conda activate semcom

echo "============================================================"
echo "SNN-Native Experiment Runner"
echo "============================================================"

for EXP in BASELINE A B C D E F ALL; do
    echo ""
    echo "============================================================"
    echo "Running experiment: $EXP"
    echo "============================================================"
    PYTHONUNBUFFERED=1 python train/train_aid_v2_snn_native.py --exp $EXP --seed 42
    echo ""
    echo "Experiment $EXP complete."
done

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "============================================================"

# Summary
echo ""
echo "Results summary:"
for d in snapshots_snn_native_*_seed42; do
    if [ -f "$d/results.json" ]; then
        exp=$(python3 -c "import json; r=json.load(open('$d/results.json')); print(f'{r[\"experiment\"]}: {r[\"best_accuracy\"]:.2f}% (Δ={r[\"delta\"]:+.2f})')")
        echo "  $exp"
    fi
done
