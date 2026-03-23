#!/bin/bash
set -e
cd /home/jpli/SemCom
source ~/miniconda/etc/profile.d/conda.sh
conda activate semcom

echo "============================================================"
echo "V4 Experiment Runner"
echo "============================================================"

for EXP in v4A v4B v4C v4D; do
    echo ""
    echo "============================================================"
    echo "Running experiment: $EXP"
    echo "============================================================"
    PYTHONUNBUFFERED=1 python train/train_aid_v4.py --exp $EXP --seed 42 --epochs_s3 40
    echo ""
    echo "Experiment $EXP complete."
done

echo ""
echo "============================================================"
echo "ALL V4 EXPERIMENTS COMPLETE"
echo "============================================================"

# Summary
echo ""
echo "Results summary:"
for d in snapshots_v4_*_seed42; do
    if [ -f "$d/results.json" ]; then
        exp=$(python3 -c "import json; r=json.load(open('$d/results.json')); print(f'{r[\"experiment\"]}: {r[\"best_accuracy\"]:.2f}% (Δ={r[\"delta_v2\"]:+.2f})')")
        echo "  $exp"
    fi
done
