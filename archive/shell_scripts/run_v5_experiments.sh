#!/bin/bash
set -e
cd /home/jpli/SemCom
source ~/miniconda/etc/profile.d/conda.sh
conda activate semcom

echo "============================================================"
echo "V5 Experiment Runner"
echo "============================================================"

for EXP in v5A v5B v5C v5D v5E; do
    echo ""
    echo "============================================================"
    echo "Running experiment: $EXP"
    echo "============================================================"
    PYTHONUNBUFFERED=1 python train/train_aid_v5.py --exp $EXP --seed 42 --epochs_s3 40
    echo ""
    echo "Experiment $EXP complete."
done

echo ""
echo "============================================================"
echo "ALL V5 EXPERIMENTS COMPLETE"
echo "============================================================"

echo ""
echo "Results summary:"
for d in snapshots_v5_*_seed42; do
    if [ -f "$d/results.json" ]; then
        exp=$(python3 -c "import json; r=json.load(open('$d/results.json')); print(f'{r[\"experiment\"]}: {r[\"best_accuracy\"]:.2f}% (Δ orig={r[\"delta_orig\"]:+.2f}, Δ v4A={r[\"delta_v4a\"]:+.2f})')")
        echo "  $exp"
    fi
done
