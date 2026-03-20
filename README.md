# SpikeAdapt-SC

**Fine-Grained Spiking Semantic Communication with Spatial Masking for Aerial Scene Classification**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> **SpikeAdapt-SC** encodes semantic features as binary spike trains on a **native 14×14 feature grid**, learns per-image spatial masking, and achieves robust aerial scene classification under noisy air-to-ground channels. Using **MPBN** (Membrane Potential Batch Normalization), the system achieves **42× energy savings** via SynOps while maintaining **96.55% accuracy** on AID-30 and **>95% through BER=0.30**. Validated on AID (10K images, 30 classes) and RESISC45 (31.5K images, 45 classes).

---

## Key Results

### Three Core Claims

1. **Fine-grained 14×14 spatial masking is decisive** — learned masks exceed random by **+45 pp** at ρ=0.50 BER=0.30 (93.35% vs 48.00%)
2. **Binary spike encoding is inherently noise-robust** — V5C degrades <1 pp from clean to BER=0.30 (vs CNN collapse from 92.5%→67%)
3. **MPBN provides 42× energy savings** — firing rate 0.148 vs 0.266 baseline, each spike carries more semantic weight

### Main Results (BSC Channel, AID)

| Method | Clean | BER=0.15 | BER=0.30 | Rate | Energy × |
|--------|-------|----------|----------|------|----------|
| **V5C-MPBN** (ρ=0.75) | **96.55%** | 96.45% | **95.75%** | 75% | **42.0×** |
| V4-A-SNN (ρ=0.75) | 96.45% | 96.55% | 95.85% | 75% | 23.2× |
| V2-Baseline (ρ=0.75) | 96.35% | 95.85% | 92.90% | 75% | 23.4× |
| SNN (no mask, ρ=1.0) | 95.70% | 95.35% | 92.20% | 100% | 23.4× |
| CNN-Uni 8-bit | 92.50% | 92.40% | 67.25% | 100% | 1.0× |

### ρ Sweep: Bandwidth–Accuracy Pareto (BER=0.30)

| ρ | BW Save | V2 | V4-A | V5C |
|---|---------|-----|------|------|
| 0.25 | 75% | 29.90% | 51.15% | **82.30%** |
| 0.50 | 50% | 85.55% | 94.00% | **94.55%** |
| 0.625 | 37.5% | 91.05% | 94.75% | **95.40%** |
| 0.75 | 25% | 92.85% | **95.55%** | 95.45% |
| 1.00 | 0% | 93.35% | **96.05%** | 95.40% |

> **V5C at ρ=0.625 (37.5% bandwidth savings) exceeds V2 at full rate** — 95.40% vs 93.35% at BER=0.30.

### Mask Comparison (V4-A, AID, ρ=0.50)

| BER | Learned | Random | Uniform | Δ_rand |
|-----|---------|--------|---------|--------|
| 0.00 | 95.60% | 92.05% | 92.10% | +3.55 |
| 0.15 | 95.40% | 81.67% | 78.70% | +13.73 |
| 0.30 | **93.35%** | 48.00% | 44.50% | **+45.35** |

### SynOps / Energy Analysis

| Version | FR | ρ | SynOps (M) | Energy × |
|---------|-----|---|------------|----------|
| V2 (IF) | 0.266 | 0.75 | 321.1 | 23.4× |
| V4-A (LIF+BNTT) | 0.268 | 0.75 | 323.5 | 23.2× |
| **V5C (MPBN)** | **0.148** | 0.75 | **178.7** | **42.0×** |
| V5C (MPBN) | 0.148 | 0.50 | 140.1 | 53.5× |

*Computed from measured firing rates using the Horowitz energy model (MAC: 4.6 pJ, SynOp: 0.9 pJ).*

---

## Repository Structure

```
SpikeAdapt-SC/
├── train/                               # Training scripts
│   ├── train_aid_v2.py                  #   V2: 14×14 native grid, C2=36
│   ├── train_aid_v4.py                  #   V4-A: LIF + BNTT + learnable slope
│   ├── train_aid_v5.py                  #   V5C: + MPBN (final model base)
│   ├── train_aid_v6.py                  #   V6: membrane shortcut (experimental)
│   ├── train_ucm.py                     #   UCM dataset training
│   ├── train_temporal_v3.py             #   Multi-exit temporal training
│   └── run_final_pipeline.py            #   Master pipeline: V5C-NA on AID + RESISC45
│
├── models/                              # Model components
│   ├── backbone.py                      #   ResNet50 front/back
│   ├── snn_modules.py                   #   SNN layers (LIF, BNTT, MPBN)
│   └── noise_aware_scorer.py            #   Noise-aware importance scorer
│
├── eval/                                # Evaluation & analysis
│   ├── compute_synops.py                #   SynOps/energy analysis
│   ├── ablation_aid.py                  #   ρ sweep + mask comparison (AID)
│   ├── run_ucm_extras.py                #   UCM ablation experiments
│   ├── slope_lr_sweep.py                #   Surrogate gradient slope analysis
│   ├── sanity_check_full.py             #   Full-dataset masking analysis
│   └── multi_trace_eval.py              #   Multi-trajectory evaluation
│
├── paper/                               # Paper assets
│   ├── main.tex                         #   Manuscript source
│   └── figures/                         #   Publication-quality figures
│
├── data/
│   ├── AID/                             #   AID dataset (30 classes, 10K images)
│   ├── NWPU-RESISC45/                   #   RESISC45 (45 classes, 31.5K images)
│   └── UCMerced_LandUse/                #   UCM dataset (21 classes, 2.1K images)
│
├── snapshots_*/                         # Model checkpoints (per version/seed)
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/JPL11/SpikeAdapt-SC.git
cd SpikeAdapt-SC

# Install dependencies
pip install -r requirements.txt

# Train final model (V5C with noise-aware scorer)
python train/run_final_pipeline.py --stage backbone_aid
python train/run_final_pipeline.py --stage v5c_aid

# Train on RESISC45
python train/run_final_pipeline.py --stage backbone_resisc
python train/run_final_pipeline.py --stage v5c_resisc

# Ablation: ρ sweep + mask comparison
python eval/ablation_aid.py

# SynOps analysis
python eval/compute_synops.py
```

---

## Training Pipeline

| Stage | Description | Epochs | LR |
|-------|-------------|--------|------|
| **S1** | ResNet50 backbone fine-tuning | 50 | 0.01 |
| **S2** | SNN encoder/decoder with MPBN (backbone frozen) | 60 | 1e-4 |
| **S3** | Joint fine-tuning with noise-aware scorer + diversity loss | 40 | 1e-5 |

---

## Datasets

| Dataset | Classes | Images | Resolution | Split |
|---------|---------|--------|------------|-------|
| **AID** | 30 | 10,000 | 600×600 | 50/50 |
| **RESISC45** | 45 | 31,500 | 256×256 | 20/80 |
| UCM | 21 | 2,100 | 256×256 | 80/20 |

---

## Citation

```bibtex
@inproceedings{li2025spikeadaptsc,
  title={Fine-Grained Spiking Semantic Communication with Spatial Masking for Aerial Scene Classification},
  author={Li, JP},
  booktitle={IEEE GLOBECOM},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
