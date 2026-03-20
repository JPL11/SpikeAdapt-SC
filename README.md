# SpikeAdapt-SC

**Fine-Grained Spiking Semantic Communication with Spatial Masking for Aerial Scene Classification**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> **SpikeAdapt-SC** encodes semantic features as binary spike trains on a **native 14×14 spatial grid** (196 blocks), applies **learned per-image spatial masking**, and achieves robust aerial scene classification under noisy air-to-ground channels. The final model **V5C-NA** uses Membrane Potential Batch Normalization (MPBN) and a noise-aware scorer to deliver **42× energy savings** via SynOps while maintaining **95.42% accuracy on AID** and **92.01% on RESISC45**. At BER=0.30, masking at ρ=0.625 (37.5% bandwidth savings) **exceeds full-rate accuracy** on both datasets.

---

## Key Results

### Three Core Claims

1. **Fine-grained 14×14 spatial masking improves accuracy under noise** — at ρ=0.625 and BER=0.30, masking exceeds full-rate accuracy on both AID (+1.14 pp) and RESISC45 (+1.98 pp). Learned masks beat random by +1.34 pp on AID at ρ=0.50/BER=0.30.

2. **Binary spike encoding provides inherent noise robustness** — V5C-NA degrades <1 pp from clean to BER=0.15 on both datasets, while CNN 8-bit quantization collapses from 92.5% to 67%.

3. **MPBN provides 42× energy savings** — firing rate 0.167 (vs. 0.266 baseline) yields 42× fewer SynOps at ρ=0.75 using the Horowitz energy model.

### Main Results (BSC Channel, AID 50/50 Split)

| Method | Clean | BER=0.15 | BER=0.30 | Rate |
|--------|-------|----------|----------|------|
| **V5C-NA** (ρ=0.75) | **95.42%** | **95.78%** | **93.36%** | 75% |
| V5C-NA (ρ=0.625) | 95.40% | 95.62% | 93.50% | 62.5% |
| SNN (no mask, ρ=1.0) | 95.20% | 95.64% | 92.36% | 100% |
| CNN-Uni 8-bit | 92.50% | 92.40% | 67.25% | 100% |

> **Key**: ρ=0.625 at BER=0.30 (93.50%) *exceeds* ρ=1.0 (92.36%) — masking actively improves noise robustness.

### Cross-Dataset Validation

| Dataset | Split | Backbone | V5C-NA Clean | BER=0.30 |
|---------|-------|----------|-------------|----------|
| **AID** | 50/50 (5K/5K) | 96.04% | **95.42%** | **93.42%** |
| **RESISC45** | 20/80 (6.3K/25.2K) | 93.44% | **92.01%** | **87.15%** |

### ρ Sweep: Bandwidth–Accuracy Pareto (BER=0.30)

| ρ | BW Save | AID | RESISC45 |
|---|---------|-----|----------|
| 0.10 | 90% | 62.68% | 53.98% |
| 0.25 | 75% | 88.12% | 78.74% |
| 0.50 | 50% | 93.26% | 86.58% |
| **0.625** | **37.5%** | **93.50%** | **87.29%** |
| 0.75 | 25% | 93.36% | 87.08% |
| 1.00 | 0% | 92.36% | 85.31% |

> **ρ=0.625 beats ρ=1.0 on both datasets**: AID +1.14 pp, RESISC45 +1.98 pp.

### Mask Comparison (V5C-NA, AID 50/50)

| ρ | BER | Learned | Random (3-draw) | Uniform | Δ_rand |
|---|-----|---------|-----------------|---------|--------|
| 0.50 | 0.00 | 95.02% | 95.31±0.04% | 95.56% | −0.29 |
| 0.50 | 0.30 | **93.32%** | 91.98±0.07% | 92.80% | **+1.34** |
| 0.75 | 0.00 | 95.42% | 95.29±0.02% | 94.70% | +0.13 |
| 0.75 | 0.30 | **93.80%** | 93.29±0.05% | 91.90% | **+0.51** |

> Learned mask advantage **grows under noise**: modest at clean, significant at BER=0.30.

### SynOps / Energy Analysis

| Version | FR | ρ | SynOps (M) | Energy × |
|---------|-----|---|------------|----------|
| V2 (IF baseline) | 0.266 | 0.75 | 321.1 | 23.4× |
| **V5C-NA (MPBN)** | **0.167** | 0.75 | **178.7** | **42.0×** |
| V5C-NA (MPBN) | 0.167 | 0.50 | 140.1 | 53.5× |

*Computed from measured firing rates using the Horowitz energy model (MAC: 4.6 pJ, SynOp: 0.9 pJ).*

### 5-Seed Reproducibility (Proper S3 Retraining)

*Running — results will be updated when complete.*

---

## Architecture

```
Input Image (3×224×224)
    │
    ▼
ResNet-50 Front (L1–L3, frozen) → Feature Map F ∈ ℝ^{1024×14×14}
    │                                    │
    │                             Noise-Aware Scorer
    │                          (BER-dependent reweighting)
    │                                    │
    ▼                                    ▼
SNN Encoder (T=8)              Importance Scores I ∈ (0,1)^{14×14}
  Conv 1024→256, BN+IF                  │
  Conv 256→36, BN+IF                    ▼
  MPBN (FR=0.167)              Learned Block Mask M ∈ {0,1}^{14×14}
    │                                    │
    ▼                                    │
Spikes S₂ ∈ {0,1}^{36×14×14} ──⊙ M──▶ Masked Spikes
                                         │
                                    BSC Channel
                                   (BER flipping)
                                         │
                                         ▼
                                  SNN Decoder (T=8)
                                  Conv 36→256, BN+IF
                                  Conv 256→1024, BN+IHF
                                         │
                                         ▼
                              Spike-to-Feature Converter
                              (Gated Linear over T steps)
                                         │
                                         ▼
                              ResNet-50 Back (L4+FC)
                                         │
                                         ▼
                               Classification ŷ ∈ ℝ^C
```

**Bits transmitted** at ρ=0.75: `T × C₂ × H × W × ρ = 8 × 36 × 14 × 14 × 0.75 = 42,336 bits`

---

## Repository Structure

```
SpikeAdapt-SC/
├── train/                               # Training scripts
│   ├── train_aid_v2.py                  #   V2: 14×14 native grid, C2=36
│   ├── train_aid_v4.py                  #   V4-A: LIF + BNTT + learnable slope
│   ├── train_aid_v5.py                  #   V5C: + MPBN (final model base)
│   └── run_final_pipeline.py            #   Master pipeline: V5C-NA on AID + RESISC45
│
├── models/                              # Model components
│   └── noise_aware_scorer.py            #   Noise-aware importance scorer
│
├── eval/                                # Evaluation & analysis
│   ├── run_ablations_final.py           #   ρ sweep + mask comparison (both datasets)
│   ├── gen_paper_figures.py             #   IEEE conference-grade figure generation
│   ├── 5seed_proper.py                  #   Proper 5-seed S3 retraining
│   ├── pareto_and_seeds.py              #   Pareto figure generation
│   └── compute_synops.py               #   SynOps/energy analysis
│
├── paper/                               # Paper assets
│   ├── main.tex                         #   Manuscript source
│   └── figures/                         #   Publication-quality figures (PDF+PNG)
│
├── data/
│   ├── AID/                             #   AID dataset (30 classes, 10K images)
│   └── NWPU-RESISC45/                   #   RESISC45 (45 classes, 31.5K images)
│
├── snapshots_*/                         # Model checkpoints (per dataset/seed)
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

# Train V5C-NA on AID (50/50 split)
python train/run_final_pipeline.py --stage backbone_aid
python train/run_final_pipeline.py --stage v5c_aid

# Train V5C-NA on RESISC45 (20/80 split)
python train/run_final_pipeline.py --stage backbone_resisc
python train/run_final_pipeline.py --stage v5c_resisc

# Run ablations (ρ sweep + mask comparison on both datasets)
python eval/run_ablations_final.py

# Generate IEEE conference-grade figures
python eval/gen_paper_figures.py

# SynOps analysis
python eval/compute_synops.py
```

---

## Training Pipeline

| Stage | Description | Epochs | LR |
|-------|-------------|--------|------|
| **S1** | ResNet-50 backbone fine-tuning (ImageNet → aerial domain) | 50 | 0.01 |
| **S2** | SNN encoder/decoder with MPBN (backbone frozen) | 60 | 1e-4 |
| **S3** | Joint fine-tuning with noise-aware scorer + diversity loss (λ_div=0.05) | 40 | 1e-5 |

**Noise curriculum**: Each batch samples BER uniformly, with 50% probability biased to high-noise [0.15, 0.40].

---

## Datasets

| Dataset | Classes | Images | Resolution | Standard Split | Train | Test |
|---------|---------|--------|------------|----------------|-------|------|
| **AID** [Xia et al. 2017] | 30 | 10,000 | 600×600 | 50/50 | 5,000 | 5,000 |
| **RESISC45** [Cheng et al. 2017] | 45 | 31,500 | 256×256 | 20/80 | 6,300 | 25,200 |

Both use their **standard benchmark splits** for comparability with prior work.

---

## Citation

```bibtex
@inproceedings{li2025spikeadaptsc,
  title={Fine-Grained Spiking Semantic Communication with Spatial Masking
         for Aerial Scene Classification},
  author={Li, JP},
  booktitle={IEEE GLOBECOM},
  year={2025}
}
```

---

## References

- **AID**: G.-S. Xia et al., "AID: A benchmark data set for performance evaluation of aerial scene classification," *IEEE TGRS*, 2017.
- **RESISC45**: G. Cheng et al., "Remote sensing image scene classification: Benchmark and state of the art," *Proc. IEEE*, 2017.
- **SNN-SC**: M. Wang et al., "SNN-SC: A spiking semantic communication framework," *IEEE TCCN*, 2024.
- **Horowitz**: M. Horowitz, "Computing's energy problem," *IEEE ISSCC*, 2014.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
