# SpikeAdapt-SC

**Fine-Grained Spiking Semantic Communication with Spatial Masking for Aerial Scene Classification**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> **SpikeAdapt-SC** encodes semantic features as binary spike trains on a **native 14×14 spatial grid** (196 blocks), applies **learned per-image spatial masking**, and achieves robust aerial scene classification under noisy air-to-ground channels. The encoder uses Membrane Potential Batch Normalization (MPBN) and a noise-aware scorer to deliver **37× energy savings** via SynOps. Over 5 seeds (full pipeline retraining), SpikeAdapt-SC achieves **94.86 ± 0.63% on AID** and **92.52 ± 0.17% on RESISC45** at ρ=0.75 (25% bandwidth savings), with only ~2 pp degradation at BER=0.30. On RESISC45, masking at ρ=0.75 significantly improves BER=0.30 accuracy over full-rate (Δ=+2.0 pp, p=0.038).

---

## Key Results

### 5-Seed Results (Seeds: 42, 123, 456, 789, 1024)

Full pipeline retraining (backbone + S2 + S3) per seed. **These are the primary reported numbers.**

| Condition | AID Mean ± Std | RESISC45 Mean ± Std |
|-----------|----------------|---------------------|
| **ρ=1.0, Clean** | **95.48 ± 0.30%** | 92.49 ± 0.16% |
| **ρ=0.75, Clean** | 94.86 ± 0.63% | **92.52 ± 0.17%** |
| ρ=0.625, Clean | 93.79 ± 1.24% | 92.35 ± 0.21% |
| ρ=1.0, BER=0.30 | 92.87 ± 1.19% | 83.55 ± 6.88% |
| **ρ=0.75, BER=0.30** | 92.66 ± 0.95% | **85.55 ± 5.53%** |
| ρ=0.625, BER=0.30 | 90.86 ± 1.93% | **86.23 ± 4.79%** |

> **Key findings**: (1) ρ=1.0 gives the highest clean accuracy (as expected — keeping all blocks). (2) On **RESISC45**, masking at ρ=0.75 *improves* BER=0.30 accuracy over full-rate by **+2.0 pp** (p=0.038, significant). (3) On AID, masking is bandwidth-neutral (Δ=−0.20 pp, p=0.69 — saves 25% bandwidth with no accuracy loss under noise). (4) The BER=0.30 variance on RESISC45 is high (±5–7 pp) because the 45-class task amplifies scorer differences across seeds.

### Seed-42 Illustrative Results (BSC Channel)

| Method | AID Clean | AID BER=0.30 | RESISC45 Clean | RESISC45 BER=0.30 | Rate |
|--------|-----------|-------------|----------------|-------------------|------|
| SpikeAdapt-SC (ρ=0.75) | 95.42% | 93.18% | 92.61% | 84.87% | 75% |
| SpikeAdapt-SC (ρ=0.625) | 95.40% | 93.38% | 92.58% | 86.48% | 62.5% |
| SNN (no mask, ρ=1.0) | 95.20% | 92.60% | 92.50% | 81.62% | 100% |
| CNN-1bit (STE, T=1) | 95.32% | 87.56% | 91.48% | 85.19% | 100% |
| CNN-Uni (8-bit) | 91.78% | 52.80% | 78.32% | 49.32% | 100% |

> **Note**: Seed-42 is shown as an illustrative operating point. Some seed-42 advantages (e.g., ρ=0.625 on AID) do not generalize across seeds — see the 5-seed table above for robust conclusions.

### Noise-Aware Ablation (Seed-42, ρ=0.75)

| Variant | BER | Div | AID Clean | AID BER=0.30 | R45 Clean | R45 BER=0.30 |
|---------|-----|-----|-----------|-------------|-----------|-------------|
| **Full** | ✓ | ✓ | 95.42 | **93.40** | 92.01 | **87.10** |
| No BER branch | ✗ | ✓ | 95.94 | 72.54 | 92.27 | 41.52 |
| No diversity loss | ✓ | ✗ | 95.94 | 72.88 | 92.35 | 42.83 |
| Neither | ✗ | ✗ | 95.96 | 72.48 | 92.18 | 39.46 |

> Both components are essential. Removing either collapses BER=0.30 accuracy by ~20 pp on AID and ~45 pp on RESISC45, while clean accuracy is unaffected.

### Mask Comparison (AID, 50 Independent Random Draws)

| ρ | BER | Learned | Random (50-draw) | Uniform | Δ_rand |
|---|-----|---------|------------------|---------|--------|
| 0.50 | 0.00 | 95.02% | 95.28 ± 0.12% | 90.58% | −0.26 |
| 0.50 | 0.30 | **93.44%** | 91.95 ± 0.15% | 85.54% | **+1.49** |
| 0.75 | 0.00 | 95.42% | 95.35 ± 0.10% | 94.70% | +0.07 |
| 0.75 | 0.30 | 93.26% | 93.25 ± 0.13% | 91.78% | +0.01 |

> The learned mask matters most under harsher compression: at ρ=0.50/BER=0.30, learned masks outperform random by **+1.49 pp**. At ρ=0.75, the advantage is minimal (+0.01 pp), consistent with the high spatial redundancy of the 14×14 feature grid at mild compression.

### Cross-Channel Comparison (Matched Equivalent BER)

| Equiv BER | BSC (AID) | AWGN (AID) | Ray. (AID) | BSC (R45) | AWGN (R45) | Ray. (R45) |
|-----------|-----------|------------|------------|-----------|------------|------------|
| 0.05 | 95.64% | 95.56% | 95.66% | 92.15% | 92.17% | 92.17% |
| 0.15 | 95.72% | 95.72% | 95.90% | 92.42% | 92.39% | 92.40% |
| 0.30 | 93.34% | 93.46% | 93.50% | 86.98% | 86.94% | 86.94% |

> Performance is **near-equivalent under matched BER**: BSC, AWGN, and Rayleigh within **±0.2 pp** at every operating point. Because all methods transmit binary representations and use BPSK hard-decision demodulation, additive noise is converted to effective bit flips.

### ρ Sweep: Bandwidth–Accuracy Pareto (BER=0.30, Seed-42)

| ρ | BW Save | AID | RESISC45 |
|---|---------|-----|----------|
| 0.10 | 90% | 62.68% | 53.98% |
| 0.25 | 75% | 88.12% | 78.74% |
| 0.50 | 50% | 93.26% | 86.58% |
| **0.625** | **37.5%** | **93.50%** | **87.29%** |
| 0.75 | 25% | 93.36% | 87.08% |
| 1.00 | 0% | 92.36% | 85.31% |

### SynOps / Energy Analysis

| Version | FR | ρ | SynOps (M) | Energy Ratio |
|---------|-----|---|------------|--------------|
| V2 (IF baseline) | 0.266 | 0.75 | 321.1 | 23.4× |
| **SpikeAdapt-SC** | **0.167** | **0.75** | **201.5** | **37.3×** |
| SpikeAdapt-SC | 0.167 | 0.50 | 140.1 | 53.5× |

*Computed from measured firing rates using the Horowitz energy model (MAC: 4.6 pJ, SynOp: 0.9 pJ). These are estimated compute savings, not hardware measurements.*

---

## Architecture
![Architecture](https://github.com/JPL11/SpikeAdapt-SC/blob/main/paper/figures/fig1_architecture.png?raw=true)
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
                                  Conv 256→1024, BN+IF
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
│   ├── run_final_pipeline.py            #   Master pipeline (AID + RESISC45)
│   ├── multi_seed_pipeline.py           #   5-seed reproducibility pipeline
│   ├── train_noise_aware_ablation.py    #   Noise-aware ablation training
│   ├── train_1bit_baseline.py           #   CNN-1bit baseline
│   ├── train_aid_v2.py                  #   Core classes (ResNet50Front/Back, BSC_Channel)
│   └── train_aid_v5.py                  #   SNN building blocks (EncoderV5, DecoderV5, MPBN)
│
├── models/                              # Model components
│   ├── spikeadapt_sc.py                 #   SpikeAdapt-SC v5c-NA model
│   ├── noise_aware_scorer.py            #   BER-conditioned importance scorer
│   ├── snn_modules.py                   #   SNN building blocks (LIF, BNTT, MPBN)
│   ├── backbone.py                      #   ResNet-50 split definitions
│   └── energy.py                        #   SynOps energy computation
│
├── eval/                                # Evaluation & figure generation
│   ├── run_ablations_final.py           #   ρ sweep + mask comparison (Table III)
│   ├── multichannel_eval.py             #   BSC/AWGN/Rayleigh eval (Table VI)
│   ├── cnn_baselines.py                 #   CNN-Uni/NonUni baselines
│   ├── cnn_multichannel.py              #   CNN cross-channel eval
│   ├── eval_mlp_baseline.py             #   MLP-FC baseline
│   ├── eval_mlp_multichannel.py         #   MLP cross-channel eval
│   ├── eval_jpeg_conv.py                #   JPEG+Conv baseline
│   ├── compute_synops.py                #   SynOps/energy analysis (Table VII)
│   ├── gen_paper_figures.py             #   Figure generation
│   ├── gen_ber_sweep_figure.py          #   BER sweep figure
│   ├── gen_multichannel_figs.py         #   Cross-channel figures
│   ├── block_importance_analysis.py     #   Block score analysis
│   └── seed_results/                    #   5-seed JSON outputs (per-seed + summary)
│
├── paper/                               # Paper source
│   ├── main.tex                         #   Full manuscript
│   ├── main_6page_revised.tex           #   6-page GlobeCom version
│   └── figures/                         #   Only figures referenced in paper (6 files)
│
├── archive/                             # Development history (old scripts, figures, docs)
│
├── ARTIFACT_TRAIL.md                    # Table → script → output mapping (reviewer guide)
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

# Train SpikeAdapt-SC on AID (50/50 split)
python train/run_final_pipeline.py --stage backbone_aid
python train/run_final_pipeline.py --stage v5c_aid

# Train SpikeAdapt-SC on RESISC45 (20/80 split)
python train/run_final_pipeline.py --stage backbone_resisc
python train/run_final_pipeline.py --stage v5c_resisc

# Run ablations (ρ sweep + mask comparison, 50 random draws)
python eval/run_ablations_final.py

# 5-seed reproducibility
python train/multi_seed_pipeline.py

# Noise-aware ablation
python train/train_noise_aware_ablation.py --dataset aid
python train/train_noise_aware_ablation.py --dataset resisc45

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

| Dataset | Classes | Images | Resolution | Split | Train | Test |
|---------|---------|--------|------------|-------|-------|------|
| **AID** [Xia et al. 2017] | 30 | 10,000 | 600×600 | 50/50 | 5,000 | 5,000 |
| **RESISC45** [Cheng et al. 2017] | 45 | 31,500 | 256×256 | 20/80 | 6,300 | 25,200 |

Both use their **commonly used benchmark splits** for comparability with prior work.

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
- **SNN-SC**: M. Wang et al., "SNN-SC: A spiking semantic communication framework for collaborative intelligence," *IEEE TCCN*, 2024.
- **Horowitz**: M. Horowitz, "Computing's energy problem," *IEEE ISSCC*, 2014.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
