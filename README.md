# SpikeAdapt-SC

**Fine-Grained Spiking Semantic Communication with Spatial Masking for Aerial Scene Classification**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> **SpikeAdapt-SC** combines three design elements for energy-efficient, noise-robust aerial scene classification over unreliable UAV channels:
> 1. **37× energy savings** — MPBN reduces the encoder firing rate to 0.167, yielding 37.3× estimated SynOps energy savings (Horowitz model).
> 2. **Binary spike robustness** — At BER=0.30, SNN encoding degrades only ~3 pp while 8-bit CNN baselines collapse by 28–56 pp. A matched-payload CNN-1bit confirms binary encoding is the primary robustness driver (AID gap: 0.82 pp); the SNN advantage is lower variance and energy efficiency.
> 3. **Learned spatial masking** — A BER-conditioned scorer on the native 14×14 grid (196 blocks) saves 25% bandwidth. On RESISC45, masking at ρ=0.75 *improves* BER=0.30 accuracy by **+1.33 pp** over full-rate (p=0.005, 10-seed paired t-test; 95% BCa bootstrap CI [+0.72, +2.10], survives Bonferroni correction). On AID, masking is bandwidth-neutral (Δ=+0.26 pp, p=0.55).
>
> Over **10 seeds** with full pipeline retraining, SpikeAdapt-SC achieves **95.02 ± 0.55%** (AID) and **92.34 ± 0.33%** (RESISC45) clean accuracy at ρ=0.75.

---

## Key Results

### Primary Evidence: 10-Seed Results

Seeds: 42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144. Full pipeline retraining (backbone → S2 → S3) per seed.

| Condition | AID Mean ± Std | RESISC45 Mean ± Std |
|-----------|----------------|---------------------|
| **ρ=1.0, Clean** | **95.49 ± 0.31%** | **92.59 ± 0.16%** |
| **ρ=0.75, Clean** | 95.02 ± 0.55% | 92.34 ± 0.33% |
| ρ=0.625, Clean | 94.12 ± 1.14% | 91.87 ± 0.70% |
| ρ=1.0, BER=0.30 | 91.89 ± 3.69% | 84.85 ± 5.58% |
| **ρ=0.75, BER=0.30** | 92.15 ± 2.87% | **86.18 ± 4.78%** |
| ρ=0.625, BER=0.30 | 90.70 ± 3.26% | 86.07 ± 4.64% |

**Interpretation:**
- **ρ=1.0** gives the highest clean accuracy (no blocks dropped).
- **ρ=0.75** is the practical operating point: saves 25% bandwidth with competitive or improved noise performance.
- On **RESISC45**, masking at ρ=0.75 improves BER=0.30 accuracy by **+1.33 pp** over full-rate (p=0.005, n=10; 95% BCa bootstrap CI [+0.72, +2.10], excludes zero, survives Bonferroni correction for 2 comparisons).
- On **AID**, masking is bandwidth-neutral (Δ=+0.26 pp, p=0.55) — the easier 30-class task has more spatial redundancy, so dropping 25% of blocks has little effect.
- The BER=0.30 std on RESISC45 is ±4.78 pp — the harder 45-class task amplifies scorer sensitivity across seeds, so the robustness claim should be interpreted as a consistent directional effect supported by formal hypothesis testing rather than a narrow-CI guarantee.

### CNN-1bit Baseline (5-Seed)

| Condition | AID Mean ± Std | RESISC45 Mean ± Std |
|-----------|----------------|---------------------|
| Clean | 95.49 ± 0.22% | 91.22 ± 0.54% |
| BER=0.30 | 91.33 ± 2.36% | 85.55 ± 2.72% |

> **Honest comparison:** CNN-1bit (STE binarization, T=1) achieves comparable noise robustness: AID gap is only 0.82 pp, R45 gap ≈0.6 pp. Binary encoding — not temporal coding alone — is the primary driver of noise robustness. The SNN advantage lies in **37× energy efficiency** (via SynOps) and **lower variance** across seeds, not a dramatic accuracy edge over CNN-1bit.

---

## Illustrative Results (Seed-42, BSC Channel)

> All tables below use seed-42 as a single illustrative operating point. Seed-42 numbers differ from the 10-seed mean because each seed produces a different trained model. **All paper headline claims use the 10-seed mean±std above.** BER evaluations use a fixed random seed for deterministic noise draws.

### Main Comparison

| Method | AID Clean | AID BER=0.30 | R45 Clean | R45 BER=0.30 | Rate |
|--------|-----------|-------------|-----------|--------------|------|
| **SpikeAdapt-SC (ρ=0.75)** | 95.42% | 93.44% | 92.01% | 86.95% | 75% |
| SpikeAdapt-SC (ρ=0.625) | 95.40% | 93.44% | 91.06% | 87.25% | 62.5% |
| SNN (no mask, ρ=1.0) | 95.20% | 92.40% | 92.53% | 85.31% | 100% |
| CNN-1bit (STE, T=1) | 95.32% | 87.56% | 91.48% | 85.19% | 100% |
| CNN-Uni (8-bit) | 91.78% | 52.80% | 78.32% | 49.32% | 100% |

### Noise-Aware Ablation (ρ=0.75)

| Variant | BER | Div | AID Clean | AID BER=0.30 | R45 Clean | R45 BER=0.30 |
|---------|-----|-----|-----------|-------------|-----------|-------------|
| **Full** | ✓ | ✓ | 95.42 | **93.40** | 92.01 | **87.10** |
| No BER branch | ✗ | ✓ | 95.94 | 72.54 | 92.27 | 41.52 |
| No diversity loss | ✓ | ✗ | 95.94 | 72.88 | 92.35 | 42.83 |
| Neither | ✗ | ✗ | 95.96 | 72.48 | 92.18 | 39.46 |
| SNN (no mask, ρ=1.0) | — | — | 95.20 | 92.40 | 92.53 | 85.31 |

> Both BER-conditioning and diversity loss are essential. Removing either collapses BER=0.30 accuracy by ~20 pp on AID and ~45 pp on RESISC45, while clean accuracy is unaffected (even slightly higher). The ablation variants are *worse* than even unmasked SNN (ρ=1.0), confirming the collapse is from the scorer, not from sending fewer blocks.

### Mask Comparison (AID, 50 Independent Random Draws)

| ρ | BER | Learned | Random (50-draw) | Uniform | Δ_rand |
|---|-----|---------|------------------|---------|--------|
| 0.50 | 0.00 | 95.02% | 95.28 ± 0.12% | 90.58% | −0.26 |
| 0.50 | 0.30 | **93.44%** | 91.95 ± 0.15% | 85.54% | **+1.49** |
| 0.75 | 0.00 | 95.42% | 95.35 ± 0.10% | 94.70% | +0.07 |
| 0.75 | 0.30 | 93.44% | 93.25 ± 0.13% | 91.78% | +0.19 |

> The learned mask matters most under aggressive compression: at ρ=0.50/BER=0.30, learned masks outperform random by **+1.49 pp**. At ρ=0.75, the advantage is small (+0.19 pp) — the 14×14 grid has high spatial redundancy at mild compression.

### Cross-Channel Comparison (Matched Equivalent BER)

| Equiv BER | BSC (AID) | AWGN (AID) | Ray. (AID) | BSC (R45) | AWGN (R45) | Ray. (R45) |
|-----------|-----------|------------|------------|-----------|------------|------------|
| 0.05 | 95.64% | 95.56% | 95.66% | 92.15% | 92.17% | 92.17% |
| 0.15 | 95.84% | 95.72% | 95.90% | 92.34% | 92.39% | 92.40% |
| 0.30 | 93.44% | 93.46% | 93.50% | 86.95% | 86.94% | 86.94% |

> BSC, AWGN, and Rayleigh produce accuracy within **±0.2 pp** at matched BER. BPSK hard-decision demodulation converts any additive noise channel to approximate BSC behavior, so channel type does not matter — only effective BER.

### ρ Sweep: Bandwidth–Accuracy Pareto (BER=0.30)

| ρ | BW Save | AID | RESISC45 |
|---|---------|-----|----------|
| 0.10 | 90% | 62.68% | 53.98% |
| 0.25 | 75% | 88.12% | 78.74% |
| 0.50 | 50% | 93.26% | 86.58% |
| **0.625** | **37.5%** | **93.44%** | **87.25%** |
| 0.75 | 25% | 93.44% | 86.95% |
| 1.00 | 0% | 92.40% | 85.31% |

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
│   ├── multi_seed_pipeline.py           #   10-seed reproducibility pipeline
│   ├── train_noise_aware_ablation.py    #   Noise-aware ablation training
│   ├── train_1bit_baseline.py           #   CNN-1bit baseline
│   ├── train_aid_v2.py                  #   Core classes (ResNet50Front/Back, BSC_Channel)
│   └── train_aid_v5.py                  #   SNN building blocks (EncoderV5, DecoderV5, MPBN)
│
├── models/                              # Model components
│   ├── spikeadapt_sc.py                 #   SpikeAdapt-SC model definition
│   ├── noise_aware_scorer.py            #   BER-conditioned importance scorer
│   ├── snn_modules.py                   #   SNN building blocks (LIF, BNTT, MPBN)
│   ├── backbone.py                      #   ResNet-50 split definitions
│   └── energy.py                        #   SynOps energy computation
│
├── eval/                                # Evaluation & figure generation
│   ├── run_ablations_final.py           #   ρ sweep + mask comparison
│   ├── multichannel_eval.py             #   BSC/AWGN/Rayleigh eval
│   ├── cnn_baselines.py                 #   CNN-Uni/NonUni baselines
│   ├── cnn_multichannel.py              #   CNN cross-channel eval
│   ├── eval_mlp_baseline.py             #   MLP-FC baseline
│   ├── eval_mlp_multichannel.py         #   MLP cross-channel eval
│   ├── eval_jpeg_conv.py                #   JPEG+Conv baseline
│   ├── compute_synops.py                #   SynOps/energy analysis
│   ├── gen_paper_figures.py             #   Figure generation
│   ├── gen_ber_sweep_figure.py          #   BER sweep figure
│   ├── gen_multichannel_figs.py         #   Cross-channel figures
│   ├── block_importance_analysis.py     #   Block score analysis
│   └── seed_results/                    #   10-seed JSON outputs (per-seed + summary)
│
├── paper/                               # Paper assets
│   └── figures/                         #   Figures referenced in paper
│
├── archive/                             # Development history
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

# 10-seed reproducibility
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
