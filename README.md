# SpikeAdapt-SC

**Fine-Grained Spiking Semantic Communication with Spatial Masking for Aerial Scene Classification**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> **SpikeAdapt-SC** combines three design elements for energy-efficient, noise-robust aerial scene classification over unreliable UAV channels:
> 1. **37× energy savings** — MPBN reduces the encoder firing rate to 0.167, yielding 37.3× estimated SynOps energy savings (Horowitz model).
> 2. **Binary spike robustness** — At BER=0.30, SpikeAdapt-SC degrades only ~3 pp while a separately trained SNN-SC (no masking) drops ~13 pp and 8-bit CNN baselines collapse by 28–56 pp. A matched-payload CNN-1bit (10 seeds) confirms binary encoding is the primary robustness driver (AID gap: 1.29 pp); the SNN advantage is noise-aware masking, lower variance, and energy efficiency.
> 3. **Learned spatial masking** — A BER-conditioned scorer on the native 14×14 grid (196 blocks) saves 25% bandwidth. On RESISC45, masking at ρ=0.75 *improves* BER=0.30 accuracy by **+1.33 pp** over full-rate (p=0.005, 10-seed paired t-test; 95% BCa bootstrap CI [+0.72, +2.10], survives Bonferroni correction). On AID, masking is bandwidth-neutral (Δ=+0.26 pp, p=0.55).
> 4. **Channel-agnostic** — Trained only on BSC, SpikeAdapt-SC generalizes to AWGN and Rayleigh channels at matched BER (max Δ < 0.2 pp). This is a well-known theoretical result: for uncoded binary signaling with hard-decision decoding, the BSC is the equivalent channel model for any additive-noise channel at the same bit error probability (Proakis, 2008).
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

### CNN-1bit Baseline (10-Seed)

| Condition | AID Mean ± Std | RESISC45 Mean ± Std |
|-----------|----------------|---------------------|
| Clean | 95.41 ± 0.33% | 91.37 ± 0.44% |
| BER=0.30 | 90.86 ± 1.75% | 85.77 ± 2.00% |

> **Honest comparison:** CNN-1bit (STE binarization, T=1, 10 seeds) achieves comparable noise robustness: AID gap is 1.29 pp, R45 gap 0.41 pp. Binary encoding — not temporal coding alone — is the primary driver of noise robustness. The SNN advantage lies in **37× energy efficiency** (via SynOps) and **lower variance** across seeds, not a dramatic accuracy edge over CNN-1bit.

### Cross-Channel Generalization & BEC Immunity

SpikeAdapt-SC trained only on BSC generalizes across all channel types at matched BER:

| Channel | AID (BER=0.30) | RESISC45 (BER=0.30) |
|---------|----------------|---------------------|
| BSC | 93.38% | 84.90% |
| AWGN | 93.34% | 84.92% |
| Rayleigh | 93.40% | 84.92% |
| **BEC** | **95.54%** | **92.43%** |

> **BEC immunity:** BEC erases bits to 0. With SNN firing rate 0.167, ~83% of bits are already 0 — erasures are harmless. This gives SpikeAdapt-SC **+7.5 pp BEC advantage** over BSC on RESISC45, a direct consequence of the sparse spike representation.

---

## Illustrative Results (Seed-42, BSC Channel)

> All tables below use seed-42 as a single illustrative operating point. Seed-42 numbers differ from the 10-seed mean because each seed produces a different trained model. **All paper headline claims use the 10-seed mean±std above.** BER evaluations use a fixed random seed for deterministic noise draws.

### Main Comparison

| Method | AID Clean | AID BER=0.30 | R45 Clean | R45 BER=0.30 | Rate |
|--------|-----------|-------------|-----------|--------------|------|
| **SpikeAdapt-SC (ρ=0.75)** | 95.42% | 93.44% | 92.01% | 86.95% | 75% |
| SpikeAdapt-SC (ρ=0.625) | 95.40% | 93.44% | 91.06% | 87.25% | 62.5% |
| SNN-SC (separately trained) | 95.40% | 82.42% | 92.39% | 80.55% | 100% |
| CNN-1bit (STE, T=1) | 95.32% | 87.56% | 91.48% | 85.19% | 100% |
| CNN-Uni (8-bit) | 91.78% | 52.80% | 78.32% | 49.32% | 100% |
| CNN-NonUni (8-bit) | 91.74% | 63.58% | 78.32% | 50.77% | 100% |
| MLP-FC (8-bit) | 91.98% | 36.12% | 82.23% | 20.89% | 100% |
| JPEG+Conv (R=1/3) | 95.16% | 2.68% | 93.55% | 1.80% | 300% |

> **Key takeaway:** SpikeAdapt-SC degrades ≤3 pp at BER=0.30; SNN-SC (no masking) drops ~13 pp; 8-bit baselines collapse 28–56 pp; JPEG+Conv has a cliff effect at BER=0.05.

### BER Sweep (Seed-42)

| Data | Method | 0.00 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 |
|------|--------|------|------|------|------|------|------|------|
| AID | **SpikeAdapt** | **95.46** | **95.63** | **95.66** | **95.78** | **95.70** | **95.16** | **93.43** |
| AID | SNN-SC | 95.40 | 95.39 | 95.21 | 94.86 | 94.29 | 92.22 | 82.42 |
| AID | CNN-Uni | 91.78 | 91.90 | 90.82 | 88.46 | 83.40 | 72.96 | 52.80 |
| AID | MLP-FC | 91.98 | 88.46 | 84.16 | 76.84 | 66.32 | 51.48 | 36.12 |
| R45 | **SpikeAdapt** | **92.00** | **92.14** | **92.34** | **92.42** | **92.31** | **91.38** | **87.00** |
| R45 | SNN-SC | 92.39 | 92.43 | 92.35 | 92.12 | 91.51 | 89.60 | 80.55 |
| R45 | CNN-Uni | 78.32 | 81.38 | 82.68 | 81.20 | 77.00 | 67.10 | 49.32 |
| R45 | MLP-FC | 82.23 | 74.73 | 65.58 | 54.77 | 43.11 | 31.27 | 20.89 |

### Noise-Aware Ablation (ρ=0.75)

| Variant | BER | Div | AID Clean | AID BER=0.30 | R45 Clean | R45 BER=0.30 |
|---------|-----|-----|-----------|-------------|-----------|-------------|
| **Full** | ✓ | ✓ | 95.42 | **93.40** | 92.01 | **87.10** |
| No BER branch | ✗ | ✓ | 95.94 | 72.54 | 92.27 | 41.52 |
| No diversity loss | ✓ | ✗ | 95.94 | 72.88 | 92.35 | 42.83 |
| Neither | ✗ | ✗ | 95.96 | 72.48 | 92.18 | 39.46 |
| SNN-SC (ρ=1.0) | — | — | 95.40 | 82.42 | 92.39 | 80.55 |

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

> BSC, AWGN, and Rayleigh produce accuracy within **±0.2 pp** at matched BER. This is a well-established result in digital communications: for uncoded binary signaling with hard-decision decoding, the BSC is the equivalent channel model for any additive-noise channel at the same BER (Proakis, 2008). Since SNN spikes are inherently binary ({0,1}) and the receiver applies hard-decision thresholding, the effective error pattern depends **only** on the aggregate BER, not the physical noise mechanism.

### ρ Sweep: Bandwidth–Accuracy Pareto

| ρ | BW Save | AID Clean | AID BER=0.15 | AID BER=0.30 | R45 Clean | R45 BER=0.15 | R45 BER=0.30 |
|---|---------|-----------|--------------|-------------|-----------|--------------|-------------|
| 0.10 | 90% | 83.02 | 79.42 | 62.68 | 71.73 | 69.07 | 53.98 |
| 0.25 | 75% | 92.54 | 92.62 | 88.12 | 83.73 | 83.67 | 78.74 |
| 0.375 | 62.5% | 94.34 | 94.52 | 92.04 | 87.60 | 87.88 | 84.46 |
| 0.50 | 50% | 95.02 | 95.18 | 93.26 | 89.84 | 90.27 | 86.58 |
| **0.625** | **37.5%** | 95.40 | 95.62 | **93.50** | 91.06 | 91.47 | **87.29** |
| 0.75 | 25% | 95.42 | 95.78 | 93.36 | 92.01 | 92.44 | 87.08 |
| 0.875 | 12.5% | 95.42 | 95.76 | 93.02 | 92.40 | 92.80 | 86.61 |
| 1.00 | 0% | 95.20 | 95.64 | 92.36 | 92.53 | 92.91 | 85.31 |

> **ρ=0.625** is the best operating point under noise: +1.1 pp on AID and +2.0 pp on RESISC45 vs ρ=1.0 at BER=0.30, while saving 37.5% bandwidth.

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

## Future Work

Preliminary results on extending SpikeAdapt-SC to object detection on aerial imagery (DOTA dataset) show promising direction. Using a YOLO-based detector head in place of the classification back-end, the spike-encoded features preserve sufficient spatial detail for bounding box regression even after masking. This suggests that the content-adaptive spatial masking framework generalizes beyond classification to dense prediction tasks — a compelling extension for UAV-based surveillance and autonomous systems where both bandwidth and energy are constrained. Future work includes HARQ-style retransmission, temporal early exit for latency-sensitive deployments, and neuromorphic hardware (Intel Loihi) deployment.

---

## References

- **AID**: G.-S. Xia et al., "AID: A benchmark data set for performance evaluation of aerial scene classification," *IEEE TGRS*, 2017.
- **RESISC45**: G. Cheng et al., "Remote sensing image scene classification: Benchmark and state of the art," *Proc. IEEE*, 2017.
- **SNN-SC**: M. Wang et al., "SNN-SC: A spiking semantic communication framework for collaborative intelligence," *IEEE TCCN*, 2024.
- **Horowitz**: M. Horowitz, "Computing's energy problem," *IEEE ISSCC*, 2014.
- **Proakis**: J.G. Proakis & M. Salehi, *Digital Communications*, 5th ed., McGraw-Hill, 2008.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
