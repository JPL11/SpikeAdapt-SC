# SpikeAdapt-SC

**Content-Adaptive Neuromorphic Semantic Communication for Collaborative Aerial Intelligence**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> **SpikeAdapt-SC** is a spiking neural network (SNN) framework for content-adaptive semantic communication in UAV aerial networks. It encodes deep features as binary spike trains on a **native 14×14 feature grid** (196 spatial blocks), learns per-image spatial masking, and achieves **96.35% top-1 accuracy on 30 aerial scene classes** (AID dataset) while **outperforming the full-rate unmasked baseline** (95.70%) with **25% spatial bandwidth savings**. A multi-exit temporal decoder adds **26% temporal savings** (avg T=5.9 at 96.55%), yielding **~45% total bandwidth reduction**. Robust down to BER=0.30 (92.90%) while CNN-Uni collapses to 67%.

---

## System Architecture

<p align="center">
  <img src="paper/figures/fig1_architecture.png" alt="SpikeAdapt-SC Architecture" width="100%">
</p>

**Key idea:** Instead of transmitting all spatial feature blocks uniformly, SpikeAdapt-SC:

1. **Encodes features as binary spikes** using IF neurons over T=8 timesteps on the native 14×14 grid
2. **Scores each spatial block's importance** via a learned channel-conditioned scorer (C₂=36)
3. **Masks unimportant blocks** — content-adaptive, per-image (2,000/2,000 unique masks at ρ=0.75)
4. **Transmits only selected blocks** over noisy channels (BSC)
5. **Decodes** using a matched SNN decoder with multi-exit capability

---

## Results

All results use **top-1 classification accuracy** on the **2,000-image AID test set** (80/20 split, seed=42).

### Headline Results (BSC Channel)

| Method | Clean | BER=0.15 | BER=0.30 | Rate |
|--------|-------|----------|----------|------|
| **SpikeAdapt-SC** (ρ=0.75) | **96.35%** | **95.85%** | **92.90%** | 75% |
| SNN (no mask, ρ=1.0) | 95.70% | 95.35% | 92.20% | 100% |
| CNN-Uni 8-bit | 92.50% | 92.40% | 67.25% | 100% |

> SpikeAdapt-SC **beats the full-rate unmasked baseline at 75% bandwidth** (96.35% > 95.70%). CNN-Uni collapses at BER=0.30 (67.25%).

### Multi-Exit Temporal Decoding

| Exit T | Clean | BER=0.15 | BER=0.30 | Temporal Savings |
|--------|-------|----------|----------|-----------------|
| T=6 | **96.60%** | **96.15%** | 92.50% | 25% |
| T=8 | 96.05% | 95.90% | **94.60%** | 0% |
| Early exit (θ=0.95) | 96.55% | 96.15% | 94.25% | 12–26% |

> T=6 **exceeds** T=8 accuracy (96.60% > 96.05%). Under noise, the system automatically uses more timesteps.

### Ablation Study (Learned vs Random Masking)

| Variant | Accuracy | BW Saved | Δ |
|---------|----------|----------|---|
| SpikeAdapt-SC (ρ=0.75) | **96.35%** | 25% | — |
| SNN (no mask, ρ=1.0) | 95.70% | 0% | −0.65 |
| Random mask (ρ=0.75, 100 draws) | 94.85 ± 0.17% | 25% | −1.50 |
| SpikeAdapt-SC (ρ=0.50) | 95.35% | 50% | −1.00 |
| Random mask (ρ=0.50, 100 draws) | 84.41 ± 0.45% | 50% | −11.94 |

### Seed Variance (5 seeds)

| Seed | Accuracy |
|------|----------|
| 42 | 96.35% |
| 101 | 96.55% |
| 123 | 96.05% |
| 456 | 95.90% |
| 789 | 96.35% |
| **Mean ± std** | **96.24 ± 0.25%** |

### Content-Adaptive Masking

<p align="center">
  <img src="paper/figures/fig2_mask_diversity.png" alt="Content-Adaptive Masks" width="100%">
</p>

### Semantic Feature Resilience

<p align="center">
  <img src="paper/figures/fig3_feature_mse.png" alt="Feature MSE vs Confidence" width="100%">
</p>

### Energy Savings (Estimated)

| Channel | Avg Firing Rate | Energy Savings |
|---------|----------------|----------------|
| BSC | 0.41 | 32% |
| AWGN | 0.28 | 48% |
| Rayleigh | 0.31 | 46% |

*Computed from firing rate statistics using the Horowitz energy model (MAC: 4.6 pJ, SynOp: 0.9 pJ).*

---

## Repository Structure

```
SpikeAdapt-SC/
├── train/                           # Training scripts
│   ├── train_aid.py                 #   AID v1 (8×8 pool, C2=128)
│   ├── train_aid_v2.py              #   AID v2 (14×14 native, C2=36) ← main
│   └── train_temporal_v3.py         #   Multi-exit temporal training
│
├── eval/                            # Evaluation & analysis
│   ├── sanity_check_full.py         #   Full-dataset score analysis (prof review)
│   ├── analyze_v2.py                #   V2 randomization analysis
│   └── multi_trace_eval.py          #   Multi-trajectory evaluation
│
├── paper/                           # Paper assets
│   ├── main.tex                     #   Manuscript source
│   ├── section_randomization_analysis.tex
│   └── figures/                     #   Publication-quality figures
│
├── snapshots_aid/                   # V1 checkpoints
├── snapshots_aid_v2_seed*/          # V2 checkpoints (per seed)
├── snapshots_aid_v3_seed*/          # V3 multi-exit checkpoints
│
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

# Train v2 on AID (14×14 grid, BSC channel)
python train/train_aid_v2.py

# Multi-exit temporal training (v3)
python train/train_temporal_v3.py

# Full-dataset sanity check (all 2000 images)
python eval/sanity_check_full.py
```

---

## Training Pipeline

| Stage | Description | Epochs | LR |
|-------|-------------|--------|------|
| **S1** | ResNet50 backbone fine-tuning on AID | 50 | 0.01 |
| **S2** | SNN encoder/decoder (backbone frozen) | 60 | 1e-4 |
| **S3** | Joint fine-tuning (back + decoder) | 30 | 1e-5 |
| **S4** | Multi-exit decoder fine-tuning (v3) | 30 | 5e-5 |

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
