# SpikeAdapt-SC: Comprehensive Data Analysis

> **Status**: ALL COMPLETE — CIFAR-100 (10+ experiments) + Tiny-ImageNet 16×16 (3 channels) + Pooled 8×8 (3 channels)


---

## 1. CIFAR-100 Baseline Comparison

### 1.1 BER Robustness (BSC Channel)

| Method | Type | BW (bits) | BER=0 | BER=0.1 | BER=0.2 | BER=0.3 | Drop₀→₀.₃ |
|--------|------|-----------|-------|---------|---------|---------|-----------|
| Backbone | Upper bound | — | 76.89% | — | — | — | — |
| **BER-Robust L3** | **Ours (best)** | **~53K** | **75.05%** | **75.21%** | **74.93%** | **72.27%** | **-2.78%** |
| CE-only L4 | Ours (no mask) | 16,384 | 75.52% | 75.62% | 75.54% | 75.15% | -0.37% |
| SNN-SC T=8 | Baseline | 16,384 | 75.78% | 75.52% | 74.89% | 71.79% | -3.99% |
| SNN-SC T=6 | Baseline | 12,288 | 75.56% | 75.48% | — | 72.26% | -3.30% |
| SpikeAdapt L4 | Ours (v2) | 12,304 | 73.45% | 73.23% | 73.09% | 72.44% | -1.01% |
| Robust Learned Imp | Ours | ~49K | 74.68% | 74.71% | 74.01% | 68.58% | -6.10% |
| CNN-Bern | Baseline | 2,048 | 75.51% | 74.95% | — | 70.12% | -5.39% |
| Random Mask | Baseline | ~12,288 | 75.44% | 75.50% | — | 74.33% | -1.11% |
| CNN-Uni | Baseline | 16,384 | 29.80% | 1.00% | — | 1.00% | cliff |
| JPEG+Conv | Baseline | ~50K | 76.88% | 1.00% | — | 1.00% | cliff |

**Key observations:**
- SNN-based methods maintain graceful degradation; CNN/JPEG show cliff effects
- BER-Robust L3 beats SNN-SC T=8 at BER≥0.2 while providing spatial masking
- CE-only L4 has the best robustness (-0.37%) but no bandwidth savings

---

## 2. Adaptive Bandwidth Analysis

### 2.1 CIFAR-100 Rate Sweep (Learned Importance, BER-Robust)

| Tx Rate | Blocks Sent | BW (bits) | Accuracy | Δ vs 100% |
|---------|-------------|-----------|----------|-----------|
| 100% | 64/64 | ~65K | 74.68% | — |
| 90% | 57/64 | ~59K | 74.73% | +0.05% |
| 80% | 51/64 | ~52K | 74.71% | +0.03% |
| 75% | 48/64 | ~49K | 74.68% | 0.00% |
| 60% | 38/64 | ~39K | 74.37% | -0.31% |
| **50%** | **32/64** | **~33K** | **73.83%** | **-0.85%** |
| 40% | 25/64 | ~26K | 72.57% | -2.11% |
| 25% | 16/64 | ~16K | 68.43% | -6.25% |

**Finding**: 80→100% range shows **zero accuracy loss** — 20% of blocks are noise. At 50% rate, only 0.85% accuracy drop = **50% bandwidth savings nearly for free**.

### 2.2 CIFAR-100 η Sweep (BER-Robust Entropy)

| η | Accuracy | Tx Rate | 
|---|----------|---------|
| 0.0 | 75.26% | 1.000 |
| 0.5 | 75.28% | 0.874 |
| 0.8 | 75.29% | 0.866 |
| 0.9 | 75.37% | 0.820 |

**Flat Pareto front**: η=0.0 to η=0.9 spans 75.26–75.37% while Tx drops from 1.0 to 0.82. **18% free bandwidth savings**.

---

## 3. Content Adaptation Analysis

### 3.1 Mask Diversity: L4 (4×4) vs L3 (8×8)

| Metric | L4 (4×4) | L3 Entropy (8×8) | Learned Imp (8×8) |
|--------|----------|------------------|-------------------|
| **Unique masks** | 2 | 2,478 | **8,987** |
| Ent-Imp correlation | -0.142 | +0.020 | N/A (learned) |
| Tx rate std | 0.007 | 0.044 | ~0.05 |
| Total blocks | 16 | 64 | 64 |

**Interpretation**: 4×4 produces a static spatial mask (same for every image). Moving to 8×8 creates genuine per-image content adaptation. Learned importance achieves near-perfect uniqueness (8,987/10,000).

### 3.2 Masking Strategy Comparison (L3, 8×8)

| Strategy | 75% keep | 50% keep |
|----------|----------|----------|
| All blocks | 73.98% | 73.98% |
| **Entropy top-k** | **73.54%** | **72.95%** |
| Random | 72.70% | 70.26% |
| Anti-entropy | 70.86% | 64.88% |

**Entropy beats random** by +0.84% at 75% and +2.69% at 50%. Anti-entropy collapses → confirms the criterion captures real signal.

### 3.3 Spatial Entropy Pattern (8×8)

```
Average entropy — what the model keeps/drops:
  0.80  0.94  0.98  0.98  0.98  0.98  0.86  0.73
  0.90  0.22  0.57  0.65  0.68  0.62  0.13  0.97   ← inner border: LOW (dropped)
  0.99  0.48  0.90  0.93  0.94  0.92  0.30  0.98
  0.99  0.49  0.91  0.95  0.95  0.93  0.31  0.98   ← center: HIGH (kept)
  0.99  0.51  0.92  0.95  0.95  0.93  0.32  0.98
  0.98  0.52  0.92  0.95  0.95  0.92  0.32  0.98
  0.98  0.35  0.76  0.81  0.81  0.76  0.20  0.98
  0.83  0.91  0.99  0.98  0.98  0.99  0.72  0.86
```

The model learns a **ring pattern**: inner-border positions (columns 1, 6) have low entropy and get dropped first. Center and edges are preserved.

---

## 4. Ablation Studies

### 4.1 Loss Function Ablation

| Config | Split | BER=0 | BER=0.3 | Δ vs Full |
|--------|-------|-------|---------|----------|
| Full (CE+Ent+Rate) | L4 4×4 | 73.45% | 72.44% | baseline |
| **CE-only** | **L4 4×4** | **75.52%** | **75.15%** | **+2.07%/+2.71%** |
| Full (CE+Ent+Rate) | L3 8×8 | 74.50% | 69.32% | baseline |
| CE-only | L3 8×8 | 74.38% | 69.25% | -0.12%/-0.07% |

**Critical finding**: On 4×4, entropy/rate losses **hurt by ~2.5%** because they force a static mask. On 8×8, they're **neutral** (±0.1%) because the mask actually works.

### 4.2 BER Training Strategy

| Training BER range | BER=0 | BER=0.3 | Δ |
|-------------------|-------|---------|---|
| [0, 0.3] uniform | 74.50% | 69.32% | baseline |
| **[0, 0.4] weighted** | **75.05%** | **72.27%** | **+0.55%/+2.95%** |

50% oversample of high BER recovers **+2.95% at BER=0.3** with minimal clean impact.

### 4.3 Importance Criterion

| Criterion | BER=0 | BER=0.3 | Masks | Advantage |
|-----------|-------|---------|-------|-----------|
| Spike-rate entropy | 75.05% | 72.27% | 2,478 | More robust |
| **Learned scorer** | 74.68% | 68.58% | **8,987** | More adaptive |

Learned importance has **4× better adaptation** but worse BER robustness (Gumbel sampling hurts temporal consistency). Entropy-based is simpler and more robust.

---

## 5. Tiny-ImageNet Results (200 classes, 64×64)

**Backbone**: 64.80% (ResNet50, Layer3 split → 1024×16×16 = 256 blocks)

### 5.1 BSC Channel

| BER | Accuracy | Δ vs clean |
|-----|----------|-----------|
| 0.0 | 59.30% | — |
| 0.05 | 60.33% | +1.03% |
| 0.1 | 61.33% | +2.03% |
| 0.15 | **61.49%** | **+2.19%** |
| 0.2 | 61.01% | +1.71% |
| 0.25 | 59.05% | -0.25% |
| 0.3 | 53.36% | -5.94% |

**Anomaly**: Accuracy **increases** from BER=0 to BER=0.15. This suggests the BER-robust training creates redundancy that helps under moderate noise (bit-flip averaging effect).

### 5.2 AWGN Channel

| SNR (dB) | Accuracy | Equiv. BER |
|----------|----------|-----------|
| 20 | 62.33% | ~0.0000 |
| 10 | 62.33% | ~0.0004 |
| 5 | 62.33% | ~0.006 |
| 3 | 62.57% | ~0.023 |
| 1 | 62.65% | ~0.056 |
| 0 | 62.22% | ~0.079 |
| -2 | 58.60% | ~0.132 |

**Remarkable**: Accuracy is **virtually flat from SNR=20 to SNR=0 dB** (62.33→62.22%). SNN encoding provides near-perfect robustness to Gaussian noise. Only at -2 dB does performance degrade.

### 5.3 Rayleigh Fading Channel (FINAL)

| SNR (dB) | Accuracy | Equiv. BER |
|----------|----------|------------|
| 20 | 61.35% | ~0.0001 |
| 15 | 61.48% | ~0.001 |
| 10 | 61.80% | ~0.004 |
| 7 | **61.88%** | ~0.013 |
| 5 | 61.83% | ~0.029 |
| 3 | 61.77% | ~0.053 |
| 1 | 60.97% | ~0.095 |
| 0 | 60.11% | ~0.126 |
| -2 | 56.87% | ~0.194 |

**Rayleigh is remarkably robust** — accuracy holds 61.35→61.88% from SNR=20 to SNR=7dB despite the fading channel. The SNN's binary encoding is naturally immune to multiplicative fading distortion.

### 5.4 Adaptive Bandwidth — Tiny-ImageNet

| Channel | Rate=100% | Rate=75% | Rate=50% | Rate=25% |
|---------|-----------|----------|----------|----------|
| BSC | 60.05% | 59.30% | 56.19% | 41.86% |
| AWGN | 61.82% | **62.29%** | 60.45% | 27.33% |
| Rayleigh | 60.97% | 60.28% | 52.20% | 20.62% |

**AWGN at Rate=75% beats Rate=100%** (62.29% > 61.82%) — dropping 25% of blocks actually helps by removing noise-sensitive blocks.

### 5.5 Energy Efficiency

| Channel | SNN SynOps | ANN MACs | Energy Ratio | **Savings** |
|---------|-----------|----------|-------------|------------|
| BSC | 537B | 155B | 0.680 | **32.0%** |
| AWGN | 409B | 155B | 0.518 | **48.2%** |
| Rayleigh | 427B | 155B | 0.540 | **46.0%** |

Based on Horowitz 2014: SynOp = 0.9 pJ, MAC = 4.6 pJ (45nm CMOS).
AWGN achieves higher energy savings because lower firing rates → fewer SynOps.

---

## 6. Cross-Dataset Consistency

| Metric | CIFAR-100 | Tiny-ImageNet |
|--------|-----------|---------------|
| Backbone accuracy | 76.89% | 64.80% |
| SpikeAdapt clean | 75.05% | 59.30–62.33% |
| Accuracy retention | 97.6% | 91.5–96.2% |
| BER=0.3 / SNR=0dB robustness | -2.78% | -5.94% / -0.2% |
| Rate=75% impact | ~0% | -0.75% to +0.47% |
| Rate=50% impact | -0.85% | -1.37% to -6.4% |

Both datasets show: (1) graceful degradation under noise, (2) near-free bandwidth savings at 75%, (3) significant energy savings.

### 6.1 Pooled 8×8 vs Native 16×16 — All Channels

**BSC**

| Grid | BER=0 | BER=0.1 | BER=0.3 | Energy |
|------|-------|---------|---------|--------|
| **16×16** | **59.30%** | **61.33%** | **53.36%** | 32% |
| 8×8 pooled | 52.24% | 52.17% | 42.35% | 45% |

**AWGN**

| Grid | SNR=20 | SNR=0 | SNR=-2 | Energy |
|------|--------|-------|--------|--------|
| **16×16** | **62.33%** | **62.22%** | **58.60%** | **48%** |
| 8×8 pooled | 54.79% | 55.76% | 53.16% | 28% |

**Rayleigh**

| Grid | SNR=20 | SNR=0 | SNR=-2 | Energy |
|------|--------|-------|--------|--------|
| **16×16** | **61.35%** | **60.11%** | **56.87%** | **46%** |
| 8×8 pooled | 56.58% | 56.84% | 54.64% | 44% |

**16×16 wins across all channels** by 5–8%. The higher spatial resolution preserves more features. Pooling sacrifices detail the decoder needs.

**Pooled AWGN rate sweep anomaly**: Rate=50%→56.59% **beats** Rate=100%→54.17%. Aggressive masking removes noise-vulnerable blocks, improving accuracy. This effect is stronger on pooled (8×8) than native (16×16).

---

## 7. Summary of Key Findings

| # | Finding | Confidence |
|---|---------|-----------|
| 1 | SNN provides excellent channel robustness vs ANN | ✅ High |
| 2 | 8×8 spatial grid enables content adaptation (2,478+ masks) | ✅ High |
| 3 | Learned importance maximizes per-image uniqueness (8,987) | ✅ High |
| 4 | BER-weighted training essential (+2.95% at BER=0.3) | ✅ High |
| 5 | Entropy loss hurts on 4×4, neutral on 8×8 | ✅ High |
| 6 | 50% bandwidth savings costs <1% accuracy (CIFAR-100) | ✅ High |
| 7 | AWGN: masking improves accuracy (noise reduction effect) | ⚠️ Medium |
| 8 | 32-48% energy savings vs ANN equivalent | ⚠️ Theoretical |
| 9 | AWGN flat from SNR=20 to SNR=0 dB | ✅ High |
| 10 | BSC accuracy peaks at BER=0.15 (redundancy effect) | ⚠️ Medium |
| 11 | Rayleigh flat from SNR=20 to SNR=3 dB | ✅ High |
| 12 | 16×16 beats pooled 8×8 on Tiny-ImageNet (+7%) | ✅ High |
