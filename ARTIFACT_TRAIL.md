# Artifact Trail

> Every number in the paper traces to **one script** and **one output file**.
> All 10-seed rows use manifest-pinned checkpoints and deterministic BER evaluation (`torch.manual_seed(42)` for noise draws).

---

## Checkpoint Pinning

Each seed directory contains a `selected_checkpoint.txt` manifest that pins the exact checkpoint filename used for all evaluations. The selection logic in `train/multi_seed_pipeline.py` reads this manifest instead of inferring from directory contents.

| Seed Directory | Pinned Checkpoint |
|---|---|
| `snapshots_aid_v5cna_seed42/` | `v5cna_best_95.42.pth` |
| `snapshots_aid_v5cna_seed123/` | `v5cna_best_94.08.pth` |
| `snapshots_aid_v5cna_seed456/` | `v5cna_best_95.00.pth` |
| `snapshots_aid_v5cna_seed789/` | `v5cna_best_94.34.pth` |
| `snapshots_aid_v5cna_seed1024/` | `v5cna_best_95.48.pth` |
| `snapshots_aid_v5cna_seed2048/` | `v5cna_best_95.38.pth` |
| `snapshots_aid_v5cna_seed3072/` | `v5cna_best_95.22.pth` |
| `snapshots_aid_v5cna_seed4096/` | `v5cna_best_94.44.pth` |
| `snapshots_aid_v5cna_seed5120/` | `v5cna_best_95.18.pth` |
| `snapshots_aid_v5cna_seed6144/` | `v5cna_best_95.68.pth` |
| `snapshots_resisc45_v5cna_seed42/` | `v5cna_best_92.01.pth` |
| `snapshots_resisc45_v5cna_seed123/` | `v5cna_best_92.73.pth` |
| `snapshots_resisc45_v5cna_seed456/` | `v5cna_best_92.29.pth` |
| `snapshots_resisc45_v5cna_seed789/` | `v5cna_best_92.55.pth` |
| `snapshots_resisc45_v5cna_seed1024/` | `v5cna_best_92.41.pth` |
| `snapshots_resisc45_v5cna_seed2048/` | `v5cna_best_92.37.pth` |
| `snapshots_resisc45_v5cna_seed3072/` | `v5cna_best_91.59.pth` |
| `snapshots_resisc45_v5cna_seed4096/` | `v5cna_best_92.38.pth` |
| `snapshots_resisc45_v5cna_seed5120/` | `v5cna_best_92.68.pth` |
| `snapshots_resisc45_v5cna_seed6144/` | `v5cna_best_92.40.pth` |

> **Note**: `snapshots_resisc45_v5cna_seed42/v5cna_best_92.61.pth` was renamed to `.SPURIOUS` — it was created by a later retraining run with higher clean accuracy but worse BER=0.30 robustness. All evals use the pinned checkpoint.

---

## Table → Script → Data Mapping

### Table 1: Main BSC Results (`tab:main_results` in `main_6page_revised.tex`)

| Row | Script | Output |
|---|---|---|
| **SpikeAdapt-SC** (ρ=0.75, 0.625) — 10-seed | `train/multi_seed_pipeline.py` | `eval/seed_results/summary_10seed.json` |
| **SNN (no mask, ρ=1.0)** — 10-seed | `train/multi_seed_pipeline.py` | `eval/seed_results/summary_10seed.json` |
| **SNN-SC†** (separately trained) — seed-42 | Separate SNN training without scorer | `eval/snn_sc_baseline_results.json` |
| **CNN-1bit** — 10-seed | `train/train_cnn1bit_10seed.py` | `eval/seed_results/cnn1bit_10seed.json` |
| **JSCC** — 10-seed | `train/train_jscc_baseline.py` | `eval/seed_results/jscc_10seed.json` |
| CNN-Uni, CNN-NonUni† | `eval/cnn_baselines.py` | `eval/cnn_baseline_results.json` |
| MLP-FC† | `eval/eval_mlp_baseline.py` | stdout (seed-42) |
| JPEG+Conv†† | `eval/eval_jpeg_conv.py` | stdout (seed-42) |

> †Seed-42 only (deterministic). ††3× payload; cliff at BER=0.05.

---

### Table 2: BER Sweep (`tab:ber_sweep`)

| Script | Output |
|---|---|
| `eval/gen_full_ber_sweep.py` | `paper/figures/ber_sweep_all_corrected.json` |

Seed-42, BER ∈ {0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30}. Seven methods: SpikeAdapt-SC, SNN-SC, CNN-Uni, CNN-NonUni, MLP-FC, JPEG+Conv.

---

### Table 3: ρ Sweep (`tab:rho_sweep`)

| Script | Output |
|---|---|
| `eval/run_ablations_final.py` | `eval/ablation_final_results.json` |

Seed-42, ρ ∈ {0.10, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 1.0}, BER ∈ {0.0, 0.15, 0.30}.

---

### Table 4: Mask Comparison (`tab:mask_compare`)

| Script | Output |
|---|---|
| `eval/run_ablations_final.py` | `eval/ablation_final_results.json` |

Seed-42. Learned vs. 50 random draws vs. uniform mask at ρ ∈ {0.50, 0.75}.

---

### Table 5: Noise-Aware Ablation (`tab:na_ablation`)

| Script | Output |
|---|---|
| `train/train_noise_aware_ablation.py` | `eval/noise_aware_ablation_aid.json` |
| `train/train_noise_aware_ablation.py` | `eval/noise_aware_ablation_resisc45.json` |

Seed-42. Ablates BER branch and diversity loss independently. Row "SNN-SC (ρ=1.0)" uses the separately trained SNN-SC baseline, not SpikeAdapt-SC at ρ=1.0.

---

### Table 6: Cross-Channel (`tab:multichannel`)

| Script | Output |
|---|---|
| `eval/multichannel_eval.py` | `eval/multichannel_results_v2.json` |

Seed-42. BSC, AWGN, Rayleigh at matched equivalent BER (BPSK hard-decision).

---

### Table 7: SynOps / Energy

| Script | Output |
|---|---|
| `eval/compute_synops.py` | stdout |

Architecture-derived. No channel noise.

---

## Figure → Script Mapping

| Figure | Script | Output |
|---|---|---|
| Fig. 2: Mask diversity | `eval/gen_paper_figures.py` | `paper/figures/fig2_mask_diversity_merged.png` |
| Fig. 3: Feature MSE | `eval/gen_paper_figures.py` | `paper/figures/fig3_feature_mse_dual.png` |
| Fig. 4: BER sweep baselines | `eval/gen_full_ber_sweep.py` | `paper/figures/fig_ber_sweep_baselines.png` |
| Fig. 5: ρ sweep Pareto | `eval/gen_rho_sweep_enhanced.py` | `paper/figures/fig5_rho_sweep_pareto.pdf` |
| Fig. 6: Block importance | `eval/block_importance_analysis.py` | `paper/figures/fig5_block_importance.png` |
| Fig. 7: Channel comparison | `eval/gen_final_figures.py` | `paper/figures/fig7_unified_ber_zoomed.pdf` |

---

## Per-Seed Data Files

```
eval/seed_results/
├── aid_seed42.json            # Per-seed eval (deterministic noise)
├── aid_seed123.json
├── aid_seed456.json
├── aid_seed789.json
├── aid_seed1024.json
├── aid_seed2048.json
├── aid_seed3072.json
├── aid_seed4096.json
├── aid_seed5120.json
├── aid_seed6144.json
├── resisc45_seed42.json
├── resisc45_seed123.json
├── resisc45_seed456.json
├── resisc45_seed789.json
├── resisc45_seed1024.json
├── resisc45_seed2048.json
├── resisc45_seed3072.json
├── resisc45_seed4096.json
├── resisc45_seed5120.json
├── resisc45_seed6144.json
├── cnn1bit_10seed.json        # CNN-1bit 10-seed baseline
├── jscc_10seed.json           # JSCC 10-seed baseline
├── summary_10seed.json        # Aggregated 10-seed mean±std + paired t-tests
└── bootstrap_cis.json         # 95% BCa bootstrap CIs for key claims
```


---

## Baseline Taxonomy

| Label in Paper | What it is | Trained how |
|---|---|---|
| **SpikeAdapt-SC** | Full proposed system (encoder + scorer + decoder) | 3-stage: backbone → SNN → joint w/ scorer |
| **SNN (no mask, ρ=1.0)** | SpikeAdapt-SC with scorer disabled, all blocks sent | Same checkpoint as SpikeAdapt-SC, evaluated at ρ=1.0 |
| **SNN-SC†** | Separately trained SNN encoder/decoder, no scorer component at all | 2-stage only: backbone → SNN (no S3 stage) |
| **CNN-1bit** | Non-spiking 1-bit CNN using STE binarization, T=1 | Same backbone, sign-based binary encoder |
| **JSCC** | Continuous-valued joint source-channel coding | AWGN-trained continuous encoder |
| **CNN-Uni / CNN-NonUni** | 8-bit uniform/non-uniform quantized CNN | Same backbone, quantized features |
| **MLP-FC** | 8-bit fully connected baseline | Same backbone, flattened features |
| **JPEG+Conv** | Traditional separate source-channel coding | JPEG Q=50 + repetition code R=1/3 |

> The distinction between **SNN (no mask)** and **SNN-SC†** is important: the former uses the SpikeAdapt-SC model at ρ=1.0 (scorer exists but all blocks pass); the latter is a completely separate model trained without any scorer component. SNN-SC† is the faithful reproduction of the prior SNN-SC work.
