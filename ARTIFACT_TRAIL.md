# Artifact Trail

> Every number in the paper traces to **one script** and **one output file**.
> An earlier mixed-checkpoint artifact in the RESISC45 seed-42 directory caused one stale BER result to be selected automatically; all reported results have been regenerated after pinning the intended checkpoint per seed via `selected_checkpoint.txt` manifests.

## Checkpoint Pinning

Each seed directory contains a `selected_checkpoint.txt` manifest that pins the exact checkpoint filename used for all evaluations. The selection logic in `train/multi_seed_pipeline.py` reads this manifest instead of inferring from directory contents.

| Seed Directory | Pinned Checkpoint |
|---|---|
| `snapshots_aid_v5cna_seed42/` | `v5cna_best_95.42.pth` |
| `snapshots_aid_v5cna_seed123/` | `v5cna_best_94.08.pth` |
| `snapshots_aid_v5cna_seed456/` | `v5cna_best_95.00.pth` |
| `snapshots_aid_v5cna_seed789/` | `v5cna_best_94.34.pth` |
| `snapshots_aid_v5cna_seed1024/` | `v5cna_best_95.48.pth` |
| `snapshots_resisc45_v5cna_seed42/` | `v5cna_best_92.01.pth` |
| `snapshots_resisc45_v5cna_seed123/` | `v5cna_best_92.73.pth` |
| `snapshots_resisc45_v5cna_seed456/` | `v5cna_best_92.29.pth` |
| `snapshots_resisc45_v5cna_seed789/` | `v5cna_best_92.55.pth` |
| `snapshots_resisc45_v5cna_seed1024/` | `v5cna_best_92.41.pth` |

> **Note**: `snapshots_resisc45_v5cna_seed42/v5cna_best_92.61.pth` was renamed to `.SPURIOUS` — it was created by a later retraining run (Mar 21) with higher clean accuracy but worse BER=0.30 robustness (84.88% vs 87.03%). All evals now use the pinned `v5cna_best_92.01.pth`.

---

## Table → Script → Data Mapping

### Table 1: Main Results (AID, `tab:main_results`)
**Paper**: `main.tex` L247, `main_6page_revised.tex` L181

| Row | Script | Output |
|---|---|---|
| SpikeAdapt-SC (ρ=0.75, 0.625) | `train/multi_seed_pipeline.py` | `eval/seed_results/summary_5seed.json` |
| SNN (no mask, ρ=1.0) | `train/multi_seed_pipeline.py` | `eval/seed_results/summary_5seed.json` |
| CNN-1bit | `train/train_cnn1bit_5seed.py` | `eval/seed_results/cnn1bit_5seed.json` |
| CNN-Uni, CNN-NonUni | `eval/cnn_baselines.py` | `eval/cnn_baseline_results.json` |
| MLP-FC | `eval/eval_mlp_baseline.py` | stdout (seed-42) |
| JPEG+Conv | `eval/eval_jpeg_conv.py` | stdout (seed-42) |

All 5-seed rows use deterministic BER eval (`torch.manual_seed(42)`) and manifest-pinned checkpoints.

---

### Table 2: Cross-Dataset Results (`tab:cross_dataset`)
**Paper**: `main.tex` L378

Same sources as Table 1. Cross-dataset table presents both AID and RESISC45 columns from `summary_5seed.json`.

---

### Table 3: BER Sweep (`tab:ber_sweep`)
**Paper**: `main.tex` L412, `main_6page_revised.tex` L213

| Script | Output |
|---|---|
| `eval/run_ablations_final.py` | `eval/ablation_final_results.json` |

Seed-42, BER ∈ {0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30}.

---

### Table 4: ρ Sweep (`tab:rho_ablation` / `tab:rho_sweep`)
**Paper**: `main.tex` L311, `main_6page_revised.tex` L285

| Script | Output |
|---|---|
| `eval/run_ablations_final.py` | `eval/ablation_final_results.json` |

Seed-42, ρ ∈ {0.10, 0.25, 0.50, 0.625, 0.75, 1.0}, BER=0.30.

---

### Table 5: Mask Comparison (`tab:mask_compare`)
**Paper**: `main.tex` L336, `main_6page_revised.tex` L256

| Script | Output |
|---|---|
| `eval/run_ablations_final.py` | `eval/ablation_final_results.json` |

Seed-42. Learned vs. 50 random draws vs. uniform mask.

---

### Table 6: Noise-Aware Ablation (`tab:na_ablation`)
**Paper**: `main.tex` L356, `main_6page_revised.tex` L312

| Script | Output |
|---|---|
| Ablation training + eval | `eval/noise_aware_ablation_aid.json` |
| Ablation training + eval | `eval/noise_aware_ablation_resisc45.json` |

Seed-42. Ablates BER branch and diversity loss independently. All variants use identical pipeline; only BER branch and diversity term toggled.

---

### Table 7: Cross-Channel / Multichannel (`tab:multichannel`)
**Paper**: `main.tex` L443, `main_6page_revised.tex` L337

| Script | Output |
|---|---|
| `eval/multichannel_eval.py` | `eval/multichannel_results_v2.json` |

Seed-42. BSC, AWGN, Rayleigh at matched equivalent BER.

---

### Table 8: SynOps / Energy (`tab:energy` / `tab:synops`)
**Paper**: `main.tex` L200, `main_6page_revised.tex` L133

| Script | Output |
|---|---|
| `eval/compute_synops.py` | stdout |

Architecture-derived. No channel noise.

---

### Table 9: Training Hyperparameters (`tab:training`)
**Paper**: `main_6page_revised.tex` L158

Documents training configuration. No eval script.

---

## Per-Seed Data Files

```
eval/seed_results/
├── aid_seed42.json          # Deterministic eval (torch.manual_seed(42))
├── aid_seed123.json
├── aid_seed456.json
├── aid_seed789.json
├── aid_seed1024.json
├── resisc45_seed42.json     # Deterministic eval (torch.manual_seed(42))
├── resisc45_seed123.json
├── resisc45_seed456.json
├── resisc45_seed789.json
├── resisc45_seed1024.json
├── cnn1bit_5seed.json       # CNN-1bit 5-seed results
└── summary_5seed.json       # Aggregated mean±std + paired t-tests
```
