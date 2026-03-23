# Artifact Trail: Paper Tables → Generator Scripts → Output Files

Every quantitative table in the paper has one generator script and one saved output file.
All scripts can be re-run from the repo root (`cd SpikeAdapt-SC`).

---

## Main Paper (`paper/main.tex`)

| Table | Description | Generator Script | Output File(s) | Seeds |
|-------|-------------|-----------------|-----------------|-------|
| Table I | AID main BSC results | `train/multi_seed_pipeline.py` | `eval/seed_results/summary_5seed.json` | 42,123,456,789,1024 |
| Table II | BER sweep (all baselines) | `eval/cnn_baselines.py` + `eval/multichannel_eval.py` | `eval/multichannel_results_v2.json` | 42 |
| Table III | Mask comparison (50-draw) | `eval/run_ablations_final.py` | `eval/ablation_final_results.json` | 42 |
| Table IV | Noise-aware ablation | `train/train_noise_aware_ablation.py` | `eval/noise_aware_ablation_aid.json`, `eval/noise_aware_ablation_resisc45.json` | 42 |
| Table V | Cross-dataset (AID + R45) | `train/multi_seed_pipeline.py` | `eval/seed_results/summary_5seed.json` | 42,123,456,789,1024 |
| Table VI | Cross-channel (BSC/AWGN/Ray) | `eval/multichannel_eval.py` | `eval/multichannel_results_v2.json` | 42 |
| Table VII | Energy/SynOps | `eval/compute_synops.py` | printed to stdout | N/A |

---

## 6-Page Paper (`paper/main_6page_revised.tex`)

| Table | Description | Generator Script | Output File(s) | Seeds |
|-------|-------------|-----------------|-----------------|-------|
| Table 1 | Main BSC results | `train/multi_seed_pipeline.py` | `eval/seed_results/summary_5seed.json` | 42,123,456,789,1024 |
| Table 2 | Mask comparison | `eval/run_ablations_final.py` | `eval/ablation_final_results.json` | 42 |
| Table 3 | Noise-aware ablation | `train/train_noise_aware_ablation.py` | `eval/noise_aware_ablation_aid.json`, `eval/noise_aware_ablation_resisc45.json` | 42 |

---

## Figure Sources

| Figure | Description | Generator Script | Saved As |
|--------|-------------|-----------------|----------|
| Fig 1 | Architecture | Manual (Illustrator) | `paper/figures/fig1_architecture.png` |
| Fig 2 | ρ sweep Pareto | `eval/gen_paper_figures.py` → `fig5_rho_sweep_pareto` | `paper/figures/fig5_rho_sweep_pareto.pdf` |
| Fig 3 | BER sweep (all baselines) | `eval/gen_ber_sweep_figure.py` | `paper/figures/fig_ber_sweep_baselines.png` |
| Fig 4 | Cross-channel comparison | `eval/gen_multichannel_figs.py` | `paper/figures/fig7_unified_ber.pdf` |

---

## Reproducibility Notes

- **5-seed results** use seeds 42, 123, 456, 789, 1024 with full pipeline retraining
  (backbone → S2 → S3) per seed.
- **Seed-42 results** are used for ablation comparisons where full 5-seed retraining
  is impractical (~12h GPU per seed).
- All eval scripts use `target_rate_override` to set ρ explicitly (never `None`),
  ensuring ρ=1.0 is genuinely full-rate.
- ρ=0.75 is the model's trained operating point; ρ=1.0 uses `target_rate_override=1.0`.
- Ablation variants (Table IV) use the identical training pipeline, backbone, and
  evaluation code — only the BER branch and diversity term are toggled.
