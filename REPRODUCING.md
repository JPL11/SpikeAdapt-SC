# Reproducing the reported numbers

Every number in the papers traces to one script and one JSON artifact in
`eval/seed_results/`. All grids and missions follow a leakage-free protocol:
policies are selected on the validation surface and scored once on test;
seeds are 42/123/456 unless stated. Python: the `semcom` conda env
(`requirements.txt`).

## Radar/array paper (ICASSP submission)

| Result | Script | Artifact |
|---|---|---|
| Table 1, SNN y_link row + F_H | `eval/radial_ylink_grid_valtest.py` | `radial_ylink_grid_valtest.json` |
| Table 1, SNN y_stress row | `eval/radial_snn_full_sweep.py` (3 seeds) | `radial_snn_full_sweep*_so.json` |
| Table 1, fair 1-bit control | `train/train_radial_adaptive1bit.py`, `eval/radial_fair1bit_grid_valtest.py` | `radial_adaptive1bit_sweep_s*.json`, `radial_fair1bit_grid_valtest.json` |
| Table 1, quant/LDPC baselines | `eval/radial_feature_quant_baseline.py`, `eval/radial_ldpc_baseline.py` | `radial_quant_baseline.json`, `radial_ldpc_baseline.json` |
| Table 1, anchors | RADIal full protocol via `radial_repo` + `patches/` | `radial_anchor_ft.json` |
| Table 2 (physical closed loop) | `eval/radial_closedloop_valtest.py --grid .../radial_ylink_grid_valtest.json`, then `eval/radial_table2_stats.py --grid ...` | `radial_closedloop_valtest_ylink.json`, `radial_table2_stats_ylink.json` |
| Stress-grid closed loop (secondary) | same scripts, default grid | `radial_closedloop_valtest.json`, `radial_table2_stats.json` |
| Support-aware receiver spot check | `eval/radial_remask_spotcheck.py` | `radial_remask_spotcheck.json` |
| Segmentation cost of the loop | mIoU fields of the y_link grid + `eval/radial_seg_closedloop_valtest.py` | `radial_ylink_grid_valtest.json`, `radial_seg_closedloop_valtest.json` |
| Payload accounting (2.32 Mbit, 25.1×, 6.3×) | `eval/radial_payload_accounting.py` | `radial_payload_accounting.json` |
| i.i.d.-BSC waveform validation | `eval/waveform_channel_validation.py` | `waveform_channel_validation.json` |
| Pilot count-variance check | `eval/pilot_count_variance.py` | `pilot_count_variance.json` |
| DoA transport (9.3×, 21×) | `eval/doa_sim.py` (stages: baselines/train/eval) | `doa_sim_baselines.json`, `doa_sim_snn_seed*.json` |
| CRB / Ziv–Zakai sweep (Fig. 3c) | `eval/doa_sim.py` (bounds stage) | `doa_bounds_snr_sweep.json` |
| Figures 2–3 | `eval/plot_array_icassp_figs.py --ylink` | reads the artifacts above |
| A2G channel/profile model | `eval/channel_a2g.py` | (library) |

Receiver conventions: `y_link = m ⊙ (s ⊕ e)` (support-aware, physical);
`y_stress` additionally scatters flips outside the support (training
curriculum + labeled stress checks). The fair 1-bit control's forward
implements `y_link` natively.

## Visual companion paper (ICC submission)

| Result | Script family | Artifacts |
|---|---|---|
| Classification grids + closed loop (AID/RESISC45) | `eval/valproto_*.py`, `train/train_spikeadapt_taug.py` | `valproto_*.json`, `spikeadapt_taug_joint_grid_*.json` |
| Fair ANN (grouped 1-bit) control | `train/train_adaptive1bit_grouped_joint.py`, `eval/compare_joint_baseline.py` | `adaptive1bit_joint_grid_*.json` |
| Learned-lossy + LDPC separation | `eval/eval_learned_ldpc.py` | `learned_ldpc_results.json` |
| HARQ-style per-frame policies | `eval/harq_style_policy.py`, `eval/radial_harq_sim.py` | `harq_*`, `radial_harq_sim.json` |
| Reconstruction / segmentation deferred tracks | `eval/recon_*_valtest.py`, `eval/radial_seg_closedloop_valtest.py` | `recon_*.json`, `radial_seg_closedloop_valtest.json` |

Statistics conventions: 3-seed mean ± sd with `ddof=1`; harmonic score
`F_H = 2·P̄·R̄/(P̄+R̄)` computed per seed before averaging; RADIal mAP/mAR are
threshold-averaged precision/recall per the dataset protocol (not PR-curve
area).
