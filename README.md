# SpikeAdapt-SC

Adaptive spiking semantic communication for aerial platforms: task-oriented
transport of visual features and radar array features over modeled UAV
air-to-ground (A2G) channels, with training-free rate adaptation (spatial
masking ρ × temporal truncation T′) driven in closed loop by a pilot-based
BER estimate.

This repository accompanies two papers by Jacky Li (California State
Polytechnic University, Pomona), both currently under review:

1. **Adaptive spiking semantic communication for robust and efficient aerial
   visual inference** (visual track: AID / NWPU-RESISC45 classification,
   DOTA detection, segmentation, reconstruction).
2. **Task-oriented spiking transport of radar features over UAV air-to-ground
   links** (radar/array track: RADIal detection + free-space segmentation
   over FFTRadNet features, DoA transport with CRB / Ziv–Zakai anchors).

Citations will be added when the papers appear; until then please cite the
repository.

## Layout

| Path | Contents |
|---|---|
| `train/` | Encoder/decoder/codec training (spiking bottlenecks, fair 1-bit controls, BER curriculum) |
| `eval/` | Evaluation, channel modeling, closed-loop missions, plotting |
| `eval/seed_results/` | JSON artifacts backing every reported number (see `REPRODUCING.md`) |
| `models/` | Spiking modules (LIF encoder/decoder, noise-aware scorer, masking) |
| `patches/` | Modifications to the third-party FFTRadNet repo (see `patches/README.md`) |
| `datasets/` | Dataset download/preparation scripts |

## Setup

```bash
conda create -n semcom python=3.11
conda activate semcom
pip install -r requirements.txt
```

Datasets: AID, NWPU-RESISC45, DOTA (visual track); RADIal (radar track —
`datasets/` contains the download helper; ~140 GB). The RADIal baseline uses
the official FFTRadNet code with the changes captured in `patches/`
(clone the upstream repo as `radial_repo/FFTRadNet` and apply the diff).

## Reproducing paper numbers

Every table/figure entry traces to one script and one JSON artifact in
`eval/seed_results/`; the mapping is in [`REPRODUCING.md`](REPRODUCING.md).
The result JSONs are committed, so paper-level statistics and figures can be
regenerated without GPU work; full re-evaluation of the grids requires the
datasets and checkpoints (training scripts and seeds included).

## License

MIT (see `LICENSE`).
