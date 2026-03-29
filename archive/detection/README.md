# Detection Extension (Exploratory)

> **Not part of the current Globecom paper artifact set.**
> These scripts are exploratory extensions of SpikeAdapt-SC to object detection on the DOTA aerial dataset.

## Contents

- `train/` — DOTA training scripts (SNN-based detection, YOLO-based detection, autoencoder baselines)
- `eval/` — DOTA evaluation scripts and result JSONs

## Status

Preliminary results show that spike-encoded features preserve sufficient spatial detail for bounding box regression even after masking. This direction is mentioned briefly as future work in the paper. No detection results are reported in the Globecom submission.
