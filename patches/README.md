# Third-party patches

`radial_repo_modifications.diff` — modifications to [valeoai/RADIal](https://github.com/valeoai/RADIal) (Apache 2.0) for the radar-track pipeline:
- `FFTRadNet/3-Evaluation.py`: torch.load weights_only=False (PyTorch 2.6) + state-dict key remap (backbone.->FPN., backbone.preproc.->FPN.pre_enc., RAmap_header.->RA_decoder.).
- `FFTRadNet/utils/metrics.py`: divide-by-zero guard when a threshold yields no detections.
- `radial_config_local.json`: local dataset-root config.

Apply: `git clone https://github.com/valeoai/RADIal && cd RADIal && git apply ../patches/radial_repo_modifications.diff`
