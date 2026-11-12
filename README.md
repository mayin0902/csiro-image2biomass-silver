# CSIRO Image2Biomass Prediction

![Kaggle](https://img.shields.io/badge/Kaggle-Silver%20Medal-C0C0C0)
![Rank](https://img.shields.io/badge/Rank-83%20%2F%203803-blue)
![Organizer](https://img.shields.io/badge/Organizer-CSIRO-green)

**Kaggle Silver Medal solution**, ranked **83 / 3803** in **CSIRO - Image2Biomass Prediction**.

This competition asks participants to predict five biomass targets from high-resolution UAV top-view grassland images. The final solution combines a dual-view vision backbone, structured multi-target regression, and a semantic compensation branch for small-sample robustness.

## Result

| Field | Value |
| --- | --- |
| Competition | CSIRO - Image2Biomass Prediction |
| Organizer | CSIRO |
| Medal | **Silver Medal** |
| Rank | **83 / 3803** |
| Awarded | January 29, 2026 |
| Metric | Weighted R2 |

## Method Overview

- Split each `2000 x 1000` UAV image into left and right views to preserve local texture and canopy-density information.
- Encode both views with a shared DINOv3 / ViT-style visual backbone.
- Fuse cross-view patch tokens with a gated local token mixer before global pooling.
- Predict `Green`, `Dead`, and `Clover` biomass directly, then reconstruct `GDM = Green + Clover` and `Total = GDM + Dead`.
- Add a SigLIP feature branch with LightGBM / CatBoost regressors as semantic compensation.
- Use cross-validation, mixed precision, test-time averaging, and late ensemble blending for the final submission.

## Repository Layout

```text
configs/
  kaggle_dino_siglip.yaml
scripts/
  train.py
  infer.py
src/biomass/
  config.py
  data.py
  model.py
  train_pipeline.py
  inference_pipeline.py
  siglip_gbdt.py
  metrics.py
  utils.py
```

## Quick Start

```bash
python -m pip install -r requirements.txt
python scripts/train.py --config configs/kaggle_dino_siglip.yaml
python scripts/infer.py --config configs/kaggle_dino_siglip.yaml
```

Inference writes `submission_wide.csv` under the configured `paths.output_dir`. Pass `--output ./submission_wide.csv` to write the CSV in the current directory.

The Kaggle data and pretrained model paths are configured in `configs/kaggle_dino_siglip.yaml`.

## Core Components

The main model is implemented in `src/biomass/model.py`. It keeps the target dependencies explicit:

```python
green = self.head_green(pooled)
dead = self.head_dead(pooled)
clover = self.head_clover(pooled)
gdm = green + clover
total = gdm + dead
```

This structure makes the model respect the biomass hierarchy instead of learning five independent regressions.

## Notes

Large competition files, checkpoints, and raw datasets are intentionally excluded. The repository contains the solution code, configuration, and reproducible training / inference entry points.
