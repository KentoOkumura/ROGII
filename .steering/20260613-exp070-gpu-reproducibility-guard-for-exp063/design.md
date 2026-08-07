# exp070_gpu_reproducibility_guard_for_exp063 Design

## Route

`ml_model`

## Parent

`exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit`

## Design

The train notebook loads exp063's saved tracker feature frame and uses it as the fixed feature surface:

- `id`
- `well`
- `target`
- `last_known_tvt`
- PF/Beam/likelihood-PF tracker columns saved by exp063

The model target is `target`, the residual `TVT - last_known_tvt`. Predictions are converted back to TVT as `last_known_tvt + pred_delta`.

The implementation trains the exp063 public Pixiux LightGBM config family (`lgb0`, `lgb1`, `lgb2`) under reproducibility-focused modes:

- `gpu_repro_guard_dp_threads8`: GPU, `gpu_use_dp=true`, fixed `num_threads=8`, deterministic flags.
- `cpu_deterministic_threads8`: CPU, `deterministic=true`, `force_col_wise=true`, fixed `num_threads=8`.
- `exp063_gpu_float32_reference`: optional exp063-like GPU float32 reference mode.

Each mode uses 5-fold GroupKFold by `well`, saves OOF predictions and boosters, and writes stable hashes for predictions and model files.

## Expected Artifacts

- `exp063_repro_guard_metrics.csv`
- `exp063_repro_guard_by_well.csv`
- `exp063_repro_guard_predictions.csv.gz`
- `exp063_repro_guard_feature_schema.csv`
- `exp063_repro_guard_summary.json`
- `exp063_repro_guard_lgb_models/manifest.json`

## Interpretation

A mode is reproducibility-safe only if two independent Kaggle runs of the same package produce matching OOF prediction SHA and matching model file hashes, or if observed differences are small enough to define an operational tolerance.
