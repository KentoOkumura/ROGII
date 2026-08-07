# exp117_linear_md_z_prior_residual_target

## Status

Completed train-side audit. Rejected.

## Hypothesis

`dTVT = TVT - T0` is strong, but it assumes a flat prior after the last known prefix row. A weak linear MD/Z prior may remove a small physical drift component before LightGBM learns the residual, without using validation-tail true TVT.

## Validation Strategy

Use the exp072 deterministic full replay train feature cache and the exp073 LightGBM config family. Keep folds, rows, features, and model config fixed. Compare only target definitions, then convert every OOF prediction back to TVT space for pooled RMSE, well RMSE, distance bucket, tail rank bucket, target distribution, and SHA records.

## Scope

- Route: `ml_model`
- Parent: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- Cache parent: `exp072_exp063_full_replay_feature_cache`
- Diagnostic parent: `exp113_linear_md_z_prior_global_search`
- Features: exp073 full replay 196 features
- Targets: `dTVT`, `linear_prior_a0p02_bm0p25`, `linear_prior_a0p02_bm0p50`, `linear_prior_a0p04_bm0p25`
- Models: `lgb0` only for the first ablation
- Inference: intentionally not selected until train-side results are reviewed

## Findings

Kaggle train v1 completed with `lgb0` only.

| target | pooled RMSE |
| --- | ---: |
| `dTVT` | 9.664291 |
| `linear_prior_a0p02_bm0p25` | 11.061642 |
| `linear_prior_a0p02_bm0p50` | 12.515352 |
| `linear_prior_a0p04_bm0p25` | 11.079209 |

The weak prior targets improve some near-prefix distance buckets, but they regress 1000+ ft and most wells. Keep `dTVT`; do not inference-port this experiment.

## Expected Outputs

- `exp117_linear_md_z_prior_residual_target_metrics.csv`
- `exp117_linear_md_z_prior_residual_target_by_well.csv`
- `exp117_linear_md_z_prior_residual_target_bucket_metrics.csv`
- `exp117_linear_md_z_prior_residual_target_target_summary.csv`
- `exp117_linear_md_z_prior_residual_target_predictions.csv.gz`
- `exp117_linear_md_z_prior_residual_target_feature_schema.csv`
- `exp117_linear_md_z_prior_residual_target_lgb_models/manifest.json`
- `exp117_linear_md_z_prior_residual_target_summary.json`
