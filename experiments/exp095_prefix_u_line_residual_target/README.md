# exp095_prefix_u_line_residual_target

## Status

Completed train-side audit. Rejected.

## Hypothesis

exp080 rejected raw U-space supervised targets, but the failure may be caused by well-specific U-space offset and slope remaining in the target. Fitting a robust `U_alpha = TVT + alpha * Z` line on known-prefix rows only and learning the residual may keep the useful trajectory de-trending while avoiding absolute U-space scale leakage.

## Validation Strategy

Use the exp072 deterministic full replay train feature cache and the exp073 LightGBM config family. Keep folds, features, rows, and model config fixed. Compare only target definitions, then convert every OOF prediction back to TVT space for pooled RMSE, well RMSE, distance bucket, tail rank bucket, target distribution, prefix-line fit diagnostics, and SHA records.

## Scope

- Route: `ml_model`
- Parent: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- Cache parent: `exp072_exp063_full_replay_feature_cache`
- Features: exp073 full replay 196 features
- Targets: `dTVT`, `prefix_u_line_alpha1p0`, `prefix_u_line_alpha0p5`
- Models: `lgb0` only for the first ablation
- Inference: intentionally not selected until train-side ablation results are reviewed

## Findings

Kaggle train v1 completed with `lgb0` only.

| target | pooled RMSE |
| --- | ---: |
| `dTVT` | 9.664067 |
| `prefix_u_line_alpha0p5` | 28.087914 |
| `prefix_u_line_alpha1p0` | 33.478794 |

Prefix-line fitting itself was mechanically valid: no fallback wells and exact `T0` / `last_known_tvt` agreement. The target definition is the problem. Do not inference-port this experiment.

## Expected Outputs

- `exp095_prefix_u_line_residual_target_metrics.csv`
- `exp095_prefix_u_line_residual_target_by_well.csv`
- `exp095_prefix_u_line_residual_target_bucket_metrics.csv`
- `exp095_prefix_u_line_residual_target_target_summary.csv`
- `exp095_prefix_u_line_residual_target_predictions.csv.gz`
- `exp095_prefix_u_line_residual_target_feature_schema.csv`
- `exp095_prefix_u_line_residual_target_lgb_models/manifest.json`
- `exp095_prefix_u_line_residual_target_summary.json`
