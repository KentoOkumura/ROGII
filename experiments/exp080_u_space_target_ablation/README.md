# exp080_u_space_target_ablation

## Status

Implemented, not run.

## Hypothesis

`TVT - last_known_tvt` target forces the model to learn part of the `Z`/`TVT` cancellation directly. A U-space target such as `(TVT - T0) + (Z - Z0)` or `TVT + Z - anchor` may make the surface smoother while keeping the exp073 full replay feature surface fixed.

## Validation Strategy

Use the exp072 deterministic full replay train feature cache and the exp073 LightGBM config family. Keep folds, features, rows, and model config fixed. Compare only target definitions, then convert every OOF prediction back to TVT space for pooled RMSE, well RMSE, distance bucket, tail rank bucket, target distribution, and SHA records.

## Scope

- Route: `ml_model`
- Parent: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- Cache parent: `exp072_exp063_full_replay_feature_cache`
- Features: exp073 full replay 196 features
- Targets: `dTVT`, `dTVT_plus_dZ`, `TVT_plus_Z_abs`, `TVT_plus_Z_minus_T0`, `TVT_plus_Z_minus_T0Z0`
- Inference: intentionally not selected until train-side ablation results are reviewed

## Findings

No run yet. Implementation and validation scaffolding are in place.

## Expected Outputs

- `exp080_u_space_target_ablation_metrics.csv`
- `exp080_u_space_target_ablation_by_well.csv`
- `exp080_u_space_target_ablation_bucket_metrics.csv`
- `exp080_u_space_target_ablation_target_summary.csv`
- `exp080_u_space_target_ablation_predictions.csv.gz`
- `exp080_u_space_target_ablation_feature_schema.csv`
- `exp080_u_space_target_ablation_lgb_models/manifest.json`
- `exp080_u_space_target_ablation_summary.json`
