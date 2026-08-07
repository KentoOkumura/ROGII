# exp085_u_projection_feature_ablation

## Status

Implemented, not run.

## Hypothesis

`exp080_u_space_target_ablation` では U-space を target として直接学習すると悪化した。一方で PF/Beam/likelihood-PF の候補軌跡を `U = TVT + Z - (T0 + Z0)` 空間に写し、smooth projection からのズレや候補間 disagreement として渡すなら、target を壊さずに信頼度・形状情報だけを exp073 系 ML に追加できる。

## Validation Strategy

exp072 deterministic full replay train feature cache と exp073 LightGBM config family を固定する。target は `TVT - last_known_tvt` のまま、base 196 features に U-space projection feature group を add-only して GroupKFold by well で比較する。

## Scope

- Route: `ml_model`
- Parent: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- Cache parent: `exp072_exp063_full_replay_feature_cache`
- Base features: exp073 full replay 196 features
- Added features: PF/Beam/likelihood-PF の U-space robust polynomial correction、residual、residual MAD、candidate disagreement
- LGB OOF U-space features: 初期実装では leakage 回避のため無効
- Inference: train-side ablation の結果確認まで未選択

## Variants

- `control_exp073_base196`
- `u_projection_correction`
- `u_disagreement`
- `u_projection_correction_plus_disagreement`

## Findings

No run yet. Implementation and validation scaffolding are in place.

## Expected Outputs

- `exp085_u_projection_feature_ablation_metrics.csv`
- `exp085_u_projection_feature_ablation_by_well.csv`
- `exp085_u_projection_feature_ablation_bucket_metrics.csv`
- `exp085_u_projection_feature_ablation_projection_feature_summary.csv`
- `exp085_u_projection_feature_ablation_feature_importance.csv`
- `exp085_u_projection_feature_ablation_feature_importance_mean.csv`
- `exp085_u_projection_feature_ablation_feature_importance_mean_top.png`
- `exp085_u_projection_feature_ablation_predictions.csv.gz`
- `exp085_u_projection_feature_ablation_feature_schema.csv`
- `exp085_u_projection_feature_ablation_lgb_models/manifest.json`
- `exp085_u_projection_feature_ablation_summary.json`
