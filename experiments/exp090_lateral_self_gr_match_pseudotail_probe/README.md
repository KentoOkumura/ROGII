# exp090_lateral_self_gr_match_pseudotail_probe

## Status

Kaggle train v1 completed. Weak positive CV result; inference is not selected.

## Hypothesis

同一 horizontal well の既知 prefix 終端側 GR と評価区間 GR が似る row では、過去 prefix 上の局所的な TVT offset や match confidence が exp073 の残差予測に効く可能性がある。exp008/017/042 で外部 typewell NCC や DTW/DWT add-only は悪化済みなので、この実験では raw GR 波形や直接置換ではなく、同一 well 内の target-free match summary だけを exp073 feature surface に追加する。

## Validation Strategy

exp072 deterministic full replay train feature cache と exp073 LightGBM config family を固定し、GroupKFold by well で base 196 features と self-GR match feature variants を比較する。評価 RMSE は通常の非加重 RMSE とし、distance bucket、tail rank bucket、worst-well の悪化を必ず見る。

## Scope

- Route: `ml_model`
- Parent: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- Cache parent: `exp072_exp063_full_replay_feature_cache`
- Base features: exp073 full replay 196 features
- Added features: same-well prefix/evaluation GR NCC score、matched prefix TVT offset、per-scale L2、GR missingness/length context
- Inference: train-side ablation result review まで未選択

## Variants

- `control_exp073_base196`
- `self_gr_core`
- `self_gr_core_multiscale`
- `self_gr_core_context`

## Findings

Best variant is `self_gr_core_multiscale`, CV 9.516732864806912 versus `control_exp073_base196` CV 9.526290307637334. The improvement is small (-0.009557 RMSE) and well-level deltas are mixed, so this is not an inference candidate yet.

## Expected Outputs

- `exp090_lateral_self_gr_match_pseudotail_probe_metrics.csv`
- `exp090_lateral_self_gr_match_pseudotail_probe_by_well.csv`
- `exp090_lateral_self_gr_match_pseudotail_probe_bucket_metrics.csv`
- `exp090_lateral_self_gr_match_pseudotail_probe_self_gr_feature_summary.csv`
- `exp090_lateral_self_gr_match_pseudotail_probe_sample_weight_summary.csv`
- `exp090_lateral_self_gr_match_pseudotail_probe_feature_importance.csv`
- `exp090_lateral_self_gr_match_pseudotail_probe_feature_importance_mean.csv`
- `exp090_lateral_self_gr_match_pseudotail_probe_feature_importance_mean_top.png`
- `exp090_lateral_self_gr_match_pseudotail_probe_predictions.csv.gz`
- `exp090_lateral_self_gr_match_pseudotail_probe_feature_schema.csv`
- `exp090_lateral_self_gr_match_pseudotail_probe_lgb_models/manifest.json`
- `exp090_lateral_self_gr_match_pseudotail_probe_summary.json`
