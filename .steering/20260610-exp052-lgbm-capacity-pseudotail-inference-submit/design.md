# 設計

## アプローチ

`exp050_xgboost_pseudo_tail_inference_submit` をコピーし、Kaggle inference flow、submission 生成、fixed bucket-shrink postprocess はそのまま使う。`config.yaml` の estimator と params を exp051 best に合わせる。

## 実験範囲

- 対象実験: `exp052_lgbm_capacity_pseudotail_inference_submit`
- Route: `ml_model`
- 親実験: `exp051_pseudo_tail_lgbm_param_micro_tune`
- 実装親: `exp050_xgboost_pseudo_tail_inference_submit`
- 変更する変数: final residual estimator を `LGBMRegressor` にし、`num_leaves=47`、`min_child_samples=60` を使う。
- 固定する変数: pseudo-tail 3 cutoffs、distance-balanced sampling、row caps、no-GR feature set、residual shrink、fixed `exp014_bucket_shrink_params`、Kaggle offline/inference flow

## リスク

- リークリスク: final inference fit は official train wells のみを使い、test well では known `TVT_input` prefix だけを使う。
- CV/LB 不一致リスク: exp051 CV は改善したが、exp021/026/050 のように Public LB に転移しない可能性がある。
- ランタイム/メモリリスク: exp026/050 と同じ full train fit で、LightGBM capacity 増加だけなので許容範囲と見込む。
