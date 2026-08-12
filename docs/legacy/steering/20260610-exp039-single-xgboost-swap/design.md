# 設計

## アプローチ

`exp038` の single LightGBM feature ablation 実装をコピーし、candidate model の `make_estimator` に `XGBRegressor` を追加する。評価は `leave_one_original_fold_out` と `well_hash_holdout` の 2 系統を維持し、各 feature variant について raw prediction と `exp014_bucket_shrink_params` 適用後を記録する。

比較対象は同一実験内の `base_geometry_bucket_shrink` と、report control の `public_pf_selector` / `pf090_hold010` / `beam` / `exp026_regenerated_bucket_shrink`。supported candidate は `base_geometry_bucket_shrink` を original-fold と well-hash の両方で上回る bucket-shrink candidate とする。

## 実験範囲

- 対象実験: `exp039_single_xgboost_swap`
- Route: `ml_model`
- 親実験: `exp038_ravaghi_public_sel15_features_single_lgbm`
- 推論 anchor: `exp039_ravaghi_single_lgbm_inference_submit`
- 変更する変数: candidate residual estimator とその params (`LGBMRegressor` -> `XGBRegressor`)
- 固定する変数: feature surface、feature variants、target、base prediction、residual shrink、max residual clip、row sampling caps、fixed bucket-shrink、split definitions、exp026 regenerated control

## リスク

- リークリスク: `target_tvt` は label/scoring のみ、`pf_error` / `beam_error` / exp026 bridge columns は feature から除外する。well-level split を維持する。
- CV/LB 不一致リスク: exp029 疑似 test surface は hidden Public LB と一致しない可能性が高い。supported candidate が出ても、別の inference-port 実験で output diff / submit-check / code-submit を確認する。
- ランタイム/メモリリスク: XGBoost は LGBM より重い可能性があるため `tree_method: hist`、`n_jobs: 2`、既存 row caps を維持する。
