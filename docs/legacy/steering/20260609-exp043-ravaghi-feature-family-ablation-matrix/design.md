# 設計

## アプローチ

`exp042` の audit script を土台に、`exp038` から `exp042` で個別に見た
Ravaghi 由来 family を同じ train well の途中以降を隠した疑似 test 生成物 上で横並びする。
LightGBM residual model の feature family ablation に限定し、raw と
fixed distance bucket shrink の両方を記録する。

## 実験範囲

- 対象実験: `exp043_ravaghi_feature_family_ablation_matrix`
- Route: `ml_model`
- 親実験: `exp042_ravaghi_ncc_gr_match_features`
- 変更する変数:
  - `pf_prediction`
  - `pf_uncertainty`
  - `public_beam_aggregate`
  - `beam_exact_paths`
  - `beam_exact_diagnostics`
  - `beam_exact_disagreement`
  - `ncc_paths`
  - `ncc_scores`
  - `ncc_disagreement`
  - `gr_match_offsets`
  - `spatial_formation_proxy`
- 固定する変数:
  - input rows: `exp029` train well の途中以降を隠した疑似 test rows
  - model class / params: single `LGBMRegressor`
  - split surfaces: leave-one-original-fold-out and well-hash holdout
  - postprocess: raw and `exp014_bucket_shrink_params`
  - report controls: `last_anchor`, `public_pf_selector`, `pf090_hold010`, `beam`

## 出力

- `single_lgbm_metrics.csv`
- `single_lgbm_bucket_metrics.csv`
- `single_lgbm_split_metrics.csv`
- `single_lgbm_well_metrics.csv`
- `single_lgbm_feature_importance.csv`
- `single_lgbm_train_summary.csv`
- `single_lgbm_family_matrix.csv`
- `single_lgbm_summary.json` / `metrics.json`

## リスク

- リークリスク: train-only formation columns や target error diagnostics を feature に混ぜると不正な改善になる。config と script で除外する。
- CV/LB 不一致リスク: train well の途中以降を隠した疑似 test rows は hidden test を完全再現しない。original-fold と well-hash の両方で改善した候補だけ supported とする。
- ランタイム/メモリリスク: exact beam と NCC/GR regeneration は重い。Kaggle full run 前に `--max-wells` smoke を行う。
