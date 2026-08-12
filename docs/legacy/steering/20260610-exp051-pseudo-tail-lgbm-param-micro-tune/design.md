# 設計

## アプローチ

`exp049` の train-side diagnostic 実装をコピーし、raw 予測と固定 `exp014_bucket_shrink_params` 後の候補を同時に集計する仕組みを使う。estimator は `LGBMRegressor` に戻し、`audit.training_variants.variants[].model_params` と row cap override を training loop で読み取る。

比較候補は control を含めて次に限定する。

- `lgbm_control_pseudo_tail_3_cutoffs_distance_balanced`
- `lgbm_regularized_leaves23_minchild120`
- `lgbm_capacity_leaves47_minchild60`
- `lgbm_subsample080_colsample085`
- `lgbm_reglambda050`
- `lgbm_rowcap700_perwell`
- `lgbm_rowcap900_perwell`

## 実験範囲

- 対象実験: `exp051_pseudo_tail_lgbm_param_micro_tune`
- Route: `ml_model`
- 親実験: `exp026_pseudo_tail_bucket_shrink_inference_submit`
- 実装親: `exp049_xgboost_pseudo_tail_residual`
- 変更する変数: LightGBM の小さな parameter override、per-well training row cap
- 固定する変数: pseudo-tail cutoff quantiles `[0.45, 0.65, 0.82]`、distance-balanced sampling cap 60000、feature set、residual shrink、fixed bucket-shrink coefficients、GroupKFold-by-well

## リスク

- リークリスク: pseudo cutoffs は train fold wells 内だけで生成し、valid fold は本来の `TVT_input` NaN evaluation zone のみで評価する。
- CV/LB 不一致リスク: exp021/026 で CV 改善が Public LB に転移しなかったため、主評価で fold / distance bucket 横断の安定性を確認してから次へ進む。
- ランタイム/メモリリスク: 7 variants x 5 folds の LightGBM fit になる。Kaggle train でまず実行し、推論 port は選択候補が明確な場合だけ別実験にする。
