# 要件

## 依頼

`lgbm_capacity_pseudotail_inference_submit` を `exp052_lgbm_capacity_pseudotail_inference_submit` として実装し、Kaggle inference で submission を生成する。

## 制約

- Route: `ml_model`
- 親実験は `exp051_pseudo_tail_lgbm_param_micro_tune` とする。
- 実装親は `exp050_xgboost_pseudo_tail_inference_submit` とし、inference flow は維持する。
- final residual estimator だけを `LGBMRegressor` に戻し、exp051 best の `num_leaves=47`、`min_child_samples=60` を使う。
- pseudo-tail cutoff、distance-balanced sampling、no-GR feature set、residual shrink、fixed `exp014_bucket_shrink_params` は exp026/exp050 と同じにする。
- Public LB は提出まで未確認として扱い、exp027 全体 / PF route 基準とは混ぜない。

## 受け入れ基準

- Kaggle inference notebook package を `run_on_push=true` で作成できる。
- Kaggle inference output に `submission.csv` が生成される。
- `submit-check` が PASS する。
- submission の行数、欠損、重複、予測範囲、exp026/exp050 との差分を記録する。
- 提出した場合は ref と Public LB を `SESSION_NOTES.md`、`result.md`、`metrics.json`、`experiment_summary.md`、`SUBMISSIONS.md` に記録する。
