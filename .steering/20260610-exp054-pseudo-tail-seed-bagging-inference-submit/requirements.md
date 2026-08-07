# 要件

## 依頼

seed 変更による LB 影響を早めに確認するため、`exp053_pseudo_tail_seed_bagging` の 3-seed bagging を推論へ移植し、Kaggle code submit まで行う。

## 制約

- Route: `ml_model`
- 親実験は `exp053_pseudo_tail_seed_bagging` とする。
- 実装親は `exp052_lgbm_capacity_pseudotail_inference_submit` とする。
- `exp052` の inference flow、fixed `exp014_bucket_shrink_params`、no-GR feature set、LightGBM capacity params は維持する。
- 変更は final residual model を member seeds `[42, 314, 2027]` の 3 本 fit し、raw prediction 平均にする点だけに限定する。
- Kaggle Notebook は offline / CPU / `run_on_push=true` とする。

## 受け入れ基準

- Kaggle inference notebook package を作成できる。
- Kaggle inference output に `submission.csv` が生成される。
- `submit-check` が PASS する。
- exp052 submission との差分、予測範囲、行数、欠損、重複を記録する。
- code submit の ref と Public LB を `SESSION_NOTES.md`、`result.md`、`metrics.json`、`experiment_summary.md`、`submissions/SUBMISSIONS.md` に記録する。
