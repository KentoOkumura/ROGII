# 要件

## 依頼

`exp049_xgboost_pseudo_tail_residual` の XGBoost pseudo-tail fixed bucket-shrink 候補を inference flow に移植し、Kaggle Notebook output の `submission.csv` を生成する。

## 制約

- Route: `ml_model`
- `exp026` の inference flow、pseudo-tail cutoff、distance-balanced sampling、fixed bucket-shrink 係数を維持する。
- 残差モデルだけを `exp049` と同じ `XGBRegressor` に変更する。
- Final fit は official train wells のみを使う。
- Test predictions は各 test well の既知 `TVT_input` prefix だけを使う。
- Train-only formation columns は使わない。
- Competition submit はユーザー確認後に限定する。

## 受け入れ基準

- `experiments/exp050_xgboost_pseudo_tail_inference_submit/` に self-contained な実験がある。
- inference notebook が Kaggle 上で `submission.csv` を生成する。
- `submission.csv` が sample submission と互換で、欠損/重複がない。
- 予測範囲、行数、exp026 submission との差分を記録する。
- `SESSION_NOTES.md`、`result.md`、`metrics.json`、`experiment_summary.md` を更新する。
