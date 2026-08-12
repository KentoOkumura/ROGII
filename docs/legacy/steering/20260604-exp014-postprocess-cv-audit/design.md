# 設計

## アプローチ

`exp013` train output の `row_oof_predictions.csv` から
`lightgbm_no_gr` の OOF rows だけを読み、以下の集計を行う。

- raw: `y_pred` をそのまま score する clean OOF CV。
- in-sample bucket fit: `exp013` と同じく、全 OOF rows で bucket ごとに
  `last_anchor + alpha * (y_pred - last_anchor)` の alpha を fit して score する。
- leave-one-fold-out bucket fit: original CV fold を 1 fold holdout し、
  残り fold で bucket alpha を fit して holdout fold を score する。
- well-bucket holdout fit: stable hash で well を audit fold に分け、
  bucket 内で held-out well group に対する alpha 汎化を確認する。

alpha は closed form で推定する。

```text
alpha = sum(pred_residual * true_residual) / sum(pred_residual^2)
```

推定後は `config.yaml` の `alpha_clip` 範囲に clip する。
RMSE は `sum(true_residual^2) - 2 alpha sum(pred_residual true_residual)
+ alpha^2 sum(pred_residual^2)` から計算し、大きな OOF CSV を全行保持しない。

## 実験範囲

- 対象実験: `exp014_postprocess_cv_audit`
- 親実験: `exp013_model_diversity_or_postprocess`
- 変更する変数: postprocess alpha の評価方法と記録。
- 固定する変数: `exp013` の OOF predictions、distance buckets、alpha clip、raw LightGBM no-GR anchor。

## リスク

- OOF artifact が `/tmp` から消えている場合は Kaggle output の再取得が必要。
- leave-one-fold-out は alpha tuning の held-out 評価だが、元 OOF prediction 自体は同じ GroupKFold で作成されたものなので、モデル汎化の新規 CV ではない。
- well-bucket holdout は alpha 安定性の stress test であり、Public LB を直接保証しない。
