# exp014_postprocess_cv_audit セッションノート

## 目的

`exp013_model_diversity_or_postprocess` の `distance_bucket_shrink_fit`
を監査し、raw clean CV、同一 OOF fit score、held-out alpha score を分離する。

## 現在の状態

- 状態: 完了
- 親実験: `exp013_model_diversity_or_postprocess`
- Raw clean CV: 13.549257
- Same-OOF bucket fit: 13.501824
- Leave-one-original-fold-out bucket fit: 13.535596
- Well-bucket holdout fit: 13.510690
- Public LB: なし。提出済み anchor は `exp013` の 12.271。

## コマンドログ

- 2026-06-04: `uv run python scripts/new_steering.py --experiment exp014_postprocess_cv_audit` で steering docs を作成。
- 2026-06-04: `uv run python scripts/new_experiment.py --name exp014_postprocess_cv_audit` で実験を作成。
- 2026-06-04: `.steering/20260604-exp014-postprocess-cv-audit/{requirements.md,design.md,tasklist.md}` を記入。
- 2026-06-04: `config.yaml` を OOF artifact audit 用に更新。
- 2026-06-04: `audit_postprocess_cv.py` を追加。
- 2026-06-04: `uv run python experiments/exp014_postprocess_cv_audit/audit_postprocess_cv.py` を実行し、artifacts と `metrics.json` を生成。
- 2026-06-04: `uv run python scripts/record_experiment.py --experiment exp014_postprocess_cv_audit --status completed --cv 13.535596 ...` で summary 用 metrics を記録。

## 変更点

- 1.1GB の `row_oof_predictions.csv` は `data/external/kaggle-output/exp013_model_diversity_or_postprocess/train/artifacts/` から読み、実験ディレクトリには常設しない。
- closed-form alpha と集計 SSE で RMSE を計算し、OOF 全体をメモリに保持しない。
- original fold holdout と stable well hash holdout の 2 通りで bucket alpha の汎化を監査した。

## Artifacts

- `artifacts/postprocess_cv_audit_metrics.csv`
- `artifacts/postprocess_cv_audit_alphas.csv`
- `artifacts/postprocess_cv_audit_bucket_summary.csv`
- `artifacts/postprocess_cv_audit_summary.json`
- `metrics.json`

## 次のアクション

1. `exp013` の Public LB 12.271 は維持し、CV 表記は raw 13.549257、same-OOF 13.501824、held-out 13.535596 に分ける。
2. 次の実験は `public_pf_beam_scale_selector_features` を優先し、PF / beam 候補も raw / held-out postprocess を分離して比較する。
