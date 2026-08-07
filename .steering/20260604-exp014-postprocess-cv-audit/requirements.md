# 要件

## 依頼

`KAGGLE_DIRECTION.md` のアイデアバックログ最上位
`exp014_postprocess_cv_audit` を実装する。

`exp013_model_diversity_or_postprocess` の `distance_bucket_shrink_fit`
は同じ OOF rows で bucket alpha を fit/evaluate しているため、
raw clean CV 13.549257 と OOF-fit postprocess score 13.501824 を分離する。
既存の `row_oof_predictions.csv` を使い、bucket alpha が fold 外でも
改善するかを監査する。

## 制約

- `exp013` の提出済み inference は Public LB anchor として維持し、再提出はしない。
- 監査は既存 OOF artifact のみを使い、新しい model training は行わない。
- `TVT_input` 評価区間の正解は alpha 監査にだけ使い、inference feature にはしない。
- 監査結果では raw clean CV、同一 OOF fit score、fold 外 alpha score を別項目として記録する。
- 1.1GB の OOF CSV は実験ディレクトリに常設せず、`/tmp/kaggle-output/...` を入力パスとして参照する。

## 受け入れ基準

- `experiments/exp014_postprocess_cv_audit/` に監査用 config、README、SESSION_NOTES、result、metrics がある。
- `audit_postprocess_cv.py` が `exp013` の OOF CSV から、raw、同一 OOF bucket fit、leave-one-fold-out bucket fit、well-holdout bucket fit の RMSE を出力する。
- artifact として metrics CSV、alpha CSV、bucket summary、JSON summary が保存される。
- 監査結果に基づき、`experiment_summary.md` と `KAGGLE_DIRECTION.md` で `exp013` の CV 表記を raw clean CV と OOF-fit score に分けて扱う。
- `validate_experiment.py`、ruff、py_compile が通る。
