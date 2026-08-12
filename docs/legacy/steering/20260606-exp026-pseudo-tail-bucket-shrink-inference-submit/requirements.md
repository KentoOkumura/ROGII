# 要件

## 依頼

`exp025_pseudo_tail_postprocess_cv_audit` の selected fixed `exp014_bucket_shrink_params` を inference / submit 候補化する。

## 制約

- exp024 の selected pseudo-tail training recipe を固定する。
- 変更は postprocess を `raw` から fixed `distance_bucket_shrink` に切り替えることに限定する。
- same-OOF alpha fit は使わない。
- Kaggle Notebook 実行を正とし、ローカル inference notebook 実行はしない。
- `submission.csv` は Kaggle output として取得し、sample submission 互換性を確認する。

## 受け入れ基準

- exp026 の steering docs と実験ディレクトリがある。
- inference notebook が exp025 selected fixed bucket shrink を使う。
- Kaggle 用 inference notebook が strict prepare できる。
- Kaggle output の `submission.csv` が submit-check を通る。
- 結果が `SESSION_NOTES.md`、`result.md`、`metrics.json`、`experiment_summary.md` に記録される。
