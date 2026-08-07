# 要件

## 依頼

`KAGGLE_DIRECTION.md` のアイデアバックログ先頭にある、exp003 で悪化した OOF wells の条件タグ付けと GR feature gating / shrink を実験化する。

## 制約

- 評価上の control baseline は `exp002_drift_minimal` とし、`exp003_residual_ablation` は no-GR alternate と診断 artifact のコピー元として扱う。
- 同じ well を train/valid にまたがらせない。
- gating に使う条件は、推論時にも取得できる `TVT_input` prefix、trajectory、GR の欠損率・信号形状に限定する。
- train-only formation columns は直接使わない。
- 初回 notebook 実行は Kaggle を正とし、ローカル notebook 実行は明示的な smoke debug なしには行わない。

## 受け入れ基準

- exp004 は `exp003_residual_ablation` からコピーされ、notebook 名・`EXPERIMENT_NAME`・config が `exp004_gr_gating` になっている。
- CV runner は `control_exp002_all`、`control_exp003_no_gr`、GR gating variants を同一 GroupKFold で比較できる。
- gating variants は exp002 の all-GR 予測を default とし、条件に合う well だけ no-GR 予測へ hard/soft に寄せる。
- `artifacts/well_metrics.csv` に gate weight と inference-safe condition tags が出力される。
- `scripts/validate_experiment.py`、ruff、pytest、Kaggle notebook prepare が通る。
