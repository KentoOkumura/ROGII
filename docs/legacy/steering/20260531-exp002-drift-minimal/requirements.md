# 要件

## 依頼

`exp001_baseline` を親にして、優先度高の `last_anchor` からの drift / residual 学習を最小構成で試す。

## 制約

- 検証は exp001 と同じ `well_id` GroupKFold、`TVT_input` が NaN の evaluation zone のみで RMSE を計算する。
- 特徴量は hidden test で利用できる horizontal well 列に限定する: `MD`, `X`, `Y`, `Z`, `GR`, `TVT_input` の既知 prefix。
- train-only formation columns (`ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA`) は直接使わない。
- evaluation zone の `TVT` や `TVT_input`、target 由来の未来集計を特徴に使わない。
- `exp001_baseline` の `last_anchor` を同一 CV 内で再計算し、改善または悪化を比較できるようにする。

## 受け入れ基準

- `experiments/exp002_drift_minimal` が `exp001_baseline` から作成され、notebook 名、config、記録ファイルが exp002 を指す。
- train notebook が `last_anchor` と residual model の OOF RMSE を出力し、`metrics.json` と `artifacts/well_metrics.csv` を更新する。
- inference notebook が train wells から residual model を fit し、test wells の `submission.csv` を `id,tvt` 形式で生成できる。
- `uv run python scripts/validate_experiment.py --experiment exp002_drift_minimal` と Kaggle notebook prepare が通る。
