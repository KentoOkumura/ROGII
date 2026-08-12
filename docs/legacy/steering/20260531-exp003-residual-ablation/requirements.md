# 要件

## 依頼

`exp002_drift_minimal` をベースラインにして、`KAGGLE_DIRECTION.md` のアイデアバックログ先頭にある residual model の sampling / shrink / feature ablation を実装する。

## 制約

- 検証は exp002 と同じ `well_id` GroupKFold、`TVT_input` が NaN の evaluation zone のみで RMSE を計算する。
- control は exp002 と同じ `max_train_rows_per_fold=300000`、`max_train_rows_per_well=800`、`residual_shrink=0.85`、full feature set とする。
- 変更は one-at-a-time に限定し、sampling cap、residual shrink、feature set を同時に変えない。
- 特徴量は hidden test で利用できる horizontal well 列と既知 `TVT_input` prefix のみに限定する。
- train-only formation columns (`ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA`) は直接使わない。
- Kaggle Notebook 実行を正とし、ローカル notebook 実行は明示依頼なしでは行わない。

## 受け入れ基準

- `experiments/exp003_residual_ablation` が `experiments/exp002_drift_minimal` を source として作成され、notebook 名、config、記録ファイルが exp003 を指す。
- `config.yaml` に ablation variants があり、control、`max_train_rows_per_well`、total sampling cap、`residual_shrink`、feature set を同じ CV runner で比較できる。
- train notebook が variant 別の `artifacts/ablation_metrics.csv`、`fold_metrics.csv`、`well_metrics.csv`、`fold_model_training.csv` と `metrics.json` を生成する。
- inference notebook が `ablation.selected_variant` を適用して最終モデルを fit し、提出 CSV を生成できる。
- `uv run python scripts/validate_experiment.py --experiment exp003_residual_ablation` と Kaggle notebook prepare が通る。
