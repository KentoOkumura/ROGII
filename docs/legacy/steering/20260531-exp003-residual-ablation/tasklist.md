# タスクリスト

## TODO

- exp002 と exp003 の CV/LB 逆転原因を確認する。

## 進行中

- なし

## ブロック中

- なし

## 完了

- `exp002_drift_minimal` を source に `exp003_residual_ablation` を作成。
- exp003 用の steering docs を作成。
- `config.yaml` に one-at-a-time ablation variants を追加。
- `baseline.py` に `model.feature_set` による active feature column 切替を追加。
- train notebook を variant 別 CV runner に更新。
- inference notebook を `ablation.selected_variant` 適用に対応。
- `uv run python scripts/validate_experiment.py --experiment exp003_residual_ablation` を通過。
- `uv run ruff check experiments/exp003_residual_ablation/baseline.py` を通過。
- `uv run pytest` を通過。
- Kaggle train/inference notebook prepare を title 付き strict で通過。
- Kaggle train full CV を完了し、`artifacts/ablation_metrics.csv` と `metrics.json` を取得。
- `feature_no_gr_signal` を selected variant に設定。
- 結果を `SESSION_NOTES.md`、`result.md`、`experiment_summary.md`、`KAGGLE_DIRECTION.md` に反映。
- `feature_no_gr_signal` で inference notebook を実行。
- `submission.csv` の形式確認を PASS。
- ref `53213975` を提出し、public LB 12.852 を取得。
- exp002 public LB 12.533 より悪化したことを記録。
