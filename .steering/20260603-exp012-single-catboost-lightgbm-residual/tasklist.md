# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- `.steering/20260603-exp012-single-catboost-lightgbm-residual/` を作成。
- `experiments/exp012_single_catboost_lightgbm_residual/` を `exp003_residual_ablation` から作成。
- notebook 名と `settings.py` の実験名を exp012 に更新。
- `config.yaml` に HGB control、LightGBM、CatBoost の 6 variants を定義。
- `baseline.py` に optional LightGBM / CatBoost model factory を追加。
- train notebook に model-class CV と group summary output を追加。
- inference notebook を selected variant の estimator 表示に対応。
- `validate_experiment`、ruff、py_compile、train / inference notebook preparation、pytest を通過。
- `experiment_summary.md` に exp012 を追加。
- Kaggle train kernel version 1 を実行し、full CV を取得。
- `lightgbm_no_gr` が best CV 13.549257 だったため、`ablation.selected_variant` を更新。
- Kaggle output の metrics、artifacts、train log を実験ディレクトリに反映。
- `lightgbm_no_gr` の inference kernel version 1 を実行。
- `submission.csv` を取得し、submit-check PASS を確認。
- submission ref `53330920` を提出し、Public LB 12.320 を記録。
