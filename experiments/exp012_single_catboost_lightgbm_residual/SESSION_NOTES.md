# exp012_single_catboost_lightgbm_residual セッションノート

## 目的

`exp003_residual_ablation` の residual pipeline を固定し、HGB control に対して CatBoost / LightGBM 単体モデルを比較する。

## 現在の状態

- 状態: inference / submit 完了
- 親実験: `exp003_residual_ablation`
- CV anchor: `exp003_residual_ablation` no-GR HGB 13.882944
- Public LB anchor: `exp012_single_catboost_lightgbm_residual` lightgbm_no_gr 12.320
- selected variant: `lightgbm_no_gr`
- CV: 13.549257
- Public LB: 12.320 (`ref=53330920`)

## コマンドログ

- 2026-06-03: `task new-steering EXP=exp012_single_catboost_lightgbm_residual` は `task` 未インストールで失敗。
- 2026-06-03: `uv run python scripts/new_steering.py --experiment exp012_single_catboost_lightgbm_residual` で steering docs を作成。
- 2026-06-03: `uv run python scripts/new_experiment.py --name exp012_single_catboost_lightgbm_residual --source experiments/exp003_residual_ablation` で exp003 から実験を作成。
- 2026-06-03: notebook 名を exp012 に変更し、`settings.py` の `EXPERIMENT_NAME` を更新。
- 2026-06-03: `config.yaml` を exp012 用に置換し、HGB control / LightGBM / CatBoost の 6 variants を定義。
- 2026-06-03: `baseline.py` の `make_drift_model` を `model.drift_model.estimator` 切り替えに対応。
- 2026-06-03: train notebook に estimator logging と `model_group_summary.csv` 出力を追加。
- 2026-06-03: inference notebook に selected estimator logging を追加。
- 2026-06-03: `uv run python scripts/validate_experiment.py --experiment exp012_single_catboost_lightgbm_residual` が通過。
- 2026-06-03: `uv run ruff check experiments/exp012_single_catboost_lightgbm_residual/baseline.py experiments/exp012_single_catboost_lightgbm_residual/settings.py` が通過。
- 2026-06-03: `uv run python -m py_compile experiments/exp012_single_catboost_lightgbm_residual/baseline.py experiments/exp012_single_catboost_lightgbm_residual/settings.py` が通過。
- 2026-06-03: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp012_single_catboost_lightgbm_residual --notebook train --strict` が通過。
- 2026-06-03: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp012_single_catboost_lightgbm_residual --notebook inference --strict` が通過。
- 2026-06-03: `uv run pytest` が通過。9 tests passed。
- 2026-06-03: `uv run python scripts/update_experiment_summary.py` で `experiment_summary.md` に exp012 を追加。
- 2026-06-03: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp012_single_catboost_lightgbm_residual --notebook train --run-on-push --title "exp012 single catboost lightgbm residual train" --strict` で train package を再生成。
- 2026-06-03: `kaggle kernels push -p experiments/exp012_single_catboost_lightgbm_residual/kaggle/train` で version 1 を push。URL: https://www.kaggle.com/code/kentookumura/exp012-single-catboost-lightgbm-residual-train
- 2026-06-03: `kaggle kernels status kentookumura/exp012-single-catboost-lightgbm-residual-train` を監視し、`KernelWorkerStatus.COMPLETE` を確認。
- 2026-06-03: `kaggle kernels output kentookumura/exp012-single-catboost-lightgbm-residual-train -p /tmp/kaggle-output/exp012_single_catboost_lightgbm_residual/train` は初回 DNS エラーで一部取得後に失敗。escalated retry で kernel log と残り output を取得。
- 2026-06-03: Kaggle output の `metrics.json`、`artifacts/*.csv`、train log を `experiments/exp012_single_catboost_lightgbm_residual/` に反映。
- 2026-06-03: best CV の `lightgbm_no_gr` を `ablation.selected_variant` に更新。
- 2026-06-03: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp012_single_catboost_lightgbm_residual --notebook inference --run-on-push --title "exp012 lightgbm no gr inference" --strict` で inference package を再生成。
- 2026-06-03: `kaggle kernels push -p experiments/exp012_single_catboost_lightgbm_residual/kaggle/inference` で version 1 を push。metadata id と title slug の warning が出たため、実際の slug `kentookumura/exp012-lightgbm-no-gr-inference` を使って監視。
- 2026-06-03: `kaggle kernels status kentookumura/exp012-lightgbm-no-gr-inference` で `KernelWorkerStatus.COMPLETE` を確認。
- 2026-06-03: `kaggle kernels output kentookumura/exp012-lightgbm-no-gr-inference -p /tmp/kaggle-output/exp012_single_catboost_lightgbm_residual/inference` で `submission.csv` と inference log を取得。
- 2026-06-03: `uv run python .agents/skills/kaggle-submit-check/scripts/check_submission.py /tmp/kaggle-output/exp012_single_catboost_lightgbm_residual/inference/submission.csv --sample data/raw/sample_submission.csv` は PASS。14,151 rows、`id,tvt`、欠損/重複なし。
- 2026-06-03: `kaggle competitions submit rogii-wellbore-geology-prediction -k kentookumura/exp012-lightgbm-no-gr-inference -v 1 -f submission.csv -m "exp012_lightgbm_no_gr CV 13.549257"` を実行。
- 2026-06-03: submission ref `53330920` が `SubmissionStatus.COMPLETE`、Public LB 12.320 と確認。
- 2026-06-03: `uv run python scripts/record_experiment.py --experiment exp012_single_catboost_lightgbm_residual --status completed --cv 13.549257 --public-lb 12.320 ...` で metrics と summary を更新。
- 2026-06-03: `uv run python scripts/record_submission.py --experiment exp012_single_catboost_lightgbm_residual --file /tmp/kaggle-output/exp012_single_catboost_lightgbm_residual/inference/submission.csv --cv 13.549257 --public-lb 12.320 ...` で `submissions/SUBMISSIONS.md` に v007 を記録。

## 変更点

- `control_hgb_all`: exp002 all-GR HGB 設定を control として再実行。
- `control_hgb_no_gr`: exp003 no-GR HGB 設定を control として再実行。
- `lightgbm_all` / `lightgbm_no_gr`: residual model を `LGBMRegressor` に差し替え。
- `catboost_all` / `catboost_no_gr`: residual model を `CatBoostRegressor` に差し替え。
- `model_group_summary.csv`: control から作る hard/no-GR、public-like、high GR missing、long eval、steep trajectory group の variant 別 RMSE 要約。

## 次のアクション

1. Public LB 12.320 を新 anchor として、次の postprocess / model diversity 実験を設計する。
2. `lightgbm_no_gr` が public-like group で all-GR HGB より悪い点を、提出後の次実験で router / blend 候補として扱う。
