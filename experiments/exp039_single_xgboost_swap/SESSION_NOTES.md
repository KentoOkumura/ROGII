# exp039_single_xgboost_swap セッションノート

## 状態

- status: completed
- route: `ml_model`
- parent: `exp038_ravaghi_public_sel15_features_single_lgbm`
- inference anchor: `exp039_ravaghi_single_lgbm_inference_submit`
- supporting artifact: `exp029_public_sel15_pf_oof_feature_generation`
- full Kaggle audit: completed
- local smoke: PASS with HGB override only
- Kaggle train package: prepared
- Kaggle train push: version 1 pushed

## 仮説

`exp039` の single LightGBM branch は ML route Public LB 11.740 を作ったが、`exp038` の train-side 疑似 test 評価では public PF controls には届いていない。`exp050` で exp026 pseudo-tail residual estimator の XGBoost swap が小幅に効いたため、Ravaghi/public sel15 特徴面でも estimator swap だけを切り出す。

## 実装メモ

- `exp038` をコピーして `exp039_single_xgboost_swap` を作成。
- `model.estimator` を `XGBRegressor` に変更。
- XGBoost params は `exp049` と同系統の `hist` / `max_depth=6` / `min_child_weight=8.0` / `n_estimators=900`。
- feature variants、target、residual shrink、max residual clip、row caps、bucket shrink alpha は `exp038` から固定。
- `audit.exp026_training` は regenerated control なので LGBM のまま固定。
- `ravaghi_single_xgboost_audit.py` を追加し、candidate estimator に `XGBRegressor` を追加。
- 生成物名は `single_xgboost_metrics.csv`、`single_xgboost_summary.json` などに変更。
- inference notebook は audit-only として停止し、supported candidate が出るまで提出物を作らない。

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp039_single_xgboost_swap
uv run python scripts/new_experiment.py --name exp039_single_xgboost_swap --source experiments/exp038_ravaghi_public_sel15_features_single_lgbm
uv run ruff check experiments/exp039_single_xgboost_swap/ravaghi_single_xgboost_audit.py experiments/exp039_single_xgboost_swap/settings.py
uv run python -m py_compile experiments/exp039_single_xgboost_swap/ravaghi_single_xgboost_audit.py experiments/exp039_single_xgboost_swap/settings.py
uv run python scripts/validate_experiment.py --experiment exp039_single_xgboost_swap
uv run python experiments/exp039_single_xgboost_swap/ravaghi_single_xgboost_audit.py --max-wells 5 --max-train-rows 500 --skip-exp026-control --base-estimator HistGradientBoostingRegressor --output-dir /tmp/exp039_xgb_swap_smoke
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp039_single_xgboost_swap --notebook train --kernel-id kentookumura/exp039-single-xgb-swap-train --title "exp039 single xgb swap train" --run-on-push --strict
```

Full audit 用:

```bash
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp039_single_xgboost_swap --notebook train --kernel-id kentookumura/exp039-single-xgb-swap-train --title "exp039 single xgb swap train" --run-on-push --strict
kaggle kernels push -p experiments/exp039_single_xgboost_swap/kaggle/train
kaggle kernels pull kentookumura/exp039-single-xgb-swap-train -p /tmp/kaggle-pull/exp039-single-xgb-swap-train -m
kaggle kernels logs kentookumura/exp039-single-xgb-swap-train
kaggle kernels output kentookumura/exp039-single-xgb-swap-train -p /tmp/kaggle-output/exp039_single_xgboost_swap/train_v1_probe
kaggle kernels output kentookumura/exp039-single-xgb-swap-train -p /tmp/kaggle-output/exp039_single_xgboost_swap/train_v1
cp /tmp/kaggle-output/exp039_single_xgboost_swap/train_v1/metrics.json experiments/exp039_single_xgboost_swap/metrics.json
cp /tmp/kaggle-output/exp039_single_xgboost_swap/train_v1/artifacts/single_xgboost_*.csv /tmp/kaggle-output/exp039_single_xgboost_swap/train_v1/artifacts/single_xgboost_summary.json experiments/exp039_single_xgboost_swap/artifacts/
cp /tmp/kaggle-output/exp039_single_xgboost_swap/train_v1/exp039-single-xgb-swap-train.log experiments/exp039_single_xgboost_swap/artifacts/
```

## 結果

- 実装: 完了
- `ruff check`: PASS
- `py_compile`: PASS
- `validate_experiment`: PASS
- local smoke:
  - command: `--max-wells 5 --max-train-rows 500 --skip-exp026-control --base-estimator HistGradientBoostingRegressor`
  - rows: 11,207
  - wells: 5
  - output: `/tmp/exp039_xgb_swap_smoke`
  - purpose: XGBoost が local env に無いため、feature loading / split / artifact writing path のみ HGB override で確認。
- local XGBoost fit: not run
  - reason: local env has no `xgboost` package.
- Kaggle train package:
  - path: `experiments/exp039_single_xgboost_swap/kaggle/train`
  - kernel id: `kentookumura/exp039-single-xgb-swap-train`
  - title: `exp039 single xgb swap train`
  - run_on_push: true
  - internet: false
  - source: `kentookumura/exp029-sel15-pf-oof-train`
- Kaggle train push:
  - version: 1
  - URL: `https://www.kaggle.com/code/kentookumura/exp039-single-xgb-swap-train`
  - `pull -m`: PASS
  - Kaggle `id_no`: `122273860`
  - initial `logs`: empty immediately after push
  - initial `output` probe: empty immediately after push
  - monitoring: stopped per user request; wait for user completion notice
- full Kaggle audit:
  - status: completed
  - output: `/tmp/kaggle-output/exp039_single_xgboost_swap/train_v1`
  - synced metrics: `experiments/exp039_single_xgboost_swap/metrics.json`
  - synced artifacts: `experiments/exp039_single_xgboost_swap/artifacts/single_xgboost_*`
  - rows: 1,782,279
  - wells: 773
  - selected candidate: `base_plus_pf_beam_diagnostics_bucket_shrink`
  - selected original-fold RMSE: 16.029777
  - selected well-hash RMSE: 16.028160
  - best overall control: `pf090_hold010` 15.089532
  - exp038 selected single-LGBM candidate: `base_plus_pf_prediction_bucket_shrink` original-fold 15.850147 / well-hash 15.820850
  - interpretation: XGBoost selected is better than base single-XGBoost but worse than exp038 selected LGBM and public PF controls.
- LB: 未提出

## 次のアクション

1. `result.md` / `experiment_summary.md` / `KAGGLE_DIRECTION.md` に反映する。
2. `exp039_single_xgboost_swap` は推論移植しない。
3. Ravaghi/public PF 系の次候補は、直接 model replacement ではなく confidence-only features / error map / weight 調整に限定する。
