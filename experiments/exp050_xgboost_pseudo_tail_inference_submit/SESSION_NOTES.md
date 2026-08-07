# exp050_xgboost_pseudo_tail_inference_submit セッションノート

## 目的

`exp049_xgboost_pseudo_tail_residual` で通常 CV が 12.779452 まで改善した XGBoost pseudo-tail fixed bucket-shrink 候補を、`exp026` の inference flow に移植して Kaggle output の `submission.csv` を生成し、提出前チェックする。

## 現在の状態

- status: submitted
- route: `ml_model`
- parent: `exp049_xgboost_pseudo_tail_residual`
- implementation parent: `exp026_pseudo_tail_bucket_shrink_inference_submit`
- selected training variant: `xgboost_pseudo_tail_3_cutoffs_distance_balanced`
- selected postprocess: `distance_bucket_shrink` / `exp014_bucket_shrink_params`
- CV reference: 12.779452
- exp026 CV reference: 12.870780
- exp026 Public LB reference: 12.102
- LB: Public 12.083
- submission ref: `53521999`

## 実装メモ

- `exp026` を土台に `exp050` を作成。
- `settings.py` と notebook 名を `exp050` に更新。
- `config.yaml` で `model.drift_model.estimator: XGBRegressor` に変更し、exp049 と同じ XGBoost パラメータを設定。
- `audit.training_variants.selected_variant` は `xgboost_pseudo_tail_3_cutoffs_distance_balanced` にした。
- `postprocess.selected_method` は exp026 と同じ `distance_bucket_shrink` を維持。
- `baseline.py` に `XGBRegressor` 対応を追加。
- `pseudo_tail_augmentation.py` の pseudo cutoff 生成で `TVT_input` を copy し、read-only view を避ける。
- ユーザー側で code submit 完了後、Kaggle submissions API で ref `53521999` / Public LB `12.083` を確認した。

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp050_xgboost_pseudo_tail_inference_submit
uv run python scripts/new_experiment.py --name exp050_xgboost_pseudo_tail_inference_submit --source experiments/exp026_pseudo_tail_bucket_shrink_inference_submit
uv run ruff check experiments/exp050_xgboost_pseudo_tail_inference_submit/baseline.py experiments/exp050_xgboost_pseudo_tail_inference_submit/pseudo_tail_augmentation.py experiments/exp050_xgboost_pseudo_tail_inference_submit/settings.py
uv run python -m py_compile experiments/exp050_xgboost_pseudo_tail_inference_submit/baseline.py experiments/exp050_xgboost_pseudo_tail_inference_submit/pseudo_tail_augmentation.py experiments/exp050_xgboost_pseudo_tail_inference_submit/settings.py
uv run python scripts/validate_experiment.py --experiment exp050_xgboost_pseudo_tail_inference_submit
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp050_xgboost_pseudo_tail_inference_submit --notebook inference --kernel-id kentookumura/exp050-xgb-pseudotail-infer --title "exp050 xgb pseudotail infer" --run-on-push --strict
kaggle kernels push -p experiments/exp050_xgboost_pseudo_tail_inference_submit/kaggle/inference
kaggle kernels pull kentookumura/exp050-xgb-pseudotail-infer -p /tmp/kaggle-pull/exp050-xgb-pseudotail-infer-check -m
kaggle kernels logs kentookumura/exp050-xgb-pseudotail-infer
kaggle kernels output kentookumura/exp050-xgb-pseudotail-infer -p /tmp/kaggle-output/exp050_xgboost_pseudo_tail_inference_submit/inference_v1
kaggle kernels status kentookumura/exp050-xgb-pseudotail-infer
uv run python scripts/validate_submission.py --submission /tmp/kaggle-output/exp050_xgboost_pseudo_tail_inference_submit/inference_v1/submission.csv
kaggle competitions submissions rogii-wellbore-geology-prediction
python .agents/skills/kaggle-submit-monitor/scripts/monitor_submission.py exp050_xgboost_pseudo_tail_inference_submit --competition rogii-wellbore-geology-prediction
uv run python scripts/record_submission.py --experiment exp050_xgboost_pseudo_tail_inference_submit --file /tmp/kaggle-output/exp050_xgboost_pseudo_tail_inference_submit/inference_v1/submission.csv --cv 12.779452 --public-lb 12.083 --notes "ref=53521999; kernel=kentookumura/exp050-xgb-pseudotail-infer v1; XGBoost pseudo-tail fixed bucket-shrink; improves exp026 Public LB 12.102 by -0.019; submit-check PASS"
```

## 検証状況

- Static checks: PASS
  - `ruff check`: PASS
  - `py_compile`: PASS
  - `scripts/validate_experiment.py --experiment exp050_xgboost_pseudo_tail_inference_submit`: PASS
  - notebook code compile: PASS
- Kaggle inference package: prepared
  - path: `experiments/exp050_xgboost_pseudo_tail_inference_submit/kaggle/inference`
  - kernel id: `kentookumura/exp050-xgb-pseudotail-infer`
  - title: `exp050 xgb pseudotail infer`
  - run_on_push: true
  - GPU: false
  - internet: false
- Kaggle inference:
  - pushed version 1 successfully.
  - URL: `https://www.kaggle.com/code/kentookumura/exp050-xgb-pseudotail-infer`
  - direct `kaggle kernels pull ... -m` succeeded, so the kernel exists on Kaggle.
  - shortly after push, normal `logs` and `output` were empty; supplemental status returned `KernelWorkerStatus.RUNNING`.
  - final `kaggle kernels status` returned `KernelWorkerStatus.COMPLETE`.
  - full inference completed on Kaggle notebook version 1.
  - output: `/tmp/kaggle-output/exp050_xgboost_pseudo_tail_inference_submit/inference_v1`
  - train_wells: 773
  - train_rows: 242,843
  - test_wells: 3
  - predicted_rows: 14,151
  - submission path: `/tmp/kaggle-output/exp050_xgboost_pseudo_tail_inference_submit/inference_v1/submission.csv`
  - synced local artifacts:
    - `artifacts/exp050-xgb-pseudotail-infer.log`
    - `artifacts/pseudo_tail_inference_source_summary.csv`
    - `artifacts/pseudo_tail_inference_summary.json`
    - `artifacts/pseudo_tail_inference_well_summaries.csv`
- submit-check: PASS
- submission diagnostics:
  - rows: 14,151
  - missing values: 0
  - duplicate ids: 0
  - prediction range: 11587.960181 - 12234.905349
  - prediction mean/std: 11907.057784 / 278.953551
  - diff vs exp026 submission: min -4.050819, max 3.799731, mean -0.244825, abs mean 1.100087, RMSE 1.431860, corr 0.999991802
- competition submit:
  - status: `SubmissionStatus.COMPLETE`
  - ref: `53521999`
  - submitted: `2026-06-10 03:15:03.410000`
  - Public LB: 12.083
  - submission log: `logs/submission_exp050_xgboost_pseudo_tail_inference_submit.log`

## 次のアクション

1. exp050 は ML route の旧自前 pseudo-tail Public LB 基準を exp026 12.102 から 12.083 へ更新したが、ML route 全体基準 exp039 11.740 には届いていないため、XGBoost 単体の深追いは小さめにする。
2. 次に進む場合は、exp044 補助 fold / bucket 確認で exp049/050 の悪化 bucket を確認するか、pseudo-tail seed bagging を小さく試す。
