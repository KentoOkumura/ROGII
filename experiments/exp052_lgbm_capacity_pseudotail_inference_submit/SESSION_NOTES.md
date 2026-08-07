# exp052_lgbm_capacity_pseudotail_inference_submit セッションノート

## 目的

`exp051_pseudo_tail_lgbm_param_micro_tune` で通常 CV が 12.634392 まで改善した LightGBM capacity pseudo-tail fixed bucket-shrink 候補を、`exp050` の inference flow に移植して Kaggle output の `submission.csv` を生成し、提出前チェックと code submit を行う。

## 現在の状態

- status: submitted
- route: `ml_model`
- parent: `exp051_pseudo_tail_lgbm_param_micro_tune`
- implementation parent: `exp050_xgboost_pseudo_tail_inference_submit`
- selected training variant: `lgbm_capacity_leaves47_minchild60`
- selected model params: `LGBMRegressor`, `num_leaves=47`, `min_child_samples=60`
- selected postprocess: `distance_bucket_shrink` / `exp014_bucket_shrink_params`
- CV reference: 12.634392
- exp050 Public LB reference: 12.083
- LB: Public 12.076
- submission ref: `53524340`

## 実装メモ

- `exp050_xgboost_pseudo_tail_inference_submit` を土台に `exp052` を作成。
- `settings.py` と notebook 名を `exp052` に更新。
- `config.yaml` で `model.drift_model.estimator: LGBMRegressor` に戻し、exp051 best の `num_leaves=47` / `min_child_samples=60` を設定した。
- `audit.training_variants.selected_variant` は `lgbm_capacity_leaves47_minchild60` にした。
- `postprocess.selected_method` は exp026/050 と同じ `distance_bucket_shrink` を維持。
- direct PF/Beam replacement、Ravaghi feature、additional tuning はこの実験範囲に含めない。

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp052_lgbm_capacity_pseudotail_inference_submit
uv run python scripts/new_experiment.py --name exp052_lgbm_capacity_pseudotail_inference_submit --source experiments/exp050_xgboost_pseudo_tail_inference_submit
uv run ruff check experiments/exp052_lgbm_capacity_pseudotail_inference_submit/baseline.py experiments/exp052_lgbm_capacity_pseudotail_inference_submit/pseudo_tail_augmentation.py experiments/exp052_lgbm_capacity_pseudotail_inference_submit/settings.py
uv run python -m py_compile experiments/exp052_lgbm_capacity_pseudotail_inference_submit/baseline.py experiments/exp052_lgbm_capacity_pseudotail_inference_submit/pseudo_tail_augmentation.py experiments/exp052_lgbm_capacity_pseudotail_inference_submit/settings.py
uv run python scripts/validate_experiment.py --experiment exp052_lgbm_capacity_pseudotail_inference_submit
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp052_lgbm_capacity_pseudotail_inference_submit --notebook inference --kernel-id kentookumura/exp052-lgbm-cap-pseudotail-infer --title "exp052 lgbm cap pseudotail infer" --run-on-push --strict
kaggle kernels push -p experiments/exp052_lgbm_capacity_pseudotail_inference_submit/kaggle/inference
kaggle kernels pull kentookumura/exp052-lgbm-cap-pseudotail-infer -p /tmp/kaggle-pull/exp052-lgbm-cap-pseudotail-infer-check -m
kaggle kernels logs kentookumura/exp052-lgbm-cap-pseudotail-infer
kaggle kernels output kentookumura/exp052-lgbm-cap-pseudotail-infer -p /tmp/kaggle-output/exp052_lgbm_capacity_pseudotail_inference_submit/inference_v1
kaggle kernels status kentookumura/exp052-lgbm-cap-pseudotail-infer
uv run python scripts/validate_submission.py --submission /tmp/kaggle-output/exp052_lgbm_capacity_pseudotail_inference_submit/inference_v1/submission.csv
kaggle competitions submit rogii-wellbore-geology-prediction -k kentookumura/exp052-lgbm-cap-pseudotail-infer -v 1 -f submission.csv -m "exp052 lgbm capacity pseudotail fixed bucket shrink"
kaggle competitions submissions rogii-wellbore-geology-prediction
python .agents/skills/kaggle-submit-monitor/scripts/monitor_submission.py exp052_lgbm_capacity_pseudotail_inference_submit --competition rogii-wellbore-geology-prediction
uv run python scripts/record_submission.py --experiment exp052_lgbm_capacity_pseudotail_inference_submit --file /tmp/kaggle-output/exp052_lgbm_capacity_pseudotail_inference_submit/inference_v1/submission.csv --cv 12.634392 --public-lb 12.076 --notes "ref=53524340; kernel=kentookumura/exp052-lgbm-cap-pseudotail-infer v1; LightGBM capacity pseudo-tail fixed bucket-shrink; improves exp050 Public LB 12.083 by -0.007 and exp026 12.102 by -0.026; submit-check PASS"
```

## 検証状況

- Static checks: PASS
  - `ruff check`: PASS
  - `py_compile`: PASS
  - `scripts/validate_experiment.py --experiment exp052_lgbm_capacity_pseudotail_inference_submit`: PASS
- Kaggle inference package: prepared
  - path: `experiments/exp052_lgbm_capacity_pseudotail_inference_submit/kaggle/inference`
  - kernel id: `kentookumura/exp052-lgbm-cap-pseudotail-infer`
  - title: `exp052 lgbm cap pseudotail infer`
  - run_on_push: true
  - GPU: false
  - internet: false
- Kaggle inference:
  - pushed version 1 successfully.
  - URL: `https://www.kaggle.com/code/kentookumura/exp052-lgbm-cap-pseudotail-infer`
  - direct `kaggle kernels pull ... -m` succeeded, so the kernel exists on Kaggle.
  - shortly after push, normal `logs` and `output` were empty; treated as Kaggle API/session output lag.
  - supplemental `kaggle kernels status` returned `KernelWorkerStatus.RUNNING`.
  - final `kaggle kernels status` returned `KernelWorkerStatus.COMPLETE`.
  - full inference completed on Kaggle notebook version 1.
  - output: `/tmp/kaggle-output/exp052_lgbm_capacity_pseudotail_inference_submit/inference_v1`
  - train_wells: 773
  - train_rows: 242,843
  - test_wells: 3
  - predicted_rows: 14,151
  - submission path: `/tmp/kaggle-output/exp052_lgbm_capacity_pseudotail_inference_submit/inference_v1/submission.csv`
  - synced local artifacts:
    - `artifacts/exp052-lgbm-cap-pseudotail-infer.log`
    - `artifacts/pseudo_tail_inference_source_summary.csv`
    - `artifacts/pseudo_tail_inference_summary.json`
    - `artifacts/pseudo_tail_inference_well_summaries.csv`
- submit-check: PASS
- submission diagnostics:
  - rows: 14,151
  - missing values: 0
  - duplicate ids: 0
  - SHA256: `657ca475d9ff8abfa7a1f482473b47815a2c9001803ae4b2b63c5074585d992b`
  - prediction range: 11587.429983 - 12236.572595
  - prediction mean/std: 11905.842985 / 279.169431
  - diff vs exp026 submission: min -6.363842, max 2.009570, mean -1.459623, abs mean 1.724878, RMSE 2.225454, corr 0.999984451265
  - diff vs exp050 submission: min -5.605740, max 3.099844, mean -1.214799, abs mean 1.656530, RMSE 2.011376, corr 0.999983797988
- code submit:
  - status: `SubmissionStatus.COMPLETE`
  - ref: `53524340`
  - submitted: `2026-06-10 05:09:36.977000`
  - Public LB: 12.076
  - submission log: `logs/submission_exp052_lgbm_capacity_pseudotail_inference_submit.log`

## 次のアクション

1. exp052 は pseudo-tail 自前系の Public LB 基準を exp050 12.083 から 12.076 へ更新したが、ML route 全体基準 exp039 11.740 には届いていない。
2. 次に進む場合は、seed bagging や cutoff distribution ablation を小さく試す。Public LB だけを見た blind tuning は避ける。
