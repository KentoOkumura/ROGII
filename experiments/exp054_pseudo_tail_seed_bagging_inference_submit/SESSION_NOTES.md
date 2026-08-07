# exp054_pseudo_tail_seed_bagging_inference_submit セッションノート

## 目的

`exp053_pseudo_tail_seed_bagging` の 3-seed bagging を `exp052` inference flow に移植し、seed 変更による Public LB 影響を早めに確認する。

## 現在の状態

- status: completed
- route: `ml_model`
- parent: `exp053_pseudo_tail_seed_bagging`
- implementation parent: `exp052_lgbm_capacity_pseudotail_inference_submit`
- selected variant: `lgbm_capacity_seed_bag3`
- member seeds: `[42, 314, 2027]`
- CV reference: 12.633797
- exp052 Public LB reference: 12.076
- submission ref: `53526321`
- submission status: complete
- LB: Public 11.856

## 実装メモ

- `exp052` を土台に `exp054` を作成。
- final fit で 3 member model を学習し、test raw prediction を平均してから fixed bucket shrink を適用する。
- `config.yaml` の `audit.training_variants.selected_variant` は `lgbm_capacity_seed_bag3`。
- direct PF/Beam replacement、追加 tuning、bucket shrink 変更は含めない。

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp054_pseudo_tail_seed_bagging_inference_submit
uv run python scripts/new_experiment.py --name exp054_pseudo_tail_seed_bagging_inference_submit --source experiments/exp052_lgbm_capacity_pseudotail_inference_submit
uv run ruff check experiments/exp054_pseudo_tail_seed_bagging_inference_submit/baseline.py experiments/exp054_pseudo_tail_seed_bagging_inference_submit/pseudo_tail_augmentation.py experiments/exp054_pseudo_tail_seed_bagging_inference_submit/settings.py
uv run python -m py_compile experiments/exp054_pseudo_tail_seed_bagging_inference_submit/baseline.py experiments/exp054_pseudo_tail_seed_bagging_inference_submit/pseudo_tail_augmentation.py experiments/exp054_pseudo_tail_seed_bagging_inference_submit/settings.py
uv run python scripts/validate_experiment.py --experiment exp054_pseudo_tail_seed_bagging_inference_submit
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp054_pseudo_tail_seed_bagging_inference_submit --notebook inference --kernel-id kentookumura/exp054-seed-bag-infer --title "exp054 seed bag infer" --run-on-push --strict
kaggle kernels push -p experiments/exp054_pseudo_tail_seed_bagging_inference_submit/kaggle/inference
kaggle kernels pull kentookumura/exp054-seed-bag-infer -p /tmp/kaggle-pull/exp054-seed-bag-infer -m
kaggle kernels status kentookumura/exp054-seed-bag-infer
kaggle kernels logs kentookumura/exp054-seed-bag-infer
kaggle kernels output kentookumura/exp054-seed-bag-infer -p /tmp/kaggle-output/exp054_pseudo_tail_seed_bagging_inference_submit/inference_v1
uv run python scripts/validate_submission.py --submission /tmp/kaggle-output/exp054_pseudo_tail_seed_bagging_inference_submit/inference_v1/submission.csv
python .agents/skills/kaggle-submit-check/scripts/check_submission.py /tmp/kaggle-output/exp054_pseudo_tail_seed_bagging_inference_submit/inference_v1/submission.csv --sample data/raw/sample_submission.csv
kaggle competitions submit rogii-wellbore-geology-prediction -k kentookumura/exp054-seed-bag-infer -v 1 -f submission.csv -m "exp054 seed bagged pseudotail fixed bucket shrink"
kaggle competitions submissions rogii-wellbore-geology-prediction
uv run python scripts/record_submission.py --experiment exp054_pseudo_tail_seed_bagging_inference_submit --file /tmp/kaggle-output/exp054_pseudo_tail_seed_bagging_inference_submit/inference_v1/submission.csv --cv 12.633797 --notes "ref=53526321; kernel=kentookumura/exp054-seed-bag-infer v1; 3-seed LightGBM capacity pseudo-tail fixed bucket-shrink; submit-check PASS; status pending at first check; user will report final LB"
```

## 検証状況

- Static checks: PASS
- Kaggle inference: completed
  - version: 1
  - URL: `https://www.kaggle.com/code/kentookumura/exp054-seed-bag-infer`
  - `kaggle kernels push -p experiments/exp054_pseudo_tail_seed_bagging_inference_submit/kaggle/inference` succeeded.
  - `kaggle kernels pull kentookumura/exp054-seed-bag-infer -p /tmp/kaggle-pull/exp054-seed-bag-infer -m` succeeded.
  - Kaggle metadata returned `id_no: 122284012`.
  - Supplemental status returned `KernelWorkerStatus.RUNNING`.
  - Normal logs/output were empty immediately after push; `logs -f` hit sandbox DNS failure, escalated retry timed out after 20 seconds with no output.
  - final status returned `KernelWorkerStatus.COMPLETE`.
  - output: `/tmp/kaggle-output/exp054_pseudo_tail_seed_bagging_inference_submit/inference_v1`
  - predicted rows: 14,151
  - train rows across 3 members: 728,843
- submit-check: PASS
- code submit: submitted
  - ref: `53526321`
  - first status check: `SubmissionStatus.PENDING`
  - final status reported via Kaggle submissions API: `SubmissionStatus.COMPLETE`
  - Public LB: 11.856

## Submission diagnostics

- rows: 14,151
- missing values: 0
- duplicate ids: 0
- SHA256: `73c978e3bff87fe6eb195d10adf318916cdc554f92870704e8b51efc5a3428bc`
- prediction range: 11590.045172 - 12236.916569
- prediction mean/std: 11906.235341 / 278.510098
- diff vs exp052 submission: min -3.158809, max 3.367355, mean 0.392356, abs mean 0.827752, RMSE 1.113177, corr 0.999995816

## 結果

- CV reference: 12.633797
- Public LB: 11.856
- exp052 single-seed capacity Public LB: 12.076
- delta vs exp052: -0.220
- ML route Public LB anchor `exp039`: 11.740
- delta vs exp039: +0.116

Seed bagging は pseudo-tail 自前系の Public LB 基準を更新したが、ML route 全体基準 `exp039` には届かなかった。

## 次のアクション

1. pseudo-tail 自前系の LB 基準は exp054 11.856 に更新する。
2. ML route 全体基準は exp039 11.740 のまま維持する。
