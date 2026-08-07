# exp053_pseudo_tail_seed_bagging セッションノート

## 目的

`exp051_pseudo_tail_lgbm_param_micro_tune` の best LightGBM capacity pseudo-tail model をベースに、pseudo-tail sampling seed と LightGBM seed の平均化で分散を下げられるかを OOF で監査する。

## 現在の状態

- status: completed
- route: `ml_model`
- parent: `exp051_pseudo_tail_lgbm_param_micro_tune`
- implementation parent: `exp051_pseudo_tail_lgbm_param_micro_tune`
- base variant: `lgbm_capacity_leaves47_minchild60_exp014_bucket_shrink_params`
- base CV reference: 12.634392
- base pseudo-tail Public LB reference: `exp052_lgbm_capacity_pseudotail_inference_submit` 12.076
- selected variant: `lgbm_capacity_seed_bag3_exp014_bucket_shrink_params`
- CV: 12.633797
- LB: 未提出

## 実装メモ

- `exp051` を土台に `exp053` を作成。
- `config.yaml` の base estimator を exp051 best の `LGBMRegressor(num_leaves=47, min_child_samples=60)` に固定した。
- training variants は次の 2 つだけに限定した。
  - `lgbm_capacity_single_seed_control`
  - `lgbm_capacity_seed_bag3`
- `pseudo_tail_augmentation.py` に `kind: seed_bagging` を追加した。
- seed bagging variant では、member ごとに pseudo-tail row sampling RNG と LightGBM `random_state` を変え、valid raw prediction を平均してから fixed `exp014_bucket_shrink_params` を適用する。
- direct PF/Beam replacement、Ravaghi feature、cutoff 分布変更、追加 LightGBM tuning、推論 port、提出処理はこの実験範囲に含めない。

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp053_pseudo_tail_seed_bagging
uv run python scripts/new_experiment.py --name exp053_pseudo_tail_seed_bagging --source experiments/exp051_pseudo_tail_lgbm_param_micro_tune
```

## 検証状況

- Static checks: PASS
- Kaggle train package: prepared
  - path: `experiments/exp053_pseudo_tail_seed_bagging/kaggle/train`
  - kernel id: `kentookumura/exp053-seed-bag-train`
  - title: `exp053 seed bag train`
  - run_on_push: true
- Kaggle train: completed
  - version: 1
  - URL: `https://www.kaggle.com/code/kentookumura/exp053-seed-bag-train`
  - `kaggle kernels push -p experiments/exp053_pseudo_tail_seed_bagging/kaggle/train` succeeded.
  - `kaggle kernels pull kentookumura/exp053-seed-bag-train -p /tmp/kaggle-pull/exp053-seed-bag-train -m` succeeded.
  - Kaggle metadata returned `id_no: 122277609`, so the kernel exists on Kaggle.
  - Normal `kaggle kernels logs` and `kaggle kernels output` were empty immediately after push; treat as Kaggle API/session output lag and retry the same kernel id later.
  - First sandboxed `logs -f` failed with DNS resolution for `api.kaggle.com`; escalated retry ran for 20 seconds and timed out with no log output.
  - After a short wait, normal `logs` and `output` were still empty.
  - Supplemental `kaggle kernels status kentookumura/exp053-seed-bag-train` returned `KernelWorkerStatus.RUNNING`.
  - user reported completion; final logs and output were retrieved successfully.
  - output: `/tmp/kaggle-output/exp053_pseudo_tail_seed_bagging/train_v1`
  - synced local artifacts:
    - `metrics.json`
    - `artifacts/exp053-seed-bag-train.log`
    - `artifacts/distance_candidate_metrics.csv`
    - `artifacts/distance_residual_bucket_summary.csv`
    - `artifacts/pseudo_tail_feature_importance.csv`
    - `artifacts/pseudo_tail_source_summary.csv`
    - `artifacts/pseudo_tail_training_metrics.csv`
    - `artifacts/pseudo_tail_training_summary.json`

Static check commands:

```bash
uv run ruff check experiments/exp053_pseudo_tail_seed_bagging/baseline.py experiments/exp053_pseudo_tail_seed_bagging/pseudo_tail_augmentation.py experiments/exp053_pseudo_tail_seed_bagging/settings.py
uv run python -m py_compile experiments/exp053_pseudo_tail_seed_bagging/baseline.py experiments/exp053_pseudo_tail_seed_bagging/pseudo_tail_augmentation.py experiments/exp053_pseudo_tail_seed_bagging/settings.py
uv run python scripts/validate_experiment.py --experiment exp053_pseudo_tail_seed_bagging
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp053_pseudo_tail_seed_bagging --notebook train --kernel-id kentookumura/exp053-seed-bag-train --title "exp053 seed bag train" --run-on-push --strict
kaggle kernels push -p experiments/exp053_pseudo_tail_seed_bagging/kaggle/train
kaggle kernels pull kentookumura/exp053-seed-bag-train -p /tmp/kaggle-pull/exp053-seed-bag-train -m
kaggle kernels logs kentookumura/exp053-seed-bag-train
kaggle kernels output kentookumura/exp053-seed-bag-train -p /tmp/kaggle-output/exp053_pseudo_tail_seed_bagging/train_v1_probe
kaggle kernels status kentookumura/exp053-seed-bag-train
kaggle kernels logs kentookumura/exp053-seed-bag-train
kaggle kernels output kentookumura/exp053-seed-bag-train -p /tmp/kaggle-output/exp053_pseudo_tail_seed_bagging/train_v1
```

## 結果

- `lgbm_capacity_seed_bag3_exp014_bucket_shrink_params`: 12.633797
- `lgbm_capacity_seed_bag3`: 12.715910
- `lgbm_capacity_single_seed_control_exp014_bucket_shrink_params`: 12.734551
- `lgbm_capacity_single_seed_control`: 12.800238

Fold:

- fixed seed bag3: fold 0 12.394983、fold 1 11.966605、fold 2 11.431497、fold 3 12.370181、fold 4 14.793445
- fixed single seed control: fold 0 12.512253、fold 1 12.064434、fold 2 11.370228、fold 3 12.578912、fold 4 14.920249

Interpretation:

- seed bag3 fixed は same-run single seed fixed 12.734551 から -0.100754 改善した。
- exp051 best 12.634392 に対する改善は -0.000595 で、実質同等と扱う。
- seed bagging は single seed の不安定さをならす効果はあるが、推論 port は高優先にしない。

## 次のアクション

1. exp053 の推論 port は低優先に留める。
2. 次に通常 CV を伸ばす場合は cutoff distribution / distance balancing / target scaling を優先する。
