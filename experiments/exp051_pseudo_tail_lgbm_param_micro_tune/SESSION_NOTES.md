# exp051_pseudo_tail_lgbm_param_micro_tune セッションノート

## 目的

バックログ高優先の `pseudo_tail_lgbm_param_micro_tune` を実装する。`exp026` の pseudo-tail + distance-balanced LightGBM 手順、no-GR feature set、残差 shrink、固定 bucket shrink 係数を保ち、LightGBM の狭い parameter / row cap 変更だけを主評価 GroupKFold で比較する。

## 現在の状態

- status: completed
- route: `ml_model`
- parent: `exp026_pseudo_tail_bucket_shrink_inference_submit`
- implementation parent: `exp049_xgboost_pseudo_tail_residual`
- comparison anchors:
  - `exp023` raw pseudo-tail LightGBM CV: 12.942938
  - `exp025/026` fixed bucket-shrink CV: 12.870780
  - `exp026` Public LB: 12.102
  - `exp049` XGBoost fixed bucket-shrink CV: 12.779452
- selected variant: `lgbm_capacity_leaves47_minchild60_exp014_bucket_shrink_params`
- CV: 12.634392
- LB: 未提出

## 実装メモ

- `exp049_xgboost_pseudo_tail_residual` を土台に `exp051` を作成。
- `config.yaml` で estimator を `LGBMRegressor` に戻した。
- training variants は control + 6 micro tune 候補に限定した。
- `subsample` tune が実際に効くように、LightGBM 分岐で `subsample_freq` を設定可能にした。
- `pseudo_tail_augmentation.py` で `variant.model_params` を `model.drift_model.params` に上書き適用できるようにした。
- `variant.max_train_rows_per_well` / `variant.max_train_rows_per_fold` を training row collection に反映できるようにした。
- raw 予測と exp025-selected fixed `exp014_bucket_shrink_params` 後の候補を同時に集計する。
- direct PF/Beam replacement、Ravaghi feature、推論 port、提出処理はこの実験範囲に含めない。

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp051_pseudo_tail_lgbm_param_micro_tune
uv run python scripts/new_experiment.py --name exp051_pseudo_tail_lgbm_param_micro_tune --source experiments/exp049_xgboost_pseudo_tail_residual
uv run ruff check experiments/exp051_pseudo_tail_lgbm_param_micro_tune/baseline.py experiments/exp051_pseudo_tail_lgbm_param_micro_tune/pseudo_tail_augmentation.py experiments/exp051_pseudo_tail_lgbm_param_micro_tune/settings.py
uv run python -m py_compile experiments/exp051_pseudo_tail_lgbm_param_micro_tune/baseline.py experiments/exp051_pseudo_tail_lgbm_param_micro_tune/pseudo_tail_augmentation.py experiments/exp051_pseudo_tail_lgbm_param_micro_tune/settings.py
uv run python scripts/validate_experiment.py --experiment exp051_pseudo_tail_lgbm_param_micro_tune
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp051_pseudo_tail_lgbm_param_micro_tune --notebook train --kernel-id kentookumura/exp051-lgbm-micro-tune-train --title "exp051 lgbm micro tune train" --run-on-push --strict
kaggle kernels push -p experiments/exp051_pseudo_tail_lgbm_param_micro_tune/kaggle/train
kaggle kernels pull kentookumura/exp051-lgbm-micro-tune-train -p /tmp/kaggle-pull/exp051-lgbm-micro-tune-train-check -m
kaggle kernels logs kentookumura/exp051-lgbm-micro-tune-train
kaggle kernels output kentookumura/exp051-lgbm-micro-tune-train -p /tmp/kaggle-output/exp051_pseudo_tail_lgbm_param_micro_tune/train_v1
kaggle kernels status kentookumura/exp051-lgbm-micro-tune-train
```

## 検証状況

- Static checks: PASS
  - `ruff check`: PASS
  - `py_compile`: PASS
  - `scripts/validate_experiment.py --experiment exp051_pseudo_tail_lgbm_param_micro_tune`: PASS
- Notebook code compile: PASS via `prepare_kaggle_notebooks --strict`
- Kaggle train package: prepared
  - path: `experiments/exp051_pseudo_tail_lgbm_param_micro_tune/kaggle/train`
  - kernel id: `kentookumura/exp051-lgbm-micro-tune-train`
  - title: `exp051 lgbm micro tune train`
  - run_on_push: true
  - GPU: false
  - internet: false
- Kaggle train:
  - pushed version 1 successfully.
  - URL: `https://www.kaggle.com/code/kentookumura/exp051-lgbm-micro-tune-train`
  - direct `kaggle kernels pull ... -m` succeeded, so the kernel exists on Kaggle.
  - shortly after push, normal `logs` and `output` were empty; treated as Kaggle API/session output lag.
  - supplemental `kaggle kernels status` returned `KernelWorkerStatus.RUNNING`.
  - user later reported completion.
  - final `kaggle kernels status` returned `KernelWorkerStatus.COMPLETE`.
  - logs and output were retrieved successfully.
  - output: `/tmp/kaggle-output/exp051_pseudo_tail_lgbm_param_micro_tune/train_v1`
  - synced local artifacts:
    - `metrics.json`
    - `artifacts/exp051-lgbm-micro-tune-train.log`
    - `artifacts/distance_candidate_metrics.csv`
    - `artifacts/distance_residual_bucket_summary.csv`
    - `artifacts/pseudo_tail_feature_importance.csv`
    - `artifacts/pseudo_tail_source_summary.csv`
    - `artifacts/pseudo_tail_training_metrics.csv`
    - `artifacts/pseudo_tail_training_summary.json`
  - best variant: `lgbm_capacity_leaves47_minchild60_exp014_bucket_shrink_params`
  - raw capacity CV: 12.706752
  - fixed bucket-shrink capacity CV: 12.634392
  - same-run control fixed bucket-shrink CV: 12.784540
  - fixed capacity delta vs same-run control: -0.150148
  - fixed capacity delta vs exp026 fixed bucket-shrink 12.870780: -0.236388
  - fixed capacity delta vs exp049 XGBoost fixed bucket-shrink 12.779452: -0.145060
  - LB: not submitted

## 次のアクション

1. 選択候補だけを inference port し、submit-check、予測範囲、exp026/exp050 submission との差分を確認する。
2. 提出する場合も exp027 PF route 基準 8.781 とは混ぜず、ML route / pseudo-tail 自前系の Public LB 比較として記録する。
