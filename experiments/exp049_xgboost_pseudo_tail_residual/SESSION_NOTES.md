# exp049_xgboost_pseudo_tail_residual セッションノート

## 目的

バックログ高優先の `xgboost_pseudo_tail_residual` を実装する。`exp026` の pseudo-tail + distance-balanced 手順、no-GR feature set、row cap、残差 shrink、固定 bucket shrink 係数を保ち、残差モデルだけを LightGBM から XGBoost に差し替えて主評価 GroupKFold で比較する。

## 現在の状態

- status: completed
- route: `ml_model`
- parent: `exp026_pseudo_tail_bucket_shrink_inference_submit`
- comparison anchors:
  - `exp023` raw pseudo-tail LightGBM CV: 12.942938
  - `exp025/026` fixed bucket-shrink CV: 12.870780
  - `exp026` Public LB: 12.102
- selected variant before run: `xgboost_pseudo_tail_3_cutoffs_distance_balanced`
- CV: 12.779452
- LB: 未提出

## 実装メモ

- `exp023_pseudo_tail_distance_augmentation` を土台に `exp049` を作成。
- `baseline.py` に `XGBRegressor` 対応を追加。
- `config.yaml` で `model.drift_model.estimator: XGBRegressor` に変更。
- training variant は `pseudo_tail_3_cutoffs_distance_balanced` 相当の 1 候補に絞った。
- `pseudo_tail_augmentation.py` で、学習した raw 予測に exp025-selected `exp014_bucket_shrink_params` を固定適用した候補も同時に集計する。
- direct PF/Beam replacement、Ravaghi feature、推論 port、提出処理はこの実験範囲に含めない。

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp049_xgboost_pseudo_tail_residual
uv run python scripts/new_experiment.py --name exp049_xgboost_pseudo_tail_residual --source experiments/exp023_pseudo_tail_distance_augmentation
uv run ruff check experiments/exp049_xgboost_pseudo_tail_residual/baseline.py experiments/exp049_xgboost_pseudo_tail_residual/pseudo_tail_augmentation.py experiments/exp049_xgboost_pseudo_tail_residual/settings.py
uv run python -m py_compile experiments/exp049_xgboost_pseudo_tail_residual/baseline.py experiments/exp049_xgboost_pseudo_tail_residual/pseudo_tail_augmentation.py experiments/exp049_xgboost_pseudo_tail_residual/settings.py
uv run python scripts/validate_experiment.py --experiment exp049_xgboost_pseudo_tail_residual
uv run python experiments/exp049_xgboost_pseudo_tail_residual/pseudo_tail_augmentation.py --max-wells 3 --output-dir /tmp/exp049_smoke
uv run python experiments/exp049_xgboost_pseudo_tail_residual/pseudo_tail_augmentation.py --max-wells 6 --output-dir /tmp/exp049_smoke
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp049_xgboost_pseudo_tail_residual --notebook train --kernel-id kentookumura/exp049-xgb-pseudotail-train --title "exp049 xgb pseudotail train" --run-on-push --strict
kaggle kernels push -p experiments/exp049_xgboost_pseudo_tail_residual/kaggle/train
kaggle kernels pull kentookumura/exp049-xgb-pseudotail-train -p /tmp/kaggle-pull/exp049-xgb-pseudotail-train-check -m
kaggle kernels logs kentookumura/exp049-xgb-pseudotail-train
kaggle kernels output kentookumura/exp049-xgb-pseudotail-train -p /tmp/kaggle-output/exp049_xgboost_pseudo_tail_residual/train_v1
kaggle kernels status kentookumura/exp049-xgb-pseudotail-train
```

## 検証状況

- Static checks: PASS
  - `ruff check`: PASS
  - `py_compile`: PASS
  - `scripts/validate_experiment.py --experiment exp049_xgboost_pseudo_tail_residual`: PASS
- Notebook code compile: PASS
- Local smoke:
  - `--max-wells 3`: expected failure because 5-fold GroupKFold cannot run with 3 wells.
  - `--max-wells 6`: first exposed a read-only NumPy view in pseudo cutoff generation; fixed by copying `TVT_input`.
  - `--max-wells 6` after fix reached XGBoost model construction, then stopped because local `uv` environment has no `xgboost` package.
  - This is a local dependency limitation. The Kaggle train package keeps `enable_internet=false` and expects Kaggle's runtime image to provide `xgboost`.
- Kaggle train package: prepared
  - path: `experiments/exp049_xgboost_pseudo_tail_residual/kaggle/train`
  - kernel id: `kentookumura/exp049-xgb-pseudotail-train`
  - title: `exp049 xgb pseudotail train`
  - run_on_push: true
  - GPU: false
  - internet: false
- Kaggle train:
  - pushed version 1 successfully.
  - URL: `https://www.kaggle.com/code/kentookumura/exp049-xgb-pseudotail-train`
  - direct `kaggle kernels pull ... -m` succeeded, so the kernel exists on Kaggle.
  - shortly after push, normal `logs` and `output` were empty; treated as Kaggle API/session output lag.
  - supplemental `kaggle kernels status` returned `KernelWorkerStatus.RUNNING`.
  - user later reported completion.
  - final `kaggle kernels status` returned `KernelWorkerStatus.COMPLETE`.
  - full CV completed on Kaggle train notebook version 1.
  - output: `/tmp/kaggle-output/exp049_xgboost_pseudo_tail_residual/train_v1`
  - synced local artifacts:
    - `metrics.json`
    - `artifacts/exp049-xgb-pseudotail-train.log`
    - `artifacts/distance_candidate_metrics.csv`
    - `artifacts/distance_residual_bucket_summary.csv`
    - `artifacts/pseudo_tail_feature_importance.csv`
    - `artifacts/pseudo_tail_source_summary.csv`
    - `artifacts/pseudo_tail_training_metrics.csv`
    - `artifacts/pseudo_tail_training_summary.json`
  - raw XGBoost CV: 12.839225
  - fixed bucket-shrink CV: 12.779452
  - fixed bucket-shrink delta vs exp026 fixed bucket-shrink 12.870780: -0.091328
  - fixed bucket-shrink delta vs exp023 raw pseudo-tail LightGBM 12.942938: -0.163486
  - best variant: `xgboost_pseudo_tail_3_cutoffs_distance_balanced_exp014_bucket_shrink_params`
  - LB: not submitted

## 次のアクション

1. exp044 補助 fold / distance bucket の破壊的悪化を確認する。
2. 問題がなければ、同じ XGBoost 構成を inference port し、submit-check と予測範囲を確認する。
3. 提出する場合も exp027 PF route 基準 8.781 とは混ぜず、ML route Public LB 基準との比較として記録する。
