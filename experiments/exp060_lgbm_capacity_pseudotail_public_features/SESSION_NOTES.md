# exp060_lgbm_capacity_pseudotail_public_features セッションノート

## 目的

高優先 backlog `lgbm_capacity_pseudotail_public_features` を実装する。
親実験は `exp056_public_sel15_pf_oof_multicutoff_artifact` とし、exp056 の
multi-cutoff public sel15 PF/Beam feature artifact を、`exp051/052`
LightGBM capacity pseudo-tail residual model の train-side audit に接続する。

## 現在の状態

- status: completed
- route: `ml_model`
- parent: `exp056_public_sel15_pf_oof_multicutoff_artifact`
- model anchor parent: `exp051_pseudo_tail_lgbm_param_micro_tune`
- inference anchor parent: `exp052_lgbm_capacity_pseudotail_inference_submit`
- implementation source: `exp059_pf_model_diff_foldsafe_surface_shrink`
- input artifact: `experiments/exp056_public_sel15_pf_oof_multicutoff_artifact/features/public_sel15_pf_oof_features.csv.gz`
- selected variant: `lgbm_capacity_public_core_spatial_multicutoff_raw`
- CV: original-fold 15.562057 / well-hash 15.731138 on exp056 pseudo-test surface
- LB: not submitted

## 実装メモ

- `exp060_lgbm_capacity_pseudotail_public_features` を作成。
- exp060 の実験親は exp056。exp059 は audit script の再利用元としてのみ記録。
- train notebook は `public_feature_model_audit.py` を呼ぶ。
- exp052/054 foldout source prediction はこの実験範囲から外した。
- 0.65-only control、multi-cutoff equal budget、0.65 preserve + augmentation を別 training policy として比較できるようにした。
- public feature family は PF prediction、PF uncertainty、beam disagreement、NCC/GR match minimal、spatial context に分けた。
- `preserve_cutoffs: [0.65]` を audit script に実装し、0.65 rows を優先保持してから 0.45 / 0.82 rows を追加できるようにした。

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp060_lgbm_capacity_pseudotail_public_features
uv run python scripts/new_experiment.py --name exp060_lgbm_capacity_pseudotail_public_features --source experiments/exp059_pf_model_diff_foldsafe_surface_shrink
uv run python -m py_compile experiments/exp060_lgbm_capacity_pseudotail_public_features/public_feature_model_audit.py experiments/exp060_lgbm_capacity_pseudotail_public_features/baseline.py experiments/exp060_lgbm_capacity_pseudotail_public_features/pseudo_tail_augmentation.py experiments/exp060_lgbm_capacity_pseudotail_public_features/settings.py
uv run ruff check experiments/exp060_lgbm_capacity_pseudotail_public_features/public_feature_model_audit.py experiments/exp060_lgbm_capacity_pseudotail_public_features/baseline.py experiments/exp060_lgbm_capacity_pseudotail_public_features/pseudo_tail_augmentation.py experiments/exp060_lgbm_capacity_pseudotail_public_features/settings.py
uv run python scripts/validate_experiment.py --experiment exp060_lgbm_capacity_pseudotail_public_features
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp060_lgbm_capacity_pseudotail_public_features --notebook train --kernel-id kentookumura/exp060-lgbm-public-features-train --title "exp060 lgbm public features train" --run-on-push --strict
kaggle kernels push -p experiments/exp060_lgbm_capacity_pseudotail_public_features/kaggle/train
kaggle kernels pull kentookumura/exp060-lgbm-public-features-train -p /tmp/kaggle-pull/exp060-lgbm-public-features-train -m
kaggle kernels logs kentookumura/exp060-lgbm-public-features-train
timeout 120 kaggle kernels logs -f --interval 10 kentookumura/exp060-lgbm-public-features-train
kaggle kernels output kentookumura/exp060-lgbm-public-features-train -p /tmp/kaggle-output/exp060_lgbm_capacity_pseudotail_public_features/train_v1
kaggle kernels status kentookumura/exp060-lgbm-public-features-train
kaggle kernels logs kentookumura/exp060-lgbm-public-features-train
kaggle kernels output kentookumura/exp060-lgbm-public-features-train -p /tmp/kaggle-output/exp060_lgbm_capacity_pseudotail_public_features/train_v1
```

## 検証状況

- Static checks: PASS
  - `py_compile`: PASS
  - `ruff check`: PASS
  - `scripts/validate_experiment.py --experiment exp060_lgbm_capacity_pseudotail_public_features`: PASS
- Feature column check: PASS
  - configured features: 79
  - required loaded columns: 47
  - missing required columns: none
  - missing configured features: none
- Kaggle train package: prepared
  - path: `experiments/exp060_lgbm_capacity_pseudotail_public_features/kaggle/train`
  - kernel id: `kentookumura/exp060-lgbm-public-features-train`
  - title: `exp060 lgbm public features train`
  - run_on_push: true
  - GPU: false
  - internet: false
- Initial Kaggle train check: running
  - version 1 push succeeded.
  - URL: `https://www.kaggle.com/code/kentookumura/exp060-lgbm-public-features-train`
  - direct `kaggle kernels pull ... -m` succeeded, so the kernel exists on Kaggle.
  - normal `logs` returned empty shortly after push.
  - `logs -f --interval 10` for 120 seconds returned no log output; treated as Kaggle API/session output lag while the kernel is still running.
  - `kaggle kernels output ...` returned no files yet.
  - supplemental `kaggle kernels status` returned `KernelWorkerStatus.RUNNING`.
- Kaggle train: completed
  - user reported completion on 2026-06-11.
  - final `kaggle kernels logs kentookumura/exp060-lgbm-public-features-train` returned the full log.
  - final `kaggle kernels output ... -p /tmp/kaggle-output/exp060_lgbm_capacity_pseudotail_public_features/train_v1` downloaded output files.
  - synced local artifacts:
    - `metrics.json`
    - `artifacts/exp060-lgbm-public-features-train.log`
    - `artifacts/public_feature_bucket_metrics.csv`
    - `artifacts/public_feature_family_matrix.csv`
    - `artifacts/public_feature_feature_importance.csv`
    - `artifacts/public_feature_feature_parity_report.csv`
    - `artifacts/public_feature_metrics.csv`
    - `artifacts/public_feature_split_metrics.csv`
    - `artifacts/public_feature_summary.json`
    - `artifacts/public_feature_train_summary.csv`
    - `artifacts/public_feature_well_metrics.csv`

## 結果

- rows / wells: 5,499,624 / 773
- best original-fold candidate: `pf090_hold010` 15.023697
- best well-hash candidate: `pf090_hold010` 15.023697
- selected model candidate: `lgbm_capacity_public_core_spatial_multicutoff_raw`
- selected model candidate RMSE: original-fold 15.562057 / well-hash 15.731138
- delta vs 0.65 geometry control: -3.339331 / -3.203251
- delta vs `public_pf_selector`: +0.441218 / +0.610299
- delta vs `pf090_hold010`: +0.538360 / +0.707441
- `lgbm_capacity_public_pf_core_multicutoff_equal_budget_raw`: 15.644047 / 15.776224
- `lgbm_capacity_public_pf_core_cutoff065_raw`: 15.772781 / 15.670501
- bucket-shrink variants were worse than corresponding raw variants except the 0.65-only core on well-hash, and none beat direct PF controls.

Interpretation:

- Public notebook derived features are strong add-only model features for the ML route.
- The selected candidate improves over the 0.65 geometry ML control and its paired NCC/GR + PF context control on both holdout audits.
- `pf090_hold010` and `public_pf_selector` are direct PF diagnostic controls, not the adoption criterion for this ML-route experiment.
- This result does not update the normal CV or Public LB anchor yet because the exp056 pseudo-test surface differs from exp051/052/054 evaluation. Inference port / hidden-branch audit is the next step if LB evidence is needed.

## 次のアクション

1. `lgbm_capacity_pseudotail_public_features` は train-side implementation として backlog から外す。
2. 次にこの線を進めるなら、同じ exp060 内で `lgbm_capacity_public_core_spatial_multicutoff_raw` の inference port / hidden-branch audit を行う。
3. 別線では exp054/059 系の seedbag anchor + model-diff distance gate、または public artifact replay integrity audit を優先する。

## Inference 実行メモ

- 2026-06-11: 同じ exp060 内で inference port を実装。
- `public_feature_inference.py` は train-side selected candidate `lgbm_capacity_public_core_spatial_multicutoff_raw` を final fit し、hidden branch に同じ public PF/Beam、NCC/GR-match、spatial features を生成して residual correction を適用する。
- `config.yaml` に `inference.selected_variant: lgbm_capacity_public_core_spatial_multicutoff` を追加。
- inference notebook は `generate_public_feature_submission(paths, config)` を実行する構成に更新。

追加実行コマンド:

```bash
uv run python -m py_compile experiments/exp060_lgbm_capacity_pseudotail_public_features/public_feature_inference.py experiments/exp060_lgbm_capacity_pseudotail_public_features/public_feature_model_audit.py experiments/exp060_lgbm_capacity_pseudotail_public_features/settings.py experiments/exp060_lgbm_capacity_pseudotail_public_features/pseudo_tail_augmentation.py
uv run ruff check experiments/exp060_lgbm_capacity_pseudotail_public_features/public_feature_inference.py experiments/exp060_lgbm_capacity_pseudotail_public_features/public_feature_model_audit.py experiments/exp060_lgbm_capacity_pseudotail_public_features/settings.py experiments/exp060_lgbm_capacity_pseudotail_public_features/pseudo_tail_augmentation.py
uv run python scripts/validate_experiment.py --experiment exp060_lgbm_capacity_pseudotail_public_features
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp060_lgbm_capacity_pseudotail_public_features --notebook inference --kernel-id kentookumura/exp060-lgbm-public-features-inference --title "exp060 lgbm public features inference" --run-on-push --strict
kaggle kernels push -p experiments/exp060_lgbm_capacity_pseudotail_public_features/kaggle/inference
kaggle kernels pull kentookumura/exp060-lgbm-public-features-inference -p /tmp/kaggle-pull/exp060-lgbm-public-features-inference -m
kaggle kernels logs kentookumura/exp060-lgbm-public-features-inference
timeout 120 kaggle kernels logs -f --interval 10 kentookumura/exp060-lgbm-public-features-inference
kaggle kernels output kentookumura/exp060-lgbm-public-features-inference -p /tmp/kaggle-output/exp060_lgbm_capacity_pseudotail_public_features/inference_v1
kaggle kernels status kentookumura/exp060-lgbm-public-features-inference
```

Inference 検証状況:

- Static checks: PASS
  - `py_compile`: PASS
  - `ruff check`: PASS
  - `scripts/validate_experiment.py --experiment exp060_lgbm_capacity_pseudotail_public_features`: PASS
- Kaggle inference package: prepared
  - path: `experiments/exp060_lgbm_capacity_pseudotail_public_features/kaggle/inference`
  - kernel id: `kentookumura/exp060-lgbm-public-features-inference`
  - title: `exp060 lgbm public features inference`
  - run_on_push: true
  - GPU: false
  - internet: false
  - kernel source: `kentookumura/exp056-sel15-pf-oof-multicutoff`
- Kaggle inference: running
  - version 1 push succeeded.
  - URL: `https://www.kaggle.com/code/kentookumura/exp060-lgbm-public-features-inference`
  - direct `kaggle kernels pull ... -m` succeeded, so the kernel exists on Kaggle.
  - normal `logs` returned empty shortly after push.
  - `logs -f --interval 10` for 120 seconds returned no log output; treated as Kaggle API/session output lag while the kernel is still running.
  - `kaggle kernels output ...` returned no files yet.
  - supplemental `kaggle kernels status` returned `KernelWorkerStatus.RUNNING`.
  - user asked not to monitor further and will report completion.
- Kaggle inference: completed
  - user reported completion on 2026-06-12.
  - final `kaggle kernels logs kentookumura/exp060-lgbm-public-features-inference` returned the full log.
  - final `kaggle kernels output ... -p /tmp/kaggle-output/exp060_lgbm_capacity_pseudotail_public_features/inference_v1` downloaded output files.
  - synced local artifacts:
    - `artifacts/exp060-lgbm-public-features-inference.log`
    - `artifacts/public_feature_inference_summary.json`
    - `artifacts/public_feature_inference_wells.csv`
    - `artifacts/public_feature_inference_source_summary.csv`
    - `artifacts/lgbm_public_feature_corrected_summary.json`
    - `artifacts/lgbm_public_feature_corrected_diff.csv`
  - `submission.csv` remains in Kaggle output only:
    - `/tmp/kaggle-output/exp060_lgbm_capacity_pseudotail_public_features/inference_v1/submission.csv`
  - submit-check PASS against `data/raw/sample_submission.csv`.
  - rows: 14,151
  - SHA256: `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`
  - prediction range: 11587.038593 to 12240.016066
  - branch counts: `physical_visible` 14,151
  - changed_rows: 0

Inference interpretation:

- The notebook package and final-fit inference flow completed successfully.
- The public sample contains only wells `000d7d20`, `00bbac68`, and `00e12e8b`, all of which took the physical-visible branch.
- The hidden `lgbm_capacity_public_core_spatial_multicutoff_raw_hidden` correction branch therefore did not change the public output.
- This validates runtime/output format, but does not update ML route Public LB evidence unless submitted as a code competition run.

## Code submission

- 2026-06-12: user reported completion after code submit.
- latest Kaggle submissions showed:
  - `ref=53581051`: Public LB 12.046
  - `ref=53581056`: Public LB 11.826, already recorded as exp061 selected submission
- Treat `ref=53581051` as exp060 code submission because it immediately precedes exp061 and matches the exp060 code-submit slot with empty Kaggle description.
- Submission record:
  - ref: `53581051`
  - Public LB: 12.046
  - Private LB: not available
  - local output: `/tmp/kaggle-output/exp060_lgbm_capacity_pseudotail_public_features/inference_v1/submission.csv`
  - SHA256: `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`
  - submit-check: PASS

LB interpretation:

- exp060 improves exp052 12.076 by -0.030.
- exp060 is worse than exp054 11.856 by +0.190.
- exp060 is worse than exp061 11.826 by +0.220.
- exp060 is worse than exp039 ML route Public LB 11.740 by +0.306.
- exp060 is worse than exp027 overall/PF route Public LB 8.781 by +3.265.
- Do not update ML route, pseudo-tail self-route, or overall/PF route anchors from exp060.
