# exp059_pf_model_diff_foldsafe_surface_shrink セッションノート

## 目的

高優先 backlog `pf_model_diff_foldsafe_surface_shrink` を実装する。`exp058` の
LightGBM capacity pseudo-test surface を固定し、`PF-vs-exp052/054` と
`Beam-vs-exp052/054` の fold-safe 差分特徴、ならびに surface-specific な
fold-out bucket shrink を検証できる形にする。

## 現在の状態

- status: submitted complete
- route: `ml_model`
- parent: `exp058_lgbm_pf_confidence_only_features`
- model anchor parent: `exp052_lgbm_capacity_pseudotail_inference_submit`
- seed-bag anchor parent: `exp054_pseudo_tail_seed_bagging_inference_submit`
- supporting artifact: `exp029_public_sel15_pf_oof_feature_generation`
- selected variant: `lgbm_capacity_pf_model_diff_foldsafe_raw`
- CV: original-fold 15.037567 / well-hash 14.735200
- LB: Public 11.878 / Private not available
- submission ref: `53549815`
- Public LB: 11.878

## 実装メモ

- `exp058_lgbm_pf_confidence_only_features` を土台に `exp059` を作成。
- train notebook は `pf_model_diff_model_audit.py` を参照する。
- audit split ごとに `exp052` / `exp054` の config を読み、train-fold wells だけで pseudo-tail source model を再学習する。
- source model を validation-fold wells の exp029 pseudo-test rows に適用し、`exp052_foldout` / `exp054_foldout` 予測を作る。
- model feature は raw source prediction ではなく、anchor delta、PF/Beam との差分、exp054-exp052 spread、最小 PF-model diff に限定する。
- paired control は `lgbm_capacity_pf_confidence_only_raw`。追加候補は `lgbm_capacity_pf_model_diff_foldsafe`。
- postprocess は `raw`、`foldout_bucket_shrink`、`confidence_foldout_bucket_shrink` に限定。
- fold-out shrink alpha は held-out split 以外の rows だけで fit し、`pf_model_diff_postprocess_alpha.csv` に保存する。
- source model 生成履歴は `pf_model_diff_source_summary.csv` に保存する。
- inference port は `pf_model_diff_inference.py` に分離し、raw 候補だけを対象にする。
- final main model は train 側も source 予測の in-sample leakage を避けるため、
  well-hash fold-out の exp052/054 source features で学習する。test hidden
  branch では full-train exp052/054 source models を fit して差分特徴を作る。
- visible wells は exp058 と同じ physical branch を維持し、hidden wells だけ
  `lgbm_capacity_pf_model_diff_foldsafe_raw_hidden` に差し替える。

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp059_pf_model_diff_foldsafe_surface_shrink
uv run python scripts/new_experiment.py --name exp059_pf_model_diff_foldsafe_surface_shrink --source experiments/exp058_lgbm_pf_confidence_only_features
```

## 検証状況

- Static checks: PASS
  - `python -m py_compile experiments/exp059_pf_model_diff_foldsafe_surface_shrink/pf_model_diff_model_audit.py experiments/exp059_pf_model_diff_foldsafe_surface_shrink/baseline.py experiments/exp059_pf_model_diff_foldsafe_surface_shrink/pseudo_tail_augmentation.py experiments/exp059_pf_model_diff_foldsafe_surface_shrink/settings.py`
  - `uv run ruff check experiments/exp059_pf_model_diff_foldsafe_surface_shrink/pf_model_diff_model_audit.py experiments/exp059_pf_model_diff_foldsafe_surface_shrink/baseline.py experiments/exp059_pf_model_diff_foldsafe_surface_shrink/pseudo_tail_augmentation.py experiments/exp059_pf_model_diff_foldsafe_surface_shrink/settings.py`
- Experiment validation: PASS
  - `uv run python scripts/validate_experiment.py --experiment exp059_pf_model_diff_foldsafe_surface_shrink`
- Local smoke: blocked by missing local dependency
  - command: `uv run python experiments/exp059_pf_model_diff_foldsafe_surface_shrink/pf_model_diff_model_audit.py --max-wells 3 --max-train-rows 200 --output-dir /tmp/exp059_pf_model_diff_smoke`
  - failure: local environment has no `lightgbm`, so `model.drift_model.estimator=LGBMRegressor` cannot fit source models. Kaggle runtime is the intended execution target.
- Kaggle train package: prepared
  - command: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp059_pf_model_diff_foldsafe_surface_shrink --notebook train --kernel-id kentookumura/exp059-pf-model-diff-train --title "exp059 pf model diff train" --run-on-push --strict`
  - path: `experiments/exp059_pf_model_diff_foldsafe_surface_shrink/kaggle/train`
  - kernel id: `kentookumura/exp059-pf-model-diff-train`
  - title: `exp059 pf model diff train`
  - run_on_push: true
  - enable_gpu: false
  - enable_internet: false
- Kaggle train: completed
  - version 1 push command: `kaggle kernels push -p experiments/exp059_pf_model_diff_foldsafe_surface_shrink/kaggle/train`
  - version 1 URL: `https://www.kaggle.com/code/kentookumura/exp059-pf-model-diff-train`
  - version 1 failed at notebook cell 5 with `FileNotFoundError: /kaggle/working/experiments/exp052_lgbm_capacity_pseudotail_inference_submit/config.yaml`.
  - cause: Kaggle package only contains current experiment support files; source configs referenced by repo-relative `experiments/...` paths were not available in `/kaggle/working`.
  - fix: copied `exp052_source_config.yaml` and `exp054_source_config.yaml` into the exp059 package, changed `audit.model_diff_sources.*.config_path` to those package-local files, and added package-local fallback in `load_external_yaml`.
  - post-fix checks: `python -m py_compile`, `uv run ruff check`, and `uv run python scripts/validate_experiment.py --experiment exp059_pf_model_diff_foldsafe_surface_shrink` all passed.
  - version 2 package includes `exp052_source_config.yaml` and `exp054_source_config.yaml`.
  - version 2 push command: `kaggle kernels push -p experiments/exp059_pf_model_diff_foldsafe_surface_shrink/kaggle/train`
  - version 2 push result: `Kernel version 2 successfully pushed`.
  - existence check: `kaggle kernels pull kentookumura/exp059-pf-model-diff-train -p /tmp/kaggle-pull/exp059-pf-model-diff-train -m` succeeded.
  - logs: normal `kaggle kernels logs` and 5-minute `logs -f --interval 10` returned empty during API lag.
  - output check: `kaggle kernels output kentookumura/exp059-pf-model-diff-train -p /tmp/kaggle-output/exp059_pf_model_diff_foldsafe_surface_shrink/train_v2` returned no files yet.
  - auxiliary status check: `kaggle kernels status kentookumura/exp059-pf-model-diff-train` returned `KernelWorkerStatus.RUNNING`.
  - user reported version 2 failed.
  - version 2 failure: after all five `exp052_foldout` splits completed, `exp054_foldout` seed-bagging failed with `TypeError: source_summary_rows() got an unexpected keyword argument 'member_index'`.
  - cause: `fit_pseudo_tail_models_from_files` was copied from exp054 but exp059's local `source_summary_rows()` did not accept seed-bagging member metadata.
  - fix: updated `source_summary_rows()` to accept optional `member_index` and `member_seed`, preserving existing callers.
  - post-v2-fix checks: `python -m py_compile`, `uv run ruff check`, and `uv run python scripts/validate_experiment.py --experiment exp059_pf_model_diff_foldsafe_surface_shrink` all passed.
  - version 3 package command: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp059_pf_model_diff_foldsafe_surface_shrink --notebook train --kernel-id kentookumura/exp059-pf-model-diff-train --title "exp059 pf model diff train" --run-on-push --strict`
  - version 3 push command: `kaggle kernels push -p experiments/exp059_pf_model_diff_foldsafe_surface_shrink/kaggle/train`
  - version 3 push result: `Kernel version 3 successfully pushed`.
  - version 3 existence check: `kaggle kernels pull kentookumura/exp059-pf-model-diff-train -p /tmp/kaggle-pull/exp059-pf-model-diff-train -m` succeeded.
  - version 3 auxiliary status check: `kaggle kernels status kentookumura/exp059-pf-model-diff-train` returned `KernelWorkerStatus.RUNNING`.
  - version 3 completion reported by user on 2026-06-11.
  - logs command: `kaggle kernels logs kentookumura/exp059-pf-model-diff-train`
  - output command: `kaggle kernels output kentookumura/exp059-pf-model-diff-train -p /tmp/kaggle-output/exp059_pf_model_diff_foldsafe_surface_shrink/train_v3`
  - output: `/tmp/kaggle-output/exp059_pf_model_diff_foldsafe_surface_shrink/train_v3`
  - synced local artifacts:
    - `metrics.json`
    - `artifacts/exp059-pf-model-diff-train.log`
    - `artifacts/pf_model_diff_bucket_metrics.csv`
    - `artifacts/pf_model_diff_family_matrix.csv`
    - `artifacts/pf_model_diff_feature_importance.csv`
    - `artifacts/pf_model_diff_feature_parity_report.csv`
    - `artifacts/pf_model_diff_metrics.csv`
    - `artifacts/pf_model_diff_postprocess_alpha.csv`
    - `artifacts/pf_model_diff_source_summary.csv`
    - `artifacts/pf_model_diff_split_metrics.csv`
    - `artifacts/pf_model_diff_summary.json`
    - `artifacts/pf_model_diff_train_summary.csv`
    - `artifacts/pf_model_diff_well_metrics.csv`
- Inference port: implemented
  - helper: `pf_model_diff_inference.py`
  - notebook: `exp059_pf_model_diff_foldsafe_surface_shrink_inference.ipynb`
  - selected candidate: `lgbm_capacity_pf_model_diff_foldsafe_raw`
  - excluded candidates: fold-out bucket shrink and confidence-conditioned fold-out shrink
  - checks:
    - `python -m py_compile experiments/exp059_pf_model_diff_foldsafe_surface_shrink/pf_model_diff_inference.py`
    - `uv run ruff check experiments/exp059_pf_model_diff_foldsafe_surface_shrink/pf_model_diff_inference.py experiments/exp059_pf_model_diff_foldsafe_surface_shrink/pf_model_diff_model_audit.py experiments/exp059_pf_model_diff_foldsafe_surface_shrink/pseudo_tail_augmentation.py experiments/exp059_pf_model_diff_foldsafe_surface_shrink/settings.py`
    - `uv run python scripts/validate_experiment.py --experiment exp059_pf_model_diff_foldsafe_surface_shrink`
- Kaggle inference package: prepared
  - command: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp059_pf_model_diff_foldsafe_surface_shrink --notebook inference --kernel-id kentookumura/exp059-pf-model-diff-infer --title "exp059 pf model diff infer" --run-on-push --strict`
  - path: `experiments/exp059_pf_model_diff_foldsafe_surface_shrink/kaggle/inference`
  - kernel id: `kentookumura/exp059-pf-model-diff-infer`
  - title: `exp059 pf model diff infer`
  - run_on_push: true
  - enable_gpu: false
  - enable_internet: false
  - kernel source: `kentookumura/exp029-sel15-pf-oof-train`
- Kaggle inference: completed
  - version 1 push command: `kaggle kernels push -p experiments/exp059_pf_model_diff_foldsafe_surface_shrink/kaggle/inference`
  - version 1 push result: `Kernel version 1 successfully pushed`
  - URL: `https://www.kaggle.com/code/kentookumura/exp059-pf-model-diff-infer`
  - existence check: `kaggle kernels pull kentookumura/exp059-pf-model-diff-infer -p /tmp/kaggle-pull/exp059-pf-model-diff-infer -m` succeeded.
  - monitoring: not started per previous user preference.
  - completion reported by user on 2026-06-11.
  - logs command: `kaggle kernels logs kentookumura/exp059-pf-model-diff-infer`
  - output command: `kaggle kernels output kentookumura/exp059-pf-model-diff-infer -p /tmp/kaggle-output/exp059_pf_model_diff_foldsafe_surface_shrink/inference_v1`
  - output: `/tmp/kaggle-output/exp059_pf_model_diff_foldsafe_surface_shrink/inference_v1`
  - submission rows: 14,151
  - submit-check: PASS
    - command: `uv run python .agents/skills/kaggle-submit-check/scripts/check_submission.py /tmp/kaggle-output/exp059_pf_model_diff_foldsafe_surface_shrink/inference_v1/submission.csv --sample data/raw/sample_submission.csv`
    - row count, header, duplicate ID, and NaN/Inf checks all passed.
  - public sample SHA256: `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`
  - same SHA as exp027 public replay and exp058 inference public output.
  - public sample branch summary:
    - rows: 14,151
    - changed_rows: 0
    - changed_wells: 0
    - branches: `physical_visible` only
    - corrected range: 11587.038593 to 12240.016066
    - diff_abs_mean: 0.0
    - source prediction columns were not populated because no public sample well used the hidden ML branch.
  - synced local artifacts:
    - `artifacts/exp059-pf-model-diff-infer.log`
    - `artifacts/pf_model_diff_inference_summary.json`
    - `artifacts/pf_model_diff_inference_wells.csv`
- Kaggle code submission: completed
  - ref: `53549815`
  - date: `2026-06-10 22:44:33.583000 UTC`
  - status: `SubmissionStatus.COMPLETE`
  - Public LB: 11.878
  - Private LB: not available
  - delta vs exp058 Public LB 12.778: -0.900
  - delta vs exp054 Public LB 11.856: +0.022
  - delta vs exp039 Public LB 11.740: +0.138
  - delta vs exp027 Public LB 8.781: +3.097

## 結果

- rows / wells: 1,782,279 / 773
- selected candidate: `lgbm_capacity_pf_model_diff_foldsafe_raw`
- selected original-fold RMSE: 15.037567
- selected well-hash RMSE: 14.735200
- selected delta vs `lgbm_capacity_pf_confidence_only_raw`: -0.908111 original-fold / -0.834016 well-hash
- selected delta vs `pf090_hold010`: -0.051965 original-fold / -0.354332 well-hash
- selected delta vs `public_pf_selector`: -0.135069 original-fold / -0.437436 well-hash
- selected delta vs `exp052_foldout_control`: -0.469332 original-fold / -0.949133 well-hash
- selected delta vs `exp054_foldout_control`: -0.331182 original-fold / -0.848632 well-hash
- supported candidates:
  - `lgbm_capacity_pf_model_diff_foldsafe_raw`: 15.037567 / 14.735200
  - `lgbm_capacity_pf_model_diff_foldsafe_foldout_bucket_shrink`: 15.092041 / 14.770612
  - `lgbm_capacity_pf_model_diff_foldsafe_confidence_foldout_bucket_shrink`: 15.098501 / 14.797292
- fold-out bucket shrink and confidence-conditioned fold-out shrink both worsened relative to raw.
- distance bucket caveat: `rows_2500_plus` remains better with direct PF controls than with selected raw model-diff.
- inference v1 produced a valid public sample `submission.csv`, but all public sample wells
  used the physical-visible branch. Therefore the public output is identical to exp027/exp058
  and does not audit hidden-branch changed rows or source prediction values.
- code submission Public LB was 11.878. This substantially improves exp058's 12.778
  confidence-only branch, but does not beat the exp054 seed-bag pseudo-tail ML anchor
  at 11.856.

Interpretation:

- Fold-safe PF/Beam-vs-exp052/054 model-diff features are a strong positive result on the exp029 pseudo-test surface.
- This is the first candidate in this branch to beat both `pf090_hold010` and `public_pf_selector` overall on original-fold and well-hash.
- The shrink variants should not be adopted; raw is the selected train-side candidate.
- The result is still a pseudo-test surface audit, not Public LB evidence. If moving toward submission, port only the raw candidate and audit hidden-branch output carefully.

## 次のアクション

1. Do not adopt exp059 as the current ML-route LB anchor; exp054 remains better by 0.022 Public LB.
2. Treat PF/Beam-vs-exp052/054 model-diff features as useful hidden-branch features because exp059 improved exp058 by 0.900 Public LB.
3. Next candidate should keep the exp059 raw feature idea but address the far-distance bucket weakness and/or combine with exp054 seed-bag pseudo-tail behavior more conservatively.
