# exp061_seedbag_anchor_model_diff_distance_gate セッションノート

## 目的

高優先 backlog `seedbag_anchor_model_diff_distance_gate` を実装する。`exp054` の
seed-bag pseudo-tail を主 anchor として固定し、`exp059` raw model-diff 予測との差分
だけを距離 bucket 別 alpha で薄く混ぜる。

## 現在の状態

- status: submitted_complete
- route: `ml_model`
- parent: `exp059_pf_model_diff_foldsafe_surface_shrink`
- seed-bag anchor parent: `exp054_pseudo_tail_seed_bagging_inference_submit`
- model-diff parent: `exp059_pf_model_diff_foldsafe_surface_shrink`
- selected inference candidate: `lgbm_capacity_pf_confidence_only_seedbag_gate_near_mid_a0p50_far0`
- selected original-fold CV: 14.872556
- selected well-hash CV: 14.737595
- LB: Public 11.826 / Private not available

## 実装メモ

- `exp059` から `exp061_seedbag_anchor_model_diff_distance_gate` を作成した。
- train audit に `seedbag_distance_gate` postprocess を追加した。
- train-side の候補式:

```text
exp054_foldout + alpha(distance_bucket) * (raw_model_diff_pred - exp054_foldout)
```

- inference-side も同じ式を full-train exp054 source prediction に適用する。
- profile は 3 つ:
  - `near_mid_a0p25_far0`
  - `near_mid_a0p50_far0`
  - `global_a0p25`
- `near_mid_*_far0` は `rows_2500_plus` を alpha 0 にして exp054 anchor へ戻す。

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp061_seedbag_anchor_model_diff_distance_gate
uv run python scripts/new_experiment.py --name exp061_seedbag_anchor_model_diff_distance_gate --source experiments/exp059_pf_model_diff_foldsafe_surface_shrink
python -m py_compile experiments/exp061_seedbag_anchor_model_diff_distance_gate/pf_model_diff_model_audit.py experiments/exp061_seedbag_anchor_model_diff_distance_gate/pf_model_diff_inference.py experiments/exp061_seedbag_anchor_model_diff_distance_gate/settings.py
uv run ruff check experiments/exp061_seedbag_anchor_model_diff_distance_gate/pf_model_diff_model_audit.py experiments/exp061_seedbag_anchor_model_diff_distance_gate/pf_model_diff_inference.py experiments/exp061_seedbag_anchor_model_diff_distance_gate/settings.py
uv run python scripts/validate_experiment.py --experiment exp061_seedbag_anchor_model_diff_distance_gate
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp061_seedbag_anchor_model_diff_distance_gate --notebook train --kernel-id kentookumura/exp061-seedbag-diff-gate-train --title "exp061 seedbag diff gate train" --run-on-push --strict
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp061_seedbag_anchor_model_diff_distance_gate --notebook inference --kernel-id kentookumura/exp061-seedbag-diff-gate-infer --title "exp061 seedbag diff gate infer" --run-on-push --strict
uv run python scripts/update_experiment_summary.py --help
```

## 検証状況

- Static checks:
  - `python -m py_compile ...`: PASS
  - `uv run ruff check ...`: PASS
- Experiment validation: PASS
  - `uv run python scripts/validate_experiment.py --experiment exp061_seedbag_anchor_model_diff_distance_gate`
- Kaggle notebook packages: prepared
  - train: `experiments/exp061_seedbag_anchor_model_diff_distance_gate/kaggle/train`
  - inference: `experiments/exp061_seedbag_anchor_model_diff_distance_gate/kaggle/inference`
- Experiment summary: updated with exp061 row
  - Note: `scripts/update_experiment_summary.py --help` behaved as an update command and rewrote `experiment_summary.md`.
- Kaggle train: v1 completed
- Kaggle inference: not run
- Kaggle train v1: running
  - package path: `experiments/exp061_seedbag_anchor_model_diff_distance_gate/kaggle/train`
  - kernel id: `kentookumura/exp061-seedbag-diff-gate-train`
  - title: `exp061 seedbag diff gate train`
  - push command: `kaggle kernels push -p experiments/exp061_seedbag_anchor_model_diff_distance_gate/kaggle/train`
  - push result: `Kernel version 1 successfully pushed`
  - URL: `https://www.kaggle.com/code/kentookumura/exp061-seedbag-diff-gate-train`
  - existence check: `kaggle kernels pull kentookumura/exp061-seedbag-diff-gate-train -p /tmp/kaggle-pull/exp061-seedbag-diff-gate-train -m` succeeded
  - `kaggle kernels logs kentookumura/exp061-seedbag-diff-gate-train`: empty immediately after push
  - `timeout 180 kaggle kernels logs -f --interval 10 kentookumura/exp061-seedbag-diff-gate-train`: timed out with no CLI log output
  - `kaggle kernels output kentookumura/exp061-seedbag-diff-gate-train -p /tmp/kaggle-output/exp061_seedbag_anchor_model_diff_distance_gate/train_v1_probe`: no files yet
  - auxiliary status: `KernelWorkerStatus.RUNNING`
- Kaggle train v1: completed
  - completion reported by user on 2026-06-11
  - logs command: `kaggle kernels logs kentookumura/exp061-seedbag-diff-gate-train`
  - output command: `kaggle kernels output kentookumura/exp061-seedbag-diff-gate-train -p /tmp/kaggle-output/exp061_seedbag_anchor_model_diff_distance_gate/train_v1`
  - auxiliary status: `KernelWorkerStatus.COMPLETE`
  - output: `/tmp/kaggle-output/exp061_seedbag_anchor_model_diff_distance_gate/train_v1`
  - synced local artifacts:
    - `metrics.json`
    - `artifacts/exp061-seedbag-diff-gate-train.log`
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

## 結果

- rows / wells: 1,782,279 / 773
- selected candidate: `lgbm_capacity_pf_confidence_only_seedbag_gate_near_mid_a0p50_far0`
- selected original-fold RMSE: 14.872556
- selected well-hash RMSE: 14.737595
- best original-fold candidate: `lgbm_capacity_pf_model_diff_foldsafe_seedbag_gate_near_mid_a0p50_far0` 14.838812
- best well-hash candidate: `lgbm_capacity_pf_model_diff_foldsafe_raw` 14.735200
- pure model-diff gated `near_mid_a0p50_far0`: 14.838812 original-fold / 14.791874 well-hash
- exp059 raw control: 15.037567 original-fold / 14.735200 well-hash
- exp054 foldout control: 15.368749 original-fold / 15.583832 well-hash
- `pf090_hold010`: 15.089532 original-fold / 15.089532 well-hash

Interpretation:

- Seed-bag anchor distance gate improved exp054 foldout control on both holdouts.
- Pure model-diff gate is strongest on original-fold, but does not beat exp059 raw on well-hash.
- The stable selected candidate is `confidence_only + seedbag gate near_mid_a0p50_far0`.
- `config.yaml` inference selection was updated to the selected candidate and profile.
- Post-result checks:
  - `uv run python scripts/validate_experiment.py --experiment exp061_seedbag_anchor_model_diff_distance_gate`: PASS
  - `python -m py_compile experiments/exp061_seedbag_anchor_model_diff_distance_gate/pf_model_diff_model_audit.py experiments/exp061_seedbag_anchor_model_diff_distance_gate/pf_model_diff_inference.py experiments/exp061_seedbag_anchor_model_diff_distance_gate/settings.py`: PASS
- Inference package regenerated after selected candidate update:
  - command: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp061_seedbag_anchor_model_diff_distance_gate --notebook inference --kernel-id kentookumura/exp061-seedbag-diff-gate-infer --title "exp061 seedbag diff gate infer" --run-on-push --strict`
  - path: `experiments/exp061_seedbag_anchor_model_diff_distance_gate/kaggle/inference`
- Kaggle inference v1: completed
  - push command: `kaggle kernels push -p experiments/exp061_seedbag_anchor_model_diff_distance_gate/kaggle/inference`
  - push result: `Kernel version 1 successfully pushed`
  - URL: `https://www.kaggle.com/code/kentookumura/exp061-seedbag-diff-gate-infer`
  - existence check: `kaggle kernels pull kentookumura/exp061-seedbag-diff-gate-infer -p /tmp/kaggle-pull/exp061-seedbag-diff-gate-infer -m` succeeded
  - logs command: `timeout 180 kaggle kernels logs -f --interval 10 kentookumura/exp061-seedbag-diff-gate-infer`
  - output command: `kaggle kernels output kentookumura/exp061-seedbag-diff-gate-infer -p /tmp/kaggle-output/exp061_seedbag_anchor_model_diff_distance_gate/inference_v1`
  - auxiliary status: `KernelWorkerStatus.COMPLETE`
  - selected candidate: `lgbm_capacity_pf_confidence_only_seedbag_gate_near_mid_a0p50_far0`
  - selected postprocess/profile: `seedbag_distance_gate` / `near_mid_a0p50_far0`
  - submission rows: 14,151
  - submit-check: PASS
    - `uv run python scripts/validate_submission.py --submission /tmp/kaggle-output/exp061_seedbag_anchor_model_diff_distance_gate/inference_v1/submission.csv`
    - `uv run python .agents/skills/kaggle-submit-check/scripts/check_submission.py /tmp/kaggle-output/exp061_seedbag_anchor_model_diff_distance_gate/inference_v1/submission.csv --sample data/raw/sample_submission.csv`
  - SHA256: `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`
  - public sample branch summary:
    - rows: 14,151
    - branch: `physical_visible` for all 3 wells
    - changed_rows: 0
    - changed_wells: 0
    - prediction range: 11587.038593 to 12240.016066
    - diff RMSE vs original selector output: 0.000000
  - synced local artifacts:
    - `artifacts/exp061-seedbag-diff-gate-infer.log`
    - `artifacts/pf_model_diff_inference_source_summary.csv`
    - `artifacts/pf_model_diff_inference_summary.json`
    - `artifacts/pf_model_diff_inference_wells.csv`
    - `artifacts/lgbm_pf_model_diff_corrected_summary.json`
    - `artifacts/lgbm_pf_model_diff_corrected_diff.csv`
- Kaggle code submission: completed
  - latest submissions command: `kaggle competitions submissions rogii-wellbore-geology-prediction -v`
  - selected ref: `53581056`
  - submitted at: `2026-06-11 21:37:05.867000 UTC`
  - status: `SubmissionStatus.COMPLETE`
  - Public LB: 11.826
  - Private LB: not available
  - record command: `uv run python scripts/record_submission.py --experiment exp061_seedbag_anchor_model_diff_distance_gate --file /tmp/kaggle-output/exp061_seedbag_anchor_model_diff_distance_gate/inference_v1/submission.csv --cv 14.872556 --public-lb 11.826 --notes "..."`
  - submission log row: `SUBMISSIONS.md` v025
  - nearby prior submission: `53581051` Public 12.046, empty description. It is recorded separately as exp060 and is not treated as the selected exp061 result.
  - delta vs exp054 Public LB 11.856: -0.030
  - delta vs exp059 Public LB 11.878: -0.052
  - delta vs exp039 Public LB 11.740: +0.086

## 次のアクション

1. exp061 is now the pseudo-tail self-route Public LB anchor at 11.826.
2. Do not promote it to the overall ML route anchor because exp039 remains better at 11.740.
3. If continuing this line, investigate why selected confidence-only gate transferred better than exp059 raw model-diff on Public LB.
