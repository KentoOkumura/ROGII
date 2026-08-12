# exp070_gpu_reproducibility_guard_for_exp063 セッションノート

## 現在の状態

- status: `discarded_superseded_invalid_feature_surface`
- route: `ml_model`
- parent: `exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit`
- Kaggle train: GPU v4 `COMPLETE`; CPU v2 `COMPLETE`
- Kaggle inference: raw-test PF/Beam regeneration v1 `COMPLETE`
- decision: exp070 is discarded for the exp063 reproducibility guard. It used the compact tracker output with 65 features, while the intended exp063 full public replay surface has 196 features. Keep the run logs as a runtime/reference audit only; do not use exp070 CV/LB as evidence for exp063 reproducibility.
- replacement path: use exp072 train-side full replay feature cache, then implement a corrected full-replay LightGBM GPU reproducibility guard. Inference must regenerate PF/Beam/likelihood-PF features from current raw test inside the downstream inference notebook.

## 実装内容

- exp063 から実験フォルダをコピーして exp070 を作成。
- `docs/legacy/steering/20260613-exp070-gpu-reproducibility-guard-for-exp063/` を作成。
- `config.yaml` を exp063 output 固定の LightGBM reproducibility guard に置換。
- `exp063_reproducibility_guard.py` を追加。
  - exp063 の `ravaghi_vs_pixiux_public_replay_tracker_features_train.csv.gz` を読む。
  - train では raw PF/Beam / likelihood-PF feature generation は実行しない。
  - inference では exp063 の public replay 実装で current raw test から PF/Beam / likelihood-PF test features を再生成する。
  - GroupKFold by `well` で LightGBM 3 configs を再学習する。
  - `gpu_repro_guard_dp_threads8` と `cpu_deterministic_threads8` を標準 mode にする。
  - metrics、by-well、OOF predictions、feature schema、model manifest、prediction SHA、model SHA を保存する。
- train notebook を exp070 用の再現性ガード実行へ差し替え。
- inference notebook は exp070 saved boosters と regenerated exp063 test features で `submission.csv` を生成する構成に変更。

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp070_gpu_reproducibility_guard_for_exp063
uv run python scripts/new_experiment.py --name exp070_gpu_reproducibility_guard_for_exp063 --source experiments/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit
```

実装後の確認:

```bash
uv run python -m py_compile experiments/exp070_gpu_reproducibility_guard_for_exp063/exp063_reproducibility_guard.py experiments/exp070_gpu_reproducibility_guard_for_exp063/settings.py
uv run ruff check experiments/exp070_gpu_reproducibility_guard_for_exp063/exp063_reproducibility_guard.py experiments/exp070_gpu_reproducibility_guard_for_exp063/settings.py
uv run python -m json.tool experiments/exp070_gpu_reproducibility_guard_for_exp063/exp070_gpu_reproducibility_guard_for_exp063_train.ipynb
uv run python -m json.tool experiments/exp070_gpu_reproducibility_guard_for_exp063/exp070_gpu_reproducibility_guard_for_exp063_inference.ipynb
uv run python scripts/validate_experiment.py --experiment exp070_gpu_reproducibility_guard_for_exp063
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp070_gpu_reproducibility_guard_for_exp063 --notebook train --kernel-id kentookumura/exp070-gpu-repro-guard-exp063-train --title "exp070 gpu repro guard exp063 train" --run-on-push --strict
```

Kaggle train 実行候補:

```bash
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp070_gpu_reproducibility_guard_for_exp063 --notebook train --kernel-id kentookumura/exp070-gpu-repro-guard-exp063-train --title "exp070 gpu repro guard exp063 train" --run-on-push --strict
kaggle kernels push -p experiments/exp070_gpu_reproducibility_guard_for_exp063/kaggle/train
kaggle kernels pull kentookumura/exp070-gpu-repro-guard-exp063-train -p /tmp/kaggle-pull/exp070-gpu-repro-guard-exp063-train-v1 -m
kaggle kernels logs kentookumura/exp070-gpu-repro-guard-exp063-train
kaggle kernels status kentookumura/exp070-gpu-repro-guard-exp063-train
kaggle kernels output kentookumura/exp070-gpu-repro-guard-exp063-train -p /tmp/kaggle-output/exp070_gpu_reproducibility_guard_for_exp063/train_v1_probe
timeout 90 kaggle kernels logs -f --interval 10 kentookumura/exp070-gpu-repro-guard-exp063-train
```

## 結果

- 実装と静的検証まで完了。
- Final decision: discarded/superseded for the original exp063 reproducibility purpose because the train feature surface is 65 compact tracker features, not the exp063 full replay 196-feature surface.
- `py_compile`: PASS
- `ruff check`: PASS
- train notebook JSON validation: PASS
- inference notebook JSON validation: PASS
- `validate_experiment.py`: PASS
- Kaggle train package: generated at `experiments/exp070_gpu_reproducibility_guard_for_exp063/kaggle/train`
- Kaggle metadata:
  - kernel id: `kentookumura/exp070-gpu-repro-guard-exp063-train`
  - `enable_gpu=true`
  - `enable_internet=false`
  - `run_on_push=true`
  - kernel source: `kentookumura/exp063-ravaghi-pixiux-strict-replay-train`
- Kaggle train version 1 pushed successfully:
  - URL: `https://www.kaggle.com/code/kentookumura/exp070-gpu-repro-guard-exp063-train`
  - pull existence check: PASS at `/tmp/kaggle-pull/exp070-gpu-repro-guard-exp063-train-v1`
  - final status: `KernelWorkerStatus.COMPLETE`
  - normal logs: empty so far
  - 90 sec `logs -f`: empty before timeout
  - final output downloaded to `/tmp/kaggle-output/exp070_gpu_reproducibility_guard_for_exp063/train_v1`
  - small audit artifacts synced into `artifacts/`; large OOF/model outputs remain under `/tmp/kaggle-output/.../train_v1`.
- Kaggle train v1 results:
  - rows / features: 3,783,989 / 65
  - caveat: invalid for main reproducibility comparison because `well` was read without `dtype=str`; local audit showed dtype-default read can produce 776 groups while dtype-fixed read produces 773.
  - source tracker feature SHA: `4ebf8f4fec0be09fba5c9c585d3699a78fbc6511b16b066098a7ca65362c5f90`
  - elapsed: 14,492.654 sec
  - model count: 30
  - `gpu_repro_guard_dp_threads8` `lgb_mean` RMSE: 9.732865121, prediction SHA `714a34feab4bbc8a3a18ddc2de1754458406dcfafecf1f5ef3e0429866c0897e`
  - `cpu_deterministic_threads8` `lgb_mean` RMSE: 9.729912845, prediction SHA `05e264f34db7184c08c2b9e439cb030d945fb12ec418c233fd5523fbb7e54098`
- Kaggle train version 2 pushed successfully with the same prepared package.
  - pull existence check: PASS at `/tmp/kaggle-pull/exp070-gpu-repro-guard-exp063-train-v2`
  - initial status: `KernelWorkerStatus.RUNNING`
  - user manually stopped v2.
  - same dtype bug as v1; superseded by v3.
- dtype fix:
  - `exp063_reproducibility_guard.py` now reads tracker features with `dtype={"id": str, "well": str}`.
  - local full-file check: default dtype read gives 776 groups; dtype-fixed read gives 773 groups.
  - package check confirmed dtype fix in `experiments/exp070_gpu_reproducibility_guard_for_exp063/kaggle/train/exp063_reproducibility_guard.py`.
- Kaggle train version 3 pushed successfully with dtype-fixed package.
  - pull existence check: PASS at `/tmp/kaggle-pull/exp070-gpu-repro-guard-exp063-train-v3`
  - initial status: `KernelWorkerStatus.RUNNING`
  - issue: v3 still used combined `active_modes=[gpu_repro_guard_dp_threads8, cpu_deterministic_threads8]` with `enable_gpu=true`, so CPU mode also spent GPU session wall time.
  - user manually stopped v3.
- Config correction after v3:
  - `active_modes` changed to `gpu_repro_guard_dp_threads8` only.
  - `runtime.kaggle.enable_gpu=true`.
  - CPU mode remains defined, but is not included in the GPU-enabled package.
- Corrected GPU-only run / output comparison は未実行。
- 2026-06-14 plan change:
  - User chose LB comparison instead of running the same GPU train twice.
  - Train v4 continues as the single corrected GPU train for CV and saved boosters.
  - Inference will run once from train v4 saved boosters and regenerated exp063 public replay test features from current raw test files.
  - Inference metadata is set to `enable_gpu=true` to match large-scale submission conditions.
  - Inference package generated at `experiments/exp070_gpu_reproducibility_guard_for_exp063/kaggle/inference`.
  - Inference metadata: `enable_gpu=true`, kernel id `kentookumura/exp070-gpu-repro-guard-exp063-infer`.
  - Inference kernel sources: exp070 train only; exp063 inference output is not used.
- CPU runtime train package:
  - Prepared separate package at `experiments/exp070_gpu_reproducibility_guard_for_exp063/kaggle/train_cpu`.
  - Kernel id: `kentookumura/exp070-cpu-repro-guard-exp063-train`.
  - Metadata: `enable_gpu=false`, `run_on_push=true`.
  - Active mode in package config: `cpu_deterministic_threads8` only.
  - Push command, when CPU runtime measurement is needed:
    `kaggle kernels push -p experiments/exp070_gpu_reproducibility_guard_for_exp063/kaggle/train_cpu`
  - CPU train version 1 pushed successfully:
    - URL: `https://www.kaggle.com/code/kentookumura/exp070-cpu-repro-guard-exp063-train`
    - pull existence check: PASS at `/tmp/kaggle-pull/exp070-cpu-repro-guard-exp063-train-v1`
    - initial status: `KernelWorkerStatus.RUNNING`
    - final status: `KernelWorkerStatus.ERROR`
    - failure reason: the package notebook bootstrap still embedded the stale GPU config, so the notebook printed `Active modes: ['gpu_repro_guard_dp_threads8']` and failed with `LightGBMError: No OpenCL device found` under `enable_gpu=false`.
    - interpretation: this was a packaging/config embedding error, not a CPU runtime result.
  - CPU train package regenerated after temporarily setting root `config.yaml` to CPU-only.
    - embedded bootstrap config check: train `active_modes=['cpu_deterministic_threads8']`.
    - metadata check: `enable_gpu=false`, `machine_shape=None`.
    - root `config.yaml` restored to GPU-only afterward for the main train/inference path.
  - CPU train version 2 pushed successfully:
    - URL: `https://www.kaggle.com/code/kentookumura/exp070-cpu-repro-guard-exp063-train`
    - pull existence check: PASS at `/tmp/kaggle-pull/exp070-cpu-repro-guard-exp063-train-v2`
    - initial status: `KernelWorkerStatus.RUNNING`
    - purpose: CPU-only train runtime comparison; do not use GPU quota.
- Inference correction for hidden/current test:
  - Problem found: the previous exp070 inference package read `ravaghi_vs_pixiux_public_replay_tracker_features_test.csv.gz` from exp063 output. That file is tied to the public replay sample_submission rows and overlaps exp063 train IDs, so it is not suitable as a hidden/current test feature source.
  - Fix: `public_notebook_replay_audit.py` from exp063 is now bundled into exp070, and `run_saved_model_inference()` regenerates exp063 public replay tracker/PF/Beam/likelihood-PF features from the current raw test files before applying exp070 saved boosters.
  - `inference.regenerate_test_features=true`.
  - Inference package metadata now uses only:
    - competition source: `rogii-wellbore-geology-prediction`
    - kernel source: `kentookumura/exp070-gpu-repro-guard-exp063-train`
  - The stale exp063 inference kernel source was removed from exp070 inference metadata to avoid accidental reuse of public replay test features.
  - Validation:
    - `py_compile`: PASS for `exp063_reproducibility_guard.py`, `public_notebook_replay_audit.py`, `settings.py`
    - inference notebook JSON validation: PASS
    - `ruff check`: PASS
    - `validate_experiment.py`: PASS
    - Kaggle inference package regenerated at `experiments/exp070_gpu_reproducibility_guard_for_exp063/kaggle/inference`
  - Kaggle inference version 1 pushed successfully:
    - URL: `https://www.kaggle.com/code/kentookumura/exp070-gpu-repro-guard-exp063-infer`
    - pull existence check: PASS at `/tmp/kaggle-pull/exp070-gpu-repro-guard-exp063-infer-v1`
    - initial status: `KernelWorkerStatus.RUNNING`
    - monitoring intentionally not continued.
- Completed Kaggle outputs downloaded:
  - GPU train v4: `/tmp/kaggle-output/exp070_gpu_reproducibility_guard_for_exp063/train_gpu_v4`
  - CPU train v2: `/tmp/kaggle-output/exp070_gpu_reproducibility_guard_for_exp063/train_cpu_v2`
  - Inference v1: `/tmp/kaggle-output/exp070_gpu_reproducibility_guard_for_exp063/infer_v1`
  - Small metrics/summary files synced into local `artifacts/` with run-specific names.
- GPU train v4 results:
  - status: `train_completed`
  - active mode: `gpu_repro_guard_dp_threads8`
  - rows / wells / features: 3,783,989 / 773 / 65
  - source tracker feature SHA: `4ebf8f4fec0be09fba5c9c585d3699a78fbc6511b16b066098a7ca65362c5f90`
  - elapsed: 6,900.889 sec
  - saved model count: 15
  - `lgb_mean` RMSE: 9.73150619943287
  - `lgb_mean` prediction SHA: `09ccb9edd59cd50057da0ee7738229749996219708f36e6c45f870d0efd026a5`
- CPU train v2 results:
  - status: `train_completed`
  - active mode: `cpu_deterministic_threads8`
  - rows / wells / features: 3,783,989 / 773 / 65
  - source tracker feature SHA: `4ebf8f4fec0be09fba5c9c585d3699a78fbc6511b16b066098a7ca65362c5f90`
  - elapsed: 6,308.689 sec
  - saved model count: 15
  - `lgb_mean` RMSE: 9.764917679392632
  - `lgb_mean` prediction SHA: `e09344e4fe0a8158150c60e018cb12867107b870e1bbb262dc9b46f0e8a3d557`
  - runtime comparison: CPU was 592.200 sec faster than GPU in this deterministic setup.
- Inference v1 results:
  - status: `inference_completed`
  - raw test feature regeneration source kind: `raw_test_regenerated_exp063_public_replay`
  - test rows / wells: 14,151 / 3
  - test feature generation elapsed: 94.557 sec
  - total inference elapsed: 122.117 sec
  - model count: 15
  - predicted rows / fallback rows: 14,151 / 0
  - submission SHA: `9d26b8b80df859b0e137e14e9fc3dba4acaf68252ebc2e87dc40153541be291b`
- CPU inference package and run:
  - GPU inference v1 was already confirmed complete with `source_kind=raw_test_regenerated_exp063_public_replay`.
  - Created separate CPU inference package at `experiments/exp070_gpu_reproducibility_guard_for_exp063/kaggle/inference_cpu`.
  - CPU inference metadata:
    - kernel id: `kentookumura/exp070-cpu-repro-guard-exp063-infer`
    - `enable_gpu=false`
    - kernel source: `kentookumura/exp070-cpu-repro-guard-exp063-train`
    - selected mode: `cpu_deterministic_threads8`
    - `inference.regenerate_test_features=true`
  - Restored canonical `experiments/exp070_gpu_reproducibility_guard_for_exp063/kaggle/inference` to GPU inference metadata after creating the CPU package.
  - Validation after restore: `validate_experiment.py` PASS.
  - CPU inference version 1 pushed successfully:
    - URL: `https://www.kaggle.com/code/kentookumura/exp070-cpu-repro-guard-exp063-infer`
    - pull existence check: PASS at `/tmp/kaggle-pull/exp070-cpu-repro-guard-exp063-infer-v1`
    - initial status: `KernelWorkerStatus.RUNNING`
    - monitoring intentionally not continued.
- Submit check:
  - `task submit-check ...` could not run because `task` is not installed in this environment.
  - Direct checker command:
    `python .agents/skills/kaggle-submit-check/scripts/check_submission.py /tmp/kaggle-output/exp070_gpu_reproducibility_guard_for_exp063/infer_v1/submission.csv --sample data/raw/sample_submission.csv`
  - Result: PASS
  - Checks passed: no duplicate IDs, no empty/NaN/Inf-like values, rows=14,151, columns=2, header matches sample, row count matches sample.

## 次のアクション

1. CPU inference v1 の完了後、output を取得して raw-test regeneration summary と submission diagnostics を記録する。
2. Public LB 用に GPU/CPU どちらの `submission.csv` を提出するか決めて提出する。
3. Public LB が出たら `metrics.json`、`result.md`、`experiment_summary.md`、`SUBMISSIONS.md` に反映する。
4. CPU/GPU runtime 差の解釈を次の実験方針に反映する。
