# exp073_gpu_reproducibility_guard_for_exp063_full_replay セッションノート

## 現在の状態

- status: `repro_check_passed_cpu_infer_v2_completed_pending_optional_lb`
- route: `ml_model`
- parent: `exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- Kaggle GPU train: v2 `COMPLETED`
- Kaggle CPU train: v1 `COMPLETED`
- Kaggle GPU inference: v1/v2 `COMPLETED`, byte-stable submission confirmed.
- Kaggle CPU inference: v2 `COMPLETED`
- blocked: none; optional competition submit is pending.

## 実装内容

- `docs/legacy/steering/20260614-exp073-gpu-reproducibility-guard-for-exp063-full-replay/` を作成。
- `experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay/` を exp070 から作成し、exp073 用に差し替え。
- `exp063_full_replay_reproducibility_guard.py` を実装。
  - train は exp072 の full replay train cache を読む。
  - feature count が 196 でない場合は fail する。
  - GroupKFold by `well`、target `TVT - last_known_tvt` で exp063 Pixiux LightGBM 3 configs を再学習する。
  - `gpu_repro_guard_dp_threads8` を既定 active mode にする。
  - metrics、by-well、OOF predictions、feature schema、model manifest、prediction SHA、model SHA を保存する。
  - inference は current raw test files から exp063 public replay PF/Beam/likelihood-PF features を再生成し、exp073 saved boosters を適用する。
  - inference の再生成 test features は compact tracker subset ではなく、train manifest と同じ `pixiux_likpf_public_replay` 196-feature schema を保存して使う。
- train/inference notebook を exp073 名に更新。
- Kaggle notebook packages を生成済み。
- 2026-06-15 に exp073 の正式スコープを end-to-end reproducibility guard に修正。
  - LightGBM training reproducibility だけでなく、inference の raw-test PF/Beam/likelihood-PF feature regeneration も対象に含める。
  - `public_notebook_replay_audit.py` を exp072 deterministic v2 と同じ stable per-well seed 実装へ差し替え。
  - inference package を再生成し、`stable_seed`、`_pf_ancc_seeded`、`_pf_z_seeded`、`stable_seed("likpf", split, wid)` が Kaggle package に含まれることを確認。
- 2026-06-15 に CPU deterministic train package を作成。
  - path: `experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay/kaggle/train_cpu`
  - kernel id: `kentookumura/exp073-full-replay-repro-guard-train-cpu`
  - title: `exp073 full replay repro guard train cpu`
  - metadata: `enable_gpu=false`, `enable_internet=false`, `run_on_push=true`
  - embedded `config.yaml`: `model.training.active_modes=[cpu_deterministic_threads8]`, `runtime.kaggle.enable_gpu=false`
  - source config と通常 `kaggle/train` package は GPU 既定 `gpu_repro_guard_dp_threads8` / `enable_gpu=true` に戻し済み。
- 2026-06-15 に CPU deterministic inference package を作成。
  - path: `experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay/kaggle/inference_cpu`
  - kernel id: `kentookumura/exp073-full-replay-repro-guard-infer-cpu`
  - title: `exp073 full replay repro guard infer cpu`
  - metadata: `enable_gpu=false`, `enable_internet=false`, `run_on_push=true`
  - kernel source: `kentookumura/exp073-full-replay-repro-guard-train-cpu`
  - embedded `config.yaml`: `inference.selected_mode=cpu_deterministic_threads8`, `inference.selected_model=lgb_mean`, `inference.feature_generation.use_gpu=cpu`
  - purpose: regenerate raw-test PF/Beam/likelihood-PF features with the same stable per-well seeds and apply CPU deterministic boosters for comparison against the GPU inference output.
- CPU deterministic inference v1 を Kaggle に push。
  - kernel id: `kentookumura/exp073-full-replay-repro-guard-infer-cpu`
  - result: failed.
  - cause: copied notebook still contained the old bootstrap support ZIP with GPU inference config, so runtime selected `gpu_repro_guard_dp_threads8` while reading the CPU train manifest.
  - error: `ValueError: No saved models for mode=gpu_repro_guard_dp_threads8 model=lgb_mean`
  - fix: rebuilt the notebook bootstrap support ZIP from `kaggle/inference_cpu` package files, then repushed as v2.
- CPU deterministic inference v2 を Kaggle に push し、完了を確認。
  - kernel id: `kentookumura/exp073-full-replay-repro-guard-infer-cpu`
  - URL: `https://www.kaggle.com/code/kentookumura/exp073-full-replay-repro-guard-infer-cpu`
  - kernel source: `kentookumura/exp073-full-replay-repro-guard-train-cpu`
  - metadata: `enable_gpu=false`, `enable_internet=false`, `run_on_push=true`
  - selected model: `cpu_deterministic_threads8` / `lgb_mean`
  - pull existence check: PASS at `/tmp/kaggle-pull/exp073-full-replay-repro-guard-infer-cpu-v2`
  - output path: `/tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/inference_cpu_v2`
  - rows/wells/features: 14,151 / 3 / 196
  - fallback rows: 0
  - elapsed seconds: `150.987`
  - elapsed feature seconds: `107.535`
  - raw gzip feature SHA: `79eea169e1e9bb138fcf9ce579deb3c12ee01e95a6ad0eef73411ef1aab7386d`
  - decompressed feature CSV content SHA: `aae82190b23af9ec0d2a9d064a87bdf5fd39cbc427da4a1515d313b3cd3e815e`
  - decompressed prediction CSV content SHA: `fcb41e5d807826369c14c14a9b4f4b28c590e9d9bc2d7d7226d173ef1c395e43`
  - prediction SHA: `b16035e8432b6dea7e8d29b1fcd8e91d51bb3257d13afa7bd6f5133f8340fee5`
  - submission SHA: `ce17f22241de85e301cae0c6241630ff50bdef8b7b685afbe6ddcd937dc46df2`
  - submit-check: PASS
  - CPU minus GPU submission diff: abs mean `0.19847654593756625`, abs max `0.982421875`, mean `-0.12577657111246554`
- CPU deterministic train v1 を Kaggle に push。
  - kernel id: `kentookumura/exp073-full-replay-repro-guard-train-cpu`
  - URL: `https://www.kaggle.com/code/kentookumura/exp073-full-replay-repro-guard-train-cpu`
  - kernel source: `kentookumura/exp072-exp063-full-replay-feature-cache-train`
  - metadata: `enable_gpu=false`, `enable_internet=false`, `run_on_push=true`
  - active mode: `cpu_deterministic_threads8`
  - push result: `Kernel version 1 successfully pushed`
  - pull existence check: PASS at `/tmp/kaggle-pull/exp073-full-replay-repro-guard-train-cpu-v1`
  - initial normal logs: empty
  - 3 minute `logs -f --interval 15` polling: no output before timeout; treated as Kaggle API log lag or still-running state, not failure.
  - post-poll output probe: `/tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/train_cpu_v1_probe` contained no files yet.
- exp072 完了確認後、train v1 を Kaggle に push。
  - kernel id: `kentookumura/exp073-full-replay-repro-guard-train`
  - URL: `https://www.kaggle.com/code/kentookumura/exp073-full-replay-repro-guard-train`
  - kernel source: `kentookumura/exp072-exp063-full-replay-feature-cache-train`
  - metadata: `enable_gpu=true`, `enable_internet=false`, `run_on_push=true`
  - pull existence check: PASS at `/tmp/kaggle-pull/exp073-full-replay-repro-guard-train-v1`
  - initial normal logs: empty
  - short `logs -f --interval 10` polling: no output before timeout; treated as Kaggle API log lag, not failure.
  - later logs confirmed:
    - exp072 cache path: `/kaggle/input/notebooks/kentookumura/exp072-exp063-full-replay-feature-cache-train/artifacts/exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz`
    - cache columns preview: 199 columns (`id`, `well`, `target` + 196 features)
    - LightGBM config print: `{"configs": 3, "features": 196, "mode": "gpu_repro_guard_dp_threads8", "rows": 3783989, "use_gpu": true}`
    - no `building ... public replay` / PF / Beam generation logs in train v1.
  - user manually stopped v1 before any metrics/model artifacts were written.
  - output downloaded to `/tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/train_v1_stopped`; only support files/log were present, no metrics or model manifest.
- exp072 deterministic v2 完了後、train v2 を Kaggle に push。
  - kernel id: `kentookumura/exp073-full-replay-repro-guard-train`
  - URL: `https://www.kaggle.com/code/kentookumura/exp073-full-replay-repro-guard-train`
  - kernel source: `kentookumura/exp072-exp063-full-replay-feature-cache-train` latest v2
  - expected exp072 source SHA: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
  - pull existence check: PASS at `/tmp/kaggle-pull/exp073-full-replay-repro-guard-train-v2`
  - initial normal logs: empty
  - 3 minute `logs -f --interval 15` polling: no output before timeout; treated as Kaggle API log lag, not failure.
  - post-scope-correction probe:
    - `kaggle kernels logs kentookumura/exp073-full-replay-repro-guard-train`: empty
    - `kaggle kernels output ... -p /tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/train_v2_probe`: no files yet
    - interpretation: train v2 completion is still not confirmed; keep inference push waiting for model artifacts.
- exp073 GPU train v2 完了を確認し、output を取得。
  - output path: `/tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/train_v2`
  - artifacts:
    - `artifacts/exp063_full_replay_repro_guard_metrics.csv`
    - `artifacts/exp063_full_replay_repro_guard_by_well.csv`
    - `artifacts/exp063_full_replay_repro_guard_predictions.csv.gz`
    - `artifacts/exp063_full_replay_repro_guard_feature_schema.csv`
    - `artifacts/exp063_full_replay_repro_guard_summary.json`
    - `artifacts/exp063_full_replay_repro_guard_lgb_models/manifest.json`
    - 15 saved LightGBM booster files under `gpu_repro_guard_dp_threads8/`
  - rows/wells/features: 3,783,989 / 773 / 196
  - exp072 source cache SHA: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
  - exp072 schema SHA: `700d38149f583c3ab6574ea7b163c3c8709c2514b675bea381d822f82f4809b8`
  - exp072 summary SHA: `133f9be7a6bcf8606e18b7d41f4d24d84e1d8e0f128660717b21fea4fad46b7f`
  - elapsed_seconds: `8246.788`
  - pooled CV:
    - `lgb_mean`: RMSE `9.526374749390682`, prediction SHA `238b7cfec82454096961cd58e2f504ee2d96fc191aa357890b55bd13aacd9c6b`
    - `lgb1`: RMSE `9.570466226132597`, prediction SHA `0a1f0caddc2f644cb71de0fd930b693d7b2271380142af65f9490a22a95f4e1b`
    - `lgb2`: RMSE `9.57705015515117`, prediction SHA `455a12d42a84e86e0365beef4e7c2bf996887411c00ab79c641a8cd24facb074`
    - `lgb0`: RMSE `9.664291131259999`, prediction SHA `4add67e488f44115130fa35bfba7f2c1ecaf073653952b0a79f92b3a53bc3bb1`
  - model manifest: 15 models, 15 model SHA values present.
- exp073 deterministic inference v1 を Kaggle に push し、完了を確認。
  - kernel id: `kentookumura/exp073-full-replay-repro-guard-infer`
  - URL: `https://www.kaggle.com/code/kentookumura/exp073-full-replay-repro-guard-infer`
  - kernel source: `kentookumura/exp073-full-replay-repro-guard-train`
  - metadata: `enable_gpu=true`, `enable_internet=false`, `run_on_push=true`
  - deterministic PF: `true`
  - seed policy: `stable_sha256_per_well`
  - push result: `Kernel version 1 successfully pushed`
  - pull existence check: PASS at `/tmp/kaggle-pull/exp073-full-replay-repro-guard-infer-v1`
  - output path: `/tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/inference_v1`
  - status: `inference_completed`
  - rows/wells/features: 14,151 / 3 / 196
  - test feature SHA: `8cd0134a646dcd6340e48b579d4a9f099b1bff5ed592133391708b1e7f4f99d0`
  - selected model: `gpu_repro_guard_dp_threads8` / `lgb_mean`
  - loaded models: 15
  - fallback_rows: 0
  - prediction range: min `11593.671875`, max `12241.693359375`, mean `11905.656258143197`, std `279.3037397081349`
  - prediction SHA: `2e47e986c013acfafaa01c652d47649778db5616e40dc3130e4e12dede7b7502`
  - submission SHA: `7335854727543eff5db04873154394acae83274b18e73ed68d76491c4504788b`
  - elapsed_seconds: `154.516`
  - feature_generation elapsed_seconds: `114.538`
  - submit-check: PASS for `/tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/inference_v1/submission.csv`
- exp073 deterministic inference v2 を同じ package で rerun し、再現性を確認。
  - kernel id: `kentookumura/exp073-full-replay-repro-guard-infer`
  - push result: `Kernel version 2 successfully pushed`
  - output path: `/tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/inference_v2`
  - status: `inference_completed`
  - raw gzip test feature SHA:
    - v1: `8cd0134a646dcd6340e48b579d4a9f099b1bff5ed592133391708b1e7f4f99d0`
    - v2: `4ef4339522a1ba255ca5f9c02bbe7cfc7a96ba2c66541c87e3fbf710d4bec91b`
    - interpretation: raw gzip SHA is not a stable feature-content identifier because gzip metadata/compression bytes can differ across runs.
  - decompressed test feature CSV content SHA:
    - v1: `e3567a64807a16c3c4d80fe6bca2611ba3fe8d13b4b20be4540e8d1ac354965c`
    - v2: `e3567a64807a16c3c4d80fe6bca2611ba3fe8d13b4b20be4540e8d1ac354965c`
  - decompressed test prediction CSV content SHA:
    - v1: `f3f04e56f3035191d651e330d26ee48e819f42cf0497acecefc88fe985cdc219`
    - v2: `f3f04e56f3035191d651e330d26ee48e819f42cf0497acecefc88fe985cdc219`
  - prediction SHA: both `2e47e986c013acfafaa01c652d47649778db5616e40dc3130e4e12dede7b7502`
  - submission SHA: both `7335854727543eff5db04873154394acae83274b18e73ed68d76491c4504788b`
  - `submission.csv` byte compare: identical (`cmp` exit code 0)
  - submit-check v2: PASS
  - conclusion: deterministic PF/Beam/likelihood-PF generation is supported at content level, and final submission is byte-stable across two Kaggle reruns.
- exp073 CPU deterministic train v1 完了を確認し、output を取得。
  - kernel id: `kentookumura/exp073-full-replay-repro-guard-train-cpu`
  - output path: `/tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/train_cpu_v1`
  - active mode: `cpu_deterministic_threads8`
  - rows/wells/features: 3,783,989 / 773 / 196
  - exp072 source cache SHA: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
  - elapsed_seconds: `11020.783`
  - model manifest: 15 models, 15 model SHA values present.
  - pooled CV:
    - `lgb_mean`: RMSE `9.540138463736211`, prediction SHA `c62fb008e8b51c7f3e74fb29899f56596eefa26bf651e16b3f80fab5b2c81ecb`
    - `lgb2`: RMSE `9.568678984737119`, prediction SHA `c79994c5d31874ceeb8125a77d0420ca9f47dba24e59da66ef5b281813754f8e`
    - `lgb1`: RMSE `9.589965477772088`, prediction SHA `e98b7eaeb110ec63b7201712542bf80cf1a79bd535f3df6d54fea1b6f872978b`
    - `lgb0`: RMSE `9.689599715432188`, prediction SHA `bc15013f33cfb3fa4e53febb314b5cd88c968044f05b441581203291d0358fb9`
  - CPU vs GPU pooled RMSE delta:
    - `lgb_mean`: +0.013763714345529365
    - `lgb1`: +0.019499251639491177
    - `lgb2`: -0.008371170414051221
    - `lgb0`: +0.02530858417218873
  - interpretation: CPU deterministic control is close but not bitwise / metric-identical to GPU DP deterministic mode. The GPU mode remains the selected inference source because inference v1/v2 already proved byte-stable submission output from the GPU train v2 boosters.

## 2026-06-15 determinism audit and scope correction

- User pointed out that exp070/exp073 inference regenerates hidden test PF/Beam/likelihood-PF features from raw test files and that public replay PF functions use `np.random.randn` / `np.random.uniform` without a stable per-well seed in several paths.
- Confirmed in `public_notebook_replay_audit.py`:
  - `_pf_ancc`, `_pf_z`, `_beam_jit` paths use global `np.random`.
  - `build_likpf()` uses `joblib.Parallel(... prefer="threads")`, so global RNG consumption order can depend on thread scheduling.
- Scope correction:
  - exp073 is now the full end-to-end reproducibility guard, not a LightGBM-only guard.
  - exp072 v2 first fixes train cache generation with stable per-well PF/Beam/likelihood-PF seeds and records the deterministic train cache SHA.
  - exp073 train reads that fixed exp072 v2 cache and audits LightGBM training reproducibility on the 196-feature full replay surface.
  - exp073 inference regenerates hidden/current raw test PF/Beam/likelihood-PF features with the same stable per-well seed policy, then applies exp073 saved LightGBM boosters.
- Interpretation:
  - `deterministic_pixiux_pf_beam_generation_guard` is treated as a sub-scope of exp073 for this work, not a separate future experiment.
  - Full train+infer/code-submit reproducibility will be established only after exp073 train v2 completes and deterministic inference is run at least twice or otherwise verified with feature SHA / submission SHA.

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp073_gpu_reproducibility_guard_for_exp063_full_replay
uv run python scripts/new_experiment.py --name exp073_gpu_reproducibility_guard_for_exp063_full_replay --source experiments/exp070_gpu_reproducibility_guard_for_exp063
```

検証:

```bash
uv run python -m py_compile experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay/exp063_full_replay_reproducibility_guard.py experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay/public_notebook_replay_audit.py experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay/settings.py
uv run ruff check experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay/exp063_full_replay_reproducibility_guard.py experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay/public_notebook_replay_audit.py experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay/settings.py
uv run python -m json.tool experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay/exp073_gpu_reproducibility_guard_for_exp063_full_replay_train.ipynb
uv run python -m json.tool experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay/exp073_gpu_reproducibility_guard_for_exp063_full_replay_inference.ipynb
uv run python scripts/validate_experiment.py --experiment exp073_gpu_reproducibility_guard_for_exp063_full_replay
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp073_gpu_reproducibility_guard_for_exp063_full_replay --notebook train --kernel-id kentookumura/exp073-full-replay-repro-guard-train --title "exp073 full replay repro guard train" --run-on-push --strict
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp073_gpu_reproducibility_guard_for_exp063_full_replay --notebook inference --kernel-id kentookumura/exp073-full-replay-repro-guard-infer --title "exp073 full replay repro guard infer" --run-on-push --strict
kaggle kernels push -p experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay/kaggle/train
kaggle kernels pull kentookumura/exp073-full-replay-repro-guard-train -p /tmp/kaggle-pull/exp073-full-replay-repro-guard-train-v1 -m
kaggle kernels logs kentookumura/exp073-full-replay-repro-guard-train
timeout 60 kaggle kernels logs -f --interval 10 kentookumura/exp073-full-replay-repro-guard-train
kaggle kernels push -p experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay/kaggle/train
kaggle kernels pull kentookumura/exp073-full-replay-repro-guard-train -p /tmp/kaggle-pull/exp073-full-replay-repro-guard-train-v2 -m
timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp073-full-replay-repro-guard-train
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp073_gpu_reproducibility_guard_for_exp063_full_replay --notebook inference --kernel-id kentookumura/exp073-full-replay-repro-guard-infer --title "exp073 full replay repro guard infer" --run-on-push --strict
rg -n "stable_seed|_pf_ancc_seeded|_pf_z_seeded|seed_policy|deterministic_pf|seed_base=stable_seed" experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay/kaggle/inference
kaggle kernels logs kentookumura/exp073-full-replay-repro-guard-train
kaggle kernels output kentookumura/exp073-full-replay-repro-guard-train -p /tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/train_v2_probe
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp073_gpu_reproducibility_guard_for_exp063_full_replay --notebook train --kernel-id kentookumura/exp073-full-replay-repro-guard-train-cpu --title "exp073 full replay repro guard train cpu" --run-on-push --strict
cp -a experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay/kaggle/train experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay/kaggle/train_cpu
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp073_gpu_reproducibility_guard_for_exp063_full_replay --notebook train --kernel-id kentookumura/exp073-full-replay-repro-guard-train --title "exp073 full replay repro guard train" --run-on-push --strict
kaggle kernels push -p experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay/kaggle/train_cpu
kaggle kernels pull kentookumura/exp073-full-replay-repro-guard-train-cpu -p /tmp/kaggle-pull/exp073-full-replay-repro-guard-train-cpu-v1 -m
kaggle kernels logs kentookumura/exp073-full-replay-repro-guard-train-cpu
timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp073-full-replay-repro-guard-train-cpu
kaggle kernels output kentookumura/exp073-full-replay-repro-guard-train-cpu -p /tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/train_cpu_v1_probe
kaggle kernels output kentookumura/exp073-full-replay-repro-guard-train -p /tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/train_v2
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp073_gpu_reproducibility_guard_for_exp063_full_replay --notebook inference --kernel-id kentookumura/exp073-full-replay-repro-guard-infer --title "exp073 full replay repro guard infer" --run-on-push --strict
kaggle kernels push -p experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay/kaggle/inference
kaggle kernels pull kentookumura/exp073-full-replay-repro-guard-infer -p /tmp/kaggle-pull/exp073-full-replay-repro-guard-infer-v1 -m
timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp073-full-replay-repro-guard-infer
kaggle kernels output kentookumura/exp073-full-replay-repro-guard-infer -p /tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/inference_v1
uv run python scripts/validate_submission.py --submission /tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/inference_v1/submission.csv
kaggle kernels push -p experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay/kaggle/inference
timeout 240 kaggle kernels logs -f --interval 15 kentookumura/exp073-full-replay-repro-guard-infer
kaggle kernels output kentookumura/exp073-full-replay-repro-guard-infer -p /tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/inference_v2
uv run python scripts/validate_submission.py --submission /tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/inference_v2/submission.csv
kaggle kernels logs kentookumura/exp073-full-replay-repro-guard-train-cpu
kaggle kernels output kentookumura/exp073-full-replay-repro-guard-train-cpu -p /tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/train_cpu_v1
uv run python -m json.tool experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay/kaggle/inference_cpu/kernel-metadata.json
uv run python -m json.tool experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay/metrics.json
uv run python scripts/validate_experiment.py --experiment exp073_gpu_reproducibility_guard_for_exp063_full_replay
uv run python scripts/update_experiment_summary.py
kaggle kernels push -p experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay/kaggle/inference_cpu
kaggle kernels pull kentookumura/exp073-full-replay-repro-guard-infer-cpu -p /tmp/kaggle-pull/exp073-full-replay-repro-guard-infer-cpu-v1 -m
kaggle kernels logs kentookumura/exp073-full-replay-repro-guard-infer-cpu
kaggle kernels output kentookumura/exp073-full-replay-repro-guard-infer-cpu -p /tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/inference_cpu_v1_probe
timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp073-full-replay-repro-guard-infer-cpu
uv run python -m json.tool experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay/kaggle/inference_cpu/exp073_gpu_reproducibility_guard_for_exp063_full_replay_inference.ipynb
kaggle kernels push -p experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay/kaggle/inference_cpu
kaggle kernels pull kentookumura/exp073-full-replay-repro-guard-infer-cpu -p /tmp/kaggle-pull/exp073-full-replay-repro-guard-infer-cpu-v2 -m
timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp073-full-replay-repro-guard-infer-cpu
kaggle kernels output kentookumura/exp073-full-replay-repro-guard-infer-cpu -p /tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/inference_cpu_v2
uv run python scripts/validate_submission.py --submission /tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/inference_cpu_v2/submission.csv
```

## 次のアクション

1. LB 確認する場合は GPU deterministic submission `/tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/inference_v1/submission.csv` または CPU deterministic submission `/tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/inference_cpu_v2/submission.csv` を submit する。

## 2026-06-20: last_known_tvt 距離別 RMSE 図

- 目的: exp073 train_v2 OOF の `lgb_mean` について、`target_tvt - last_known_tvt` の絶対距離ビン別に TVT RMSE を確認する。
- 入力: `/tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/train_v2/artifacts/exp063_full_replay_repro_guard_predictions.csv.gz`
- 実行:

```bash
uv run --with matplotlib python experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay/plot_last_known_tvt_distance_rmse.py
uv run python -m py_compile experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay/plot_last_known_tvt_distance_rmse.py
```

- 生成物:
  - `artifacts/exp073_lgb_mean_rmse_by_last_known_tvt_abs_distance.png`
  - `artifacts/exp073_lgb_mean_rmse_by_last_known_tvt_abs_distance.csv`
  - `artifacts/exp073_lgb_mean_rmse_by_last_known_tvt_signed_distance.csv`
- 概要: overall RMSE 9.5264。絶対距離 0-1 は RMSE 4.7416、10-20 は 8.3013、20-40 は 13.0265、40-60 は 26.9237、80+ は 59.1912。

### tail by-well / signed distance 追加確認

- 実行: exp073 train_v2 OOF `lgb_mean` から `|target_tvt-last_known_tvt| >= 20` / `>= 40` の by-well と signed distance 寄与率を集計。
- 生成物:
  - `artifacts/exp073_lgb_mean_tail_abs_ge20_by_well.csv`
  - `artifacts/exp073_lgb_mean_tail_abs_ge40_by_well.csv`
  - `artifacts/exp073_lgb_mean_rmse_by_last_known_tvt_signed_distance_contribution.csv`
- `abs>=20`: 599,906 rows、407 wells、RMSE 18.1072、全体 MSE 寄与 57.28%。top5 wells が tail MSE の 31.03%、top20 が 55.91%。
- `abs>=40`: 101,367 rows、80 wells、RMSE 33.2540、全体 MSE 寄与 32.64%。top5 wells が tail MSE の 52.81%、top20 が 86.14%。
- signed distance: 正距離 long tail では error_mean が負、負距離 long tail では error_mean が正。遠距離で予測が `last_known_tvt` 側へ縮む傾向が強い。

### 訂正: last known からの距離は TVT 差ではなく row step

- 上記 `target_tvt-last_known_tvt` 診断は TVT delta 診断であり、ユーザー意図の last known からの距離ではなかった。
- 正しい step 距離は `id` の suffix row index と raw train horizontal well の `TVT_input` 最終非欠損 index から `row_index - last_known_index` として復元。
- 実行:

```bash
uv run --with matplotlib python experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay/plot_last_known_step_rmse.py
uv run python -m py_compile experiments/exp073_gpu_reproducibility_guard_for_exp063_full_replay/plot_last_known_step_rmse.py
```

- 生成物:
  - `artifacts/exp073_lgb_mean_rmse_by_last_known_step.png`
  - `artifacts/exp073_lgb_mean_rmse_by_last_known_step.csv`
  - `artifacts/exp073_lgb_mean_tail_step_ge500_by_well.csv`
  - `artifacts/exp073_lgb_mean_tail_step_ge1000_by_well.csv`
  - `artifacts/exp073_lgb_mean_tail_step_ge1500_by_well.csv`
- step bucket 概要: `1501_plus` が 2,626,171 rows / 69.40% を占め、RMSE 10.86、MSE 寄与 90.12%。step tail は希少 tail ではなく評価領域の大半。
- by-well 概要: `step>=1500` では top5 wells が tail MSE の 18.96%、top20 が 35.43%。TVT delta `abs>=40` のような少数 well 集中ではない。
