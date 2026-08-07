# exp073_gpu_reproducibility_guard_for_exp063_full_replay Result

## Status

Kaggle GPU train v2, deterministic GPU inference v1/v2, CPU train v1, and deterministic CPU inference v2 completed.

This experiment is the corrected end-to-end full-replay reproducibility guard for exp063. Train v2 was pushed after exp072 deterministic v2 completed and uses exp072's latest full replay train feature cache as a Kaggle kernel source.

The stopped train v1 log confirmed that train used the fixed exp072 cache:

- source: `kentookumura/exp072-exp063-full-replay-feature-cache-train`
- rows: 3,783,989
- features: 196
- mode: `gpu_repro_guard_dp_threads8`

No PF/Beam generation ran inside exp073 train v1. It entered LightGBM training and was stopped before metrics/model artifacts were written. Train v2 reran the same LightGBM guard using exp072 deterministic v2 cache SHA `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`.

## GPU Train v2

The GPU train v2 run verified:

- exp072 source cache SHA: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`.
- rows / wells / features: 3,783,989 / 773 / 196.
- elapsed seconds: 8246.788.
- saved booster manifest: 15 models, 15 SHA values present.

Pooled RMSE:

- `lgb_mean`: 9.526374749390682, prediction SHA `238b7cfec82454096961cd58e2f504ee2d96fc191aa357890b55bd13aacd9c6b`.
- `lgb1`: 9.570466226132597, prediction SHA `0a1f0caddc2f644cb71de0fd930b693d7b2271380142af65f9490a22a95f4e1b`.
- `lgb2`: 9.57705015515117, prediction SHA `455a12d42a84e86e0365beef4e7c2bf996887411c00ab79c641a8cd24facb074`.
- `lgb0`: 9.664291131259999, prediction SHA `4add67e488f44115130fa35bfba7f2c1ecaf073653952b0a79f92b3a53bc3bb1`.

## Inference v1

The inference package was regenerated with the exp072 deterministic PF/Beam implementation and pushed as `kentookumura/exp073-full-replay-repro-guard-infer` v1. It regenerated current raw test PF/Beam/likelihood-PF features with stable SHA256-derived per-well seeds, then applied exp073 GPU train v2 saved boosters.

- status: `inference_completed`
- rows / wells / features: 14,151 / 3 / 196
- test feature SHA: `8cd0134a646dcd6340e48b579d4a9f099b1bff5ed592133391708b1e7f4f99d0`
- selected model: `gpu_repro_guard_dp_threads8` / `lgb_mean`
- fallback rows: 0
- prediction SHA: `2e47e986c013acfafaa01c652d47649778db5616e40dc3130e4e12dede7b7502`
- submission SHA: `7335854727543eff5db04873154394acae83274b18e73ed68d76491c4504788b`
- elapsed seconds: 154.516
- submit-check: PASS

Inference was rerun as v2 with the same package to verify determinism. `submission.csv` was byte-identical between v1 and v2 and both runs produced submission SHA `7335854727543eff5db04873154394acae83274b18e73ed68d76491c4504788b`. The decompressed feature CSV content SHA was also identical (`e3567a64807a16c3c4d80fe6bca2611ba3fe8d13b4b20be4540e8d1ac354965c`). The raw gzip file SHA differed, so raw `.csv.gz` SHA should not be used as the feature-content determinism key.

## CPU Train v1

The CPU deterministic train package completed as `kentookumura/exp073-full-replay-repro-guard-train-cpu` v1. It used the same exp072 deterministic cache SHA `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`.

- active mode: `cpu_deterministic_threads8`
- rows / wells / features: 3,783,989 / 773 / 196
- elapsed seconds: 11020.783
- saved booster manifest: 15 models, 15 SHA values present

Pooled CPU RMSE:

- `lgb_mean`: 9.540138463736211, prediction SHA `c62fb008e8b51c7f3e74fb29899f56596eefa26bf651e16b3f80fab5b2c81ecb`.
- `lgb2`: 9.568678984737119, prediction SHA `c79994c5d31874ceeb8125a77d0420ca9f47dba24e59da66ef5b281813754f8e`.
- `lgb1`: 9.589965477772088, prediction SHA `e98b7eaeb110ec63b7201712542bf80cf1a79bd535f3df6d54fea1b6f872978b`.
- `lgb0`: 9.689599715432188, prediction SHA `bc15013f33cfb3fa4e53febb314b5cd88c968044f05b441581203291d0358fb9`.

CPU versus GPU pooled RMSE deltas:

- `lgb_mean`: +0.013763714345529365
- `lgb1`: +0.019499251639491177
- `lgb2`: -0.008371170414051221
- `lgb0`: +0.02530858417218873

CPU deterministic control is close but not bitwise / metric-identical to GPU DP deterministic mode. The GPU v2 boosters remain the selected inference source because inference v1/v2 produced byte-stable submissions.

## CPU Inference

CPU deterministic inference completed as `kentookumura/exp073-full-replay-repro-guard-infer-cpu` v2. v1 failed because the copied notebook still contained the old bootstrap support ZIP with GPU inference config; the bootstrap ZIP was rebuilt from `kaggle/inference_cpu` files and repushed as v2.

- kernel id: `kentookumura/exp073-full-replay-repro-guard-infer-cpu`
- kernel source: `kentookumura/exp073-full-replay-repro-guard-train-cpu`
- metadata: `enable_gpu=false`, `enable_internet=false`, `run_on_push=true`
- selected model: `cpu_deterministic_threads8` / `lgb_mean`
- feature generation: raw-test PF/Beam/likelihood-PF replay with stable per-well seeds and `use_gpu=cpu`
- rows / wells / features: 14,151 / 3 / 196
- fallback rows: 0
- elapsed seconds: 150.987
- decompressed feature CSV content SHA: `aae82190b23af9ec0d2a9d064a87bdf5fd39cbc427da4a1515d313b3cd3e815e`
- prediction SHA: `b16035e8432b6dea7e8d29b1fcd8e91d51bb3257d13afa7bd6f5133f8340fee5`
- submission SHA: `ce17f22241de85e301cae0c6241630ff50bdef8b7b685afbe6ddcd937dc46df2`
- submit-check: PASS

CPU vs GPU submission diff:

- absolute mean: 0.19847654593756625
- absolute max: 0.982421875
- mean: -0.12577657111246554

## Next

Submit the deterministic GPU or CPU inference output only when LB verification is desired.
