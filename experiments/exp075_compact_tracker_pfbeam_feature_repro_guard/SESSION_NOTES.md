# exp075_compact_tracker_pfbeam_feature_repro_guard セッションノート

## 現在の状態

- status: `submitted_repro_guarded_gpu_v3_public_lb_8_489`
- route: `ml_model`
- parent: `exp074_compact_tracker_surface_lgbm_candidate_audit`
- feature parent: `exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit`
- purpose: compact PF/Beam tracker feature generation を LightGBM 学習から分離し、後続実験で使い回せる train feature notebook として実装する。

## 実装内容

- `.steering/20260616-exp075-compact-tracker-pfbeam-feature-repro-guard/` を作成。
- exp074 を source として exp075 を作成。
- `compact_tracker_surface_audit.py` を `compact_tracker_pfbeam_repro_guard.py` にリネーム。
- `run_pfbeam_feature_generation()` を追加し、raw train から `ravaghi_vs_pixiux_public_replay_tracker_features_train.csv.gz` を 1 回生成する構成にした。
- `run_reproducibility_guard()` は generated train feature CSV を読む構成に変更。
- LightGBM feature importance を fold/model 単位で保存し、平均重要度トップを matplotlib PNG に保存するようにした。
- `exp075_compact_tracker_pfbeam_feature_repro_guard_pfbeam_features.ipynb` を追加。
- train notebook は generated feature CSV を読み込む構成に更新。
- inference notebook は raw test feature regeneration + saved booster inference の構成を exp075 naming に更新。
- `config.yaml` / README / result / metrics を exp075 目的へ更新。
- PF/Beam feature generation notebook は GPU を使わない方針に修正。
  - `runtime.kaggle.pfbeam_features.enable_gpu: false`
  - `feature_generation.use_gpu: cpu`

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp075_compact_tracker_pfbeam_feature_repro_guard
uv run python scripts/new_experiment.py --name exp075_compact_tracker_pfbeam_feature_repro_guard --source experiments/exp074_compact_tracker_surface_lgbm_candidate_audit
```

実装後の検証:

```bash
uv run python -m py_compile experiments/exp075_compact_tracker_pfbeam_feature_repro_guard/compact_tracker_pfbeam_repro_guard.py experiments/exp075_compact_tracker_pfbeam_feature_repro_guard/public_notebook_replay_audit.py experiments/exp075_compact_tracker_pfbeam_feature_repro_guard/settings.py
uv run ruff check experiments/exp075_compact_tracker_pfbeam_feature_repro_guard/compact_tracker_pfbeam_repro_guard.py experiments/exp075_compact_tracker_pfbeam_feature_repro_guard/public_notebook_replay_audit.py experiments/exp075_compact_tracker_pfbeam_feature_repro_guard/settings.py
uv run python -m json.tool experiments/exp075_compact_tracker_pfbeam_feature_repro_guard/exp075_compact_tracker_pfbeam_feature_repro_guard_pfbeam_features.ipynb
uv run python -m json.tool experiments/exp075_compact_tracker_pfbeam_feature_repro_guard/exp075_compact_tracker_pfbeam_feature_repro_guard_train.ipynb
uv run python -m json.tool experiments/exp075_compact_tracker_pfbeam_feature_repro_guard/exp075_compact_tracker_pfbeam_feature_repro_guard_inference.ipynb
uv run python scripts/validate_experiment.py --experiment exp075_compact_tracker_pfbeam_feature_repro_guard
```

Kaggle feature generation は `prepare_kaggle_notebooks.py --notebook pfbeam_features` で package 化できるようにした。

```bash
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp075_compact_tracker_pfbeam_feature_repro_guard --notebook pfbeam_features --kernel-id kentookumura/exp075-compact-pfbeam-features-train --title "exp075 compact pfbeam features train" --run-on-push --strict
```

標準 train / inference の想定:

```bash
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp075_compact_tracker_pfbeam_feature_repro_guard --notebook train --kernel-id kentookumura/exp075-compact-pfbeam-lgbm-train --title "exp075 compact pfbeam lgbm train" --run-on-push --strict
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp075_compact_tracker_pfbeam_feature_repro_guard --notebook inference --kernel-id kentookumura/exp075-compact-pfbeam-lgbm-infer --title "exp075 compact pfbeam lgbm infer" --run-on-push --strict
```

## 結果

- 静的検証:
  - `py_compile`: PASS
  - `ruff check`: PASS
  - PF/Beam feature notebook JSON validation: PASS
  - train notebook JSON validation: PASS
  - inference notebook JSON validation: PASS
  - `validate_experiment.py`: PASS
- Kaggle package:
  - `pfbeam_features`: PASS at `experiments/exp075_compact_tracker_pfbeam_feature_repro_guard/kaggle/pfbeam_features`
  - `train`: PASS at `experiments/exp075_compact_tracker_pfbeam_feature_repro_guard/kaggle/train`
  - `inference`: PASS at `experiments/exp075_compact_tracker_pfbeam_feature_repro_guard/kaggle/inference`
- Kaggle feature generation:
  - v1: pushed with `enable_gpu: true`; superseded before using output.
  - v2: pushed with `enable_gpu: false`; failed after feature generation with `ValueError: too many values to unpack (expected 2)`.
    - Cause: `generate_exp063_tracker_train_frame()` called `load_exp063_tracker_frame()` and unpacked 2 values, but the loader returns `(frame, feature_columns, metadata)`.
    - Evidence: logs show base features `3,783,989` rows, Pixiux features `3,783,989` rows, then failure at `tracker_frame, read_meta = load_exp063_tracker_frame(tracker_path)`.
  - v3: fixed unpack to `tracker_frame, _, read_meta = load_exp063_tracker_frame(tracker_path)` and pushed with `enable_gpu: false`.
  - kernel id: `kentookumura/exp075-compact-pfbeam-features-train`
  - notebook file: `exp075_compact_tracker_pfbeam_feature_repro_guard_pfbeam_features.ipynb`
  - v3 pull existence check: PASS at `/tmp/kaggle-pull/exp075-compact-pfbeam-features-train-v3`
  - v3 metadata: `enable_gpu=false`, `machine_shape=None`
  - v3 status: complete
  - v3 output: `/tmp/kaggle-output/exp075_compact_tracker_pfbeam_feature_repro_guard/pfbeam_features_v3`
  - rows / wells / features: 3,783,989 / 773 / 65
  - raw gzip SHA: `280e94c8a7256e455ab5e3595096e56e4baf2c8e69083fc740aba48c818caa0a`
  - decompressed CSV content SHA: `d8bf4a133599f0c822335d66783cbffed74a244fd442585cb9b83d2d5b481e7c`
  - tracker CSV size: 845,817,669 bytes
  - feature generation elapsed: 9,793.583 sec
  - total notebook elapsed: 10,450.066 sec
  - summary SHA: `d37624cb312646d2c4c659798e3a9b7496e8a02d077a9dc01f40c80ce5d112de`
  - log SHA: `77bb4b67a5474ad67ff0ef8534d93ea6785a133fcd401958e1a446bd8fcc5f3d`
  - local evidence:
    - `artifacts/compact_tracker_pfbeam_repro_guard_feature_generation_v3_summary.json`
    - `artifacts/exp075-compact-pfbeam-features-train-v3.log`
    - `artifacts/exp075-compact-pfbeam-features-train-v3-kernel-metadata.json`
- Kaggle train:
  - v1: completed.
  - kernel id: `kentookumura/exp075-compact-pfbeam-lgbm-train`
  - URL: `https://www.kaggle.com/code/kentookumura/exp075-compact-pfbeam-lgbm-train`
  - pull existence check: PASS at `/tmp/kaggle-pull/exp075-compact-pfbeam-lgbm-train-v1-complete`
  - output: `/tmp/kaggle-output/exp075_compact_tracker_pfbeam_feature_repro_guard/train_v1`
  - metadata: `enable_gpu=true`, `machine_shape=Gpu`
  - kernel source: `kentookumura/exp075-compact-pfbeam-features-train`
  - feature source: `/kaggle/input/notebooks/kentookumura/exp075-compact-pfbeam-features-train/artifacts/ravaghi_vs_pixiux_public_replay_tracker_features_train.csv.gz`
  - feature source raw gzip SHA: `280e94c8a7256e455ab5e3595096e56e4baf2c8e69083fc740aba48c818caa0a`
  - feature source decompressed CSV content SHA: `d8bf4a133599f0c822335d66783cbffed74a244fd442585cb9b83d2d5b481e7c`
  - rows / wells / features: 3,783,989 / 773 / 65
  - active mode: `gpu_repro_guard_dp_threads8`
  - pooled CV RMSE:
    - `lgb0`: 9.76191961499488, prediction SHA `4d137f00186ada0607afb663e060f6ada3d9e56fbf4a839d91644e19338a60b9`
    - `lgb1`: 9.651408152898307, prediction SHA `ade34b512629afccd9e92eeb012c99cadc20e5f89dc810f222bcbbf021f8f607`
    - `lgb2`: 9.65450091719246, prediction SHA `5dbe3a7e78e9e2364edded63c5c90e93ca701d41bd4289e96171a41bef9fbac7`
    - `lgb_mean`: 9.624618332949836, prediction SHA `4621598e334539194aaa60a448f32eaa446061c325ef433c11e11ffdb67cc846`
  - elapsed: 7,324.503 sec
  - train summary SHA: `7ff6e0e928d3c769852e062ccc2e0c3fd39d1f9bac92d43aa1d3f76313786bab`
  - metrics SHA: `f10749ce9549402b39e560813f273286833154706d74495524372ef08bebd6e7`
  - feature importance mean SHA: `81d77182823765f8860f571f9cb56cc45d243fdbbfd64f7e6f02d3ee8d01dc6e`
  - feature importance plot SHA: `f4798afebccdd470679cc16b8bf7bd9a14423c7be6f805b65e44027f28074740`
  - model manifest SHA: `39553e4342a1cf830f92ecf40eca7c218205359dce4008818965e908da9f2179`
  - predictions CSV.gz SHA: `465a91829894119b88ddd730338b45482cbcf8dab8878a8d4d6f70ff36428de6`
  - train log SHA: `081a428f5c4498c2085787e8a7d33344c79b64dc16092759cca340d8510a587d`
  - top feature importance mean:
    - `beam_vs_spatial`: 6210.533333
    - `pf_vs_dense`: 5845.666667
    - `pf_vs_spatial`: 5428.200000
    - `beam_vcons_d`: 5352.266667
    - `last_known_tvt`: 5181.266667
  - local evidence:
    - `artifacts/compact_tracker_pfbeam_repro_guard_train_v1_summary.json`
    - `artifacts/compact_tracker_pfbeam_repro_guard_train_v1_metrics.csv`
    - `artifacts/compact_tracker_pfbeam_repro_guard_train_v1_feature_importance_mean.csv`
    - `artifacts/compact_tracker_pfbeam_repro_guard_train_v1_feature_importance_mean_top.png`
    - `artifacts/compact_tracker_pfbeam_repro_guard_train_v1_lgb_manifest.json`
    - `artifacts/exp075-compact-pfbeam-lgbm-train-v1.log`
    - `artifacts/exp075-compact-pfbeam-lgbm-train-v1-kernel-metadata.json`
- Kaggle inference:
  - v1: completed.
  - kernel id: `kentookumura/exp075-compact-pfbeam-lgbm-infer`
  - URL: `https://www.kaggle.com/code/kentookumura/exp075-compact-pfbeam-lgbm-infer`
  - pull existence check: PASS at `/tmp/kaggle-pull/exp075-compact-pfbeam-lgbm-infer-v1-complete`
  - output: `/tmp/kaggle-output/exp075_compact_tracker_pfbeam_feature_repro_guard/inference_v1`
  - metadata: `id_no=123521629`, `enable_gpu=true`, `machine_shape=Gpu`, `enable_internet=false`
  - kernel source: `kentookumura/exp075-compact-pfbeam-lgbm-train`
  - initial `kaggle kernels logs` response was empty, treated as push-time API lag.
  - model manifest: `/kaggle/input/notebooks/kentookumura/exp075-compact-pfbeam-lgbm-train/artifacts/compact_tracker_pfbeam_repro_guard_lgb_models/manifest.json`
  - raw-test generated feature rows / wells / columns: 14,151 / 3 / 67
  - raw-test feature raw gzip SHA: `1ad141560fe42b2021b46fef534464c1ab6cd8d53513d79c9452d5fc0a31b723`
  - raw-test feature decompressed CSV content SHA: `fa82323cff7d24712f109348313a47aaf965c8c8acaa8275f7224d33a35412e8`
  - feature generation elapsed: 90.529 sec
  - selected mode/model/count: `gpu_repro_guard_dp_threads8` / `lgb_mean` / 15
  - submission rows / predicted rows / fallback rows: 14,151 / 14,151 / 0
  - prediction range / mean / std: 11599.796875 - 12240.2919921875 / 11905.121053241048 / 277.8403758000316
  - prediction SHA: `03e5cf79fcf9e6f03e0725ea4acaa35b08b0748a1886f8ac10931b47f54cf07e`
  - submission SHA: `c962cbd1602511c973a7d92b6973c5db790ad7eab310649e025dff411a7c991e`
  - summary SHA: `384b425f44c76752dafcb3b6d09fb9cae6a7f82bd32ab10b4ec75441680e243d`
  - metrics SHA: `e218a0bbc4ced418cc2b49880fb42ce684ff75b97270d8abf21ac27056ca8443`
  - log SHA: `1de2fd59c41164f9c9c3fed9eb61be056fdb21cf04810ec03eb48cff79cb982c`
  - metadata SHA: `e64891158b60162b9ad92222122df0fdeb10d6fe39a5e8ca540dc032b0074333`
  - submit-check: `uv run python scripts/validate_submission.py --submission /tmp/kaggle-output/exp075_compact_tracker_pfbeam_feature_repro_guard/inference_v1/submission.csv` PASS
  - note: `task submit-check ...` was not available in this runtime because `task` command was not installed; the same validation script was run directly.
  - local evidence:
    - `artifacts/compact_tracker_pfbeam_repro_guard_inference_v1_summary.json`
    - `artifacts/compact_tracker_pfbeam_repro_guard_inference_v1_metrics.csv`
    - `artifacts/exp075-compact-pfbeam-lgbm-infer-v1.log`
    - `artifacts/exp075-compact-pfbeam-lgbm-infer-v1-kernel-metadata.json`
- Reproducibility correction:
  - User submitted inference v1 twice and observed different Public LB scores.
  - Kaggle submissions confirm `ref=53790771` Public LB 8.535 and `ref=53790878` Public LB 8.447 on 2026-06-18.
  - Root cause: exp075 inherited unseeded numba `np.random` paths in `run_pf_ancc()` / `run_pf_z()` and did not pass stable split/well `seed_base` to likelihood-PF.
  - The same issue applies to `exp075-compact-pfbeam-features-train` v3, so feature generation v3 / train v1 / inference v1 are historical only, not reproducibility evidence.
  - Patch applied to `public_notebook_replay_audit.py`:
    - `stable_seed()` added.
    - `run_pf_ancc()` uses `stable_seed("pf_ancc", wid)`.
    - `run_pf_z()` uses `stable_seed("pf_z", wid)`.
    - `lik_pf()` uses `stable_seed("likpf", split, wid)` as `seed_base`.
- Kaggle feature generation stable-seed rerun:
  - v4: completed.
  - kernel id: `kentookumura/exp075-compact-pfbeam-features-train`
  - URL: `https://www.kaggle.com/code/kentookumura/exp075-compact-pfbeam-features-train`
  - pull existence check: PASS at `/tmp/kaggle-pull/exp075-compact-pfbeam-features-train-v4-complete`
  - output: `/tmp/kaggle-output/exp075_compact_tracker_pfbeam_feature_repro_guard/pfbeam_features_v4`
  - metadata: `id_no=123237004`, `enable_gpu=false`, `machine_shape=None`, `enable_internet=false`
  - packaged source check: `stable_seed`, `_pf_ancc_seeded`, `_pf_z_seeded`, `seed=stable_seed(...)`, and `seed_base=stable_seed(...)` are present.
  - initial `kaggle kernels logs` response was empty, treated as push-time API lag.
  - User requested execution again; v4 had already been pushed with `run_on_push=true`, so no duplicate v5 push was made.
  - Recheck commands:
    - `kaggle kernels logs kentookumura/exp075-compact-pfbeam-features-train`: empty response.
    - `kaggle kernels output kentookumura/exp075-compact-pfbeam-features-train -p /tmp/kaggle-output/exp075_compact_tracker_pfbeam_feature_repro_guard/pfbeam_features_v4`: empty response.
    - `kaggle kernels pull kentookumura/exp075-compact-pfbeam-features-train -p /tmp/kaggle-pull/exp075-compact-pfbeam-features-train-v4-check -m`: PASS.
    - `timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp075-compact-pfbeam-features-train`: timed out with no log output, treated as Kaggle CLI/API log lag rather than a reason to change slug or push a duplicate version.
  - completed output rows / wells / features: 3,783,989 / 773 / 65
  - tracker CSV size: 845,549,177 bytes
  - raw gzip SHA: `7a091800aeb068bfa5fceba4d6331a61945afcef2197e7fb973b30287dca9a91`
  - decompressed CSV content SHA: `047b80b32e64b595f2a75e7593ecb513e1f27d43de87614dd2de82dae416d5b4`
  - feature generation elapsed: 18,209.562 sec
  - total notebook elapsed: 19,185.915 sec
  - summary SHA: `61c743ae86695ed13b5c06a44a365c7a17864e055a84abbdeb9429953e3207fb`
  - log SHA: `b6e30ba9cc08ab97992b350eb152f0d99a83f91e03b757ac2586110d4f95aa20`
  - metadata SHA: `79b7729ff66dfd5638d109530dce157d387bed4b34801624858f7ad051491b61`
  - local evidence:
    - `artifacts/compact_tracker_pfbeam_repro_guard_feature_generation_v4_summary.json`
    - `artifacts/exp075-compact-pfbeam-features-train-v4.log`
    - `artifacts/exp075-compact-pfbeam-features-train-v4-kernel-metadata.json`

## 次のアクション

1. exp075 compact surface を後続実験で使う場合は、PF/Beam feature generation v4、train v2、inference v3 GPU の SHA を基準にする。

## 2026-06-18 17:36 JST update

- User requested train execution and matplotlib feature importance that helps explain why exp075 LB is better than exp076.
- Implemented fold-mean feature importance in `compact_tracker_pfbeam_repro_guard.py`:
  - keep raw importance and existing direct mean outputs for compatibility.
  - add `compact_tracker_pfbeam_repro_guard_feature_importance_fold_mean_by_fold.csv`.
  - add `compact_tracker_pfbeam_repro_guard_feature_importance_fold_mean.csv`.
  - add matplotlib plot `compact_tracker_pfbeam_repro_guard_feature_importance_fold_mean_top.png`.
  - aggregation: average feature_importances_ across LightGBM configs within each fold, then compute mean/std across folds.
- Updated train notebook display cell to show the fold-mean table and matplotlib PNG.
- Updated expected train artifacts in `config.yaml`.
- Validation:
  - `uv run python -m json.tool experiments/exp075_compact_tracker_pfbeam_feature_repro_guard/exp075_compact_tracker_pfbeam_feature_repro_guard_train.ipynb`: PASS.
  - `uv run python -m py_compile experiments/exp075_compact_tracker_pfbeam_feature_repro_guard/compact_tracker_pfbeam_repro_guard.py experiments/exp075_compact_tracker_pfbeam_feature_repro_guard/public_notebook_replay_audit.py experiments/exp075_compact_tracker_pfbeam_feature_repro_guard/settings.py`: PASS.
  - `uv run ruff check experiments/exp075_compact_tracker_pfbeam_feature_repro_guard/compact_tracker_pfbeam_repro_guard.py experiments/exp075_compact_tracker_pfbeam_feature_repro_guard/public_notebook_replay_audit.py experiments/exp075_compact_tracker_pfbeam_feature_repro_guard/settings.py`: PASS.
  - `uv run python scripts/validate_experiment.py --experiment exp075_compact_tracker_pfbeam_feature_repro_guard`: PASS.
- Kaggle train v2:
  - prepared with `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp075_compact_tracker_pfbeam_feature_repro_guard --notebook train --kernel-id kentookumura/exp075-compact-pfbeam-lgbm-train --title "exp075 compact pfbeam lgbm train" --run-on-push --strict`.
  - metadata: `enable_gpu=true`, `enable_internet=false`, kernel source `kentookumura/exp075-compact-pfbeam-features-train`.
  - packaged source contains stable seed patch and fold-mean feature importance outputs.
  - pushed with `kaggle kernels push -p experiments/exp075_compact_tracker_pfbeam_feature_repro_guard/kaggle/train`.
  - Kaggle response: `Kernel version 2 successfully pushed`.
  - URL: `https://www.kaggle.com/code/kentookumura/exp075-compact-pfbeam-lgbm-train`.
  - pull existence check: PASS at `/tmp/kaggle-pull/exp075-compact-pfbeam-lgbm-train-v2`.
  - initial `kaggle kernels logs kentookumura/exp075-compact-pfbeam-lgbm-train`: empty response immediately after push, treated as Kaggle CLI/API log lag.
  - current status: train v2 running from stable-seed PF/Beam feature generation v4.

## 2026-06-18 train v2 completed

- User reported train completion.
- Retrieval commands:
  - `kaggle kernels logs kentookumura/exp075-compact-pfbeam-lgbm-train`
  - `kaggle kernels output kentookumura/exp075-compact-pfbeam-lgbm-train -p /tmp/kaggle-output/exp075_compact_tracker_pfbeam_feature_repro_guard/train_v2`
  - `kaggle kernels pull kentookumura/exp075-compact-pfbeam-lgbm-train -p /tmp/kaggle-pull/exp075-compact-pfbeam-lgbm-train-v2-complete -m`
- Kaggle train v2:
  - status: completed.
  - kernel id: `kentookumura/exp075-compact-pfbeam-lgbm-train`
  - version: v2.
  - id_no: `123429196`.
  - URL: `https://www.kaggle.com/code/kentookumura/exp075-compact-pfbeam-lgbm-train`
  - output: `/tmp/kaggle-output/exp075_compact_tracker_pfbeam_feature_repro_guard/train_v2`
  - metadata: `enable_gpu=true`, `machine_shape=Gpu`, `enable_internet=false`
  - docker image: `gcr.io/kaggle-private-byod/python@sha256:57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c`
  - kernel source: `kentookumura/exp075-compact-pfbeam-features-train`
  - feature source raw gzip SHA: `7a091800aeb068bfa5fceba4d6331a61945afcef2197e7fb973b30287dca9a91`
  - feature source decompressed CSV content SHA: `047b80b32e64b595f2a75e7593ecb513e1f27d43de87614dd2de82dae416d5b4`
  - rows / wells / features: 3,783,989 / 773 / 65
  - elapsed seconds: 9,484.825
- CV RMSE:
  - `lgb0`: 9.841188080569813
  - `lgb1`: 9.727603469600522
  - `lgb2`: 9.719506950985725
  - `lgb_mean`: 9.699548082062895
- Prediction SHA:
  - `lgb0`: `00e629a072119282a708a74e14a764c05f0f939c11711822f82e92b7a73c9607`
  - `lgb1`: `db120d6e34290dd4d04b1a651ff66156905379222186b349d89c2d8100b90e77`
  - `lgb2`: `a875efddd87a172c743384ab23e8d6abeed9c787a228060fac1557203646da9d`
  - `lgb_mean`: `0afe646b8f52adbcea775a401c6d5af77e77df166bd4186d039ac438c9c46320`
- Train artifact SHA:
  - summary: `07a154e9949cb9fb63e6822f944f7970b11691e7db0f69e1a4f7dfb10b7d6510`
  - metrics: `e4d518ec27a8e94b13ddef5e5d40a853f5fc423c090efa9512280629be438b86`
  - model manifest: `ce4cf6897596c043a7a3a286a67607155092295f06715fdaf57539bba8fc1247`
  - predictions gzip: `a41f27848c2937950492c6c391da3d12f6d537ad85afc0aaf2d68fa613f09c0a`
  - log: `0f0468761d4229c7f14797345bb389cf563dda680b8ecca7a7bb60f524816584`
  - metadata: `fd24809c48bb7bc7c79b7729454b7eeb386b7eed57ffd1237d6fc053fe33f125`
- Fold-mean feature importance artifacts:
  - CSV: `compact_tracker_pfbeam_repro_guard_feature_importance_fold_mean.csv`, SHA `6ce887c8d9a5669fd8eaa2c590ae8d0b9f8e2cf9b9c4c92db503d6238081061a`
  - by-fold CSV: `compact_tracker_pfbeam_repro_guard_feature_importance_fold_mean_by_fold.csv`, SHA `ccd60df82b116f4641d2c0ba6f890268045dbf9f57c26be2d78c60c9d7c6a6b8`
  - matplotlib PNG: `compact_tracker_pfbeam_repro_guard_feature_importance_fold_mean_top.png`, SHA `78c6f0e36e0ee2970908d42a00f9058878b5fb4192bd9cf8bbd0988c608de7f6`
- Top fold-mean feature importance:
  - `beam_vs_spatial`: 8602.4
  - `pf_vs_dense`: 7885.8
  - `pf_vs_spatial`: 7475.6
  - `beam_vcons_d`: 6899.466666666667
  - `pf_z_delta`: 6673.133333333334
  - `last_known_tvt`: 6128.533333333333
  - `beam_stiff_d`: 5872.8
  - `pf_vs_z`: 5689.333333333334
  - `pf_ancc_delta`: 5514.666666666666
  - `beam_vloose_d`: 5460.733333333334
- Local evidence copied:
  - `artifacts/compact_tracker_pfbeam_repro_guard_train_v2_summary.json`
  - `artifacts/compact_tracker_pfbeam_repro_guard_train_v2_metrics.csv`
  - `artifacts/compact_tracker_pfbeam_repro_guard_train_v2_feature_importance_fold_mean.csv`
  - `artifacts/compact_tracker_pfbeam_repro_guard_train_v2_feature_importance_fold_mean_by_fold.csv`
  - `artifacts/compact_tracker_pfbeam_repro_guard_train_v2_feature_importance_fold_mean_top.png`
  - `artifacts/compact_tracker_pfbeam_repro_guard_train_v2_lgb_manifest.json`
  - `artifacts/exp075-compact-pfbeam-lgbm-train-v2.log`
  - `artifacts/exp075-compact-pfbeam-lgbm-train-v2-kernel-metadata.json`
- Interpretation:
  - Stable-seed v4 features make the valid train CV `lgb_mean=9.699548082062895`.
  - This is worse than pre-patch train v1 CV `9.624618332949836`; v1 remains invalid as reproducibility evidence.
  - Next step is inference rerun from train v2, then duplicate inference/submission SHA comparison.

## 2026-06-18 21:03 JST inference v2 started

- User requested inference execution from train v2.
- Validation before push:
  - `uv run python -m json.tool experiments/exp075_compact_tracker_pfbeam_feature_repro_guard/exp075_compact_tracker_pfbeam_feature_repro_guard_inference.ipynb`: PASS.
  - `uv run python -m py_compile experiments/exp075_compact_tracker_pfbeam_feature_repro_guard/compact_tracker_pfbeam_repro_guard.py experiments/exp075_compact_tracker_pfbeam_feature_repro_guard/public_notebook_replay_audit.py experiments/exp075_compact_tracker_pfbeam_feature_repro_guard/settings.py`: PASS.
  - `uv run python scripts/validate_experiment.py --experiment exp075_compact_tracker_pfbeam_feature_repro_guard`: PASS.
  - `uv run ruff check experiments/exp075_compact_tracker_pfbeam_feature_repro_guard/compact_tracker_pfbeam_repro_guard.py experiments/exp075_compact_tracker_pfbeam_feature_repro_guard/public_notebook_replay_audit.py experiments/exp075_compact_tracker_pfbeam_feature_repro_guard/settings.py`: PASS.
- Prepared with:
  - `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp075_compact_tracker_pfbeam_feature_repro_guard --notebook inference --kernel-id kentookumura/exp075-compact-pfbeam-lgbm-infer --title "exp075 compact pfbeam lgbm infer" --run-on-push --strict`
- First push attempt with GPU metadata failed:
  - command: `kaggle kernels push -p experiments/exp075_compact_tracker_pfbeam_feature_repro_guard/kaggle/inference`
  - Kaggle response: `Kernel push error: Maximum weekly GPU quota of 30.00 hours reached.`
- CPU fallback:
  - edited generated `kaggle/inference/kernel-metadata.json` only, changing `enable_gpu` from `true` to `false`.
  - reason: inference only regenerates test PF/Beam features and applies saved LightGBM boosters; GPU is not required for prediction.
- Kaggle inference v2:
  - push command: `kaggle kernels push -p experiments/exp075_compact_tracker_pfbeam_feature_repro_guard/kaggle/inference`
  - Kaggle response: `Kernel version 2 successfully pushed`.
  - kernel id: `kentookumura/exp075-compact-pfbeam-lgbm-infer`
  - URL: `https://www.kaggle.com/code/kentookumura/exp075-compact-pfbeam-lgbm-infer`
  - metadata: `enable_gpu=false`, `machine_shape=None`, `enable_internet=false`
  - docker image: `gcr.io/kaggle-images/python@sha256:e5452ce6268c2e8345cfe5141f31ca7ff47032aca46a7ea532bbb87481281d0c`
  - kernel source: `kentookumura/exp075-compact-pfbeam-lgbm-train`
  - train source intended: latest train kernel output, v2, model manifest SHA `ce4cf6897596c043a7a3a286a67607155092295f06715fdaf57539bba8fc1247`.
  - pull existence check: PASS at `/tmp/kaggle-pull/exp075-compact-pfbeam-lgbm-infer-v2`.
  - initial `kaggle kernels logs kentookumura/exp075-compact-pfbeam-lgbm-infer`: empty response immediately after push, treated as Kaggle CLI/API log lag.
  - current status: inference v2 running on CPU from train v2.

## 2026-06-18 21:17 JST GPU inference v3 started

- `timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp075-compact-pfbeam-lgbm-infer` showed CPU inference v2 had completed.
- CPU inference v2 summary:
  - output: `/tmp/kaggle-output/exp075_compact_tracker_pfbeam_feature_repro_guard/inference_v2`
  - metadata: `enable_gpu=false`, `machine_shape=None`
  - model source: train v2, manifest SHA `ce4cf6897596c043a7a3a286a67607155092295f06715fdaf57539bba8fc1247`
  - raw-test generated feature rows / wells / columns: 14,151 / 3 / 67
  - raw-test feature raw gzip SHA: `21f15e8a0424a6f119c1a69979e3eb73b796e56c305a2892424a54e0d3b0bbcc`
  - raw-test feature decompressed CSV content SHA: `38512547d23528e713134493311dae26707d30c9b302958cb8c6ff3ce02bb0a2`
  - feature generation elapsed: 107.979 sec
  - prediction SHA: `137baa962a02ca6b2d63c6e1085250ccb3b2ec158e098278bafc04799291bf51`
  - submission SHA: `79da931b3dd651fd7cc983d0b90de4d298995b8d68d1665a8a48edf621725284`
  - summary SHA: `42de08e0b1988d887514e8d6277babd151536b14e0c095d055adc8e33209a208`
  - metrics SHA: `254369a9a1f72946a302bd4b4ab940288f4a3ddd1b92b4bbc6673692a5f63e8e`
  - log SHA: `4853a771af955b7f2bd5b56ada002df4f6624ba0f12ce6d92274d87577290405`
  - metadata SHA: `101d788c984766235c60ef45e82f050ebc40b4a294eef82edf9685470e62ed9c`
  - submit-check: `uv run python scripts/validate_submission.py --submission /tmp/kaggle-output/exp075_compact_tracker_pfbeam_feature_repro_guard/inference_v2/submission.csv` PASS.
  - local evidence:
    - `artifacts/compact_tracker_pfbeam_repro_guard_inference_v2_cpu_summary.json`
    - `artifacts/compact_tracker_pfbeam_repro_guard_inference_v2_cpu_metrics.csv`
    - `artifacts/exp075-compact-pfbeam-lgbm-infer-v2-cpu.log`
    - `artifacts/exp075-compact-pfbeam-lgbm-infer-v2-cpu-kernel-metadata.json`
- User increased GPU quota and requested GPU inference execution.
- Re-prepared inference package:
  - `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp075_compact_tracker_pfbeam_feature_repro_guard --notebook inference --kernel-id kentookumura/exp075-compact-pfbeam-lgbm-infer --title "exp075 compact pfbeam lgbm infer" --run-on-push --strict`
  - generated metadata confirmed `enable_gpu=true`.
- Kaggle inference v3:
  - push command: `kaggle kernels push -p experiments/exp075_compact_tracker_pfbeam_feature_repro_guard/kaggle/inference`
  - Kaggle response: `Kernel version 3 successfully pushed`.
  - kernel id: `kentookumura/exp075-compact-pfbeam-lgbm-infer`
  - URL: `https://www.kaggle.com/code/kentookumura/exp075-compact-pfbeam-lgbm-infer`
  - metadata: `enable_gpu=true`, `machine_shape=Gpu`, `enable_internet=false`
  - docker image: `gcr.io/kaggle-private-byod/python@sha256:57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c`
  - kernel source: `kentookumura/exp075-compact-pfbeam-lgbm-train`
  - pull existence check: PASS at `/tmp/kaggle-pull/exp075-compact-pfbeam-lgbm-infer-v3`.
  - initial `kaggle kernels logs kentookumura/exp075-compact-pfbeam-lgbm-infer`: empty response immediately after push, treated as Kaggle CLI/API log lag.
  - current status: GPU inference v3 running from train v2.

## 2026-06-18 GPU inference v3 completed

- `timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp075-compact-pfbeam-lgbm-infer` showed GPU inference v3 completed.
- Retrieval commands:
  - `kaggle kernels output kentookumura/exp075-compact-pfbeam-lgbm-infer -p /tmp/kaggle-output/exp075_compact_tracker_pfbeam_feature_repro_guard/inference_v3_gpu`
  - `kaggle kernels pull kentookumura/exp075-compact-pfbeam-lgbm-infer -p /tmp/kaggle-pull/exp075-compact-pfbeam-lgbm-infer-v3-complete -m`
- GPU inference v3:
  - status: completed.
  - output: `/tmp/kaggle-output/exp075_compact_tracker_pfbeam_feature_repro_guard/inference_v3_gpu`
  - metadata: `enable_gpu=true`, `machine_shape=Gpu`, `enable_internet=false`
  - model source: train v2, manifest SHA `ce4cf6897596c043a7a3a286a67607155092295f06715fdaf57539bba8fc1247`
  - raw-test generated feature rows / wells / columns: 14,151 / 3 / 67
  - raw-test feature raw gzip SHA: `c9532d83e761d489ec42ea82f19b7814e9c50065861517e8ca58d4eff9863baf`
  - raw-test feature decompressed CSV content SHA: `6a00cb045dc1bdd3e8627bd669b445a97790ff7234054c3968225be62d49401d`
  - feature generation elapsed: 93.524 sec
  - prediction range / mean / std: 11601.1953125 - 12239.947265625 / 11905.696790283151 / 277.01090836741366
  - prediction SHA: `137baa962a02ca6b2d63c6e1085250ccb3b2ec158e098278bafc04799291bf51`
  - submission SHA: `79da931b3dd651fd7cc983d0b90de4d298995b8d68d1665a8a48edf621725284`
  - summary SHA: `a3a21b0eb370415c032f504df8aa93ff2bb0528e3b520d828834ebf5b9a6609b`
  - metrics SHA: `254369a9a1f72946a302bd4b4ab940288f4a3ddd1b92b4bbc6673692a5f63e8e`
  - predictions gzip SHA: `43c14467c183e9bef34c6ac3d537b850c07f18bbf517ea932fb2a3fa64aa73ea`
  - log SHA: `df7fbfe79d8335005b68faa4e9fbff160ad9d583d6f463f02d4f54d91aad9fdd`
  - metadata SHA: `e64891158b60162b9ad92222122df0fdeb10d6fe39a5e8ca540dc032b0074333`
  - submit-check: `uv run python scripts/validate_submission.py --submission /tmp/kaggle-output/exp075_compact_tracker_pfbeam_feature_repro_guard/inference_v3_gpu/submission.csv` PASS.
- CPU v2 vs GPU v3:
  - prediction SHA matches: `137baa962a02ca6b2d63c6e1085250ccb3b2ec158e098278bafc04799291bf51`
  - submission SHA matches: `79da931b3dd651fd7cc983d0b90de4d298995b8d68d1665a8a48edf621725284`
  - regenerated test feature content SHA differs:
    - CPU v2: `38512547d23528e713134493311dae26707d30c9b302958cb8c6ff3ce02bb0a2`
    - GPU v3: `6a00cb045dc1bdd3e8627bd669b445a97790ff7234054c3968225be62d49401d`
  - Interpretation: final prediction/submission is stable across CPU/GPU inference here, but feature content SHA is not identical across runtime mode. Use same runtime mode for strict feature-content reproducibility comparisons.
- Local evidence copied:
  - `artifacts/compact_tracker_pfbeam_repro_guard_inference_v3_gpu_summary.json`
  - `artifacts/compact_tracker_pfbeam_repro_guard_inference_v3_gpu_metrics.csv`
  - `artifacts/exp075-compact-pfbeam-lgbm-infer-v3-gpu.log`
  - `artifacts/exp075-compact-pfbeam-lgbm-infer-v3-gpu-kernel-metadata.json`

## 2026-06-18 submission attribution

- User submitted exp075 GPU inference v3 twice and both completed with the same Public LB.
  - `ref=53807892`: Public LB `8.489`
  - `ref=53807896`: Public LB `8.489`
- exp075 submission path: `/tmp/kaggle-output/exp075_compact_tracker_pfbeam_feature_repro_guard/inference_v3_gpu/submission.csv`
- exp075 submission SHA: `79da931b3dd651fd7cc983d0b90de4d298995b8d68d1665a8a48edf621725284`
- exp075 prediction SHA: `137baa962a02ca6b2d63c6e1085250ccb3b2ec158e098278bafc04799291bf51`
- `ref=53809333` is the latest observed submission, but the user clarified that it is not related to this experiment. It is not attributed to exp075.
- Final exp075 interpretation:
  - pre-patch inference v1 refs `53790771` / `53790878` remain invalid as reproducibility evidence because they produced different Public LB scores.
  - stable-seed PF/Beam feature generation v4 + train v2 + GPU inference v3 is the valid compact PF/Beam chain.
  - CPU inference v2 and GPU inference v3 produced identical prediction/submission SHA, but their regenerated test feature content SHA differs by runtime mode. Same-runtime reruns are required for strict feature-content equality checks.
