# exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit セッションノート

## 現在の状態

- status: `completed`
- route: `ml_model`
- 旧 v1: exp056 artifact ベース audit は公開ノートブック再現として不十分だったため無効化。
- 現 v2: raw competition files から公開ノートブック由来の特徴生成を再生する strict replay 実装。
- 追加 v3: selected Pixiux LGBM replay candidate の inference notebook を実装。初回 Kaggle inference version 1 は train features を再生成する設計だったため手動停止。
- 追加 v4: train notebook で saved booster / reusable tracker features を保存し、inference notebook は saved booster + test-only feature generation を使う設計へ修正。

## 実装内容

- `public_notebook_replay_audit.py` を追加。
  - Pixiux public notebook の `build_features` / `build_likpf` / `add_likpf_features` / public LGBM configs を実験内に同梱。
  - Ravaghi 側は public Ravaghi-style base features を使用。
  - Pixiux 側は base features に public likelihood-PF delta features を追加。
  - CatBoost、Ridge stack、final blend、projection、pretrained booster、static visible override は除外。
- train notebook を raw replay 実行に差し替え。
- `config.yaml` を artifact audit から strict public replay audit に置換。
- inference notebook の no-op guard を外し、`pixiux_likpf_public_replay` `lgb_mean` で `submission.csv` を作る構成に変更。
- `run_public_replay_inference` を追加。
  - 修正前 v1: raw train/test files から features を作り、inference 内で LightGBM を再学習していたため手動停止。
  - 修正後 v4: train notebook が `pixiux_likpf_public_replay` の 3 configs x 5 folds の LightGBM booster を `ravaghi_vs_pixiux_public_replay_lgb_models/` に保存。
  - inference notebook は保存済み booster を読み、test-side replay features だけを生成して `last_known_tvt + pred_delta` を submission に保存。
  - 後続実験の train で `id` join して再利用できるよう、PF/Beam/likelihood-PF tracker feature frame を csv.gz で保存。
  - hidden-specific branch、guarded overlap override、static visible override、CatBoost、Ridge stack、final public notebook blend、projection postprocess は除外。
- `scripts/prepare_kaggle_notebooks.py` を修正し、experiment config の `runtime.kaggle.enable_gpu` / `enable_internet` を project default より優先。
- 旧 `ravaghi_vs_pixiux_feature_parity_audit.py`、`baseline.py`、`pseudo_tail_augmentation.py`、旧生成物を削除。

## 実行コマンド

```bash
uv run python -m py_compile experiments/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/public_notebook_replay_audit.py experiments/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/settings.py scripts/prepare_kaggle_notebooks.py
uv run ruff check experiments/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/public_notebook_replay_audit.py experiments/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/settings.py scripts/prepare_kaggle_notebooks.py
uv run python scripts/validate_experiment.py --experiment exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit --notebook train --kernel-id kentookumura/exp063-ravaghi-pixiux-parity-train --title "exp063 ravaghi pixiux strict replay train" --run-on-push --strict
kaggle kernels push -p experiments/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/kaggle/train
kaggle kernels pull kentookumura/exp063-ravaghi-pixiux-strict-replay-train -p /tmp/kaggle-pull/exp063-ravaghi-pixiux-strict-replay-train-v2 -m
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit --notebook train --kernel-id kentookumura/exp063-ravaghi-pixiux-strict-replay-train --title "exp063 ravaghi pixiux strict replay train" --run-on-push --strict
kaggle kernels push -p experiments/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/kaggle/train
kaggle kernels logs kentookumura/exp063-ravaghi-pixiux-strict-replay-train
kaggle kernels output kentookumura/exp063-ravaghi-pixiux-strict-replay-train -p /tmp/kaggle-output/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/train_v3
kaggle kernels status kentookumura/exp063-ravaghi-pixiux-strict-replay-train
uv run python -m py_compile experiments/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/public_notebook_replay_audit.py experiments/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/settings.py
uv run ruff check experiments/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/public_notebook_replay_audit.py experiments/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/settings.py
uv run python -m json.tool experiments/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/metrics.json
uv run python -m json.tool experiments/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit_inference.ipynb
uv run python scripts/validate_experiment.py --experiment exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit --notebook inference --kernel-id kentookumura/exp063-ravaghi-pixiux-strict-replay-infer --title "exp063 ravaghi pixiux strict replay infer" --run-on-push --strict
kaggle kernels push -p experiments/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/kaggle/inference
kaggle kernels pull kentookumura/exp063-ravaghi-pixiux-strict-replay-infer -p /tmp/kaggle-pull/exp063-ravaghi-pixiux-strict-replay-infer-v1 -m
kaggle kernels logs kentookumura/exp063-ravaghi-pixiux-strict-replay-infer
timeout 60 kaggle kernels logs -f --interval 5 kentookumura/exp063-ravaghi-pixiux-strict-replay-infer
kaggle kernels output kentookumura/exp063-ravaghi-pixiux-strict-replay-infer -p /tmp/kaggle-output/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/infer_v1_probe
uv run python -m py_compile experiments/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/public_notebook_replay_audit.py experiments/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/settings.py scripts/prepare_kaggle_notebooks.py
uv run ruff check experiments/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/public_notebook_replay_audit.py experiments/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/settings.py scripts/prepare_kaggle_notebooks.py
uv run python -m json.tool experiments/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/metrics.json
uv run python -m json.tool experiments/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit_train.ipynb
uv run python -m json.tool experiments/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit_inference.ipynb
uv run python scripts/validate_experiment.py --experiment exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit --notebook train --kernel-id kentookumura/exp063-ravaghi-pixiux-strict-replay-train --title "exp063 ravaghi pixiux strict replay train" --run-on-push --strict
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit --notebook inference --kernel-id kentookumura/exp063-ravaghi-pixiux-strict-replay-infer --title "exp063 ravaghi pixiux strict replay infer" --run-on-push --strict
kaggle kernels push -p experiments/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/kaggle/train
kaggle kernels pull kentookumura/exp063-ravaghi-pixiux-strict-replay-train -p /tmp/kaggle-pull/exp063-ravaghi-pixiux-strict-replay-train-v4 -m
kaggle kernels status kentookumura/exp063-ravaghi-pixiux-strict-replay-train
kaggle kernels logs kentookumura/exp063-ravaghi-pixiux-strict-replay-train
kaggle kernels output kentookumura/exp063-ravaghi-pixiux-strict-replay-train -p /tmp/kaggle-output/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/train_v4
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit --notebook inference --kernel-id kentookumura/exp063-ravaghi-pixiux-strict-replay-infer --title "exp063 ravaghi pixiux strict replay infer" --run-on-push --strict
kaggle kernels push -p experiments/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/kaggle/inference
kaggle kernels pull kentookumura/exp063-ravaghi-pixiux-strict-replay-infer -p /tmp/kaggle-pull/exp063-ravaghi-pixiux-strict-replay-infer-v2 -m
kaggle kernels logs kentookumura/exp063-ravaghi-pixiux-strict-replay-infer
kaggle kernels status kentookumura/exp063-ravaghi-pixiux-strict-replay-infer
kaggle kernels output kentookumura/exp063-ravaghi-pixiux-strict-replay-infer -p /tmp/kaggle-output/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/infer_v2_probe
```

## 検証結果

- `py_compile`: PASS
- `ruff check`: PASS
- `validate_experiment.py`: PASS
- Kaggle train package: generated at `experiments/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/kaggle/train`
- Kaggle metadata: `enable_gpu=true`, `enable_internet=false`, `run_on_push=true`
- First push used mismatched kernel id/title and Kaggle created/used slug `kentookumura/exp063-ravaghi-pixiux-strict-replay-train`; local package metadata was updated to that canonical id.
- Kaggle version 2 failed before feature generation because one notebook display line still called `get_nested(config, "model.variants", [])`.
- The notebook was fixed to use `cfg_get(...)`; validation and package generation passed again.
- Kaggle version 3 was pushed successfully: `https://www.kaggle.com/code/kentookumura/exp063-ravaghi-pixiux-strict-replay-train`
- Kaggle version 3 completed.
- Output was downloaded to `/tmp/kaggle-output/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/train_v3`.
- Generated artifacts were synced into `artifacts/`.
- Inference implementation checks:
  - `py_compile`: PASS
  - `ruff check`: PASS
  - `metrics.json` JSON validation: PASS
  - inference notebook JSON validation: PASS
  - `validate_experiment.py`: PASS
  - Kaggle inference package: generated at `experiments/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/kaggle/inference`
  - Kaggle inference metadata: `enable_gpu=true`, `enable_internet=false`, `run_on_push=true`
  - Kaggle inference version 1 was pushed successfully: `https://www.kaggle.com/code/kentookumura/exp063-ravaghi-pixiux-strict-replay-infer`
  - Pull existence check succeeded at `/tmp/kaggle-pull/exp063-ravaghi-pixiux-strict-replay-infer-v1`.
  - Initial normal logs were empty; 60 sec `logs -f` also returned no log before timeout.
  - Initial output probe at `/tmp/kaggle-output/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/infer_v1_probe` returned no files yet.
  - User manually stopped inference v1 because it regenerated train features and was too slow.
  - Code was corrected to saved booster + test-only inference. Existing train v3 does not contain saved boosters, so train must be rerun once before corrected inference can run.
  - Corrected implementation checks:
    - `py_compile`: PASS
    - `ruff check`: PASS
    - `metrics.json` JSON validation: PASS
    - train notebook JSON validation: PASS
    - inference notebook JSON validation: PASS
    - `validate_experiment.py`: PASS
    - corrected train package generated with no kernel source.
    - corrected inference package generated with kernel source `kentookumura/exp063-ravaghi-pixiux-strict-replay-train`.
  - Corrected train version 4 was pushed successfully: `https://www.kaggle.com/code/kentookumura/exp063-ravaghi-pixiux-strict-replay-train`
  - Pull existence check succeeded at `/tmp/kaggle-pull/exp063-ravaghi-pixiux-strict-replay-train-v4`.
  - Status after completion check: `KernelWorkerStatus.COMPLETE`.
  - Initial normal logs were empty.
  - Final logs and output were downloaded to `/tmp/kaggle-output/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/train_v4`.
  - Generated v4 artifacts were synced into `artifacts/`.
  - Corrected inference package was regenerated with kernel source `kentookumura/exp063-ravaghi-pixiux-strict-replay-train`.
  - Corrected inference version 2 was pushed successfully: `https://www.kaggle.com/code/kentookumura/exp063-ravaghi-pixiux-strict-replay-infer`
  - Pull existence check succeeded at `/tmp/kaggle-pull/exp063-ravaghi-pixiux-strict-replay-infer-v2`.
  - Initial normal logs were empty.
  - Initial status: `KernelWorkerStatus.RUNNING`.
  - Initial output probe returned no files yet.
  - Completion status: `KernelWorkerStatus.COMPLETE`.
  - Final logs and output were downloaded to `/tmp/kaggle-output/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/infer_v2`.
  - Inference artifacts were synced into `artifacts/`; `submission.csv` remains in `/tmp/kaggle-output/.../infer_v2/`.
  - `uv run python scripts/validate_submission.py --submission /tmp/kaggle-output/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/infer_v2/submission.csv`: PASS.
  - `.agents/skills/kaggle-submit-check/scripts/check_submission.py`: PASS, no FAIL/WARN.

## Kaggle version 4 結果

- kernel: `kentookumura/exp063-ravaghi-pixiux-strict-replay-train`
- status: `KernelWorkerStatus.COMPLETE`
- rows / wells: 3,783,989 / 773
- feature generation: 13,034.108 sec
- total runtime: 27,742.572 sec
- feature counts:
  - `ravaghi_public_lgbm_replay`: 195
  - `pixiux_likpf_public_replay`: 196
- selected: `pixiux_likpf_public_replay` `lgb2`
- selected OOF RMSE: 9.628965
- inference ensemble candidate: `pixiux_likpf_public_replay` `lgb_mean` 9.630105
- Ravaghi `lgb_mean` OOF RMSE: 10.560537
- delta Pixiux mean vs Ravaghi mean: -0.930432
- best Ravaghi single LGBM: `lgb2` 10.538333
- delta Pixiux best vs best Ravaghi single: -0.909367
- mean feature importance plot: `artifacts/ravaghi_vs_pixiux_public_replay_feature_importance_mean_top.png`
- Pixiux top mean feature importance: `likpf_mean_d`
- saved LightGBM boosters: `artifacts/ravaghi_vs_pixiux_public_replay_lgb_models/` (15 model files + manifest)
- reusable tracker train features: `artifacts/ravaghi_vs_pixiux_public_replay_tracker_features_train.csv.gz` (3,783,989 rows x 68 columns)

## Kaggle inference version 2 結果

- kernel: `kentookumura/exp063-ravaghi-pixiux-strict-replay-infer`
- status: `KernelWorkerStatus.COMPLETE`
- mode: `strict_public_replay_saved_model_inference_no_override`
- model: `pixiux_likpf_public_replay` `lgb_mean`
- train source: `kentookumura/exp063-ravaghi-pixiux-strict-replay-train` version 4 output
- saved LightGBM boosters loaded: 15
- test wells / rows: 3 / 14,151
- feature generation: 98.101 sec
- total runtime: 127.648 sec
- submission rows / predicted rows / fallback rows: 14,151 / 14,151 / 0
- prediction range: 11,593.674805 - 12,240.098633
- prediction mean / std: 11,905.529255 / 279.332552
- submission path: `/tmp/kaggle-output/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/infer_v2/submission.csv`
- submission sha256: `36486e2e5a049ae02b51daa2a06e317bc6c7b841d5fe25841427b792a24f2499`
- submit-check: PASS
- reusable tracker test features: `artifacts/ravaghi_vs_pixiux_public_replay_tracker_features_test.csv.gz` (14,151 rows x 67 columns)
- code submission ref: `53632725`
- submission status: COMPLETE
- Public LB: 8.811
- submission history row: `SUBMISSIONS.md` v028

## 2026-06-13 再現性調査

- 背景: exp063 は Kaggle metadata で `enable_gpu=true`、`audit.use_gpu=auto` のため、LightGBM は GPU device で学習された。
- 通常使っている PyTorch/CuDNN 向け seed 設定だけでは、この実験の主要経路は制御できない。
  - exp063 は PyTorch を使わず、学習は LightGBM GPU、特徴生成は NumPy/Numba/joblib が中心。
  - LightGBM config は seed / random_state を持つが、GPU 実行では `device_type="gpu"`、`gpu_use_dp=false`、`n_jobs=-1` で、deterministic / force_col_wise / num_threads 固定はない。
  - likelihood-PF は Numba JIT 内で `np.random.seed(seed_base + s)` を使うが、`build_likpf` は `joblib.Parallel(..., prefer="threads")` で well 並列実行するため、global RNG state 競合の余地がある。
- 実測比較:
  - `train_v3` と `train_v4` は feature schema SHA が同一: `a041dbf4f7c5e6d64ee7ee572fea0be104ccb921fe6842629281d97385d9c822`。
  - code/config は同一ではない。v4 は saved booster と inference port 用 helper を追加しているが、train feature schema と public LightGBM configs は同じ。
  - `pixiux_likpf_public_replay` `lgb_mean` pooled OOF RMSE は `train_v3=9.599138098927096`、`train_v4=9.630105123038494`、差分 `+0.030967024111397734`。
  - `pixiux_likpf_public_replay` `lgb2` pooled OOF RMSE は `train_v3=9.61511336497242`、`train_v4=9.628965463901501`、差分 `+0.01385209892908179`。
  - OOF prediction content SHA も不一致: `train_v3=98f52bf325a78989e5645b353fe6f472f904d4c73d4edc80b6303a2012e078a9`、`train_v4=b2ae003d84c4a76683a4974b7c9fc93896f281b306520b1b1ffae20e2e7914ae`。
- 判定:
  - exp063 の Public LB 8.811 は saved booster 由来の `submission.csv` SHA `36486e2e5a049ae02b51daa2a06e317bc6c7b841d5fe25841427b792a24f2499` として提出物単位では固定できている。
  - 一方で、train-side CV は GPU rerun で bitwise 再現されておらず、v4 の CV 値は単発実行値として扱う。
  - ML route anchor の LB 記録は有効だが、exp063 の CV を後続実験の細かい差分比較基準に使う場合は再現性リスクあり。
- 推奨:
  - 次に exp063 系を学習再利用するなら、まず `use_gpu: cpu` または LightGBM に `deterministic=true`、`force_col_wise=true`、`num_threads` 固定、`gpu_use_dp=true` の候補を分けて rerun し、OOF SHA / metrics 差分を確認する。
  - PF feature を厳密に固定したい場合は、likelihood-PF の `joblib` threads を避け、well 単位で独立 RNG を使う実装に切り替える。

## 次のアクション

1. ML route では exp039 Public LB 11.740 から exp063 Public LB 8.811 へ anchor 更新。全体 / PF route では exp027 Public LB 8.781 を維持し、exp063 raw replay は後処理/gate 監査の入力としても扱う。
2. exp063 の train-side CV は再現性調査の結果、GPU rerun でぶれる単発値として扱う。後続の細かい CV 差分比較には、CPU deterministic rerun か saved booster/submission SHA 比較を優先する。
