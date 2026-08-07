# exp224_well_scaled_z_dz_features_on_exp218 セッションノート

## 目的

`well_scaled_z_dz_features_on_exp218` を実装する。exp218 の GRWR add-only ML anchor に、target-free な well 内 robust scale の `z` / `dz` / `dzdmd` / `slp_z` 特徴を追加する。

## 現在の状態

- 状態: Kaggle CPU split train 完了。不採用
- route: `ml_model`
- 親実験: `exp218_gr_wavelet_rotation_confidence_features_on_exp148`
- baseline: exp218 `lgb_mean` CV 8.475793752 / Public LB 7.843
- result: exp224 split aggregate `lgb_mean` CV 8.538687042。exp218 比 +0.062893290 悪化

## 実装内容

- `well_scaled_z_dz_features_on_exp218.py` を exp218 実装から派生。
- `build_well_scaled_z_dz_features()` を追加。
- 追加 feature group: `well_scaled_z_dz`
- active variant: `well_scaled_z_dz_addonly`
- raw `z` / `dz` / `dzdmd` / `slp_z` は削除せず、base feature として残す。
- well 内 median / MAD / IQR / p05-p95 range / rank / relative range を生成。
- `likpf_tvt = last_known_tvt + likpf_mean_d` の well 内 p05-p95 range で `z` / `dz` を割る feature を追加。これは scale feature のみで、candidate correction には使わない。
- interaction は `z` / `dz` の p05-p95 scaled 版と、well-scaled raw `gr`、`grwr_fft_rotation_ratio_x_log1p_md_since`、`grwr_candidate_tvt_std`、`grwr_candidate_tvt_range`、`grwr_known_prefix_fraction` に限定。
- train は CPU split 実行に変更し、`train_lgb0` / `train_lgb1` / `train_lgb2` が各 1 LightGBM config だけを学習する。
- inference は split train の aggregate manifest または明示的な split output handling ができるまで実行しない。
- train / inference の両方で同じ builder を呼び、manifest の feature group 一致チェックを入れる。

## 実行コストガード

- active variant: 1 (`well_scaled_z_dz_addonly`)
- runtime: CPU (`enable_gpu=false`)
- train split: `train_lgb0` / `train_lgb1` / `train_lgb2`
- LightGBM configs: 3 (`lgb0`, `lgb1`, `lgb2`)
- folds per split: 5
- boosters per split: 5
- total boosters: 15
- parent/control retraining: なし。exp218 の保存済み CV / Public LB を baseline として参照する。

## リスク

- likPF range scale は `likpf_mean_d` 由来なので過信しない。
- target-derived scaler、direct correction、replacement、blend、postprocess、hard selector、sample-weight 変更は禁止。
- 100-1000 bucket と worst-well regression を重点確認する。

## 次のコマンド

Kaggle CPU split train push の承認後に実行する。各 split は 1 config x 5 folds = 5 boosters。

```bash
kaggle kernels push -p experiments/exp224_well_scaled_z_dz_features_on_exp218/kaggle/train_lgb0
kaggle kernels push -p experiments/exp224_well_scaled_z_dz_features_on_exp218/kaggle/train_lgb1
kaggle kernels push -p experiments/exp224_well_scaled_z_dz_features_on_exp218/kaggle/train_lgb2
```

## 実装時検証

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp224_well_scaled_z_dz_features_on_exp218/exp224_well_scaled_z_dz_features_on_exp218_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp224_well_scaled_z_dz_features_on_exp218/exp224_well_scaled_z_dz_features_on_exp218_train_lgb0.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp224_well_scaled_z_dz_features_on_exp218/exp224_well_scaled_z_dz_features_on_exp218_train_lgb1.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp224_well_scaled_z_dz_features_on_exp218/exp224_well_scaled_z_dz_features_on_exp218_train_lgb2.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp224_well_scaled_z_dz_features_on_exp218/exp224_well_scaled_z_dz_features_on_exp218_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp224_well_scaled_z_dz_features_on_exp218/exp224_well_scaled_z_dz_features_on_exp218_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp224_well_scaled_z_dz_features_on_exp218/exp224_well_scaled_z_dz_features_on_exp218_train_lgb0.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp224_well_scaled_z_dz_features_on_exp218/exp224_well_scaled_z_dz_features_on_exp218_train_lgb1.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp224_well_scaled_z_dz_features_on_exp218/exp224_well_scaled_z_dz_features_on_exp218_train_lgb2.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp224_well_scaled_z_dz_features_on_exp218/exp224_well_scaled_z_dz_features_on_exp218_inference.py
.venv/bin/python -m py_compile experiments/exp224_well_scaled_z_dz_features_on_exp218/well_scaled_z_dz_features_on_exp218.py experiments/exp224_well_scaled_z_dz_features_on_exp218/exp224_well_scaled_z_dz_features_on_exp218_train.py experiments/exp224_well_scaled_z_dz_features_on_exp218/exp224_well_scaled_z_dz_features_on_exp218_train_lgb0.py experiments/exp224_well_scaled_z_dz_features_on_exp218/exp224_well_scaled_z_dz_features_on_exp218_train_lgb1.py experiments/exp224_well_scaled_z_dz_features_on_exp218/exp224_well_scaled_z_dz_features_on_exp218_train_lgb2.py experiments/exp224_well_scaled_z_dz_features_on_exp218/exp224_well_scaled_z_dz_features_on_exp218_inference.py experiments/exp224_well_scaled_z_dz_features_on_exp218/settings.py
.venv/bin/ruff check experiments/exp224_well_scaled_z_dz_features_on_exp218/well_scaled_z_dz_features_on_exp218.py experiments/exp224_well_scaled_z_dz_features_on_exp218/exp224_well_scaled_z_dz_features_on_exp218_train.py experiments/exp224_well_scaled_z_dz_features_on_exp218/exp224_well_scaled_z_dz_features_on_exp218_train_lgb0.py experiments/exp224_well_scaled_z_dz_features_on_exp218/exp224_well_scaled_z_dz_features_on_exp218_train_lgb1.py experiments/exp224_well_scaled_z_dz_features_on_exp218/exp224_well_scaled_z_dz_features_on_exp218_train_lgb2.py experiments/exp224_well_scaled_z_dz_features_on_exp218/exp224_well_scaled_z_dz_features_on_exp218_inference.py experiments/exp224_well_scaled_z_dz_features_on_exp218/settings.py --select F821,F401
uv run python scripts/validate_experiment.py --experiment exp224_well_scaled_z_dz_features_on_exp218
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp224_well_scaled_z_dz_features_on_exp218 --notebook train --kernel-id kentookumura/exp224-wsz-exp218-train --title 'exp224 wsz exp218 train index' --strict
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp224_well_scaled_z_dz_features_on_exp218 --notebook train_lgb0 --kernel-id kentookumura/exp224-wsz-exp218-lgb0 --title 'exp224 wsz exp218 lgb0' --run-on-push --strict
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp224_well_scaled_z_dz_features_on_exp218 --notebook train_lgb1 --kernel-id kentookumura/exp224-wsz-exp218-lgb1 --title 'exp224 wsz exp218 lgb1' --run-on-push --strict
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp224_well_scaled_z_dz_features_on_exp218 --notebook train_lgb2 --kernel-id kentookumura/exp224-wsz-exp218-lgb2 --title 'exp224 wsz exp218 lgb2' --run-on-push --strict
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp224_well_scaled_z_dz_features_on_exp218 --notebook inference --kernel-id kentookumura/exp224-wsz-exp218-infer --title 'exp224 wsz exp218 infer' --strict
.venv/bin/python -m py_compile experiments/exp224_well_scaled_z_dz_features_on_exp218/kaggle/train/well_scaled_z_dz_features_on_exp218.py experiments/exp224_well_scaled_z_dz_features_on_exp218/kaggle/train/exp224_well_scaled_z_dz_features_on_exp218_train.py experiments/exp224_well_scaled_z_dz_features_on_exp218/kaggle/train/settings.py
.venv/bin/python -m py_compile experiments/exp224_well_scaled_z_dz_features_on_exp218/kaggle/train_lgb0/well_scaled_z_dz_features_on_exp218.py experiments/exp224_well_scaled_z_dz_features_on_exp218/kaggle/train_lgb0/exp224_well_scaled_z_dz_features_on_exp218_train_lgb0.py experiments/exp224_well_scaled_z_dz_features_on_exp218/kaggle/train_lgb0/settings.py
.venv/bin/python -m py_compile experiments/exp224_well_scaled_z_dz_features_on_exp218/kaggle/train_lgb1/well_scaled_z_dz_features_on_exp218.py experiments/exp224_well_scaled_z_dz_features_on_exp218/kaggle/train_lgb1/exp224_well_scaled_z_dz_features_on_exp218_train_lgb1.py experiments/exp224_well_scaled_z_dz_features_on_exp218/kaggle/train_lgb1/settings.py
.venv/bin/python -m py_compile experiments/exp224_well_scaled_z_dz_features_on_exp218/kaggle/train_lgb2/well_scaled_z_dz_features_on_exp218.py experiments/exp224_well_scaled_z_dz_features_on_exp218/kaggle/train_lgb2/exp224_well_scaled_z_dz_features_on_exp218_train_lgb2.py experiments/exp224_well_scaled_z_dz_features_on_exp218/kaggle/train_lgb2/settings.py
.venv/bin/python -m py_compile experiments/exp224_well_scaled_z_dz_features_on_exp218/kaggle/inference/well_scaled_z_dz_features_on_exp218.py experiments/exp224_well_scaled_z_dz_features_on_exp218/kaggle/inference/exp224_well_scaled_z_dz_features_on_exp218_inference.py experiments/exp224_well_scaled_z_dz_features_on_exp218/kaggle/inference/settings.py
```

- Jupytext conversion / `--test`: pass
- `py_compile`: pass
- `ruff --select F821,F401`: pass
- `validate_experiment`: pass
- synthetic feature builder smoke: pass。10 rows / 2 wells で 51 features を生成し、row alignment と finite values を確認した。
- selected `lgb_config_indices` の actual fit smoke は local venv に `lightgbm` がないため未実行。Kaggle package 側の構文検証は pass。
- Kaggle train package: `experiments/exp224_well_scaled_z_dz_features_on_exp218/kaggle/train`
- Kaggle train_lgb0 package: `experiments/exp224_well_scaled_z_dz_features_on_exp218/kaggle/train_lgb0`
- Kaggle train_lgb1 package: `experiments/exp224_well_scaled_z_dz_features_on_exp218/kaggle/train_lgb1`
- Kaggle train_lgb2 package: `experiments/exp224_well_scaled_z_dz_features_on_exp218/kaggle/train_lgb2`
- Kaggle inference package: `experiments/exp224_well_scaled_z_dz_features_on_exp218/kaggle/inference`
- Metadata check: train_lgb0 / train_lgb1 / train_lgb2 は `enable_gpu=false`、`run_on_push=true`。

## 2026-07-09 Kaggle CPU split train push

実行前確認:

- `validate_experiment`: pass
- active variant: 1 (`well_scaled_z_dz_addonly`)
- active mode: 1 (`cpu_deterministic_threads8`)
- runtime: CPU (`enable_gpu=false`)
- split: `train_lgb0` / `train_lgb1` / `train_lgb2`
- boosters: 各 split 1 config x 5 folds = 5、合計 15
- parent/control retraining: なし

実行コマンド:

```bash
kaggle kernels push -p experiments/exp224_well_scaled_z_dz_features_on_exp218/kaggle/train_lgb0
kaggle kernels push -p experiments/exp224_well_scaled_z_dz_features_on_exp218/kaggle/train_lgb1
kaggle kernels push -p experiments/exp224_well_scaled_z_dz_features_on_exp218/kaggle/train_lgb2
```

push 結果:

- `kentookumura/exp224-wsz-exp218-lgb0` v1 pushed。URL: https://www.kaggle.com/code/kentookumura/exp224-wsz-exp218-lgb0
- `kentookumura/exp224-wsz-exp218-lgb1` v1 pushed。URL: https://www.kaggle.com/code/kentookumura/exp224-wsz-exp218-lgb1
- `kentookumura/exp224-wsz-exp218-lgb2` v1 pushed。URL: https://www.kaggle.com/code/kentookumura/exp224-wsz-exp218-lgb2

status 確認:

```bash
kaggle kernels status kentookumura/exp224-wsz-exp218-lgb0
kaggle kernels status kentookumura/exp224-wsz-exp218-lgb1
kaggle kernels status kentookumura/exp224-wsz-exp218-lgb2
```

- `train_lgb0`: `KernelWorkerStatus.RUNNING`
- `train_lgb1`: `KernelWorkerStatus.RUNNING`
- `train_lgb2`: `KernelWorkerStatus.RUNNING`

## 2026-07-09 Kaggle CPU split train 完了 / OOF 集計

status:

- `train_lgb0`: `KernelWorkerStatus.COMPLETE`
- `train_lgb1`: `KernelWorkerStatus.COMPLETE`
- `train_lgb2`: `KernelWorkerStatus.COMPLETE`

logs / output:

- `train_lgb0`: RMSE TVT 8.683606336244、RMSE target 8.683606208479、elapsed 8760.786 sec、full coverage pass
- `train_lgb1`: RMSE TVT 8.573438104937、RMSE target 8.573438257423、elapsed 9626.092 sec、full coverage pass
- `train_lgb2`: RMSE TVT 8.534973570049、RMSE target 8.534973421997、elapsed 12992.098 sec、full coverage pass
- coverage は 3 split すべて 3,783,989 rows / 773 wells、dropped rows 0

downloaded outputs:

- `kaggle/output/train_lgb0_v1/artifacts/`
- `kaggle/output/train_lgb1_v1/artifacts/`
- `kaggle/output/train_lgb2_v1/artifacts/`

split OOF aggregate:

- output: `kaggle/output/train_split_aggregate_v1/artifacts/`
- 3-config `lgb_mean` RMSE TVT: 8.538687041980
- RMSE target: 8.538683614280
- prediction SHA256: `83bec3ddef9e187c65ca17f1c8ee6daff6a5e0e0292ac1436b0c098768e498a2`
- exp218 parent `lgb_mean` 8.475793751657 から +0.062893290324 悪化
- exp148 feature surface `lgb_mean` 8.501281181896 から +0.037405860085 悪化

readout:

- top importance: `slp_b_d_50`, `grwr_fft_rotation_ratio_x_log1p_md_since`, `spatial_knn_dist`, `wsz_dz_over_likpf_tvt_p05_p95_range`, `dense_dist`
- worst wells top3: `86454a6f` RMSE 48.803342、`fb03ae90` RMSE 46.074345、`1b1eba53` RMSE 43.999814
- aggregate bucket は local exp072 `md_since` cache がないため tail-rank bucket のみ厳密再計算。distance bucket は各 split の既存 CSV を `split_individual_bucket_metrics.csv` に連結した。

判定:

- CV が exp218 / exp148 の両方より悪いため不採用。
- inference port、saved-booster aggregate manifest、submit は行わない。
- 後続では `likpf_mean_d` 由来 robust range を correction / replacement に使わず、readout または限定的な feature probe に留める。
