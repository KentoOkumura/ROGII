# exp164_spatial_prior_confidence_features_on_exp092_kaggle セッションノート

## 2026-07-01 実装

- `exp159_spatial_prior_confidence_features_on_exp092` の Colab 前提ディレクトリは採用せず、Kaggle Notebook 前提の clean 実験として `exp164_spatial_prior_confidence_features_on_exp092_kaggle` を作成した。
- `docs/legacy/steering/20260701-exp164-spatial-prior-confidence-features-on-exp092-kaggle/` を作成。
- `experiments/exp164_spatial_prior_confidence_features_on_exp092_kaggle/` を exp151 scaffold から作成し、実装本体は exp159 の spatial prior feature builder を checkpoint なしの Kaggle-first 版として移植した。
- 親実験は `exp092_u_projection_correction_disagreement_fullrun`、base cache は `exp072_exp063_full_replay_feature_cache`、spatial prior cache は `exp114_spatial_neighbor_prior_signal_audit`。
- Colab runner / manual upload / checkpoint 再開機構はこの exp164 には含めない。実行は Kaggle Notebook train を正とする。

## 実行コストガード

- active variant 数: 1 (`spatial_prior_confidence_addonly`)
- LightGBM config 数: 3 (`lgb0`, `lgb1`, `lgb2`)
- folds: 5 GroupKFold by `well`
- 合計 booster 数: 15
- 2026-07-02 時点の実行形態: CPU notebook 3 本に分割し、各 notebook は 1 config x 5 folds = 5 boosters を学習する。
- exp092 control 再学習: なし (`exp092_full_row_control.enabled=false`)
- baseline は保存済み exp092 `lgb1` CV 9.322479896 / Public LB 8.350 を参照する。

## 実装内容

- exp072 full replay 196 features に、exp092 と同じ U-projection correction / disagreement surface を再生成する。
- exp114 spatial prior OOF から `xy_only_k8` と `xy_plus_trajectory_shape_k8` の prior value / quality / disagreement / interaction features を追加する。
- exp118 best gate proxy は target-free な prior quality threshold と small correction proxy として特徴量化する。
- spatial prior を hard selector、direct correction、oracle path としては使わない。

## 検証

- `exp164_spatial_prior_confidence_features_on_exp092_kaggle_train.py` / `..._inference.py` を Jupytext percent 形式の notebook source として追加した。
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb ..._train.py ..._inference.py`: PASS
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test ..._train.py`: PASS
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test ..._inference.py`: PASS
- `python3 -m py_compile experiments/exp164_spatial_prior_confidence_features_on_exp092_kaggle/spatial_prior_confidence_features_on_exp092_kaggle.py experiments/exp164_spatial_prior_confidence_features_on_exp092_kaggle/settings.py experiments/exp164_spatial_prior_confidence_features_on_exp092_kaggle/exp164_spatial_prior_confidence_features_on_exp092_kaggle_train.py experiments/exp164_spatial_prior_confidence_features_on_exp092_kaggle/exp164_spatial_prior_confidence_features_on_exp092_kaggle_inference.py`: PASS
- `python3 -m json.tool ..._train.ipynb` / `python3 -m json.tool ..._inference.ipynb`: PASS
- `.venv/bin/ruff check experiments/exp164_spatial_prior_confidence_features_on_exp092_kaggle/spatial_prior_confidence_features_on_exp092_kaggle.py experiments/exp164_spatial_prior_confidence_features_on_exp092_kaggle/settings.py experiments/exp164_spatial_prior_confidence_features_on_exp092_kaggle/exp164_spatial_prior_confidence_features_on_exp092_kaggle_train.py experiments/exp164_spatial_prior_confidence_features_on_exp092_kaggle/exp164_spatial_prior_confidence_features_on_exp092_kaggle_inference.py`: PASS
- `make validate-exp EXP=exp164_spatial_prior_confidence_features_on_exp092_kaggle`: PASS

## 2026-07-02 CPU 分割実行

- ユーザー指示により、Kaggle CPU 実行へ切り替え、timeout 対策として train notebook を `train_lgb0` / `train_lgb1` / `train_lgb2` に分割した。
- `config.yaml` は `runtime.kaggle.enable_gpu=false`、`model.training.active_modes=[cpu_deterministic_threads8]` に変更した。
- 学習関数に `selected_lgb_configs` を追加し、各 notebook が 1 つの LightGBM config だけを学習できるようにした。
- Jupytext 生成時に kernelspec が無いと Kaggle Papermill が `ValueError: No kernel name found in notebook and no override provided.` で失敗するため、`--set-kernel python3` 付きで `.ipynb` を再生成した。
- 検証:
  - `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --set-kernel python3 --to ipynb ..._train.py ..._train_lgb0.py ..._train_lgb1.py ..._train_lgb2.py ..._inference.py`: PASS
  - `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test ..._train_lgb0.py ..._train_lgb1.py ..._train_lgb2.py`: PASS
  - `python3 -m py_compile` for helper/settings/train split scripts: PASS
  - `python3 -m json.tool ..._train_lgb0.ipynb ..._train_lgb1.ipynb ..._train_lgb2.ipynb`: PASS
  - `.venv/bin/ruff check --fix` then `ruff check`: PASS
  - `make validate-exp EXP=exp164_spatial_prior_confidence_features_on_exp092_kaggle`: PASS
- Prepared CPU Kaggle packages:
  - `make prepare-kaggle-notebooks EXP=exp164_spatial_prior_confidence_features_on_exp092_kaggle EXTRA_ARGS="--notebook train_lgb0 --kernel-id kentookumura/exp164-spatial-prior-conf-exp092-lgb0-train --title 'exp164 spatial prior conf exp092 lgb0 train' --run-on-push --strict"`: PASS
  - `make prepare-kaggle-notebooks EXP=exp164_spatial_prior_confidence_features_on_exp092_kaggle EXTRA_ARGS="--notebook train_lgb1 --kernel-id kentookumura/exp164-spatial-prior-conf-exp092-lgb1-train --title 'exp164 spatial prior conf exp092 lgb1 train' --run-on-push --strict"`: PASS
  - `make prepare-kaggle-notebooks EXP=exp164_spatial_prior_confidence_features_on_exp092_kaggle EXTRA_ARGS="--notebook train_lgb2 --kernel-id kentookumura/exp164-spatial-prior-conf-exp092-lgb2-train --title 'exp164 spatial prior conf exp092 lgb2 train' --run-on-push --strict"`: PASS
- Push / execution:
  - `kaggle kernels push -p experiments/exp164_spatial_prior_confidence_features_on_exp092_kaggle/kaggle/train_lgb0`: v1 pushed, then ERROR due missing kernelspec.
  - `kaggle kernels push -p experiments/exp164_spatial_prior_confidence_features_on_exp092_kaggle/kaggle/train_lgb1`: v1 pushed.
  - `kaggle kernels push -p experiments/exp164_spatial_prior_confidence_features_on_exp092_kaggle/kaggle/train_lgb2`: v1 pushed.
  - kernelspec fix 後、同じ 3 kernel id に v2 を再 push。
- Kaggle kernel ids:
  - `kentookumura/exp164-spatial-prior-conf-exp092-lgb0-train` / id_no `125586957` / URL `https://www.kaggle.com/code/kentookumura/exp164-spatial-prior-conf-exp092-lgb0-train`
  - `kentookumura/exp164-spatial-prior-conf-exp092-lgb1-train` / id_no `125586966` / URL `https://www.kaggle.com/code/kentookumura/exp164-spatial-prior-conf-exp092-lgb1-train`
  - `kentookumura/exp164-spatial-prior-conf-exp092-lgb2-train` / id_no `125586974` / URL `https://www.kaggle.com/code/kentookumura/exp164-spatial-prior-conf-exp092-lgb2-train`
- metadata pull で 3 本とも `enable_gpu=false`、`enable_internet=false`、`machine_shape=None`、competition source と exp072 / exp114 kernel sources を確認した。
- 2026-07-02 push 後 1 分時点の status:
  - lgb0 v2: `KernelWorkerStatus.RUNNING`
  - lgb1 v2: `KernelWorkerStatus.RUNNING`
  - lgb2 v2: `KernelWorkerStatus.RUNNING`

## 2026-07-02 CPU 分割実行完了

- ユーザー完了連絡後に `kaggle kernels status` を確認し、3 本とも `KernelWorkerStatus.COMPLETE`。
- Kaggle logs / notebook cell output から pooled RMSE を確認した。各 notebook は 1 config のみなので、logs 上の `lgb_mean` は単一 config と同一。
- Results:
  - `lgb0`: RMSE 9.660879008、exp092 `lgb1` baseline 9.322479896 比 +0.338399112、elapsed 11562.454 sec。
  - `lgb1`: RMSE 9.429441976、baseline 比 +0.106962080、elapsed 14093.613 sec。
  - `lgb2`: RMSE 9.415444308、baseline 比 +0.092964412、elapsed 15924.085 sec。
- 最良は `lgb2` だが baseline を上回れず、train-side CV は negative。inference / submit へは進めない。
- 3 config ensemble OOF を再計算するには各 notebook output の prediction を取得して結合する必要がある。現時点では個別 config がすべて negative のため、output archive は取得していない。
