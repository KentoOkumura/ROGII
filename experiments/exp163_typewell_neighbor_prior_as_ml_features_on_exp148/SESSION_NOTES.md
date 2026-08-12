# exp163_typewell_neighbor_prior_as_ml_features_on_exp148 セッションノート

## 2026-07-01 実装

- `docs/legacy/steering/20260701-exp163-typewell-neighbor-prior-as-ml-features-on-exp148/` を作成。
- `experiments/exp163_typewell_neighbor_prior_as_ml_features_on_exp148/` を exp162 の CPU split notebook 構成から作成。
- 親実験は `exp148_learned_likelihood_fulltrain_addonly_on_exp092`。
- exp148 の U-projection / learned likelihood confidence surface は残し、exp099 row context と exp065 native typewell overlap cluster から fold-safe neighbor prior を作る。
- 追加 feature group:
  - `typewell_neighbor_prior_value`
  - `typewell_neighbor_prior_quality`
  - `typewell_neighbor_prior_interaction`
  - `typewell_neighbor_prior_correction_proxy`
- Prior TVT path の direct selector、soft average、blend、postprocess replacement は入れない。
- 推論は未実装。まず split CPU train で OOF / worst-well / near-row / feature importance を見る。

## CPU split 実行ガード

- active variant 数: 1 (`typewell_neighbor_prior_addonly`)
- LightGBM config 数: 3 (`lgb0`, `lgb1`, `lgb2`)
- fold 数: 5
- 合計 booster 数: 15。ただし notebook は `lgb0` / `lgb1` / `lgb2` に分割し、1 notebook あたり 5 boosters。
- active mode: `cpu_deterministic_threads8`
- `runtime.kaggle.enable_gpu`: false
- exp148 control 再学習: なし
- baseline は保存済み exp148 `lgb_mean` CV 8.50128118189582 / Public LB 7.960 を参照する。

## 検証ログ

- `uv run python -m py_compile ...`: PASS
- `uv run ruff check ... --select F821`: PASS
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb ...train_lgb0.py`: PASS
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb ...train_lgb1.py`: PASS
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb ...train_lgb2.py`: PASS
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb ...train.py`: PASS
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb ...inference.py`: PASS
- `make validate-exp EXP=exp163_typewell_neighbor_prior_as_ml_features_on_exp148`: PASS
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test ...train_lgb0.py`: PASS
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test ...train_lgb1.py`: PASS
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test ...train_lgb2.py`: PASS
- `make prepare-kaggle-notebooks ... --notebook train_lgb0 ... --run-on-push --strict`: PASS
- `make prepare-kaggle-notebooks ... --notebook train_lgb1 ... --run-on-push --strict`: PASS
- `make prepare-kaggle-notebooks ... --notebook train_lgb2 ... --run-on-push --strict`: PASS
- `make validate-exp EXP=exp163_typewell_neighbor_prior_as_ml_features_on_exp148`: PASS after README section fix

## Kaggle package 方針

- `train_lgb0`: `kentookumura/exp163-typewell-prior-exp148-lgb0-train`
- `train_lgb1`: `kentookumura/exp163-typewell-prior-exp148-lgb1-train`
- `train_lgb2`: `kentookumura/exp163-typewell-prior-exp148-lgb2-train`
- 各 package は CPU / internet off / run-on-push 前提。
- 生成済み package:
  - `experiments/exp163_typewell_neighbor_prior_as_ml_features_on_exp148/kaggle/train_lgb0/`
  - `experiments/exp163_typewell_neighbor_prior_as_ml_features_on_exp148/kaggle/train_lgb1/`
  - `experiments/exp163_typewell_neighbor_prior_as_ml_features_on_exp148/kaggle/train_lgb2/`

## Kaggle train push

- 2026-07-01 に CPU split train を Kaggle へ push。
- `train_lgb0`: version 1、id_no `125546928`
  - https://www.kaggle.com/code/kentookumura/exp163-typewell-prior-exp148-lgb0-train
- `train_lgb1`: version 1、id_no `125546942`
  - https://www.kaggle.com/code/kentookumura/exp163-typewell-prior-exp148-lgb1-train
- `train_lgb2`: version 1、id_no `125546954`
  - https://www.kaggle.com/code/kentookumura/exp163-typewell-prior-exp148-lgb2-train
- `kaggle kernels pull ... -m`: 3 kernels PASS。CPU (`enable_gpu: false`, `machine_shape: "None"`), internet off を確認。
- push 直後の `kaggle kernels logs`: 3 kernels とも CLI logs は空。Kaggle CLI は実行中ログを返さないことがあるため、完了後に通常 `logs` で CV / fold 別 score / 保存先を確認する。
- `kaggle kernels status`: 3 kernels とも `KernelWorkerStatus.RUNNING`。

## Kaggle train result

- 2026-07-02 に `kaggle kernels status` で 3 kernels とも `KernelWorkerStatus.COMPLETE` を確認。
- `kaggle kernels logs`: 3 kernels PASS。fold 別 score と summary を取得。
- `kaggle kernels output`: 3 kernels の output を `/tmp/kaggle-output/exp163_typewell_neighbor_prior_as_ml_features_on_exp148/train_lgb{0,1,2}/` に取得。
- rows / wells / features: 3,783,989 rows / 773 wells / 315 features。
- pooled OOF:
  - `lgb0`: 8.575290758
  - `lgb1`: 8.572174727
  - `lgb2`: 8.571366316
  - 3-model `lgb_mean`: 8.519739843
- exp148 historical `lgb_mean` 8.501281182 から +0.018458661 悪化。
- 3-model mean prediction SHA proxy (`id,pred_tvt` rounded 8 decimals): `d594dec0eb36b8fef8ce3ed9d49c1dd70232926eb77a46507f5f232169f2a8ed`
- worst wells by 3-model mean: `86454a6f` 46.788967, `1b1eba53` 46.058109, `fb03ae90` 44.960031, `91b301ce` 36.829506。
- typewell prior features are used by the models, but the add-only surface does not improve the exp148 anchor. Decision: train-side rejected, no inference port, no submit.
