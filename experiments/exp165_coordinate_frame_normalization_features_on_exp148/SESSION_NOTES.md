# exp165_coordinate_frame_normalization_features_on_exp148 セッションノート

## 2026-07-02 実装

- `docs/legacy/steering/20260702-exp165-coordinate-frame-normalization-features-on-exp148/` を作成。
- `experiments/exp165_coordinate_frame_normalization_features_on_exp148/` を exp163 の CPU split notebook 構成から作成。
- 親実験は `exp148_learned_likelihood_fulltrain_addonly_on_exp092`。
- exp148 の U-projection / learned likelihood confidence surface は残し、raw horizontal well の `MD/X/Y/Z` を known-prefix anchor と prefix-tail azimuth で正規化した coordinate-frame features を add-only する。
- 追加 feature group:
  - `coordinate_frame_geometry`
  - `coordinate_frame_direction`
  - `coordinate_frame_derivative`
  - `coordinate_frame_interaction`
- 座標値の direct TVT candidate、hard correction、row-wise selector、blend、postprocess replacement は入れない。
- 推論は未実装。まず split CPU train で OOF / worst-well / near-row / feature importance を見る。

## CPU split 実行ガード

- active variant 数: 1 (`coordinate_frame_addonly`)
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
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb ...train.py`: PASS
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb ...train_lgb0.py`: PASS
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb ...train_lgb1.py`: PASS
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb ...train_lgb2.py`: PASS
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb ...inference.py`: PASS
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test ...train.py`: PASS
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test ...train_lgb0.py`: PASS
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test ...train_lgb1.py`: PASS
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test ...train_lgb2.py`: PASS
- `JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test ...inference.py`: PASS
- `make validate-exp EXP=exp165_coordinate_frame_normalization_features_on_exp148`: PASS
- 合成 raw well 20 rows / eval 10 rows で `build_coordinate_frame_features()` smoke: PASS。38 features、feature groups は `coordinate_frame_geometry` 10、`coordinate_frame_direction` 10、`coordinate_frame_derivative` 9、`coordinate_frame_interaction` 8、finite check PASS。
- `.venv/bin/python -m json.tool` for train / train_lgb0 / train_lgb1 / train_lgb2 / inference notebooks: PASS
- `rg -n "__file__|Path\\(__file__\\)" ...train.py ...inference.py`: no matches

## Kaggle package 方針

- `train_lgb0`: `kentookumura/exp165-coordinate-frame-exp148-lgb0-train`
- `train_lgb1`: `kentookumura/exp165-coordinate-frame-exp148-lgb1-train`
- `train_lgb2`: `kentookumura/exp165-coordinate-frame-exp148-lgb2-train`
- 各 package は CPU / internet off / run-on-push 前提。
- 生成済み package:
  - `experiments/exp165_coordinate_frame_normalization_features_on_exp148/kaggle/train_lgb0/`
  - `experiments/exp165_coordinate_frame_normalization_features_on_exp148/kaggle/train_lgb1/`
  - `experiments/exp165_coordinate_frame_normalization_features_on_exp148/kaggle/train_lgb2/`
- `kernel-metadata.json`: 3 package とも `enable_gpu=false`、`enable_internet=false`、`run_on_push=true`、competition source `rogii-wellbore-geology-prediction`、kernel sources `kentookumura/exp072-exp063-full-replay-feature-cache-train` / `kentookumura/exp145-train`。

## Kaggle train push

- 2026-07-02 に CPU split train を Kaggle へ push。
- `train_lgb0`: version 1、id_no `125610549`
  - https://www.kaggle.com/code/kentookumura/exp165-coordinate-frame-exp148-lgb0-train
- `train_lgb1`: version 1、id_no `125610563`
  - https://www.kaggle.com/code/kentookumura/exp165-coordinate-frame-exp148-lgb1-train
- `train_lgb2`: version 1、id_no `125610586`
  - https://www.kaggle.com/code/kentookumura/exp165-coordinate-frame-exp148-lgb2-train
- `kaggle kernels pull ... -m`: 3 kernels PASS。CPU (`enable_gpu: false`, `machine_shape: "None"`), internet off を確認。
- push 後の `kaggle kernels status`: 3 kernels とも `KernelWorkerStatus.RUNNING`。
- push 直後の `kaggle kernels logs`: 3 kernels とも warning 以外は空。Kaggle CLI は実行中ログを返さないことがあるため、完了後に通常 `logs` で CV / fold 別 score / 保存先を確認する。

## Kaggle train 完了

- 2026-07-02 に `lgb0` / `lgb1` / `lgb2` はすべて `KernelWorkerStatus.COMPLETE`。
- ログ取得:
  - `/tmp/exp165_lgb0_logs.json`
  - `/tmp/exp165_lgb1_logs.json`
  - `/tmp/exp165_lgb2_logs.json`
- fold 別 RMSE:
  - `lgb0`: fold0 9.203312584、fold1 8.997133959、fold2 7.497428861、fold3 8.292190260、fold4 9.008076354
  - `lgb1`: fold0 9.027207676、fold1 8.785035816、fold2 7.475383269、fold3 8.600408558、fold4 8.951083386
  - `lgb2`: fold0 9.021196787、fold1 8.769281625、fold2 7.469313911、fold3 8.616152468、fold4 9.105819094
- split 単体 pooled RMSE:
  - `lgb0`: 8.623039477
  - `lgb1`: 8.586673413
  - `lgb2`: 8.616753590
- Kaggle output の prediction CSV はローカル大容量取得が接続断で不安定だったため、3 split output を input にした軽量集約 notebook を追加した。
- `train_aggregate`: `kentookumura/exp165-coordinate-frame-exp148-aggregate-train`
  - v1 は generated notebook に kernelspec が無く `No kernel name found` で ERROR。
  - v2 で kernelspec を追加して COMPLETE。id_no `125649416`。
  - output: `/tmp/kaggle-output/exp165_coordinate_frame_normalization_features_on_exp148/train_aggregate/`
- 3-model `lgb_mean`: 8.549931602
- exp148 historical `lgb_mean` 8.501281182 から +0.048650420 悪化。
- 3-model mean prediction SHA proxy (`id,pred_tvt` rounded 8 decimals): `e58c971423e3972bd29a8bd3cfd328835964df93e95315c1c07528849909e535`
- worst wells top5:
  - `86454a6f`: RMSE 47.182148
  - `1b1eba53`: RMSE 46.856087
  - `fb03ae90`: RMSE 45.687022
  - `91b301ce`: RMSE 37.431130
  - `81bf5923`: RMSE 32.283569

## 判定

- exp148 historical `lgb_mean` を改善できなかったため、推論化・提出はしない。
- raw coordinate-frame normalization features は exp148 の既存 learned likelihood / projection surface に対して add-only 改善を出せなかった。
- direct TVT candidate、hard correction、row-wise selector、postprocess replacement には展開しない。
