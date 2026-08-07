# exp228_direct_residual_correction_on_exp226 セッションノート

## 目的

KAGGLE_DIRECTION backlog `exp226_direct_residual_correction` を実装する。exp226 K16 fallback の group-safe OOF 残差 `TVT - exp226_oof_pred` を、exp218 と同じ ML feature surface で LightGBM 学習する。推論では exp226 inference v1 submission を base にして residual prediction を加える。

## 現在の状態

- Route: `ensemble`
- 状態: 実装済み / Kaggle train 未実行
- CV: 未計測
- LB: 未提出
- GPU cost: なし
- Active variants: 1 (`exp218_surface_direct_residual`)
- LightGBM config count: 3 (`lgb0`, `lgb1`, `lgb2`)
- Fold count: 5
- Total boosters: 15
- Train split:
  - `train_lgb0`: 5 boosters
  - `train_lgb1`: 5 boosters
  - `train_lgb2`: 5 boosters
- Parent/control retraining: なし

## 実装内容

- `.steering/20260709-exp228-direct-residual-correction-on-exp226/` を作成。
- `experiments/exp228_direct_residual_correction_on_exp226/` を exp218 からコピーして作成。
- `direct_residual_correction_on_exp226.py`
  - exp218 feature surface generation を再利用。
  - exp226 train v1 OOF artifact を `well_id + row_idx` で join。
  - `target_tvt = last_known_tvt + target` と exp226 OOF true TVT の alignment を検査。
  - `exp226_residual_target = target_tvt - exp226_oof_pred` を LightGBM target にする。
  - `selected_lgb_models` によって `lgb0/lgb1/lgb2` の単独 split train を可能にした。
  - inference は複数 split manifest を検出して全 fold boosters を平均し、exp226 inference submission に residual を加える。
- notebook source:
  - `exp228_direct_residual_correction_on_exp226_train.py`: split 実行の索引。全15 boosters 一括学習はしない。
  - `exp228_direct_residual_correction_on_exp226_train_lgb0.py`
  - `exp228_direct_residual_correction_on_exp226_train_lgb1.py`
  - `exp228_direct_residual_correction_on_exp226_train_lgb2.py`
  - `exp228_direct_residual_correction_on_exp226_train_aggregate.py`
  - `exp228_direct_residual_correction_on_exp226_inference.py`

## リークガード

- train residual は exp226 group-safe OOF prediction からのみ作る。
- full-train exp226 prediction は train residual に使わない。
- exp226 OOF true TVT は exp218 feature target との alignment check のみに使う。
- exp218 feature surface の target-free policy を維持する。
- LightGBM fold は well group split。
- true error / oracle / LB weight search はしない。

## push 前コスト確認

- Runtime: CPU (`runtime.kaggle.enable_gpu=false`)
- active variants: 1
- LightGBM configs: 3, split into 3 notebooks
- folds: 5
- total boosters: 15
- parent/control retraining: なし
- expected train loop:
  - `train_lgb0`: 1 config x 5 folds = 5 boosters
  - `train_lgb1`: 1 config x 5 folds = 5 boosters
  - `train_lgb2`: 1 config x 5 folds = 5 boosters
  - `train_aggregate`: 0 boosters

## コマンドログ

```bash
make new-steering EXP=exp228_direct_residual_correction_on_exp226
make new-exp EXP=exp228_direct_residual_correction_on_exp226 SOURCE=experiments/exp218_gr_wavelet_rotation_confidence_features_on_exp148
```

- result: steering / experiment scaffold 作成。

## 次の確認

```bash
.venv/bin/python -m py_compile \
  experiments/exp228_direct_residual_correction_on_exp226/direct_residual_correction_on_exp226.py \
  experiments/exp228_direct_residual_correction_on_exp226/exp228_direct_residual_correction_on_exp226_train.py \
  experiments/exp228_direct_residual_correction_on_exp226/exp228_direct_residual_correction_on_exp226_train_lgb0.py \
  experiments/exp228_direct_residual_correction_on_exp226/exp228_direct_residual_correction_on_exp226_train_lgb1.py \
  experiments/exp228_direct_residual_correction_on_exp226/exp228_direct_residual_correction_on_exp226_train_lgb2.py \
  experiments/exp228_direct_residual_correction_on_exp226/exp228_direct_residual_correction_on_exp226_train_aggregate.py \
  experiments/exp228_direct_residual_correction_on_exp226/exp228_direct_residual_correction_on_exp226_inference.py \
  experiments/exp228_direct_residual_correction_on_exp226/settings.py
.venv/bin/ruff check experiments/exp228_direct_residual_correction_on_exp226 --select F821,F401
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp228_direct_residual_correction_on_exp226/exp228_direct_residual_correction_on_exp226_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp228_direct_residual_correction_on_exp226/exp228_direct_residual_correction_on_exp226_train_lgb0.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp228_direct_residual_correction_on_exp226/exp228_direct_residual_correction_on_exp226_train_lgb1.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp228_direct_residual_correction_on_exp226/exp228_direct_residual_correction_on_exp226_train_lgb2.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp228_direct_residual_correction_on_exp226/exp228_direct_residual_correction_on_exp226_train_aggregate.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp228_direct_residual_correction_on_exp226/exp228_direct_residual_correction_on_exp226_inference.py
.venv/bin/python scripts/validate_experiment.py --experiment exp228_direct_residual_correction_on_exp226
```

- result: PASS
- `py_compile`: PASS
- `ruff check --select F821,F401`: PASS
- Jupytext convert: train / train_lgb0 / train_lgb1 / train_lgb2 / train_aggregate / inference すべて PASS
- Jupytext `--test`: train / train_lgb0 / train_lgb1 / train_lgb2 / train_aggregate / inference すべて PASS
- `validate_experiment.py`: PASS
- `prepare_kaggle_notebooks.py --strict`: train_lgb0 / train_lgb1 / train_lgb2 / train_aggregate / inference すべて PASS
- package metadata: all `enable_gpu=false`
- package `py_compile`: PASS

## Kaggle package

```bash
.venv/bin/python scripts/prepare_kaggle_notebooks.py --experiment exp228_direct_residual_correction_on_exp226 --notebook train_lgb0 --kernel-id kentookumura/exp228-direct-residual-exp226-train-lgb0 --title 'exp228 direct residual exp226 train lgb0' --run-on-push --strict
.venv/bin/python scripts/prepare_kaggle_notebooks.py --experiment exp228_direct_residual_correction_on_exp226 --notebook train_lgb1 --kernel-id kentookumura/exp228-direct-residual-exp226-train-lgb1 --title 'exp228 direct residual exp226 train lgb1' --run-on-push --strict
.venv/bin/python scripts/prepare_kaggle_notebooks.py --experiment exp228_direct_residual_correction_on_exp226 --notebook train_lgb2 --kernel-id kentookumura/exp228-direct-residual-exp226-train-lgb2 --title 'exp228 direct residual exp226 train lgb2' --run-on-push --strict
.venv/bin/python scripts/prepare_kaggle_notebooks.py --experiment exp228_direct_residual_correction_on_exp226 --notebook train_aggregate --kernel-id kentookumura/exp228-direct-residual-exp226-train-aggregate --title 'exp228 direct residual exp226 train aggregate' --run-on-push --strict
.venv/bin/python scripts/prepare_kaggle_notebooks.py --experiment exp228_direct_residual_correction_on_exp226 --notebook inference --kernel-id kentookumura/exp228-direct-residual-exp226-inference --title 'exp228 direct residual exp226 inference' --strict
```

- Generated package dirs:
  - `experiments/exp228_direct_residual_correction_on_exp226/kaggle/train_lgb0`
  - `experiments/exp228_direct_residual_correction_on_exp226/kaggle/train_lgb1`
  - `experiments/exp228_direct_residual_correction_on_exp226/kaggle/train_lgb2`
  - `experiments/exp228_direct_residual_correction_on_exp226/kaggle/train_aggregate`
  - `experiments/exp228_direct_residual_correction_on_exp226/kaggle/inference`

## 次の実行

```bash
kaggle kernels push -p experiments/exp228_direct_residual_correction_on_exp226/kaggle/train_lgb0
kaggle kernels push -p experiments/exp228_direct_residual_correction_on_exp226/kaggle/train_lgb1
kaggle kernels push -p experiments/exp228_direct_residual_correction_on_exp226/kaggle/train_lgb2
```

3 split が COMPLETE した後:

```bash
kaggle kernels push -p experiments/exp228_direct_residual_correction_on_exp226/kaggle/train_aggregate
```

aggregate CV と stress readout を確認してから inference / submit 判断する。

## Kaggle 実行ログ

- 2026-07-10 06:44 JST: CPU notebook として split train 3 本を push 済み。
  - `kentookumura/exp228-direct-residual-exp226-train-lgb0`: version 1 pushed, status `RUNNING`
  - `kentookumura/exp228-direct-residual-exp226-train-lgb1`: version 1 pushed, status `RUNNING`
  - `kentookumura/exp228-direct-residual-exp226-train-lgb2`: version 1 pushed, status `RUNNING`
- CLI の `logs -f` は途中出力なし。これはこの環境の既知挙動として扱い、空ログだけでは失敗判定しない。
- 2026-07-11: split train 3 本は `COMPLETE`。
  - `train_lgb0`: pooled RMSE TVT 9.042170562084868、elapsed 6485.358 sec
  - `train_lgb1`: pooled RMSE TVT 8.940004291330437、elapsed 3213.399 sec
  - `train_lgb2`: pooled RMSE TVT 8.94036689323167、elapsed 3007.949 sec
- 2026-07-11: `train_aggregate` v1 を push し、`COMPLETE`。
  - `lgb_mean_from_split_lgb0_lgb1_lgb2`: RMSE TVT 8.94408550082682
  - RMSE residual: 8.944085601265096
  - rows: 3,783,989、wells: 773
  - aggregate prediction SHA256: `239d53622af7cf3f3b421522de9a1f9cdda0a6ac3b99ba283dc8796032209da8`
  - aggregate summary: `/kaggle/working/artifacts/exp228_direct_residual_correction_on_exp226_split_aggregate_summary.json`
  - local temporary output check: `/tmp/exp228_aggregate_output/artifacts/exp228_direct_residual_correction_on_exp226_split_aggregate_summary.json`
- 比較:
  - exp226 CV 9.427109596582213 からは -0.4830240957553931 改善。
  - exp218 CV 8.475793751656624 からは +0.4682917491701961 悪化。
- 判断: 現行 exp218 ML anchor に届かないため、inference / submit は行わない。
