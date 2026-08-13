# exp184_heatmap_selector_compact_addonly_on_exp148 セッションノート

## 2026-07-05 実装

### 狙い

`backlog/KAGGLE_DIRECTION.md` の `exp184_heatmap_selector_compact_addonly_on_exp148` backlog を実装する。exp184 の heatmap selector signal を、exp148 の ML route anchor に compact add-only feature として足す。

### 実装方針

- 親: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- heatmap / selector 親: `exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158`
- heatmap source: `exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe`
- route: `ml_model`
- active variant: `heatmap_selector_compact_addonly`
- control retraining: なし
- direct exp184 selected TVT replacement / blend / postprocess / hard gate: なし
- inference / submit: 初期実装では対象外

### Feature

exp184 OOF predictions の best Viterbi selected path を読む。

- variant: `viterbi_sw200_bias000_jw000_jf025_d0150_std999999_md0000_seg012`
- mode: `viterbi`

downstream feature に使う列:

- selected candidate code / family flag
- selected TVT minus `likpf_mean`
- selected TVT minus exp148 OOF
- segment length / boundary distance / local switch / path jump
- exp182 heatmap real score / margin / entropy / confidence proxy
- heatmap sparse sample distance / far sparse flag
- real-vs-shuffled/no-GR confidence gap
- heatmap real top1/top3 vs selected path absolute deltas

leakage 防止として、以下は明示的に使わない:

- `true_tvt`
- `abs_error`
- `within10`
- `target_in_grid`
- `best_mode`
- `oracle_candidate`
- `oracle_label`
- oracle / true-error rank 系列

### Kaggle train 実行前確認

- 実行対象: `exp184_heatmap_selector_compact_addonly_on_exp148`
- runtime: CPU
- Kaggle GPU: disabled (`runtime.kaggle.enable_gpu=false`)
- active mode: `cpu_deterministic_threads8`
- active variants: 1
- LightGBM configs: 3 (`lgb0`, `lgb1`, `lgb2`)
- folds: 5
- planned boosters: 15
- train notebook split: `train_lgb0` / `train_lgb1` / `train_lgb2`
- planned boosters per split notebook: 5
- control / parent retraining: なし

### 再現性メモ

- feature merge 自体に新規乱数はない。
- exp072 / exp145 / exp182 / exp184 は upstream fixed Kaggle output として読む。
- LightGBM CPU は `deterministic=true`、`force_col_wise=true`、`n_jobs/num_threads=8`。
- deterministic submission anchor ではない。`submission.csv` は作らない。

### 検証予定

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp184_heatmap_selector_compact_addonly_on_exp148/exp184_heatmap_selector_compact_addonly_on_exp148_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp184_heatmap_selector_compact_addonly_on_exp148/exp184_heatmap_selector_compact_addonly_on_exp148_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp184_heatmap_selector_compact_addonly_on_exp148/exp184_heatmap_selector_compact_addonly_on_exp148_train_lgb0.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp184_heatmap_selector_compact_addonly_on_exp148/exp184_heatmap_selector_compact_addonly_on_exp148_train_lgb0.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp184_heatmap_selector_compact_addonly_on_exp148/exp184_heatmap_selector_compact_addonly_on_exp148_train_lgb1.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp184_heatmap_selector_compact_addonly_on_exp148/exp184_heatmap_selector_compact_addonly_on_exp148_train_lgb1.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp184_heatmap_selector_compact_addonly_on_exp148/exp184_heatmap_selector_compact_addonly_on_exp148_train_lgb2.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp184_heatmap_selector_compact_addonly_on_exp148/exp184_heatmap_selector_compact_addonly_on_exp148_train_lgb2.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp184_heatmap_selector_compact_addonly_on_exp148/exp184_heatmap_selector_compact_addonly_on_exp148_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp184_heatmap_selector_compact_addonly_on_exp148/exp184_heatmap_selector_compact_addonly_on_exp148_inference.py
.venv/bin/python -m py_compile experiments/exp184_heatmap_selector_compact_addonly_on_exp148/heatmap_selector_compact_addonly_on_exp148.py experiments/exp184_heatmap_selector_compact_addonly_on_exp148/exp184_heatmap_selector_compact_addonly_on_exp148_train.py experiments/exp184_heatmap_selector_compact_addonly_on_exp148/exp184_heatmap_selector_compact_addonly_on_exp148_train_lgb0.py experiments/exp184_heatmap_selector_compact_addonly_on_exp148/exp184_heatmap_selector_compact_addonly_on_exp148_train_lgb1.py experiments/exp184_heatmap_selector_compact_addonly_on_exp148/exp184_heatmap_selector_compact_addonly_on_exp148_train_lgb2.py experiments/exp184_heatmap_selector_compact_addonly_on_exp148/exp184_heatmap_selector_compact_addonly_on_exp148_inference.py experiments/exp184_heatmap_selector_compact_addonly_on_exp148/settings.py
.venv/bin/ruff check experiments/exp184_heatmap_selector_compact_addonly_on_exp148/heatmap_selector_compact_addonly_on_exp148.py experiments/exp184_heatmap_selector_compact_addonly_on_exp148/exp184_heatmap_selector_compact_addonly_on_exp148_train.py experiments/exp184_heatmap_selector_compact_addonly_on_exp148/exp184_heatmap_selector_compact_addonly_on_exp148_train_lgb0.py experiments/exp184_heatmap_selector_compact_addonly_on_exp148/exp184_heatmap_selector_compact_addonly_on_exp148_train_lgb1.py experiments/exp184_heatmap_selector_compact_addonly_on_exp148/exp184_heatmap_selector_compact_addonly_on_exp148_train_lgb2.py experiments/exp184_heatmap_selector_compact_addonly_on_exp148/exp184_heatmap_selector_compact_addonly_on_exp148_inference.py --select F821
make validate-exp EXP=exp184_heatmap_selector_compact_addonly_on_exp148
```

### 検証結果

- `py_compile`: pass
- `ruff --select F821`: pass
- Jupytext train / train_lgb0 / train_lgb1 / train_lgb2 / inference `--to ipynb --test`: pass
- `make validate-exp EXP=exp184_heatmap_selector_compact_addonly_on_exp148`: pass
- local helper smoke: exp072 先頭 100 行で exp184 selected path + exp182 heatmap compact + exp148 OOF 差分を生成し、31 features、missing rate 0.0、exp148 OOF available を確認した。

### Kaggle train push 前ガード

- active variants: 1 (`heatmap_selector_compact_addonly`)
- LightGBM configs: 3 (`lgb0`, `lgb1`, `lgb2`)
- folds: 5
- planned boosters: 15
- split train notebooks: `train_lgb0` / `train_lgb1` / `train_lgb2`
- planned boosters per split: 5
- control / parent retraining: なし
- runtime: CPU、Kaggle `enable_gpu=false`、`deterministic=true`、`force_col_wise=true`

## 2026-07-05 CPU split train 変更

ユーザー指示により、Kaggle train を GPU 単一 notebook から CPU split train に変更した。

- `config.yaml`: active mode を `cpu_deterministic_threads8` に変更し、`runtime.kaggle.enable_gpu=false` を設定した。
- helper: `selected_lgb_config_indices` を追加し、`lgb0` / `lgb1` / `lgb2` の個別学習を manifest / summary に記録できるようにした。
- notebook source: `exp184_heatmap_selector_compact_addonly_on_exp148_train_lgb0.py` / `train_lgb1.py` / `train_lgb2.py` を追加した。
- 各 split notebook は active variant 1、active mode 1、fold 5 の 5 boosters だけを実行する。
- 3 split 合計は 15 boosters。parent/control 再学習は引き続きなし。
- `make prepare-kaggle-notebooks` は `train_lgb0` / `train_lgb1` / `train_lgb2` の 3本で PASS。
- prepared metadata は 3本とも `enable_gpu=false`、`enable_internet=false`、`run_on_push=true`、competition source `rogii-wellbore-geology-prediction`。
- prepared kernel ids:
  - `kentookumura/exp184-heatmap-selcompact-exp148-train-lgb0`
  - `kentookumura/exp184-heatmap-selcompact-exp148-train-lgb1`
  - `kentookumura/exp184-heatmap-selcompact-exp148-train-lgb2`
- Kaggle train はまだ push / 実行していない。

## 2026-07-05 Kaggle CPU split train push

ユーザー指示「kaggleで実行してください」により、prepared 済み CPU split train 3本を Kaggle に push した。

### 実行前ガード

- active variants: 1 (`heatmap_selector_compact_addonly`)
- active modes: 1 (`cpu_deterministic_threads8`)
- Kaggle GPU: disabled (`enable_gpu=false`)
- split notebooks: `train_lgb0` / `train_lgb1` / `train_lgb2`
- boosters per split: 5
- total planned boosters: 15
- parent/control retraining: なし

### Push 結果

- `kaggle kernels push -p experiments/exp184_heatmap_selector_compact_addonly_on_exp148/kaggle/train_lgb0`: success、Kernel version 1、URL `https://www.kaggle.com/code/kentookumura/exp184-heatmap-selcompact-exp148-train-lgb0`
- `kaggle kernels push -p experiments/exp184_heatmap_selector_compact_addonly_on_exp148/kaggle/train_lgb1`: success、Kernel version 1、URL `https://www.kaggle.com/code/kentookumura/exp184-heatmap-selcompact-exp148-train-lgb1`
- `kaggle kernels push -p experiments/exp184_heatmap_selector_compact_addonly_on_exp148/kaggle/train_lgb2`: success、Kernel version 1、URL `https://www.kaggle.com/code/kentookumura/exp184-heatmap-selcompact-exp148-train-lgb2`

### Status

- `kentookumura/exp184-heatmap-selcompact-exp148-train-lgb0`: `KernelWorkerStatus.RUNNING`
- `kentookumura/exp184-heatmap-selcompact-exp148-train-lgb1`: `KernelWorkerStatus.RUNNING`
- `kentookumura/exp184-heatmap-selcompact-exp148-train-lgb2`: `KernelWorkerStatus.RUNNING`

CLI は Kaggle CLI 2.2.0 の upgrade warning を出したが、push/status は成功した。train-side CV は未完了。完了後に logs を確認し、必要なら split 3本の output を取得して横断 `lgb_mean` ensemble CV を計算する。

## 2026-07-05 Kaggle CPU split train 完了

ユーザーの「完了しました」後に Kaggle status を再確認し、CPU split train 3本がすべて `COMPLETE` であることを確認した。

### Status

- `kentookumura/exp184-heatmap-selcompact-exp148-train-lgb0`: `KernelWorkerStatus.COMPLETE`
- `kentookumura/exp184-heatmap-selcompact-exp148-train-lgb1`: `KernelWorkerStatus.COMPLETE`
- `kentookumura/exp184-heatmap-selcompact-exp148-train-lgb2`: `KernelWorkerStatus.COMPLETE`

### Kaggle output

必要な OOF prediction と summary を確認するため、3本の output を取得した。

- `kaggle/output/train_lgb0_v1`
- `kaggle/output/train_lgb1_v1`
- `kaggle/output/train_lgb2_v1`

各 split の common summary:

- rows: 3,783,989
- wells: 773
- feature join coverage: pass
- features: 322
- hmp184 compact generated features: 28

Kaggle split train の input source に exp148 train output がなかったため、optional exp148 OOF delta features は unavailable だった。local helper smoke では exp148 OOF 差分込みで 31 features を生成していたが、Kaggle 実行は 28 hmp184 features で完了した。

### CV

| model | RMSE TVT | RMSE target | elapsed sec |
| --- | ---: | ---: | ---: |
| lgb0 | 8.710685277 | 8.710685059 | 11120.622 |
| lgb1 | 8.639432353 | 8.639432520 | 8422.584 |
| lgb2 | 8.611075285 | 8.611075086 | 10234.615 |

3 split の OOF prediction を chunked streaming で結合し、cross-split `lgb_mean` を計算した。最初の full DataFrame merge は local memory OOM で killed になったため、chunked に切り替えた。

- cross-split `lgb_mean` RMSE TVT: 8.604130846223319
- cross-split `lgb_mean` RMSE target: 8.604130684382842
- delta vs exp148 GPU historical `lgb_mean`: +0.10284966432749876
- delta vs exp148 CPU runtime `lgb_mean`: +0.0754327322233177
- output summary: `artifacts/exp184_heatmap_selector_compact_addonly_on_exp148_split_lgb_mean_summary.json`
- output metrics: `artifacts/exp184_heatmap_selector_compact_addonly_on_exp148_split_lgb_mean_metrics.csv`

### 判定

`completed_train_side_rejected_no_submit`。

exp148 anchor から大きく悪化し、exp188 add-only selector confidence よりも悪いため、inference port と submit は行わない。厳密な 31-feature rerun は split train kernel sources に exp148 train output を追加すれば可能だが、今回の 28-feature result が十分 negative なので現時点では実行しない。
