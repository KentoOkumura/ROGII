# exp188_exp183_selector_confidence_addonly_on_exp148 セッションノート

## 2026-07-04 実装

### 狙い

`KAGGLE_DIRECTION.md` の `exp183_selector_confidence_addonly_on_exp148` backlog を実験化する。実験開始時点で番号を確定し、`exp188_exp183_selector_confidence_addonly_on_exp148` とする。

exp183 の cluster-outlier prior 入り selector selected path が、exp148 の ML route anchor に add-only confidence feature として効くかを確認する。

### 実装方針

- 親: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- selector 親: `exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector`
- route: `ml_model`
- active variant: `exp183_selector_confidence_addonly`
- control retraining: なし
- direct exp183 selected TVT replacement / blend / postprocess / hard gate: なし
- inference / submit: 初期実装では対象外

### Feature

exp183 OOF predictions の best Viterbi selected path を読む。

- variant: `viterbi_sw200_bias000_jw100_jf025_d0075_std999999_md0000_seg001`
- mode: `viterbi`

downstream feature に使う列:

- selected candidate code / family flag
- selected TVT minus last-known / PF/Beam/dense candidates
- candidate spread / range
- local switch / path jump / segment length / segment boundary distance
- exp148 OOF prediction artifact が見つかる場合のみ selected TVT minus exp148 OOF

leakage 防止として、以下は明示的に使わない:

- `true_tvt`
- `abs_error`
- `oracle_candidate`
- `oracle_label`

### Kaggle train 実行前確認

- 実行対象: `exp188_exp183_selector_confidence_addonly_on_exp148`
- GPU: enabled
- active variants: 1
- LightGBM configs: 3 (`lgb0`, `lgb1`, `lgb2`)
- folds: 5
- planned boosters: 15
- train notebook split: なし
- control / parent retraining: なし

### 再現性メモ

- exp188 の feature merge 自体に新規乱数はない。
- exp072 / exp145 / exp183 は upstream fixed Kaggle output として読む。
- LightGBM GPU は `gpu_use_dp=true`、`deterministic=true`、`force_col_wise=true`、`n_jobs/num_threads=8`。
- deterministic submission anchor ではない。`submission.csv` は作らない。

### 検証予定

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp188_exp183_selector_confidence_addonly_on_exp148/exp188_exp183_selector_confidence_addonly_on_exp148_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp188_exp183_selector_confidence_addonly_on_exp148/exp188_exp183_selector_confidence_addonly_on_exp148_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp188_exp183_selector_confidence_addonly_on_exp148/exp188_exp183_selector_confidence_addonly_on_exp148_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp188_exp183_selector_confidence_addonly_on_exp148/exp188_exp183_selector_confidence_addonly_on_exp148_inference.py
.venv/bin/python -m py_compile experiments/exp188_exp183_selector_confidence_addonly_on_exp148/exp183_selector_confidence_addonly_on_exp148.py experiments/exp188_exp183_selector_confidence_addonly_on_exp148/exp188_exp183_selector_confidence_addonly_on_exp148_train.py experiments/exp188_exp183_selector_confidence_addonly_on_exp148/exp188_exp183_selector_confidence_addonly_on_exp148_inference.py experiments/exp188_exp183_selector_confidence_addonly_on_exp148/settings.py
.venv/bin/ruff check experiments/exp188_exp183_selector_confidence_addonly_on_exp148/exp183_selector_confidence_addonly_on_exp148.py experiments/exp188_exp183_selector_confidence_addonly_on_exp148/exp188_exp183_selector_confidence_addonly_on_exp148_train.py experiments/exp188_exp183_selector_confidence_addonly_on_exp148/exp188_exp183_selector_confidence_addonly_on_exp148_inference.py --select F821
make validate-exp EXP=exp188_exp183_selector_confidence_addonly_on_exp148
```

### 検証結果

- `py_compile`: pass
- `ruff --select F821`: pass
- Jupytext train / inference convert and `--test`: pass
- `make validate-exp EXP=exp188_exp183_selector_confidence_addonly_on_exp148`: pass

### Kaggle package prepare

```bash
make prepare-kaggle-notebooks EXP=exp188_exp183_selector_confidence_addonly_on_exp148 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp188-exp183-selector-confidence-addonly-on-exp148-train --title 'exp188 exp183 selector confidence addonly on exp148 train' --run-on-push --strict"
```

- package: `experiments/exp188_exp183_selector_confidence_addonly_on_exp148/kaggle/train`
- kernel id: `kentookumura/exp188-exp183-selector-confidence-addonly-on-exp148-train`
- title: `exp188 exp183 selector confidence addonly on exp148 train`
- GPU: true
- internet: false
- kernel sources:
  - `kentookumura/exp072-exp063-full-replay-feature-cache-train`
  - `kentookumura/exp145-train`
  - `kentookumura/exp183-copcf-train`
- bootstrap config confirms `selected_variant=exp183_selector_confidence_addonly` and `active_modes=[gpu_repro_guard_dp_threads8]`.

### Kaggle train 初回 push 失敗

```bash
make push-kaggle-train EXP=exp188_exp183_selector_confidence_addonly_on_exp148
```

`kaggle kernels push` が `SaveKernel` 400 を返した。詳細 message はなし。id/title slug は一致しているが、exp183 と同様に長い slug が Kaggle 側制約に触れた可能性があるため、実験番号は変えず kernel id/title だけ短縮する。

- old train id: `kentookumura/exp188-exp183-selector-confidence-addonly-on-exp148-train`
- new train id: `kentookumura/exp188-exp183-selconf-exp148-train`
- new train title: `exp188 exp183 selconf exp148 train`

### Kaggle train v1 実行開始

```bash
make prepare-kaggle-notebooks EXP=exp188_exp183_selector_confidence_addonly_on_exp148 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp188-exp183-selconf-exp148-train --title 'exp188 exp183 selconf exp148 train' --run-on-push --strict"
make push-kaggle-train EXP=exp188_exp183_selector_confidence_addonly_on_exp148
```

- kernel id: `kentookumura/exp188-exp183-selconf-exp148-train`
- URL: https://www.kaggle.com/code/kentookumura/exp188-exp183-selconf-exp148-train
- version: 1
- push result: `Kernel version 1 successfully pushed`
- initial status: `KernelWorkerStatus.RUNNING`
- initial logs: empty (expected while running with current Kaggle CLI behavior)
- metadata: GPU true, internet false, 3 kernel sources

### Kaggle train v1 失敗

```bash
kaggle kernels status kentookumura/exp188-exp183-selconf-exp148-train
kaggle kernels logs kentookumura/exp188-exp183-selconf-exp148-train
```

- status: `KernelWorkerStatus.ERROR`
- notebook log:
  - support files bootstrap は成功。
  - exp072 full replay cache、exp145 learned features、exp183 selector OOF predictions はすべて検出。
  - `learned feature rows: 3783989 wells: 773 columns: 51`
  - `selector feature rows: 3783989 wells: 773`
  - その後 `Kernel died while waiting for execute reply.` / `nbclient.exceptions.DeadKernelError: Kernel died`
- Python 例外 traceback ではなく kernel 死亡のため、学習 cell 内のメモリ不足が主因と判断。

### Kaggle train v2 修正

- `exp183_selector_confidence_addonly_on_exp148.py` の LightGBM 学習で、全行・全特徴の `x_matrix = frame[feature_columns].to_numpy(...)` を事前生成しないよう変更。
- 各 fold の直前に `x_train` / `x_valid` だけを `float32` 化し、fold 完了後に `del` と `gc.collect()` で解放する。
- 変更範囲は学習時のメモリ保持方法のみ。特徴量、variant、LightGBM config、fold 数、booster 数は v1 と同じ。
- validation:
  - `py_compile`: pass
  - `ruff --select F821`: pass
  - Jupytext train convert and `--test`: pass
  - `make validate-exp EXP=exp188_exp183_selector_confidence_addonly_on_exp148`: pass

```bash
make prepare-kaggle-notebooks EXP=exp188_exp183_selector_confidence_addonly_on_exp148 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp188-exp183-selconf-exp148-train --title 'exp188 exp183 selconf exp148 train' --run-on-push --strict"
make push-kaggle-train EXP=exp188_exp183_selector_confidence_addonly_on_exp148
kaggle kernels status kentookumura/exp188-exp183-selconf-exp148-train
```

- v2 package prepare: pass
- v2 push: `Kernel version 2 successfully pushed`
- URL: https://www.kaggle.com/code/kentookumura/exp188-exp183-selconf-exp148-train
- initial status: `KernelWorkerStatus.RUNNING`

### Kaggle train v2 失敗

```bash
kaggle kernels status kentookumura/exp188-exp183-selconf-exp148-train
kaggle kernels logs kentookumura/exp188-exp183-selconf-exp148-train
```

- status: `KernelWorkerStatus.ERROR`
- notebook log:
  - support files bootstrap、設定表示、入力 path 解決は成功。
  - notebook preview cell で `learned feature rows: 3783989 wells: 773 columns: 51`、`selector feature rows: 3783989 wells: 773` まで表示。
  - `_fit_one_variant_mode` の開始 JSON は出ていない。
  - その後 `Kernel died while waiting for execute reply.` / `nbclient.exceptions.DeadKernelError: Kernel died`
- v1 と同じく Python 例外ではない。LightGBM fold 学習前、notebook preview の全件 DataFrame 保持と本実行側の全件再ロード・merge/concat のピークメモリで落ちたと判断。

### Kaggle train v3 修正

- train notebook の入力確認セルを軽量化:
  - exp145 learned feature preview は `pd.read_csv(..., nrows=5)` のみ。
  - exp183 selector preview は `pd.read_csv(..., nrows=1000)` から selected variant/mode の先頭だけ表示。
  - preview DataFrame は表示後に `del` + `gc.collect()`。
- train 実行側のピークメモリ削減:
  - `add_anchor_columns` を full-frame merge から well-key `map` に変更。
  - exp072 / exp145 / projection / exp183 の有限値チェックを全列一括 `to_numpy` から column-wise に変更。
  - exp145 learned features と exp183 selector features は row-order 一致時に full merge せず、float32 列を in-place assignment。
  - projection / exp183 の追加も full-frame `pd.concat` ではなく追加列 assignment。
  - exp183 candidate summary は `np.column_stack` を使わず、sum/sumsq/min/max の逐次集計。
  - row-level prediction DataFrame は `lgb_mean` のみ保持し、各 lgb config の fold metrics / feature importance / saved model は維持。
- 変更範囲はメモリ保持と preview のみに限定。学習対象は 1 variant、3 LightGBM configs、5 folds、15 boosters のまま。
- validation:
  - `py_compile`: pass
  - `ruff --select F821`: pass
  - Jupytext train convert and `--test`: pass
  - `make validate-exp EXP=exp188_exp183_selector_confidence_addonly_on_exp148`: pass

```bash
make prepare-kaggle-notebooks EXP=exp188_exp183_selector_confidence_addonly_on_exp148 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp188-exp183-selconf-exp148-train --title 'exp188 exp183 selconf exp148 train' --run-on-push --strict"
make push-kaggle-train EXP=exp188_exp183_selector_confidence_addonly_on_exp148
kaggle kernels status kentookumura/exp188-exp183-selconf-exp148-train
```

- v3 package prepare: pass
- v3 push: `Kernel version 3 successfully pushed`
- URL: https://www.kaggle.com/code/kentookumura/exp188-exp183-selconf-exp148-train
- initial status: `KernelWorkerStatus.RUNNING`

### Kaggle train v3 完了

```bash
kaggle kernels status kentookumura/exp188-exp183-selconf-exp148-train
kaggle kernels logs kentookumura/exp188-exp183-selconf-exp148-train
kaggle kernels output kentookumura/exp188-exp183-selconf-exp148-train -p experiments/exp188_exp183_selector_confidence_addonly_on_exp148/kaggle/output/train_v3
```

- status: `KernelWorkerStatus.COMPLETE`
- kernel: `kentookumura/exp188-exp183-selconf-exp148-train`
- version: 3
- output: `experiments/exp188_exp183_selector_confidence_addonly_on_exp148/kaggle/output/train_v3/`
- generated artifacts:
  - `artifacts/exp188_exp183_selector_confidence_addonly_on_exp148_summary.json`
  - `artifacts/exp188_exp183_selector_confidence_addonly_on_exp148_metrics.csv`
  - `artifacts/exp188_exp183_selector_confidence_addonly_on_exp148_predictions.csv.gz`
  - `artifacts/exp188_exp183_selector_confidence_addonly_on_exp148_lgb_models/manifest.json`
  - 15 LightGBM booster files
- rows / wells: 3,783,989 / 773
- features: 326
- exp183 selector features: 32
- feature join coverage: pass、dropped rows 0、dropped wells 0
- elapsed: 14,426.84 sec

Pooled CV:

| model | RMSE TVT | prediction sha256 |
| --- | ---: | --- |
| lgb0 | 8.620017124 | `ba1c3c72a986771d0cfc8e1ee7b192dbe2ab2a42093fb8f0da7f3c94e98b574b` |
| lgb1 | 8.568069226 | `deae5a8bb3cf7be3ba2fc4f2cb1509504a5f6b884718ed293a91b50714f91cbe` |
| lgb2 | 8.576058237 | `fb213c84e77a035ce7074b243cc2c88bca927a43fa4d5028370dee0390521705` |
| lgb_mean | 8.539573790 | `f008092f52656be999db63db751d96e14f4f1f87be685742d3c5cab289db74f5` |

exp148 historical `lgb_mean` CV 8.501281182 から +0.038292608 悪化。train-side negative のため、inference port / submit には進めない。一方、replacement-only は exp148 の既存 `ll_*` block との置換比較として別仮説なので、`exp183_selector_confidence_replacement_only_on_exp148` は backlog に戻して扱う。
