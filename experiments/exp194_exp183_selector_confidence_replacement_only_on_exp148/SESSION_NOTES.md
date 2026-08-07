# exp194_exp183_selector_confidence_replacement_only_on_exp148 セッションノート

## 2026-07-04 実装

### 狙い

`KAGGLE_DIRECTION.md` の `exp183_selector_confidence_replacement_only_on_exp148` backlog を実験化する。実験番号は次の未使用番号として `exp194_exp183_selector_confidence_replacement_only_on_exp148` を使う。

exp188 は exp148 anchor に exp183 selector confidence features を add-only して悪化した。今回は別仮説として、exp148 の既存 `learned_likelihood_confidence` block を外し、exp183 selector confidence block へ置換する。

### 実装方針

- 親: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- selector 親: `exp183_cluster_outlier_prior_confidence_addonly_on_exp158_selector`
- route: `ml_model`
- active variant: `exp183_selector_confidence_replacement_only`
- feature groups: `projection_correction`, `u_disagreement`, `exp183_selector_confidence`
- excluded active group: `learned_likelihood_confidence`
- control retraining: なし
- direct exp183 selected TVT replacement / blend / postprocess / hard gate: なし
- inference / submit: 初期実装では対象外

### Kaggle train 実行前確認

- active variants: 1
- LightGBM configs: 3 (`lgb0`, `lgb1`, `lgb2`)
- folds: 5
- planned boosters: 15
- GPU: enabled
- train notebook split: なし
- control / parent retraining: なし

### 再現性メモ

- exp194 の feature merge 自体に新規乱数はない。
- exp072 / exp145 / exp183 は upstream fixed Kaggle output として読む。
- exp145 learned-likelihood cache は coverage / inventory として確認するが、active variant には入れない。
- LightGBM GPU は `gpu_use_dp=true`、`deterministic=true`、`force_col_wise=true`、`n_jobs/num_threads=8`。
- deterministic submission anchor ではない。`submission.csv` は作らない。

### 実装内容

- `exp188` の exp183 selector confidence feature builder を複製し、実験名と output prefix を exp194 に変更した。
- active variant の feature groups から `learned_likelihood_confidence` を外した。
- replacement-only では `ll_*` columns を学習に使わないため、active variant が要求しない限り exp145 generated columns を `full_frame` に attach しないようにしてメモリを下げた。
- train / inference notebook は Jupytext percent `.py` を正として作成した。

### 検証予定

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp194_exp183_selector_confidence_replacement_only_on_exp148/exp194_exp183_selector_confidence_replacement_only_on_exp148_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp194_exp183_selector_confidence_replacement_only_on_exp148/exp194_exp183_selector_confidence_replacement_only_on_exp148_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp194_exp183_selector_confidence_replacement_only_on_exp148/exp194_exp183_selector_confidence_replacement_only_on_exp148_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp194_exp183_selector_confidence_replacement_only_on_exp148/exp194_exp183_selector_confidence_replacement_only_on_exp148_inference.py
.venv/bin/python -m py_compile experiments/exp194_exp183_selector_confidence_replacement_only_on_exp148/exp183_selector_confidence_replacement_only_on_exp148.py experiments/exp194_exp183_selector_confidence_replacement_only_on_exp148/exp194_exp183_selector_confidence_replacement_only_on_exp148_train.py experiments/exp194_exp183_selector_confidence_replacement_only_on_exp148/exp194_exp183_selector_confidence_replacement_only_on_exp148_inference.py experiments/exp194_exp183_selector_confidence_replacement_only_on_exp148/settings.py
.venv/bin/ruff check experiments/exp194_exp183_selector_confidence_replacement_only_on_exp148/exp183_selector_confidence_replacement_only_on_exp148.py experiments/exp194_exp183_selector_confidence_replacement_only_on_exp148/exp194_exp183_selector_confidence_replacement_only_on_exp148_train.py experiments/exp194_exp183_selector_confidence_replacement_only_on_exp148/exp194_exp183_selector_confidence_replacement_only_on_exp148_inference.py --select F821
make validate-exp EXP=exp194_exp183_selector_confidence_replacement_only_on_exp148
```

### 検証結果

- Jupytext train / inference convert: pass
- Jupytext train / inference `--test`: pass
- `py_compile`: pass
- `ruff --select F821`: pass
- `make validate-exp EXP=exp194_exp183_selector_confidence_replacement_only_on_exp148`: pass
- 初回 `validate-exp` は README の `## 仮説` / `## 所見` 不足で失敗したため、README を補完して再実行した。

### Kaggle package prepare

```bash
make prepare-kaggle-notebooks EXP=exp194_exp183_selector_confidence_replacement_only_on_exp148 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp194-exp183-selconf-repl-exp148-train --title 'exp194 exp183 selconf repl exp148 train' --run-on-push --strict"
```

- package: `experiments/exp194_exp183_selector_confidence_replacement_only_on_exp148/kaggle/train`
- kernel id: `kentookumura/exp194-exp183-selconf-repl-exp148-train`
- title: `exp194 exp183 selconf repl exp148 train`
- GPU: true
- internet: false
- run_on_push: true
- kernel sources:
  - `kentookumura/exp072-exp063-full-replay-feature-cache-train`
  - `kentookumura/exp145-train`
  - `kentookumura/exp183-copcf-train`
- bootstrap config confirms active variant `exp183_selector_confidence_replacement_only`; disabled control is the only config path containing `learned_likelihood_confidence`.

### 次のアクション

Kaggle train push が必要なら、上記 package を `make push-kaggle-train EXP=exp194_exp183_selector_confidence_replacement_only_on_exp148` で実行する。push 前にこの notes の booster 数と control 再学習なしを再確認する。

### Kaggle 実行記録

- 2026-07-04 22:32:11 JST: `make push-kaggle-train EXP=exp194_exp183_selector_confidence_replacement_only_on_exp148` を実行し、`kentookumura/exp194-exp183-selconf-repl-exp148-train` version 1 を開始（実行中は継続）。
- 状態確認: `kaggle kernels status kentookumura/exp194-exp183-selconf-repl-exp148-train` は `KernelWorkerStatus.RUNNING`。
- logs: `kaggle kernels logs` は初回ネットワーク取得で空出力だったため、`KernelWorkerStatus` の結果で実行継続を確認。
- 次アクション: 実行完了後、`kaggle kernels logs` で Fold / CV 結果を回収し、必要であれば `kaggle kernels output` で artifacts を取得。
- 2026-07-04 22:32:11 JST: `kaggle kernels status kentookumura/exp194-exp183-selconf-repl-exp148-train` を再確認。状態は `KernelWorkerStatus.RUNNING` のまま。

## 2026-07-05 Kaggle train v1 完了

- 2026-07-05 09:20:45 JST: `kaggle kernels status kentookumura/exp194-exp183-selconf-repl-exp148-train` は `KernelWorkerStatus.COMPLETE`。
- `kaggle kernels logs kentookumura/exp194-exp183-selconf-repl-exp148-train` で CV と生成物保存を確認。
- `kaggle kernels output kentookumura/exp194-exp183-selconf-repl-exp148-train -p experiments/exp194_exp183_selector_confidence_replacement_only_on_exp148/kaggle/output/train_v1` で output を取得。
- output: `experiments/exp194_exp183_selector_confidence_replacement_only_on_exp148/kaggle/output/train_v1`
- rows / wells / features: 3,783,989 / 773 / 272
- feature join coverage: pass。dropped rows 0、dropped wells 0。
- elapsed: 15,226.192 sec

| model | pooled RMSE |
| --- | ---: |
| lgb0 | 9.489350463 |
| lgb1 | 9.338468847 |
| lgb2 | 9.306169088 |
| lgb_mean | 9.329893102 |

比較:

- exp148 `lgb_mean` 8.501281182 から +0.828611921 悪化。
- exp188 add-only `lgb_mean` 8.539573790 から +0.790319312 悪化。

判断:

- `learned_likelihood_confidence` block を exp183 selector confidence block で置換する仮説は negative。
- current-test exp183 selector feature generation、saved-booster inference port、submit には進めない。
- `exp183_selector_confidence_replacement_only_on_exp148` backlog は完了/不採用として閉じる。
