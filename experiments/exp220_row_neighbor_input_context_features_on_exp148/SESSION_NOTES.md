# exp220_row_neighbor_input_context_features_on_exp148 セッションノート

## 2026-07-08 実装

### 狙い

`backlog/KAGGLE_DIRECTION.md` の `row_neighbor_input_context_features_on_exp148` backlog を exp220 として実装する。

### 実装方針

- 親: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- route: `ml_model`
- active variant: `row_neighbor_input_context_addonly`
- control retraining: なし。exp148 / exp193 / exp198 の保存済み CV・Public LB を historical baseline として参照する。
- runtime: CPU
- train split: `train_lgb0`, `train_lgb1`, `train_lgb2`

### Feature

`row_neighbor_input_context_features_on_exp148.py` で、exp148 の full-row feature frame に対して同一 well 内の MD 順 context を追加する。

- feature prefix: `rnic_`
- lag/lead periods: 1 / 3 / 5
- rolling window: 5
- source columns: `gr`, `dzdmd`, `md_since`, `beam_mean_d`, `likpf_mean_d`, `ll_candidate_tvt_std`, `ll_learned_prob_entropy`, `uproj_source_u_std`
- max feature count: 60

leakage 防止として、`TVT_input`、OOF prediction、前行 model prediction、valid/test true TVT、oracle best、true-error rank、evaluation label は feature source に使わない。

### Kaggle train 実行前確認

- active variants: 1
- active modes: 1 (`cpu_deterministic_threads8`)
- LightGBM configs: 3 (`lgb0`, `lgb1`, `lgb2`)
- folds: 5
- planned boosters: 15 total
- per split notebook: 5 boosters
- train notebook split: あり (`train_lgb0`, `train_lgb1`, `train_lgb2`)
- control / parent retraining: なし
- GPU: disabled

### 再現性メモ

- row-neighbor context feature generation に乱数は使わない。
- lead / centered rolling は hidden inference でも同一 well の評価区間全体が見える前提で使う。
- upstream exp072 / exp145 cache は固定 artifact として読む。
- LightGBM は CPU deterministic flags、固定 `n_jobs/num_threads=8`。
- deterministic submission anchor ではない。初期実装では `submission.csv` を作らない。

### 検証予定

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp220_row_neighbor_input_context_features_on_exp148/exp220_row_neighbor_input_context_features_on_exp148_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp220_row_neighbor_input_context_features_on_exp148/exp220_row_neighbor_input_context_features_on_exp148_train_lgb0.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp220_row_neighbor_input_context_features_on_exp148/exp220_row_neighbor_input_context_features_on_exp148_train_lgb1.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp220_row_neighbor_input_context_features_on_exp148/exp220_row_neighbor_input_context_features_on_exp148_train_lgb2.py
.venv/bin/python -m py_compile experiments/exp220_row_neighbor_input_context_features_on_exp148/row_neighbor_input_context_features_on_exp148.py experiments/exp220_row_neighbor_input_context_features_on_exp148/exp220_row_neighbor_input_context_features_on_exp148_train.py experiments/exp220_row_neighbor_input_context_features_on_exp148/exp220_row_neighbor_input_context_features_on_exp148_train_lgb0.py experiments/exp220_row_neighbor_input_context_features_on_exp148/exp220_row_neighbor_input_context_features_on_exp148_train_lgb1.py experiments/exp220_row_neighbor_input_context_features_on_exp148/exp220_row_neighbor_input_context_features_on_exp148_train_lgb2.py experiments/exp220_row_neighbor_input_context_features_on_exp148/settings.py
.venv/bin/ruff check experiments/exp220_row_neighbor_input_context_features_on_exp148/row_neighbor_input_context_features_on_exp148.py experiments/exp220_row_neighbor_input_context_features_on_exp148/exp220_row_neighbor_input_context_features_on_exp148_train.py experiments/exp220_row_neighbor_input_context_features_on_exp148/exp220_row_neighbor_input_context_features_on_exp148_train_lgb0.py experiments/exp220_row_neighbor_input_context_features_on_exp148/exp220_row_neighbor_input_context_features_on_exp148_train_lgb1.py experiments/exp220_row_neighbor_input_context_features_on_exp148/exp220_row_neighbor_input_context_features_on_exp148_train_lgb2.py experiments/exp220_row_neighbor_input_context_features_on_exp148/settings.py --select F821,F401
uv run python scripts/validate_experiment.py --experiment exp220_row_neighbor_input_context_features_on_exp148
```

### 検証結果

- Jupytext convert: pass (`train`, `inference`, `train_lgb0`, `train_lgb1`, `train_lgb2`)
- Jupytext `--test`: pass (`train`, `inference`, `train_lgb0`, `train_lgb1`, `train_lgb2`)
- `py_compile`: pass
- `ruff --select F821,F401`: pass
- `validate_experiment.py --experiment exp220_row_neighbor_input_context_features_on_exp148`: pass
- synthetic row-neighbor feature builder smoke: pass。5 rows / 2 wells から 56 features を生成し、row count / group count が期待通り。

### Kaggle package prepare

```bash
.venv/bin/python scripts/prepare_kaggle_notebooks.py --experiment exp220_row_neighbor_input_context_features_on_exp148 --notebook train_lgb0 --kernel-id kentookumura/exp220-row-neighbor-input-context-features-on-exp148-train-lgb0 --title 'exp220 row neighbor input context features on exp148 train lgb0' --strict
.venv/bin/python scripts/prepare_kaggle_notebooks.py --experiment exp220_row_neighbor_input_context_features_on_exp148 --notebook train_lgb1 --kernel-id kentookumura/exp220-row-neighbor-input-context-features-on-exp148-train-lgb1 --title 'exp220 row neighbor input context features on exp148 train lgb1' --strict
.venv/bin/python scripts/prepare_kaggle_notebooks.py --experiment exp220_row_neighbor_input_context_features_on_exp148 --notebook train_lgb2 --kernel-id kentookumura/exp220-row-neighbor-input-context-features-on-exp148-train-lgb2 --title 'exp220 row neighbor input context features on exp148 train lgb2' --strict
```

- train_lgb0 package: `experiments/exp220_row_neighbor_input_context_features_on_exp148/kaggle/train_lgb0`
- train_lgb1 package: `experiments/exp220_row_neighbor_input_context_features_on_exp148/kaggle/train_lgb1`
- train_lgb2 package: `experiments/exp220_row_neighbor_input_context_features_on_exp148/kaggle/train_lgb2`
- package `py_compile`: pass for all three split packages.
- metadata check: `enable_gpu=false`; kernel sources are exp072 full replay cache train and exp145 train.

## 2026-07-08 Kaggle CPU split train v1 実行

ユーザー依頼により、exp220 の CPU split train を Kaggle に push して実行開始した。

実行前コスト確認:

- active variants: 1 (`row_neighbor_input_context_addonly`)
- active modes: 1 (`cpu_deterministic_threads8`)
- LightGBM configs: 3 (`lgb0`, `lgb1`, `lgb2`)
- folds: 5
- planned boosters: 15 total
- per split notebook: 5 boosters
- control / parent retraining: なし
- GPU: disabled (`enable_gpu=false`)

初回 prepare は長い canonical slug:

- `kentookumura/exp220-row-neighbor-input-context-features-on-exp148-train-lgb0`
- `kentookumura/exp220-row-neighbor-input-context-features-on-exp148-train-lgb1`
- `kentookumura/exp220-row-neighbor-input-context-features-on-exp148-train-lgb2`

で作成したが、`train_lgb0` push が Kaggle `SaveKernel` 400 で失敗した。exp193 と同じく slug 長が原因の可能性が高いため、同じ exp220 のまま短い id/title に再 prepare した。

再 prepare / push:

```bash
.venv/bin/python scripts/prepare_kaggle_notebooks.py --experiment exp220_row_neighbor_input_context_features_on_exp148 --notebook train_lgb0 --kernel-id kentookumura/exp220-row-neighbor-exp148-lgb0 --title 'exp220 row neighbor exp148 lgb0' --run-on-push --strict
.venv/bin/python scripts/prepare_kaggle_notebooks.py --experiment exp220_row_neighbor_input_context_features_on_exp148 --notebook train_lgb1 --kernel-id kentookumura/exp220-row-neighbor-exp148-lgb1 --title 'exp220 row neighbor exp148 lgb1' --run-on-push --strict
.venv/bin/python scripts/prepare_kaggle_notebooks.py --experiment exp220_row_neighbor_input_context_features_on_exp148 --notebook train_lgb2 --kernel-id kentookumura/exp220-row-neighbor-exp148-lgb2 --title 'exp220 row neighbor exp148 lgb2' --run-on-push --strict
kaggle kernels push -p experiments/exp220_row_neighbor_input_context_features_on_exp148/kaggle/train_lgb0
kaggle kernels push -p experiments/exp220_row_neighbor_input_context_features_on_exp148/kaggle/train_lgb1
kaggle kernels push -p experiments/exp220_row_neighbor_input_context_features_on_exp148/kaggle/train_lgb2
```

Kaggle kernels:

- `kentookumura/exp220-row-neighbor-exp148-lgb0` v1: push success, status `KernelWorkerStatus.RUNNING`
- `kentookumura/exp220-row-neighbor-exp148-lgb1` v1: push success, status `KernelWorkerStatus.RUNNING`
- `kentookumura/exp220-row-neighbor-exp148-lgb2` v1: push success, status `KernelWorkerStatus.RUNNING`

存在確認:

```bash
kaggle kernels pull kentookumura/exp220-row-neighbor-exp148-lgb0 -p /tmp/kaggle-pull/exp220-row-neighbor-exp148-lgb0 -m
kaggle kernels pull kentookumura/exp220-row-neighbor-exp148-lgb1 -p /tmp/kaggle-pull/exp220-row-neighbor-exp148-lgb1 -m
kaggle kernels pull kentookumura/exp220-row-neighbor-exp148-lgb2 -p /tmp/kaggle-pull/exp220-row-neighbor-exp148-lgb2 -m
```

すべて成功。CLI version warning は `installed: 2.2.0`, latest `2.2.2`。

## 2026-07-08 Kaggle CPU split train v1 完了確認

ユーザー報告後に Kaggle status を確認し、3 split すべて `COMPLETE`。

- `kentookumura/exp220-row-neighbor-exp148-lgb0` v1: COMPLETE
- `kentookumura/exp220-row-neighbor-exp148-lgb1` v1: COMPLETE
- `kentookumura/exp220-row-neighbor-exp148-lgb2` v1: COMPLETE

`kaggle kernels output` で生成物を取得した。

- `kaggle/output/train_lgb0_v1`
- `kaggle/output/train_lgb1_v1`
- `kaggle/output/train_lgb2_v1`

各 split は 3,783,989 rows / 773 wells / 350 features。feature join coverage はすべて pass、dropped rows/wells は 0。各 split は 1 LightGBM config x 5 folds = 5 boosters、合計 15 boosters。control / parent retraining はなし、GPU disabled。

### split CV

| model | pooled RMSE TVT | pooled RMSE target | prediction SHA |
| --- | ---: | ---: | --- |
| `lgb0` | 8.577046760 | 8.577046998 | `f96b7932dda27fc60542aadbcee40906e89209ac2e6dfede2264e2410091f29f` |
| `lgb1` | 8.532166021 | 8.532166024 | `ea39574d56476ae2559b19f0178ffe44349a91de9c55f42a7566d92b87dcca8e` |
| `lgb2` | 8.539115349 | 8.539115667 | `aced87ccb87a001cbd2ee824ae2dab0b0b0e4b5f1345d724a68b21f485d87912` |

### cross-split `lgb_mean` 集計

各 split の `pred_target` を streaming で平均し、full-run 相当の `lgb_mean` を検算した。一括読みはメモリで落ちたため、150,000 row chunks で行順検証、RMSE、SHA、bucket、by-well、feature importance を集計した。巨大な結合済み prediction CSV は保存していない。

- row order check: `lgb0/lgb1/lgb2` all pass
- `lgb_mean` pooled RMSE TVT: 8.496282588
- `lgb_mean` pooled RMSE target: 8.496282412
- prediction SHA: `5be47377f9ffcf0ddd3023c3e93e57764316588a9a6c94693ea5ee6666bc4e21`
- max `last_known_tvt + pred_target_mean - pred_tvt_mean` abs error: 0

比較:

- vs exp148 GPU `lgb_mean` 8.501281182: -0.004998594
- vs exp148 CPU runtime `lgb_mean` 8.528698114: -0.032415526
- vs exp193 `lgb_mean` 8.456665439: +0.039617150
- vs exp198 `lgb_mean` 8.457923653: +0.038358935
- vs exp218 `lgb_mean` 8.475793752: +0.020488837

Distance bucket:

| bucket | rows | RMSE TVT |
| --- | ---: | ---: |
| `000_050` | 38,650 | 0.986487 |
| `050_100` | 38,650 | 1.334240 |
| `100_250` | 115,950 | 2.097604 |
| `250_500` | 193,157 | 3.309812 |
| `500_1000` | 385,911 | 4.826862 |
| `1000_plus` | 3,011,671 | 9.316960 |

Top `rnic_` feature importance:

| feature | mean importance | records |
| --- | ---: | ---: |
| `rnic_likpf_mean_d_lead5_minus_cur` | 340.533333 | 15 |
| `rnic_likpf_mean_d_cur_minus_lag5` | 310.466667 | 15 |
| `rnic_uproj_source_u_std_lead5_minus_cur` | 188.600000 | 15 |
| `rnic_uproj_source_u_std_cur_minus_lag5` | 175.533333 | 15 |
| `rnic_likpf_mean_d_roll5_std` | 152.466667 | 15 |

集計成果物:

- `artifacts/exp220_row_neighbor_input_context_features_on_exp148_split_lgb_mean_summary.json`
- `artifacts/exp220_row_neighbor_input_context_features_on_exp148_split_lgb_mean_metrics.csv`
- `artifacts/exp220_row_neighbor_input_context_features_on_exp148_split_lgb_mean_bucket_metrics.csv`
- `artifacts/exp220_row_neighbor_input_context_features_on_exp148_split_lgb_mean_by_well.csv`
- `artifacts/exp220_row_neighbor_input_context_features_on_exp148_split_lgb_mean_feature_importance_mean.csv`

### 判断

row-neighbor input context は exp148 GPU historical OOF をわずかに改善したが、exp193 / exp198 / 現行 ML route submitted anchor の exp218 には届かない。train-side completed / no submit とし、inference 化や competition submission は行わない。
