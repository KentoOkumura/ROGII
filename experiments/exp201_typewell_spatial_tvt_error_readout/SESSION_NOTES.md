# exp201_typewell_spatial_tvt_error_readout セッションノート

## 目的

exp148 OOF の残差を、共通 typewell group、XY 近傍、true TVT 急変、well 全体 offset の観点で診断する。

## 現在の状態

- Route: ml_model
- 状態: Kaggle 実行完了
- CV: まだなし
- LB: まだなし

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

### 実行済み

```bash
make new-steering EXP=exp201_typewell_spatial_tvt_error_readout
make new-exp EXP=exp201_typewell_spatial_tvt_error_readout
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp201_typewell_spatial_tvt_error_readout/exp201_typewell_spatial_tvt_error_readout_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp201_typewell_spatial_tvt_error_readout/exp201_typewell_spatial_tvt_error_readout_train.py
.venv/bin/python -m py_compile experiments/exp201_typewell_spatial_tvt_error_readout/typewell_spatial_tvt_error_readout.py experiments/exp201_typewell_spatial_tvt_error_readout/exp201_typewell_spatial_tvt_error_readout_train.py
.venv/bin/ruff check experiments/exp201_typewell_spatial_tvt_error_readout/typewell_spatial_tvt_error_readout.py experiments/exp201_typewell_spatial_tvt_error_readout/exp201_typewell_spatial_tvt_error_readout_train.py --select F821
make validate-exp EXP=exp201_typewell_spatial_tvt_error_readout
make prepare-kaggle-notebooks EXP=exp201_typewell_spatial_tvt_error_readout EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp201-typewell-spatial-tvt-error-readout-train --title 'exp201 typewell spatial tvt error readout train' --run-on-push --strict"
```

Kaggle push:

```bash
make push-kaggle-train EXP=exp201_typewell_spatial_tvt_error_readout
```

結果: Kaggle API は到達したが `Kernel push error: Maximum batch CPU session count of 5 reached.` で未投入。2026-07-05 時点で次の5本が CPU batch 枠を使っていることを確認した。

- `kentookumura/exp083-v12-ml-oof-known-tvt-probe`: RUNNING
- `kentookumura/exp184-heatmap-selcompact-exp148-train-lgb0`: RUNNING
- `kentookumura/exp184-heatmap-selcompact-exp148-train-lgb1`: RUNNING
- `kentookumura/exp184-heatmap-selcompact-exp148-train-lgb2`: RUNNING
- `kentookumura/exp200-pf-step-delta-prior-train`: RUNNING

同じ slug の再 push は `Kernel push error: Notebook not found` になったため、exp161 と同じ回避として train slug を変更した。

```bash
make prepare-kaggle-notebooks EXP=exp201_typewell_spatial_tvt_error_readout EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp201-typewell-spatial-tvt-error-readout-train-a --title 'exp201 typewell spatial tvt error readout train a' --run-on-push --strict"
make push-kaggle-train EXP=exp201_typewell_spatial_tvt_error_readout
kaggle kernels status kentookumura/exp201-typewell-spatial-tvt-error-readout-train-a
```

結果: `kentookumura/exp201-typewell-spatial-tvt-error-readout-train-a` v1 の push に成功。URL は https://www.kaggle.com/code/kentookumura/exp201-typewell-spatial-tvt-error-readout-train-a 。起動直後の status は `KernelWorkerStatus.RUNNING`。

v1 は Kaggle papermill 起動時に `ValueError: No kernel name found in notebook and no override provided.` で失敗。Jupytext source に kernelspec header を追加し、notebook を再生成して再投入した。

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp201_typewell_spatial_tvt_error_readout/exp201_typewell_spatial_tvt_error_readout_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp201_typewell_spatial_tvt_error_readout/exp201_typewell_spatial_tvt_error_readout_train.py
.venv/bin/python -m py_compile experiments/exp201_typewell_spatial_tvt_error_readout/typewell_spatial_tvt_error_readout.py experiments/exp201_typewell_spatial_tvt_error_readout/exp201_typewell_spatial_tvt_error_readout_train.py
.venv/bin/ruff check experiments/exp201_typewell_spatial_tvt_error_readout/typewell_spatial_tvt_error_readout.py experiments/exp201_typewell_spatial_tvt_error_readout/exp201_typewell_spatial_tvt_error_readout_train.py --select F821
make prepare-kaggle-notebooks EXP=exp201_typewell_spatial_tvt_error_readout EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp201-typewell-spatial-tvt-error-readout-train-a --title 'exp201 typewell spatial tvt error readout train a' --run-on-push --strict"
make push-kaggle-train EXP=exp201_typewell_spatial_tvt_error_readout
kaggle kernels status kentookumura/exp201-typewell-spatial-tvt-error-readout-train-a
kaggle kernels logs kentookumura/exp201-typewell-spatial-tvt-error-readout-train-a
```

結果: v2 push 成功。起動直後の status は `KernelWorkerStatus.RUNNING`。直後ログは warning のみで notebook 側の traceback はなし。

## 変更点

- 新規学習なしの OOF readout 実験として作成。
- 入力は exp148 train v1 `lgb_mean` OOF prediction、raw train、native_overlap_1 typewell summary。
- ローカル実行は 3,783,989 row OOF のメモリで `exit 137` になったため、streaming 集計化し、Kaggle CPU notebook 実行に切り替えた。
- typewell summary は `inputs/native_overlap_1_well_position_typewell_summary.csv` として notebook bootstrap に同梱する。

## 再現性メモ

- seed policy: 乱数なし。
- stochastic components: なし。
- CPU/GPU runtime: CPU 集計のみ。新規 GPU 学習なし。
- Kaggle kernel id / version: 入力 prediction は exp148 train v1。
- input / feature schema SHA: readout 実行後に `readout_summary.json` に記録。
- feature content SHA: readout 実行後に記録。
- model manifest / model SHA: 新規 model なし。
- prediction SHA: 新規 submission prediction なし。入力 OOF prediction の decompressed SHA を記録する。
- submission SHA: なし。
- rerun check: deterministic 集計なので同一入力なら同一出力。

## 次のアクション

1. `result.md` の次候補に沿って、直接補正ではなく add-only confidence / uncertainty feature として実験化するか判断する。

## 完了確認

2026-07-05 に Kaggle train v2 が COMPLETE。

```bash
kaggle kernels status kentookumura/exp201-typewell-spatial-tvt-error-readout-train-a
kaggle kernels logs kentookumura/exp201-typewell-spatial-tvt-error-readout-train-a
kaggle kernels output kentookumura/exp201-typewell-spatial-tvt-error-readout-train-a -p experiments/exp201_typewell_spatial_tvt_error_readout/kaggle/output/train_v2
```

- kernel: `kentookumura/exp201-typewell-spatial-tvt-error-readout-train-a`
- version: 2
- status: `KernelWorkerStatus.COMPLETE`
- output: `kaggle/output/train_v2/`
- rows: 3,783,989
- wells: 773
- typewell groups: 54
- exp148 OOF RMSE: 8.50128118189582
- offset wells: 66 / 773
- generated SVG: `typewell_group_bias_rmse_top.svg`, `xy_bias_map.svg`, `offset_residual_profiles.svg`, `sharp_step_true_vs_pred.svg`

主要な解釈は `result.md` に記録した。結論は、same-typewell / XY 近傍の residual shape が一貫して似る証拠は弱い。一方で typewell group 単位の high-RMSE / offset hotspot と whole-well offset は強い。高 error top30 wells の 27 wells が offset flag で、well RMSE と abs_bias の相関は 0.948。次に使うなら直接 postprocess ではなく、offset/outlier/uncertainty feature として扱う。

## 追加確認: offset 方向が揃う typewell group

ユーザー指摘を受け、test-only 補正候補として offset wells の方向が揃っている typewell group を追加集計した。

```bash
.venv/bin/python - <<'PY'
# artifacts/well_error_profile_summary.csv と offset_wells.csv から
# typewell_offset_direction_summary.csv を作成
PY
```

出力: `artifacts/typewell_offset_direction_summary.csv`

- offset wells が 2 本以上あり、offset 方向が全て一致する group は 4 つ。
- `cluster_0004`: 38 wells / offset 4 / all underpredict / offset bias mean -15.288。ただし group 全体は positive 19 / negative 19。
- `cluster_0029`: 10 wells / offset 2 / all underpredict / offset bias mean -18.968。group 全体は positive 6 / negative 4 で、候補中では最も test-only prior に近い。
- `cluster_0009`: 28 wells / offset 2 / all underpredict / group 全体は positive 13 / negative 15。
- `cluster_0013`: 22 wells / offset 2 / all underpredict / group 全体は positive 9 / negative 13。
- group 全体の bias 方向が 75% 以上揃う group は `cluster_0032`, `cluster_0036`, `cluster_0022`, `cluster_0006`, `cluster_0030` だが、offset wells は 0-1 本で強い offset 補正対象ではない。

test-only に使うなら、固定補正の主候補は `cluster_0029` だけ。`cluster_0004/0009/0013` は offset wells 方向は揃うが、group 全体の方向が弱いため、補正するなら小さい alpha / clip を推奨。
