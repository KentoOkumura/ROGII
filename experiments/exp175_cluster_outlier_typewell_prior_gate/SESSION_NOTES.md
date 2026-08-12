# exp175_cluster_outlier_typewell_prior_gate セッションノート

## 目的

exp109 / exp114 の typewell / spatial neighbor prior を、exp065 native typewell cluster の空間外れ well にだけ弱く適用する gate を train-side OOF で診断する。

## 現在の状態

- Route: `ml_model`
- 状態: 実装済み、Kaggle train 未実行
- CV: 未計測
- LB: なし

## コマンドログ

### 2026-07-03 実装

```bash
make new-steering EXP=exp175_cluster_outlier_typewell_prior_gate
make new-exp EXP=exp175_cluster_outlier_typewell_prior_gate SOURCE=experiments/exp118_spatial_neighbor_prior_confidence_gate_on_exp092
.venv/bin/python -m py_compile experiments/exp175_cluster_outlier_typewell_prior_gate/cluster_outlier_typewell_prior_gate.py experiments/exp175_cluster_outlier_typewell_prior_gate/exp175_cluster_outlier_typewell_prior_gate_train.py experiments/exp175_cluster_outlier_typewell_prior_gate/exp175_cluster_outlier_typewell_prior_gate_inference.py
.venv/bin/ruff check experiments/exp175_cluster_outlier_typewell_prior_gate/cluster_outlier_typewell_prior_gate.py experiments/exp175_cluster_outlier_typewell_prior_gate/exp175_cluster_outlier_typewell_prior_gate_train.py experiments/exp175_cluster_outlier_typewell_prior_gate/exp175_cluster_outlier_typewell_prior_gate_inference.py --select F821,F401,E501
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp175_cluster_outlier_typewell_prior_gate/exp175_cluster_outlier_typewell_prior_gate_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp175_cluster_outlier_typewell_prior_gate/exp175_cluster_outlier_typewell_prior_gate_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp175_cluster_outlier_typewell_prior_gate/exp175_cluster_outlier_typewell_prior_gate_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp175_cluster_outlier_typewell_prior_gate/exp175_cluster_outlier_typewell_prior_gate_inference.py
make validate-exp EXP=exp175_cluster_outlier_typewell_prior_gate
make prepare-kaggle-notebooks EXP=exp175_cluster_outlier_typewell_prior_gate EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp175-cluster-outlier-typewell-prior-gate-train --title 'exp175 cluster outlier typewell prior gate train' --run-on-push --strict"
make prepare-kaggle-notebooks EXP=exp175_cluster_outlier_typewell_prior_gate EXTRA_ARGS="--notebook inference --kernel-id kentookumura/exp175-cluster-outlier-typewell-prior-gate-infer --title 'exp175 cluster outlier typewell prior gate infer' --run-on-push --strict"
```

- `docs/legacy/steering/20260703-exp175-cluster-outlier-typewell-prior-gate/` を作成。
- exp118 から実験ディレクトリを派生し、cluster-outlier gated prior audit に差し替えた。
- 入力は exp065 cluster assignment、exp109 OOF typewell prior、exp114 OOF spatial prior / geometry、exp148 / exp092 OOF prediction。
- inference notebook は no-submission guard。
- `py_compile` と ruff `F821,F401,E501` は成功。
- Jupytext train / inference 変換と `--test` は成功。
- `make validate-exp` は strict PASS。
- Kaggle train / inference package 生成は成功。
- train kernel metadata:
  - id: `kentookumura/exp175-cluster-outlier-typewell-prior-gate-train`
  - GPU: false
  - internet: false
  - kernel sources: exp109 / exp114 / exp065 / exp115 / exp148 / exp092

## 実行予定

- active prediction sources: `exp148_lgb_mean`, `exp092_lgb1` のうち mount できるもの。
- prior variants: 4
- cluster gates: 8
- prior quality gates: 5
- alpha: 3
- clip: 4
- posthoc policy 上限: prediction source ごとに 4 * 8 * 5 * 3 * 4 = 1920 policies。
- LightGBM config 数: 0
- fold 数: 0
- booster 数: 0
- control / parent 再学習: なし

### 2026-07-03 Kaggle train 実行

push 前確認:

- 実行対象: train-side deterministic posthoc grid audit
- active variant 数: 0 model variants / 1 audit grid
- LightGBM config 数: 0
- fold 数: 0
- booster 数: 0
- GPU: false
- internet: false
- control / parent 再学習: なし
- 既存 parent OOF / prior / cluster 生成物を参照するだけで、新規 model training はしない。

```bash
make validate-exp EXP=exp175_cluster_outlier_typewell_prior_gate
make push-kaggle-train EXP=exp175_cluster_outlier_typewell_prior_gate
kaggle kernels pull kentookumura/exp175-cluster-outlier-typewell-prior-gate-train -p /tmp/kaggle-pull/exp175-cluster-outlier-typewell-prior-gate-train -m
kaggle kernels logs kentookumura/exp175-cluster-outlier-typewell-prior-gate-train
kaggle kernels status kentookumura/exp175-cluster-outlier-typewell-prior-gate-train
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp175-cluster-outlier-typewell-prior-gate-train
```

- `make validate-exp` は strict PASS。
- Kaggle train kernel version 1 を push 成功。
- URL: `https://www.kaggle.com/code/kentookumura/exp175-cluster-outlier-typewell-prior-gate-train`
- `kaggle kernels pull -m` で同 kernel id の存在確認は成功。
- `kaggle kernels status` は `KernelWorkerStatus.RUNNING`。
- 実行中の `kaggle kernels logs` / `logs -f` は空出力。Kaggle CLI の既知挙動として扱い、失敗判定や再 push はしない。
- ユーザー指示によりローカル監視を停止。完了連絡後に同じ kernel id で logs / 必要時 output を確認する。

### 2026-07-03 Kaggle train v1 失敗と修正

ユーザー連絡後に同じ kernel id の logs を確認。

```bash
kaggle kernels status kentookumura/exp175-cluster-outlier-typewell-prior-gate-train
kaggle kernels logs kentookumura/exp175-cluster-outlier-typewell-prior-gate-train
```

- status: `KernelWorkerStatus.ERROR`
- 失敗セル: `In [5]`, `run_cluster_outlier_typewell_prior_gate(...)`
- 失敗箇所: `detailed_metrics()` の distance bucket 集計
- error: `AttributeError: 'Series' object has no attribute 'categories'`
- 原因: `pd.cut(pd.Series(...))` は `Categorical` ではなく categorical dtype の `Series` を返すため、`buckets.categories` が存在しない。

修正:

- `distance_bucket()` の戻り型を `pd.Series | pd.Categorical` に変更。
- `distance_bucket_categories()` を追加し、Series の場合は `buckets.cat.categories` を使う。
- `subgroup_masks()` と `detailed_metrics()` の bucket loop を同 helper 経由に修正。

検証:

```bash
.venv/bin/python -m py_compile experiments/exp175_cluster_outlier_typewell_prior_gate/cluster_outlier_typewell_prior_gate.py
.venv/bin/ruff check experiments/exp175_cluster_outlier_typewell_prior_gate/cluster_outlier_typewell_prior_gate.py --select F821,F401,E501
make validate-exp EXP=exp175_cluster_outlier_typewell_prior_gate
make prepare-kaggle-notebooks EXP=exp175_cluster_outlier_typewell_prior_gate EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp175-cluster-outlier-typewell-prior-gate-train --title 'exp175 cluster outlier typewell prior gate train' --run-on-push --strict"
```

- `py_compile`: PASS
- ruff `F821,F401,E501`: PASS
- `make validate-exp`: strict PASS
- Kaggle train package 再生成: PASS
- 次: 同じ kernel id に v2 として再 push。

再 push:

```bash
make push-kaggle-train EXP=exp175_cluster_outlier_typewell_prior_gate
kaggle kernels status kentookumura/exp175-cluster-outlier-typewell-prior-gate-train
kaggle kernels pull kentookumura/exp175-cluster-outlier-typewell-prior-gate-train -p /tmp/kaggle-pull/exp175-cluster-outlier-typewell-prior-gate-train-v2 -m
```

- Kaggle train kernel version 2 を push 成功。
- URL: `https://www.kaggle.com/code/kentookumura/exp175-cluster-outlier-typewell-prior-gate-train`
- status 確認時点: `KernelWorkerStatus.RUNNING`
- `kaggle kernels pull -m` で同 kernel id の存在確認は成功。
- 以後は長時間監視せず、完了連絡後に logs / 必要時 output を確認する。

### 2026-07-03 Kaggle train v2 完了

ユーザー連絡後に同じ kernel id の status / logs を確認。logs には生成物パスだけで数値テーブルが出なかったため、CV 記録に必要な `summary.json` / `metrics.json` / small CSV を確認する目的で output を `/tmp` に取得した。

```bash
kaggle kernels status kentookumura/exp175-cluster-outlier-typewell-prior-gate-train
kaggle kernels logs kentookumura/exp175-cluster-outlier-typewell-prior-gate-train
kaggle kernels output kentookumura/exp175-cluster-outlier-typewell-prior-gate-train -p /tmp/kaggle-output/exp175_cluster_outlier_typewell_prior_gate/train_v2
```

- status: `KernelWorkerStatus.COMPLETE`
- kernel version: 2
- output: `/tmp/kaggle-output/exp175_cluster_outlier_typewell_prior_gate/train_v2`
- rows: 3,783,989 per source / 773 wells
- loaded sources: exp148 `lgb_mean`, exp092 `lgb1`, exp109 prior, exp114 prior/geometry, exp065 cluster, exp115 roles
- decision: `cluster_outlier_prior_gate_not_supported`

結果:

- exp148 `lgb_mean` baseline: RMSE 8.501281181895820 / MAE 5.335650953431683 / within10 0.856332034791856
- exp092 `lgb1` baseline: RMSE 9.322479895503928 / MAE 5.980980324939915 / within10 0.822047051405276
- best policy: exp148 `lgb_mean` baseline
- best non-baseline: `exp148_lgb_mean__lgb_mean__typewell_native_overlap_0p999__own_z_gt2p0__std_le20__a0p05__c5`
- best non-baseline RMSE: 8.501592821010115, delta +0.0003116391142956587
- best non-baseline gate: 271,090 rows / 62 wells / gate rate 0.07164132876707623
- best non-baseline by-well: 29 improved / 33 worse / 711 same, max regression +0.18117324601574047 RMSE, max improvement -0.21004396329086283 RMSE
- near `000_050` bucket improved by -0.004189 RMSE, but `250_500`, `500_1000`, `1000_plus`, exp115 hidden-like subgroups, and global RMSE worsened.

結論:

- cluster-outlier gate は ML output への prior correction の大きな worst-well regression を抑えたが、global RMSE / MAE を改善しない。
- inference port / submit はしない。
- この実験は exp109/114 と同じ PF/Beam/likPF 候補への correction を cluster-outlier well に限定したものではなく、補正対象を exp148/exp092 ML output に変えている。したがって exp109/114 の直接 follow-up としては軸がずれていた。
- ML output への `cluster_outlier_typewell_prior_gate` は完了/不採用。PF/Beam/likPF 候補への cluster-outlier gated prior correction は未検証として backlog に戻す。

## 次のコマンド

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp175_cluster_outlier_typewell_prior_gate/exp175_cluster_outlier_typewell_prior_gate_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp175_cluster_outlier_typewell_prior_gate/exp175_cluster_outlier_typewell_prior_gate_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp175_cluster_outlier_typewell_prior_gate/exp175_cluster_outlier_typewell_prior_gate_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp175_cluster_outlier_typewell_prior_gate/exp175_cluster_outlier_typewell_prior_gate_inference.py
make validate-exp EXP=exp175_cluster_outlier_typewell_prior_gate
```

上記は実行済み。次は Kaggle train push。

## 再現性メモ

- seed policy: deterministic_posthoc_grid_no_model_rng
- stochastic components: upstream exp092 / exp148 / exp109 / exp114 の固定生成物のみ
- exp175 内の model training / PF/Beam regeneration: なし
- gzip 生成物は raw SHA と decompressed SHA を summary に記録する。
