# exp181_cluster_outlier_pfbeam_prior_gate セッションノート

## 2026-07-03 初期実装

### 目的

`cluster_outlier_pfbeam_prior_gate` backlog を実装する。exp175 の cluster-outlier gate を再利用するが、補正対象を exp148/exp092 ML output ではなく、exp109/114 と同じ PF/Beam/likPF OOF 候補へ戻す。

### 実行予定

- Route: `pf_beam`
- base candidates: `likpf_mean`, `pf_ancc`, `beam_mean`
- prior variants: typewell native overlap 1 / 0.999, spatial xy+trajectory k8, spatial xy-only k8
- cluster gates: own z, nearest other closer, nearby majority diff, AND/OR variants
- correction grid: alpha `0.05/0.10/0.20`, clip `5/10/20/40ft`
- reference policies: exp109 global best, exp114 global best
- LightGBM configs: 0
- folds: 0
- boosters: 0
- control / parent retraining: なし

### 再現性メモ

- exp181 自体に乱数、学習、PF/Beam 再生成はない。
- upstream の PF/Beam/likPF、typewell prior、spatial prior、cluster assignment は固定 Kaggle output として扱う。
- gzip 出力は decompressed content SHA を主証拠として summary に記録する。
- deterministic submission anchor ではない。`submission.csv` は作らない。

### 実装メモ

- `docs/legacy/steering/20260703-exp181-cluster-outlier-pfbeam-prior-gate/` を作成。
- `experiments/exp181_cluster_outlier_pfbeam_prior_gate/` を exp175 からコピーして作成。
- `cluster_outlier_pfbeam_prior_gate.py` を実装し、exp109 OOF 内の `likpf_mean` / `pf_ancc` / `beam_mean` を base candidate として評価するよう変更。
- `config.yaml`、train/inference percent notebook source、README、result、metrics を exp181 用に更新。

### 検証

```bash
.venv/bin/python -m py_compile experiments/exp181_cluster_outlier_pfbeam_prior_gate/cluster_outlier_pfbeam_prior_gate.py experiments/exp181_cluster_outlier_pfbeam_prior_gate/exp181_cluster_outlier_pfbeam_prior_gate_train.py experiments/exp181_cluster_outlier_pfbeam_prior_gate/exp181_cluster_outlier_pfbeam_prior_gate_inference.py
.venv/bin/ruff check experiments/exp181_cluster_outlier_pfbeam_prior_gate/cluster_outlier_pfbeam_prior_gate.py experiments/exp181_cluster_outlier_pfbeam_prior_gate/exp181_cluster_outlier_pfbeam_prior_gate_train.py experiments/exp181_cluster_outlier_pfbeam_prior_gate/exp181_cluster_outlier_pfbeam_prior_gate_inference.py --select F821,F401,E501
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp181_cluster_outlier_pfbeam_prior_gate/exp181_cluster_outlier_pfbeam_prior_gate_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp181_cluster_outlier_pfbeam_prior_gate/exp181_cluster_outlier_pfbeam_prior_gate_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp181_cluster_outlier_pfbeam_prior_gate/exp181_cluster_outlier_pfbeam_prior_gate_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp181_cluster_outlier_pfbeam_prior_gate/exp181_cluster_outlier_pfbeam_prior_gate_inference.py
make validate-exp EXP=exp181_cluster_outlier_pfbeam_prior_gate
make prepare-kaggle-notebooks EXP=exp181_cluster_outlier_pfbeam_prior_gate EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp181-cluster-outlier-pfbeam-prior-gate-train --title 'exp181 cluster outlier pfbeam prior gate train' --run-on-push --strict"
make prepare-kaggle-notebooks EXP=exp181_cluster_outlier_pfbeam_prior_gate EXTRA_ARGS="--notebook inference --kernel-id kentookumura/exp181-cluster-outlier-pfbeam-prior-gate-infer --title 'exp181 cluster outlier pfbeam prior gate infer' --run-on-push --strict"
make update-summary
```

- `py_compile`: pass
- `ruff --select F821,F401,E501`: pass
- Jupytext train / inference convert and `--test`: pass
- `make validate-exp`: pass
- Kaggle package metadata:
  - train id: `kentookumura/exp181-cluster-outlier-pfbeam-prior-gate-train`
  - inference id: `kentookumura/exp181-cluster-outlier-pfbeam-prior-gate-infer`
  - GPU: disabled
  - internet: disabled
  - kernel sources: exp109 / exp114 / exp065 / exp115

### 次アクション

Kaggle train を push して、`gate_metrics`、`by_well_delta`、bucket / subgroup / path continuity を確認する。train result が出るまでは backlog から削除しない。

## 2026-07-03 Kaggle train v1

### 実行

```bash
make push-kaggle-train EXP=exp181_cluster_outlier_pfbeam_prior_gate
kaggle kernels pull kentookumura/exp181-cluster-outlier-pfbeam-prior-gate-train -p /tmp/kaggle-pull/exp181-cluster-outlier-pfbeam-prior-gate-train-v1 -m
kaggle kernels logs kentookumura/exp181-cluster-outlier-pfbeam-prior-gate-train
kaggle kernels status kentookumura/exp181-cluster-outlier-pfbeam-prior-gate-train
```

- push: success
- kernel version: v1
- URL: `https://www.kaggle.com/code/kentookumura/exp181-cluster-outlier-pfbeam-prior-gate-train`
- pull metadata: success, `/tmp/kaggle-pull/exp181-cluster-outlier-pfbeam-prior-gate-train-v1`
- initial logs: empty, expected while running
- initial status: `KernelWorkerStatus.RUNNING`
- `timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp181-cluster-outlier-pfbeam-prior-gate-train` で polling したが、CLI logs は空のまま。ユーザー指示によりローカル監視だけ停止。Kaggle kernel v1 は停止していない。

### 完了確認

ユーザーから Kaggle 実行完了の連絡を受け、同じ kernel id で状態、logs、output を確認した。

```bash
kaggle kernels status kentookumura/exp181-cluster-outlier-pfbeam-prior-gate-train
kaggle kernels logs kentookumura/exp181-cluster-outlier-pfbeam-prior-gate-train
kaggle kernels output kentookumura/exp181-cluster-outlier-pfbeam-prior-gate-train -p /tmp/kaggle-output/exp181_cluster_outlier_pfbeam_prior_gate/train_v1
```

- status: `KernelWorkerStatus.COMPLETE`
- output: `/tmp/kaggle-output/exp181_cluster_outlier_pfbeam_prior_gate/train_v1`
- runtime: `1610.2290246486664` sec
- rows: `11,351,967` total candidate rows (`3,783,989` rows x 3 base candidates)
- wells: `773`
- policies: `5,760`
- LightGBM configs / folds / boosters: `0 / 0 / 0`

### train v1 結果

- `likpf_mean` baseline: RMSE `11.594897672217703`, MAE `7.067632584311985`, within10 `0.772807479091509`
- exp109 reference global best: RMSE `11.143359413899551`, delta `-0.451538258318152`, max well regression `+6.5941827073495265`
- exp114 reference global best: RMSE `11.151818277057291`, delta `-0.44307939516041195`
- best gated policy: `pfbeam_oof__likpf_mean__typewell_native_overlap_0p999__any_outlier_signal_k8__std_le20__a0p2__c40`
  - RMSE `11.479140437818206`, delta `-0.11575723439949748`
  - MAE `7.04176347410932`, within10 `0.7754673705446818`
  - gate rows / wells / rate: `908,309` / `215` / `0.2400400741122662`
  - by-well improved / worse / same: `118` / `97` / `558`
  - max well regression / improvement: `+4.359665989725492` / `-6.324879137309921`
- guarded clip20 policy: `pfbeam_oof__likpf_mean__typewell_native_overlap_0p999__any_outlier_signal_k8__std_le20__a0p2__c20`
  - RMSE `11.497560716475249`, delta `-0.09733695574245438`
  - max well regression / improvement: `+3.0323877696964523` / `-3.441585801542562`
  - by-well improved / worse / same: `120` / `95` / `558`

best gated c40 は全 distance bucket で `likpf_mean` baseline を改善した。`000_050` は RMSE `1.1888775182606774 -> 1.1525695272383354`、`1000_plus` は `12.704015215339874 -> 12.580753080335171`。exp115 stress も `spatial_valid` が `13.643807871154252 -> 13.60424306513178`、`typewell_purged_valid` が `13.50680108059878 -> 13.450540494236355`。

### 判断

PF/Beam/likPF 候補上では exp109/114 prior signal は有効で、cluster-outlier gate は global reference correction より worst-well regression を下げる。ただし best gated でも max well regression が `+4.359665989725492`、guarded clip20 でも `+3.0323877696964523` 残るため、direct posthoc correction として inference port / submit には進めない。

exp181 は train-side no-training audit として完了。`cluster_outlier_pfbeam_prior_gate` backlog は完了として閉じ、再利用する場合は selector / confidence feature / candidate scoring の材料に限定する。
