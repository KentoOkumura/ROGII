# exp187_cluster_outlier_alt_typewell_pfbeam_audit セッションノート

## 目的

`cluster_outlier_alt_typewell_pfbeam_audit` backlog を実装する。cluster 外れ well だけを対象に、query well 自身の typewell と別 cluster composite typewell を使った PF/Beam 生成を比較する。

## 現在の状態

- Route: pf_beam
- 状態: completed_train_side_rejected_no_submit
- CV: own PF RMSE 17.011319 / own Beam RMSE 16.287400
- LB: まだなし
- 注記: v1 は artificial 192-row prefix holdout / representative typewell 参照だったため、validation と typewell source を修正して v2 を実行済み。結論は v2 を正とする。

## 実行予定

- active variant 数: typewell strategy 3 種 (`own_typewell`, `nearest_other_cluster_composite`, `nearby_majority_cluster_composite_k8`)
- model/config 数: LightGBM なし、PF/Beam generation audit のみ
- fold 数: 0
- 合計 booster 数: 0
- control / 親実験の再学習: なし
- target wells: 最大 64 wells
- score rows: exp072/099 と同じ `TVT_input_missing_equivalent_exp063_rows`
- alt typewell: source cluster member typewell を TVT bin ごとに結合した composite
- PF config: 260 particles x 8 seeds / strategy
- Beam config: beam size 14、move radius 2

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

### 実装時

```bash
make new-steering EXP=exp187_cluster_outlier_alt_typewell_pfbeam_audit
make new-exp EXP=exp187_cluster_outlier_alt_typewell_pfbeam_audit
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp187_cluster_outlier_alt_typewell_pfbeam_audit/exp187_cluster_outlier_alt_typewell_pfbeam_audit_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp187_cluster_outlier_alt_typewell_pfbeam_audit/exp187_cluster_outlier_alt_typewell_pfbeam_audit_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp187_cluster_outlier_alt_typewell_pfbeam_audit/exp187_cluster_outlier_alt_typewell_pfbeam_audit_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp187_cluster_outlier_alt_typewell_pfbeam_audit/exp187_cluster_outlier_alt_typewell_pfbeam_audit_inference.py
.venv/bin/python -m py_compile experiments/exp187_cluster_outlier_alt_typewell_pfbeam_audit/cluster_outlier_alt_typewell_pfbeam_audit.py experiments/exp187_cluster_outlier_alt_typewell_pfbeam_audit/exp187_cluster_outlier_alt_typewell_pfbeam_audit_train.py experiments/exp187_cluster_outlier_alt_typewell_pfbeam_audit/exp187_cluster_outlier_alt_typewell_pfbeam_audit_inference.py
.venv/bin/ruff check experiments/exp187_cluster_outlier_alt_typewell_pfbeam_audit
make validate-exp EXP=exp187_cluster_outlier_alt_typewell_pfbeam_audit
make prepare-kaggle-notebooks EXP=exp187_cluster_outlier_alt_typewell_pfbeam_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp187-cluster-outlier-alt-typewell-pfbeam-audit-train --title 'exp187 cluster outlier alt typewell pfbeam audit train' --run-on-push --strict"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experiments/exp187_cluster_outlier_alt_typewell_pfbeam_audit .venv/bin/python -c "from settings import ExperimentPaths, load_config; from cluster_outlier_alt_typewell_pfbeam_audit import build_cluster_features, select_target_wells, summarize_strategy_sources; paths=ExperimentPaths(); cfg=load_config(); cf, meta=build_cluster_features(cfg); tf=select_target_wells(cf, paths.train_data_dir, cfg); ss=summarize_strategy_sources(tf, paths.train_data_dir, cfg); print({'cluster_rows': len(cf), 'target_wells': len(tf), 'strategy_rows': len(ss), 'strategies': sorted(ss['strategy'].unique().tolist()) if len(ss) else []})"
make update-summary
```

### v2 validation/typewell source 修正

ユーザー指摘により、v1 の validation は従来 PF/Beam audit と同じではないため修正する。また、別 cluster の typewell は representative 1本ではなく、source cluster の available member typewell を結合した composite typewell として参照する。

```bash
.venv/bin/python -m py_compile experiments/exp187_cluster_outlier_alt_typewell_pfbeam_audit/cluster_outlier_alt_typewell_pfbeam_audit.py experiments/exp187_cluster_outlier_alt_typewell_pfbeam_audit/exp187_cluster_outlier_alt_typewell_pfbeam_audit_train.py experiments/exp187_cluster_outlier_alt_typewell_pfbeam_audit/exp187_cluster_outlier_alt_typewell_pfbeam_audit_inference.py
.venv/bin/ruff check experiments/exp187_cluster_outlier_alt_typewell_pfbeam_audit
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experiments/exp187_cluster_outlier_alt_typewell_pfbeam_audit .venv/bin/python - <<'PY'
from settings import ExperimentPaths, load_config
from cluster_outlier_alt_typewell_pfbeam_audit import build_cluster_features, select_target_wells, summarize_strategy_sources
paths=ExperimentPaths(); cfg=load_config()
cf, meta=build_cluster_features(cfg)
tf=select_target_wells(cf, paths.train_data_dir, cfg)
ss=summarize_strategy_sources(tf, paths.train_data_dir, cfg)
print({'cluster_rows': len(cf), 'target_wells': len(tf), 'strategy_rows': len(ss), 'strategies': sorted(ss['strategy'].unique().tolist()) if len(ss) else []})
print(ss.groupby('strategy')['source_member_count'].describe().to_string())
PY
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp187_cluster_outlier_alt_typewell_pfbeam_audit/exp187_cluster_outlier_alt_typewell_pfbeam_audit_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp187_cluster_outlier_alt_typewell_pfbeam_audit/exp187_cluster_outlier_alt_typewell_pfbeam_audit_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp187_cluster_outlier_alt_typewell_pfbeam_audit/exp187_cluster_outlier_alt_typewell_pfbeam_audit_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp187_cluster_outlier_alt_typewell_pfbeam_audit/exp187_cluster_outlier_alt_typewell_pfbeam_audit_inference.py
make validate-exp EXP=exp187_cluster_outlier_alt_typewell_pfbeam_audit
```

結果:

- py_compile: PASS
- ruff: PASS
- target selection smoke: `cluster_rows=773`、`target_wells=64`、`strategy_rows=164`
- strategies: `own_typewell` / `nearest_other_cluster_composite` / `nearby_majority_cluster_composite_k8`
- source_member_count: nearest composite mean 19.73 wells、nearby composite mean 26.06 wells、own 1 well
- jupytext conversion/test: PASS
- validate-exp: PASS
- local constraint: `experiments/exp072_exp063_full_replay_feature_cache/artifacts/` には schema のみがあり、train feature cache 本体は Kaggle input から読むため、本体 audit のローカル実行はしない。

### Kaggle train v2

```bash
make prepare-kaggle-notebooks EXP=exp187_cluster_outlier_alt_typewell_pfbeam_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp187-alt-typewell-pfbeam-audit-train --title 'exp187 alt typewell pfbeam audit train' --run-on-push --strict"
make push-kaggle-train EXP=exp187_cluster_outlier_alt_typewell_pfbeam_audit
kaggle kernels pull kentookumura/exp187-alt-typewell-pfbeam-audit-train -p /tmp/kaggle-pull/exp187-alt-typewell-pfbeam-audit-train-v2 -m
kaggle kernels status kentookumura/exp187-alt-typewell-pfbeam-audit-train
kaggle kernels logs kentookumura/exp187-alt-typewell-pfbeam-audit-train
kaggle kernels output kentookumura/exp187-alt-typewell-pfbeam-audit-train -p experiments/exp187_cluster_outlier_alt_typewell_pfbeam_audit/kaggle/output/train_v2
```

結果:

- kernel: `kentookumura/exp187-alt-typewell-pfbeam-audit-train` v2
- URL: https://www.kaggle.com/code/kentookumura/exp187-alt-typewell-pfbeam-audit-train
- id_no: 125890196
- status: COMPLETE
- runtime: summary 1,049.754 sec / logs last time 1,081.803 sec
- output: `experiments/exp187_cluster_outlier_alt_typewell_pfbeam_audit/kaggle/output/train_v2`
- validation source: exp072 train feature cache 3,783,989 rows / 773 wells
- exp072 cache decompressed SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- eval rows: 306,490 rows / 64 wells
- strategy rows: 164
- typewell cache entries: 89

主要 metrics:

- primary baseline `pf_own_typewell_lik_mean`: RMSE 17.011319 / MAE 11.503088 / within10 0.586701
- best non-oracle `beam_own_typewell_top1`: RMSE 16.287400 / MAE 11.798330 / within10 0.548997
- `pf_nearest_other_cluster_composite_lik_mean`: RMSE 191.623692 / MAE 106.358104 / within10 0.295429 / delta +174.612373
- `pf_nearby_majority_cluster_composite_k8_lik_mean`: RMSE 189.864085 / MAE 112.125988 / within10 0.302059 / delta +172.852766
- `beam_nearest_other_cluster_composite_top1`: RMSE 205.606487 / MAE 115.216130 / within10 0.273210 / delta +188.595168
- `beam_nearby_majority_cluster_composite_k8_top1`: RMSE 195.101081 / MAE 114.264089 / within10 0.249461 / delta +178.089763

By-well:

- nearest composite PF improved 11/64 wells、最大 regression +572.507ft。
- nearby-majority composite PF improved 9/64 wells、最大 regression +615.042ft。
- nearest composite Beam improved 11/64 wells、最大 regression +614.988ft。
- nearby-majority composite Beam improved 9/64 wells、最大 regression +653.803ft。

解釈:

- validation surface は exp072/099 と同じ `TVT_input_missing_equivalent_exp063_rows` に修正済み。
- alt typewell は representative 1本ではなく cluster member composite を参照済み。
- composite にしても絶対 TVT range / GR depth alignment 問題は解消せず、global candidate としては明確に不採用。
- hard switch、direct candidate replacement、inference port、submit は行わない。

結果:

- `py_compile`: PASS
- `ruff check`: PASS
- `jupytext --to ipynb --test`: train / inference とも PASS
- `validate-exp`: PASS
- Kaggle train package: `experiments/exp187_cluster_outlier_alt_typewell_pfbeam_audit/kaggle/train`
- kernel id: `kentookumura/exp187-cluster-outlier-alt-typewell-pfbeam-audit-train`
- metadata: GPU false / internet false / run_on_push true
- bootstrap manifest: `config.yaml`、`cluster_outlier_alt_typewell_pfbeam_audit.py`、train/inference `.py`、`settings.py` を含む
- local selection smoke: `cluster_rows=773`、`target_wells=64`、`strategy_rows=164`、strategies は `own_typewell` / `nearest_other_cluster_rep` / `nearby_majority_cluster_rep_k8`

### Kaggle train v1

最初の long slug は `SaveKernel` 400 で失敗した。id/title slug は一致していたが、過去の長い exp と同様に slug 長が疑わしいため、同じ exp187 のまま意味を残した short slug に変更した。

```bash
make push-kaggle-train EXP=exp187_cluster_outlier_alt_typewell_pfbeam_audit
make prepare-kaggle-notebooks EXP=exp187_cluster_outlier_alt_typewell_pfbeam_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp187-alt-typewell-pfbeam-audit-train --title 'exp187 alt typewell pfbeam audit train' --run-on-push --strict"
make push-kaggle-train EXP=exp187_cluster_outlier_alt_typewell_pfbeam_audit
kaggle kernels pull kentookumura/exp187-alt-typewell-pfbeam-audit-train -p /tmp/kaggle-pull/exp187-alt-typewell-pfbeam-audit-train -m
kaggle kernels logs kentookumura/exp187-alt-typewell-pfbeam-audit-train
timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp187-alt-typewell-pfbeam-audit-train
kaggle kernels output kentookumura/exp187-alt-typewell-pfbeam-audit-train -p experiments/exp187_cluster_outlier_alt_typewell_pfbeam_audit/kaggle/output/train_v1
```

結果:

- kernel: `kentookumura/exp187-alt-typewell-pfbeam-audit-train` v1
- URL: https://www.kaggle.com/code/kentookumura/exp187-alt-typewell-pfbeam-audit-train
- id_no: 125890196
- runtime: 34.014935 sec
- rows / wells: 12,288 rows / 64 wells
- target wells selected: 64
- strategy rows: 164
- output: `experiments/exp187_cluster_outlier_alt_typewell_pfbeam_audit/kaggle/output/train_v1`
- best non-oracle: `beam_own_typewell_top1` RMSE 3.167908 / MAE 2.042757 / within10 0.983643
- primary baseline: `pf_own_typewell_lik_mean` RMSE 3.333187 / MAE 2.324363 / within10 0.984701
- alt nearest PF: `pf_nearest_other_cluster_rep_lik_mean` RMSE 181.225170 / delta +177.891982
- alt nearby-majority PF: `pf_nearby_majority_cluster_rep_k8_lik_mean` RMSE 138.856987 / delta +135.523799
- alt nearest Beam: `beam_nearest_other_cluster_rep_top1` RMSE 233.086263 / delta +229.753076
- alt nearby-majority Beam: `beam_nearby_majority_cluster_rep_k8_top1` RMSE 186.146368 / delta +182.813180
- improved wells: nearest PF 11/64、nearby PF 8/64、nearest Beam 17/64、nearby Beam 10/64
- max regression: nearest PF +522.385、nearby PF +550.800、nearest Beam +623.574、nearby Beam +650.562

判定:

- Alternative representative typewell をそのまま PF/Beam generation に使う設計は不採用。
- hard switch、direct candidate replacement、inference port、submit はしない。
- 原因は別 typewell の絶対 TVT range / GR depth alignment が query well と合わず、wrong depth に吸い込まれること。

### 予定

```bash
```

## 変更点

- exp065 cluster assignment と exp114 geometry から target-free cluster outlier features を構築する helper を追加。
- inverse-distance weighted nearby-majority cluster から代表 typewell を選ぶ strategy を追加。
- scoped prefix holdout 上で PF likelihood-weighted mean、best seed、top3 oracle diagnostic、Beam top1 を生成する audit を追加。
- candidate metrics、own-vs-alt delta metrics、bucket/group/by-well metrics、PF diagnostics、target well features、strategy sources、row candidates を保存する。

## 再現性メモ

- seed policy: `stable_sha256_per_query_well_seed_index_shared_across_typewell_strategies`
- stochastic components: PF particle propagation / resampling
- CPU/GPU runtime: CPU only、GPU disabled、internet disabled
- Kaggle kernel id / version: `kentookumura/exp187-alt-typewell-pfbeam-audit-train` v1
- input / feature schema SHA: exp065 assignment `dcda8588cc1dd9261bafae7de00c890393e38b8a0ca0eb86fbba18a2cffc4a50`、exp114 geometry `079fbd65ed92bbe21a1cbf0006c445562fc451c026daa805c526d34496c4bf2a`
- feature content SHA: row candidates raw gzip `c177dde5651cdaea61768ea25ad8b90cdc2bc3a3bccd1a8f24c520bffb147733`、decompressed `d2d6a040284342bfaee344a462d342980541b0a2ee16cfe2c1856d9c0276207f`
- model manifest / model SHA: model なし
- prediction SHA: candidate metrics `26be45815b7a880e19bdcf309d99f9793674b0048cc6796f159584d919007b7c`、strategy delta `7faec9d4d17a8ed99052a0ceffd15c780b8550d379562130a247096f708cb785`
- submission SHA: submission なし
- rerun check: 未実行

## 次のアクション

1. 完了。不採用として閉じる。
