# exp212_heatmap_mdn_full_grid_path_generation_probe セッションノート

## 2026-07-07 実装

目的: backlog `exp212_heatmap_mdn_full_grid_path_generation_probe` を実験化する。exp210 は exp099 candidate-cache intersection の covered rows では contract が成立したが、coverage は 0.430091631 で、selector の通常候補として全 row で読むには不足していた。exp212 では exp208 dense local paths を source に、exp099 feature-cache row grid 全体を覆う topK full-grid candidate paths を作る。

### 設計

- Route: `pf_beam`
- 親: `exp202_heatmap_mdn_candidate_generator_probe`
- dense source: `exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe`
- stitch parent: `exp207_heatmap_mdn_overlapping_window_path_stitch_probe`
- comparison: `exp099_pf_multi_observation_likelihood_probe`
- downstream: `exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158`
- 入力: exp208 `dense_candidate_paths_top10.npz` / `dense_path_samples.csv.gz`
- active local topK: `10` のみ。full-grid output は exp099 rows x path ranks で大きいため、初回は primary artifact に絞る。
- full-grid fill: source rows は stitched prediction を直接使い、未カバー row は row_index 線形補間または端点外挿で埋める。
- 出力: `*_full_grid_candidate_paths.csv.gz`、`*_full_grid_path_schema.csv`、`*_full_grid_contract_metrics.csv`、candidate union metrics / distance bucket / by-well readout。

### リークガード

- stitch score、full-grid fill、contract table には true TVT、oracle best、abs-error、within10、candidate true-error rank を入れない。
- exp099 target は full-grid paths 固定後の oracle readout にのみ使う。
- `md_since` は exp099 cache の distance/alignment 診断列で、path selection target ではない。
- direct TVT replacement、softmax average、PF weight replacement、postprocess blend、inference、submit はしない。

### 予定コスト

- CNN models: 0
- LightGBM configs: 0
- folds: 0
- boosters: 0
- parent/control retraining: なし
- Kaggle GPU: disabled

### 実装ファイル

- `docs/legacy/steering/20260707-exp212-heatmap-mdn-full-grid-path-generation-probe/`
- `experiments/exp212_heatmap_mdn_full_grid_path_generation_probe/config.yaml`
- `experiments/exp212_heatmap_mdn_full_grid_path_generation_probe/heatmap_mdn_full_grid_path_generation_probe.py`
- `experiments/exp212_heatmap_mdn_full_grid_path_generation_probe/exp212_heatmap_mdn_full_grid_path_generation_probe_train.py`
- `experiments/exp212_heatmap_mdn_full_grid_path_generation_probe/exp212_heatmap_mdn_full_grid_path_generation_probe_inference.py`

### 検証ログ

実行:

```bash
.venv/bin/python -m py_compile experiments/exp212_heatmap_mdn_full_grid_path_generation_probe/heatmap_mdn_full_grid_path_generation_probe.py experiments/exp212_heatmap_mdn_full_grid_path_generation_probe/exp212_heatmap_mdn_full_grid_path_generation_probe_train.py experiments/exp212_heatmap_mdn_full_grid_path_generation_probe/exp212_heatmap_mdn_full_grid_path_generation_probe_inference.py experiments/exp212_heatmap_mdn_full_grid_path_generation_probe/settings.py
.venv/bin/ruff check experiments/exp212_heatmap_mdn_full_grid_path_generation_probe/heatmap_mdn_full_grid_path_generation_probe.py experiments/exp212_heatmap_mdn_full_grid_path_generation_probe/exp212_heatmap_mdn_full_grid_path_generation_probe_train.py experiments/exp212_heatmap_mdn_full_grid_path_generation_probe/exp212_heatmap_mdn_full_grid_path_generation_probe_inference.py experiments/exp212_heatmap_mdn_full_grid_path_generation_probe/settings.py --select F821,E501
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp212_heatmap_mdn_full_grid_path_generation_probe/exp212_heatmap_mdn_full_grid_path_generation_probe_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp212_heatmap_mdn_full_grid_path_generation_probe/exp212_heatmap_mdn_full_grid_path_generation_probe_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp212_heatmap_mdn_full_grid_path_generation_probe/exp212_heatmap_mdn_full_grid_path_generation_probe_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp212_heatmap_mdn_full_grid_path_generation_probe/exp212_heatmap_mdn_full_grid_path_generation_probe_inference.py
make validate-exp EXP=exp212_heatmap_mdn_full_grid_path_generation_probe
make prepare-kaggle-notebooks EXP=exp212_heatmap_mdn_full_grid_path_generation_probe EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp212-hmdn-full-grid-path-generation-train --title 'exp212 hmdn full grid path generation train' --run-on-push --strict"
```

結果:

- `py_compile`: pass
- `ruff --select F821,E501`: pass
- Jupytext train / inference conversion: pass
- Jupytext train / inference `--test`: pass
- `make validate-exp`: pass
- Kaggle train package: `experiments/exp212_heatmap_mdn_full_grid_path_generation_probe/kaggle/train`
- kernel id: `kentookumura/exp212-hmdn-full-grid-path-generation-train`
- title: `exp212 hmdn full grid path generation train`
- metadata: GPU false、internet false、run_on_push true
- kernel sources: `kentookumura/exp208-hmdn-dense-stride-train`、`kentookumura/exp099-pf-multiobs-likelihood-train`

local helper debug smoke は、exp208 npz と exp099 cache を同時に読むためローカル環境で SIGKILL された。notebook のローカル実行は行わず、代わりに synthetic table smoke で `build_full_grid_path_table`、`contract_metrics_frame`、`evaluate_union` を検証した。

synthetic smoke 結果:

- `row_coverage_rate_vs_cache`: `1.0`
- `duplicate_key_rows`: `0`
- `null_required_value_count`: `0`
- `source_coverage_rate_vs_grid`: `0.4`
- `fallback_unique_row_rate`: `0.6`
- fill methods: `left_extrapolated`、`source_window`、`interpolated`、`right_extrapolated`

## 2026-07-07 Kaggle train 実行

ユーザー依頼により Kaggle train を実行する。

実行前ガード:

- route: `pf_beam`
- active local topK values: `[10]`
- CNN training models: `0`
- LightGBM configs / boosters: `0 / 0`
- folds: `0`
- parent/control retraining: なし
- Kaggle GPU: disabled
- internet: disabled
- inference / submit: 対象外

予定 kernel:

- id: `kentookumura/exp212-hmdn-full-grid-path-generation-train`
- title: `exp212 hmdn full grid path generation train`

予定コマンド:

```bash
make validate-exp EXP=exp212_heatmap_mdn_full_grid_path_generation_probe
make prepare-kaggle-notebooks EXP=exp212_heatmap_mdn_full_grid_path_generation_probe EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp212-hmdn-full-grid-path-generation-train --title 'exp212 hmdn full grid path generation train' --run-on-push --strict"
make push-kaggle-train EXP=exp212_heatmap_mdn_full_grid_path_generation_probe
kaggle kernels pull kentookumura/exp212-hmdn-full-grid-path-generation-train -p /tmp/kaggle-pull/exp212-hmdn-full-grid-path-generation-train -m
kaggle kernels logs kentookumura/exp212-hmdn-full-grid-path-generation-train
```

実行結果:

- `make validate-exp`: pass
- `make prepare-kaggle-notebooks`: pass
- `make push-kaggle-train`: success
- kernel version: `1`
- URL: <https://www.kaggle.com/code/kentookumura/exp212-hmdn-full-grid-path-generation-train>
- id_no: `126206534`
- pulled metadata: CPU / GPU false / internet false / competition sourceあり / kernel sources `kentookumura/exp099-pf-multiobs-likelihood-train`, `kentookumura/exp208-hmdn-dense-stride-train`
- initial logs: Kaggle CLI version warning のみで本文空。実行中 logs 空は既知挙動なので失敗扱いしない。
- initial status: `KernelWorkerStatus.RUNNING`
- `timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp212-hmdn-full-grid-path-generation-train` で follow を開始したが、ユーザー指示によりローカル監視を停止した。Kaggle kernel 実行自体は継続中。

## 2026-07-07 Kaggle train 完了

ユーザーから Kaggle 側の完了連絡を受け、status / logs / output を確認した。

確認コマンド:

```bash
kaggle kernels status kentookumura/exp212-hmdn-full-grid-path-generation-train
kaggle kernels logs kentookumura/exp212-hmdn-full-grid-path-generation-train
kaggle kernels output kentookumura/exp212-hmdn-full-grid-path-generation-train -p experiments/exp212_heatmap_mdn_full_grid_path_generation_probe/kaggle/output/train_v1
```

結果:

- status: `COMPLETE`
- kernel version: `1`
- elapsed log seconds: `4635.838601125`
- output: `experiments/exp212_heatmap_mdn_full_grid_path_generation_probe/kaggle/output/train_v1`
- primary artifact: `artifacts/exp212_heatmap_mdn_full_grid_path_generation_probe_localtopk10_full_grid_candidate_paths.csv.gz`
- primary artifact gzip SHA256: `6fb44a0e4728717448194ba8e7f3a0e5ce8a078c2d7a8dd8ff9e9942cf5878d0`
- primary artifact decompressed SHA256: `66c110cc14e72167d11d51d9054a9cab0830eea15000d3e012347a7032da1ec2`
- summary SHA256: `61fb91f0c07b258cf1aa173bd4c5d07b454623932e763b4ac9e552a65b6f9ecb`

Full-grid contract:

- required columns present: `true`
- missing required columns: `[]`
- duplicate key rows: `0`
- null required value count: `0`
- rows: `18,919,945`
- unique row ids: `3,783,989`
- wells: `773`
- row coverage vs exp099 cache: `1.0`
- rows by rank: `1..5` each `3,783,989`
- path count: `3,865`
- source-covered unique row ids: `1,627,462`
- source coverage vs grid: `0.4300916308160515`
- fallback unique row ids: `2,156,527`
- fallback unique row rate: `0.5699083691839485`
- fill method rows: `source_window=8,137,310`, `right_extrapolated=10,782,635`

Physicality:

- source overlap wells: `773`
- source overlap pair count: `24,679`
- source gap pair count: `0`
- stitched row gap count total: `0`
- stitched path step abs mean ft: `0.24851098854292333`
- full-grid path step abs mean ft: `0.10306967794895172`
- full-grid curvature abs mean ft: `0.20302779972553253`
- assignment overlap abs mean ft: `7.243898115924085`

Candidate union oracle readout:

- existing union on full-grid rows: RMSE `7.434029841171774`, MAE `3.7452277525553224`, within10 `0.9065253625208741`
- stitched only top5: RMSE `50.08523757271513`, MAE `31.42772917835104`, within10 `0.3142620129181137`
- existing + stitched top5: RMSE `5.9414799954943645`, MAE `3.1105414066470263`, within10 `0.9334601659782837`, new-best rate `0.11573500874341865`
- top5 delta vs existing: RMSE `-1.4925498456774093`, within10 `+0.026934803457409617`
- `1000_plus`: existing `8.16179657742627` -> union `6.491812972797192`, new-best rate `0.13256985528684037`
- by-well top5: `567 improved / 206 same / 0 worse`, mean RMSE delta `-0.7565241378294933`
- best well: `86454a6f`, delta `-22.712819184338446`

判定:

- Full-grid artifact contract は成立した。
- ただし source support は `43.0%` に留まり、`57.0%` は右端点外挿なので fallback-heavy。
- stitched-only top5 RMSE は `50.08523757271513` と弱く、direct replacement、softmax average、PF weight replacement、inference、submit はしない。
- existing + stitched top5 oracle は `7.434029841171774 -> 5.9414799954943645` と headroom があるが、これは oracle readout であり deployable proof ではない。
- 後続 exp204 系では、`coverage_flag`、`fallback_flag`、`fill_method`、`candidate_score`、`path_step_abs`、`curvature_abs` を selector features / guards に入れる前提で guarded-only に進める。fallback bucket 監査なしに通常候補化しない。

## 2026-07-07 plot audit

ユーザーが exp212 path overlay を確認し、「ある箇所から直線になっていて正しく最後まで生成しようとしていない」と指摘した。確認結果:

- 指摘は正しい。これは描画の問題ではなく exp212 artifact の生成仕様と入力 source coverage の問題。
- exp208 dense source は `model.training.max_tail_rows=2048`、`row_center_stride=64`、`horizontal_window_rows=128` で、各 well 約 33 windows / 2175 source rows しか生成していない。
- exp212 `build_full_grid_path_table` は source rows の外側を `np.interp(grid_rows, source_rows, source_tvt)` の endpoint hold で埋め、`row_index > source_rows[-1]` を `right_extrapolated` としている。
- Kaggle output の `fill_method_rows` は `source_window=8,137,310`、`right_extrapolated=10,782,635` で、interpolated / left_extrapolated は実 run では 0。
- したがって直線 tail は「最後まで heatmap path を生成した結果」ではなく、最後の source window 以降の外挿 tail。

結論:

- exp212 は full-grid schema / row coverage の contract 診断としては成立したが、visual path generation としては失敗。
- exp204 にそのまま通常候補として渡さない。仮に使う場合も fallback guard 付き diagnostic candidate に限定する。
- 正しく最後まで生成するには、exp212 の fill だけを変えるのでは不十分。exp208 相当の dense path generation を `max_tail_rows` 制限なし、または exp099 feature-cache row grid の末尾まで rerun し、source coverage 自体を full tail に近づける必要がある。
