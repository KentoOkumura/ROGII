# exp210_heatmap_mdn_full_well_path_generation_probe セッションノート

## 2026-07-07 実装

目的: backlog `heatmap_mdn_full_well_path_generation_probe` を実験化する。exp208 は dense stride local paths の source overlap と coverage を改善したが、後続 selector が直接読める `well,row_id,path_rank,tvt_pred` 形式の full-row path artifact は明示的に作っていなかった。exp210 では exp208 dense path artifact を再利用し、target-free stitch 後の topK full-well candidates を contract 付きで保存する。

### 設計

- Route: `pf_beam`
- 親: `exp202_heatmap_mdn_candidate_generator_probe`
- dense source: `exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe`
- stitch parent: `exp207_heatmap_mdn_overlapping_window_path_stitch_probe`
- comparison: `exp099_pf_multi_observation_likelihood_probe`
- downstream: `exp212_heatmap_mdn_full_grid_path_generation_probe`
- 実行内容: exp208 `dense_candidate_paths_top10.npz` / `dense_path_samples.csv.gz` を読み、local topK 5/10 で target-free beam stitch を top5 full-well candidates まで再実行する。
- 出力: `*_full_well_candidate_paths.csv.gz`、`*_full_well_path_schema.csv`、`*_full_well_contract_metrics.csv`、candidate union metrics / distance bucket / by-well readout。

### リークガード

- stitch score と contract table には true TVT、oracle best、abs-error、within10、candidate true-error rank を入れない。
- exp099 target は full-well paths 固定後の oracle readout にのみ使う。
- `md_from_ps` は exp099 cache の distance/alignment 診断列で、path selection target ではない。
- direct TVT replacement、softmax average、PF weight replacement、postprocess blend、inference、submit はしない。

### 予定コスト

- CNN models: 0
- LightGBM configs: 0
- folds: 0
- boosters: 0
- parent/control retraining: なし
- Kaggle GPU: disabled

### 実装ファイル

- `.steering/20260707-exp210-heatmap-mdn-full-well-path-generation-probe/`
- `experiments/exp210_heatmap_mdn_full_well_path_generation_probe/config.yaml`
- `experiments/exp210_heatmap_mdn_full_well_path_generation_probe/heatmap_mdn_full_well_path_generation_probe.py`
- `experiments/exp210_heatmap_mdn_full_well_path_generation_probe/exp210_heatmap_mdn_full_well_path_generation_probe_train.py`
- `experiments/exp210_heatmap_mdn_full_well_path_generation_probe/exp210_heatmap_mdn_full_well_path_generation_probe_inference.py`

### 検証ログ

実装時のローカル smoke は notebook 実行ではなく helper 関数のみ。Kaggle train は未実行。

実行:

```bash
.venv/bin/python -m py_compile experiments/exp210_heatmap_mdn_full_well_path_generation_probe/heatmap_mdn_full_well_path_generation_probe.py experiments/exp210_heatmap_mdn_full_well_path_generation_probe/exp210_heatmap_mdn_full_well_path_generation_probe_train.py experiments/exp210_heatmap_mdn_full_well_path_generation_probe/exp210_heatmap_mdn_full_well_path_generation_probe_inference.py experiments/exp210_heatmap_mdn_full_well_path_generation_probe/settings.py
.venv/bin/ruff check experiments/exp210_heatmap_mdn_full_well_path_generation_probe/heatmap_mdn_full_well_path_generation_probe.py experiments/exp210_heatmap_mdn_full_well_path_generation_probe/exp210_heatmap_mdn_full_well_path_generation_probe_train.py experiments/exp210_heatmap_mdn_full_well_path_generation_probe/exp210_heatmap_mdn_full_well_path_generation_probe_inference.py experiments/exp210_heatmap_mdn_full_well_path_generation_probe/settings.py --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp210_heatmap_mdn_full_well_path_generation_probe/exp210_heatmap_mdn_full_well_path_generation_probe_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp210_heatmap_mdn_full_well_path_generation_probe/exp210_heatmap_mdn_full_well_path_generation_probe_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp210_heatmap_mdn_full_well_path_generation_probe/exp210_heatmap_mdn_full_well_path_generation_probe_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp210_heatmap_mdn_full_well_path_generation_probe/exp210_heatmap_mdn_full_well_path_generation_probe_inference.py
make validate-exp EXP=exp210_heatmap_mdn_full_well_path_generation_probe
env PYTHONPATH=experiments/exp210_heatmap_mdn_full_well_path_generation_probe EXPERIMENT_DEBUG=1 EXPERIMENT_MAX_WELLS=2 .venv/bin/python -c "from settings import ExperimentPaths, load_config; from heatmap_mdn_full_well_path_generation_probe import run_stitch_probe; s=run_stitch_probe(config=load_config(), paths=ExperimentPaths(), max_wells=2, debug=True); print(s['status']); print(s['primary_local_topk']); print(s['topk_summaries'][-1]['full_path_contract'])"
make prepare-kaggle-notebooks EXP=exp210_heatmap_mdn_full_well_path_generation_probe EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp210-hmdn-full-well-path-generation-train --title 'exp210 hmdn full well path generation train' --run-on-push --strict"
```

結果:

- py_compile: pass
- ruff F821: pass
- jupytext train/inference conversion + `--test`: pass
- `make validate-exp`: pass
- debug smoke: `debug_completed`
- primary local topK: `10`
- required columns present: `True`
- duplicate key rows: `0`
- null required value count: `0`
- debug rows: `21,110`
- debug unique row ids: `4,222`
- path count: `10`
- prepare-kaggle-notebooks train: pass
- prepared kernel id: `kentookumura/exp210-hmdn-full-well-path-generation-train`
- prepared title: `exp210 hmdn full well path generation train`
- GPU / internet: disabled / disabled
- kernel sources: `kentookumura/exp208-hmdn-dense-stride-train`, `kentookumura/exp099-pf-multiobs-likelihood-train`

debug smoke は `artifacts/` に 2 wells 分の小さな生成物を作った。これは Kaggle 評価結果ではなく、実装時の schema / join 確認用。

## 2026-07-07 Kaggle train v1 push

ユーザー依頼で Kaggle train を実行開始した。

実行:

```bash
kaggle kernels push -p experiments/exp210_heatmap_mdn_full_well_path_generation_probe/kaggle/train
kaggle kernels pull kentookumura/exp210-hmdn-full-well-path-generation-train -p /tmp/kaggle-pull/exp210-hmdn-full-well-path-generation-train -m
kaggle kernels logs kentookumura/exp210-hmdn-full-well-path-generation-train
timeout 600 kaggle kernels logs -f --interval 30 kentookumura/exp210-hmdn-full-well-path-generation-train
kaggle kernels status kentookumura/exp210-hmdn-full-well-path-generation-train
```

結果:

- push: success
- kernel version: `1`
- URL: <https://www.kaggle.com/code/kentookumura/exp210-hmdn-full-well-path-generation-train>
- pull metadata: success
- initial logs / logs -f: empty while running
- last checked status: `KernelWorkerStatus.RUNNING`
- monitoring: ユーザー指示により停止

次アクション: ユーザーから完了連絡を受けたら、同じ kernel id の logs を取得し、必要なら output archive を取得して full-well contract metrics / candidate union metrics / SHA を `result.md`、`metrics.json`、`experiment_summary.md`、`KAGGLE_DIRECTION.md` に反映する。

## 2026-07-07 Kaggle train v1 完了記録

ユーザーから完了連絡を受け、logs と output を取得した。

実行:

```bash
kaggle kernels status kentookumura/exp210-hmdn-full-well-path-generation-train
kaggle kernels logs kentookumura/exp210-hmdn-full-well-path-generation-train
kaggle kernels output kentookumura/exp210-hmdn-full-well-path-generation-train -p experiments/exp210_heatmap_mdn_full_well_path_generation_probe/kaggle/output/train_v1
```

結果:

- status: `KernelWorkerStatus.COMPLETE`
- output: `experiments/exp210_heatmap_mdn_full_well_path_generation_probe/kaggle/output/train_v1`
- exp208 dense path input: `25,452` samples / `773` wells / topK `10`
- exp099 candidate cache: `3,783,989` rows / `773` wells
- primary local topK: `10`
- full-well contract rows: `8,137,310`
- unique row ids: `1,627,462`
- coverage vs exp099 cache: `0.4300916308160515`
- required columns present: `True`
- duplicate key rows: `0`
- null required value count: `0`
- source overlap wells / pairs: `773 / 24,679`
- source gap pair count: `0`
- stitched row gap count: `0`
- path step abs mean: `0.24851098854292333`
- curvature abs mean: `0.4875124523192066`
- existing union oracle RMSE / within10: `5.139413348675699` / `0.9475225842446705`
- existing + stitched top5 oracle RMSE / delta / within10: `4.407737500103614` / `-0.7316758485720856` / `0.9608648312525884`
- stitched only top5 oracle RMSE / within10: `46.95894604882118` / `0.2927656682613788`
- `1000_plus` bucket union RMSE: `6.352450934 -> 5.403899359`
- by-well: `524 improved / 249 same / 0 worse`
- best well: `1b1eba53`, RMSE `37.5347308526953 -> 21.331056019270253`
- primary full-well candidate path decompressed SHA256: `f22808f0c0af8cc8a2953680284db9d8564fcecfa401a8921c6130e29f8509f0`

判断:

- full-well artifact contract は成立したため backlog は完了扱い。
- stitched-only path は弱いため direct replacement、softmax average、PF weight replacement、postprocess blend、inference、submit はしない。
- この artifact は exp099 candidate-cache intersection の covered rows に限定され、exp072/exp083 plot の全 `md_since` 区間を覆う full-grid trajectory ではない。
- selector の通常候補として使うには、別 backlog `exp212_heatmap_mdn_full_grid_path_generation_probe` で全 row grid coverage、gap count、path continuity、stitched-only RMSE、existing+new oracle、longtail を先に検証する。
