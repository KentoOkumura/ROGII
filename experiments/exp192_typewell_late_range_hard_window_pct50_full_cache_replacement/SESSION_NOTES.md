# exp192_typewell_late_range_hard_window_pct50_full_cache_replacement セッションノート

## 2026-07-04 実装

- ユーザー依頼により `typewell_late_range_hard_window_pct50_full_cache_replacement` backlog の実装を開始。
- `docs/legacy/steering/20260704-exp192-typewell-late-range-hard-window-pct50-full-cache-replacement/` を作成。
- `experiments/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement/` を exp186 から作成。
- Route: `pf_beam`
- GPU 学習: なし。CPU-only full replay train feature cache generation。
- feature cache variant 数: 1 (`pixiux_likpf_hard_window_pct50_public_replay`)
- hard-window threshold: `typewell_pct >= 0.50` のみ。`0.60/0.70` grid なし。
- LightGBM config 数: 0
- fold 数: 0
- 合計 booster 数: 0
- control / parent 再学習: なし
- inference / submit: なし

## 実装内容

- `hard_window_public_replay.py` を exp186 の corrected full replay implementation から派生。
- soft prior default を `no_prior` に戻し、penalty を 0 にした。
- `apply_typewell_hard_window()` を追加し、raw typewell の元 finite TVT min/max から `typewell_pct` を計算。
- `load_well()` と `build_well()` の typewell 読み込み直後に `0.50 <= typewell_pct <= 1.00` filter を適用。
- `feature_cache.py` は `hard_window` config を runtime へ渡し、exp192 用 output prefix / variant を出力する。
- train notebook 起点 `.py` は raw input check、hard-window contract、feature cache generation、generated artifacts のセル構成に更新。
- inference notebook 起点 `.py` は train-cache-only no-op として更新。

## 再現性メモ

- seed policy: stable SHA seed per well / split / feature family。
- stochastic components: PF particle propagation / resampling / likelihood-PF seed ensemble。
- Beam: deterministic。
- CPU/GPU runtime: Kaggle CPU、`enable_gpu=false`。
- deterministic anchor: false。train feature cache only で submission anchor ではない。
- gzip 生成物は raw gzip SHA と decompressed content SHA を分け、decompressed content SHA を主証拠にする。

## コマンドログ

```bash
make new-steering EXP=exp192_typewell_late_range_hard_window_pct50_full_cache_replacement
make new-exp EXP=exp192_typewell_late_range_hard_window_pct50_full_cache_replacement SOURCE=experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior
```

- result: steering 作成は PASS。最初の `new-exp` は並行実行で steering check より先に走って失敗したため、単独再実行で PASS。

```bash
.venv/bin/python -m py_compile \
  experiments/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement/hard_window_public_replay.py \
  experiments/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement/feature_cache.py \
  experiments/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement/settings.py \
  experiments/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement_train.py \
  experiments/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement_inference.py

.venv/bin/ruff check \
  experiments/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement/hard_window_public_replay.py \
  experiments/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement/feature_cache.py \
  experiments/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement/settings.py \
  experiments/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement_train.py \
  experiments/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement_inference.py --select F821
```

- result: PASS

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement_inference.py
```

- result: PASS

```bash
uv run python scripts/validate_experiment.py --experiment exp192_typewell_late_range_hard_window_pct50_full_cache_replacement
```

- result: PASS (`experiment validation passed (strict)`)

```bash
make prepare-kaggle-notebooks EXP=exp192_typewell_late_range_hard_window_pct50_full_cache_replacement EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp192-typewell-hard-window-pct50-train --title 'exp192 typewell hard window pct50 train' --run-on-push --strict"
```

- result: PASS
- output: `experiments/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement/kaggle/train`
- kernel id: `kentookumura/exp192-typewell-hard-window-pct50-train`
- title: `exp192 typewell hard window pct50 train`
- metadata `enable_gpu`: false
- metadata `enable_internet`: false
- competition source: `rogii-wellbore-geology-prediction`
- kernel sources: none
- bootstrap/package config confirmed `model.hard_window.name: pct50`, `min_typewell_pct: 0.50`, `variant: pixiux_likpf_hard_window_pct50_public_replay`

## 次のアクション

Kaggle train v1 は完了。以降の記録を参照する。

## 2026-07-04 Kaggle train v1 push

### push 前コスト確認

- Runtime: CPU (`enable_gpu=false`)
- Feature cache variant 数: 1 (`pixiux_likpf_hard_window_pct50_public_replay`)
- hard-window threshold: `typewell_pct >= 0.50`
- PF seeds: 128
- PF particles: 500
- target wells: all horizontal wells (`max_wells: null`)
- LightGBM config 数: 0
- fold 数: 0
- 合計 booster 数: 0
- control / parent 再学習: なし
- inference / submit: なし

```bash
make push-kaggle-train EXP=exp192_typewell_late_range_hard_window_pct50_full_cache_replacement
```

- result: PASS
- Kaggle kernel version: v1
- URL: https://www.kaggle.com/code/kentookumura/exp192-typewell-hard-window-pct50-train

```bash
kaggle kernels pull kentookumura/exp192-typewell-hard-window-pct50-train -p /tmp/kaggle-pull/exp192-typewell-hard-window-pct50-train-v1 -m
kaggle kernels status kentookumura/exp192-typewell-hard-window-pct50-train
kaggle kernels logs kentookumura/exp192-typewell-hard-window-pct50-train
```

- pull: PASS
- metadata `id_no`: 125911807
- metadata `enable_gpu`: false
- metadata `enable_internet`: false
- metadata `competition_sources`: `rogii-wellbore-geology-prediction`
- status: `KernelWorkerStatus.RUNNING`
- logs: empty while running, consistent with previous Kaggle CLI behavior in this environment.

```bash
timeout 600 kaggle kernels logs -f --interval 30 kentookumura/exp192-typewell-hard-window-pct50-train
```

- result: user requested to stop monitoring before timeout.
- local `logs -f` process was interrupted.
- Kaggle kernel v1 itself was not stopped and should continue running on Kaggle.

## 2026-07-04 Kaggle train v1 完了確認

```bash
kaggle kernels status kentookumura/exp192-typewell-hard-window-pct50-train
kaggle kernels logs kentookumura/exp192-typewell-hard-window-pct50-train
kaggle kernels pull kentookumura/exp192-typewell-hard-window-pct50-train -p /tmp/kaggle-pull/exp192-typewell-hard-window-pct50-train-v1-complete -m
```

- status: `KernelWorkerStatus.COMPLETE`
- kernel version: v1
- metadata `id_no`: 125911807
- metadata `enable_gpu`: false
- metadata `enable_internet`: false
- logs summary:
  - status: `train_feature_cache_completed`
  - route: `pf_beam`
  - variant: `pixiux_likpf_hard_window_pct50_public_replay`
  - hard window: `pct50`, `min_typewell_pct=0.5`, `max_typewell_pct=1.0`
  - rows / wells / features: 3,783,989 / 773 / 196
  - feature generation elapsed: 11,643.326 sec
  - total elapsed: 13,275.591 sec
  - raw gzip SHA: `1040d7d3b9254b5a36d2a3f7fd526ae28e3ddd5b29059926b44bbe9d84436e6a`

## Kaggle output 取得と SHA 検証

Small artifacts:

```bash
kaggle kernels output kentookumura/exp192-typewell-hard-window-pct50-train \
  -p /tmp/kaggle-output/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement/train_v1_small \
  --file-pattern '.*(summary\.json|feature_schema\.csv)$' -o
```

Full gzip は Kaggle signed URL を streaming 取得した。通常の CLI full output 取得は大きな gzip を一度に持つため避けた。

- downloaded gzip bytes: 2,066,723,848
- gzip test: PASS
- raw gzip SHA: `1040d7d3b9254b5a36d2a3f7fd526ae28e3ddd5b29059926b44bbe9d84436e6a`
- decompressed SHA: `a86dff450b108e4481208a5f5699f8624eaf736cb6eb6aa735d39b4044c6f0e1`
- decompressed bytes: 7,428,711,117
- line count: 3,783,990
- data rows: 3,783,989
- header columns: 199
- schema SHA: `ad59c4f998433bdd9105effa081ec620fd7f6ced5f9dc32d68b97d4c757f6ed0`
- summary SHA: `f7c244621a197068b948e6753076f898c4171ed9a6b1086745d93567b4eb6b50`

Artifacts copied to:

- `experiments/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement/artifacts/`

## exp072 direct PF/Beam 比較

Comparison input:

- exp072: `/tmp/kaggle-output/exp072_exp063_full_replay_feature_cache/train_v2_stream/artifacts/exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz`
- exp192: `artifacts/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement_full_replay_cache_pixiux_likpf_hard_window_pct50_public_replay_train_features.csv.gz`

Validation:

- `rows_checked`: 3,783,989
- `unique_wells`: 773
- `id_mismatches`: 0
- `missing_typewell_range_wells`: 0

Overall direct TVT metrics:

| candidate | exp072 RMSE | exp192 RMSE | delta RMSE | exp072 MAE | exp192 MAE | delta within10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pf_ancc` | 14.493061 | 13.821178 | -0.671884 | 8.921569 | 8.053430 | +0.042730 |
| `pf_z` | 17.788174 | 19.705112 | +1.916938 | 10.677493 | 11.543506 | -0.006206 |
| `beam_mean` | 15.774328 | 15.677016 | -0.097311 | 10.898586 | 10.926740 | -0.002106 |
| `beam_sm5` | 16.313542 | 16.152930 | -0.160612 | 11.300928 | 11.317035 | -0.002140 |
| `likpf_mean` | 11.594898 | 11.544812 | -0.050086 | 7.067633 | 7.063503 | +0.001386 |

By-well guard:

- `likpf_mean`: 396 wells improved / 375 worsened / 2 same、max regression +30.263337 RMSE (`78a4a386`)
- `pf_ancc`: 428 improved / 344 worsened / 1 same、max regression +49.512070 RMSE (`fef8af96`)
- `pf_z`: 351 improved / 421 worsened / 1 same、max regression +39.362221 RMSE (`a4719920`)

Subset guard:

- `likpf_mean` true typewell pct `<0.50`: RMSE 7.784425 -> 33.538635、delta +25.754210
- `beam_mean` true typewell pct `<0.50`: RMSE 8.471420 -> 40.974027、delta +32.502607
- `pf_ancc` true typewell pct `<0.50`: RMSE 17.008325 -> 26.935806、delta +9.927481
- `pf_z` true typewell pct `<0.50`: RMSE 8.286518 -> 7.812296、delta -0.474222

Comparison artifacts:

- `artifacts/exp192_vs_exp072_overall_metrics.csv`
- `artifacts/exp192_vs_exp072_distance_bucket_metrics.csv`
- `artifacts/exp192_vs_exp072_true_typewell_pct_metrics.csv`
- `artifacts/exp192_vs_exp072_by_well_delta.csv`
- `artifacts/exp192_vs_exp072_summary.json`

Comparison SHA:

- overall metrics: `f7250d78e4637dba7a102bfc5d5f8f37ddf58c546a5db888b4dfbc3f7f2c332d`
- distance bucket metrics: `546001218856dba3f01007fdc81b9b023bf838323b24dd07f5d010ef760ae525`
- true typewell pct metrics: `a11f952052e3898c554eed93362e3c538f6f03f5b5415669c95e65a1b581c9f7`
- by-well delta: `527b5d9d065e2448620e472a3e365867b511a5ad73782fb1106c5489e534d87a`
- comparison summary: `a051bd2ef1dc41ced8a43908bd88e33e52551886ef806757d8fc084c2eac1b76`

## 判断

hard-window pct50 は `likpf_mean` を 11.594898 -> 11.544812、`pf_ancc` を 14.493061 -> 13.821178 へ改善したため、direct cache candidate としては支持する。

ただし `pf_z` は 17.788174 -> 19.705112 へ大きく悪化し、true typewell pct `<0.50` subset の regression も大きい。したがって PF/Beam route の direct inference / submit には進めない。続ける場合は downstream ML replacement-only 学習で、`pf_z` 悪化と early-range exception を吸収できるか確認する。
