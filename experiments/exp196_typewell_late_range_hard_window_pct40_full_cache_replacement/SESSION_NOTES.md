# exp196_typewell_late_range_hard_window_pct40_full_cache_replacement セッションノート

## 2026-07-04 実装

- ユーザー依頼により `typewell_late_range_hard_window_pct40_full_cache_replacement` backlog の実装を開始。
- `docs/legacy/steering/20260704-exp196-typewell-late-range-hard-window-pct40-full-cache-replacement/` を作成。
- `experiments/exp196_typewell_late_range_hard_window_pct40_full_cache_replacement/` を exp192 から作成。
- Route: `pf_beam`
- GPU 学習: なし。CPU-only full replay train feature cache generation。
- feature cache variant 数: 1 (`pixiux_likpf_hard_window_pct40_public_replay`)
- hard-window threshold: `typewell_pct >= 0.40` のみ。`0.30/0.60/0.70` grid なし。
- LightGBM config 数: 0
- fold 数: 0
- 合計 booster 数: 0
- control / parent 再学習: なし
- inference / submit: なし

```bash
make push-kaggle-train EXP=exp196_typewell_late_range_hard_window_pct40_full_cache_replacement
```

- result: PASS
- Kaggle kernel version: v1
- URL: https://www.kaggle.com/code/kentookumura/exp196-typewell-hard-window-pct40-train

```bash
kaggle kernels pull kentookumura/exp196-typewell-hard-window-pct40-train -p /tmp/kaggle-pull/exp196-typewell-hard-window-pct40-train-v1 -m
kaggle kernels status kentookumura/exp196-typewell-hard-window-pct40-train
kaggle kernels logs kentookumura/exp196-typewell-hard-window-pct40-train
```

- pull: PASS
- metadata `id_no`: 125944394
- metadata `enable_gpu`: false
- metadata `enable_internet`: false
- metadata `competition_sources`: `rogii-wellbore-geology-prediction`
- status: `KernelWorkerStatus.RUNNING`
- logs: empty while running, consistent with previous Kaggle CLI behavior in this environment.

```bash
uv run python scripts/update_experiment_summary.py
uv run python scripts/validate_experiment.py --experiment exp196_typewell_late_range_hard_window_pct40_full_cache_replacement
kaggle kernels status kentookumura/exp196-typewell-hard-window-pct40-train
```

- summary update: PASS
- validate-exp: PASS
- final status check in this turn: `KernelWorkerStatus.RUNNING`

## 実装内容

- `hard_window_public_replay.py` は exp192 の corrected hard-window full replay implementation から派生。
- soft prior は `no_prior` / penalty 0 のまま維持。
- `apply_typewell_hard_window()` は raw typewell の元 finite TVT min/max から `typewell_pct` を計算し、`0.40 <= typewell_pct <= 1.00` の rows だけを残す。
- `feature_cache.py` は exp196 用 output prefix / pct40 variant を出力する。
- train notebook 起点 `.py` は raw input check、hard-window contract、feature cache generation、generated artifacts のセル構成を維持。
- inference notebook 起点 `.py` は train-cache-only no-op。

## 再現性メモ

- seed policy: stable SHA seed per well / split / feature family。
- stochastic components: PF particle propagation / resampling / likelihood-PF seed ensemble。
- Beam: deterministic。
- CPU/GPU runtime: Kaggle CPU、`enable_gpu=false`。
- deterministic anchor: false。train feature cache only で submission anchor ではない。
- gzip 生成物は raw gzip SHA と decompressed content SHA を分け、decompressed content SHA を主証拠にする。

## コマンドログ

```bash
make new-steering EXP=exp196_typewell_late_range_hard_window_pct40_full_cache_replacement
make new-exp EXP=exp196_typewell_late_range_hard_window_pct40_full_cache_replacement SOURCE=experiments/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement
```

- result: PASS

## 次のアクション

- Kaggle train v1 を push する場合は、上記のコスト確認を再掲してから実行する。

## 2026-07-04 ローカル検証と Kaggle package 準備

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp196_typewell_late_range_hard_window_pct40_full_cache_replacement/exp196_typewell_late_range_hard_window_pct40_full_cache_replacement_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp196_typewell_late_range_hard_window_pct40_full_cache_replacement/exp196_typewell_late_range_hard_window_pct40_full_cache_replacement_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp196_typewell_late_range_hard_window_pct40_full_cache_replacement/exp196_typewell_late_range_hard_window_pct40_full_cache_replacement_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp196_typewell_late_range_hard_window_pct40_full_cache_replacement/exp196_typewell_late_range_hard_window_pct40_full_cache_replacement_inference.py
```

- result: PASS

```bash
.venv/bin/python -m py_compile \
  experiments/exp196_typewell_late_range_hard_window_pct40_full_cache_replacement/hard_window_public_replay.py \
  experiments/exp196_typewell_late_range_hard_window_pct40_full_cache_replacement/feature_cache.py \
  experiments/exp196_typewell_late_range_hard_window_pct40_full_cache_replacement/settings.py \
  experiments/exp196_typewell_late_range_hard_window_pct40_full_cache_replacement/exp196_typewell_late_range_hard_window_pct40_full_cache_replacement_train.py \
  experiments/exp196_typewell_late_range_hard_window_pct40_full_cache_replacement/exp196_typewell_late_range_hard_window_pct40_full_cache_replacement_inference.py

.venv/bin/ruff check \
  experiments/exp196_typewell_late_range_hard_window_pct40_full_cache_replacement/hard_window_public_replay.py \
  experiments/exp196_typewell_late_range_hard_window_pct40_full_cache_replacement/feature_cache.py \
  experiments/exp196_typewell_late_range_hard_window_pct40_full_cache_replacement/settings.py \
  experiments/exp196_typewell_late_range_hard_window_pct40_full_cache_replacement/exp196_typewell_late_range_hard_window_pct40_full_cache_replacement_train.py \
  experiments/exp196_typewell_late_range_hard_window_pct40_full_cache_replacement/exp196_typewell_late_range_hard_window_pct40_full_cache_replacement_inference.py --select F821
```

- result: PASS

```bash
uv run python scripts/validate_experiment.py --experiment exp196_typewell_late_range_hard_window_pct40_full_cache_replacement
```

- first result: FAIL。`README.md` に `## 所見` が不足していた。
- fix: `README.md` に未実行時点の所見を追加。
- rerun result: PASS (`experiment validation passed (strict)`)

```bash
make prepare-kaggle-notebooks EXP=exp196_typewell_late_range_hard_window_pct40_full_cache_replacement EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp196-typewell-hard-window-pct40-train --title 'exp196 typewell hard window pct40 train' --run-on-push --strict"
```

- result: PASS
- output: `experiments/exp196_typewell_late_range_hard_window_pct40_full_cache_replacement/kaggle/train`
- kernel id: `kentookumura/exp196-typewell-hard-window-pct40-train`
- title: `exp196 typewell hard window pct40 train`
- metadata `enable_gpu`: false
- metadata `enable_internet`: false
- competition source: `rogii-wellbore-geology-prediction`
- kernel sources: none
- package config confirmed `model.hard_window.name: pct40`, `min_typewell_pct: 0.40`, `variant: pixiux_likpf_hard_window_pct40_public_replay`
- notebook support manifest SHA matched loose package files:
  - `config.yaml`: `f6e2d03e92198e6568434bf3d6a99026157cd04e4e4b28d3909e5db4fe53b9d7`
  - train script: `981c48b720f0da62e5b377d674a4a71319580df42fde6062d2a05227849c2862`
  - `feature_cache.py`: `ed0754a560448cf1529d5e2787b15e489003cd9565f69d303ec70a57771e56f5`
  - `hard_window_public_replay.py`: `6192719abe156193cb5fc813b45a39e6940f232384254737e38cca23162cf8a3`

```bash
uv run python scripts/update_experiment_summary.py
uv run python scripts/validate_experiment.py --experiment exp196_typewell_late_range_hard_window_pct40_full_cache_replacement
```

- result: PASS
- `experiment_summary.md` に exp196 行を追加。
- `KAGGLE_DIRECTION.md` は exp196 実装済み / Kaggle train 未実行として更新。結果待ちのため backlog 完了扱いにはしていない。

## 2026-07-04 Kaggle train v1 push

### push 前コスト確認

- Runtime: CPU (`enable_gpu=false`)
- Feature cache variant 数: 1 (`pixiux_likpf_hard_window_pct40_public_replay`)
- hard-window threshold: `typewell_pct >= 0.40`
- PF seeds: 128
- PF particles: 500
- target wells: all horizontal wells (`max_wells: null`)
- LightGBM config 数: 0
- fold 数: 0
- 合計 booster 数: 0
- control / parent 再学習: なし
- inference / submit: なし

## 2026-07-05 Kaggle train v1 完了確認

```bash
kaggle kernels status kentookumura/exp196-typewell-hard-window-pct40-train
kaggle kernels logs kentookumura/exp196-typewell-hard-window-pct40-train
```

- status: `KernelWorkerStatus.COMPLETE`
- rows / wells / features: 3,783,989 / 773 / 196
- feature generation seconds: 7,497.889
- elapsed seconds: 8,616.007
- raw gzip SHA: `7b1f51b1c4de16bbff59c9a0c1bd015fc3b6d6152c32a3a93b2f5a694a37576b`
- output:
  - `exp196_typewell_late_range_hard_window_pct40_full_cache_replacement_full_replay_cache_pixiux_likpf_hard_window_pct40_public_replay_train_features.csv.gz`
  - `exp196_typewell_late_range_hard_window_pct40_full_cache_replacement_full_replay_cache_feature_schema.csv`
  - `exp196_typewell_late_range_hard_window_pct40_full_cache_replacement_full_replay_cache_summary.json`

```bash
kaggle kernels output kentookumura/exp196-typewell-hard-window-pct40-train \
  --file-pattern '.*(summary\.json|feature_schema\.csv)$' \
  -p /tmp/kaggle-output/exp196-typewell-hard-window-pct40-train-v1
```

- small output fetch: PASS
- summary SHA: `8b0843b5fc617faf16926bf2f5415b332f0993c0e4890da82f800e093b832eeb`
- schema SHA: `b1946bee3db4cec5311f1ffd4a47a4e1db0f0635e153b3882871b7c34ef5e9e5`
- log SHA: `38eaed8def8028ab08e680bfa94eec6fc57d2d8e6fa74ec8c71701e21a9b4feb`

Kaggle CLI の `kernels output --file-pattern '.*train_features\.csv\.gz$'` は exit code 137 で 0-byte tmp を残したため、Kaggle Python API で signed URL を取得し、`requests.iter_content` で直接 stream download した。

- downloaded bytes: 2,082,017,667
- raw gzip SHA match: `7b1f51b1c4de16bbff59c9a0c1bd015fc3b6d6152c32a3a93b2f5a694a37576b`
- decompressed SHA: `106cdfb266f93a0e45f25b281d3238c1fab0a24a84dac4c23187044022b5127e`
- decompressed bytes: 7,429,469,755
- line_count: 3,783,990
- data_rows: 3,783,989
- header_columns: 199

## 2026-07-05 direct PF/Beam comparison

同一 row で exp072 full replay cache と exp192 pct50 hard-window cache に直接比較した。

- rows_checked: 3,783,989
- chunks: 19
- unique_wells: 773
- id_mismatches: 0
- missing_typewell_range_wells: 0

### vs exp072

| candidate | exp072 RMSE | exp196 RMSE | delta RMSE | delta within10 |
| --- | ---: | ---: | ---: | ---: |
| `pf_ancc` | 14.493061 | 14.020904 | -0.472157 | +0.027218 |
| `pf_z` | 17.788174 | 18.834133 | +1.045959 | -0.008121 |
| `beam_mean` | 15.774328 | 15.711042 | -0.063285 | +0.000035 |
| `beam_sm5` | 16.313542 | 16.185190 | -0.128352 | +0.000050 |
| `likpf_mean` | 11.594898 | 11.576062 | -0.018835 | +0.001023 |

true typewell pct `<0.50` bucket は exp072 比で `likpf_mean` -2.635833 RMSE、`beam_mean` -1.027486、`pf_ancc` -11.853791 と改善。`pf_z` は +0.303997 と小悪化。

by-well `likpf_mean`: 364 wells 改善 / 405 wells 悪化 / 4 same、max regression +10.213776 RMSE (`8f201368`)。

### vs exp192 pct50

| candidate | exp192 RMSE | exp196 RMSE | delta RMSE | delta within10 |
| --- | ---: | ---: | ---: | ---: |
| `pf_ancc` | 13.821178 | 14.020904 | +0.199726 | -0.015511 |
| `pf_z` | 19.705112 | 18.834133 | -0.870979 | -0.001915 |
| `beam_mean` | 15.677016 | 15.711042 | +0.034026 | +0.002141 |
| `beam_sm5` | 16.152930 | 16.185190 | +0.032260 | +0.002191 |
| `likpf_mean` | 11.544812 | 11.576062 | +0.031251 | -0.000363 |

true typewell pct `<0.50` bucket は pct50 比で `likpf_mean` -28.390042 RMSE、`beam_mean` -33.530093、`pf_ancc` -21.781272 と大幅改善。一方、`0.50-0.70` bucket は pct50 比で `likpf_mean` +2.380595、`beam_mean` +2.500450、`pf_ancc` +4.023544 と悪化した。

by-well `likpf_mean`: 370 wells 改善 / 396 wells 悪化 / 7 same、max regression +19.695207 RMSE (`ba48188d`)。

### 生成 comparison artifacts

- `artifacts/exp196_vs_exp072_overall_metrics.csv` SHA `21b1b75e1664535d5aeb3cd161199119f2ff2e6913082ba31868585de1ed4d29`
- `artifacts/exp196_vs_exp072_distance_bucket_metrics.csv` SHA `677578b3527f94a6033ad823acafeb9f715fc26541749c743a455a61b2ada59d`
- `artifacts/exp196_vs_exp072_true_typewell_pct_metrics.csv` SHA `b6ada72b0224665541d93283583d4b7c354c8a2b42325f7b606061c20ceb4912`
- `artifacts/exp196_vs_exp072_by_well_delta.csv` SHA `f08928cb2cf9e81b14239ada268dbb9b9dcae67123ddb1b66a44900fc7efeff0`
- `artifacts/exp196_vs_exp192_pct50_overall_metrics.csv` SHA `b61f2f2f49e3e2eff204b845ac3531468c19751b9e33db23ab1948a65af87ef1`
- `artifacts/exp196_vs_exp192_pct50_distance_bucket_metrics.csv` SHA `d5ea457408437b6de9b685217134c8eec6404cd7b462bc4d6d742974a38a108a`
- `artifacts/exp196_vs_exp192_pct50_true_typewell_pct_metrics.csv` SHA `b1420a278bb3890088c4ce24fe814886f8c9a80502bda9b541ee9e8da21f2de5`
- `artifacts/exp196_vs_exp192_pct50_by_well_delta.csv` SHA `abe23b35280314069d413f221352b3063ca0ceecd14ebd27d512acfb8b42de47`
- `artifacts/exp196_direct_pfbeam_comparison_summary.json` SHA `d5f595b0adcaa68ab164e6b9eb865f8e56f73ecf1c62fced301761440deec106`

## 判断

`typewell_late_range_hard_window_pct40_full_cache_replacement` backlog は完了。pct40 は pct50 の early-range exception と `pf_z` regression を緩めたが、global `likpf_mean` / `pf_ancc` / Beam の改善幅は pct50 より弱い。direct PF/Beam inference / submit には進めない。次に使う場合は downstream ML replacement-only で pct40 と pct50 を同条件比較する。
