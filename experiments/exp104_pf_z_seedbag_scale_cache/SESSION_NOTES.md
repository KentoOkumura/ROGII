# exp104_pf_z_seedbag_scale_cache セッションノート

## 目的

exp100 の best `pf_z_xy_slope` は単発 PF 候補で、exp072 `likpf_mean` は seed bagging された likelihood-PF ensemble だった。exp104 では `pf_z_xy_slope` を 128 seed の pf_z seedbag / scale cache として再生成し、exp072 保存済み `pf_z` / `likpf_*` と同じ candidate metrics で比較する。

ユーザー指定に従い、占有済みの `exp072` は実験名の接頭辞に使わず、最新の空き番号 `exp104` を接頭辞にした。

## 現在の状態

- Route: `pf_beam`
- 状態: `completed_train_side_audit_rejected`
- CV: Kaggle train v1 完了
- LB: なし
- 提出: なし

## 実装内容

- `exp104_pf_z_seedbag_scale_cache` を `exp103_pf_z_xy_likpf_ensemble_parity` から作成した。
- `.steering/20260622-exp104-pf-z-seedbag-scale-cache/` に requirements / design / tasklist を記入した。
- `pf_z_seedbag_scale_cache.py` を実装対象にし、出力 prefix を `exp104_pf_z_seedbag_scale_cache` に変更した。
- `model.pf_z_seedbag` から粒子数、seed 数、scale を読む。
- 生成候補名を `pf_z_seedbag_mean` / `pf_z_seedbag_scale_3/5/8/12` に統一した。
- exp072 cache の既存候補 `exp072_pf_z` / `exp072_likpf_mean` / 存在する `exp072_likpf_scale_*` と同じ行集合で比較する。

## 生成候補

exp072 cache から読む baseline:

- `exp072_pf_z`
- `exp072_likpf_mean`
- `exp072_likpf_scale_3/5/8/12` は cache に存在する場合だけ読む

exp104 で生成する pf_z seedbag:

- `pf_z_seedbag_mean`
- `pf_z_seedbag_scale_3`
- `pf_z_seedbag_scale_5`
- `pf_z_seedbag_scale_8`
- `pf_z_seedbag_scale_12`

## コマンドログ

### 2026-06-22 JST 実装

```bash
make new-steering EXP=exp104_pf_z_seedbag_scale_cache
make new-exp EXP=exp104_pf_z_seedbag_scale_cache SOURCE=experiments/exp103_pf_z_xy_likpf_ensemble_parity
```

`task` はこの環境になかったため `make` を使用した。

設計判断:

- `pos = TVT + Z` の state を使い、prediction は `pos - Z` にする。
- prefix の `d(TVT_input + Z)/dMD ~ dZ/dMD + dXY/dMD` で rate prior を fitting する。
- rate likelihood は粒子重みと seed log likelihood の両方に入れる。
- exp072 baseline は再生成せず、Kaggle input の exp072 train cache から読む。
- ローカルには exp072 cache 本体がないため、full 比較は Kaggle train で実行する。

### 予定

```bash
python3 -m py_compile experiments/exp104_pf_z_seedbag_scale_cache/pf_z_seedbag_scale_cache.py experiments/exp104_pf_z_seedbag_scale_cache/settings.py
python3 -m json.tool experiments/exp104_pf_z_seedbag_scale_cache/exp104_pf_z_seedbag_scale_cache_train.ipynb
python3 -m json.tool experiments/exp104_pf_z_seedbag_scale_cache/exp104_pf_z_seedbag_scale_cache_inference.ipynb
uv run ruff check experiments/exp104_pf_z_seedbag_scale_cache/pf_z_seedbag_scale_cache.py experiments/exp104_pf_z_seedbag_scale_cache/settings.py
make validate-exp EXP=exp104_pf_z_seedbag_scale_cache
make prepare-kaggle-notebooks EXP=exp104_pf_z_seedbag_scale_cache EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp104-pf-z-seedbag-scale-train --title 'exp104 pf z seedbag scale train' --run-on-push --strict"
make push-kaggle-train EXP=exp104_pf_z_seedbag_scale_cache
```

Kaggle train 完了後に `candidate_metrics.csv`、summary、生成物 SHA、exp072 候補との比較結果を追記する。

### 2026-06-22 JST validation / package / push

```bash
python3 -m py_compile experiments/exp104_pf_z_seedbag_scale_cache/pf_z_seedbag_scale_cache.py experiments/exp104_pf_z_seedbag_scale_cache/settings.py
python3 -m json.tool experiments/exp104_pf_z_seedbag_scale_cache/exp104_pf_z_seedbag_scale_cache_train.ipynb
python3 -m json.tool experiments/exp104_pf_z_seedbag_scale_cache/exp104_pf_z_seedbag_scale_cache_inference.ipynb
uv run ruff check experiments/exp104_pf_z_seedbag_scale_cache/pf_z_seedbag_scale_cache.py experiments/exp104_pf_z_seedbag_scale_cache/settings.py
make validate-exp EXP=exp104_pf_z_seedbag_scale_cache
make prepare-kaggle-notebooks EXP=exp104_pf_z_seedbag_scale_cache EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp104-pf-z-seedbag-scale-train --title 'exp104 pf z seedbag scale train' --run-on-push --strict"
make push-kaggle-train EXP=exp104_pf_z_seedbag_scale_cache
```

結果:

- `py_compile`: PASS
- train / inference notebook JSON: PASS
- `ruff check`: PASS
- `validate-exp`: PASS
- Kaggle train package: `experiments/exp104_pf_z_seedbag_scale_cache/kaggle/train`
- kernel id: `kentookumura/exp104-pf-z-seedbag-scale-train`
- version: 1
- URL: https://www.kaggle.com/code/kentookumura/exp104-pf-z-seedbag-scale-train
- metadata: CPU / internet false / competition source `rogii-wellbore-geology-prediction` / kernel source `kentookumura/exp072-exp063-full-replay-feature-cache-train`

### 2026-06-22 JST Kaggle train v1 完了

```bash
kaggle kernels logs kentookumura/exp104-pf-z-seedbag-scale-train
kaggle kernels output kentookumura/exp104-pf-z-seedbag-scale-train -p experiments/exp104_pf_z_seedbag_scale_cache/kaggle/output/train_v1
timeout 240 kaggle kernels output kentookumura/exp104-pf-z-seedbag-scale-train -p experiments/exp104_pf_z_seedbag_scale_cache/kaggle/output/train_v1
```

結果:

- Kernel: `KernelWorkerStatus.COMPLETE`
- runtime: 41,132.91 sec
- rows: 3,783,989
- wells: 773
- output: `experiments/exp104_pf_z_seedbag_scale_cache/kaggle/output/train_v1`
- exp072 cache columns: `pf_z`, `likpf_mean_d`
- exp072 `likpf_scale_3/5/8/12` は cache に存在せず比較対象外

`candidate_long.csv.gz` が大きく、1 回目の `kaggle kernels output` は途中で中断した。中断直前に `candidate_metrics.csv` までは取得済み。2 回目は既存ファイルを skip し、残りの `candidate_wide.csv.gz`、`pf_z_seedbag_quality.csv`、`summary.json`、log を取得した。

候補比較:

| candidate | RMSE | MAE | within10 | bias |
| --- | ---: | ---: | ---: | ---: |
| `exp072_likpf_mean` | 11.594898 | 7.067633 | 0.772807 | -1.099423 |
| `pf_z_seedbag_scale_12` | 14.145856 | 8.776381 | 0.695260 | -0.953733 |
| `pf_z_seedbag_scale_8` | 14.171680 | 8.776667 | 0.694138 | -0.892882 |
| `pf_z_seedbag_scale_5` | 14.178127 | 8.768430 | 0.693555 | -0.819223 |
| `pf_z_seedbag_scale_3` | 14.215698 | 8.777034 | 0.692961 | -0.747068 |
| `pf_z_seedbag_mean` | 14.587060 | 9.664454 | 0.651736 | -1.047900 |
| `exp072_pf_z` | 17.788171 | 10.677487 | 0.647668 | -0.934560 |

判定:

- best seedbag は `pf_z_seedbag_scale_12`。
- exp072 plain `pf_z` より RMSE -3.642315 改善。
- exp072 `likpf_mean` より RMSE +2.550958 悪化。
- 直接推論移植、提出、exp073/exp092 系への add-only feature 化はしない。

主な SHA:

- exp072 cache raw SHA: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
- exp072 cache decompressed SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- candidate metrics SHA: `b3343b394a7acd356a547ba134e70cd958aacd38731c2f09f996f4a55efef19b`
- candidate wide decompressed SHA: `3d27bb3c580f5c2df3542c9e6dcccb981c20b37f81a2a6a10cda17153e134fd2`
- candidate long decompressed SHA: `17e8a58595a4fc2fce62d7a30634cf3da48e41a64d7f11e93f10f131ae2851f6`
- summary SHA: `f5339a5fdb5855b3b15ad4e349f3b98695f384d762dffb8ec60483803fdb7fb3`
