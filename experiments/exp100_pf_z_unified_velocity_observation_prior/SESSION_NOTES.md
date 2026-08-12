# exp100_pf_z_unified_velocity_observation_prior セッションノート

## 目的

既存 `pf_z` の Z-aware velocity prior を control とし、XY slope、prefix TVT slope、GR affine calibration を弱い observation / velocity prior として粒子重みに追加した場合の train pseudo-tail 改善を Stage 1 ablation として確認する。

## 現在の状態

- Route: pf_beam
- 状態: implemented_not_run
- CV: 未実行
- LB: なし
- 提出: なし

## コマンドログ

### 2026-06-21 JST 実装

```bash
make new-steering EXP=exp100_pf_z_unified_velocity_observation_prior
make new-exp EXP=exp100_pf_z_unified_velocity_observation_prior SOURCE=experiments/exp099_pf_multi_observation_likelihood_probe
```

実装内容:

- `docs/legacy/steering/20260621-exp100-pf-z-unified-velocity-observation-prior/` を作成し、requirements / design / tasklist を記入した。
- `config.yaml` を `pf_beam` route の Stage 1 ablation 用に更新した。
- `pf_z_unified_velocity_observation_prior.py` を追加した。
- `settings.py` の experiment name を exp100 に更新した。
- train / inference notebook 名を exp100 に変更した。

実装済み variant:

- `pf_z_control`
- `pf_z_xy_slope`
- `pf_z_prefix_slope`
- `pf_z_gr_calibrated`
- `pf_z_xy_plus_prefix`
- `pf_z_xy_plus_gr_calibrated`
- `pf_z_prefix_plus_gr_calibrated`
- `pf_z_xy_prefix_gr_calibrated`

### 予定

```bash
python3 -m py_compile experiments/exp100_pf_z_unified_velocity_observation_prior/pf_z_unified_velocity_observation_prior.py experiments/exp100_pf_z_unified_velocity_observation_prior/settings.py
python3 -m json.tool experiments/exp100_pf_z_unified_velocity_observation_prior/exp100_pf_z_unified_velocity_observation_prior_train.ipynb >/tmp/exp100_train_nb.json
uv run ruff check experiments/exp100_pf_z_unified_velocity_observation_prior/pf_z_unified_velocity_observation_prior.py experiments/exp100_pf_z_unified_velocity_observation_prior/settings.py
make validate-exp EXP=exp100_pf_z_unified_velocity_observation_prior
make prepare-kaggle-notebooks EXP=exp100_pf_z_unified_velocity_observation_prior EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp100-pf-z-unified-prior-train --title 'exp100 pf z unified prior train' --run-on-push --strict"
```

Kaggle train 完了後に `variant_metrics`、`summary.json`、生成物 SHA、解釈を追記する。

### 2026-06-21 JST validation / package

```bash
python3 -m py_compile experiments/exp100_pf_z_unified_velocity_observation_prior/pf_z_unified_velocity_observation_prior.py experiments/exp100_pf_z_unified_velocity_observation_prior/settings.py
python3 -m json.tool experiments/exp100_pf_z_unified_velocity_observation_prior/exp100_pf_z_unified_velocity_observation_prior_train.ipynb
python3 -m json.tool experiments/exp100_pf_z_unified_velocity_observation_prior/exp100_pf_z_unified_velocity_observation_prior_inference.ipynb
uv run ruff check experiments/exp100_pf_z_unified_velocity_observation_prior/pf_z_unified_velocity_observation_prior.py experiments/exp100_pf_z_unified_velocity_observation_prior/settings.py
make validate-exp EXP=exp100_pf_z_unified_velocity_observation_prior
make prepare-kaggle-notebooks EXP=exp100_pf_z_unified_velocity_observation_prior EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp100-pf-z-unified-prior-train --title 'exp100 pf z unified prior train' --run-on-push --strict"
```

結果:

- `py_compile`: PASS
- train notebook JSON: PASS
- inference notebook JSON: PASS
- `ruff check`: 1 回目は行長 1 件と未使用変数 1 件で失敗。修正後 PASS。
- `validate-exp`: PASS
- Kaggle train package: `experiments/exp100_pf_z_unified_velocity_observation_prior/kaggle/train`
- kernel id: `kentookumura/exp100-pf-z-unified-prior-train`
- title: `exp100 pf z unified prior train`
- metadata: GPU false / internet false / run_on_push true / competition source `rogii-wellbore-geology-prediction`

補足:

- ローカル system Python と `.venv` に `numba` が入っていないため、Numba 実行 smoke は未実施。Kaggle runtime は exp072/exp091 の PF/Beam notebook と同じく `numba` 前提で実行する。

### 2026-06-21 JST Numba wrapper hardening

`_pf_z_unified_seeded` を `*args` wrapper から明示引数 wrapper に修正した。既存 exp098 / exp091 と同じ seed wrapper 形に寄せ、Numba JIT 上の可変引数解釈に依存しないようにした。

再実行:

```bash
uv run ruff check experiments/exp100_pf_z_unified_velocity_observation_prior/pf_z_unified_velocity_observation_prior.py experiments/exp100_pf_z_unified_velocity_observation_prior/settings.py
python3 -m py_compile experiments/exp100_pf_z_unified_velocity_observation_prior/pf_z_unified_velocity_observation_prior.py experiments/exp100_pf_z_unified_velocity_observation_prior/settings.py
make validate-exp EXP=exp100_pf_z_unified_velocity_observation_prior
make prepare-kaggle-notebooks EXP=exp100_pf_z_unified_velocity_observation_prior EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp100-pf-z-unified-prior-train --title 'exp100 pf z unified prior train' --run-on-push --strict"
```

結果:

- ruff: PASS
- py_compile: PASS
- validate-exp: PASS
- Kaggle train package: 再生成済み

### 2026-06-21 JST Kaggle train v2 push

```bash
kaggle kernels push -p experiments/exp100_pf_z_unified_velocity_observation_prior/kaggle/train
kaggle kernels pull kentookumura/exp100-pf-z-unified-prior-train -p /tmp/kaggle-pull/exp100-pf-z-unified-prior-train -m
kaggle kernels status kentookumura/exp100-pf-z-unified-prior-train
kaggle kernels logs kentookumura/exp100-pf-z-unified-prior-train
```

結果:

- Kernel version 2 push 成功。
- URL: https://www.kaggle.com/code/kentookumura/exp100-pf-z-unified-prior-train
- `kaggle kernels pull ... -m`: 成功。
- 初期 status: `KernelWorkerStatus.RUNNING`
- 初期 logs: warning 以外は空。
- ユーザー希望により長時間監視はしない。完了連絡後に logs / output を取得する。

### 2026-06-21 JST Kaggle train v2 output 取得

ユーザー完了連絡後に logs / output を取得した。

```bash
kaggle kernels status kentookumura/exp100-pf-z-unified-prior-train
kaggle kernels logs kentookumura/exp100-pf-z-unified-prior-train
kaggle kernels output kentookumura/exp100-pf-z-unified-prior-train -p experiments/exp100_pf_z_unified_velocity_observation_prior/kaggle/output/train_v2
```

結果:

- status: `KernelWorkerStatus.COMPLETE`
- runtime: 3,986.42 sec
- rows: 3,783,989
- wells: 773
- output: `experiments/exp100_pf_z_unified_velocity_observation_prior/kaggle/output/train_v2`
- best: `pf_z_xy_slope` RMSE 29.404162 / MAE 10.959212 / within10 0.655593
- control: `pf_z_control` RMSE 163.301551 / MAE 92.155466 / within10 0.179258
- best-control delta: RMSE -133.897390 / within10 +0.476334

Variant ranking:

| variant | RMSE | within10 |
| --- | ---: | ---: |
| `pf_z_xy_slope` | 29.404162 | 0.655593 |
| `pf_z_xy_plus_prefix` | 30.157085 | 0.439449 |
| `pf_z_xy_plus_gr_calibrated` | 31.700123 | 0.637807 |
| `pf_z_xy_prefix_gr_calibrated` | 38.796222 | 0.389975 |
| `pf_z_prefix_slope` | 107.007015 | 0.224214 |
| `pf_z_prefix_plus_gr_calibrated` | 135.734575 | 0.179921 |
| `pf_z_control` | 163.301551 | 0.179258 |
| `pf_z_gr_calibrated` | 196.811290 | 0.137380 |

生成物:

- `artifacts/exp100_pf_z_unified_velocity_observation_prior_variant_metrics.csv`
- `artifacts/exp100_pf_z_unified_velocity_observation_prior_bucket_metrics.csv`
- `artifacts/exp100_pf_z_unified_velocity_observation_prior_by_well.csv`
- `artifacts/exp100_pf_z_unified_velocity_observation_prior_well_fit_summary.csv`
- `artifacts/exp100_pf_z_unified_velocity_observation_prior_candidate_wide.csv.gz`
- `artifacts/exp100_pf_z_unified_velocity_observation_prior_candidate_long.csv.gz`
- `artifacts/exp100_pf_z_unified_velocity_observation_prior_summary.json`

SHA / row count 検査:

- `variant_metrics.csv` SHA: `bc64dc86fa500721000b395fe8b70da51bde79d38b766063e0cdeac7ef7a0ccd`
- `candidate_wide.csv.gz` raw SHA: `2a6f24058b7b026248661cfd408aa8ada12d8183882b1299708f4393cc39595c`
- `candidate_wide.csv.gz` decompressed SHA: `0de40234e68e1cf956da0fafa08323a828865157052eb3af616faa677b0a0389`
- `candidate_wide.csv.gz` lines: 3,783,990 including header
- `candidate_long.csv.gz` raw SHA: `78c8524ea113b4ddfc7a3998c0031d8d4dbd1adf11bcbe6d7cfbdae364fcca22`
- `candidate_long.csv.gz` decompressed SHA: `591675720c4f3af1c3368faab95c2ecf0bb31dbee62ee0c220271ffb61fe4139`
- `candidate_long.csv.gz` lines: 30,271,913 including header
- `summary.json` actual local SHA: `a862b5d8d327e8388507baec9b4b927dddc3e6e9fe36e2e68c92e8960b4fa721`

注意:

- v2 summary 内に記録された summary 自身の SHA は自己参照のため final file SHA と一致しない。CSV / gzip の SHA は summary 記録とローカル検査が一致した。
- post-run で helper を修正し、次回以降は summary 自身の SHA を記録しない。
- `pf_z_control` は exp100 standalone rerun 内 control であり、exp072 保存済み `pf_z` との strict parity ではない。

解釈:

- XY velocity prior は standalone control から大幅改善したが、best RMSE 29.404162 は既存 `likpf_mean` RMSE 11.594897 より弱い。
- prefix slope と GR calibration は単独 / 組み合わせともに不採用。
- direct inference port / submit はしない。

### 2026-06-21 JST Kaggle train v1 failed

```bash
kaggle kernels push -p experiments/exp100_pf_z_unified_velocity_observation_prior/kaggle/train
kaggle kernels pull kentookumura/exp100-pf-z-unified-prior-train -p /tmp/kaggle-pull/exp100-pf-z-unified-prior-train -m
kaggle kernels logs kentookumura/exp100-pf-z-unified-prior-train
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp100-pf-z-unified-prior-train
kaggle kernels status kentookumura/exp100-pf-z-unified-prior-train
kaggle kernels output kentookumura/exp100-pf-z-unified-prior-train -p experiments/exp100_pf_z_unified_velocity_observation_prior/kaggle/output/train_v1
```

結果:

- Kernel version 1 push 成功。
- URL: https://www.kaggle.com/code/kentookumura/exp100-pf-z-unified-prior-train
- status: `KernelWorkerStatus.ERROR`
- runtime はログ上で約 3,873 sec。
- PF 本体と CSV / gzip 書き出し後、summary SHA 記録で失敗した。
- 失敗箇所: `run_audit()` の `output_files` に `summary.json` を入れたまま、summary を書く前に `sha256_path(summary_path)` を呼んでいた。
- error: `FileNotFoundError: /kaggle/working/artifacts/exp100_pf_z_unified_velocity_observation_prior_summary.json`
- partial output として `bucket_metrics.csv`、`by_well.csv`、`candidate_long.csv.gz` まではローカルに一部取得済み。`variant_metrics.csv` / `candidate_wide.csv.gz` / `summary.json` は取得できていない。

修正:

- `summary.json` は summary を一度書いてから SHA を計算するようにした。
- 成功時 summary status を `completed_train_side_audit` に変更した。

再実行した確認:

```bash
uv run ruff check experiments/exp100_pf_z_unified_velocity_observation_prior/pf_z_unified_velocity_observation_prior.py experiments/exp100_pf_z_unified_velocity_observation_prior/settings.py
python3 -m py_compile experiments/exp100_pf_z_unified_velocity_observation_prior/pf_z_unified_velocity_observation_prior.py experiments/exp100_pf_z_unified_velocity_observation_prior/settings.py
make validate-exp EXP=exp100_pf_z_unified_velocity_observation_prior
make prepare-kaggle-notebooks EXP=exp100_pf_z_unified_velocity_observation_prior EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp100-pf-z-unified-prior-train --title 'exp100 pf z unified prior train' --run-on-push --strict"
```

結果:

- ruff: PASS
- py_compile: PASS
- validate-exp: PASS
- Kaggle train package: 再生成済み
