# exp106_strict_exp072_pf_z_multiseed_scale_cache セッションノート

## 目的

`strict_exp072_pf_z_multiseed_scale_cache` backlog を実装する。exp104 は exp100 派生 `pf_z_xy_slope` の seedbag 化だったため、exp072 cache の `pf_z` と strict parity ではなかった。この実験では exp072 の `_pf_z` / `run_pf_z` と同一ロジックを seed 1 で再生成し、parity 通過後に multi-seed / scale cache を生成する。

## 現在の状態

- Route: `pf_beam`
- 状態: `completed_train_side_audit_rejected`
- CV: Kaggle train v3 完了
- LB: なし
- 提出: なし

## 実装内容

- `exp106_strict_exp072_pf_z_multiseed_scale_cache` を `exp104_pf_z_seedbag_scale_cache` から作成した。
- `.steering/20260622-exp106-strict-exp072-pf-z-multiseed-scale-cache/` に requirements / design / tasklist を記入した。
- `config.yaml` を exp106 用に更新し、`model.strict_pf_z` に exp072 `pf_z` と同じ PF パラメータを置いた。
- 実装対象を `strict_exp072_pf_z_multiseed_scale_cache.py` とし、出力 prefix を `exp106_strict_exp072_pf_z_multiseed_scale_cache` に変更した。
- 予定候補:
  - baseline: `exp072_pf_z`, `exp072_likpf_mean`, optional `exp072_likpf_scale_*`
  - parity: `strict_pf_z_parity_seed`
  - multi-seed: `pf_z_ms_mean`, `pf_z_ms_scale_3/5/8/12`
  - wide cache features: `pf_z_ms_std`, `pf_z_ms_best_lik_seed`, `pf_z_ms_delta_vs_pf_z`, `pf_z_ms_delta_vs_likpf_mean`
- 初回 full run は 64 seeds / 600 particles。exp104 128 seeds / 500 particles が約 11.4 時間だったため、parity と 64-seed metrics を確認してから 128 seeds へ上げる。
- v2 では exp072 と同じ well-level `joblib.Parallel(... prefer="threads")` を使い、`num_workers=8` にする。seed は well / seed index で固定済みなので thread scheduling には依存しない。

## コマンドログ

### 2026-06-22 JST 実装

```bash
make new-steering EXP=exp106_strict_exp072_pf_z_multiseed_scale_cache
make new-exp EXP=exp106_strict_exp072_pf_z_multiseed_scale_cache SOURCE=experiments/exp104_pf_z_seedbag_scale_cache
```

`git status --short` は `fatal: not a git repository` で失敗。この作業場所では通常の Git checkout として見えていない。

### 予定

```bash
python3 -m py_compile experiments/exp106_strict_exp072_pf_z_multiseed_scale_cache/strict_exp072_pf_z_multiseed_scale_cache.py experiments/exp106_strict_exp072_pf_z_multiseed_scale_cache/settings.py
python3 -m json.tool experiments/exp106_strict_exp072_pf_z_multiseed_scale_cache/exp106_strict_exp072_pf_z_multiseed_scale_cache_train.ipynb
python3 -m json.tool experiments/exp106_strict_exp072_pf_z_multiseed_scale_cache/exp106_strict_exp072_pf_z_multiseed_scale_cache_inference.ipynb
uv run ruff check experiments/exp106_strict_exp072_pf_z_multiseed_scale_cache/strict_exp072_pf_z_multiseed_scale_cache.py experiments/exp106_strict_exp072_pf_z_multiseed_scale_cache/settings.py
make validate-exp EXP=exp106_strict_exp072_pf_z_multiseed_scale_cache
make prepare-kaggle-notebooks EXP=exp106_strict_exp072_pf_z_multiseed_scale_cache EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp106-strict-exp072-pfz-ms-train --title 'exp106 strict exp072 pfz ms train' --run-on-push --strict"
make push-kaggle-train EXP=exp106_strict_exp072_pf_z_multiseed_scale_cache
```

ローカル smoke はユーザーが明示した場合のみ実行する。Kaggle train 完了後に `candidate_metrics.csv`、parity diff、summary、生成物 SHA、exp072 候補との比較結果を追記する。

### 2026-06-22 JST validation / package

```bash
python3 -m py_compile experiments/exp106_strict_exp072_pf_z_multiseed_scale_cache/strict_exp072_pf_z_multiseed_scale_cache.py experiments/exp106_strict_exp072_pf_z_multiseed_scale_cache/settings.py
python3 -m json.tool experiments/exp106_strict_exp072_pf_z_multiseed_scale_cache/exp106_strict_exp072_pf_z_multiseed_scale_cache_train.ipynb
python3 -m json.tool experiments/exp106_strict_exp072_pf_z_multiseed_scale_cache/exp106_strict_exp072_pf_z_multiseed_scale_cache_inference.ipynb
uv run ruff check experiments/exp106_strict_exp072_pf_z_multiseed_scale_cache/strict_exp072_pf_z_multiseed_scale_cache.py experiments/exp106_strict_exp072_pf_z_multiseed_scale_cache/settings.py
make validate-exp EXP=exp106_strict_exp072_pf_z_multiseed_scale_cache
make prepare-kaggle-notebooks EXP=exp106_strict_exp072_pf_z_multiseed_scale_cache EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp106-strict-exp072-pfz-ms-train --title 'exp106 strict exp072 pfz ms train' --run-on-push --strict"
make update-summary
```

結果:

- `py_compile`: PASS
- train / inference notebook JSON: PASS
- `ruff check`: PASS
- `validate-exp`: PASS
- Kaggle train package: `experiments/exp106_strict_exp072_pf_z_multiseed_scale_cache/kaggle/train`
- kernel id: `kentookumura/exp106-strict-exp072-pfz-ms-train`
- metadata: CPU / internet false / competition source `rogii-wellbore-geology-prediction` / kernel source `kentookumura/exp072-exp063-full-replay-feature-cache-train`
- `experiment_summary.md`: 107 experiments に更新

### 2026-06-22 JST Kaggle train v1 push / initial monitoring

```bash
make push-kaggle-train EXP=exp106_strict_exp072_pf_z_multiseed_scale_cache
kaggle kernels status kentookumura/exp106-strict-exp072-pfz-ms-train
kaggle kernels logs kentookumura/exp106-strict-exp072-pfz-ms-train
kaggle kernels pull kentookumura/exp106-strict-exp072-pfz-ms-train -p /tmp/kaggle-pull/exp106-strict-exp072-pfz-ms-train -m
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp106-strict-exp072-pfz-ms-train
kaggle kernels output kentookumura/exp106-strict-exp072-pfz-ms-train -p /tmp/kaggle-output/exp106-strict-exp072-pfz-ms-train-probe
```

結果:

- push: PASS
- kernel version: 1
- URL: https://www.kaggle.com/code/kentookumura/exp106-strict-exp072-pfz-ms-train
- status: `KernelWorkerStatus.RUNNING`
- logs: initial normal logs and 5-minute follow logs are empty
- output probe: no files yet

解釈:

- Kernel は Kaggle 上で実行開始済み。起動直後の logs/output が空なので、重複 push はしない。
- 初回 full run は 64 seeds / 600 particles の CPU run で長時間想定。完了後に output を取得し、parity diff と candidate metrics を記録する。

### 2026-06-22 JST v2 parallel fix / push

ユーザー指摘により、v1 の `num_workers=1` は exp072 と同等の高速化ではないと判断。exp072 と同じ well-level `joblib.Parallel(... prefer="threads")` を追加し、`runtime.num_workers=8` に変更した。

実装内容:

- `strict_exp072_pf_z_multiseed_scale_cache.py` に `joblib.Parallel` / `delayed` を追加。
- `run_audit` の well loop を `Parallel(n_jobs=num_workers, prefer="threads")` に変更。
- `warm_up_strict_pf_z_kernel()` を追加し、parallel 前に Numba kernel を一度 compile して thread compile 競合を避ける。
- seed は parity seed `stable_seed("pf_z", well)` と multiseed `stable_seed(exp106, "strict_pf_z", seed_root, well, seed_index)` のままなので、thread scheduling に依存しない。
- `config.yaml` の `runtime.num_workers` を `8` に変更。

```bash
python3 -m py_compile experiments/exp106_strict_exp072_pf_z_multiseed_scale_cache/strict_exp072_pf_z_multiseed_scale_cache.py experiments/exp106_strict_exp072_pf_z_multiseed_scale_cache/settings.py
uv run ruff check experiments/exp106_strict_exp072_pf_z_multiseed_scale_cache/strict_exp072_pf_z_multiseed_scale_cache.py experiments/exp106_strict_exp072_pf_z_multiseed_scale_cache/settings.py
make validate-exp EXP=exp106_strict_exp072_pf_z_multiseed_scale_cache
make prepare-kaggle-notebooks EXP=exp106_strict_exp072_pf_z_multiseed_scale_cache EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp106-strict-exp072-pfz-ms-train --title 'exp106 strict exp072 pfz ms train' --run-on-push --strict"
make push-kaggle-train EXP=exp106_strict_exp072_pf_z_multiseed_scale_cache
kaggle kernels status kentookumura/exp106-strict-exp072-pfz-ms-train
kaggle kernels logs kentookumura/exp106-strict-exp072-pfz-ms-train
kaggle kernels pull kentookumura/exp106-strict-exp072-pfz-ms-train -p /tmp/kaggle-pull/exp106-strict-exp072-pfz-ms-train-v2 -m
```

結果:

- `py_compile`: PASS
- `ruff check`: PASS
- `validate-exp`: PASS
- package config: `runtime.num_workers=8`, `model.strict_pf_z.n_seeds=64`
- v2 push: PASS
- kernel version: 2
- URL: https://www.kaggle.com/code/kentookumura/exp106-strict-exp072-pfz-ms-train
- v2 status: `KernelWorkerStatus.RUNNING`
- logs: initial normal logs are empty
- pull metadata: PASS (`/tmp/kaggle-pull/exp106-strict-exp072-pfz-ms-train-v2`)

以後は v2 を正として監視する。v1 は `num_workers=1` の superseded run として扱う。

### 2026-06-23 JST v2 failure / exact-kernel v3 patch

```bash
kaggle kernels status kentookumura/exp106-strict-exp072-pfz-ms-train
kaggle kernels logs kentookumura/exp106-strict-exp072-pfz-ms-train
kaggle kernels output kentookumura/exp106-strict-exp072-pfz-ms-train -p /tmp/kaggle-output/exp106-strict-exp072-pfz-ms-train-v2-failed
```

結果:

- v2 status: `KernelWorkerStatus.ERROR`
- failure cell: `summary = run_audit(config)`
- error: `ValueError: strict pf_z parity failed on full run`
- parity diff: rows 3,783,989 / wells 773 / max_abs_diff 69.723633 / mean_abs_diff 2.501835 / p95_abs_diff 15.295898 / rmse_diff 6.415956
- output取得: support files と kernel log のみ。candidate metrics / parity diff CSV は parity failure raise 前のため未保存。

原因:

- exp072 `_pf_z` と等価に再実装した `_strict_pf_z_allseeds` が exact ではなかった。well-level parallel / `num_workers=8` そのものではなく、PF kernel の細部差が原因。

修正:

- exp072 の `_resamp` / `_pf_z_seeded` 相当を exact seeded kernel `_strict_pf_z_seeded_exp072` として追加。
- parity seed と multi-seed の両方を exact seeded kernel の seed loop で生成するよう変更。
- exp072 と同じ `_grid` / `_gr_sig` 相当 helper を追加し、typewell grid と GR sigma を exact path に寄せた。
- well-level `joblib.Parallel(... prefer="threads")` と `num_workers=8` は維持。

```bash
python3 -m py_compile experiments/exp106_strict_exp072_pf_z_multiseed_scale_cache/strict_exp072_pf_z_multiseed_scale_cache.py experiments/exp106_strict_exp072_pf_z_multiseed_scale_cache/settings.py
uv run ruff check experiments/exp106_strict_exp072_pf_z_multiseed_scale_cache/strict_exp072_pf_z_multiseed_scale_cache.py experiments/exp106_strict_exp072_pf_z_multiseed_scale_cache/settings.py
make validate-exp EXP=exp106_strict_exp072_pf_z_multiseed_scale_cache
```

結果:

- `py_compile`: PASS
- `ruff check`: PASS
- `validate-exp`: PASS

### 2026-06-23 JST v3 push

```bash
make prepare-kaggle-notebooks EXP=exp106_strict_exp072_pf_z_multiseed_scale_cache EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp106-strict-exp072-pfz-ms-train --title 'exp106 strict exp072 pfz ms train' --run-on-push --strict"
make push-kaggle-train EXP=exp106_strict_exp072_pf_z_multiseed_scale_cache
kaggle kernels status kentookumura/exp106-strict-exp072-pfz-ms-train
kaggle kernels logs kentookumura/exp106-strict-exp072-pfz-ms-train
kaggle kernels pull kentookumura/exp106-strict-exp072-pfz-ms-train -p /tmp/kaggle-pull/exp106-strict-exp072-pfz-ms-train-v3 -m
```

結果:

- package: PASS
- package config: `runtime.num_workers=8`, `model.strict_pf_z.n_seeds=64`
- push: PASS
- kernel version: 3
- URL: https://www.kaggle.com/code/kentookumura/exp106-strict-exp072-pfz-ms-train
- status: `KernelWorkerStatus.RUNNING`
- logs: initial normal logs are empty
- pull metadata: PASS (`/tmp/kaggle-pull/exp106-strict-exp072-pfz-ms-train-v3`)

以後は v3 を正として監視する。v2 は parity failed run、v1 は `num_workers=1` superseded run として扱う。

### 2026-06-23 JST v3 completion / output

```bash
kaggle kernels status kentookumura/exp106-strict-exp072-pfz-ms-train
kaggle kernels logs kentookumura/exp106-strict-exp072-pfz-ms-train
kaggle kernels output kentookumura/exp106-strict-exp072-pfz-ms-train -p experiments/exp106_strict_exp072_pf_z_multiseed_scale_cache/kaggle/output/train_v3
```

結果:

- v3 status: `KernelWorkerStatus.COMPLETE`
- output: `experiments/exp106_strict_exp072_pf_z_multiseed_scale_cache/kaggle/output/train_v3`
- rows / wells: 3,783,989 / 773
- runtime: 10,111.57 sec
- parity: PASS (`max_abs_diff=0.0`, `mean_abs_diff=0.0`, `p95_abs_diff=0.0`, `rmse_diff=0.0`)
- best overall: `exp072_likpf_mean` RMSE 11.594898 / MAE 7.067633 / within10 0.772807 / bias -1.099423
- best strict multiseed: `pf_z_ms_scale_3` RMSE 16.145943 / MAE 9.155580 / within10 0.708807 / bias -0.752507
- `exp072_pf_z`: RMSE 17.788171 / MAE 10.677487 / within10 0.647668 / bias -0.934560
- delta `pf_z_ms_scale_3` vs `exp072_pf_z`: RMSE -1.642228
- delta `pf_z_ms_scale_3` vs `exp072_likpf_mean`: RMSE +4.551045

生成物:

- `exp106_strict_exp072_pf_z_multiseed_scale_cache_candidate_metrics.csv`
- `exp106_strict_exp072_pf_z_multiseed_scale_cache_bucket_metrics.csv`
- `exp106_strict_exp072_pf_z_multiseed_scale_cache_by_well.csv`
- `exp106_strict_exp072_pf_z_multiseed_scale_cache_strict_pf_z_quality.csv`
- `exp106_strict_exp072_pf_z_multiseed_scale_cache_parity_diff.csv.gz`
- `exp106_strict_exp072_pf_z_multiseed_scale_cache_candidate_wide.csv.gz`
- `exp106_strict_exp072_pf_z_multiseed_scale_cache_summary.json`

解釈:

- exact exp072 seeded PF kernel と `num_workers=8` well-level thread parallelism の組み合わせで再現性は維持された。
- multiseed / scale 化は exp072 plain `pf_z` の lucky/unlucky seed 問題を一部改善したが、`likpf_mean` との差は大きい。
- direct inference port / submit / exp073 or exp092 への置換特徴量化はしない。PF 側は候補生成や target-free likelihood / confidence feature へ戻す。
