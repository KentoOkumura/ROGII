# exp143_multimode_pfbeam_local_correlation_audit セッションノート

## 目的

`multimode_pfbeam_local_correlation_audit` backlog を実装する。exp083 v11 の `1b1eba53` 型 failure を念頭に、PF/Beam が局所相関ずれ候補を保持できず早期に単一モードへ潰れていないかを train pseudo-tail で診断する。

## 現在の状態

- Route: `pf_beam`
- 状態: `implemented_not_run`
- CV: 未実行
- LB: 未提出
- blocked: none

## 変更点

- `docs/legacy/steering/20260627-exp143-multimode-pfbeam-local-correlation-audit/` を作成した。
- `experiments/exp143_multimode_pfbeam_local_correlation_audit/` を `exp142_trajectory_aware_pf_transition_prior` から作成した。
- 実装ファイルを `multimode_pfbeam_local_correlation_audit.py` に変更した。
- exp072 strict PF-Z parity / multiseed PF-Z / exp142-style multimode PF variants は候補比較基準として残した。
- seed path 分布から以下の診断列を生成する処理を追加した。
  - `mode_count`: seed TVT を 5ft bin に切った row-local mode 数。
  - `mode_entropy`: mode bin 分布の正規化 entropy。
  - `seed_spread_p90_p10`: seed TVT の p90-p10 spread。
  - `local_corr_topk_spread`: eval GR window と typewell GR window の local correlation 上位 seed の TVT spread。
  - `local_corr_mean` / `local_corr_max` / `local_corr_topk_mean`。
  - `best_local_corr_tvt`: target-free local correlation が最大の seed TVT。
- `strict_pf_z_quality.csv` に strict multiseed の mode/correlation aggregate を追加した。
- `multimode_pf_z_quality.csv` に variant 別の ESS、resampling、collapse、mode/correlation aggregate を追加した。
- train notebook は設定確認、入力 preview、variant plan、audit 実行、metrics/artifacts 表示の構成にした。
- inference notebook は diagnostic-only marker のみ。

## 再現性メモ

- seed policy: `stable_sha256_seed_from_experiment_multimode_pfbeam_local_correlation_well_variant_seed_index`
- stochastic components: strict PF-Z / multimode PF-Z particle initialization, process noise, resampling, upstream exp072 cache
- parallel RNG policy: well-level thread parallel; each well/variant gets stable seed vector before Numba kernel
- CPU/GPU runtime: CPU-only、GPU 不使用
- deterministic anchor: false。train-side diagnostic only。
- gzip output: decompressed content SHA を summary JSON に記録する。
- submission / prediction SHA: 推論・提出なしのため対象外。

## コマンドログ

```bash
uv run python scripts/new_steering.py --experiment exp143_multimode_pfbeam_local_correlation_audit
uv run python scripts/new_experiment.py --name exp143_multimode_pfbeam_local_correlation_audit --source experiments/exp142_trajectory_aware_pf_transition_prior
mv experiments/exp143_multimode_pfbeam_local_correlation_audit/trajectory_aware_pf_transition_prior.py experiments/exp143_multimode_pfbeam_local_correlation_audit/multimode_pfbeam_local_correlation_audit.py
mv experiments/exp143_multimode_pfbeam_local_correlation_audit/exp142_trajectory_aware_pf_transition_prior_train.ipynb experiments/exp143_multimode_pfbeam_local_correlation_audit/exp143_multimode_pfbeam_local_correlation_audit_train.ipynb
mv experiments/exp143_multimode_pfbeam_local_correlation_audit/exp142_trajectory_aware_pf_transition_prior_inference.ipynb experiments/exp143_multimode_pfbeam_local_correlation_audit/exp143_multimode_pfbeam_local_correlation_audit_inference.ipynb
```

## 検証

```bash
uv run python -m py_compile experiments/exp143_multimode_pfbeam_local_correlation_audit/multimode_pfbeam_local_correlation_audit.py experiments/exp143_multimode_pfbeam_local_correlation_audit/settings.py
python3 -m json.tool experiments/exp143_multimode_pfbeam_local_correlation_audit/exp143_multimode_pfbeam_local_correlation_audit_train.ipynb
python3 -m json.tool experiments/exp143_multimode_pfbeam_local_correlation_audit/exp143_multimode_pfbeam_local_correlation_audit_inference.ipynb
uv run ruff check experiments/exp143_multimode_pfbeam_local_correlation_audit/multimode_pfbeam_local_correlation_audit.py experiments/exp143_multimode_pfbeam_local_correlation_audit/settings.py
uv run ruff format --check experiments/exp143_multimode_pfbeam_local_correlation_audit/multimode_pfbeam_local_correlation_audit.py experiments/exp143_multimode_pfbeam_local_correlation_audit/settings.py
uv run python scripts/validate_experiment.py --experiment exp143_multimode_pfbeam_local_correlation_audit
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp143_multimode_pfbeam_local_correlation_audit --notebook train --kernel-id kentookumura/exp143-multimode-pfbeam-corr-train --title 'exp143 multimode pfbeam corr train' --run-on-push --strict
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp143_multimode_pfbeam_local_correlation_audit --notebook inference --kernel-id kentookumura/exp143-multimode-pfbeam-corr-infer --title 'exp143 multimode pfbeam corr infer' --run-on-push --strict
python3 -m json.tool experiments/exp143_multimode_pfbeam_local_correlation_audit/kaggle/train/exp143_multimode_pfbeam_local_correlation_audit_train.ipynb
python3 -m json.tool experiments/exp143_multimode_pfbeam_local_correlation_audit/kaggle/inference/exp143_multimode_pfbeam_local_correlation_audit_inference.ipynb
uv run python -m py_compile experiments/exp143_multimode_pfbeam_local_correlation_audit/kaggle/train/multimode_pfbeam_local_correlation_audit.py experiments/exp143_multimode_pfbeam_local_correlation_audit/kaggle/train/settings.py experiments/exp143_multimode_pfbeam_local_correlation_audit/kaggle/inference/settings.py
```

- `py_compile`: PASS
- train notebook JSON: PASS
- inference notebook JSON: PASS
- `ruff check`: PASS
- `ruff format --check`: PASS after formatting `multimode_pfbeam_local_correlation_audit.py`
- `validate_experiment.py`: PASS
- `prepare_kaggle_notebooks.py --notebook train --strict`: PASS
- `prepare_kaggle_notebooks.py --notebook inference --strict`: PASS
- packaged train notebook JSON: PASS
- packaged inference notebook JSON: PASS
- packaged support `.py` py_compile: PASS
- generated train package: `experiments/exp143_multimode_pfbeam_local_correlation_audit/kaggle/train`
- generated inference package: `experiments/exp143_multimode_pfbeam_local_correlation_audit/kaggle/inference`
- generated train metadata:
  - kernel id: `kentookumura/exp143-multimode-pfbeam-corr-train`
  - title: `exp143 multimode pfbeam corr train`
  - GPU: false
  - internet: false
  - run_on_push: true
  - competition source: `rogii-wellbore-geology-prediction`
  - kernel source: `kentookumura/exp072-exp063-full-replay-feature-cache-train`
- generated inference metadata:
  - kernel id: `kentookumura/exp143-multimode-pfbeam-corr-infer`
  - title: `exp143 multimode pfbeam corr infer`
  - GPU: false
  - internet: false
  - run_on_push: true
  - competition source: `rogii-wellbore-geology-prediction`
  - kernel source: `kentookumura/exp072-exp063-full-replay-feature-cache-train`

## Kaggle 実行

```bash
kaggle kernels push -p experiments/exp143_multimode_pfbeam_local_correlation_audit/kaggle/train
kaggle kernels pull kentookumura/exp143-multimode-pfbeam-corr-train -p /tmp/kaggle-pull/exp143-multimode-pfbeam-corr-train-v1 -m
kaggle kernels logs kentookumura/exp143-multimode-pfbeam-corr-train
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp143-multimode-pfbeam-corr-train
```

- `kaggle kernels push`: PASS。Kernel version 1 successfully pushed.
- URL: https://www.kaggle.com/code/kentookumura/exp143-multimode-pfbeam-corr-train
- `kaggle kernels pull -m`: PASS。metadata を `/tmp/kaggle-pull/exp143-multimode-pfbeam-corr-train-v1` に取得。
- initial `kaggle kernels logs`: 空。
- `logs -f`: CLI logs が空のまま推移。ユーザー指示により監視停止。完了連絡後に output / logs を取得する。

## 次のアクション

1. ユーザーから Kaggle 完了連絡を受けたら output / logs を取得する。
2. output 取得後に `result.md`、`metrics.json`、`experiment_summary.md`、`KAGGLE_DIRECTION.md` を更新する。

## Kaggle v1 timeout 対応

- ユーザー連絡: Kaggle v1 が timeout。
- `kaggle kernels logs kentookumura/exp143-multimode-pfbeam-corr-train`:
  - bootstrap と config 表示は完了。
  - exp072 cache は見つかり、required columns missing は `[]`。
  - `quality metrics...` 以降の artifact 書き出しログなし。
- `kaggle kernels output kentookumura/exp143-multimode-pfbeam-corr-train -p experiments/exp143_multimode_pfbeam_local_correlation_audit/kaggle/output/train_v1`:
  - `artifacts/` なし。
  - bootstrap/support files と Numba cache のみ。
- 推定原因:
  - full 773 wells / 約 3.78M rows。
  - strict PF-Z parity + 32 seeds、multimode 3 variants x 32 seeds、各 600 particles。
  - local GR correlation と mode diagnostics を seed/path ごとに計算するため Kaggle notebook 制限内に収まらなかった。

## Kaggle v2 scope

- 同一 exp143 / 同一 kernel id で timeout 復旧版を push する。
- scope: `targeted_timeout_recovery_v2`
- focus wells:
  - `1b1eba53`
  - `86454a6f`
  - `fb03ae90`
  - `91b301ce`
  - `ba48188d`
  - `fef8af96`
- `audit.max_rows_per_well`: 2000
- strict PF-Z: 300 particles / 8 seeds
- multimode PF-Z: 300 particles / 8 seeds / 2 transition variants
- runtime workers: 4
- 追加実装:
  - `audit.focus_wells`
  - `audit.max_rows_per_well`
  - well start/done progress logs
  - summary status `completed_scoped_train_side_audit`

## Kaggle v2 実行

```bash
uv run python -m py_compile experiments/exp143_multimode_pfbeam_local_correlation_audit/multimode_pfbeam_local_correlation_audit.py experiments/exp143_multimode_pfbeam_local_correlation_audit/settings.py
python3 -m json.tool experiments/exp143_multimode_pfbeam_local_correlation_audit/exp143_multimode_pfbeam_local_correlation_audit_train.ipynb
python3 -m json.tool experiments/exp143_multimode_pfbeam_local_correlation_audit/exp143_multimode_pfbeam_local_correlation_audit_inference.ipynb
uv run ruff check experiments/exp143_multimode_pfbeam_local_correlation_audit/multimode_pfbeam_local_correlation_audit.py experiments/exp143_multimode_pfbeam_local_correlation_audit/settings.py
uv run ruff format --check experiments/exp143_multimode_pfbeam_local_correlation_audit/multimode_pfbeam_local_correlation_audit.py experiments/exp143_multimode_pfbeam_local_correlation_audit/settings.py
uv run python scripts/validate_experiment.py --experiment exp143_multimode_pfbeam_local_correlation_audit
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp143_multimode_pfbeam_local_correlation_audit --notebook train --kernel-id kentookumura/exp143-multimode-pfbeam-corr-train --title 'exp143 multimode pfbeam corr train' --run-on-push --strict
python3 -m json.tool experiments/exp143_multimode_pfbeam_local_correlation_audit/kaggle/train/exp143_multimode_pfbeam_local_correlation_audit_train.ipynb
uv run python -m py_compile experiments/exp143_multimode_pfbeam_local_correlation_audit/kaggle/train/multimode_pfbeam_local_correlation_audit.py experiments/exp143_multimode_pfbeam_local_correlation_audit/kaggle/train/settings.py
kaggle kernels push -p experiments/exp143_multimode_pfbeam_local_correlation_audit/kaggle/train
kaggle kernels pull kentookumura/exp143-multimode-pfbeam-corr-train -p /tmp/kaggle-pull/exp143-multimode-pfbeam-corr-train-v2 -m
```

- `py_compile`: PASS
- train notebook JSON: PASS
- inference notebook JSON: PASS
- `ruff check`: PASS
- `ruff format --check`: PASS
- `validate_experiment.py`: PASS
- `prepare_kaggle_notebooks.py --notebook train --strict`: PASS
- packaged train notebook JSON: PASS
- packaged support `.py` py_compile: PASS
- packaged config confirmed:
  - strict/multimode `n_particles: 300`
  - strict/multimode `n_seeds: 8`
  - `runtime.num_workers: 4`
  - `audit.focus_wells`: 6 wells
  - `audit.max_rows_per_well: 2000`
- `kaggle kernels push`: PASS。Kernel version 2 successfully pushed.
- URL: https://www.kaggle.com/code/kentookumura/exp143-multimode-pfbeam-corr-train
- `kaggle kernels pull -m`: PASS。metadata を `/tmp/kaggle-pull/exp143-multimode-pfbeam-corr-train-v2` に取得。
- ユーザー指示により v2 も監視しない。完了連絡後に output / logs を取得する。

## Kaggle v2 failure

- ユーザー連絡: v2 failed。
- `kaggle kernels logs kentookumura/exp143-multimode-pfbeam-corr-train`: PASS。
- `kaggle kernels output kentookumura/exp143-multimode-pfbeam-corr-train -p experiments/exp143_multimode_pfbeam_local_correlation_audit/kaggle/output/train_v2`: PASS。
- v2 log:
  - scope: 6 wells / 12,000 rows / 4 workers / max rows per well 2000。
  - PF task は全 well 完了。
  - elapsed by well: `86454a6f` 43.9s、`1b1eba53` 61.4s、`91b301ce` 64.5s、`ba48188d` 64.7s、`fb03ae90` 29.8s、`fef8af96` 13.5s。
  - failure: `ValueError: strict pf_z parity failed on full run`
  - parity summary: rows 12000、wells 6、max abs diff 60.40625、mean abs diff 19.92836、p95 abs diff 52.67388、rmse diff 25.16116。
- 原因:
  - `audit.focus_wells` と `audit.max_rows_per_well` を追加したが、parity guard の full-run 判定が `max_rows` / `max_wells` だけを見ていた。
  - scoped run なのに full-run parity failure として raise していた。
- v3 fix:
  - `scoped_run = any(max_rows, max_wells, focus_wells, max_rows_per_well)` を run_audit 冒頭で定義。
  - parity tolerance と `require_parity_for_full` は `strict_config` から読む。
  - `not scoped_run` の場合だけ full-run parity guard で raise する。
  - scoped v3 では parity summary は生成物に残し、診断結果の解釈で扱う。

## Kaggle v3 実行

```bash
uv run python -m py_compile experiments/exp143_multimode_pfbeam_local_correlation_audit/multimode_pfbeam_local_correlation_audit.py experiments/exp143_multimode_pfbeam_local_correlation_audit/settings.py
uv run ruff check experiments/exp143_multimode_pfbeam_local_correlation_audit/multimode_pfbeam_local_correlation_audit.py experiments/exp143_multimode_pfbeam_local_correlation_audit/settings.py
uv run ruff format --check experiments/exp143_multimode_pfbeam_local_correlation_audit/multimode_pfbeam_local_correlation_audit.py experiments/exp143_multimode_pfbeam_local_correlation_audit/settings.py
uv run python scripts/validate_experiment.py --experiment exp143_multimode_pfbeam_local_correlation_audit
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp143_multimode_pfbeam_local_correlation_audit --notebook train --kernel-id kentookumura/exp143-multimode-pfbeam-corr-train --title 'exp143 multimode pfbeam corr train' --run-on-push --strict
python3 -m json.tool experiments/exp143_multimode_pfbeam_local_correlation_audit/kaggle/train/exp143_multimode_pfbeam_local_correlation_audit_train.ipynb
uv run python -m py_compile experiments/exp143_multimode_pfbeam_local_correlation_audit/kaggle/train/multimode_pfbeam_local_correlation_audit.py experiments/exp143_multimode_pfbeam_local_correlation_audit/kaggle/train/settings.py
kaggle kernels push -p experiments/exp143_multimode_pfbeam_local_correlation_audit/kaggle/train
kaggle kernels pull kentookumura/exp143-multimode-pfbeam-corr-train -p /tmp/kaggle-pull/exp143-multimode-pfbeam-corr-train-v3 -m
```

- `py_compile`: PASS
- `ruff check`: PASS
- `ruff format --check`: PASS
- `validate_experiment.py`: PASS
- `prepare_kaggle_notebooks.py --notebook train --strict`: PASS
- packaged train notebook JSON: PASS
- packaged support `.py` py_compile: PASS
- packaged config confirmed:
  - status: `kaggle_v3_running`
  - strict/multimode `n_particles: 300`
  - strict/multimode `n_seeds: 8`
  - `audit.focus_wells`: 6 wells
  - `audit.max_rows_per_well: 2000`
  - scoped parity guard: `not scoped_run`
- `kaggle kernels push`: PASS。Kernel version 3 successfully pushed.
- URL: https://www.kaggle.com/code/kentookumura/exp143-multimode-pfbeam-corr-train
- `kaggle kernels pull -m`: PASS。metadata を `/tmp/kaggle-pull/exp143-multimode-pfbeam-corr-train-v3` に取得。
- ユーザー指示により v3 も監視しない。完了連絡後に output / logs を取得する。

## Kaggle v3 completion readout

```bash
kaggle kernels logs kentookumura/exp143-multimode-pfbeam-corr-train
kaggle kernels output kentookumura/exp143-multimode-pfbeam-corr-train -p experiments/exp143_multimode_pfbeam_local_correlation_audit/kaggle/output/train_v3
uv run python - <<'PY'
# summary / candidate_metrics / quality readout
PY
```

- ユーザー連絡: v3 完了。
- `kaggle kernels logs`: PASS。
- `kaggle kernels output`: PASS。
- output dir: `experiments/exp143_multimode_pfbeam_local_correlation_audit/kaggle/output/train_v3`
- artifact dir: `experiments/exp143_multimode_pfbeam_local_correlation_audit/kaggle/output/train_v3/artifacts`
- 保存生成物:
  - `exp143_multimode_pfbeam_local_correlation_audit_candidate_metrics.csv`
  - `exp143_multimode_pfbeam_local_correlation_audit_bucket_metrics.csv`
  - `exp143_multimode_pfbeam_local_correlation_audit_by_well.csv`
  - `exp143_multimode_pfbeam_local_correlation_audit_strict_pf_z_quality.csv`
  - `exp143_multimode_pfbeam_local_correlation_audit_multimode_pf_z_quality.csv`
  - `exp143_multimode_pfbeam_local_correlation_audit_parity_diff.csv.gz`
  - `exp143_multimode_pfbeam_local_correlation_audit_candidate_wide.csv.gz`
  - `exp143_multimode_pfbeam_local_correlation_audit_summary.json`
- summary status: `completed_scoped_train_side_audit`
- runtime: 294.693 sec
- rows / wells: 12,000 / 6
- best overall: `exp072_pf_ancc`, RMSE 50.721842, MAE 43.416754, within10 0.164833
- `exp072_likpf_mean`: RMSE 52.758772, MAE 47.635939, within10 0.000833
- `exp072_pf_z`: RMSE 57.641691, MAE 45.050145, within10 0.216667
- best strict multiseed: `pf_z_ms_scale_12`, RMSE 64.039534
- best multimode: `multimode_pf_zacc_s010_a020_noise050_best_lik_seed`, RMSE 60.763085
- best multimode delta:
  - vs `exp072_likpf_mean`: +8.004313 RMSE
  - vs `exp072_pf_z`: +3.121394 RMSE
- multimode quality aggregate:
  - `zmean_s006_noise025`: mean mode count 1.309000、mode_count_le1_rate 0.717000、mean topK spread 2.214723
  - `zacc_s010_a020_noise050`: mean mode count 1.462917、mode_count_le1_rate 0.645333、mean topK spread 3.605253
- interpretation:
  - `1b1eba53` / `91b301ce` では z-accel variant が mode diversity と topK spread を残した。
  - `fb03ae90` / `86454a6f` はほぼ単一 mode collapse。
  - non-oracle scorer は既存候補を上回れないため、direct candidate / inference port / submit はしない。
- updated:
  - `config.yaml`: `completed_scoped_train_side_diagnostic_no_submit`
  - `result.md`
  - `README.md`
  - `metrics.json`
  - `experiment_summary.md`
  - `KAGGLE_DIRECTION.md`

## Interpretation correction

- ユーザー指摘: 比較対象は `likpf_mean` ではなく従来 Beam ではないか。
- 修正:
  - exp143 の主比較対象は従来 Beam `exp072_beam_mean` とする。
  - 採用 guard / submit 判断では `pf_ancc`、`likpf_mean`、`pf_z` も併記する。
- corrected readout:
  - `exp072_beam_mean`: RMSE 70.297647 / MAE 66.370694
  - best multimode: RMSE 60.763085 / MAE 55.599689
  - best multimode delta vs Beam: RMSE -9.534561 / MAE -10.771005
  - best strict multiseed delta vs Beam: RMSE -6.258113
- conclusion:
  - Beam 比では positive。
  - ただし `pf_ancc` / `likpf_mean` / `pf_z` には届かないため direct candidate / inference port / submit はしない。
- updated:
  - `result.md`
  - `README.md`
  - `metrics.json`
  - `experiment_summary.md`
  - `KAGGLE_DIRECTION.md`
