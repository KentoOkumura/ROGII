# exp142_trajectory_aware_pf_transition_prior セッションノート

## 目的

`trajectory_aware_pf_transition_prior` backlog を実装する。exp106 strict PF-Z parity 実装を親に、`dZ/dMD`、`d2Z/dMD2`、prefix TVT slope を PF transition prior の平均速度・process noise・velocity likelihood 幅へ入れた候補を train pseudo-tail で評価する。

## 現在の状態

- Route: `pf_beam`
- 状態: `completed_train_side_rejected_no_submit`
- CV: best `exp072_likpf_mean` RMSE 11.594897672。best trajectory-aware PF は RMSE 23.132450295 で不採用。
- LB: 未提出
- blocked: none

## 変更点

- `.steering/20260627-exp142-trajectory-aware-pf-transition-prior/` を作成し、要件・設計・タスクを記入した。
- `experiments/exp142_trajectory_aware_pf_transition_prior/` を `exp106_strict_exp072_pf_z_multiseed_scale_cache` から作成した。
- 実装ファイルを `trajectory_aware_pf_transition_prior.py` に変更した。
- exp106 strict PF-Z parity / multiseed candidate は比較基準として残した。
- `_trajectory_pf_z_seeded()` を追加した。
  - target-free eval-zone `MD` / `Z` から `dZ/dMD` と `d2Z/dMD2` を計算する。
  - local TVT velocity の transition mean を `beta*dZ/dMD + intercept`、`d2Z/dMD2`、prefix TVT slope で調整する。
  - high-curvature 区間では velocity / position process noise と velocity likelihood sigma を広げる。
  - seed ごとの `mean_neff_frac`、`min_neff_frac`、`mean_resample_count`、`mean_collapse_rate`、`mean_particle_std` を保存する。
- `model.trajectory_pf_z.transition_variants` に 3 variants を追加した。
- bucket metrics に `abs_dzdmd` と `abs_d2zdmd2` を追加した。
- train notebook を exp142 用に更新し、設定確認、入力 preview、variant plan、audit 実行、metrics/artifacts 表示の構成にした。
- inference notebook は diagnostic-only marker を表示する構成にした。
- `README.md`、`result.md`、`metrics.json` を未実行状態で初期化した。

## 再現性メモ

- seed policy: `stable_sha256_seed_from_experiment_trajectory_pf_z_well_variant_seed_index`
- stochastic components: strict PF-Z / trajectory PF-Z particle initialization, process noise, resampling, upstream exp072 cache
- parallel RNG policy: well-level thread parallel; each well/variant gets stable seed vector before Numba kernel
- CPU/GPU runtime: CPU-only、GPU 不使用
- deterministic anchor: false。train-side diagnostic only。
- gzip output: decompressed content SHA を summary JSON に記録する。
- submission / prediction SHA: 推論・提出なしのため対象外。

## コマンドログ

```bash
uv run python scripts/new_steering.py --experiment exp142_trajectory_aware_pf_transition_prior
uv run python scripts/new_experiment.py --name exp142_trajectory_aware_pf_transition_prior --source experiments/exp106_strict_exp072_pf_z_multiseed_scale_cache
mv experiments/exp142_trajectory_aware_pf_transition_prior/strict_exp072_pf_z_multiseed_scale_cache.py experiments/exp142_trajectory_aware_pf_transition_prior/trajectory_aware_pf_transition_prior.py
mv experiments/exp142_trajectory_aware_pf_transition_prior/exp106_strict_exp072_pf_z_multiseed_scale_cache_train.ipynb experiments/exp142_trajectory_aware_pf_transition_prior/exp142_trajectory_aware_pf_transition_prior_train.ipynb
mv experiments/exp142_trajectory_aware_pf_transition_prior/exp106_strict_exp072_pf_z_multiseed_scale_cache_inference.ipynb experiments/exp142_trajectory_aware_pf_transition_prior/exp142_trajectory_aware_pf_transition_prior_inference.ipynb
```

## 検証

```bash
uv run python -m py_compile experiments/exp142_trajectory_aware_pf_transition_prior/trajectory_aware_pf_transition_prior.py experiments/exp142_trajectory_aware_pf_transition_prior/settings.py
python3 -m json.tool experiments/exp142_trajectory_aware_pf_transition_prior/exp142_trajectory_aware_pf_transition_prior_train.ipynb
python3 -m json.tool experiments/exp142_trajectory_aware_pf_transition_prior/exp142_trajectory_aware_pf_transition_prior_inference.ipynb
uv run ruff check experiments/exp142_trajectory_aware_pf_transition_prior/trajectory_aware_pf_transition_prior.py experiments/exp142_trajectory_aware_pf_transition_prior/settings.py
uv run ruff format --check experiments/exp142_trajectory_aware_pf_transition_prior/trajectory_aware_pf_transition_prior.py experiments/exp142_trajectory_aware_pf_transition_prior/settings.py
uv run python scripts/validate_experiment.py --experiment exp142_trajectory_aware_pf_transition_prior
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp142_trajectory_aware_pf_transition_prior --notebook train --kernel-id kentookumura/exp142-traj-pf-trans-prior-train --title 'exp142 traj pf trans prior train' --run-on-push --strict
python3 -m json.tool experiments/exp142_trajectory_aware_pf_transition_prior/kaggle/train/exp142_trajectory_aware_pf_transition_prior_train.ipynb
uv run python -m py_compile experiments/exp142_trajectory_aware_pf_transition_prior/kaggle/train/trajectory_aware_pf_transition_prior.py experiments/exp142_trajectory_aware_pf_transition_prior/kaggle/train/settings.py
kaggle kernels push -p experiments/exp142_trajectory_aware_pf_transition_prior/kaggle/train
kaggle kernels pull kentookumura/exp142-traj-pf-trans-prior-train -p /tmp/kaggle-pull/exp142-traj-pf-trans-prior-train-v1 -m
kaggle kernels logs kentookumura/exp142-traj-pf-trans-prior-train
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp142-traj-pf-trans-prior-train
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
- generated train package: `experiments/exp142_trajectory_aware_pf_transition_prior/kaggle/train`
- generated metadata:
  - kernel id: `kentookumura/exp142-traj-pf-trans-prior-train`
  - title: `exp142 traj pf trans prior train`
  - GPU: false
  - internet: false
  - run_on_push: true
  - competition source: `rogii-wellbore-geology-prediction`
  - kernel source: `kentookumura/exp072-exp063-full-replay-feature-cache-train`
- Kaggle push:
  - command: `kaggle kernels push -p experiments/exp142_trajectory_aware_pf_transition_prior/kaggle/train`
  - result: Kernel version 1 successfully pushed
  - URL: https://www.kaggle.com/code/kentookumura/exp142-traj-pf-trans-prior-train
  - pull metadata: PASS (`/tmp/kaggle-pull/exp142-traj-pf-trans-prior-train-v1`)
  - initial `kaggle kernels logs`: empty
  - `logs -f`: stopped by user request; monitoring will resume after user completion notice if needed
- synthetic PF kernel smoke: NOT RUN locally because the local uv environment does not have `numba` installed (`ModuleNotFoundError: No module named 'numba'`). Existing PF experiments are Kaggle/Numba-runtime oriented; no dependency install was performed.

## 2026-06-27 修正: 誤混在 v2 の扱い

- `tvt_plus_z_beam_smoothness_penalty` backlog を誤って exp142 に追加した v2 は、`trajectory_aware_pf_transition_prior` と仮説が混在しているため invalid とする。
- exp142 の正しい実装は `trajectory_aware_pf_transition_prior` の 3 variants のみ。
- `tvt_plus_z_beam_smoothness_penalty` は exp142 から除去し、別実験 `exp146_tvt_plus_z_beam_smoothness_penalty` として扱う。
- exp142 復旧後の検証:
  - `uv run python -m py_compile ...`: PASS
  - train / inference notebook JSON: PASS
  - `uv run ruff check ...`: PASS
  - `uv run ruff format --check ...`: PASS after `uv run ruff format ...`
  - `uv run python scripts/validate_experiment.py --experiment exp142_trajectory_aware_pf_transition_prior`: PASS
  - `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp142_trajectory_aware_pf_transition_prior --notebook train --kernel-id kentookumura/exp142-traj-pf-trans-prior-train --title 'exp142 traj pf trans prior train' --run-on-push --strict`: PASS
  - packaged train notebook JSON / packaged support `.py` py_compile: PASS
  - source `trajectory_aware_pf_transition_prior.py` と packaged copy の diff: no diff
  - source `config.yaml` と packaged copy の diff: no diff
  - local Numba compile smoke: NOT RUN。`uv run python -c "import numba"` が `ModuleNotFoundError: No module named 'numba'` で失敗したため、Kaggle runtime で確認する。
- 既存 Kaggle v2 の扱い:
  - command: `kaggle kernels push -p experiments/exp142_trajectory_aware_pf_transition_prior/kaggle/train`
  - result: Kernel version 2 successfully pushed before this separation was corrected
  - URL: https://www.kaggle.com/code/kentookumura/exp142-traj-pf-trans-prior-train
  - pull metadata: PASS (`/tmp/kaggle-pull/exp142-traj-pf-trans-prior-train-v2`)
  - initial `kaggle kernels logs`: empty
  - `timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp142-traj-pf-trans-prior-train`: stopped by user request; user will report completion.
  - status for records: invalid_mixed_backlog_do_not_use
- 復旧済み exp142 source/package は 2026-06-27 に v3 として再 push 済み。v3 が trajectory-only の正しい実行。

## 2026-06-27 復旧版 v3 push

- ユーザー指示により、復旧済み trajectory-only 版を v3 として Kaggle 実行した。
- 事前確認:
  - `uv run python scripts/validate_experiment.py --experiment exp142_trajectory_aware_pf_transition_prior`: PASS
  - source/package の `config.yaml` diff: no diff
  - source/package の `trajectory_aware_pf_transition_prior.py` diff: no diff
  - code/config/steering に `tvt_plus_z` / `beam_smoothness` / `U = TVT + Z` / `dU/dMD` の残留なし
- first push attempt:
  - `Kernel push error: Maximum batch CPU session count of 5 reached.`
  - retry でも同じエラー。
- after exp146 completion, retry:
  - command: `make push-kaggle-train EXP=exp142_trajectory_aware_pf_transition_prior`
  - result: Kernel version 3 successfully pushed
  - URL: https://www.kaggle.com/code/kentookumura/exp142-traj-pf-trans-prior-train
  - status check: `KernelWorkerStatus.RUNNING`
  - v3 status: running_unmonitored

## 2026-06-27 v3 完了結果

- ユーザー完了連絡後、latest kernel output を取得した。version 指定なしの output 取得は最新 v3 を返すため、local output は `experiments/exp142_trajectory_aware_pf_transition_prior/kaggle/output/train_v3` として保存した。
- 取得コマンド:

```bash
kaggle kernels logs kentookumura/exp142-traj-pf-trans-prior-train
kaggle kernels output kentookumura/exp142-traj-pf-trans-prior-train -p experiments/exp142_trajectory_aware_pf_transition_prior/kaggle/output/train_v1
kaggle kernels output kentookumura/exp142-traj-pf-trans-prior-train -p experiments/exp142_trajectory_aware_pf_transition_prior/kaggle/output/train_v1 --force --file-pattern '.*(summary\.json|strict_pf_z_quality\.csv|trajectory_pf_z_quality\.csv|parity_diff\.csv\.gz)$'
mv experiments/exp142_trajectory_aware_pf_transition_prior/kaggle/output/train_v1 experiments/exp142_trajectory_aware_pf_transition_prior/kaggle/output/train_v3
```

- full output download は `candidate_wide.csv.gz` が大きいため途中停止した。必要な `candidate_metrics`、`bucket_metrics`、`by_well`、`summary.json`、`strict_pf_z_quality`、`trajectory_pf_z_quality`、`parity_diff` は取得済み。`candidate_wide.csv.gz` は Kaggle 側では生成済みで summary に SHA を記録しているが、ローカルには完全取得していない。
- rows / wells: 3,783,989 rows / 773 wells
- runtime: 13,936.071507 sec
- parity: PASS。`max_abs_diff=0.0` / `rmse_diff=0.0`
- best overall: `exp072_likpf_mean` RMSE 11.594897672 / MAE 7.067632584 / within10 0.772807479
- best strict multiseed: `pf_z_ms_scale_8` RMSE 16.871842167 / within10 0.693143664
- best trajectory-aware PF: `traj_pf_zmean_s006_noise025_scale_3` RMSE 23.132450295 / MAE 12.927040469 / within10 0.614682812
- trajectory best delta:
  - vs `likpf_mean`: +11.537552623 RMSE
  - vs `exp072_pf_z`: +5.344279124 RMSE
- bucket finding:
  - near 0-50 ft: trajectory best RMSE 0.550729 vs `likpf_mean` 1.188878
  - 50-100 ft: trajectory best 1.266820 vs `likpf_mean` 1.925625
  - 100-250 ft: trajectory best 2.629969 vs `likpf_mean` 2.934160
  - 1000+ ft: trajectory best 25.710669 vs `likpf_mean` 12.704015
- PF diagnostics:
  - `zmean_s006_noise025`: mean ESS frac 0.732249 / min 0.003540 / mean resample count 249.903 / collapse rate 0.110213
  - `zacc_s010_a020_noise050`: mean ESS frac 0.697873 / min 0.001667 / mean resample count 517.725 / collapse rate 0.120437
  - `zacc_s014_a035_noise075`: mean ESS frac 0.681301 / min 0.001667 / mean resample count 637.089 / collapse rate 0.136930
- conclusion: completed_train_side_rejected_no_submit。Z trajectory を transition prior に直接入れる設計は longtail を壊すため、raw-test inference port / submit はしない。Z は near-prefix guard、confidence feature、または local mode/correlation diagnostic に限定する。

## 次のアクション

1. `trajectory_aware_pf_transition_prior` backlog は完了/不採用として閉じる。
2. 後続は direct transition ではなく、`exp143_multimode_pfbeam_local_correlation_audit` のような mode diversity / local correlation 診断か、near-prefix confidence feature に分ける。
