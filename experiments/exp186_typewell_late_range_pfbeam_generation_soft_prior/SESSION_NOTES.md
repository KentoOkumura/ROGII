# exp186_typewell_late_range_pfbeam_generation_soft_prior セッションノート

## 2026-07-04 実装

- ユーザー依頼により `typewell_late_range_pfbeam_generation_soft_prior` の実装を開始。
- `docs/legacy/steering/20260704-exp186-typewell-late-range-pfbeam-generation-soft-prior/` を作成。
- `experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/` を新規作成。
- Route: `pf_beam`
- GPU 学習: なし
- active variant 数: 4 soft-prior variants (`no_prior` を含む)
- LightGBM config 数: 0
- fold 数: 0
- 合計 booster 数: 0
- control / parent 再学習: なし
- inference / submit: なし

## 実装内容

- raw train well の既知 `TVT_input` prefix 末尾を人工的に mask し、同じ horizontal/typewell から PF と Beam を再生成する train-side audit を追加した。
- `typewell_late_range_pfbeam_generation_soft_prior.py` を追加した。
  - PF particle likelihood に typewell range `candidate_pct` soft penalty を掛ける。
  - Beam path cost に同じ soft penalty を足す。
  - PF は per-well/per-seed stable SHA seed を使い、global RNG に依存しない。
  - 出力: candidate metrics、bucket metrics、by-well、group metrics、PF diagnostics、well status、row candidates。
- train / inference notebook の正となる Jupytext `.py` を追加した。
- inference notebook は train-side audit only として明示的に no-op。

## 再現性メモ

- seed policy: `sha256(experiment, well, "pf", seed_index)` から stable seed を作る。
- stochastic components: PF particle propagation / resampling。
- Beam: deterministic。
- CPU/GPU runtime: Kaggle CPU、`enable_gpu=false`。
- deterministic anchor: false。train-side diagnostic であり submission anchor ではない。
- model manifest / model SHA: not applicable。
- submission SHA: not applicable。

## コマンドログ

```bash
make new-steering EXP=exp186_typewell_late_range_pfbeam_generation_soft_prior
make new-exp EXP=exp186_typewell_late_range_pfbeam_generation_soft_prior
```

- result: PASS

```bash
python3 -m py_compile \
  experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/typewell_late_range_pfbeam_generation_soft_prior.py \
  experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/settings.py \
  experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/exp186_typewell_late_range_pfbeam_generation_soft_prior_train.py \
  experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/exp186_typewell_late_range_pfbeam_generation_soft_prior_inference.py
```

- result: PASS

```bash
.venv/bin/ruff check \
  experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/typewell_late_range_pfbeam_generation_soft_prior.py \
  experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/settings.py \
  experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/exp186_typewell_late_range_pfbeam_generation_soft_prior_train.py \
  experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/exp186_typewell_late_range_pfbeam_generation_soft_prior_inference.py
.venv/bin/ruff format --check \
  experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/typewell_late_range_pfbeam_generation_soft_prior.py \
  experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/settings.py \
  experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/exp186_typewell_late_range_pfbeam_generation_soft_prior_train.py \
  experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/exp186_typewell_late_range_pfbeam_generation_soft_prior_inference.py
```

- result: PASS

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/exp186_typewell_late_range_pfbeam_generation_soft_prior_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/exp186_typewell_late_range_pfbeam_generation_soft_prior_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/exp186_typewell_late_range_pfbeam_generation_soft_prior_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/exp186_typewell_late_range_pfbeam_generation_soft_prior_inference.py
```

- result: PASS

```bash
uv run python scripts/validate_experiment.py --experiment exp186_typewell_late_range_pfbeam_generation_soft_prior
```

- result: PASS

```bash
PYTHONPATH=experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior .venv/bin/python - <<'PY'
...
PY
```

- result: PASS
- synthetic holdout output: `(4, 16) 1 True True`
- 目的: 実データ train ではなく、in-memory の極小 holdout で PF / Beam helper API が finite output を返すことだけ確認した。
- `experiment_summary.md` に exp186 の implemented_pending_kaggle_train 行と主な発見を追記した。
- `backlog/KAGGLE_DIRECTION.md` に exp186 実装済み / Kaggle train 未実行の注記を追加した。backlog 自体は結果未評価のため削除していない。

## 次のアクション

1. Kaggle train を push する場合は、CPU runtime / variant 数 / booster 0 を再確認してから package を準備する。
2. train completion 後に metrics、bucket、PF diagnostics、row candidate SHA を記録する。

## 2026-07-04 Kaggle train push

### push 前コスト確認

- Runtime: CPU (`enable_gpu=false`)
- active soft-prior variant 数: 4 (`no_prior` を含む)
- PF particles / seeds: 260 particles x 10 seeds
- target wells: max 64
- Beam config: beam_size 14 / move_radius 2
- LightGBM config 数: 0
- fold 数: 0
- 合計 booster 数: 0
- control / parent 再学習: なし
- inference / submit: なし

```bash
make prepare-kaggle-notebooks EXP=exp186_typewell_late_range_pfbeam_generation_soft_prior EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp186-typewell-late-soft-pfbeam-train --title 'exp186 typewell late soft pfbeam train' --run-on-push --strict"
```

- result: PASS
- output: `experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/kaggle/train`
- kernel id: `kentookumura/exp186-typewell-late-soft-pfbeam-train`
- title: `exp186 typewell late soft pfbeam train`
- metadata `enable_gpu`: false
- metadata `enable_internet`: false
- competition source: `rogii-wellbore-geology-prediction`
- kernel sources: none

```bash
make push-kaggle-train EXP=exp186_typewell_late_range_pfbeam_generation_soft_prior
```

- result: PASS
- Kaggle kernel version: v1
- URL: https://www.kaggle.com/code/kentookumura/exp186-typewell-late-soft-pfbeam-train

```bash
kaggle kernels pull kentookumura/exp186-typewell-late-soft-pfbeam-train -p /tmp/kaggle-pull/exp186-typewell-late-soft-pfbeam-train -m
kaggle kernels status kentookumura/exp186-typewell-late-soft-pfbeam-train
timeout 600 kaggle kernels logs -f --interval 30 kentookumura/exp186-typewell-late-soft-pfbeam-train
```

- pull: PASS
- metadata `id_no`: 125873124
- status: `KernelWorkerStatus.COMPLETE`
- runtime from notebook summary: 140.088093 sec
- CLI log: support files restored, train input found, generated artifacts printed.

```bash
kaggle kernels output kentookumura/exp186-typewell-late-soft-pfbeam-train -p /tmp/kaggle-output/exp186_typewell_late_range_pfbeam_generation_soft_prior/train_v1
```

- result: PASS
- output archive を取得した理由: CLI logs には numeric tables が出ず、metrics / SHA / generated CSV をローカル確認する必要があったため。
- copied output artifacts to `experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/artifacts/`
- copied Kaggle `metrics.json` to local `metrics.json`

### Kaggle train v1 結果

- status: completed_train_side_rejected_no_submit
- rows: 12,288
- wells: 64
- variants: `no_prior`, `pct50_strong2_pct70_weak0p5`, `pct50_strong4_pct70_weak1`, `pct50_strong8_pct70_weak2`
- best non-oracle: `pf_no_prior_lik_mean`
  - RMSE 2.367781676
  - MAE 1.660929680
  - within10 1.0
  - path_jump_rate 0.005127
- best soft-prior non-oracle: `pf_pct50_strong2_pct70_weak0p5_lik_mean`
  - RMSE 2.501685767
  - delta vs baseline +0.133904092
  - max well regression +3.282318859
- strong soft-prior `pf_pct50_strong8_pct70_weak2_lik_mean`:
  - RMSE 4.314599346
  - delta vs baseline +1.946817671
  - max well regression +16.631115910
- Beam best soft-prior `beam_pct50_strong8_pct70_weak2_top1`:
  - RMSE 2.844968502
  - delta vs PF baseline +0.477186827
- topK oracle diagnostic `pf_no_prior_top3_oracle`:
  - RMSE 2.322535637
  - delta vs baseline -0.045246039
  - 真値を使う oracle diagnostic なので採用候補ではない。

Subgroup:

- `candidate_pct_baseline_lt_0p70`: no-prior PF RMSE 2.035520、weak soft prior RMSE 3.454565、strong soft prior RMSE 13.009466。
- `late_prefix_ge_0p75`: PF soft-prior variants は no-prior と同値で、penalty が有効に働いた subset ではない。
- `near_000_050`: Beam は no-prior 1.190801、strong prior 1.164089 と微改善だが、PF no-prior 1.489632 に対する全体採用根拠にはならない。

PF diagnostics:

- `no_prior`: ESS mean 189.679496、resampling_rate 0.055697、log_likelihood_mean -26.834601
- weak prior: ESS mean 189.598540、resampling_rate 0.056299、log_likelihood_mean -30.918355
- medium prior: ESS mean 189.787043、resampling_rate 0.056429、log_likelihood_mean -34.729683
- strong prior: ESS mean 189.695350、resampling_rate 0.056860、log_likelihood_mean -41.776490

### 生成物 SHA

- candidate_metrics: `b98ba6bc93cf56ddc7d04895b17a2e87fc2732aeb72b6131cae94d6b48a1b6d4`
- bucket_metrics: `98b21528e3435734453b584a3cb3c706cafafc28a67c52cffea136ce82e651d4`
- by_well: `4bc08adfc57535ca65c381cc0d74810e34711d0dac0b14c5d5a1eed1fa2fe579`
- group_metrics: `73ef542d1815dc6d0d9bd580d335d01621ed1be6449883ea50ac2c2f6217c4d9`
- pf_diagnostics: `ab3aefd27e9b1b5ffcd4314d8ca99e51a85b6b02f8e7d1deefbe72ee3f5cb68c`
- well_status: `702682dd23d4294c41b35866a0a4c9c4c07c809023b5549dd653c9bc4984b0a4`
- row_candidates raw gzip: `2a7ba40d68e5e33d8fc418abdc78bbe7fb42e87c12ea6b9b8d0620880855dad9`
- row_candidates decompressed: `1793893c95187e5ef978113438f8c80461fd707bfaf44e4dd09ac5864c451b5e`
- summary: `33ed9d0e327ade060be793653a9909c5191b91197a831c75ce93b56343f648ce`

### 判断

- `typewell_late_range_pfbeam_generation_soft_prior` は scoped audit では不採用。
- PF/Beam generation への soft prior、hard invalid、clip、inference port、submit はしない。
- late-range signal を続ける場合は exp176 continuity selector または ML / selector confidence feature に限定する。

## 2026-07-04 Kaggle train v2 all-well rerun

User request: v1 の実行時間が早すぎる理由を確認したところ、`max_wells: 64` の scoped audit だったため、全 well 実行に変更する。

### v2 push 前コスト確認

- Runtime: CPU (`enable_gpu=false`)
- active soft-prior variant 数: 4 (`no_prior` を含む)
- PF particles / seeds: 260 particles x 10 seeds
- target wells: all horizontal wells (`max_wells: null`)
- local raw train horizontal well files: 773
- expected scored rows: up to 773 x 192 = 148,416 rows, depending on per-well minimum prefix eligibility
- Beam config: beam_size 14 / move_radius 2
- LightGBM config 数: 0
- fold 数: 0
- 合計 booster 数: 0
- control / parent 再学習: なし
- inference / submit: なし
- runtime estimate from v1: 64 wells / 140.088 sec -> all-well simple scale about 28.2 minutes, plus Kaggle overhead

Config change:

- `experiment.description`: scoped audit -> all-well audit
- `experiment.status`: `running_kaggle_train_v2_all_wells`
- `model.prefix_holdout.max_wells`: `64` -> `null`

### Kaggle train v2 completion

```bash
kaggle kernels status kentookumura/exp186-typewell-late-soft-pfbeam-train
kaggle kernels logs kentookumura/exp186-typewell-late-soft-pfbeam-train
kaggle kernels pull kentookumura/exp186-typewell-late-soft-pfbeam-train -p /tmp/kaggle-pull/exp186-typewell-late-soft-pfbeam-train-v2-complete -m
kaggle kernels output kentookumura/exp186-typewell-late-soft-pfbeam-train -p /tmp/kaggle-output/exp186_typewell_late_range_pfbeam_generation_soft_prior/train_v2
```

- status: `KernelWorkerStatus.COMPLETE`
- Kaggle kernel version: v2
- metadata `id_no`: 125873124
- URL: https://www.kaggle.com/code/kentookumura/exp186-typewell-late-soft-pfbeam-train
- notebook summary runtime: 1734.081479 sec
- CLI log last summary timestamp: about 1747 sec
- output archive を取得した理由: metrics / SHA / generated CSV を全well結果としてローカル確認する必要があったため。
- copied v2 output artifacts to `experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/artifacts/`
- copied Kaggle v2 `metrics.json` to local `metrics.json`

### Kaggle train v2 all-well 結果

- status: completed_train_side_rejected_no_submit
- rows: 148,416
- wells: 773
- rows per well: 192
- well status: `ok` 773
- scope caveat: this is an all-well lightweight prefix-holdout audit, not the same condition as exp072 / exp073 full replay cache generation. It regenerates only 192 masked known-prefix suffix rows per well, with 260 particles x 10 seeds and a single Beam config per variant. It does not regenerate 3,783,989 OOF rows, 128-seed likelihood-PF, multi-scale PF, Beam ensemble, or the full feature cache used by previous multi-hour runs.
- variants: `no_prior`, `pct50_strong2_pct70_weak0p5`, `pct50_strong4_pct70_weak1`, `pct50_strong8_pct70_weak2`
- best non-oracle: `pf_no_prior_lik_mean`
  - RMSE 3.563341582
  - MAE 2.261443329
  - within10 0.977657395
  - path_jump_rate 0.005188
- best soft-prior non-oracle: `pf_pct50_strong2_pct70_weak0p5_lik_mean`
  - RMSE 3.726103563
  - delta vs baseline +0.162761982
  - max well regression +9.719488
- strong soft-prior `pf_pct50_strong8_pct70_weak2_lik_mean`:
  - RMSE 4.238622094
  - delta vs baseline +0.675280512
  - max well regression +16.631116
- Beam best soft-prior `beam_pct50_strong8_pct70_weak2_top1`:
  - RMSE 4.337859056
  - delta vs PF baseline +0.774517475
  - max well regression +42.867263
- topK oracle diagnostic `pf_no_prior_top3_oracle`:
  - RMSE 3.282851440
  - delta vs baseline -0.280490141
  - 真値を使う oracle diagnostic なので採用候補ではない。

Subgroup:

- `candidate_pct_baseline_lt_0p70`: 8,446 rows。no-prior PF RMSE 3.042562667、weak soft prior RMSE 5.440496413、strong soft prior RMSE 10.043615437。
- `late_prefix_ge_0p75`: 126,720 rows。PF soft-prior variants は no-prior と同値で、penalty が有効に働いた subset ではない。
- `near_000_050`: Beam は no-prior 1.472783436、strong prior 1.471564121 と微改善だが、PF no-prior 2.044080421 に対する全体採用根拠にはならない。
- changed rows: PF weak 12,626、PF strong 12,786、Beam strong 773。

PF diagnostics:

- `no_prior`: ESS mean 190.207875、resampling_rate 0.055271、log_likelihood_mean -30.299095
- weak prior: ESS mean 190.172450、resampling_rate 0.055517、log_likelihood_mean -31.887999
- medium prior: ESS mean 190.219425、resampling_rate 0.055671、log_likelihood_mean -33.336260
- strong prior: ESS mean 190.219569、resampling_rate 0.055878、log_likelihood_mean -36.094450

### v2 生成物 SHA

- candidate_metrics: `73a83709f1dfe5afe4bafedb6fa749909ae23acf436bd412f2568d905d31023f`
- bucket_metrics: `8b2c5841cb0a804f936165dded537084eacf658e889db346325ebfd08e99e5a6`
- by_well: `1f41041008997bb13572b044a707286fcc065e0d342bace1ab724728a38ea96e`
- group_metrics: `5dbdac1673d2e63890575b5ac21ab3f02fbaeee026df68ab6ca7a34c1c1c65b8`
- pf_diagnostics: `71f8fbd7b1fff5d56f725db679c53cff19948759fa02a3d8575a153a3c51b352`
- well_status: `6cb49929ba493938653a9ec257a286754641aa4c3d75c8aa3ecd29d728b8352e`
- row_candidates raw gzip: `311a15bba4058dfe6739797671ae20b452af82dd090fa425b1ed68709eb5fd81`
- row_candidates decompressed: `977348bdbbaacb3adc322d9ffee2f28ee34c3410850a1cb5636b7be2b2f3d815`
- summary: `778091cc1e5d4cf998e86b0c4643dc7f34e73d57fa37bea7107d806d3f793430`

### v2 判断

- `typewell_late_range_pfbeam_generation_soft_prior` は all-well audit でも不採用。
- PF/Beam generation への soft prior、hard invalid、clip、inference port、submit はしない。
- late-range signal を続ける場合は exp176 continuity selector または ML / selector confidence feature に限定する。

## 2026-07-04 corrected full replay cache implementation

### ユーザー指摘による scope 修正

- ユーザー指摘: 既存 full replay cache は入力ではなく、raw well/typewell から full replay cache を作り直すのが意図。
- 追加確認: soft prior という処理自体は backlog 名どおり許容。外していた主因は、前回実装が 192-row prefix-holdout audit であり、exp072-style full replay cache 出力ではなかったこと。
- 対応:
  - exp072 の full replay builder を exp186 内に `late_soft_prior_public_replay.py` として取り込み。
  - `feature_cache.py` を exp186 用 full replay train cache wrapper に差し替え。
  - `exp186_typewell_late_range_pfbeam_generation_soft_prior_train.py` / `.ipynb` を full replay train feature cache 生成に差し替え。
  - 既存 `typewell_late_range_pfbeam_generation_soft_prior.py` は superseded lightweight audit source として残す。

### 実装内容

- 入力: raw competition train horizontal/typewell files under `data/raw/train`。
- 既存 full replay cache: generation input としては読まない。比較対象 / downstream baseline としてのみ扱う。
- 出力:
  - `exp186_typewell_late_range_pfbeam_generation_soft_prior_full_replay_cache_pixiux_likpf_late_soft_prior_public_replay_train_features.csv.gz`
  - `exp186_typewell_late_range_pfbeam_generation_soft_prior_full_replay_cache_feature_schema.csv`
  - `exp186_typewell_late_range_pfbeam_generation_soft_prior_full_replay_cache_summary.json`
- selected soft prior: `pct50_strong2_pct70_weak0p5`
  - weak pct 0.70 / weak penalty 0.5
  - strong pct 0.50 / strong penalty 2.0
  - known_last_pct_threshold 0.75 / known_last_multiplier 1.0
- prior insertion points:
  - `PF_ANCC`
  - `PF_Z`
  - Beam path cost
  - 128-seed likelihood-PF

### Kaggle train push 前確認

- route: `pf_beam`
- GPU: false / CPU-only
- feature cache variants: 1
- selected soft-prior variant: 1
- LightGBM configs: 0
- folds: 0
- total boosters: 0
- PF seeds: 128
- PF particles: 500
- max_wells: null
- 親実験 control / baseline の再学習: なし

### local validation

```bash
.venv/bin/python -m py_compile \
  experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/late_soft_prior_public_replay.py \
  experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/feature_cache.py \
  experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/exp186_typewell_late_range_pfbeam_generation_soft_prior_train.py \
  experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/exp186_typewell_late_range_pfbeam_generation_soft_prior_inference.py

.venv/bin/ruff check \
  experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/late_soft_prior_public_replay.py \
  experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/feature_cache.py \
  experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/exp186_typewell_late_range_pfbeam_generation_soft_prior_train.py \
  experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/exp186_typewell_late_range_pfbeam_generation_soft_prior_inference.py \
  --select F821

JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/exp186_typewell_late_range_pfbeam_generation_soft_prior_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/exp186_typewell_late_range_pfbeam_generation_soft_prior_inference.py
uv run python scripts/validate_experiment.py --experiment exp186_typewell_late_range_pfbeam_generation_soft_prior
```

- status: passed
- local JIT import caveat: local Python/uv environment lacks `numba`; Numba-backed import is deferred to Kaggle runtime.

### Kaggle train v3 corrected full replay push

```bash
make prepare-kaggle-notebooks EXP=exp186_typewell_late_range_pfbeam_generation_soft_prior EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp186-typewell-late-soft-pfbeam-train --title 'exp186 typewell late soft pfbeam train' --run-on-push --strict"
make push-kaggle-train EXP=exp186_typewell_late_range_pfbeam_generation_soft_prior
```

- pushed kernel version: v3
- URL: https://www.kaggle.com/code/kentookumura/exp186-typewell-late-soft-pfbeam-train
- scope: corrected full replay train cache rebuild from raw train files
- expected runtime: multi-hour possible; this should not finish as quickly as the previous lightweight audit unless it fails early.

Initial checks:

```bash
kaggle kernels pull kentookumura/exp186-typewell-late-soft-pfbeam-train -p /tmp/kaggle-pull/exp186-typewell-late-soft-pfbeam-train-v3 -m
kaggle kernels status kentookumura/exp186-typewell-late-soft-pfbeam-train
kaggle kernels logs kentookumura/exp186-typewell-late-soft-pfbeam-train
```

- metadata pull: success
- status: `KernelWorkerStatus.RUNNING`
- logs: empty while running, consistent with previous Kaggle CLI behavior in this environment.

### Kaggle train v3 corrected full replay completion

User reported completion, then checked:

```bash
kaggle kernels status kentookumura/exp186-typewell-late-soft-pfbeam-train
kaggle kernels logs kentookumura/exp186-typewell-late-soft-pfbeam-train
kaggle kernels pull kentookumura/exp186-typewell-late-soft-pfbeam-train -p /tmp/kaggle-pull/exp186-typewell-late-soft-pfbeam-train-v3-complete -m
```

- status: `KernelWorkerStatus.COMPLETE`
- kernel version: v3
- URL: https://www.kaggle.com/code/kentookumura/exp186-typewell-late-soft-pfbeam-train
- logs confirmed:
  - raw train horizontal wells: 773
  - raw train typewells: 773
  - selected soft prior: `pct50_strong2_pct70_weak0p5`
  - base features: 3,783,989 rows / 198 cols / elapsed 2,379.4 sec
  - pixiux features: 3,783,989 rows / 208 cols / elapsed 14,053.5 sec
  - final feature cache: 3,783,989 rows / 773 wells / 196 features
  - summary elapsed: 15,783.764 sec

Small output files:

```bash
kaggle kernels output kentookumura/exp186-typewell-late-soft-pfbeam-train \
  -p /tmp/kaggle-output/exp186_typewell_late_range_pfbeam_generation_soft_prior/train_v3_small \
  --file-pattern '.*(summary\.json|feature_schema\.csv)$' -o
```

- summary/schema download: success
- schema lines: 197 = header + 196 features

Large output note:

- `kaggle kernels output` for the full output and for the single large train feature gzip both exited 137 after creating a 0-byte gzip placeholder.
- Cause inspected in Kaggle CLI 2.2.0 source: `KaggleApi.kernels_output` uses `download_response.content`, loading the entire file into memory before writing.
- Workaround: use Kaggle API output signed URL and `requests.iter_content()` chunk streaming.

Streaming large output:

```bash
uv run python - <<'PY'
# Uses KaggleApi list_kernel_session_output, matches .*train_features\.csv\.gz$,
# and streams item.url to /tmp/kaggle-output/.../train_v3_stream in 8 MiB chunks.
PY
```

- downloaded gzip bytes: 2,093,362,668
- train feature raw gzip SHA: `4bb7a43278ec65143d61c3451353735093995d5258aad665b901237a6a469185`
- SHA matches Kaggle summary.

Copied v3 output to local artifacts:

- `experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/artifacts/exp186_typewell_late_range_pfbeam_generation_soft_prior_full_replay_cache_pixiux_likpf_late_soft_prior_public_replay_train_features.csv.gz`
- `experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/artifacts/exp186_typewell_late_range_pfbeam_generation_soft_prior_full_replay_cache_feature_schema.csv`
- `experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/artifacts/exp186_typewell_late_range_pfbeam_generation_soft_prior_full_replay_cache_summary.json`

Local validation:

```bash
gzip -t experiments/exp186_typewell_late_range_pfbeam_generation_soft_prior/artifacts/exp186_typewell_late_range_pfbeam_generation_soft_prior_full_replay_cache_pixiux_likpf_late_soft_prior_public_replay_train_features.csv.gz
```

- gzip integrity: pass
- header columns: 199 = `id`, `well`, `target` + 196 features
- first row id/well: `000d7d20_1442` / `000d7d20`
- decompressed bytes: 7,430,756,999
- decompressed lines: 3,783,990
- data rows: 3,783,989
- train feature decompressed SHA: `b4dd75312d91b21f55b8d1ad09a8590c6bb75857ddfbbbc84d7db175dbb75d15`
- feature schema SHA: `8c875703e3c009c74cc28430c4a8451f239f11fd4dcd3e6e55c705a5adfb7830`
- summary SHA: `8e85db2e6d48b93b2a436b160a6041e478426e8bf7ef62406b6be6e31f215c5f`

### v3 判断

- corrected scope を満たす full replay train cache rebuild は完了。
- 既存 full replay cache は generation input として使っていない。
- この実験では model training / inference / submission はしない。
- v1/v2 prefix-holdout audit は superseded として扱う。
- 改善有無は downstream で既存 exp072 cache と同条件比較して判断する。

## 2026-07-04 exp072 direct PF/Beam RMSE TVT 比較

最初の確認では plain `python3` 側に pandas がなく、さらに exp072 full replay train cache が local に存在しなかったため比較を実行できなかった。
これは計算不能という意味ではなく、実行環境と artifact 取得の問題だった。

exp072 v2 full replay train feature cache は、Kaggle CLI の通常 output download が大容量 gzip で exit 137 になるため、exp186 と同じ signed URL streaming workaround で取得した。

```bash
uv run python - <<'PY'
# KaggleApi list_kernel_session_output で exp072 train_features csv.gz を探し、
# item.url を requests.iter_content() で 8 MiB chunk streaming download。
PY
```

- exp072 kernel: `kentookumura/exp072-exp063-full-replay-feature-cache-train`
- exp072 artifact: `/tmp/kaggle-output/exp072_exp063_full_replay_feature_cache/train_v2_stream/artifacts/exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz`
- exp072 gzip bytes: 2,093,372,344
- exp072 gzip SHA: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
- exp186 rows: 3,783,989
- exp072 rows: 3,783,989
- id alignment: first `000d7d20_1442`, last `ffefef30_6420`

比較定義:

- 真値 TVT: `last_known_tvt + target`
- `pf_ancc` / `pf_z`: absolute TVT prediction として採点
- `beam_*_d` / `likpf_mean_d`: `last_known_tvt` を足して absolute TVT に戻して採点

結果:

| candidate | exp072 RMSE | exp186 RMSE | delta RMSE | exp072 MAE | exp186 MAE | delta MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pf_ancc` | 14.493061 | 14.220030 | -0.273031 | 8.921569 | 8.851547 | -0.070022 |
| `pf_z` | 17.788174 | 17.679589 | -0.108585 | 10.677493 | 10.697447 | +0.019954 |
| `beam_mean` | 15.774328 | 15.753703 | -0.020624 | 10.898586 | 10.888194 | -0.010392 |
| `beam_cons` | 16.023008 | 16.025383 | +0.002374 | 11.106713 | 11.112429 | +0.005715 |
| `beam_sm5` | 16.313542 | 16.309361 | -0.004181 | 11.300928 | 11.300965 | +0.000037 |
| `beam_med` | 15.987519 | 15.988469 | +0.000950 | 11.060277 | 11.067241 | +0.006964 |
| `likpf_mean` | 11.594898 | 12.942278 | +1.347381 | 7.067633 | 7.805225 | +0.737592 |

判断:

- `pf_ancc` は RMSE -0.273、`pf_z` は -0.109、`beam_mean` は -0.021 と小改善。
- Beam variants はほぼ横ばい。
- exp072 の最強候補 `likpf_mean` が RMSE +1.347、MAE +0.738 と大きく悪化。
- したがって exp186 soft-prior generated cache は exp072 の direct replacement として不採用。
- LightGBM 学習、inference port、submit には進めない。
