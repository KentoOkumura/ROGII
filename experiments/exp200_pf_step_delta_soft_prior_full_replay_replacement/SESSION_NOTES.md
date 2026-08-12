# exp200_pf_step_delta_soft_prior_full_replay_replacement セッションノート

## 2026-07-05 実装

- ユーザー依頼により `pf_step_delta_soft_prior_full_replay_replacement` の実装を開始。
- `docs/legacy/steering/20260705-exp200-pf-step-delta-soft-prior-full-replay-replacement/` を作成。
- `experiments/exp200_pf_step_delta_soft_prior_full_replay_replacement/` を `exp186` から scaffold し、実装本体は exp072 baseline full replay code を `step_delta_public_replay.py` として取り込み直した。
- Route: `pf_beam`
- GPU 学習: なし
- active feature cache variant 数: 1 (`delta_free010_cost0025_scale003`)
- configured follow-up variant: `delta_free008_cost005_scale003` は inactive。full replay runtime / output size を倍にしないため、初回 default run には含めない。
- LightGBM config 数: 0
- fold 数: 0
- 合計 booster 数: 0
- control / parent 再学習: なし
- inference / submit: なし

## 実装内容

- `step_delta_public_replay.py`
  - exp072 baseline の full replay builder をベースにする。
  - `PF_ANCC`、`PF_Z`、128-seed likelihood-PF の particle likelihood に per-step TVT delta soft prior を追加。
  - Beam search は exp072 baseline のまま変更しない。
  - likelihood-PF の `pf_ess_mean` と `pf_resampling_rate` を generation summary に入れる。
- `feature_cache.py`
  - exp200 用の output prefix / variant に変更。
  - selected step-delta prior を runtime config として渡す。
- `direct_pfbeam_comparison.py`
  - 生成後に exp072 cache と `id` 完全一致を確認する。
  - `pf_ancc`、`pf_z`、`beam_mean`、`beam_cons`、`beam_sm5`、`beam_med`、`likpf_mean` を overall / distance bucket / by-well / step-delta rate で比較する。

## prior contract

- selected: `delta_free010_cost0025_scale003`
- formula:
  - `dtvt = current_tvt - previous_particle_tvt`
  - `excess = max(0, abs(dtvt) - 0.10)`
  - `prior = 0.025 * (excess / 0.03)^2`
  - `likelihood *= exp(-prior)`

## 再現性メモ

- seed policy: exp072 と同じ `stable_seed("pf_ancc" / "pf_z" / "likpf", split, well)`。
- stochastic components: PF particle propagation / resampling、128-seed likelihood-PF。
- parallel RNG policy: per-well stable seed を使うため joblib thread scheduling に依存しない。
- Beam: deterministic、かつ今回変更なし。
- CPU/GPU runtime: Kaggle CPU、`enable_gpu=false`。
- deterministic anchor: false。train feature cache generation であり submission anchor ではない。
- gzip 生成物は raw gzip SHA と decompressed content SHA を分け、decompressed content SHA を主証拠にする。

## push 前コスト確認

- Runtime: CPU (`enable_gpu=false`)
- active feature cache variant 数: 1
- PF seeds / particles: 128 seeds x 500 particles
- target wells: all train horizontal wells (`max_wells: null`)
- expected rows: 3,783,989
- expected feature_count: 196
- LightGBM config 数: 0
- fold 数: 0
- total boosters: 0
- parent/control 再学習: なし

## 次のアクション

1. validation / Jupytext 変換を通す。
2. Kaggle train を push する場合は、canonical kernel id を `kentookumura/exp200-pf-step-delta-prior-train` にする。
3. Kaggle output 取得後、`direct_pfbeam_comparison.py` を実行して `metrics.json`、`result.md`、`experiment_summary.md`、`KAGGLE_DIRECTION.md` を更新する。

## local validation

```bash
.venv/bin/python -m py_compile \
  experiments/exp200_pf_step_delta_soft_prior_full_replay_replacement/step_delta_public_replay.py \
  experiments/exp200_pf_step_delta_soft_prior_full_replay_replacement/feature_cache.py \
  experiments/exp200_pf_step_delta_soft_prior_full_replay_replacement/direct_pfbeam_comparison.py \
  experiments/exp200_pf_step_delta_soft_prior_full_replay_replacement/settings.py \
  experiments/exp200_pf_step_delta_soft_prior_full_replay_replacement/exp200_pf_step_delta_soft_prior_full_replay_replacement_train.py \
  experiments/exp200_pf_step_delta_soft_prior_full_replay_replacement/exp200_pf_step_delta_soft_prior_full_replay_replacement_inference.py
```

- result: PASS

```bash
.venv/bin/ruff check \
  experiments/exp200_pf_step_delta_soft_prior_full_replay_replacement/step_delta_public_replay.py \
  experiments/exp200_pf_step_delta_soft_prior_full_replay_replacement/feature_cache.py \
  experiments/exp200_pf_step_delta_soft_prior_full_replay_replacement/direct_pfbeam_comparison.py \
  experiments/exp200_pf_step_delta_soft_prior_full_replay_replacement/settings.py \
  experiments/exp200_pf_step_delta_soft_prior_full_replay_replacement/exp200_pf_step_delta_soft_prior_full_replay_replacement_train.py \
  experiments/exp200_pf_step_delta_soft_prior_full_replay_replacement/exp200_pf_step_delta_soft_prior_full_replay_replacement_inference.py \
  --select F821
```

- result: PASS

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp200_pf_step_delta_soft_prior_full_replay_replacement/exp200_pf_step_delta_soft_prior_full_replay_replacement_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp200_pf_step_delta_soft_prior_full_replay_replacement/exp200_pf_step_delta_soft_prior_full_replay_replacement_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp200_pf_step_delta_soft_prior_full_replay_replacement/exp200_pf_step_delta_soft_prior_full_replay_replacement_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp200_pf_step_delta_soft_prior_full_replay_replacement/exp200_pf_step_delta_soft_prior_full_replay_replacement_inference.py
```

- result: PASS

```bash
uv run python scripts/validate_experiment.py --experiment exp200_pf_step_delta_soft_prior_full_replay_replacement
```

- result: PASS
- 2026-07-05 pre-Kaggle check: この時点では実装済み・評価待ちとして `KAGGLE_DIRECTION.md` に記録した。後続の Kaggle train / direct comparison 完了により、この扱いは下の最終判定で supersede された。
- 2026-07-05 final check: `uv run python scripts/validate_experiment.py --experiment exp200_pf_step_delta_soft_prior_full_replay_replacement` を再実行し、strict validation PASS。

## Kaggle train v1

```bash
make prepare-kaggle-notebooks EXP=exp200_pf_step_delta_soft_prior_full_replay_replacement EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp200-pf-step-delta-soft-prior-full-replay-replacement-train --title 'exp200 pf step delta soft prior full replay replacement train' --run-on-push --strict"
kaggle kernels push -p experiments/exp200_pf_step_delta_soft_prior_full_replay_replacement/kaggle/train
```

- result: `SaveKernel` 400。package は小さく、metadata JSON も正常だったが、slug/title が 61 chars で既存 kernel slug より長いため、Kaggle 側 slug 制約の可能性が高いと判断。
- 方針: 同じ exp のまま短縮 canonical slug/title に寄せる。以降の train kernel id は `kentookumura/exp200-pf-step-delta-prior-train`。

```bash
make prepare-kaggle-notebooks EXP=exp200_pf_step_delta_soft_prior_full_replay_replacement EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp200-pf-step-delta-prior-train --title 'exp200 pf step delta prior train' --run-on-push --strict"
kaggle kernels push -p experiments/exp200_pf_step_delta_soft_prior_full_replay_replacement/kaggle/train
```

- result: `Kernel version 1 successfully pushed`
- URL: `https://www.kaggle.com/code/kentookumura/exp200-pf-step-delta-prior-train`

```bash
kaggle kernels pull kentookumura/exp200-pf-step-delta-prior-train -p /tmp/kaggle-pull/exp200-pf-step-delta-prior-train -m
kaggle kernels logs kentookumura/exp200-pf-step-delta-prior-train
kaggle kernels status kentookumura/exp200-pf-step-delta-prior-train
```

- pull: success。
- logs: initial CLI logs は warning のみで本文空。実行中 logs が空のことは既知なので失敗扱いにしない。
- push 直後 status: running。最終 status は下の「Kaggle train v1 完了」を正とする。

## Kaggle train v1 完了

```bash
kaggle kernels status kentookumura/exp200-pf-step-delta-prior-train
kaggle kernels output kentookumura/exp200-pf-step-delta-prior-train -p /tmp/kaggle-output/exp200_pf_step_delta_soft_prior_full_replay_replacement/train_v1_small --file-pattern '.*(summary|feature_schema).*'
```

- status: COMPLETE
- train kernel: `kentookumura/exp200-pf-step-delta-prior-train` v1
- URL: `https://www.kaggle.com/code/kentookumura/exp200-pf-step-delta-prior-train`
- summary status: `train_feature_cache_completed`
- variant: `pixiux_likpf_step_delta_prior_public_replay`
- rows / wells / feature_count: 3,783,989 / 773 / 196
- elapsed seconds: 14,719.448
- feature elapsed seconds: 13,059.303
- raw gzip SHA256: `f2bc3026bcb1491716fcf8845a158badff8f229a7b4a124659cbbc7bde032233`
- likelihood-PF diagnostics: `pf_ess_mean_mean=372.6444300040633`, `pf_resampling_rate_mean=0.021144307205656018`, `pf_ll_spread_mean=20.46068370312951`
- full gzip output は Kaggle 上に存在し、comparison notebook の input として使用できた。
- local full gzip download は connection reset で複数回失敗したため、decompressed content SHA は未計算。小さい summary/schema/log と comparison artifacts を local artifacts に保存した。

## Kaggle direct comparison v5

```bash
kaggle kernels output kentookumura/exp200-pf-step-delta-prior-comparison -p /tmp/kaggle-output/exp200_pf_step_delta_soft_prior_full_replay_replacement/comparison_v5 --file-pattern '.*exp200_vs_exp072_.*'
```

- comparison kernel: `kentookumura/exp200-pf-step-delta-prior-comparison`
- authoritative version: v5
- v1: kernelspec なしで失敗。
- v2/v3: self-contained notebook ではない import/package 問題で失敗。
- v4: 完了したが `well` dtype inference により `unique_wells=779` になったため、by-well 集計は採用しない。
- v5: `id` と `well` を `str` dtype 固定にして完了。以降の記録は v5 を正とする。
- rows_checked / unique_wells / id_mismatches: 3,783,989 / 773 / 0

Overall:

| candidate | exp072 RMSE | exp200 RMSE | delta RMSE |
| --- | ---: | ---: | ---: |
| `pf_ancc` | 14.493061 | 14.736794 | +0.243733 |
| `pf_z` | 17.788174 | 17.662427 | -0.125747 |
| `beam_mean` | 15.774328 | 15.774328 | +0.000000 |
| `likpf_mean` | 11.594898 | 11.618341 | +0.023444 |

- guard: `likpf_mean` の許容悪化は +0.02 RMSE。実測 +0.0234436847 で fail。
- `likpf_mean` は MAE -0.083430、within10 +0.002485 と改善したが、RMSE guard を満たさない。
- `likpf_mean` bucket delta RMSE: `000_050=-0.848621`, `050_100=-0.982801`, `100_250=-0.694795`, `250_500=-0.643344`, `500_1000=-0.378803`, `1000_plus=+0.073129`。
- by-well `likpf_mean`: 426 improved / 347 worsened。最大悪化は well `70925e23` の +24.605219 RMSE。
- 判定: completed / rejected / no submit。

Local artifacts:

- `artifacts/exp200_pf_step_delta_soft_prior_full_replay_replacement_full_replay_cache_summary.json`
- `artifacts/exp200_pf_step_delta_soft_prior_full_replay_replacement_full_replay_cache_feature_schema.csv`
- `artifacts/exp200-pf-step-delta-prior-train.log`
- `artifacts/exp200_vs_exp072_overall_metrics.csv`
- `artifacts/exp200_vs_exp072_distance_bucket_metrics.csv`
- `artifacts/exp200_vs_exp072_by_well_delta.csv`
- `artifacts/exp200_vs_exp072_step_delta_rates.csv`
- `artifacts/exp200_vs_exp072_summary.json`
- `artifacts/exp200-pf-step-delta-prior-comparison.log`
