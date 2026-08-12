# exp092_u_projection_correction_disagreement_fullrun セッションノート

## 現在の状態

- status: `submitted_complete_public_lb_8_350_hidden_assert_probe_ready`
- route: `ml_model`
- parent: `exp085_u_projection_feature_ablation`
- base surface parent: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- CV: best `lgb1` pooled RMSE 9.322479896
- LB: Public LB 8.350 (`ref=53927479`)
- inference: Kaggle inference v1 complete / submit-check PASS / user manual submit complete
- blocked: none

## 実装内容

- `docs/legacy/steering/20260620-exp092-u-projection-correction-disagreement-fullrun/` を作成。
- `experiments/exp092_u_projection_correction_disagreement_fullrun/` を exp085 から作成。
- `settings.py` の experiment name を exp092 に更新。
- 補助実装を `u_projection_correction_disagreement_fullrun.py` にリネームし、出力 prefix / summary experiment 名を exp092 に更新。
- `config.yaml` を exp085 の最有望 variant fullrun 用に更新。
  - active variant は `u_projection_correction_plus_disagreement` のみ。
  - LightGBM family は `lgb0/lgb1/lgb2` の 3 model を維持。
  - target は `TVT - last_known_tvt` のまま。
  - LGB OOF U-space features は nested fold が必要なため無効。
  - comparison parent として exp073 / exp077 を記録。
- train notebook を exp092 用に更新。
- inference notebook は fullrun 結果レビューまで停止する guard notebook として更新。

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp092_u_projection_correction_disagreement_fullrun
uv run python scripts/new_experiment.py --name exp092_u_projection_correction_disagreement_fullrun --source experiments/exp085_u_projection_feature_ablation
```

## 次のアクション

1. 静的検証と synthetic smoke test を通す。
2. Kaggle train package を作成し、metadata と bootstrap manifest を確認する。
3. Kaggle train を実行して正式 pooled OOF / bucket / importance / prediction SHA を取得する。

## 検証

- `uv run python -m py_compile experiments/exp092_u_projection_correction_disagreement_fullrun/u_projection_correction_disagreement_fullrun.py experiments/exp092_u_projection_correction_disagreement_fullrun/public_notebook_replay_audit.py experiments/exp092_u_projection_correction_disagreement_fullrun/settings.py`: PASS
- `uv run ruff check experiments/exp092_u_projection_correction_disagreement_fullrun/u_projection_correction_disagreement_fullrun.py experiments/exp092_u_projection_correction_disagreement_fullrun/public_notebook_replay_audit.py experiments/exp092_u_projection_correction_disagreement_fullrun/settings.py`: PASS
- `uv run ruff format --check experiments/exp092_u_projection_correction_disagreement_fullrun/u_projection_correction_disagreement_fullrun.py experiments/exp092_u_projection_correction_disagreement_fullrun/public_notebook_replay_audit.py experiments/exp092_u_projection_correction_disagreement_fullrun/settings.py`: PASS
- `python3 -m json.tool experiments/exp092_u_projection_correction_disagreement_fullrun/exp092_u_projection_correction_disagreement_fullrun_train.ipynb`: PASS
- `python3 -m json.tool experiments/exp092_u_projection_correction_disagreement_fullrun/exp092_u_projection_correction_disagreement_fullrun_inference.ipynb`: PASS
- `uv run python scripts/validate_experiment.py --experiment exp092_u_projection_correction_disagreement_fullrun`: PASS
- synthetic frame による `build_u_projection_features()` smoke test: PASS、16 rows / 71 columns、projection correction 20、U-disagreement 24、selected feature count 48、source summary 5。
- `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp092_u_projection_correction_disagreement_fullrun --notebook train --kernel-id kentookumura/exp092-u-projection-correction-disagreement-fullrun-train --title "exp092 u projection correction disagreement fullrun train" --run-on-push --strict`: PASS
- generated train package: `experiments/exp092_u_projection_correction_disagreement_fullrun/kaggle/train`
- generated kernel id: `kentookumura/exp092-u-projection-correction-disagreement-fullrun-train`
- generated metadata: GPU enabled, internet disabled, run_on_push true, competition source `rogii-wellbore-geology-prediction`, kernel source `kentookumura/exp072-exp063-full-replay-feature-cache-train`
- generated bootstrap manifest includes:
  - `config.yaml` SHA `9a3df780938954c2ebaadf62615f30e7ff14c20f2e3b4f6274308c526130c5a9`
  - `u_projection_correction_disagreement_fullrun.py` SHA `1dfd107be19e1f33ce2d6caa15f6815f3de13f4814ef57c5c42b1b411d070ab7`
  - `public_notebook_replay_audit.py` SHA `c46da772d09595cb3ff6d1c7f04233f0522fd672a2d83f70808b8f7e0e117a60`
  - `settings.py` SHA `e661dd3efd99ce47e3b332d39122864013d28f80fcc8ac365039666239904ee1`

## Kaggle train v1

```bash
make push-kaggle-train EXP=exp092_u_projection_correction_disagreement_fullrun
kaggle kernels pull kentookumura/exp092-uproj-corr-disagree-train -p /tmp/kaggle-pull/exp092-uproj-corr-disagree-train-v1 -m
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp092-uproj-corr-disagree-train
kaggle kernels logs kentookumura/exp092-uproj-corr-disagree-train
kaggle kernels output kentookumura/exp092-uproj-corr-disagree-train -p /tmp/kaggle-output/exp092_u_projection_correction_disagreement_fullrun/train_v1_probe
kaggle kernels status kentookumura/exp092-uproj-corr-disagree-train
```

- 初回 push は long slug `kentookumura/exp092-u-projection-correction-disagreement-fullrun-train` / title `exp092 u projection correction disagreement fullrun train` で `SaveKernel` 400 になった。長すぎる slug/title の疑いが強いため、短い canonical kernel id に変更した。
- short canonical kernel id: `kentookumura/exp092-uproj-corr-disagree-train`
- short title: `exp092 uproj corr disagree train`
- version: 1
- URL: `https://www.kaggle.com/code/kentookumura/exp092-uproj-corr-disagree-train`
- push: `Kernel version 1 successfully pushed`
- pull existence check: PASS at `/tmp/kaggle-pull/exp092-uproj-corr-disagree-train-v1`
- 5 minute `logs -f --interval 20`: no output before timeout; treated as Kaggle API log lag or still-running state, not failure.
- normal logs after follow timeout: empty
- output probe: `/tmp/kaggle-output/exp092_u_projection_correction_disagreement_fullrun/train_v1_probe` contained no files yet.
- status probe: `KernelWorkerStatus.RUNNING`
- User requested to stop monitoring for now and will report when the Kaggle run completes.

## Kaggle train v1 completion

```bash
kaggle kernels status kentookumura/exp092-uproj-corr-disagree-train
kaggle kernels logs kentookumura/exp092-uproj-corr-disagree-train
kaggle kernels output kentookumura/exp092-uproj-corr-disagree-train -p experiments/exp092_u_projection_correction_disagreement_fullrun/kaggle/output/train
```

- completed status: `KernelWorkerStatus.COMPLETE`
- output: `experiments/exp092_u_projection_correction_disagreement_fullrun/kaggle/output/train/artifacts/`
- log: `experiments/exp092_u_projection_correction_disagreement_fullrun/kaggle/output/train/exp092-uproj-corr-disagree-train.log`
- runtime: 13,563.193 sec
- rows: 3,783,989
- wells: 773
- base features: 196
- model features: 240
- active mode: `gpu_repro_guard_dp_threads8`
- log confirmed `use_gpu=true`
- source cache SHA: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
- schema SHA: `700d38149f583c3ab6574ea7b163c3c8709c2514b675bea381d822f82f4809b8`
- model manifest count: 15
- OOF predictions gzip SHA: `67617007921e762868063c27e1bfa6156622c4357b3c922b6b37eda3c21235b9`
- OOF predictions decompressed SHA: `6dc3d53d6cb5621b86360929d638f2a5d853c58ea3e7a53d3da86e614e5f2f69`

### Pooled OOF metrics

| model | pooled RMSE | prediction SHA |
| --- | ---: | --- |
| `lgb1` | 9.322479896 | `dd631f28f3cfc6da3cab1ec3e939bb7185c5546e5b77c33e4425ec8080ef42e0` |
| `lgb2` | 9.338192405 | `85d0c7fd1a7482299fdf3a91527b3bea2656ac64529f2e5d778b961e6d271c56` |
| `lgb_mean` | 9.343064066 | `adbadb268707032f0ce3cd493a7a0ee81d43269574069c2c36826dd387c56a3b` |
| `lgb0` | 9.533126438 | `b98e93ae49283204cec06b12256b05326c2f7dff10a388c30e1944f0d45fc89b` |

### Readout

- Best model is `lgb1`, RMSE 9.322479896.
- exp077 policy OOF 9.470514801 から -0.148034906 改善。
- exp073 raw anchor 9.526374749 から -0.203894854 改善。
- exp085 log-derived `lgb1` 9.291006 よりは悪いが、正式 pooled OOF でも改善方向は再現した。
- `lgb_mean` は near-prefix distance 0-50 ft RMSE 1.154526、50-100 ft RMSE 1.443261。near-row は大きく壊していない。
- worst wells are still severe: `86454a6f` RMSE 57.5-57.7, `1b1eba53` RMSE 41-44, `fb03ae90` RMSE 39-40.
- Top added U-projection features in importance include `uproj_likpf_mean_resid_mad` and `uproj_pf_ancc_resid_mad`.

### Next action

Inference port へ進む前に、exp073 / exp077 OOF と exp092 predictions を align し、by-well delta、near-row delta、long-tail delta、prediction range、path continuity を guard する。guard 通過後に raw-test projection feature parity を監査し、`lgb1` 単体または `lgb1/lgb2` 近傍の inference port を検討する。

## OOF delta guard

```bash
uv run python -m py_compile experiments/exp092_u_projection_correction_disagreement_fullrun/oof_delta_guard.py
uv run ruff check experiments/exp092_u_projection_correction_disagreement_fullrun/oof_delta_guard.py
uv run ruff format --check experiments/exp092_u_projection_correction_disagreement_fullrun/oof_delta_guard.py
uv run python experiments/exp092_u_projection_correction_disagreement_fullrun/oof_delta_guard.py
```

- script: `experiments/exp092_u_projection_correction_disagreement_fullrun/oof_delta_guard.py`
- artifacts: `experiments/exp092_u_projection_correction_disagreement_fullrun/artifacts/oof_delta_guard/`
- context mode: `tail_rank_fallback_feature_cache_missing_or_empty`
- reason: local exp072 train feature cache gzip under `/tmp/kaggle-output/...` was 0 byte, so row-distance buckets use `id` tail rank.
- aligned rows: 3,783,989
- aligned wells: 773
- exp073 `lgb_mean` RMSE: 9.526374826
- exp077 policy RMSE: 9.470514801
- exp092 `lgb1` RMSE: 9.322480157
- exp092 `lgb_mean` RMSE: 9.343064070
- exp092 `lgb1` delta vs exp077 policy: -0.148034643
- exp092 `lgb1` delta vs exp073: -0.203894669
- exp092 `lgb_mean` delta vs exp077 policy: -0.127450731
- exp092 `lgb_mean` delta vs exp073: -0.183310756
- long-tail bucket `1000_plus`: 3,783,839 rows, `lgb1` delta vs exp077 -0.148037816, vs exp073 -0.203898884.
- near-row buckets `000_050`, `050_100`, `100_250`: no rows in this aligned OOF surface. Tail-rank 500-999 has only 149/150 rows and regresses by about +0.10 vs exp077, so near-row guard remains inconclusive rather than pass.
- by-well guard: `lgb1` improves vs exp077 on 459 wells and worsens on 314 wells. Max well regression vs exp077 is +4.164459617 (`b8c49c1a`), and max regression vs exp073 is +5.141955760. This exceeds the 0.25 warning threshold.
- worst regressions vs exp077: `b8c49c1a` +4.164460, `3417285d` +3.590497, `f074d277` +3.352546, `f9fc81aa` +3.007717, `86454a6f` +2.951191.
- best improvements vs exp077: `389ae58f` -5.348768, `059c8f24` -4.574808, `f6d009f4` -3.828164, `896d15b9` -3.724801, `f5859199` -3.462032.
- path continuity: `pred_exp092_lgb1_step_abs_p95` mean 0.327492 / p95 0.526094 / max 0.903320 by well; `pred_exp092_lgb1_step_abs_max` p95 4.116602 / max 10.460938; lgb1 step spikes ge10 = 1, ge25 = 0.
- correction continuity vs exp077: correction abs p95 mean 3.489870 / p95 6.794648 / max 14.631055; correction step p95 mean 0.302272 / p95 0.456641 / max 0.857471; correction step max p95 3.010937 / max 8.844727; correction step ge5 = 23.
- guard decision: overall and long-tail improve strongly, and path continuity does not show broad step collapse. However, by-well max regression is far above threshold and near-row guard is not covered by this OOF surface. Inference was later submitted per user request; targeted regression guard or gating review for the worst wells plus raw-test projection feature parity remain follow-up checks.

## Code submission attribution correction

- CLI `kaggle competitions submit` failed with submit limit 400, so the user submitted manually later.
- user correction on 2026-06-22: Public LB 8.350 belongs to exp092, not exp098.
- submission ref: `53927479`
- submitted at: `2026-06-22 00:01:47 UTC`
- Public LB: `8.350`
- selected variant: `u_projection_correction_plus_disagreement`
- selected mode: `gpu_repro_guard_dp_threads8`
- selected model: `lgb1`
- local submission SHA: not recorded in this repo output; Kaggle submission descriptions were blank.
- readout: exp092 improves exp077 submitted/postprocessed anchor 8.611 by -0.261 and exp098 8.441 by -0.091, so exp092 is now the ML route submitted anchor.
- caution remains: train-side OOF guard had by-well max-regression warning and near-row inconclusive coverage; this LB result supersedes the submit anchor but does not remove the diagnostic risk.

## Kaggle inference v1

```bash
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp092_u_projection_correction_disagreement_fullrun --notebook inference --kernel-id kentookumura/exp092-uproj-corr-disagree-infer --title "exp092 uproj corr disagree infer" --run-on-push --strict
make push-kaggle-infer EXP=exp092_u_projection_correction_disagreement_fullrun
kaggle kernels status kentookumura/exp092-uproj-corr-disagree-infer
kaggle kernels output kentookumura/exp092-uproj-corr-disagree-infer -p experiments/exp092_u_projection_correction_disagreement_fullrun/kaggle/output/inference
make submit-check EXP=exp092_u_projection_correction_disagreement_fullrun SUBMISSION=experiments/exp092_u_projection_correction_disagreement_fullrun/kaggle/output/inference/submission.csv
```

- User requested creating inference and submitting once before the worst-well guard follow-up.
- inference kernel: `kentookumura/exp092-uproj-corr-disagree-infer` v1.
- status: `KernelWorkerStatus.COMPLETE`.
- output: `experiments/exp092_u_projection_correction_disagreement_fullrun/kaggle/output/inference`.
- selected: `u_projection_correction_plus_disagreement` / `gpu_repro_guard_dp_threads8` / `lgb1`.
- fold boosters: 5.
- feature count: 240.
- test rows / submission rows: 14,151 / 14,151.
- fallback rows: 0.
- prediction min / max / mean / std: 11591.2509765625 / 12240.0703125 / 11905.651226820873 / 278.9331175018898.
- prediction SHA: `988b148fe7e6bbd5ffd1292613d53bc5242c80773787dea3d246b0eb02edc3dc`.
- submission SHA: `55a69b615f766ee4045c61c7d1c78bd418fc82152b1f7554368b1286834e5aad`.
- submit-check: PASS.

## Worst-well raw-test guard implementation

```bash
uv run python scripts/new_steering.py --experiment exp092_worst_well_rawtest_guard
python3 -m py_compile experiments/exp092_u_projection_correction_disagreement_fullrun/worst_well_rawtest_guard.py
uv run ruff check experiments/exp092_u_projection_correction_disagreement_fullrun/worst_well_rawtest_guard.py
uv run ruff format --check experiments/exp092_u_projection_correction_disagreement_fullrun/worst_well_rawtest_guard.py
uv run python experiments/exp092_u_projection_correction_disagreement_fullrun/worst_well_rawtest_guard.py --self-test
```

- steering: `docs/legacy/steering/20260622-exp092-worst-well-rawtest-guard/`
- script: `worst_well_rawtest_guard.py`
- output dir: `artifacts/worst_well_rawtest_guard/`
- mode: `target_free_raw_test_regression_guard`
- duplicate `experiments/exp092_worst_well_rawtest_guard/` は作らず、既存 exp092 の follow-up として実装した。
- visible test に target はないため、guard は CV/LB 改善や hidden LB 安全性を主張しない。OOF worst-regression profile を基準に、通常 kernel で見える exp092 inference prediction の visible-test well ごとの step continuity、optional exp073 / exp077 比 correction、prefix anchor parity、feature schema parity、projection summary parity を監査する。
- required inputs:
  - `exp092_oof_delta_guard_by_well.csv`
  - `exp092_u_projection_correction_disagreement_fullrun_inference_test_predictions.csv.gz`
- optional inputs:
  - `exp092_oof_delta_guard_path_continuity.csv`
  - train / inference feature schema
  - train / inference projection feature summary
  - exp073 inference predictions
  - exp077 inference submission
- outputs:
  - `exp092_worst_well_rawtest_guard_test_well_metrics.csv`
  - `exp092_worst_well_rawtest_guard_test_bucket_metrics.csv`
  - `exp092_worst_well_rawtest_guard_oof_worst_wells.csv`
  - `exp092_worst_well_rawtest_guard_schema_parity.csv`
  - `exp092_worst_well_rawtest_guard_projection_summary_parity.csv`
  - `exp092_worst_well_rawtest_guard_summary.json`
- self-test: PASS。合成 visible test wells / exp092 prediction / exp073 baseline / OOF guard / schema / projection summary で pass を確認した。
- static checks: `py_compile`, `ruff check`, `ruff format --check` PASS.
- visible-test guard run: completed on Kaggle v2 using exp092 train/inference kernel outputs. This is not hidden LB evidence.

## Kaggle worst-well raw-test guard v1/v2

```bash
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp092_u_projection_correction_disagreement_fullrun --notebook guard --kernel-id kentookumura/exp092-worst-well-rawtest-guard --title "exp092 worst well rawtest guard" --run-on-push --strict
kaggle kernels push -p experiments/exp092_u_projection_correction_disagreement_fullrun/kaggle/guard
timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp092-worst-well-rawtest-guard
kaggle kernels output kentookumura/exp092-worst-well-rawtest-guard -p experiments/exp092_u_projection_correction_disagreement_fullrun/kaggle/output/guard_v2
```

- Added `guard` notebook kind to `scripts/prepare_kaggle_notebooks.py`.
- Added `exp092_u_projection_correction_disagreement_fullrun_guard.ipynb`.
- Guard kernel: `kentookumura/exp092-worst-well-rawtest-guard`
- v1 completed, but optional `exp077_submission` auto-detected mounted exp092 `submission.csv` because the filename was generic. This made correction-vs-exp077 columns meaningless. Fixed `worst_well_rawtest_guard.py` so generic `submission.csv` is not globbed from `/kaggle/input`; exp077 is used only when explicitly provided.
- v2 completed with corrected input resolution.
- Local output: `kaggle/output/guard_v2/artifacts/worst_well_rawtest_guard/`
- status: `visible_test_completed_pass` after 2026-06-22 correction; original summary file says `completed_pass`
- warning wells: 0 / 3
- schema parity: PASS, train/inference feature count 240 / 240, missing 0, extra 0, order mismatch 0
- visible test context: 14,151 rows / 3 wells, anchor T0 vs last_known abs max 0.0
- projection summary parity: sources 5, max abs ratio 1.4706396688327277
- exp092 inference prediction decompressed SHA: `c863d55011690f4cc7f96e1a814619c5cf68ba4b9b83ed038756ccb25e302c5e`
- guard prediction SHA: `9882797f249273e6f1911e58b6a8ad8b385b91cea3d4dfebf9e03f47b9d07332`
- OOF profile context: 773 train wells, 231 wells above +0.25 RMSE regression threshold vs exp077; max regression +4.164459617, max improvement -5.348767590
- visible test well metrics:
  - `00bbac68`: 6,014 rows, pred range 41.996094, step p95 0.279297, step max 2.516602, warnings 0
  - `00e12e8b`: 4,301 rows, pred range 23.789062, step p95 0.182666, step max 1.534180, warnings 0
  - `000d7d20`: 3,836 rows, pred range 14.574219, step p95 0.114258, step max 0.595703, warnings 0

Interpretation at the time: exp092 normal-notebook inference showed no schema drift, no prefix-anchor mismatch, and no continuity warning under the OOF-derived thresholds.

2026-06-22 correction:
- The previous interpretation overstated the result for a Code Competition. The guard kernel ran as a normal Kaggle notebook and therefore evaluated only the exposed sample / visible test, not the hidden LB test injected during code submission rerun.
- This output is not evidence that hidden LB raw test satisfies the same continuity/schema assumptions. Hidden-test conditions can only be probed indirectly by placing assertions inside the submitted inference notebook and observing whether the submission rerun passes or fails.
- `config.yaml` and `result.md` were updated to downgrade the guard status to `visible_test_completed_pass` / `not_hidden_evidence`.

## Hidden assert probe implementation

User requested that the correction be incorporated into this experiment and that the experiment purpose match Code Competition reality: hidden test cannot be inspected by normal kernel output, so the only useful hidden-side probe is opt-in assertions inside the submission-rerun inference notebook.

Implemented:
- Added `docs/legacy/steering/20260622-exp092-hidden-assert-probe/`.
- Added `hidden_assert_probe` support to `run_saved_model_inference()`.
- Updated `exp092_u_projection_correction_disagreement_fullrun_inference.ipynb` to pass `inference.hidden_assert_probe`.
- Added disabled-by-default config under `inference.hidden_assert_probe`.
- Probe failure message includes only failed check names; hidden rows / wells / metrics are not printed.
- Probe-enabled hidden context redacts summary / metrics / detailed prediction artifacts and leaves `submission.csv` for scoring.

Default assert conditions:
- `non_visible_signature`: input is not the exposed visible test signature.
- `sample_id_coverage`: all `sample_submission.id` rows are predicted, with zero fallback rows.
- `finite_predictions`: `last_known_tvt`, `pred_delta`, and `pred_tvt` are finite.
- `anchor_t0_abs_max`: recovered prefix anchor TVT and `last_known_tvt` differ by at most 0.05.
- `known_prefix_rows_min`: every well has at least one known TVT_input prefix row.
- `well_step_abs_p95_max`: per-well adjacent `pred_tvt` step p95 is at most 2.0.
- `well_step_abs_max_max`: per-well adjacent `pred_tvt` step max is at most 10.0.

Exp092 concern proxy conditions added after review:
- `pred_delta_abs_p95_max`: per-well `|pred_tvt - last_known_tvt|` p95 is at most 100.0.
- `pred_delta_abs_max_max`: per-well `|pred_tvt - last_known_tvt|` max is at most 160.0.
- `pred_range_max`: per-well `pred_tvt` range is at most 180.0.
- `near_prefix_delta_abs_p95_max`: first 250 prediction rows per well have `|pred_delta|` p95 at most 25.0.
- `near_prefix_delta_abs_max_max`: first 250 prediction rows per well have `|pred_delta|` max at most 50.0.
- `near_prefix_step_abs_p95_max`: first 250 prediction rows per well have adjacent `pred_tvt` step p95 at most 1.5.
- `near_prefix_step_abs_max_max`: first 250 prediction rows per well have adjacent `pred_tvt` step max at most 5.0.
- `projection_feature_finite`: U-projection / disagreement features are finite.
- `projection_correction_abs_p95_max`: projection correction / residual feature column abs p95 is at most 20.0.
- `projection_correction_abs_max_max`: projection correction / residual feature column abs max is at most 80.0.
- `u_disagreement_abs_p95_max`: PF/Beam/likelihood-PF U-space disagreement feature column abs p95 is at most 250.0.
- `u_disagreement_abs_max_max`: PF/Beam/likelihood-PF U-space disagreement feature column abs max is at most 500.0.

These are the actual label-free proxy checks for the exp092 concern: by-well regression shape, near-prefix instability, and U-projection/disagreement over-correction.

## Hidden assert probe v2 scoring failure

Kaggle code submission `ref=53931397` / description `exp092 hidden assert proxy probe v2` completed with empty Public LB. Kaggle UI and raw API expose only the generic rerun failure:

`Notebook Threw Exception` / `Your notebook hit an unhandled error while rerunning your code. Note that the hidden dataset can be larger/smaller/different than the public dataset`

The visible notebook run completed normally and skipped assertions on the exposed visible test, so this is a hidden submission-rerun exception. Because v2 enabled many assert checks at once, Kaggle does not reveal which condition failed. The v2 result is therefore not interpretable beyond "at least one hidden-rerun assert or hidden-only runtime path failed".

Implementation correction:
- Added `hidden_assert_probe.active_checks` so each code submission can enable exactly one assert or a small named group.
- Set the next probe to `active_checks: [sample_id_coverage]`.
- Interpretation rule for the next submission: if the description-named single check gets an empty score / rerun exception, that condition failed; if it scores, that condition passed and later submissions can probe the exp092-specific shape checks one at a time.

## Hidden assert probe v3 sample_id_coverage result

Kaggle code submission `ref=53933465` / description `exp092 probe sample_id_coverage v3` completed with Public LB `8.350`.

Interpretation:
- `sample_id_coverage` passed on the hidden submission rerun.
- Hidden `sample_submission.id` rows were all covered by generated exp092 predictions, and fallback rows were zero under this single-check probe.
- The v2 empty-score failure was therefore not caused by sample-id coverage. The remaining candidates are hidden-only runtime behavior in the other assert paths or one of the exp092-specific proxy checks: prediction continuity, `pred_delta` limits, near-prefix limits, projection feature finiteness, projection correction magnitude, or U-space disagreement magnitude.

## OOF check for projection/disagreement relation to degraded wells

User asked whether `projection_correction_abs_*` and `u_disagreement_abs_*` should be checked on hidden test only after confirming they relate to degraded train/OOF wells.

Checked available local evidence:
- Existing `artifacts/oof_delta_guard/exp092_oof_delta_guard_by_well.csv`
- Existing `artifacts/oof_delta_guard/exp092_oof_delta_guard_path_continuity.csv`
- Kaggle train output metadata / feature importance downloaded under `/tmp/exp092_train_output_check/artifacts/`

Direct limitation:
- The exp092 projection/disagreement feature frame by row/well was not saved locally.
- The exp072 train feature cache body and exp092 train predictions remained unavailable / incomplete through Kaggle output downloads, so direct by-well `projection_correction_abs_*` and `u_disagreement_abs_*` values could not be recomputed in this pass.

What could be checked:
- Feature importance confirms the added exp092 groups were used by the model:
  - base features: 196 features, importance share `0.7350`
  - projection correction group: 20 features, importance share `0.1228`, top `uproj_likpf_mean_resid_mad`
  - U-space disagreement group: 24 features, importance share `0.1422`, top `uproj_diff_pf_ancc_minus_likpf_mean`
- Existing OOF guard proxy for actual prediction correction magnitude does not support a simple "larger correction => worse well" rule:
  - `lgb1_minus_exp077_policy_correction_abs_p95`: all median `3.119`, degraded top30 median `5.080`, improved bottom30 median `6.966`, Spearman vs RMSE delta `-0.0859`
  - `lgb1_minus_exp073_correction_abs_p95`: all median `3.216`, degraded top30 median `5.326`, improved bottom30 median `6.750`, Spearman vs RMSE delta `-0.1174`
  - step/continuity proxies have near-zero Spearman correlations with RMSE degradation.

Conclusion:
- The current OOF evidence does not justify hidden assert checks that simply reject large projection/disagreement or large correction magnitude as "bad".
- Large exp092 correction is present in degraded wells, but it is also present, and often larger, in improved wells. It appears more like a model activity signal than a degradation-specific signal.
- Hidden checks for `projection_correction_abs_*` / `u_disagreement_abs_*` should be treated as broad sanity guards only, not as validated exp092 worst-well degradation probes, unless a new readout first saves and analyzes the projection/disagreement feature values by OOF well.

Saved local analysis:
- `artifacts/oof_delta_guard/exp092_projection_disagreement_worstwell_relation_summary.csv`
- `artifacts/oof_delta_guard/exp092_projection_disagreement_worstwell_relation_summary.json`

## Hidden assert probe discontinued

Decision:
- Stop further hidden assert submissions for exp092.
- Restore the exp092 inference execution path to normal saved-booster inference so downstream experiments can copy/reference exp092 without inheriting discontinued probe logic.

Reason:
- `sample_id_coverage` passed on hidden rerun (`ref=53933465`, Public LB `8.350`).
- The remaining assert candidates are not validated OOF worst-well degradation probes.
- OOF evidence showed projection/disagreement features are used by the model, but large correction magnitude is not degradation-specific; improved wells can have even larger correction magnitude.

Implementation cleanup:
- Removed hidden assert probe helpers and redaction branch from `u_projection_correction_disagreement_fullrun.py`.
- Removed `hidden_assert_probe` config from `config.yaml`.
- Removed hidden assert print / argument from `exp092_u_projection_correction_disagreement_fullrun_inference.ipynb`.
- Kept the probe results only as historical diagnostics in `SESSION_NOTES.md`, `result.md`, and `SUBMISSIONS.md`.
