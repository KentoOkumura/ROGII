# exp117_linear_md_z_prior_residual_target セッションノート

## 2026-06-24 実装

- `.steering/20260624-exp117-linear-md-z-prior-residual-target/` を作成。
- `experiments/exp117_linear_md_z_prior_residual_target/` を `exp095_prefix_u_line_residual_target` からコピーして作成。
- 実装元を `linear_md_z_prior_residual_target.py` に整理。
- `dTVT` control と弱い `linear_prior_*` residual targets を同一 folds / 同一 features / `lgb0` で比較する config に更新。
- prior は `T0 + a * (MD - MD0) + b * (Z - Z0)`。`T0/MD0/Z0` は各 well の known-prefix 最終行だけから復元する。
- 係数は exp113 の診断結果を踏まえた小さい固定値だけにし、validation tail true TVT で係数 fit / 選択はしない。

## 実行コマンド

```bash
uv run python scripts/new_steering.py --experiment exp117_linear_md_z_prior_residual_target
uv run python scripts/new_experiment.py --name exp117_linear_md_z_prior_residual_target --source experiments/exp095_prefix_u_line_residual_target
uv run python -m py_compile experiments/exp117_linear_md_z_prior_residual_target/linear_md_z_prior_residual_target.py experiments/exp117_linear_md_z_prior_residual_target/public_notebook_replay_audit.py experiments/exp117_linear_md_z_prior_residual_target/settings.py
uv run python -m json.tool experiments/exp117_linear_md_z_prior_residual_target/exp117_linear_md_z_prior_residual_target_train.ipynb
uv run python -m json.tool experiments/exp117_linear_md_z_prior_residual_target/exp117_linear_md_z_prior_residual_target_inference.ipynb
uv run ruff check experiments/exp117_linear_md_z_prior_residual_target/linear_md_z_prior_residual_target.py experiments/exp117_linear_md_z_prior_residual_target/public_notebook_replay_audit.py experiments/exp117_linear_md_z_prior_residual_target/settings.py
uv run python scripts/validate_experiment.py --experiment exp117_linear_md_z_prior_residual_target
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp117_linear_md_z_prior_residual_target --notebook train --run-on-push --strict
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp117_linear_md_z_prior_residual_target --notebook train --kernel-id kentookumura/exp117-linear-md-z-prior-residual-target-train --title "exp117 linear md z prior residual target train" --run-on-push --strict
kaggle kernels push -p experiments/exp117_linear_md_z_prior_residual_target/kaggle/train
kaggle kernels pull kentookumura/exp117-linear-md-z-prior-residual-target-train -p /tmp/kaggle-pull/exp117-linear-md-z-prior-residual-target-train-v1 -m
kaggle kernels logs kentookumura/exp117-linear-md-z-prior-residual-target-train
kaggle kernels status kentookumura/exp117-linear-md-z-prior-residual-target-train
kaggle kernels output kentookumura/exp117-linear-md-z-prior-residual-target-train -p /tmp/kaggle-output/exp117_linear_md_z_prior_residual_target/train_v1_probe
timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp117-linear-md-z-prior-residual-target-train
kaggle kernels output kentookumura/exp117-linear-md-z-prior-residual-target-train -p /tmp/kaggle-output/exp117_linear_md_z_prior_residual_target/train_v1
```

## 検証

- Python compile: PASS
- train notebook JSON: PASS
- inference notebook JSON: PASS
- ruff: PASS
- `validate_experiment.py --experiment exp117_linear_md_z_prior_residual_target`: PASS
- Kaggle train package generation: PASS
- generated train package: `experiments/exp117_linear_md_z_prior_residual_target/kaggle/train`

## 2026-06-24 Kaggle train v1

- first push failed because the generated title `ROGII - Wellbore Geology Prediction exp117_linear_md_z_prior_residual_target train` did not resolve to the requested kernel id slug.
- regenerated train package with canonical id/title:
  - kernel id: `kentookumura/exp117-linear-md-z-prior-residual-target-train`
  - title: `exp117 linear md z prior residual target train`
- push: `Kernel version 1 successfully pushed`
- URL: `https://www.kaggle.com/code/kentookumura/exp117-linear-md-z-prior-residual-target-train`
- pull existence check: PASS at `/tmp/kaggle-pull/exp117-linear-md-z-prior-residual-target-train-v1`
- initial `logs`: empty
- output probe: no output files downloaded yet
- status after push: `KernelWorkerStatus.RUNNING`
- `logs -f`: ended with `ConnectionResetError(104, 'Connection reset by peer')`; Kaggle kernel itself was not cancelled.
- status after follow-log disconnect: `KernelWorkerStatus.RUNNING`
- regular `logs` after follow-log disconnect: empty

## 2026-06-24 Kaggle train v1 完了確認

- status: `KernelWorkerStatus.COMPLETE`
- output: `/tmp/kaggle-output/exp117_linear_md_z_prior_residual_target/train_v1`
- elapsed_seconds: 13598.921
- rows: 3,783,989
- wells: 773
- feature count: 196
- active mode: `gpu_repro_guard_dp_threads8`
- active models: `lgb0` only

Pooled RMSE:

| target | pooled RMSE | verdict |
| --- | ---: | --- |
| `dTVT` | 9.664291 | best / keep |
| `linear_prior_a0p02_bm0p25` | 11.061642 | reject |
| `linear_prior_a0p02_bm0p50` | 12.515352 | reject |
| `linear_prior_a0p04_bm0p25` | 11.079209 | reject |

Fold RMSE:

| target | fold0 | fold1 | fold2 | fold3 | fold4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dTVT` | 8.875566 | 10.178201 | 8.538341 | 10.440596 | 10.134981 |
| `linear_prior_a0p02_bm0p25` | 10.537594 | 11.318329 | 10.323318 | 11.680138 | 11.386884 |
| `linear_prior_a0p02_bm0p50` | 12.404166 | 13.078191 | 12.145430 | 12.787320 | 12.133872 |
| `linear_prior_a0p04_bm0p25` | 10.396035 | 11.251801 | 10.458824 | 11.999382 | 11.211363 |

Distance bucket の要点:

- `linear_prior_a0p02_bm0p25` は 0-50 ft / 50-100 ft / 100-250 ft / 250-500 ft で `dTVT` より改善。
- ただし 500-1000 ft は 5.562122 -> 5.656289、1000+ ft は 10.594760 -> 12.190477 に悪化。
- `linear_prior_a0p02_bm0p50` と `linear_prior_a0p04_bm0p25` も pooled / long-tail で悪化。

Well-level delta vs `dTVT`:

| target | improved wells | worse wells | max regression RMSE |
| --- | ---: | ---: | ---: |
| `linear_prior_a0p02_bm0p25` | 268 | 505 | 20.655735 |
| `linear_prior_a0p02_bm0p50` | 248 | 525 | 51.263206 |
| `linear_prior_a0p04_bm0p25` | 250 | 523 | 15.136597 |

Anchor diagnostics:

- `anchor_t0_vs_last_known_abs_max`: 0.0
- known-prefix rows: min 851 / max 2392
- known-prefix MD span: min 850 / max 2391
- `md_delta_abs_p99`: 7054.0
- `z_delta_abs_p99`: 267.330139

同期した小さい生成物:

- `artifacts/exp117_linear_md_z_prior_residual_target_metrics.csv`
- `artifacts/exp117_linear_md_z_prior_residual_target_by_well.csv`
- `artifacts/exp117_linear_md_z_prior_residual_target_bucket_metrics.csv`
- `artifacts/exp117_linear_md_z_prior_residual_target_target_summary.csv`
- `artifacts/exp117_linear_md_z_prior_residual_target_feature_schema.csv`
- `artifacts/exp117_linear_md_z_prior_residual_target_lgb_models/manifest.json`
- `artifacts/exp117_linear_md_z_prior_residual_target_summary.json`

大きい生成物は repo に同期しない:

- `/tmp/kaggle-output/exp117_linear_md_z_prior_residual_target/train_v1/artifacts/exp117_linear_md_z_prior_residual_target_predictions.csv.gz`
- `/tmp/kaggle-output/exp117_linear_md_z_prior_residual_target/train_v1/artifacts/exp117_linear_md_z_prior_residual_target_lgb_models/`

解釈:

- weak linear prior target は近距離 bucket で一部改善したが、long-tail と well-level regression が大きく、global supervised target としては不採用。
- `linear_md_z_prior_residual_target` は inference port しない。
- 使う場合は target 変更ではなく、distance-aware feature、confidence diagnostic、near-prefix 限定 gate の材料に下げる。

## 現在の状態

- Kaggle train v1 完了。
- linear MD/Z prior residual target は不採用。
- 推論 target は未選択。
