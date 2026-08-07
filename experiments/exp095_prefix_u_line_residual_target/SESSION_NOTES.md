# exp095_prefix_u_line_residual_target セッションノート

## 2026-06-20 実装

- `.steering/20260620-exp095-prefix-u-line-residual-target/` を作成。
- `experiments/exp095_prefix_u_line_residual_target/` を `exp080_u_space_target_ablation` からコピーして作成。
- 実装元を `prefix_u_line_residual_target.py` に整理。
- `dTVT` control と `prefix_u_line_alpha1p0` / `prefix_u_line_alpha0p5` を同一 folds / 同一 features / `lgb0` で比較する config に更新。
- raw train known-prefix rows だけから `U_alpha = TVT_input + alpha * Z` の robust line を fit し、target を `U_alpha - prefix_line(MD)`、inverse を `pred + prefix_line(MD) - alpha * Z` とする。
- prefix が短い、または MD span が小さい well は最後の known prefix row の constant `U_alpha` へ fallback する。

## 実行コマンド

```bash
make new-steering EXP=exp095_prefix_u_line_residual_target
make new-exp EXP=exp095_prefix_u_line_residual_target SOURCE=experiments/exp080_u_space_target_ablation
.venv/bin/python -m py_compile experiments/exp095_prefix_u_line_residual_target/prefix_u_line_residual_target.py experiments/exp095_prefix_u_line_residual_target/public_notebook_replay_audit.py experiments/exp095_prefix_u_line_residual_target/settings.py
.venv/bin/python -m json.tool experiments/exp095_prefix_u_line_residual_target/exp095_prefix_u_line_residual_target_train.ipynb
.venv/bin/python -m json.tool experiments/exp095_prefix_u_line_residual_target/exp095_prefix_u_line_residual_target_inference.ipynb
uv run ruff check experiments/exp095_prefix_u_line_residual_target/prefix_u_line_residual_target.py experiments/exp095_prefix_u_line_residual_target/public_notebook_replay_audit.py experiments/exp095_prefix_u_line_residual_target/settings.py
uv run python scripts/validate_experiment.py --experiment exp095_prefix_u_line_residual_target
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp095_prefix_u_line_residual_target --notebook train --kernel-id kentookumura/exp095-prefix-u-line-residual-target-train --title "exp095 prefix u line residual target train" --run-on-push --strict
uv run python scripts/update_experiment_summary.py
```

## 検証

- Python compile: PASS
- train notebook JSON: PASS
- inference notebook JSON: PASS
- ruff: PASS
- `validate_experiment.py --experiment exp095_prefix_u_line_residual_target`: PASS
- Kaggle train package generation: PASS
- generated train package: `experiments/exp095_prefix_u_line_residual_target/kaggle/train`
- generated kernel id: `kentookumura/exp095-prefix-u-line-residual-target-train`
- `experiment_summary.md`: updated, exp095 appears as `implemented_not_run`

## 現在の状態

- Kaggle train v1 完了。
- prefix U-line residual target は不採用。
- 推論 target は未選択。

## 2026-06-20 Kaggle train v1

```bash
kaggle kernels push -p experiments/exp095_prefix_u_line_residual_target/kaggle/train
kaggle kernels pull kentookumura/exp095-prefix-u-line-residual-target-train -p /tmp/kaggle-pull/exp095-prefix-u-line-residual-target-train-v1 -m
kaggle kernels logs kentookumura/exp095-prefix-u-line-residual-target-train
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp095-prefix-u-line-residual-target-train
kaggle kernels output kentookumura/exp095-prefix-u-line-residual-target-train -p /tmp/kaggle-output/exp095_prefix_u_line_residual_target/train_v1
kaggle kernels status kentookumura/exp095-prefix-u-line-residual-target-train
timeout 600 kaggle kernels logs -f --interval 30 kentookumura/exp095-prefix-u-line-residual-target-train
```

- push: `Kernel version 1 successfully pushed`
- URL: `https://www.kaggle.com/code/kentookumura/exp095-prefix-u-line-residual-target-train`
- kernel id: `kentookumura/exp095-prefix-u-line-residual-target-train`
- pull existence check: PASS at `/tmp/kaggle-pull/exp095-prefix-u-line-residual-target-train-v1`
- initial `logs`: empty
- 5 minute `logs -f`: no log output before timeout
- initial output probe: no files downloaded yet
- status check: `KernelWorkerStatus.RUNNING`
- 10 minute `logs -f`: stopped by user request; Kaggle kernel itself was not cancelled

## 2026-06-21 Kaggle train v1 完了確認

```bash
kaggle kernels status kentookumura/exp095-prefix-u-line-residual-target-train
kaggle kernels logs kentookumura/exp095-prefix-u-line-residual-target-train
kaggle kernels output kentookumura/exp095-prefix-u-line-residual-target-train -p /tmp/kaggle-output/exp095_prefix_u_line_residual_target/train_v1
```

- status: `KernelWorkerStatus.COMPLETE`
- output: `/tmp/kaggle-output/exp095_prefix_u_line_residual_target/train_v1`
- elapsed_seconds: 8228.319
- rows: 3,783,989
- wells: 773
- feature count: 196
- active mode: `gpu_repro_guard_dp_threads8`
- active models: `lgb0` only

Pooled RMSE:

| target | pooled RMSE | verdict |
| --- | ---: | --- |
| `dTVT` | 9.664067 | best / keep |
| `prefix_u_line_alpha0p5` | 28.087914 | reject |
| `prefix_u_line_alpha1p0` | 33.478794 | reject |

Fold RMSE:

| target | fold0 | fold1 | fold2 | fold3 | fold4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dTVT` | 8.875566 | 10.177136 | 8.538341 | 10.440596 | 10.134981 |
| `prefix_u_line_alpha0p5` | 22.001913 | 28.078428 | 26.262255 | 33.539633 | 29.287922 |
| `prefix_u_line_alpha1p0` | 33.150654 | 34.630264 | 33.105576 | 37.855243 | 27.867127 |

Prefix-line diagnostics:

- `anchor_t0_vs_last_known_abs_max`: 0.0
- known-prefix rows: min 851 / max 2392
- `prefix_alpha1_fallback_wells`: 0
- `prefix_alpha0p5_fallback_wells`: 0

同期した小さい生成物:

- `artifacts/exp095_prefix_u_line_residual_target_metrics.csv`
- `artifacts/exp095_prefix_u_line_residual_target_by_well.csv`
- `artifacts/exp095_prefix_u_line_residual_target_bucket_metrics.csv`
- `artifacts/exp095_prefix_u_line_residual_target_target_summary.csv`
- `artifacts/exp095_prefix_u_line_residual_target_feature_schema.csv`
- `artifacts/exp095_prefix_u_line_residual_target_lgb_models/manifest.json`
- `artifacts/exp095_prefix_u_line_residual_target_summary.json`

大きい生成物は repo に同期しない:

- `/tmp/kaggle-output/exp095_prefix_u_line_residual_target/train_v1/artifacts/exp095_prefix_u_line_residual_target_predictions.csv.gz`
- `/tmp/kaggle-output/exp095_prefix_u_line_residual_target/train_v1/artifacts/exp095_prefix_u_line_residual_target_lgb_models/`

解釈:

- prefix-line fit 自体は fallback なしで成立しているが、target 分布が大きくなり、near rows から long-tail まで `dTVT` より大幅に悪化した。
- `prefix_u_line_residual_target` は inference port しない。
- target 変更方向はここで閉じ、U-space は target-free projection / correction feature / gated postprocess に戻す。
