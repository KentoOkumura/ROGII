# exp002_drift_minimal セッションノート

## 目的

`exp001_baseline` の `last_anchor` を基準に、`TVT - last_anchor_tvt` の drift / residual を推論可能な trajectory、GR、prefix 特徴で学習する。

## 現在の状態

- 状態: 完了
- 親実験: `exp001_baseline`
- 優先アイデア: `last_anchor` からの drift / residual 学習
- CV: Kaggle train full CV 完了。`drift_hgb` 14.124569
- LB: public 12.533 (`ref=53211155`)

## コマンドログ

- 2026-05-31: `uv run python scripts/new_steering.py --experiment exp002_drift_minimal`
- 2026-05-31: `uv run python scripts/new_experiment.py --name exp002_drift_minimal --source experiments/exp001_baseline`
- 2026-05-31: `uv run python scripts/validate_experiment.py --experiment exp002_drift_minimal` が通過。
- 2026-05-31: `uv run python scripts/execute_experiment_notebook.py --experiment exp002_drift_minimal --notebook train --debug --allow-local` が通過。
- 2026-05-31: inference smoke を非書き込みで実行し、sample submission 14,151 IDs に対して missing 0 を確認。
- 2026-05-31: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp002_drift_minimal --strict`、`uv run pytest`、`uv run ruff check .` が通過。
- 2026-05-31: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp002_drift_minimal --notebook train --run-on-push --strict` を実行。
- 2026-05-31: `kaggle kernels push -p experiments/exp002_drift_minimal/kaggle/train` は title と id slug 不一致で 400 error。
- 2026-05-31: title を `exp002 drift minimal train` にして train notebook を再生成。
- 2026-05-31: `kaggle kernels push -p experiments/exp002_drift_minimal/kaggle/train` が成功。Kaggle kernel version 1、URL: https://www.kaggle.com/code/kentookumura/exp002-drift-minimal-train
- 2026-05-31: `kaggle kernels status kentookumura/exp002-drift-minimal-train` は `KernelWorkerStatus.RUNNING`。
- 2026-05-31: ユーザー依頼で再確認。`kaggle kernels status kentookumura/exp002-drift-minimal-train` はまだ `KernelWorkerStatus.RUNNING`。Kaggle full CV output は未取得。
- 2026-05-31: `kaggle kernels status kentookumura/exp002-drift-minimal-train` は `KernelWorkerStatus.COMPLETE`。
- 2026-05-31: `kaggle kernels output kentookumura/exp002-drift-minimal-train -p /tmp/kaggle-output/exp002_drift_minimal/train` で output を取得。
- 2026-05-31: Kaggle output の `metrics.json` と `artifacts/*.csv` を `experiments/exp002_drift_minimal/` に反映。
- 2026-05-31: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp002_drift_minimal --notebook inference --run-on-push --title "exp002 drift minimal inference" --strict` を実行。
- 2026-05-31: `kaggle kernels push -p experiments/exp002_drift_minimal/kaggle/inference` が成功。Kaggle kernel version 1、URL: https://www.kaggle.com/code/kentookumura/exp002-drift-minimal-inference
- 2026-05-31: `kaggle kernels status kentookumura/exp002-drift-minimal-inference` は `KernelWorkerStatus.COMPLETE`。
- 2026-05-31: `kaggle kernels output kentookumura/exp002-drift-minimal-inference -p /tmp/kaggle-output/exp002_drift_minimal/inference` で `submission.csv` を取得。
- 2026-05-31: `.agents/skills/kaggle-submit-check/scripts/check_submission.py /tmp/kaggle-output/exp002_drift_minimal/inference/submission.csv --sample data/raw/sample_submission.csv` は PASS。14,151 rows、`id,tvt`、欠損/重複なし。
- 2026-05-31: `kaggle competitions submit rogii-wellbore-geology-prediction -k kentookumura/exp002-drift-minimal-inference -v 1 -f submission.csv -m "exp002_drift_minimal drift_hgb CV 14.124569"` を実行。
- 2026-05-31: submission ref `53211155` が complete。public LB 12.533。

## 予定

```bash
uv run python scripts/record_experiment.py --experiment exp002_drift_minimal --status completed --cv 14.124569 --public-lb 12.533 --metric rmse
uv run python scripts/update_experiment_summary.py
```

## 変更点

- `config.yaml` を exp002 用に更新し、primary strategy を `drift_hgb` に変更。
- `baseline.py` に residual feature builder、HGB model、training row sampling、drift prediction を実装。
- train notebook は fold ごとに train wells で residual model を fit し、valid wells で OOF RMSE を計算する。
- inference notebook は train wells 全体から residual model を fit し、test wells の submission を生成する。

## 結果

| Strategy | Debug OOF RMSE | Mean Fold RMSE | Rows |
| --- | ---: | ---: | ---: |
| `last_anchor` | 12.145321 | 12.023445 | 143,091 |
| `drift_hgb` | 17.770814 | 16.707989 | 143,091 |

30 wells debug では `drift_hgb` が悪化した。これは smoke 用の小標本で、full CV の代替にはしない。ただし、初期 residual model は anchor を壊しやすい可能性があるため、full CV でも悪化する場合は local matcher または fold-safe structural guide に切り替える。

Kaggle train full CV:

| Strategy | Full OOF RMSE | Mean Fold RMSE | Rows |
| --- | ---: | ---: | ---: |
| `last_anchor` | 15.909853 | 15.894391 | 3,783,989 |
| `drift_hgb` | 14.124569 | 14.101909 | 3,783,989 |

`drift_hgb` は exp001/同一 split の `last_anchor` から RMSE 1.785284 改善。fold RMSE は 13.431936 / 13.573716 / 13.510140 / 14.060831 / 15.932922。

Kaggle inference / submission:

| Item | Value |
| --- | --- |
| Kernel | `kentookumura/exp002-drift-minimal-inference` |
| Version | 1 |
| Submission ref | 53211155 |
| Public LB | 12.533 |
| Private LB | - |

Public LB は exp001 の 15.883 から 3.350 改善。CV 14.124569 より public LB の方が 1.591569 良い。

## 次のアクション

1. sampling / shrink / feature ablation を次の小実験で試す。
2. Public LB は小さい visible test による可能性があるため、次実験も well-level CV を主判断にする。
