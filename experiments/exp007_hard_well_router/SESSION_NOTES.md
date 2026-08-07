# exp007_hard_well_router セッションノート

## 目的

exp006 の router diagnostic を実装に進め、inference-safe な well condition だけで `exp002` all-GR、`exp003` no-GR、`exp005` guarded prediction を選択する hard well router を検証する。

## 現在の状態

- 状態: Kaggle inference / submission 完了
- 親実験: `exp006_hard_well_router_diagnostic`
- selected candidate: `hard_router_low_gr_guarded`
- CV: 13.921559
- LB: public 12.675 (`ref=53254030`)

## コマンドログ

- 2026-06-01: `uv run python scripts/new_steering.py --experiment exp007_hard_well_router` で steering docs を作成。
- 2026-06-01: `uv run python scripts/new_experiment.py --name exp007_hard_well_router --source experiments/exp006_hard_well_router_diagnostic` で exp006 から実験を作成。
- 2026-06-01: notebook ファイル名、`settings.py`、`config.yaml`、README、SESSION_NOTES、result、metrics、steering docs を exp007 用に更新。
- 2026-06-01: `baseline.py` に `HardRouterModelBundle`、router route 判定、guarded prediction helper を追加。
- 2026-06-01: train notebook を hard router CV と route artifact 出力に更新。
- 2026-06-01: inference notebook を selected hard router variant で submission 生成できる形に更新。
- 2026-06-01: `uv run python scripts/validate_experiment.py --experiment exp007_hard_well_router` が通過。
- 2026-06-01: `uv run ruff check experiments/exp007_hard_well_router/baseline.py experiments/exp007_hard_well_router/diagnostics.py experiments/exp007_hard_well_router/settings.py` が通過。
- 2026-06-01: `python3 -m py_compile experiments/exp007_hard_well_router/baseline.py experiments/exp007_hard_well_router/diagnostics.py experiments/exp007_hard_well_router/settings.py` が通過。
- 2026-06-01: train / inference notebook の JSON 検査が通過。
- 2026-06-01: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp007_hard_well_router --notebook train --run-on-push --title "exp007 hard well router train" --strict` が通過。
- 2026-06-01: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp007_hard_well_router --notebook inference --run-on-push --title "exp007 hard well router inference" --strict` が通過。
- 2026-06-01: `uv run pytest` が通過。9 tests passed。
- 2026-06-01: exp006 の `router_diagnostic_well_tags.csv` に selected router 条件を当てる sanity check を実行。route counts は all-GR 476 / no-GR 247 / guarded 50。
- 2026-06-01: `kaggle kernels push -p experiments/exp007_hard_well_router/kaggle/train` で version 1 を push。URL: https://www.kaggle.com/code/kentookumura/exp007-hard-well-router-train
- 2026-06-01: 5 分間隔で `kaggle kernels status kentookumura/exp007-hard-well-router-train` を監視し、20 分時点で `KernelWorkerStatus.COMPLETE` を確認。
- 2026-06-01: `kaggle kernels output kentookumura/exp007-hard-well-router-train -p /tmp/kaggle-output/exp007_hard_well_router/train` を実行。初回は DNS エラーで途中停止したが、再実行で `metrics.json` と kernel log まで取得完了。
- 2026-06-01: Kaggle output の `metrics.json`、`ablation_metrics.csv`、`well_metrics.csv`、`fold_metrics.csv`、`fold_model_training.csv`、router diagnostic artifacts、train log を `experiments/exp007_hard_well_router/` に反映。
- 2026-06-01: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp007_hard_well_router --notebook inference --run-on-push --title "exp007 hard well router inference" --strict` で inference package を再生成。
- 2026-06-01: `kaggle kernels push -p experiments/exp007_hard_well_router/kaggle/inference` で inference version 1 を push。URL: https://www.kaggle.com/code/kentookumura/exp007-hard-well-router-inference
- 2026-06-01: `kaggle kernels status kentookumura/exp007-hard-well-router-inference` で `KernelWorkerStatus.COMPLETE` を確認。
- 2026-06-01: `kaggle kernels output kentookumura/exp007-hard-well-router-inference -p /tmp/kaggle-output/exp007_hard_well_router/inference` で `submission.csv`、`inference_well_summaries.csv`、kernel log を取得。
- 2026-06-01: `.agents/skills/kaggle-submit-check/scripts/check_submission.py /tmp/kaggle-output/exp007_hard_well_router/inference/submission.csv --sample data/raw/sample_submission.csv` は PASS。
- 2026-06-01: `uv run python scripts/validate_submission.py --submission /tmp/kaggle-output/exp007_hard_well_router/inference/submission.csv` は PASS。
- 2026-06-01: visible route sanity は `000d7d20=guarded`、`00bbac68=all_gr`、`00e12e8b=all_gr`。
- 2026-06-01: `kaggle competitions submit rogii-wellbore-geology-prediction -k kentookumura/exp007-hard-well-router-inference -v 1 -f submission.csv -m "exp007_hard_well_router hard_router_low_gr_guarded CV 13.921559"` で提出。
- 2026-06-01: submission ref `53254030` は `SubmissionStatus.COMPLETE`、Public LB 12.675。
- 2026-06-01: `uv run python scripts/record_submission.py --experiment exp007_hard_well_router --file /tmp/kaggle-output/exp007_hard_well_router/inference/submission.csv --cv 13.921559 --public-lb 12.675 ...` で `submissions/SUBMISSIONS.md` に v006 を記録。
- 2026-06-01: `uv run python scripts/record_experiment.py --experiment exp007_hard_well_router --status completed --cv 13.921559 --public-lb 12.675 ...` で `metrics.json` と `experiment_summary.md` を更新。

## 変更点

- `control_exp002_all`: exp002 と同じ all-GR residual model。
- `control_exp003_no_gr`: exp003 selected と同じ no-GR residual model。
- `control_exp005_guarded`: exp005 selected と同じ strict guarded gate。
- `hard_router_low_gr_any`: exp004 diagnostic の any-low-GR to no-GR rule を hard router 実装で再現する control。
- `hard_router_low_gr_guarded`: selected router。`gr_weak_all`、`short_prefix_low_gr`、`large_gr_shift_low_gr` を no-GR に送り、残りの low-GR は guarded、それ以外は all-GR。

## 事前根拠

exp006 artifact 上の推定では、`gr_weak_all | short_prefix_low_gr | large_gr_shift_low_gr` を no-GR に送る rule は CV 13.921559 相当で、exp004 any-low-GR rule の 13.932968、exp005 strict guarded の 13.936732 より少し良い。ただし同じ OOF artifact 上での rule selection なので、Kaggle full CV と Public LB で確認する。

## 結果

| Variant | CV | exp002 差分 |
| --- | ---: | ---: |
| `control_exp003_no_gr` | 13.882944 | -0.241625 |
| `hard_router_low_gr_guarded` | 13.921559 | -0.203010 |
| `hard_router_low_gr_any` | 13.932968 | -0.191601 |
| `control_exp005_guarded` | 13.936732 | -0.187837 |
| `control_exp002_all` | 14.124569 | 0.000000 |

Selected `hard_router_low_gr_guarded` route counts:

| Route | Wells | Eval Rows |
| --- | ---: | ---: |
| `all_gr` | 476 | 2,277,855 |
| `no_gr` | 247 | 1,300,064 |
| `guarded` | 50 | 206,070 |

Selected router は exp005 guarded から CV を 0.015173 改善し、exp002 all-GR から 0.203010 改善した。ただし pure no-GR control の CV 13.882944 には届かない。

## Inference / Submission

| Well | Route | Guarded Weight |
| --- | --- | ---: |
| `000d7d20` | `guarded` | 0.0 |
| `00bbac68` | `all_gr` | 0.0 |
| `00e12e8b` | `all_gr` | 0.0 |

Public LB は 12.675。exp004 12.730 と exp003 12.852 は上回ったが、exp005 12.579 と exp002 12.533 には届かない。

## 次のアクション

1. LB anchor は exp002 のまま維持する。
2. router は CV 改善が LB に十分転写しなかったため、次は backlog 先頭の GR local matcher / multi-scale NCC を add-only で検証する。
