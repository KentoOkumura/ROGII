# exp026_pseudo_tail_bucket_shrink_inference_submit セッションノート

## 目的

`exp025_pseudo_tail_postprocess_cv_audit` で selected になった fixed `exp014_bucket_shrink_params` を `exp024` raw pseudo-tail inference flow に反映し、Kaggle inference output の `submission.csv` を生成して提出前チェックする。

## 現在の状態

- 状態: 完了
- 親実験: `exp025_pseudo_tail_postprocess_cv_audit`
- inference flow: `exp024_pseudo_tail_inference_postprocess`
- selected training variant: `pseudo_tail_3_cutoffs_distance_balanced`
- selected postprocess: `distance_bucket_shrink` / `exp014_bucket_shrink_params`
- CV reference: 12.870780
- Raw exp024 Public LB: 12.166
- 提出: ref `53411137`
- Public LB: 12.102

## コマンドログ

- 2026-06-06: `uv run python scripts/new_steering.py --experiment exp026_pseudo_tail_bucket_shrink_inference_submit` で steering docs を作成。
- 2026-06-06: `uv run python scripts/new_experiment.py --name exp026_pseudo_tail_bucket_shrink_inference_submit --source experiments/exp024_pseudo_tail_inference_postprocess` で exp024 から実験を作成。
- 2026-06-06: notebook 名、`settings.py`、`config.yaml`、README、SESSION_NOTES、result、metrics を exp026 用に更新。
- 2026-06-06: `uv run python scripts/validate_experiment.py --experiment exp026_pseudo_tail_bucket_shrink_inference_submit` が通過。
- 2026-06-06: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp026_pseudo_tail_bucket_shrink_inference_submit --notebook inference --run-on-push --title "exp026 pseudo tail bucket shrink inference" --strict` が通過。
- 2026-06-06: `uv run pytest` が通過。9 tests passed。
- 2026-06-06: `kaggle kernels push -p experiments/exp026_pseudo_tail_bucket_shrink_inference_submit/kaggle/inference` で inference version 1 を push。Kaggle URL slug は `kentookumura/exp026-pseudo-tail-bucket-shrink-inference`。
- 2026-06-06: `kaggle kernels status kentookumura/exp026-pseudo-tail-bucket-shrink-inference` は Kaggle API 500 を返したが、`kaggle kernels output kentookumura/exp026-pseudo-tail-bucket-shrink-inference -p /tmp/kaggle-output/exp026_pseudo_tail_bucket_shrink_inference_submit/inference` で output を取得。
- 2026-06-06: `python3 .agents/skills/kaggle-submit-check/scripts/check_submission.py /tmp/kaggle-output/exp026_pseudo_tail_bucket_shrink_inference_submit/inference/submission.csv --sample data/raw/sample_submission.csv` は PASS。
- 2026-06-06: ユーザーが Kaggle submit を完了。`kaggle competitions submissions rogii-wellbore-geology-prediction` で ref `53411137`、Public LB 12.102、status COMPLETE を確認。
- 2026-06-06: `uv run python scripts/record_submission.py --experiment exp026_pseudo_tail_bucket_shrink_inference_submit ... --public-lb 12.102 ...` で `SUBMISSIONS.md` に v011 を記録。
- 2026-06-06: `uv run python scripts/record_experiment.py --experiment exp026_pseudo_tail_bucket_shrink_inference_submit --status completed --cv 12.87078 --public-lb 12.102 ...` で metrics / summary を更新。

## 変更点

- `postprocess.selected_method` を `raw` から `distance_bucket_shrink` に変更。
- `postprocess.methods.distance_bucket_shrink` に exp025 selected `exp014_bucket_shrink_params` を設定。
- `generate_pseudo_tail_submission` の summary 実験名を config の `experiment.name` から出すよう更新。

## Artifacts

- `submission.csv`
- `artifacts/pseudo_tail_inference_summary.json`
- `artifacts/pseudo_tail_inference_well_summaries.csv`
- `artifacts/pseudo_tail_inference_source_summary.csv`

## 結果

- submission rows: 14,151
- postprocess: `distance_bucket_shrink`
- prediction range: 11590.725143 - 12237.368348
- prediction mean: 11907.302608
- exp024 raw submission との差分: min -1.838460、max 1.803698、mean 0.102789、abs mean 0.438886、RMSE 0.611885
- submit-check: PASS
- competition submit: ref `53411137`
- Public LB: 12.102
- exp024 raw Public LB 12.166 から -0.064 改善

## 次のアクション

1. `KAGGLE_DIRECTION.md` の次候補に進む。Public LB anchor は exp026 12.102。
2. 追加 postprocess は同一 OOF fit ではなく、held-out 監査または seed / cutoff diversity で比較する。
