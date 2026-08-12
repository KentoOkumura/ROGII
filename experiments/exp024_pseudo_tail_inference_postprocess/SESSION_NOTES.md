# exp024_pseudo_tail_inference_postprocess セッションノート

## 目的

`exp023_pseudo_tail_distance_augmentation` の best `pseudo_tail_3_cutoffs_distance_balanced` を final inference 化し、Kaggle 上で `submission.csv` を生成する。

## 現在の状態

- 状態: submit 完了
- 親実験: `exp023_pseudo_tail_distance_augmentation`
- Parent clean CV: `pseudo_tail_3_cutoffs_distance_balanced` 12.942938
- Public LB anchor: `exp013 distance_bucket_shrink` 12.271
- Public LB: 12.166 (`ref=53408921`)
- selected inference postprocess: `raw`

## コマンドログ

- 2026-06-06: `uv run python scripts/new_steering.py --experiment exp024_pseudo_tail_inference_postprocess` で steering docs を作成。
- 2026-06-06: `uv run python scripts/new_experiment.py --name exp024_pseudo_tail_inference_postprocess --source experiments/exp023_pseudo_tail_distance_augmentation` で exp023 から実験を作成。
- 2026-06-06: notebook 名、`settings.py`、`config.yaml`、README、SESSION_NOTES、result、metrics を exp024 用に更新。
- 2026-06-06: `pseudo_tail_augmentation.py` に `generate_pseudo_tail_submission` を追加し、inference notebook を raw pseudo-tail submission 生成用に更新。
- 2026-06-06: `uv run python -m py_compile experiments/exp024_pseudo_tail_inference_postprocess/pseudo_tail_augmentation.py experiments/exp024_pseudo_tail_inference_postprocess/baseline.py experiments/exp024_pseudo_tail_inference_postprocess/settings.py` が通過。
- 2026-06-06: `uv run ruff check experiments/exp024_pseudo_tail_inference_postprocess/pseudo_tail_augmentation.py experiments/exp024_pseudo_tail_inference_postprocess/baseline.py experiments/exp024_pseudo_tail_inference_postprocess/settings.py` が通過。
- 2026-06-06: train / inference notebook の code cell compile が通過。
- 2026-06-06: `uv run python scripts/validate_experiment.py --experiment exp024_pseudo_tail_inference_postprocess` が通過。
- 2026-06-06: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp024_pseudo_tail_inference_postprocess --notebook inference --run-on-push --title "exp024 pseudo tail inference" --strict` が通過。
- 2026-06-06: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp024_pseudo_tail_inference_postprocess --notebook train --run-on-push --title "exp024 pseudo tail inference train diag" --strict` が通過。
- 2026-06-06: `kaggle kernels push -p experiments/exp024_pseudo_tail_inference_postprocess/kaggle/inference` で inference version 1 を push。Kaggle URL slug は `kentookumura/exp024-pseudo-tail-inference`。
- 2026-06-06: `kaggle kernels status kentookumura/exp024-pseudo-tail-inference` は Kaggle API 500 を返したが、`kaggle kernels output kentookumura/exp024-pseudo-tail-inference -p /tmp/kaggle-output/exp024_pseudo_tail_inference_postprocess/inference` で output を取得。
- 2026-06-06: Kaggle inference は `submission.csv` 14,151 rows を生成。final train rows 242,843、train wells 773、postprocess `raw`。
- 2026-06-06: `uv run python .agents/skills/kaggle-submit-check/scripts/check_submission.py /tmp/kaggle-output/exp024_pseudo_tail_inference_postprocess/inference/submission.csv --sample data/raw/sample_submission.csv` は PASS。欠損/重複なし、行数/header は sample と一致。
- 2026-06-06: `submission.csv`、inference artifacts、kernel log を `experiments/exp024_pseudo_tail_inference_postprocess/artifacts/` に保存。
- 2026-06-06: ユーザーが Kaggle submit を完了。`kaggle competitions submissions rogii-wellbore-geology-prediction` で ref `53408921`、Public LB 12.166、status COMPLETE を確認。
- 2026-06-06: `uv run python scripts/record_experiment.py --experiment exp024_pseudo_tail_inference_postprocess --status completed --cv 12.942938 --public-lb 12.166 ...` で metrics / summary を更新。
- 2026-06-06: `uv run python scripts/record_submission.py --experiment exp024_pseudo_tail_inference_postprocess --file experiments/exp024_pseudo_tail_inference_postprocess/artifacts/submission.csv --cv 12.942938 --public-lb 12.166 ...` で `SUBMISSIONS.md` に v010 を記録。

## 変更点

- exp023 selected variant `pseudo_tail_3_cutoffs_distance_balanced` を final train に使用する。
- final fit は all train wells で、pseudo cutoff は train wells 内だけに生成する。
- inference は raw pseudo-tail prediction を採用する。bucket shrink は未監査なので今回の selected method にはしない。

## 次のアクション

1. exp024 を新 Public LB anchor として、次は public LB 改善が CV と一貫している理由を確認する。
2. 追加 postprocess や sequence diversity は、exp024 raw candidate を壊さない小さな比較として切り出す。
