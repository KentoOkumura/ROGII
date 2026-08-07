# exp022_distance_uncertainty_shrink セッションノート

## 目的

バックログ先頭の `distance_uncertainty_shrink` を実装する。`exp021` の clean CV 改善候補である weighted LightGBM + bucket shrink を親にし、距離・GR 欠損・tail 長・Z span・raw residual magnitude から inference-safe な不確実性 proxy を作って residual を追加で縮める。

## 現在の状態

- 状態: Kaggle train 完了、inference / submit なし
- 親実験: `exp021_distance_weighted_inference_postprocess`
- Parent clean CV: `weighted_distance_bucket_shrink` 13.415799
- Parent weighted raw CV: 13.470015
- Raw clean CV anchor: `exp013 lightgbm_no_gr` 13.549257
- Public LB anchor: `exp013 distance_bucket_shrink` 12.271
- exp021 Public LB: 12.523

## コマンドログ

- 2026-06-06: `uv run python scripts/new_steering.py --experiment exp022_distance_uncertainty_shrink` で steering docs を作成。
- 2026-06-06: `uv run python scripts/new_experiment.py --name exp022_distance_uncertainty_shrink --source experiments/exp021_distance_weighted_inference_postprocess` で exp021 から実験を作成。
- 2026-06-06: notebook 名、`settings.py`、`config.yaml`、README、SESSION_NOTES、result、metrics を exp022 用に更新。
- 2026-06-06: `distance_uncertainty_shrink.py` を追加し、fixed uncertainty shrink candidates と OOF proxy column 保存を実装。
- 2026-06-06: `uv run python -m py_compile experiments/exp022_distance_uncertainty_shrink/distance_uncertainty_shrink.py experiments/exp022_distance_uncertainty_shrink/baseline.py experiments/exp022_distance_uncertainty_shrink/settings.py` が通過。
- 2026-06-06: `uv run ruff check experiments/exp022_distance_uncertainty_shrink/distance_uncertainty_shrink.py experiments/exp022_distance_uncertainty_shrink/baseline.py experiments/exp022_distance_uncertainty_shrink/settings.py` が通過。
- 2026-06-06: `uv run python scripts/validate_experiment.py --experiment exp022_distance_uncertainty_shrink` が通過。
- 2026-06-06: notebook code cell compile が train / inference とも通過。
- 2026-06-06: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp022_distance_uncertainty_shrink --notebook train --run-on-push --kernel-id kentookumura/exp022-unc-shrink-train --title "exp022 uncertainty shrink train" --strict` が通過。
- 2026-06-06: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp022_distance_uncertainty_shrink --notebook inference --run-on-push --kernel-id kentookumura/exp022-unc-shrink-infer --title "exp022 uncertainty shrink infer" --strict` が通過。
- 2026-06-06: `kaggle kernels push -p experiments/exp022_distance_uncertainty_shrink/kaggle/train` で train v1 を push。Kaggle URL slug は title 由来の `kentookumura/exp022-uncertainty-shrink-train`。
- 2026-06-06: `kaggle kernels status kentookumura/exp022-uncertainty-shrink-train` は Kaggle API 500 を返したが、`kaggle kernels output kentookumura/exp022-uncertainty-shrink-train -p /tmp/kaggle-output/exp022_distance_uncertainty_shrink/train` で output を取得。
- 2026-06-06: Kaggle train result は `weighted_distance_bucket_shrink` 13.415799、`weighted_raw` 13.470015、`weighted_uncertainty_shrink_conservative` 13.555887、`weighted_uncertainty_shrink_medium` 13.747715、`weighted_uncertainty_shrink_aggressive` 13.935058。
- 2026-06-06: 小さい artifact と log を `artifacts/` に保存。855MB の `weighted_oof_predictions.csv` は `/tmp/kaggle-output/exp022_distance_uncertainty_shrink/train/artifacts/` のみ保持。

## 変更点

- exp021 の selected weight profile `near_down_far_up_lightgbm` は固定。
- `postprocess.candidate_methods` に `uncertainty_shrink_conservative`、`uncertainty_shrink_medium`、`uncertainty_shrink_aggressive` を追加。
- uncertainty shrink は `distance_bucket_shrink` の alpha を基準に、inference-time に使える proxy だけで shrink factor を決める。
- 同じ OOF target residual で係数を fit しない。初回は config 固定候補として clean CV を監査する。

## 次のアクション

1. exp022 は inference / submit に進めない。
2. 単純な fixed uncertainty shrink は打ち切り、次は backlog 先頭の `pseudo_tail_distance_augmentation` を検討する。
3. uncertainty を再開する場合は同一 OOF 固定式ではなく、original-fold 外 selection か abs-error model の held-out 監査に限定する。
