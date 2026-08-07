# exp020_distance_weighted_training_audit セッションノート

## 目的

バックログ先頭の `distance_weighted_training_audit` を実装する。`exp013`
raw LightGBM no-GR anchor の距離 bucket 別誤差を診断し、near-row を anchor に
任せる重み付け、far-row の軽い強調、near/mid/far 別 LightGBM の価値を同一
GroupKFold で小さく比較する。

## 現在の状態

- 状態: 完了
- 親実験: `exp013_model_diversity_or_postprocess`
- Raw clean CV anchor: `exp013 lightgbm_no_gr` 13.549257
- Held-out postprocess reference: `exp014` leave-one-original-fold-out 13.535596
- Public LB anchor: `exp013 distance_bucket_shrink` 12.271

## コマンドログ

- 2026-06-06: `uv run python scripts/new_steering.py --experiment exp020_distance_weighted_training_audit` で steering docs を作成。
- 2026-06-06: `uv run python scripts/new_experiment.py --name exp020_distance_weighted_training_audit --source experiments/exp013_model_diversity_or_postprocess` で exp013 から実験を作成。
- 2026-06-06: notebook 名と `settings.py` を exp020 用に更新。
- 2026-06-06: `config.yaml`、README、train/inference notebook、`audit_distance_weighted_training.py` を実装。
- 2026-06-06: `uv run ruff format experiments/exp020_distance_weighted_training_audit/audit_distance_weighted_training.py experiments/exp020_distance_weighted_training_audit/settings.py` を実行。
- 2026-06-06: `uv run python -m py_compile experiments/exp020_distance_weighted_training_audit/audit_distance_weighted_training.py experiments/exp020_distance_weighted_training_audit/baseline.py experiments/exp020_distance_weighted_training_audit/settings.py` が通過。
- 2026-06-06: `uv run ruff check experiments/exp020_distance_weighted_training_audit/audit_distance_weighted_training.py experiments/exp020_distance_weighted_training_audit/baseline.py experiments/exp020_distance_weighted_training_audit/settings.py` が通過。
- 2026-06-06: `uv run python scripts/validate_experiment.py --experiment exp020_distance_weighted_training_audit` が通過。
- 2026-06-06: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp020_distance_weighted_training_audit --notebook train --run-on-push --title "exp020 distance weighted training audit train" --strict` が通過。
- 2026-06-06: `kaggle kernels push -p experiments/exp020_distance_weighted_training_audit/kaggle/train` で version 1 を push。URL: https://www.kaggle.com/code/kentookumura/exp020-distance-weighted-training-audit-train
- 2026-06-06: 実行完了の連絡を受け、`kaggle kernels output kentookumura/exp020-distance-weighted-training-audit-train -p /tmp/kaggle-output/exp020_distance_weighted_training_audit/train` で output を取得。`kaggle kernels status` は Kaggle API 500 を返したが、output は取得できた。
- 2026-06-06: 小さい artifact と log を `experiments/exp020_distance_weighted_training_audit/artifacts/` に保存し、`metrics.json` を更新。
- 2026-06-06: train notebook を薄い entrypoint から、過去実験と同じように setup / OOF audit / training CV / metrics をセル単位で読める構成へ変更。notebook JSON、code cell compile、`validate-exp`、ruff、Kaggle train package generation が通過。
- 2026-06-06: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp020_distance_weighted_training_audit --notebook train --run-on-push --title "exp020 distance weighted training audit train" --strict` で読みやすい notebook 版の Kaggle train package を再生成。
- 2026-06-06: `kaggle kernels push -p experiments/exp020_distance_weighted_training_audit/kaggle/train` で version 2 を push。URL: https://www.kaggle.com/code/kentookumura/exp020-distance-weighted-training-audit-train
- 2026-06-06: version 2 実行完了の連絡を受け、`kaggle kernels output kentookumura/exp020-distance-weighted-training-audit-train -p /tmp/kaggle-output/exp020_distance_weighted_training_audit/train_v2` で output を取得。`kaggle kernels status` は前回同様 Kaggle API 500 を返した。
- 2026-06-06: version 2 の metrics は version 1 と一致。best `near_down_far_up_lightgbm` 13.470015、`control_lightgbm_no_gr` 13.549257。小さい artifact と log を version 2 のものに差し替え、`record_experiment.py` で summary を更新。

## 変更点

- `exp013` の `row_oof_predictions.csv` から raw / last_anchor / recent_linear / exp014 bucket shrink を距離 bucket 別に比較する。
- raw residual の bias、error std、target residual std、near-row raw 悪化を artifact 化する。
- LightGBM no-GR の control、near downweight、far upweight、near+far、near/mid/far segmented model を同一 fold で比較する。
- feature importance は variant / fold / segment 別に保存する。

## Artifacts

- `artifacts/distance_candidate_metrics.csv`
- `artifacts/distance_residual_bucket_summary.csv`
- `artifacts/distance_weighted_training_metrics.csv`
- `artifacts/distance_weighted_feature_importance.csv`
- `artifacts/distance_weighted_training_summary.json`
- `artifacts/exp020-distance-weighted-training-audit-train.log` (version 2)
- `metrics.json`

## 結果

- Raw LightGBM no-GR: 13.549257
- Held-out postprocess reference: 13.535596
- Best training variant: `near_down_far_up_lightgbm` 13.470015
- `far_upweight_lightgbm`: 13.550841
- `near_downweight_lightgbm`: 13.580536
- `near_mid_far_segmented_lightgbm`: 13.655239
- OOF candidate overall: raw 13.549257、`exp014_bucket_shrink_params` 13.501824、`last_anchor` 15.909853、`recent_linear` 41.022355
- `near_down_far_up_lightgbm` distance buckets: rows 0-49 3.576164 vs raw 3.231596、rows 50-249 4.078820 vs raw 3.829747、rows 1000-2499 11.790864 vs raw 11.870592、rows 2500+ 16.284294 vs raw 16.391606。

## 次のアクション

1. `near_down_far_up_lightgbm` を inference notebook に反映する。
2. weighted raw、weighted + exp014 bucket shrink、現行 exp013 submission の比較を作り、提出候補を決める。
3. その後、uncertainty shrink / pseudo-tail augmentation に進む。
