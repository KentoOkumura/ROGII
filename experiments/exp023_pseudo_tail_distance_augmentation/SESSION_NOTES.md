# exp023_pseudo_tail_distance_augmentation セッションノート

## 目的

バックログ先頭の `pseudo_tail_distance_augmentation` を実装する。train fold 内の well だけに仮想 `TVT_input` cutoff を作り、疑似 hidden tail の residual 学習データを増やすことで、`exp012/exp013` LightGBM no-GR anchor を同一 GroupKFold で改善できるか確認する。

## 現在の状態

- 状態: 完了
- 親実験: `exp020_distance_weighted_training_audit`
- Raw clean CV anchor: `exp013 lightgbm_no_gr` 13.549257
- Distance-weighted training reference: `exp020 near_down_far_up_lightgbm` 13.470015
- Distance-weighted postprocess reference: `exp021 weighted_distance_bucket_shrink` 13.415799
- Public LB anchor: `exp013 distance_bucket_shrink` 12.271
- selected variant: `pseudo_tail_3_cutoffs_distance_balanced`
- CV: 12.942938

## コマンドログ

- 2026-06-06: `uv run python scripts/new_steering.py --experiment exp023_pseudo_tail_distance_augmentation` で steering docs を作成。
- 2026-06-06: `uv run python scripts/new_experiment.py --name exp023_pseudo_tail_distance_augmentation --source experiments/exp020_distance_weighted_training_audit` で exp020 から実験を作成。
- 2026-06-06: notebook 名、`settings.py`、`config.yaml`、README、SESSION_NOTES、result、metrics を exp023 用に更新。
- 2026-06-06: `pseudo_tail_augmentation.py` を実装し、pseudo cutoff generation、distance-balanced sampling、source summary artifact を追加。
- 2026-06-06: `uv run ruff format experiments/exp023_pseudo_tail_distance_augmentation/pseudo_tail_augmentation.py experiments/exp023_pseudo_tail_distance_augmentation/settings.py` を実行。
- 2026-06-06: `uv run python -m py_compile experiments/exp023_pseudo_tail_distance_augmentation/pseudo_tail_augmentation.py experiments/exp023_pseudo_tail_distance_augmentation/baseline.py experiments/exp023_pseudo_tail_distance_augmentation/settings.py` が通過。
- 2026-06-06: `uv run ruff check experiments/exp023_pseudo_tail_distance_augmentation/pseudo_tail_augmentation.py experiments/exp023_pseudo_tail_distance_augmentation/baseline.py experiments/exp023_pseudo_tail_distance_augmentation/settings.py` が通過。
- 2026-06-06: train / inference notebook の code cell compile が通過。
- 2026-06-06: `uv run python scripts/validate_experiment.py --experiment exp023_pseudo_tail_distance_augmentation` が通過。
- 2026-06-06: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp023_pseudo_tail_distance_augmentation --notebook train --run-on-push --title "exp023 pseudo tail augmentation train" --strict` が通過。
- 2026-06-06: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp023_pseudo_tail_distance_augmentation --notebook inference --run-on-push --title "exp023 pseudo tail augmentation infer" --strict` が通過。
- 2026-06-06: `uv run pytest` が通過。9 tests passed。
- 2026-06-06: `task validate-template` は `task` 未インストールで失敗。代替として `make validate-template` を実行し、`project.yml validation passed (template)` を確認。
- 2026-06-06: `uv run python scripts/update_experiment_summary.py` で `experiment_summary.md` に exp023 を追加。
- 2026-06-06: `kaggle kernels push -p experiments/exp023_pseudo_tail_distance_augmentation/kaggle/train` で train version 1 を push。Kaggle URL slug は title 由来の `kentookumura/exp023-pseudo-tail-augmentation-train`。
- 2026-06-06: `kaggle kernels status kentookumura/exp023-pseudo-tail-augmentation-train` は Kaggle API 500 を返した。`kaggle kernels output kentookumura/exp023-pseudo-tail-augmentation-train -p /tmp/kaggle-output/exp023_pseudo_tail_distance_augmentation/train` はエラーなしだが、push 直後の確認では output はまだ空。
- 2026-06-06: 実行完了の連絡を受け、`kaggle kernels output kentookumura/exp023-pseudo-tail-augmentation-train -p /tmp/kaggle-output/exp023_pseudo_tail_distance_augmentation/train` で output を取得。
- 2026-06-06: Kaggle result は best `pseudo_tail_3_cutoffs_distance_balanced` 12.942938。`pseudo_tail_1_cutoff` 12.971839、`pseudo_tail_3_cutoffs` 13.012302、`distance_balanced_sampling` 13.441648、`control_lightgbm_no_gr` 13.494554。
- 2026-06-06: 小さい artifact、metrics、kernel log を `experiments/exp023_pseudo_tail_distance_augmentation/artifacts/` に保存し、`metrics.json` と `result.md` を更新。

## 変更点

- `exp013` の `row_oof_predictions.csv` から raw / last_anchor / recent_linear / exp014 bucket shrink を距離 bucket 別 reference として比較する。
- LightGBM no-GR の control、1 cutoff/well、3 cutoffs/well、distance-balanced sampling、pseudo-tail + distance-balanced sampling を同一 fold で比較する。
- pseudo cutoff は train fold の well 内だけで生成し、valid fold は本来の `TVT_input.isna()` 行だけを評価する。
- feature importance は variant / fold / segment 別に保存する。

## Artifacts

- `artifacts/distance_candidate_metrics.csv`
- `artifacts/distance_residual_bucket_summary.csv`
- `artifacts/pseudo_tail_training_metrics.csv`
- `artifacts/pseudo_tail_feature_importance.csv`
- `artifacts/pseudo_tail_source_summary.csv`
- `artifacts/pseudo_tail_training_summary.json`
- `metrics.json`

## 結果

- Best: `pseudo_tail_3_cutoffs_distance_balanced` 12.942938
- `pseudo_tail_1_cutoff`: 12.971839
- `pseudo_tail_3_cutoffs`: 13.012302
- `distance_balanced_sampling`: 13.441648
- `control_lightgbm_no_gr`: 13.494554
- Raw exp013 anchor: 13.549257
- exp021 weighted bucket shrink reference: 13.415799

`pseudo_tail_3_cutoffs_distance_balanced` は raw anchor から -0.606319、exp021 weighted bucket shrink から -0.472861 改善した。距離 bucket 平均でも near / mid / far の全域で control より改善している。

## 次のアクション

1. `pseudo_tail_3_cutoffs_distance_balanced` を inference notebook に反映する。
2. pseudo-tail raw、pseudo-tail + bucket shrink、exp021 weighted bucket shrink、exp013 LB anchor を比較し、提出候補を決める。
