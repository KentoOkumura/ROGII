# exp017_deterministic_dtw_addonly セッションノート

## 目的

`exp013_model_diversity_or_postprocess` を親に、`exp012/exp013` の raw `lightgbm_no_gr` anchor へ fold-safe な deterministic DTW/DWT alignment features を追加し、Kaggle full CV で比較する。

## 現在の状態

- 状態: Kaggle full train 完了、結果記録済み
- 親実験: `exp013_model_diversity_or_postprocess`
- Raw CV anchor: `exp012/exp013 lightgbm_no_gr` 13.549257
- selected variant before CV: `dtw_dwt_no_gr`
- CV: `control_lightgbm_no_gr` 13.549257、`dtw_dwt_no_gr` 13.949718
- Public LB: なし

## コマンドログ

- 2026-06-05: `uv run python scripts/new_steering.py --experiment exp017_deterministic_dtw_addonly` で steering docs を作成。
- 2026-06-05: `uv run python scripts/new_experiment.py --name exp017_deterministic_dtw_addonly --source experiments/exp015_public_pf_beam_scale_selector_features` で exp015 から実験を作成。
- 2026-06-05: notebook 名を exp017 に変更し、`settings.py` の `EXPERIMENT_NAME` を更新。
- 2026-06-05: `.steering/20260605-exp017-deterministic-dtw-addonly/{requirements.md,design.md,tasklist.md}` に仮説、設計、タスクを記入。
- 2026-06-05: `config.yaml` を exp017 用に置換し、`control_lightgbm_no_gr` と `dtw_dwt_no_gr` の 2 variants を定義。
- 2026-06-05: `baseline.py` に deterministic DTW/DWT feature generator を追加。
- 2026-06-05: `uv run python -m py_compile experiments/exp017_deterministic_dtw_addonly/baseline.py experiments/exp017_deterministic_dtw_addonly/settings.py` が通過。
- 2026-06-05: `uv run ruff check experiments/exp017_deterministic_dtw_addonly/baseline.py experiments/exp017_deterministic_dtw_addonly/settings.py` が通過。
- 2026-06-05: `uv run python scripts/validate_experiment.py --experiment exp017_deterministic_dtw_addonly` が通過。
- 2026-06-05: feature sanity check で control 25 features、DTW/DWT variant 62 features、`dtw_dwt_*` 37 features を確認。
- 2026-06-05: train/inference notebook を `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp017_deterministic_dtw_addonly --notebook <train|inference> --strict` で準備できることを確認。
- 2026-06-05: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp017_deterministic_dtw_addonly --notebook train --run-on-push --title "exp017 deterministic dtw addonly train" --strict` が通過。
- 2026-06-05: `uv run pytest` が通過。9 tests passed。
- 2026-06-05: `uv run python scripts/update_experiment_summary.py` で `experiment_summary.md` に exp017 を追加。
- 2026-06-05: `kaggle kernels push -p experiments/exp017_deterministic_dtw_addonly/kaggle/train` で version 1 を push。URL: https://www.kaggle.com/code/kentookumura/exp017-deterministic-dtw-addonly-train
- 2026-06-05: `kaggle kernels status kentookumura/exp017-deterministic-dtw-addonly-train` は Kaggle API の 500 で取得不可。既存 exp013/exp015 でも同じ 500 のため status endpoint 側の問題として扱う。
- 2026-06-05: `kaggle kernels output kentookumura/exp017-deterministic-dtw-addonly-train -p /tmp/kaggle-output/exp017_deterministic_dtw_addonly/train` で output を取得。
- 2026-06-05: Kaggle output の小さい `artifacts/*.csv/json` を `experiments/exp017_deterministic_dtw_addonly/artifacts/` に反映。`row_oof_predictions.csv` は 1.1GB のため `/tmp/kaggle-output/exp017_deterministic_dtw_addonly/train/artifacts/row_oof_predictions.csv` に残し、実験ディレクトリには常設しない。
- 2026-06-05: `artifacts/ablation_metrics.csv` を確認。raw anchor の `control_lightgbm_no_gr` は CV 13.549257 を再現、`dtw_dwt_no_gr` は CV 13.949718 で +0.400461 悪化。`distance_bucket_shrink_fit` 後も 13.910963 で raw anchor に届かないため、DTW/DWT add-only features は採用しない。

## 変更点

- `control_lightgbm_no_gr`: exp012/exp013 selected LightGBM no-GR raw anchor の再実行。
- `dtw_dwt_no_gr`: `no_gr_signal` features に `dtw_dwt_*` を追加。
- `dtw_dwt_*` features:
  - typewell GR に対する bounded shift search。
  - rolling smooth / detail energy による pywt なし multi-scale DWT 近似。
  - best shift、cost margin、NCC、banded DTW cost、route slope。
  - scale 別 smooth/energy mismatch と row-level typewell GR mismatch。

## 次のアクション

1. `distance_bucket_router` で alignment 候補を直接特徴追加ではなく routing / confidence 目的に限定して検討する。
