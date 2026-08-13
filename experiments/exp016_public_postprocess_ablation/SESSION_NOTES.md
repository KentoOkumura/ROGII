# exp016_public_postprocess_ablation セッションノート

## 目的

`exp013_model_diversity_or_postprocess` の `lightgbm_no_gr` OOF を使い、
公開 notebook 由来の後処理候補を同じ評価面で切り分ける。

## 現在の状態

- 状態: 完了
- CV: 13.549257
- LB: まだなし

## コマンドログ

- 2026-06-05: `uv run python scripts/new_steering.py --experiment exp016_public_postprocess_ablation` で steering docs を作成。
- 2026-06-05: `uv run python scripts/new_experiment.py --name exp016_public_postprocess_ablation` で実験を作成。
- 2026-06-05: `docs/legacy/steering/20260605-exp016-public-postprocess-ablation/{requirements.md,design.md,tasklist.md}` を記入。
- 2026-06-05: `config.yaml` を OOF 後処理 ablation 用に更新。
- 2026-06-05: `audit_public_postprocess.py` を追加。
- 2026-06-05: `uv run python experiments/exp016_public_postprocess_ablation/audit_public_postprocess.py` を実行し、artifacts と `metrics.json` を生成。
- 2026-06-05: `exp013_bucket_shrink` は同一 OOF fit 由来のため selectable から外し、audit を再実行。

### 検証

```bash
uv run python experiments/exp016_public_postprocess_ablation/audit_public_postprocess.py
```

## 変更点

- 新規学習は行わず、`data/external/kaggle-output/exp013_model_diversity_or_postprocess/train/artifacts/row_oof_predictions.csv` の `lightgbm_no_gr` を監査する。
- raw、last anchor、SG smoothing、fade-in、hold blend、alpha/tau shrink、exp013 bucket shrink を比較する。
- same-OOF RMSE と fold 外 candidate selection RMSE を分離して記録する。
- `exp013_bucket_shrink` は same-OOF 比較には残すが、fold 外 candidate selection からは除外する。

## 結果

- Raw clean CV: 13.549257
- Best same-OOF: `exp013_bucket_shrink` 13.501824。ただし同一 OOF fit 由来。
- Best fixed same-OOF: `alpha_tau_250_a020_115` 13.515133。
- Leave-one-original-fold-out candidate selection: 13.551561。raw から +0.002303 悪化。
- Well-hash holdout candidate selection: 13.515133。raw から -0.034125 改善。
- Clean CV として採用する値: 13.549257。original fold 外で改善が残らないため、固定 public-style postprocess は診断扱い。

## Artifacts

- `artifacts/public_postprocess_ablation_metrics.csv`
- `artifacts/public_postprocess_ablation_selection.csv`
- `artifacts/public_postprocess_ablation_bucket_summary.csv`
- `artifacts/public_postprocess_ablation_summary.json`
- `metrics.json`

## 次のアクション

1. `experiment_summary.md` と `backlog/KAGGLE_DIRECTION.md` に反映する。
2. 次は DWT/DTW route または candidate quality/routing の検証に進む。
