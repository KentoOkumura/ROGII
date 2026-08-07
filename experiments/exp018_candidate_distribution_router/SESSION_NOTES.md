# exp018_candidate_distribution_router セッションノート

## 目的

バックログ先頭の `candidate_distribution_router` を実装する。既存 OOF 候補を
row distance と候補間 disagreement で切り替え、PF/beam や DTW/DWT を
直接特徴追加ではなく局所 routing 候補として評価する。

## 現在の状態

- 状態: 完了
- 親実験: `exp013_model_diversity_or_postprocess`
- Raw CV anchor: `exp012/exp013 lightgbm_no_gr` 13.549257
- Public LB anchor: `exp013` 12.271
- CV: raw clean CV 13.549257、selected clean CV 13.549257

## コマンドログ

- 2026-06-05: `task new-steering EXP=exp018_candidate_distribution_router` と `task new-exp ...` は `task` が未インストールで失敗。
- 2026-06-05: `uv run python scripts/new_steering.py --experiment exp018_candidate_distribution_router` で steering docs を作成。
- 2026-06-05: `uv run python scripts/new_experiment.py --name exp018_candidate_distribution_router --source experiments/exp016_public_postprocess_ablation` で exp016 から実験を作成。
- 2026-06-05: notebook 名、`settings.py`、`config.yaml` を exp018 用に更新。
- 2026-06-05: `audit_candidate_distribution_router.py` を実装。
- 2026-06-05: `uv run python -m py_compile experiments/exp018_candidate_distribution_router/audit_candidate_distribution_router.py experiments/exp018_candidate_distribution_router/settings.py` が通過。
- 2026-06-05: `uv run ruff check experiments/exp018_candidate_distribution_router/audit_candidate_distribution_router.py experiments/exp018_candidate_distribution_router/settings.py` が通過。
- 2026-06-05: `uv run python scripts/validate_experiment.py --experiment exp018_candidate_distribution_router` が通過。
- 2026-06-05: 初回 audit 実行は `distance_router` の read-only 配列代入で失敗。`candidate_pred()` が writable copy を返すよう修正。
- 2026-06-05: `uv run python experiments/exp018_candidate_distribution_router/audit_candidate_distribution_router.py` を実行し、artifacts と `metrics.json` を生成。

## 変更点

- `last_anchor`、`raw_lightgbm_no_gr`、`control_hgb_no_gr`、`dtw_dwt_no_gr`、任意 `pf_beam_no_gr` を候補として扱う。
- 欠けている任意 OOF artifact はスキップする。
- fixed / weighted blend / distance router / disagreement damping / bucket oracle を比較する。
- same-OOF RMSE、bucket 別 RMSE、leave-one-original-fold-out selection、well-hash holdout selection を出力する。

## Artifacts

- `artifacts/candidate_router_metrics.csv`
- `artifacts/candidate_router_selection.csv`
- `artifacts/candidate_router_bucket_summary.csv`
- `artifacts/candidate_router_summary.json`
- `metrics.json`

## 結果

- Loaded candidates: `raw_lightgbm_no_gr`, `control_hgb_no_gr`, `dtw_dwt_no_gr`
- Skipped candidates: `pf_beam_no_gr`。exp015 の row OOF artifact がローカルにないため。
- Raw clean CV: 13.549257
- Best same-OOF router: `disagreement_damped_raw` 13.537122。62,757 rows を damping。
- Bucket oracle same-OOF: 13.545073。`rows_0_49=last_anchor`、`rows_50_249=dtw_dwt_no_gr`、以降は raw。
- Leave-one-original-fold-out router selection: 13.644470。raw から +0.095213 悪化。
- Well-hash holdout router selection: 13.646503。raw から +0.097246 悪化。
- Clean CV として採用する値: 13.549257。router は診断扱い。

## 次のアクション

1. `experiment_summary.md` と `KAGGLE_DIRECTION.md` に反映する。
2. candidate routing は提出実装へ進めず、PF/beam artifact が復元された場合だけ品質監査として再確認する。
