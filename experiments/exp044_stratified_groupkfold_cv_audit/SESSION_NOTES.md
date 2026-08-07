# exp044_stratified_groupkfold_cv_audit セッションノート

## 目的

signed azimuth / median TVT / spatial location / eval length / GR coverage で層化した well-level `StratifiedGroupKFold` を作り、既存 OOF artifact を stress bucket 別に再集計する。primary CV の置換ではなく、primary CV で改善した候補に対する補助的な red-flag report として使う。exp044 だけで候補採用、ハイパラ調整、postprocess fit は行わない。

## 現在の状態

- Route: ml_model
- 状態: ローカル full audit 完了
- CV: なし。診断実験のため primary score は記録しない。
- LB: まだなし

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

- 2026-06-09: `uv run python scripts/new_steering.py --experiment exp044_stratified_groupkfold_cv_audit` で steering docs を作成。
- 2026-06-09: `uv run python scripts/new_experiment.py --name exp044_stratified_groupkfold_cv_audit` で実験ディレクトリを作成。
- 2026-06-09: `stratified_groupkfold_cv_audit.py`、config、train/inference notebooks、docs を実装。
- 2026-06-09: `uv run python scripts/validate_experiment.py --experiment exp044_stratified_groupkfold_cv_audit` が PASS。
- 2026-06-09: `uv run python experiments/exp044_stratified_groupkfold_cv_audit/stratified_groupkfold_cv_audit.py --max-wells 30 --skip-oof` が PASS。30 wells で `strat_labels=4`。
- 2026-06-09: `uv run python experiments/exp044_stratified_groupkfold_cv_audit/stratified_groupkfold_cv_audit.py` で full local audit 完了。773 wells、49 strat labels、exp013/exp017 OOF を再集計。
- 2026-06-09: `uv run python scripts/record_experiment.py --experiment exp044_stratified_groupkfold_cv_audit --status completed ...` で metrics / summary を更新。
- 2026-06-09: `uv run python scripts/prepare_kaggle_notebooks.py --experiment exp044_stratified_groupkfold_cv_audit --notebook train --run-on-push --strict` が PASS。

### 予定

```bash
task prepare-kaggle-notebooks EXP=exp044_stratified_groupkfold_cv_audit EXTRA_ARGS="--notebook train --run-on-push --strict"
task push-kaggle-train EXP=exp044_stratified_groupkfold_cv_audit
task kaggle-status KERNEL=<username>/<train-kernel-slug>
```

## 変更点

- train well metadata を `artifacts/well_metadata_stratified_folds.csv` に保存する。
- usual `GroupKFold` と diagnostic `StratifiedGroupKFold` の fold balance を `artifacts/fold_balance_summary.csv` / `artifacts/fold_bucket_distribution.csv` に保存する。
- 設定済み OOF source を chunk 読み込みし、overall / fold / metadata bin / distance bucket 別 RMSE を `artifacts/stratified_oof_segment_metrics.csv` に保存する。
- inference notebook は `submission.csv` を生成せず、`NO_SUBMISSION.txt` のみ作る。

## 結果

- wells: 773
- strat labels: 49
- exp013 `lightgbm_no_gr` raw: 13.549257
- exp013 fixed `exp014_bucket_shrink_params`: 13.501824
- exp013 last anchor: 15.909853
- exp017 `dtw_dwt_no_gr` raw: 13.949718
- exp017 fixed `exp014_bucket_shrink_params`: 13.911474
- exp013 raw StratifiedGroupKFold fold RMSE: 14.127027 / 13.337070 / 13.058390 / 14.126412 / 13.072675

## 次のアクション

1. 必要なら Kaggle train notebook でも同じ診断 artifact を生成する。
2. `xgboost_pseudo_tail_residual` と `pseudo_tail_lgbm_param_micro_tune` では、primary CV 改善候補だけ exp044 の stress bucket で破壊的悪化がないか確認する。
3. exp044 の stratified fold / bucket に合わせた最適化はしない。
