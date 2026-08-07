# 要件

## 依頼

`stratified_groupkfold_cv_audit` を実装し、signed azimuth、median TVT、spatial location を含む well-level stratification で GroupKFold の stress report を作る。

## 制約

- Route: `ml_model`
- 新 primary CV へ即置換しない。通常の `GroupKFold(well_id)` は anchor として維持する。
- spatial BlockKFold は採用しない。spatial location は層化 bucket の一部として使う。
- train-only formation columns は使わない。
- OOF artifact がない候補は欠損として記録し、fold balance 監査は完了できること。

## 受け入れ基準

- train well ごとに azimuth / TVT / spatial / eval length / GR coverage の metadata と coarse bin が保存される。
- `StratifiedGroupKFold(groups=well_id)` と通常 `GroupKFold` の fold 分布比較が保存される。
- 設定済み OOF source が存在する場合、candidate RMSE が fold / bucket / distance bucket 別に再集計される。
- `metrics.json`、`SESSION_NOTES.md`、`result.md` に diagnostic-only 実験として記録される。
