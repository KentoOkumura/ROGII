# 設計

## アプローチ

train の `*__horizontal_well.csv` から 1 well 1 row の metadata を作成し、coarse bin を組み合わせた `strat_label` を作る。`StratifiedGroupKFold` は `well_id` を group とし、rare label は `rare` に畳み込んで fold が破綻しないようにする。

既存 OOF artifact は chunk 読み込みで処理し、`well_id` に fold assignment と metadata bins を merge する。raw、last anchor、fixed `exp014_bucket_shrink_params` を candidate として、overall / original fold / new stratified fold / metadata bucket / distance bucket ごとに RMSE を保存する。

## 実験範囲

- 対象実験: `exp044_stratified_groupkfold_cv_audit`
- Route: `ml_model`
- 親実験: `exp026_pseudo_tail_bucket_shrink_inference_submit`
- 変更する変数: validation stress split と監査集計軸
- 固定する変数: モデル、特徴量、提出 flow、Public LB anchor

## リスク

- リークリスク: median TVT は split 診断にだけ使い、モデル特徴や inference には使わない。
- CV/LB 不一致リスク: この split のスコアだけで提出判断しない。既存 clean CV / Public LB anchor と分けて記録する。
- ランタイム/メモリリスク: 1.1GB 級 OOF CSV は chunk 読み込みし、欠損時は source status に記録して fold metadata 監査だけ通す。
