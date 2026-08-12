# 設計

## アプローチ

`exp029` の train well の途中以降を隠した疑似 test public sel15 PF/Beam artifact を読み、`last_anchor_tvt` からの residual を単体 LightGBM で学習する。特徴量候補は `base_geometry`、`base + PF prediction`、`base + Beam prediction`、`base + PF/Beam diagnostics` に分ける。

各候補は well-level split で cross-fit し、raw prediction と `exp014_bucket_shrink_params` を適用した prediction の両方を評価する。参考 control として `last_anchor`、public PF selector、Beam、必要に応じて regenerated exp026 bucket-shrink anchor を出す。

## 実験範囲

- 対象実験: `exp038_ravaghi_public_sel15_features_single_lgbm`
- Route: `ml_model`
- 親実験: `exp026_pseudo_tail_bucket_shrink_inference_submit`
- Supporting artifact: `exp029_public_sel15_pf_oof_feature_generation`
- 変更する変数: Ravaghi/public sel15 PF/Beam feature family
- 固定する変数: LightGBM recipe、row cap、distance bucket shrink、well-level audit split

## リスク

- リークリスク: exp029 artifact は train well の途中以降を隠した疑似 test 生成だが、error diagnostic や exp026_oof bridge columns を feature に入れると stack/diagnostic leakage になるため除外する。
- CV/LB 不一致リスク: exp029 train well の途中以降を隠した疑似 test 評価条件 は exp026 clean CV surface と違うため、改善しても inference port に直結しない。
- ランタイム/メモリリスク: full artifact は約 1.78M rows のため、chunked load と per-split row cap を使う。
