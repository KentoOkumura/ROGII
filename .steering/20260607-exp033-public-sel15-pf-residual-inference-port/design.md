# 設計

## アプローチ

`exp031` の inference notebook を source にし、公開 sel15 PF/Beam/selector の流れを保つ。notebook 冒頭で `exp029` の `public_sel15_pf_oof_features.csv.gz` を探し、`exp032` と同じ feature columns、`SimpleImputer`、`StandardScaler`、`Ridge(alpha=20)` で `target_tvt - pf_pred` residual を fit する。

見えない test well では、既存 selector prediction に対して inference-time features を作り、Ridge residual を予測する。最終予測は `tvt_selector + 0.5 * clip(residual, -20, 20)` とする。visible train well は従来の physical model を使い、監査上の original prediction と corrected prediction を同一にする。

## 実験範囲

- 対象実験: `exp033_public_sel15_pf_residual_inference_port`
- Route: `pf_beam`
- 親実験: `exp031_public_sel15_pf_hold_blend_inference_audit`
- 変更する変数: 見えない test well の selected residual correction
- 固定する変数: public sel15 PF particles/seeds/scales、Beam configs、selector bins、visible physical model、Kaggle metadata

## リスク

- リークリスク: 見えない test well 用処理は known `TVT_input` prefix と PF/Beam/GR/trajectory/typewell features だけを使う。train well の途中以降を隠した疑似 test 生成物の `target_tvt` は Ridge residual training target に限定する。
- CV/LB 不一致リスク: `exp032` の train-side train well の途中以降を隠した疑似 test 改善が Public LB 8.781 anchor に直結する保証はない。original-fold split 3 は raw PF より悪化した。
- ランタイム/メモリリスク: inference notebook が exp029 1,782,279-row artifact を読み、最大 500,000 rows に sampling して Ridge を fit する。Kaggle input source とメモリ使用を確認する。
