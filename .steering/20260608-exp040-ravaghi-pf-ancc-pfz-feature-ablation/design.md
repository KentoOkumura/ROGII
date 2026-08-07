# 設計

## アプローチ

`exp038` の single-LGBM audit を土台に、Ravaghi notebook の `pf_ancc` / `pf_z` 系 feature を `exp029` artifact から作れる proxy に落とす。

- `pf_ancc_delta_proxy`: `pf_pred - last_anchor_tvt`、absolute delta、`TVT + Z` surface delta、delta per eval step。
- `pf_z_proxy`: cutoff 先頭からの `Z` / `MD` delta、`last_anchor_tvt - z_delta` proxy、PF-vs-Z proxy。
- `pf_uncertainty`: PF scale mean/std/range、selected scale disagreement、seed std、likelihood margin、weight entropy。

## 実験範囲

- 対象実験: `exp040_ravaghi_pf_ancc_pfz_feature_ablation`
- Route: `ml_model`
- 親実験: `exp026_pseudo_tail_bucket_shrink_inference_submit`
- supporting artifact: `exp029_public_sel15_pf_oof_feature_generation`
- 変更する変数: single LightGBM に入れる Ravaghi PF ANCC/PFZ-style feature family。
- 固定する変数: split、target、LightGBM params、distance bucket shrink、report controls。

## リスク

- リークリスク: `ANCC` 等の train-only formation columns は読まない。train well の途中以降を隠した疑似 test 生成物 の error columns と exp026 bridge columns は feature から除外する。
- CV/LB 不一致リスク: `exp029` train well の途中以降を隠した疑似 test 評価条件 は 見えない test well 評価の LB に転移しない可能性がある。original-fold と well-hash の両方を要求する。
- ランタイム/メモリリスク: exp038 と同じ chunked CSV load と per-split row cap を使う。full audit は Kaggle Notebook で実行する。
