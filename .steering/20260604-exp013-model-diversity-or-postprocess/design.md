# 設計

## アプローチ

`exp012` を親にコピーし、まず `lightgbm_no_gr` を raw anchor として再実行する。
train notebook では validation well ごとに row-level OOF を作成し、以下を fold 外で採点する。

- `raw_lightgbm_no_gr`: exp012 相当の raw residual model。
- `sg_smooth`: well 内の raw prediction を Savitzky-Golay 風の rolling median/mean smoothing で平滑化する。
- `global_residual_shrink`: `last_anchor + alpha * (raw - last_anchor)` の alpha を小さく比較する。
- `near_anchor_damping`: `eval_step` が小さい row だけ anchor 寄りに damp し、遠方は raw を維持する。
- `distance_bucket_shrink`: `eval_step` bucket ごとに OOF で alpha を推定し、bucket 外挿を避けるため alpha を保守範囲に clip する。
- `hgb_lightgbm_nnls`: 既存 HGB no-GR control と LightGBM no-GR の OOF を非負重みで小さく blend する。

## 実験範囲

- 対象実験: `exp013_model_diversity_or_postprocess`
- 親実験: `experiments/exp012_single_catboost_lightgbm_residual`
- 変更する変数: postprocess rule、row-level OOF 保存、selected postprocess の inference 適用。
- 固定する変数: GroupKFold by well、target residual、last-anchor baseline、LightGBM no-GR hyperparameters、sampling cap、seed、train-only formation columns 不使用。

## リスク

- リークリスク: distance bucket alpha は OOF 全体で推定するため valid target を使う。これはモデル選択としてのみ使い、inference では固定 alpha を config に書き込む。row ごとの future target や test aggregate は使わない。
- CV/LB 不一致リスク: bucket / smoothing の過最適化で Public LB に出ない可能性がある。候補数を少なくし、well group summary と距離 bucket summary を出して局所改善だけか確認する。
- ランタイム/メモリリスク: OOF CSV は row-level だが train evaluation rows のみなので許容範囲。model diversity は HGB no-GR と LightGBM no-GR の 2 モデルに限定する。
