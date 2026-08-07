# 設計

## アプローチ

`exp020` の LightGBM no-GR residual CV audit を親にし、学習データ生成部分だけを pseudo-tail variants に差し替える。各 train well で既知 `TVT_input` prefix 内の quantile cutoff を選び、cutoff 後を NaN にしたコピーから `build_drift_feature_frame` を再利用して residual 学習行を作る。

## 実験範囲

- 対象実験: `exp023_pseudo_tail_distance_augmentation`
- 親実験: `exp020_distance_weighted_training_audit`
- 変更する変数: train-fold-only pseudo cutoff 数、distance-balanced sampling の有無
- 固定する変数: LightGBM no-GR feature set、GroupKFold by well、評価 mask、postprocess なしの raw residual CV

## リスク

- リークリスク: valid well から pseudo rows を作ると fold leakage になるため、cutoff generation は train fold path のみに限定する。
- CV/LB 不一致リスク: pseudo-tail は train CV の distribution を増やすが、hidden test の cutoff 分布とずれる可能性がある。距離 bucket 別 RMSE と control 再現を必ず見る。
- ランタイム/メモリリスク: 3 cutoffs/well は行数が増えるため、per pseudo tail sampling、fold total cap、distance-balanced cap を config で固定する。
