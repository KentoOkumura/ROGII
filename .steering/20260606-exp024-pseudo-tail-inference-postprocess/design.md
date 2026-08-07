# 設計

## アプローチ

`exp023_pseudo_tail_distance_augmentation` を親にし、selected `pseudo_tail_3_cutoffs_distance_balanced` の学習行生成を final fit に再利用する。all train wells から original tail と pseudo tail rows を作り、distance-balanced sampling 後に LightGBM を学習する。

## 実験範囲

- 対象実験: `exp024_pseudo_tail_inference_postprocess`
- 親実験: `exp023_pseudo_tail_distance_augmentation`
- 変更する変数: inference final fit と submission generation
- 固定する変数: exp023 selected variant、LightGBM no-GR feature set、raw postprocess

## リスク

- リークリスク: test well では target の未知区間を参照しない。
- CV/LB 不一致リスク: exp023 は train-side CV のみで Public LB 未確認。submit 前に visible prediction sanity と submit-check を通す。
- ランタイム/メモリリスク: final fit は pseudo rows を含むため、`max_train_rows_final` で cap する。
