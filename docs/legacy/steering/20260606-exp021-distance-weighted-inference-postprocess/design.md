# 設計

## アプローチ

`exp020` で clean CV 13.470015 を出した `near_down_far_up_lightgbm` を提出候補化する。実装は `exp020` の LightGBM residual pipeline を継承し、sample weight の selected profile を final training と fold CV の両方に適用する。

postprocess は `exp014` の distance bucket shrink parameter を再利用する。weighted model は rows 0-249 で raw より悪化していたため、near bucket を anchor に寄せる候補を weighted raw と同時に監査する。

## 実験範囲

- 対象実験: `exp021_distance_weighted_inference_postprocess`
- 親実験: `exp020_distance_weighted_training_audit`
- 変更する変数: final/inference training の sample weight、weighted OOF に対する distance bucket shrink
- 固定する変数: feature set `no_gr_signal`、LightGBM params、row caps、well-level GroupKFold、exp014 bucket alpha

## リスク

- リークリスク: bucket shrink の alpha は過去 OOF 由来の固定値として使う。exp021 で同じ OOF に fit し直した値を clean CV として扱わない。
- CV/LB 不一致リスク: Public LB は 3 visible wells 由来で、weighted far-row 改善がそのまま反映されない可能性がある。CV と Public LB は別 anchor として記録する。
- ランタイム/メモリリスク: full OOF は 5-fold LightGBM を再学習するため Kaggle 実行前提。row cap は exp020 と同じ `300000/fold`、final は `450000` に固定する。
