# 設計

## アプローチ

`exp051` の train-side pseudo-tail audit をコピーし、variant kind として `seed_bagging` を追加する。通常 variant は従来通り 1 回だけ training rows を sample して 1 model を fit する。`seed_bagging` variant では member seed ごとに training rows を再サンプルし、同じ LightGBM capacity parameter で model を fit する。

valid prediction は member raw prediction の平均とし、postprocess は平均後の raw prediction に fixed `exp014_bucket_shrink_params` を適用する。これにより、bagging の効果を「モデル/サンプリング分散の低減」として切り出す。

## 実験範囲

- 対象実験: `exp053_pseudo_tail_seed_bagging`
- Route: `ml_model`
- 親実験: `exp051_pseudo_tail_lgbm_param_micro_tune`
- 変更する変数: pseudo-tail sampling RNG、LightGBM `random_state`、seed member prediction average
- 固定する変数: cutoff quantiles、distance-balanced sampling cap、feature set、LightGBM capacity params、residual shrink、fixed bucket-shrink params、GroupKFold

## ベース選定

古い backlog では `exp024/026` single seed 再現を前提にしていたが、2026-06-10 時点では `exp051` が通常 CV 12.634392、`exp052` が pseudo-tail 自前系 Public LB 12.076 で直近基準になっている。seed bagging は構造変更ではなく分散低減の検証なので、現行最良の `exp051` capacity model をベースにする。

## リスク

- リークリスク: pseudo-tail rows は train-fold wells の中だけで作り、valid-fold wells は本来の evaluation zone のみで score する。
- CV/LB 不一致リスク: exp021/026/050/052 と同様に CV 改善が Public LB に転移しない可能性がある。full CV 後も inference port 前に予測範囲と差分を監査する。
- ランタイム/メモリリスク: 3 seed bagging は fold ごとに 3 model fit するが、exp051 の 7 variants より総 model 数は少ない。5 seed 以上はこの実験では扱わない。
