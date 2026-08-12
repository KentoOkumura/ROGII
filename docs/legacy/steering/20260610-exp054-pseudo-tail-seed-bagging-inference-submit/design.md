# 設計

## アプローチ

`exp052` の inference flow をコピーし、final model fit のみ `kind: seed_bagging` に対応させる。member ごとに full train wells から pseudo-tail rows を同じ設定で再サンプルし、LightGBM `random_state` も member seed に合わせる。test well の raw prediction は 3 member の平均にし、その平均 raw prediction に fixed bucket shrink を適用する。

## 実験範囲

- 対象実験: `exp054_pseudo_tail_seed_bagging_inference_submit`
- Route: `ml_model`
- 親実験: `exp053_pseudo_tail_seed_bagging`
- 実装親: `exp052_lgbm_capacity_pseudotail_inference_submit`
- 変更する変数: final residual model の seed member 平均
- 固定する変数: cutoff quantiles、distance-balanced sampling、feature set、LightGBM params、residual shrink、bucket shrink、submission format

## リスク

- リークリスク: final fit は official train wells のみを使用し、test は known prefix のみで予測する。
- CV/LB 不一致リスク: exp053 の CV 改善は exp051 から -0.000595 と極小なので、LB が悪化する可能性もある。目的は seed 変更の LB 影響確認。
- ランタイム/メモリリスク: exp052 の final fit を 3 回行うため inference runtime は増える。Kaggle CPU で実行時間をログから確認する。
