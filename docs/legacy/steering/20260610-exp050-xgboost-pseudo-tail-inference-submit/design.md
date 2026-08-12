# 設計

## アプローチ

`exp026_pseudo_tail_bucket_shrink_inference_submit` を実装親としてコピーし、Kaggle inference notebook と `generate_pseudo_tail_submission` を再利用する。`baseline.py` に `XGBRegressor` 対応を追加し、`config.yaml` で exp049 と同じ XGBoost パラメータへ切り替える。後処理は exp026 と同じ fixed `exp014_bucket_shrink_params` を使う。

## 実験範囲

- 対象実験: `exp050_xgboost_pseudo_tail_inference_submit`
- Route: `ml_model`
- 親実験: `exp049_xgboost_pseudo_tail_residual`
- 実装親: `exp026_pseudo_tail_bucket_shrink_inference_submit`
- 変更する変数: final residual estimator (`LGBMRegressor` -> `XGBRegressor`)
- 固定する変数: no-GR feature set、pseudo cutoff quantiles `[0.45, 0.65, 0.82]`、distance-balanced sampling、row caps、residual shrink、fixed bucket-shrink alpha

## リスク

- リークリスク: final fit は official train wells のみ、test は visible prefix のみを使う。train-only formation columns は使わない。
- CV/LB 不一致リスク: exp021/026 では CV 改善が Public LB に完全転移しなかった。submit 前に予測範囲と exp026 差分を確認する。
- ランタイム/メモリリスク: XGBoost final fit は LightGBM より重い可能性がある。Kaggle CPU inference で実行時間を確認する。
