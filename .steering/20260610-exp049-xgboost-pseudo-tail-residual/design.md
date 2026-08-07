# 設計

## アプローチ

`exp023_pseudo_tail_distance_augmentation` を実装親としてコピーし、`exp026` で採用された pseudo-tail + distance-balanced 構成だけを残す。`baseline.py` の model factory に `XGBRegressor` を追加し、`config.yaml` で estimator を切り替える。OOF 予測は raw と fixed bucket-shrink 後の両方を保存し、LightGBM の `exp023/026` と同じ GroupKFold surface で比較する。

## 実験範囲

- 対象実験: `exp049_xgboost_pseudo_tail_residual`
- Route: `ml_model`
- 親実験: `exp026_pseudo_tail_bucket_shrink_inference_submit`
- 変更する変数: residual estimator (`LGBMRegressor` -> `XGBRegressor`)
- 固定する変数: no-GR feature set、pseudo cutoff quantiles `[0.45, 0.65, 0.82]`、distance-balanced sampling、row caps、residual shrink、bucket shrink alpha、GroupKFold split 方針

## リスク

- リークリスク: pseudo cutoff は train fold well 内だけで生成し、valid fold は本来の missing tail だけを評価する。train-only formation columns は使わない。
- CV/LB 不一致リスク: `exp021/026` で CV 改善が Public LB に十分転移しない例があるため、改善しても即提出せず、distance bucket と補助 fold を確認する。
- ランタイム/メモリリスク: XGBoost hist は LightGBM より遅い可能性がある。最初は既存 row cap と `n_jobs=2` を維持し、Kaggle train notebook で full CV を実行する。
