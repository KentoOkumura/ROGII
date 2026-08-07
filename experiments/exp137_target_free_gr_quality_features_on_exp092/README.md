# exp137_target_free_gr_quality_features_on_exp092

## 概要

exp092 の U-projection correction / disagreement LightGBM surface に、target-free な GR 品質特徴量を add-only で入れる実験。

生の GR 値、NCC/DTW の波形 score、GR 由来の candidate TVT は入れない。追加するのは、GR coverage、missing run、row 周辺の finite GR interpolation gap、prefix/eval の GR 分布 mismatch 要約、exp065 native typewell overlap context に限定する。

## 仮説

GR 波形特徴そのものは過去実験で不安定だったが、GR が欠けている、prefix と eval の GR regime がずれている、同じ native typewell group の信頼度が高い、といった品質情報だけなら exp092 の誤差 regime を補足できる可能性がある。

## 検証方針

GroupKFold by well で `exp092_full_row_control` と `target_free_gr_quality_addonly` を同一 row 上で比較する。pooled RMSE、well-level regression、near-row、1000+ longtail、PF-dense disagreement、feature importance を確認し、改善しても raw-test parity と exp115 hidden-like stress を確認するまで submission candidate にはしない。

## 生成物

- `exp137_target_free_gr_quality_features_on_exp092_metrics.csv`
- `exp137_target_free_gr_quality_features_on_exp092_by_well.csv`
- `exp137_target_free_gr_quality_features_on_exp092_bucket_metrics.csv`
- `exp137_target_free_gr_quality_features_on_exp092_projection_feature_summary.csv`
- `exp137_target_free_gr_quality_features_on_exp092_diagnostic_feature_summary.csv`
- `exp137_target_free_gr_quality_features_on_exp092_feature_importance.csv`
- `exp137_target_free_gr_quality_features_on_exp092_feature_importance_mean.csv`
- `exp137_target_free_gr_quality_features_on_exp092_feature_importance_mean_top.png`
- `exp137_target_free_gr_quality_features_on_exp092_predictions.csv.gz`
- `exp137_target_free_gr_quality_features_on_exp092_feature_schema.csv`
- `exp137_target_free_gr_quality_features_on_exp092_lgb_models/manifest.json`
- `exp137_target_free_gr_quality_features_on_exp092_summary.json`

## 状態

完了。不採用。

## 所見

Kaggle train v1 は CPU runtime timeout/cancel。v2 は `lgb0` のみの full-row 比較として完走した。

`target_free_gr_quality_addonly` は pooled RMSE 9.729657 で、`exp092_full_row_control` 9.535793 から +0.193863 悪化した。near-prefix bucket は微小改善したが、`1000_plus` longtail と複数 well で悪化したため、inference port / submission / 3-config full rerun は行わない。
