# exp192_typewell_late_range_hard_window_pct50_full_cache_replacement

## 概要

typewell の前半 TVT range を生成前に切り、`typewell_pct >= 0.50` の typewell rows だけで exp072-style full replay train feature cache を作り直した。

- Route: `pf_beam`
- Status: `completed_supported_for_downstream_replacement_test`
- 親実装: `exp186_typewell_late_range_pfbeam_generation_soft_prior`
- 比較対象: `exp072_exp063_full_replay_feature_cache`
- Kaggle kernel: `kentookumura/exp192-typewell-hard-window-pct50-train` v1

## 状態

Kaggle train v1 完了。feature cache と exp072 direct comparison を保存済み。model、prediction、submission は生成していない。

## 仮説

EDA 上の typewell 後半集中を、soft penalty ではなく generation input の hard support として入れると、PF_ANCC / PF_Z / Beam / likelihood-PF の direct candidate が改善する可能性がある。ただし exp186 soft prior で `likpf_mean` が大きく悪化しているため、full replacement としては高リスク。

## 実装

`hard_window_public_replay.py` は raw typewell 読み込み直後に、元の finite TVT min/max から `typewell_pct` を計算し、`0.50 <= typewell_pct <= 1.00` の行だけを PF_ANCC、PF_Z、Beam、128-seed likelihood-PF に渡す。

soft prior、`0.60/0.70` grid、LightGBM 学習、inference、submit はこの実験では実行していない。

## 検証方針

raw train horizontal/typewell files から exp072-style full replay train feature cache を再生成する。既存 exp072 cache は生成 input に使わず、生成後に同一 row で direct PF/Beam RMSE/MAE/within10、distance bucket、true typewell pct bucket、by-well regression を比較する。

## 結果

- rows / wells / features: 3,783,989 / 773 / 196
- runtime: 13,275.591 sec
- raw gzip SHA: `1040d7d3b9254b5a36d2a3f7fd526ae28e3ddd5b29059926b44bbe9d84436e6a`
- decompressed SHA: `a86dff450b108e4481208a5f5699f8624eaf736cb6eb6aa735d39b4044c6f0e1`
- exp072 comparison: `rows_checked=3,783,989`, `unique_wells=773`, `id_mismatches=0`

Direct PF/Beam RMSE TVT:

| candidate | exp072 | exp192 | delta |
| --- | ---: | ---: | ---: |
| `pf_ancc` | 14.493061 | 13.821178 | -0.671884 |
| `pf_z` | 17.788174 | 19.705112 | +1.916938 |
| `beam_mean` | 15.774328 | 15.677016 | -0.097311 |
| `beam_sm5` | 16.313542 | 16.152930 | -0.160612 |
| `likpf_mean` | 11.594898 | 11.544812 | -0.050086 |

## 所見

`likpf_mean` と `pf_ancc` は改善したため、direct cache candidate としては支持する。ただし `pf_z` の大幅悪化、by-well regression、true typewell pct 前半 subset の悪化があるため、PF/Beam route の提出候補にはしない。続ける場合は downstream ML replacement-only 比較で guard を確認する。

## 生成物

- `artifacts/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement_full_replay_cache_pixiux_likpf_hard_window_pct50_public_replay_train_features.csv.gz`
- `artifacts/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement_full_replay_cache_feature_schema.csv`
- `artifacts/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement_full_replay_cache_summary.json`
- `artifacts/exp192_vs_exp072_overall_metrics.csv`
- `artifacts/exp192_vs_exp072_distance_bucket_metrics.csv`
- `artifacts/exp192_vs_exp072_true_typewell_pct_metrics.csv`
- `artifacts/exp192_vs_exp072_by_well_delta.csv`
- `artifacts/exp192_vs_exp072_summary.json`

## Files

- 学習 notebook: `exp192_typewell_late_range_hard_window_pct50_full_cache_replacement_train.ipynb`
- 推論 notebook: `exp192_typewell_late_range_hard_window_pct50_full_cache_replacement_inference.ipynb`
- 実装: `hard_window_public_replay.py`, `feature_cache.py`
