# exp196_typewell_late_range_hard_window_pct40_full_cache_replacement

## 概要

typewell の hard-window full replay cache replacement を exp192 pct50 から pct40 へ緩め、`typewell_pct >= 0.40` の typewell rows だけで exp072-style full replay train feature cache を作り直した。

- Route: `pf_beam`
- Status: `completed_supported_for_downstream_replacement_test_no_submit`
- 親実験: `exp192_typewell_late_range_hard_window_pct50_full_cache_replacement`
- 比較対象: `exp072_exp063_full_replay_feature_cache`, `exp192_typewell_late_range_hard_window_pct50_full_cache_replacement`
- Kaggle kernel: `kentookumura/exp196-typewell-hard-window-pct40-train` v1

## 状態

Kaggle train v1 完了。model、prediction、submission は生成していない。pct40 は direct PF/Beam submit 候補ではなく、downstream ML replacement-only で pct50 と比較するための感度結果として扱う。

## 仮説

exp192 pct50 は `likpf_mean` と `pf_ancc` を改善した一方、`pf_z` と true typewell pct `<0.50` subset を大きく悪化させた。pct40 は `0.40-0.50` support を戻すことで、late-range prior の効果を残しながら early-range exception と `pf_z` regression を緩められる可能性があった。

## 実装

`hard_window_public_replay.py` は raw typewell 読み込み直後に、元の finite TVT min/max から `typewell_pct` を計算し、`0.40 <= typewell_pct <= 1.00` の行だけを PF_ANCC、PF_Z、Beam、128-seed likelihood-PF に渡す。

soft prior、`0.30/0.60/0.70` grid、LightGBM 学習、inference、submit はこの実験では実行していない。

## 検証方針

raw train horizontal/typewell files から exp072-style full replay train feature cache を再生成した。既存 exp072 cache と exp192 pct50 cache は生成 input に使わず、生成後に同一 row で direct PF/Beam RMSE/MAE/within10、distance bucket、true typewell pct bucket、by-well regression を比較した。

## 所見

Kaggle train v1 は 3,783,989 rows / 773 wells / 196 features を生成し、runtime は 8,616.007 sec、feature generation は 7,497.889 sec。raw gzip SHA は `7b1f51b1c4de16bbff59c9a0c1bd015fc3b6d6152c32a3a93b2f5a694a37576b`、decompressed SHA は `106cdfb266f93a0e45f25b281d3238c1fab0a24a84dac4c23187044022b5127e`。

exp072 比では `likpf_mean` RMSE 11.594898 -> 11.576062、`pf_ancc` 14.493061 -> 14.020904、`beam_mean` 15.774328 -> 15.711042 と改善したが、`pf_z` は 17.788174 -> 18.834133 と悪化した。pct50 比では `pf_z` が -0.870979 RMSE 改善し、true typewell pct `<0.50` の破綻も大きく回復した一方、global `likpf_mean` は +0.031251 RMSE 悪化した。

## 生成物

- `artifacts/exp196_typewell_late_range_hard_window_pct40_full_cache_replacement_full_replay_cache_pixiux_likpf_hard_window_pct40_public_replay_train_features.csv.gz`
- `artifacts/exp196_typewell_late_range_hard_window_pct40_full_cache_replacement_full_replay_cache_feature_schema.csv`
- `artifacts/exp196_typewell_late_range_hard_window_pct40_full_cache_replacement_full_replay_cache_summary.json`
- `artifacts/exp196_direct_pfbeam_comparison_summary.json`
- `artifacts/exp196_vs_exp072_overall_metrics.csv`
- `artifacts/exp196_vs_exp192_pct50_overall_metrics.csv`

## Files

- 学習 notebook: `exp196_typewell_late_range_hard_window_pct40_full_cache_replacement_train.ipynb`
- 推論 notebook: `exp196_typewell_late_range_hard_window_pct40_full_cache_replacement_inference.ipynb`
- 実装: `hard_window_public_replay.py`, `feature_cache.py`
