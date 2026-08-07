# exp196_typewell_late_range_hard_window_pct40_full_cache_replacement 結果

## 状態

Kaggle train v1 完了。pct40 hard-window full replay cache は感度実験として支持。ただし direct PF/Beam inference / submit には進めず、使う場合は downstream ML の replacement-only 比較で pct50 と並べて評価する。

## 実験意図

exp192 pct50 hard-window full replay cache replacement は `likpf_mean` と `pf_ancc` を改善したが、`pf_z` と true typewell pct `<0.50` subset を大きく悪化させた。pct40 へ緩め、`0.40-0.50` support を戻すことで、pct50 の global 改善をどの程度維持しながら regression を緩和できるかを確認した。

## 実行結果

- Kaggle kernel: `kentookumura/exp196-typewell-hard-window-pct40-train` v1
- URL: https://www.kaggle.com/code/kentookumura/exp196-typewell-hard-window-pct40-train
- status: `COMPLETE`
- metadata `id_no`: 125944394
- runtime: CPU (`enable_gpu=false`, `enable_internet=false`)
- rows / wells / features: 3,783,989 / 773 / 196
- runtime: 8,616.007 sec
- feature generation: 7,497.889 sec
- raw gzip SHA: `7b1f51b1c4de16bbff59c9a0c1bd015fc3b6d6152c32a3a93b2f5a694a37576b`
- decompressed SHA: `106cdfb266f93a0e45f25b281d3238c1fab0a24a84dac4c23187044022b5127e`
- schema SHA: `b1946bee3db4cec5311f1ffd4a47a4e1db0f0635e153b3882871b7c34ef5e9e5`

exp072 / exp192 pct50 との同一 row 比較は `rows_checked=3,783,989`、`unique_wells=773`、`id_mismatches=0`、`missing_typewell_range_wells=0`。

### vs exp072

| candidate | exp072 RMSE | exp196 RMSE | delta RMSE | exp196 MAE | delta within10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pf_ancc` | 14.493061 | 14.020904 | -0.472157 | 8.382114 | +0.027218 |
| `pf_z` | 17.788174 | 18.834133 | +1.045959 | 11.255989 | -0.008121 |
| `beam_mean` | 15.774328 | 15.711042 | -0.063285 | 10.883766 | +0.000035 |
| `beam_sm5` | 16.313542 | 16.185190 | -0.128352 | 11.272562 | +0.000050 |
| `likpf_mean` | 11.594898 | 11.576062 | -0.018835 | 7.062574 | +0.001023 |

### vs exp192 pct50

| candidate | exp192 RMSE | exp196 RMSE | delta RMSE | exp196 MAE | delta within10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pf_ancc` | 13.821178 | 14.020904 | +0.199726 | 8.382114 | -0.015511 |
| `pf_z` | 19.705112 | 18.834133 | -0.870979 | 11.255989 | -0.001915 |
| `beam_mean` | 15.677016 | 15.711042 | +0.034026 | 10.883766 | +0.002141 |
| `beam_sm5` | 16.152930 | 16.185190 | +0.032260 | 11.272562 | +0.002191 |
| `likpf_mean` | 11.544812 | 11.576062 | +0.031251 | 7.062574 | -0.000363 |

## 所見

pct40 は pct50 より global 改善幅は小さい。最重要の `likpf_mean` は exp072 比 -0.018835 RMSE に留まり、pct50 の -0.050086 には届かない。`pf_ancc`、`beam_mean`、`beam_sm5` も同様に pct50 より弱い。一方、`pf_z` の悪化は exp072 比 +1.045959 で、pct50 の +1.916938 よりは明確に緩和した。

true typewell pct `<0.50` subset では pct50 の破綻を大きく回復した。`likpf_mean` は pct50 比 -28.390042 RMSE、`beam_mean` は -33.530093 RMSE、`pf_ancc` は -21.781272 RMSE で、exp072 比でも改善している。ただし `0.50-0.70` bucket では pct50 比で `likpf_mean` +2.380595、`beam_mean` +2.500450、`pf_ancc` +4.023544 RMSE と悪化した。

by-well では exp072 比 `likpf_mean` が 364 wells 改善 / 405 wells 悪化で、最大 regression は +10.213776 RMSE。pct50 より最大 regression は抑えたが、全体としては改善 wells が優勢ではない。

## 判断

実装した backlog は完了。pct40 は pct50 の early-range exception と `pf_z` regression を緩める感度実験として有効だが、direct PF/Beam submit 候補ではない。次に進めるなら、exp148 などの downstream ML replacement-only で pct40 と pct50 を同じ条件で比較し、ML が pct50 の `pf_z` / `<0.50` 破綻を吸収できるか、または pct40 の安定性が OOF に効くかを確認する。

## 生成物

- `artifacts/exp196_typewell_late_range_hard_window_pct40_full_cache_replacement_full_replay_cache_pixiux_likpf_hard_window_pct40_public_replay_train_features.csv.gz`
- `artifacts/exp196_typewell_late_range_hard_window_pct40_full_cache_replacement_full_replay_cache_feature_schema.csv`
- `artifacts/exp196_typewell_late_range_hard_window_pct40_full_cache_replacement_full_replay_cache_summary.json`
- `artifacts/exp196_kaggle_train_v1.log`
- `artifacts/exp196_vs_exp072_overall_metrics.csv`
- `artifacts/exp196_vs_exp072_distance_bucket_metrics.csv`
- `artifacts/exp196_vs_exp072_true_typewell_pct_metrics.csv`
- `artifacts/exp196_vs_exp072_by_well_delta.csv`
- `artifacts/exp196_vs_exp192_pct50_overall_metrics.csv`
- `artifacts/exp196_vs_exp192_pct50_distance_bucket_metrics.csv`
- `artifacts/exp196_vs_exp192_pct50_true_typewell_pct_metrics.csv`
- `artifacts/exp196_vs_exp192_pct50_by_well_delta.csv`
- `artifacts/exp196_direct_pfbeam_comparison_summary.json`
