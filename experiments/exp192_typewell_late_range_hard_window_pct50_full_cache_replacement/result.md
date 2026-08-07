# exp192_typewell_late_range_hard_window_pct50_full_cache_replacement 結果

## 状態

Kaggle train v1 完了。direct PF/Beam cache としては支持。ただし inference / submit は未実行で、full replacement を提出へ直行しない。

## 実験意図

exp186 の soft prior full replay cache は `pf_ancc` / `pf_z` / `beam_mean` を小幅改善した一方、最重要候補 `likpf_mean` を大きく悪化させた。ユーザー指定により、soft penalty ではなく typewell support 自体を `typewell_pct >= 0.50` に切る hard-window full cache replacement を直接検証した。

## 実行結果

- Kaggle kernel: `kentookumura/exp192-typewell-hard-window-pct50-train` v1
- status: `COMPLETE`
- rows / wells / features: 3,783,989 / 773 / 196
- runtime: 13,275.591 sec
- feature generation: 11,643.326 sec
- raw gzip SHA: `1040d7d3b9254b5a36d2a3f7fd526ae28e3ddd5b29059926b44bbe9d84436e6a`
- decompressed SHA: `a86dff450b108e4481208a5f5699f8624eaf736cb6eb6aa735d39b4044c6f0e1`
- schema SHA: `ad59c4f998433bdd9105effa081ec620fd7f6ced5f9dc32d68b97d4c757f6ed0`

exp072 full replay cache との同一 row 比較は `rows_checked=3,783,989`、`unique_wells=773`、`id_mismatches=0`、`missing_typewell_range_wells=0`。

| candidate | exp072 RMSE | exp192 RMSE | delta RMSE | exp072 MAE | exp192 MAE | delta within10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pf_ancc` | 14.493061 | 13.821178 | -0.671884 | 8.921569 | 8.053430 | +0.042730 |
| `pf_z` | 17.788174 | 19.705112 | +1.916938 | 10.677493 | 11.543506 | -0.006206 |
| `beam_mean` | 15.774328 | 15.677016 | -0.097311 | 10.898586 | 10.926740 | -0.002106 |
| `beam_sm5` | 16.313542 | 16.152930 | -0.160612 | 11.300928 | 11.317035 | -0.002140 |
| `likpf_mean` | 11.594898 | 11.544812 | -0.050086 | 7.067633 | 7.063503 | +0.001386 |

## 所見

hard-window pct50 は exp186 soft prior と違い、最重要候補 `likpf_mean` を壊さず、global RMSE を -0.050086 改善した。`pf_ancc` は -0.671884 と強く改善し、`beam_mean` / `beam_sm5` も RMSE は小改善した。

一方で `pf_z` は +1.916938 RMSE と大きく悪化した。by-well でも `likpf_mean` は 396 wells 改善 / 375 wells 悪化で拮抗し、最大 regression は +30.263337 RMSE。`pf_ancc` も global では強いが最大 regression +49.512070 RMSE がある。さらに true typewell pct `lt_0p50` の 10,500 rows では `likpf_mean` が +25.754210 RMSE、`beam_mean` が +32.502607 RMSE と大きく悪化し、typewell 前半に正解 mode がある例外を壊す。

したがって、この cache は direct replacement 候補としては exp186 より有望だが、PF/Beam route の提出候補ではない。次に進める場合は、exp073 / exp148 系 downstream feature consumer の replacement-only 学習で、`pf_z` 悪化と early-range exception を ML が吸収できるかを確認する。global 小改善だけで inference port / submit へ進めない。

## 生成物

- `artifacts/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement_full_replay_cache_pixiux_likpf_hard_window_pct50_public_replay_train_features.csv.gz`
- `artifacts/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement_full_replay_cache_feature_schema.csv`
- `artifacts/exp192_typewell_late_range_hard_window_pct50_full_cache_replacement_full_replay_cache_summary.json`
- `artifacts/exp192_vs_exp072_overall_metrics.csv`
- `artifacts/exp192_vs_exp072_distance_bucket_metrics.csv`
- `artifacts/exp192_vs_exp072_true_typewell_pct_metrics.csv`
- `artifacts/exp192_vs_exp072_by_well_delta.csv`
- `artifacts/exp192_vs_exp072_summary.json`
