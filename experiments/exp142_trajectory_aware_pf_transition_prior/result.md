# exp142_trajectory_aware_pf_transition_prior 結果

## 結論

Kaggle train v3 は完了したが、trajectory-aware PF transition prior は不採用。best trajectory-aware candidate は `traj_pf_zmean_s006_noise025_scale_3` で RMSE 23.132450 / MAE 12.927040 / within10 0.614683。`exp072_likpf_mean` 11.594898 から +11.537553、`exp072_pf_z` 17.788171 からも +5.344279 悪化した。

直接の inference port / submit はしない。`dZ/dMD` と `d2Z/dMD2` を PF transition mean / noise に強く入れる設計は、near-prefix では良いが longtail で path を壊す。

## 主な数値

| candidate | RMSE | MAE | within10 | bias |
| --- | ---: | ---: | ---: | ---: |
| `exp072_likpf_mean` | 11.594898 | 7.067633 | 0.772807 | -1.099423 |
| `exp072_pf_ancc` | 14.493051 | 8.921559 | 0.691741 | -1.167419 |
| `exp072_beam_mean` | 15.774328 | 10.898586 | 0.591646 | -1.499448 |
| `pf_z_ms_scale_8` | 16.871842 | 9.708059 | 0.693144 | -0.939723 |
| `exp072_pf_z` / parity | 17.788171 | 10.677487 | 0.647668 | -0.934560 |
| `traj_pf_zmean_s006_noise025_scale_3` | 23.132450 | 12.927040 | 0.614683 | -2.135705 |
| `traj_pf_zacc_s010_a020_noise050_scale_3` | 24.564859 | 14.156565 | 0.576947 | -2.471684 |
| `traj_pf_zacc_s014_a035_noise075_scale_3` | 24.979228 | 14.499017 | 0.570488 | -2.519479 |

strict parity は `max_abs_diff=0.0` / `rmse_diff=0.0` で pass。32-seed strict multiseed は exp106 v3 の 64-seed結果より弱く、best が `pf_z_ms_scale_8` RMSE 16.871842 だった。

## Bucket 所見

near-prefix は改善した。`traj_pf_zmean_s006_noise025_scale_3` は `md_since` 0-50 ft で RMSE 0.550729、50-100 ft で 1.266820、100-250 ft で 2.629969 と、`likpf_mean` の 1.188878 / 1.925625 / 2.934160 より良い。

一方、global を支える 1000+ ft longtail で崩壊した。`likpf_mean` の `1000_plus` RMSE 12.704015 / within10 0.733571 に対し、best trajectory-aware は RMSE 25.710669 / within10 0.540462。`abs_dzdmd` 0.1-0.25 bucket でも `likpf_mean` 19.984848 に対し 47.842545 まで悪化した。

## PF Diagnostics

| variant | mean_neff_frac | min_neff_frac | mean_resample_count | mean_collapse_rate | mean_particle_std |
| --- | ---: | ---: | ---: | ---: | ---: |
| `zmean_s006_noise025` | 0.732249 | 0.003540 | 249.903 | 0.110213 | 0.680116 |
| `zacc_s010_a020_noise050` | 0.697873 | 0.001667 | 517.725 | 0.120437 | 0.627684 |
| `zacc_s014_a035_noise075` | 0.681301 | 0.001667 | 637.089 | 0.136930 | 0.593857 |

trajectory strength を上げるほど likelihood と path quality が悪化した。collapse rate だけが主因ではなく、transition prior の target velocity 自体が longtail で誤誘導している可能性が高い。

## 生成物

- output: `experiments/exp142_trajectory_aware_pf_transition_prior/kaggle/output/train_v3`
- summary: `artifacts/exp142_trajectory_aware_pf_transition_prior_summary.json`
- metrics: `artifacts/exp142_trajectory_aware_pf_transition_prior_candidate_metrics.csv`
- bucket metrics: `artifacts/exp142_trajectory_aware_pf_transition_prior_bucket_metrics.csv`
- by-well: `artifacts/exp142_trajectory_aware_pf_transition_prior_by_well.csv`
- quality: `artifacts/exp142_trajectory_aware_pf_transition_prior_trajectory_pf_z_quality.csv`

`candidate_wide.csv.gz` は Kaggle 側では生成済みで summary に SHA が記録されているが、ローカル full output download は巨大ファイルで途中停止したため完全取得していない。

## 次のアクション

`trajectory_aware_pf_transition_prior` は完了/不採用として閉じる。Z trajectory は PF transition の直接置換ではなく、near-prefix guard、confidence feature、または `exp143_multimode_pfbeam_local_correlation_audit` のような mode diversity / local correlation 診断に下げて使う。
