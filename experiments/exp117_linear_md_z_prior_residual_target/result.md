# exp117_linear_md_z_prior_residual_target 結果

## Status

Kaggle train v1 完了。linear MD/Z prior residual target は不採用。

## Summary

exp072 の deterministic full replay feature cache と exp073 の LightGBM 設定を固定し、supervised target だけを比較した。control は `dTVT = TVT - T0`。比較対象は `prior = T0 + a*dMD + b*dZ` に対する residual target。

## Result

Kaggle train v1 は 3,783,989 rows / 773 wells / 196 features、`gpu_repro_guard_dp_threads8`、`lgb0` の 5-fold GroupKFold で完了した。

| target | pooled RMSE | verdict |
| --- | ---: | --- |
| `dTVT` | 9.664291 | best / keep |
| `linear_prior_a0p02_bm0p25` | 11.061642 | reject |
| `linear_prior_a0p02_bm0p50` | 12.515352 | reject |
| `linear_prior_a0p04_bm0p25` | 11.079209 | reject |

Fold RMSE:

| target | fold0 | fold1 | fold2 | fold3 | fold4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dTVT` | 8.875566 | 10.178201 | 8.538341 | 10.440596 | 10.134981 |
| `linear_prior_a0p02_bm0p25` | 10.537594 | 11.318329 | 10.323318 | 11.680138 | 11.386884 |
| `linear_prior_a0p02_bm0p50` | 12.404166 | 13.078191 | 12.145430 | 12.787320 | 12.133872 |
| `linear_prior_a0p04_bm0p25` | 10.396035 | 11.251801 | 10.458824 | 11.999382 | 11.211363 |

Distance bucket:

| target | 0-50 ft | 50-100 ft | 100-250 ft | 250-500 ft | 500-1000 ft | 1000+ ft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dTVT` | 0.865871 | 1.321617 | 2.272692 | 3.757856 | 5.562122 | 10.594760 |
| `linear_prior_a0p02_bm0p25` | 0.608066 | 1.089810 | 2.102435 | 3.620263 | 5.656289 | 12.190477 |
| `linear_prior_a0p02_bm0p50` | 1.028961 | 1.730610 | 2.719795 | 4.015090 | 6.007545 | 13.813262 |
| `linear_prior_a0p04_bm0p25` | 0.605215 | 1.155673 | 2.233174 | 3.679816 | 5.687558 | 12.206532 |

Weak linear prior residuals improve some near-prefix buckets, but the 1000+ ft bucket regresses strongly. Well-level behavior is also unsafe:

| target | improved wells | worse wells | max regression RMSE |
| --- | ---: | ---: | ---: |
| `linear_prior_a0p02_bm0p25` | 268 | 505 | 20.655735 |
| `linear_prior_a0p02_bm0p50` | 248 | 525 | 51.263206 |
| `linear_prior_a0p04_bm0p25` | 250 | 523 | 15.136597 |

Anchor diagnostics were clean: 773 wells, `anchor_t0_vs_last_known_abs_max = 0.0`, known-prefix rows 851-2392. The failure is therefore target definition / long-tail behavior, not anchor recovery.

## Artifacts

Small artifacts synced to repo:

- `artifacts/exp117_linear_md_z_prior_residual_target_metrics.csv`
- `artifacts/exp117_linear_md_z_prior_residual_target_by_well.csv`
- `artifacts/exp117_linear_md_z_prior_residual_target_bucket_metrics.csv`
- `artifacts/exp117_linear_md_z_prior_residual_target_target_summary.csv`
- `artifacts/exp117_linear_md_z_prior_residual_target_feature_schema.csv`
- `artifacts/exp117_linear_md_z_prior_residual_target_lgb_models/manifest.json`
- `artifacts/exp117_linear_md_z_prior_residual_target_summary.json`

Large outputs are kept outside the repo:

- `/tmp/kaggle-output/exp117_linear_md_z_prior_residual_target/train_v1/artifacts/exp117_linear_md_z_prior_residual_target_predictions.csv.gz`
- `/tmp/kaggle-output/exp117_linear_md_z_prior_residual_target/train_v1/artifacts/exp117_linear_md_z_prior_residual_target_lgb_models/`

## Next

Do not port linear MD/Z prior residual targets to inference. Keep `dTVT` as the supervised target. If this signal is reused, use it only as a weak distance-aware feature, confidence diagnostic, or gated near-prefix correction, not as a global target transform.
