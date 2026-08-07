# exp095_prefix_u_line_residual_target Result

## Status

Kaggle train v1 completed.

## Summary

This experiment compares prefix de-trended U-space residual targets while keeping the exp073 deterministic full replay feature surface, folds, and LightGBM `lgb0` config fixed.

## Result

Kaggle train v1 compared `dTVT` control against prefix de-trended U-space residual targets with the exp073 deterministic full replay feature surface and only `lgb0`.

| target | pooled RMSE | fold RMSE range | interpretation |
| --- | ---: | ---: | --- |
| `dTVT` | 9.664067 | 8.538341 - 10.440596 | best; keep baseline target |
| `prefix_u_line_alpha0p5` | 28.087914 | 22.001913 - 33.539633 | much worse |
| `prefix_u_line_alpha1p0` | 33.478794 | 27.867127 - 37.855243 | broken |

Prefix-line diagnostics were clean mechanically: 773 wells, no prefix-line fallbacks, `T0` matched `last_known_tvt` exactly, and known-prefix row counts were 851-2392. The degradation is therefore not a short-prefix fallback issue. The prefix U-line residual target itself is poorly conditioned for this LightGBM setup.

Distance buckets show the failure is broad, including near rows:

| target | 0-50 ft RMSE | 500-1000 ft RMSE | 1000+ ft RMSE |
| --- | ---: | ---: | ---: |
| `dTVT` | 0.865586 | 5.561822 | 10.594535 |
| `prefix_u_line_alpha0p5` | 6.250421 | 10.919722 | 31.119892 |
| `prefix_u_line_alpha1p0` | 5.909423 | 13.157698 | 37.116570 |

Artifacts:

- `artifacts/exp095_prefix_u_line_residual_target_metrics.csv`
- `artifacts/exp095_prefix_u_line_residual_target_by_well.csv`
- `artifacts/exp095_prefix_u_line_residual_target_bucket_metrics.csv`
- `artifacts/exp095_prefix_u_line_residual_target_target_summary.csv`
- `artifacts/exp095_prefix_u_line_residual_target_feature_schema.csv`
- `artifacts/exp095_prefix_u_line_residual_target_lgb_models/manifest.json`
- `artifacts/exp095_prefix_u_line_residual_target_summary.json`

Large outputs are kept outside the repo:

- `/tmp/kaggle-output/exp095_prefix_u_line_residual_target/train_v1/artifacts/exp095_prefix_u_line_residual_target_predictions.csv.gz`
- `/tmp/kaggle-output/exp095_prefix_u_line_residual_target/train_v1/artifacts/exp095_prefix_u_line_residual_target_lgb_models/`

## Next

Do not port a prefix U-line residual target to inference. Keep `dTVT` as the supervised target. Future U-space work should stay target-free, such as projection postprocess / correction features with explicit near-prefix guards.
