# exp080_u_space_target_ablation Result

## Status

Kaggle train v1 timed out, but target/model/fold metrics were recovered from logs.

## Summary

This experiment compares U-space target definitions while keeping the exp073 deterministic full replay feature surface, folds, and LightGBM config family fixed.

## Result

The run did not finish all targets before Kaggle timeout, so no official pooled CV / prediction SHA was written. However, logs contain fold RMSE for 56 completed target/model/fold jobs.

Log-derived fold RMSE means:

| target | best completed model | mean fold RMSE | interpretation |
| --- | --- | ---: | --- |
| `dTVT` | `lgb1` | 9.534549 | best; keep baseline target |
| `dTVT_plus_dZ` | `lgb0` | 12.058333 | worse |
| `TVT_plus_Z_minus_T0` | `lgb0` | 18.893572 | much worse |
| `TVT_plus_Z_abs` | `lgb1` | 52.900531 | broken |
| `TVT_plus_Z_minus_T0Z0` | - | - | not reached before timeout |

These are simple means of fold RMSE, not pooled RMSE. The margin is large enough to reject the tested U-space target variants.

Artifacts:

- `artifacts/exp080_train_v1_log_fold_metrics.csv`
- `artifacts/exp080_train_v1_log_target_model_summary.csv`

## Next

Do not port a U-space target to inference from this experiment. Keep `dTVT` as the supervised target. If U-space is revisited, prefer add-only projection/disagreement features or postprocess rather than changing the core target.
