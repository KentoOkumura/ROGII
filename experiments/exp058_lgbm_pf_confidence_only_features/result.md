# exp058_lgbm_pf_confidence_only_features result

## Summary

Status: submitted complete.

Kaggle train version 1 completed on
`kentookumura/exp058-lgbm-pf-confidence-train`. The run used 1,782,279 rows and
773 wells from the exp029 cutoff-0.65 public sel15 PF/Beam feature artifact.

Kaggle inference version 1 completed on
`kentookumura/exp058-lgbm-pf-confidence-inference`. Submit-check passed and code
submission ref `53535327` was created on 2026-06-10 12:26:23.747000. Public LB
was 12.778.

## Metrics

| Candidate | Original-fold RMSE | Well-hash RMSE | Paired control delta original | Paired control delta well-hash |
| --- | ---: | ---: | ---: | ---: |
| `pf090_hold010` | 15.089532 | 15.089532 | - | - |
| `public_pf_selector` | 15.172636 | 15.172636 | - | - |
| `lgbm_capacity_pf_confidence_only_raw` | 15.945678 | 15.569216 | -2.701498 | -3.272481 |
| `lgbm_capacity_pf_confidence_only_bucket_shrink` | 16.071589 | 15.671243 | -2.806396 | -3.447887 |
| `beam` | 18.122632 | 18.122632 | - | - |
| `last_anchor` | 18.284054 | 18.284054 | - | - |
| `lgbm_capacity_geometry_control_raw` | 18.647176 | 18.841697 | - | - |
| `lgbm_capacity_geometry_control_bucket_shrink` | 18.877985 | 19.119130 | - | - |

## Interpretation

PF/Beam confidence-only features strongly improved the LightGBM capacity model
over its paired geometry control. The selected paired-control candidate is
`lgbm_capacity_pf_confidence_only_raw` with original-fold RMSE 15.945678 and
well-hash RMSE 15.569216.

For the ML route, the relevant paired comparison is positive: the confidence
variant beat the same LightGBM geometry control on both holdouts. Direct public
PF controls are reported only as diagnostic ceiling values on this exp029
pseudo-test surface, not as the primary route adoption criterion. The run also
used only the available cutoff 0.65 artifact; requested cutoffs 0.45 and 0.82
were missing and recorded as missing in `pf_confidence_train_summary.csv`.

## Decision

The raw candidate was inference-ported after user request. The public sample has
only visible train wells, so the local `submission.csv` is unchanged from the
public replay SHA `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`
with `changed_rows=0`; hidden code execution still uses the
`lgbm_capacity_pf_confidence_only_raw_hidden` branch.

The submitted hidden branch is not adopted. Public LB 12.778 is worse than
exp052 12.076 by +0.702 and worse than exp054 11.856 by +0.922.

The next submit-oriented experiment should not repeat the same surface as-is.
It should add fold-safe PF-vs-exp052/054 disagreement features and fit
bucket-shrink alphas out-of-fold on the exp058 surface instead of reusing the
exp014 shrink parameters.
