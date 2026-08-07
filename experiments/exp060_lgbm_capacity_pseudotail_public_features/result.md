# exp060_lgbm_capacity_pseudotail_public_features 結果

## Summary

Status: completed.

Kaggle train version 1 completed on
`kentookumura/exp060-lgbm-public-features-train`. The run used the exp056
multi-cutoff public sel15 PF/Beam artifact: 5,499,624 rows and 773 wells.

## Metrics

| Candidate | Original-fold RMSE | Well-hash RMSE | Delta vs 0.65 geometry original | Delta vs 0.65 geometry well-hash |
| --- | ---: | ---: | ---: | ---: |
| `pf090_hold010` | 15.023697 | 15.023697 | -3.877691 | -3.910691 |
| `public_pf_selector` | 15.120839 | 15.120839 | -3.780549 | -3.813550 |
| `lgbm_capacity_public_core_spatial_multicutoff_raw` | 15.562057 | 15.731138 | -3.339331 | -3.203251 |
| `lgbm_capacity_public_pf_core_multicutoff_equal_budget_raw` | 15.644047 | 15.776224 | -3.257341 | -3.158165 |
| `lgbm_capacity_public_pf_core_cutoff065_raw` | 15.772781 | 15.670501 | -3.128607 | -3.263888 |

## Interpretation

Public notebook derived features are useful model features for the ML route.
The spatial + NCC/GR + PF context candidate is the best model candidate. It
improves over the 0.65-only geometry ML control by -3.339331 / -3.203251, and
over the NCC/GR + PF context paired control by -0.221582 / -0.064659.

`pf090_hold010` and `public_pf_selector` are direct PF diagnostic controls on
the exp056 pseudo-test surface, not the adoption criterion for this ML-route
experiment. They remain useful as ceiling/context checks, but they should not be
used to reject an add-only ML feature candidate.

## Decision

Treat `lgbm_capacity_public_core_spatial_multicutoff_raw` as a positive
train-side ML-route candidate. Do not claim it updates the normal CV or Public
LB anchor yet, because this exp056 pseudo-test surface is not the same
evaluation as exp051 CV 12.634392, exp052 Public LB 12.076, exp054 Public LB
11.856, or exp039 Public LB 11.740. The next supported step is an inference
port / hidden-branch audit in the same experiment if we want LB evidence.

## Inference

Kaggle inference version 1 completed on
`kentookumura/exp060-lgbm-public-features-inference`.

- output: `/tmp/kaggle-output/exp060_lgbm_capacity_pseudotail_public_features/inference_v1`
- submission rows: 14,151
- submission SHA256: `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`
- submit-check: PASS against `data/raw/sample_submission.csv`
- prediction range: 11587.038593 to 12240.016066
- branch counts: `physical_visible` 14,151
- changed rows: 0

The inference notebook ran successfully and generated a valid `submission.csv`,
but the public sample contains only the three visible train wells. Therefore all
rows used the physical-visible branch and the hidden
`lgbm_capacity_public_core_spatial_multicutoff_raw_hidden` correction branch did
not change this public output. This confirms package/runtime compatibility and
submission format, but it does not provide Public LB evidence for the hidden ML
branch unless submitted as a code competition run.

## Code Submission

The code submission completed as `ref=53581051` with Public LB 12.046.

- Public LB: 12.046
- delta vs exp052 Public LB 12.076: -0.030
- delta vs exp054 Public LB 11.856: +0.190
- delta vs exp061 Public LB 11.826: +0.220
- delta vs exp039 ML route Public LB 11.740: +0.306
- delta vs exp027 overall/PF route Public LB 8.781: +3.265

This is a small improvement over exp052 but worse than the current pseudo-tail
self-route anchor exp061 and the broader ML-route anchor exp039. Do not update
route anchors from exp060.
