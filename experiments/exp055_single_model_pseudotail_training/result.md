# exp055_single_model_pseudotail_training result

## Summary

Status: completed, no supported pseudo-tail training candidate.

Kaggle train version 1 completed on `kentookumura/exp055-single-model-pseudotail-train`.
The audit used 1,782,279 rows and 773 wells from the exp029 cutoff-0.65 public sel15 PF/Beam feature artifact.

## Metrics

| Candidate | Original-fold RMSE | Well-hash RMSE |
| --- | ---: | ---: |
| `pf090_hold010` | 15.089532 | 15.089532 |
| `public_pf_selector` | 15.172636 | 15.172636 |
| `exp039_same_surface_control_raw` | 15.722062 | 15.667445 |
| `single_model_pseudotail_training_raw` | 15.764607 | 15.959310 |
| `exp039_same_surface_control_bucket_shrink` | 15.875275 | 15.837223 |
| `single_model_pseudotail_training_bucket_shrink` | 15.911149 | 16.159852 |

## Interpretation

The pseudo-tail training policy did not improve the exp039 same-surface single-model control.

- Raw pseudo-tail training was worse than raw control by +0.042545 on original-fold and +0.291865 on well-hash.
- Bucket-shrink pseudo-tail training was worse than bucket-shrink control by +0.035874 on original-fold and +0.322629 on well-hash.
- The requested exp051-style cutoffs `[0.45, 0.65, 0.82]` could not be fully evaluated because the available exp029 artifact contains only cutoff 0.65; cutoffs 0.45 and 0.82 were missing and the run fell back to available cutoff 0.65.
- Direct public PF controls remained stronger than the single-model candidates on this train-side surface.

The Kaggle-generated `single_lgbm_summary.json` lists `exp039_same_surface_control_raw` as selected because the run config only marked the bucket-shrink control as required. For interpretation, that is a control variant, not a new supported candidate. The local config now also marks `exp039_same_surface_control_raw` as a required control for future reruns.

## Decision

Do not inference-port or submit `single_model_pseudotail_training` from this run.

Next useful step is to either generate a true multi-cutoff exp029 public-feature artifact before revisiting this hypothesis, or prioritize adding public/PF confidence features into the stronger exp051/052 pseudo-tail capacity model instead.
