# exp057_xgb_catboost_pf_confidence_only_features result

## Summary

Status: completed.

Kaggle train version 1 completed on
`kentookumura/exp057-xgb-catboost-pf-confidence-train`. The run used 1,782,279
rows and 773 wells from the exp029 cutoff-0.65 public sel15 PF/Beam feature
artifact.

## Metrics

| Candidate | Original-fold RMSE | Well-hash RMSE | Paired control delta original | Paired control delta well-hash |
| --- | ---: | ---: | ---: | ---: |
| `pf090_hold010` | 15.089532 | 15.089532 | - | - |
| `public_pf_selector` | 15.172636 | 15.172636 | - | - |
| `catboost_pf_confidence_only_raw` | 15.609958 | 15.440706 | -2.898545 | -3.062035 |
| `catboost_pf_confidence_only_bucket_shrink` | 15.645771 | 15.451603 | -3.053753 | -3.259695 |
| `xgb_pf_confidence_only_raw` | 15.836064 | 15.662857 | -2.950842 | -3.466265 |
| `xgb_pf_confidence_only_bucket_shrink` | 15.958626 | 15.777023 | -3.090676 | -3.674658 |
| `catboost_geometry_control_raw` | 18.508503 | 18.502741 | - | - |
| `catboost_geometry_control_bucket_shrink` | 18.699524 | 18.711298 | - | - |
| `xgb_geometry_control_raw` | 18.786906 | 19.129122 | - | - |
| `xgb_geometry_control_bucket_shrink` | 19.049302 | 19.451681 | - | - |

## Interpretation

PF/Beam confidence-only features strongly improved both XGBoost and CatBoost over
their paired geometry controls. The selected paired-control candidate is
`catboost_pf_confidence_only_raw` with original-fold RMSE 15.609958 and
well-hash RMSE 15.440706.

This is still weaker than direct public PF controls on the same exp029
pseudo-test surface: `pf090_hold010` scored 15.089532 and `public_pf_selector`
scored 15.172636 on both audits. The run also used only the available cutoff
0.65 artifact; requested cutoffs 0.45 and 0.82 were missing and recorded as
missing in `pf_confidence_train_summary.csv`.

## Decision

Do not inference-port or submit directly from this run. Treat the result as
evidence that confidence-only diagnostics are useful model features, but not a
direct replacement for public PF controls. Next useful steps are to test the same
confidence-only family inside the stronger exp051/052 LGBM capacity pseudo-tail
surface, or use PF/Beam disagreement for sample weighting / clipping rather than
direct hidden-branch replacement.
