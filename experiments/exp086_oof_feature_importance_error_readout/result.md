# Result

## Summary

Kaggle train v1 completed. This experiment is an OOF diagnostic readout and does not create an inference notebook output or submission.

- kernel: `kentookumura/exp086-oof-feature-importance-error-readout-train`
- output path: `/tmp/kaggle-output/exp086_oof_feature_importance_error_readout/train_v1`
- rows / wells: 3,783,989 / 773
- features read: 34
- elapsed seconds: 355.393

Policy metrics:

- `baseline_exp073_lgb_mean`: RMSE `9.52637482601992`, MAE `6.159766047813564`
- `longtail_likpf_tiny_gate_w006`: RMSE `9.47051480056479`, MAE `6.110920032404956`
- RMSE delta vs baseline: `-0.05586002545513047`

## Current Interpretation

The largest baseline-error lift appears in feature buckets tied to PF/dense-surface disagreement and dense TVT deltas:

- `pf_vs_dense` worst bucket MAE lift `+2.632059`, baseline RMSE `12.783611`
- `tvt_densew_d` worst bucket MAE lift `+2.533809`, baseline RMSE `12.851624`
- `tvt_dense50_d` worst bucket MAE lift `+2.524084`, baseline RMSE `12.812007`
- `tvt_dense_d` worst bucket MAE lift `+2.328562`, baseline RMSE `12.686914`
- `dense_dist` has a relatively strong Spearman absolute-error correlation `0.119421`
- `beam_std_d` has the strongest selected-feature Spearman absolute-error correlation `0.175250`

This supports using disagreement/uncertainty features for confidence or sample weighting. It does not support direct PF/Beam replacement, and it does not update the ML route anchor.

## Artifacts

- `artifacts/exp086_oof_feature_importance_error_readout_policy_metrics.csv`
- `artifacts/exp086_oof_feature_importance_error_readout_feature_summary.csv`
- `artifacts/exp086_oof_feature_importance_error_readout_feature_quantile_metrics.csv`
- `artifacts/exp086_oof_feature_importance_error_readout_well_summary.csv`
- `artifacts/exp086_oof_feature_importance_error_readout_feature_error_lift_top20.png`
- `artifacts/exp086_oof_feature_importance_error_readout_feature_error_correlation_top20.png`
- `artifacts/exp086_oof_feature_importance_error_readout_summary.json`

Output path:

- `/tmp/kaggle-output/exp086_oof_feature_importance_error_readout/train_v1`
