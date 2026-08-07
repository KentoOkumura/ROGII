# Result

## Summary

Kaggle train v1 and inference v1 completed. The user reported that submission was also made manually; ref `53809333` completed with Public LB `8.611`.

`exp077_full_replay_postprocess_guard` audits conservative fixed postprocess policies on top of the deterministic exp073 full replay OOF prediction. It also adds fold/model averaged LightGBM feature importance rendering from exp073 saved boosters.

Best same-OOF policy:

- policy: `longtail_likpf_tiny_gate_w006`
- OOF RMSE: `9.470514771712411`
- exp073 baseline OOF RMSE in this audit: `9.526374749390682`
- delta vs baseline: `-0.055859977678271`
- prediction SHA: `9813e6ba2e008f87c37ca0185fb754e17435a9a70c9a5f559ccd7c9a3dce3d24`

## Current Interpretation

The OOF gain was ported to inference as a fixed policy. The inference output passed submit-check and produced submission SHA `ccf17704959274d9e38f6eb8a7fe3c55a19128a8f24ba1a3d555f6af73bc8538`.

Latest observed submission ref `53809333` completed with Public LB `8.611`. This improves over the exp073 deterministic raw ML anchor Public LB `8.780`, so exp077 updates the ML route submitted/postprocessed anchor. Nearby refs `53807892` and `53807896` completed with Public LB `8.489`, but those are exp075 duplicate submissions and are not used as the ML route anchor.

Keep exp073 as the deterministic raw ML anchor because it is the byte-stable unpostprocessed LightGBM replay baseline. Use exp077 as the submitted/postprocessed ML route anchor for downstream LB comparisons.

## Artifacts

- `artifacts/exp077_full_replay_postprocess_guard_metrics.csv`
- `artifacts/exp077_full_replay_postprocess_guard_bucket_metrics.csv`
- `artifacts/exp077_full_replay_postprocess_guard_predictions.csv.gz`
- `artifacts/exp077_full_replay_postprocess_guard_summary.json`
- `artifacts/exp063_full_replay_repro_guard_feature_importance_by_fold.csv`
- `artifacts/exp063_full_replay_repro_guard_feature_importance_mean.csv`
- `artifacts/exp063_full_replay_repro_guard_feature_importance_mean_top40.png`

Output path:

- `/tmp/kaggle-output/exp077-full-replay-postprocess-guard-train-v1`
