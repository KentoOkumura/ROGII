# exp077_full_replay_postprocess_guard

## Status

Kaggle train v1 and inference v1 completed. Submission ref `53809333` completed with Public LB `8.611`; this updates the ML route submitted/postprocessed anchor. Keep exp073 as the deterministic raw ML anchor.

## Hypothesis

The deterministic exp073 full replay ML anchor may have small OOF headroom from conservative postprocess guards, especially residual clipping, tail-start fade, flat-prefix shrink, and tiny PF-vs-ML disagreement gates. The policy must be fixed and small because same-OOF selection can overfit.

## Scope

- Reuse exp073 OOF predictions for `gpu_repro_guard_dp_threads8 / lgb_mean`.
- Join exp072 full replay train features by `id` when available.
- Compare fixed postprocess policies and save global and distance-bucket metrics.
- Recover exp073 saved booster feature importance by fold/model, average it, and render a matplotlib plot.

## Validation Strategy

The primary audit is OOF-only: compare each fixed postprocess policy against the exp073 OOF baseline with global RMSE and distance-bucket metrics. Treat any same-OOF improvement as diagnostic until the policy is fixed, ported to inference, and checked with submit-check / LB.

## Findings

Kaggle train v1 completed. Best same-OOF policy is `longtail_likpf_tiny_gate_w006` with RMSE `9.470514771712411`, improving over the exp073 baseline `9.526374749390682` by `-0.055859977678271`.

Kaggle inference v1 completed with the fixed policy and submit-check passed. The inference regenerated raw test PF/Beam/likelihood-PF replay features on Kaggle for 3 wells / 14,151 rows. Latest observed submission ref `53809333` scored Public LB `8.611`, improving over exp073 Public LB `8.780`. Therefore exp077 is the ML route submitted/postprocessed anchor, while exp073 remains the deterministic raw ML anchor.

## Expected Outputs

- `exp077_full_replay_postprocess_guard_metrics.csv`
- `exp077_full_replay_postprocess_guard_bucket_metrics.csv`
- `exp077_full_replay_postprocess_guard_predictions.csv.gz`
- `exp077_full_replay_postprocess_guard_summary.json`
- `exp063_full_replay_repro_guard_feature_importance_by_fold.csv`
- `exp063_full_replay_repro_guard_feature_importance_mean.csv`
- `exp063_full_replay_repro_guard_feature_importance_mean_top40.png`
