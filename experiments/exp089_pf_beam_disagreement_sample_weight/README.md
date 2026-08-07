# exp089_pf_beam_disagreement_sample_weight

## Status

Kaggle train v1 completed. No submit candidate.

## Hypothesis

PF/Beam direct replacement is worse overall, but exp083/087 show that low PF/Beam/likelihood-PF disagreement identifies stable rows and high disagreement identifies many high-error rows. Instead of routing to PF/Beam, this experiment absorbs those signals as target-free confidence features and a conservative sample-weight policy on top of the exp073 deterministic ML surface.

## Validation Strategy

Use the exp072 deterministic full replay train feature cache and exp073 LightGBM config family. Keep the target as `TVT - last_known_tvt`, group by well, and compare unweighted RMSE for control, feature-only, weight-only, and feature+weight variants.

## Scope

- Route: `ml_model`
- Parent: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- Cache parent: `exp072_exp063_full_replay_feature_cache`
- Base features: exp073 full replay 196 features
- Added features: `pf_likpf_abs`, `pf_beam_abs`, `beam_likpf_abs`
- Sample weight: rank-average instability from PF/Beam / likelihood-PF / density disagreement, normalized to mean 1.0
- Inference: train-side ablation result reviewまで未選択

## Variants

- `control_exp073_base196`
- `confidence_features_core`
- `sample_weight_unstable_downweight`
- `confidence_features_plus_weight`

## Findings

Best `lgb_mean` variant is `sample_weight_unstable_downweight` with RMSE `9.521212047`, improving the exp073 control `9.526374573` by `-0.005162526`.

Confidence feature add-only and feature+weight variants worsened. The sample-weight-only improvement is not robust enough for submission: improved wells 374, worsened wells 399, max well worsen +1.096752, and mid-distance buckets worsened.

## Expected Outputs

- `exp089_pf_beam_disagreement_sample_weight_metrics.csv`
- `exp089_pf_beam_disagreement_sample_weight_by_well.csv`
- `exp089_pf_beam_disagreement_sample_weight_bucket_metrics.csv`
- `exp089_pf_beam_disagreement_sample_weight_confidence_feature_summary.csv`
- `exp089_pf_beam_disagreement_sample_weight_sample_weight_summary.csv`
- `exp089_pf_beam_disagreement_sample_weight_feature_importance.csv`
- `exp089_pf_beam_disagreement_sample_weight_feature_importance_mean.csv`
- `exp089_pf_beam_disagreement_sample_weight_feature_importance_mean_top.png`
- `exp089_pf_beam_disagreement_sample_weight_predictions.csv.gz`
- `exp089_pf_beam_disagreement_sample_weight_feature_schema.csv`
- `exp089_pf_beam_disagreement_sample_weight_lgb_models/manifest.json`
- `exp089_pf_beam_disagreement_sample_weight_summary.json`
