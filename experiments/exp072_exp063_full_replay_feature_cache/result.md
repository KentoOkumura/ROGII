# exp072_exp063_full_replay_feature_cache Result

## Status

Completed on Kaggle CPU runtime. Latest completed version is deterministic v2.

## Evaluation

No model training is performed in this experiment. The success criterion is that the generated train feature cache has 196 features for `pixiux_likpf_public_replay`, matching exp063's full public replay feature surface.

Kaggle train v2 result:

| item | value |
| --- | ---: |
| rows | 3,783,989 |
| wells | 773 |
| feature count | 196 |
| elapsed seconds | 17,728.972 |
| feature generation seconds | 15,380.262 |

Train feature cache SHA256:

`14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`

Previous v1 stochastic train feature cache SHA256:

`86d4777ddf44134cc8e1c7ce4eebf56cc1537ce6baf2e39f75c5c65cf26335ae`

Generated files:

- `exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz`
- `exp063_full_replay_feature_cache_feature_schema.csv`
- `exp063_full_replay_feature_cache_summary.json`

## Interpretation

The exp063 full replay train feature cache is available as the Kaggle output of `kentookumura/exp072-exp063-full-replay-feature-cache-train` v2. Downstream experiments should use this deterministic train cache as a kernel source and regenerate test features inside their own inference notebooks with the same stable per-well seed policy.

## Next

Use this cache for `exp073_gpu_reproducibility_guard_for_exp063_full_replay`.
