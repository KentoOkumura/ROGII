# exp092_u_projection_correction_disagreement_fullrun

## Status

Kaggle train v1, OOF delta guard, inference, and code submission completed. Public LB is 8.350.

## Hypothesis

`exp085_u_projection_feature_ablation` は timeout したが、log-derived fold metrics では `u_projection_correction_plus_disagreement` が control より明確に良かった。全 variant を再実行せず、この最有望 variant だけに絞れば、正式 pooled OOF、bucket metrics、feature importance、prediction SHA を完走できる可能性が高い。

## Validation Strategy

exp072 deterministic full replay train feature cache と exp073 LightGBM config family を固定する。target は `TVT - last_known_tvt` のまま、base 196 features に U-space projection correction と U-space disagreement feature group を add-only して GroupKFold by well で評価する。

## Scope

- Route: `ml_model`
- Parent: `exp085_u_projection_feature_ablation`
- Base surface parent: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- Cache parent: `exp072_exp063_full_replay_feature_cache`
- Variant: `u_projection_correction_plus_disagreement`
- Models: `lgb0` / `lgb1` / `lgb2` and `lgb_mean`
- Inference: user-requested `lgb1` port submitted as `ref=53927479`

## Findings

Kaggle train v1 completed on `kentookumura/exp092-uproj-corr-disagree-train`.

| model | pooled OOF RMSE |
| --- | ---: |
| `lgb1` | 9.322479896 |
| `lgb2` | 9.338192405 |
| `lgb_mean` | 9.343064066 |
| `lgb0` | 9.533126438 |

Best `lgb1` improves over exp077 policy OOF 9.470514801 by -0.148034906 and exp073 raw anchor 9.526374749 by -0.203894854.

OOF delta guard in `artifacts/oof_delta_guard/` confirms the overall and long-tail gains, but raises a by-well warning: 459 wells improve vs exp077 and 314 worsen, with max regression +4.164460. Near-row 0-250 is not covered by the aligned OOF surface, so it remains inconclusive. Path continuity does not show broad collapse (`lgb1` ge10 step spike = 1, ge25 = 0).

## Submission

| ref | Public LB | note |
| --- | ---: | --- |
| `53927479` | 8.350 | user-corrected exp092 submission |

Public LB 8.350 improves exp077 8.611 and exp098 8.441, and is now the ML route submitted anchor. Kaggle submission description was blank, and the local submission output SHA is not recorded in this repo.

## Hidden Assert Probe

Normal Kaggle notebook runs only see the exposed visible test. Hidden LB test checks are implemented as opt-in submission-rerun assertions under `inference.hidden_assert_probe`, disabled by default. The probe reports pass/fail through assertion success or failed check names only; hidden rows, well counts, and aggregate values are not logged.

The probe now targets exp092-specific label-free failure proxies: `pred_delta` magnitude, per-well prediction range, near-prefix delta/step instability, projection correction outliers, and PF/Beam/likelihood-PF U-space disagreement outliers.

## Expected Outputs

- `exp092_u_projection_correction_disagreement_fullrun_metrics.csv`
- `exp092_u_projection_correction_disagreement_fullrun_by_well.csv`
- `exp092_u_projection_correction_disagreement_fullrun_bucket_metrics.csv`
- `exp092_u_projection_correction_disagreement_fullrun_projection_feature_summary.csv`
- `exp092_u_projection_correction_disagreement_fullrun_feature_importance.csv`
- `exp092_u_projection_correction_disagreement_fullrun_feature_importance_mean.csv`
- `exp092_u_projection_correction_disagreement_fullrun_feature_importance_mean_top.png`
- `exp092_u_projection_correction_disagreement_fullrun_predictions.csv.gz`
- `exp092_u_projection_correction_disagreement_fullrun_feature_schema.csv`
- `exp092_u_projection_correction_disagreement_fullrun_lgb_models/manifest.json`
- `exp092_u_projection_correction_disagreement_fullrun_summary.json`
