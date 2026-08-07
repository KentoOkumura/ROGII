# metric_weighted_tail_error_map readout

Generated on 2026-06-16 with `exp027` as the anchor.

## Inputs

- `exp027`: `experiments/exp027_public_replay_needless090_sel15_spread3/artifacts/submission.csv`
- `exp073`: `/tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/inference_v2/submission.csv`
- `exp063`: `/tmp/kaggle-output/exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit/infer_v2/submission.csv`
- `exp070`: `/tmp/kaggle-output/exp070_gpu_reproducibility_guard_for_exp063/infer_v1/submission.csv`
- `exp069`: `/tmp/kaggle-output/exp069_pixiux_pf_beam_direct_submit_audit/infer_v3/submission.csv`
- `exp039`: `experiments/exp039_ravaghi_single_lgbm_inference_submit/artifacts/submission.csv`
- `exp054`: `experiments/exp054_pseudo_tail_seed_bagging_inference_submit/artifacts/submission.csv`
- `exp026`: `experiments/exp026_pseudo_tail_bucket_shrink_inference_submit/artifacts/submission.csv`

`exp050` was not included because no local `submission.csv` was present under its experiment directory or `/tmp/kaggle-output`.

## Output files

- `weighted_tail_overall_metrics.csv`
- `weighted_tail_well_error_map.csv`
- `weighted_tail_bucket_metrics.csv`
- `weighted_tail_row_error_map.csv`

An additional exp073-anchor run is stored in `../metric_weighted_tail_error_map_anchor_exp073/`.

## Overall readout

Visible/public sample RMSE with `exp027` anchor:

| Candidate | RMSE | MAE | Bias | Note |
| --- | ---: | ---: | ---: | --- |
| exp027 | 0.005251 | 0.004173 | -0.001179 | Public replay anchor; nearly exact on visible train-derived sample. |
| exp039 | 0.005251 | 0.004173 | -0.001179 | Same visible output as exp027. |
| exp070 | 4.341515 | 3.143055 | 1.139688 | Best non-copy visible score, driven by `00bbac68`. |
| exp073 | 4.382939 | 3.008014 | 2.025006 | Current deterministic ML anchor; close to exp070 overall. |
| exp063 | 4.533153 | 3.035690 | 1.898003 | Better than exp073 on two smaller visible wells, worse on `00bbac68`. |
| exp054 | 6.097466 | 4.392115 | 2.604088 | Worse than exp063/070/073. |
| exp026 | 8.097674 | 5.777167 | 3.671356 | Worse visible pseudo-tail anchor. |
| exp069 | 12.851383 | 8.329371 | 6.958231 | Direct PF/Beam not viable on visible sample. |

## Well-level readout

Among non-copy candidates:

| Well | Rows | Best | RMSE | Second | RMSE | Interpretation |
| --- | ---: | --- | ---: | --- | ---: | --- |
| `000d7d20` | 3,836 | exp063 | 2.207111 | exp073 | 2.244715 | exp063 is slightly better than exp073; exp070 is much worse. |
| `00bbac68` | 6,014 | exp070 | 4.860883 | exp073 | 5.944230 | exp070 wins the largest visible well and drives the overall result. |
| `00e12e8b` | 4,301 | exp063 | 2.797817 | exp073 | 3.050241 | exp063 is clearly better than exp073; exp070 is weak here. |

## Distance-bucket readout

Non-copy best by distance from tail start:

| Distance bucket | Rows | Best | RMSE | Runner-up | RMSE |
| --- | ---: | --- | ---: | --- | ---: |
| `000-100` | 300 | exp063 | 0.722390 | exp073 | 0.751819 |
| `100-250` | 450 | exp070 | 1.122182 | exp063 | 1.261120 |
| `250-500` | 750 | exp054 | 2.256988 | exp063 | 2.348906 |
| `500-1000` | 1,500 | exp054 | 2.056092 | exp063 | 2.105900 |
| `1000+` | 11,151 | exp070 | 4.497412 | exp073 | 4.804359 |

The long-tail `1000+` bucket dominates row-weighted score. exp070's visible advantage mostly comes from this bucket and the large `00bbac68` well.

## Decision

Do not blend exp073/exp063/exp070 into exp027 based on visible-tail score: exp027/exp039 are nearly exact on the public train-derived sample, so any ML/PF candidate necessarily looks worse there.

For hidden-test strategy, treat this as a diagnostic only:

- exp073 remains the deterministic ML anchor because it has a reproducible full replay CV of 9.526374749 and its LB has already been checked.
- exp070 is not a clean replacement for exp073; it wins visible overall by one large well but loses badly on the other two visible wells.
- exp063 is useful as a local comparison for near/mid rows and two visible wells, but its overall visible score is behind exp073/exp070.
- exp069 direct PF/Beam should stay rejected.

Next actionable experiment: if pursuing a guarded blend, use a very narrow exp073/exp070 gate keyed to `00bbac68`-like long-tail geometry rather than a global blend. Do not spend another cycle on exp073 LB confirmation; this map is for error attribution and router/blend design.
