# pf_beam_disagreement_error_map

PF/Beam disagreement, confidence, and tail-length buckets from exp083 well summaries.

## Inputs
- well_summary: `experiments/exp083_pf_beam_true_tvt_2d_well_eda/artifacts/pf_beam_true_tvt_2d_well_eda_clean_all_well_summary.csv`
- plot_manifest: `experiments/exp083_pf_beam_true_tvt_2d_well_eda/artifacts/pf_beam_true_tvt_2d_well_eda_clean_all_plot_manifest.csv`
- ml_well_metrics: `/tmp/kaggle-output/exp073_gpu_reproducibility_guard_for_exp063_full_replay/train_v2/artifacts/exp063_full_replay_repro_guard_by_well.csv`

## Overall

| Metric | Value |
|---|---:|
| wells | 773 |
| rows | 3783989 |
| PF pooled RMSE | 14.493061 |
| Beam pooled RMSE | 15.774328 |
| ML pooled RMSE | 9.526375 |
| PF minus ML RMSE | 4.966687 |
| PF minus Beam RMSE | -1.281266 |

## Outputs

- `pf_beam_disagreement_overall_metrics.csv`
- `pf_beam_disagreement_well_map.csv`
- `pf_beam_disagreement_bucket_metrics.csv`

This is diagnostic only; it does not define a hard router or selector.
