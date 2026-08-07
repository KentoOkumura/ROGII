# Result

## Summary

Implementation and local OOF audit are complete. Kaggle train/inference were not run because no compact gate policy passed the discussion metric guard.

`exp078_compact_surface_longtail_gate` audits whether exp075 compact surface predictions can be added lightly to the exp073 deterministic full replay anchor only on long-tail rows.

## Current Interpretation

The compact branch improves global OOF RMSE/SSE under several long-tail gates, but the improvement comes with large worst-well regressions. Under the current guard, `baseline_exp073` remains selected.

Best diagnostic compact policy by RMSE:

- policy: `tail_or_len_long_w020`
- RMSE: `9.362945426943881`
- delta RMSE: `-0.16342932244680064`
- delta SSE: `-11681440.0`
- max well RMSE regression: `2.908365249633789`

Decision: no inference port or submission candidate from exp078 as currently implemented.

## Artifacts

Planned train outputs:

- `artifacts/exp078_compact_surface_longtail_gate_metrics.csv`
- `artifacts/exp078_compact_surface_longtail_gate_bucket_metrics.csv`
- `artifacts/exp078_compact_surface_longtail_gate_well_metrics.csv`
- `artifacts/exp078_compact_surface_longtail_gate_best_predictions.csv.gz`
- `artifacts/exp078_compact_surface_longtail_gate_summary.json`
