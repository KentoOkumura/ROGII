# exp078_compact_surface_longtail_gate

## Status

Implemented. Local OOF audit completed; no submit candidate selected.

## Hypothesis

The exp075 compact PF/Beam surface has useful signal on long-tail rows, but visible evidence does not support global replacement of the exp073 deterministic full replay anchor. A small conditional blend can capture long-tail SSE gains while limiting short/mid and well-level regressions.

## Scope

- Reuse exp073 OOF predictions for `gpu_repro_guard_dp_threads8 / lgb_mean`.
- Reuse exp075 compact surface OOF predictions for the same mode/model.
- Evaluate fixed long-tail gate policies with weights `0.05`, `0.10`, and `0.20`.
- Save RMSE, SSE/SSR deltas, tail-bucket deltas, well-level regression metrics, and best-policy predictions.
- Inference reads saved exp073/exp075 test predictions and writes a gated `submission.csv`.

## Validation Strategy

The primary audit is OOF-only. Discussion #698860 motivates tracking SSE because RMSE is a square-root transform of mean squared error. Discussion #700340 motivates tracking worst-well regressions instead of trusting only global OOF RMSE. A submit candidate must improve long-tail SSE and avoid large well-level regressions.

## Findings

Local OOF audit completed on saved exp073/exp075 predictions. Compact gated candidates improved global RMSE/SSE, but all non-baseline candidates violated the worst-well regression guard from the metric discussion framing.

- Selected policy: `baseline_exp073`
- Baseline RMSE: `9.526374749390682`
- Best compact diagnostic by RMSE: `tail_or_len_long_w020`, RMSE `9.362945426943881`, delta `-0.16342932244680064`
- Rejection reason: max well RMSE regression `2.908365249633789` exceeds the configured `0.25` guard.
- Decision: do not port compact gate to inference without a narrower guard.

## Expected Outputs

- `exp078_compact_surface_longtail_gate_metrics.csv`
- `exp078_compact_surface_longtail_gate_bucket_metrics.csv`
- `exp078_compact_surface_longtail_gate_well_metrics.csv`
- `exp078_compact_surface_longtail_gate_best_predictions.csv.gz`
- `exp078_compact_surface_longtail_gate_summary.json`
