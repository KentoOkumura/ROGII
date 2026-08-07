# exp073-anchor weighted tail readout

Generated on 2026-06-16 with `exp073` as the anchor.

## Output files

- `weighted_tail_overall_metrics.csv`
- `weighted_tail_well_error_map.csv`
- `weighted_tail_bucket_metrics.csv`

## Overall versus exp073

| Candidate | RMSE | Anchor RMSE | Delta RMSE vs exp073 | Decision |
| --- | ---: | ---: | ---: | --- |
| exp027 | 0.005251 | 4.382939 | -4.377688 | Public sample copy behavior; not a hidden-test model signal. |
| exp073 | 4.382939 | 4.382939 | 0.000000 | Deterministic ML anchor. |
| exp070 | 4.341515 | 4.382939 | -0.041424 | Slight visible improvement, concentrated in `00bbac68`. |
| exp063 | 4.533153 | 4.382939 | +0.150214 | Worse overall, but better on two visible wells. |
| exp054 | 6.097466 | 4.382939 | +1.714527 | Reject. |
| exp026 | 8.097674 | 4.382939 | +3.714735 | Reject. |
| exp069 | 12.851383 | 4.382939 | +8.468444 | Reject. |

## Segment notes

- exp070 improves `00bbac68` by RMSE 5.944230 -> 4.860883, but worsens `000d7d20` by 2.244715 -> 3.095805 and `00e12e8b` by 3.050241 -> 4.519830.
- exp063 improves `000d7d20` by 2.244715 -> 2.207111 and `00e12e8b` by 3.050241 -> 2.797817, but worsens `00bbac68` by 5.944230 -> 6.296650.
- exp070 improves the long-tail `1000+` bucket versus exp073: 4.804359 -> 4.497412.
- exp063 improves short/mid buckets versus exp073 except `1000+`, where it is worse: 4.804359 -> 5.003211.

## Decision

The visible evidence supports only a guarded, geometry-dependent router between exp073, exp070, and possibly exp063. It does not support a global average or global replacement.
