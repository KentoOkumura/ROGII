# exp061_seedbag_anchor_model_diff_distance_gate result

## Summary

Kaggle train v1 completed on 2026-06-11.

- rows / wells: 1,782,279 / 773
- selected candidate: `lgbm_capacity_pf_confidence_only_seedbag_gate_near_mid_a0p50_far0`
- selected original-fold RMSE: 14.872556
- selected well-hash RMSE: 14.737595
- best original-fold candidate: `lgbm_capacity_pf_model_diff_foldsafe_seedbag_gate_near_mid_a0p50_far0` 14.838812
- best well-hash candidate: `lgbm_capacity_pf_model_diff_foldsafe_raw` 14.735200

## Key Comparisons

| candidate | original-fold | well-hash | delta vs exp054 original | delta vs exp054 well-hash |
| --- | ---: | ---: | ---: | ---: |
| `lgbm_capacity_pf_confidence_only_seedbag_gate_near_mid_a0p50_far0` | 14.872556 | 14.737595 | -0.496193 | -0.846237 |
| `lgbm_capacity_pf_model_diff_foldsafe_seedbag_gate_near_mid_a0p50_far0` | 14.838812 | 14.791874 | -0.529937 | -0.791958 |
| `lgbm_capacity_pf_model_diff_foldsafe_raw` | 15.037567 | 14.735200 | -0.331182 | -0.848632 |
| `exp054_foldout_control` | 15.368749 | 15.583832 | 0.000000 | 0.000000 |
| `pf090_hold010` | 15.089532 | 15.089532 | -0.279217 | -0.494301 |

## Interpretation

The fixed seed-bag anchor gate worked. The conservative `near_mid_a0p50_far0`
profile improved both original-fold and well-hash holdouts versus `exp054_foldout_control`.

The pure model-diff gated candidate is the best original-fold score, but its well-hash
RMSE is slightly worse than exp059 raw. The automatic selected candidate is therefore
the more stable `confidence_only + seedbag gate` variant, not the pure model-diff
variant.

For inference, `config.yaml` now selects:

- `inference.selected_variant: lgbm_capacity_pf_confidence_only`
- `inference.selected_candidate: lgbm_capacity_pf_confidence_only_seedbag_gate_near_mid_a0p50_far0`
- `inference.selected_gate_profile: near_mid_a0p50_far0`

Next action: prepare and run the inference notebook if we want to test Public LB.

## Inference

Kaggle inference v1 completed.

- kernel: `kentookumura/exp061-seedbag-diff-gate-infer`
- selected candidate: `lgbm_capacity_pf_confidence_only_seedbag_gate_near_mid_a0p50_far0`
- submission rows: 14,151
- submit-check: PASS
- SHA256: `2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`
- prediction range: 11587.038593 to 12240.016066
- public sample branch: all 3 wells used `physical_visible`
- changed_rows: 0
- changed_wells: 0
- diff RMSE vs original selector output: 0.000000

The public sample output is identical to exp027/public physical replay output because
all public sample wells are visible train wells. The hidden branch was built and fit,
but public sample changed-row behavior does not test it. Code submit is required to
measure Public LB.

## Submission

Code submission completed.

- ref: `53581056`
- status: `SubmissionStatus.COMPLETE`
- Public LB: 11.826
- Private LB: not available
- submission record: `submissions/SUBMISSIONS.md` v025

This improves exp054 Public LB 11.856 by -0.030 and exp059 Public LB 11.878 by
-0.052. It still does not beat the ML route Public LB anchor exp039 at 11.740.
