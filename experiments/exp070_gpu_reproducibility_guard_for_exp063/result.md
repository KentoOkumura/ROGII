# exp070_gpu_reproducibility_guard_for_exp063 Result

## Status

Discarded/superseded for the original exp063 reproducibility purpose.

GPU train v4, CPU train v2, and raw-test-regenerating inference v1 completed on Kaggle, but this experiment used the compact tracker output with 65 features. The intended exp063 full public replay surface has 196 features, so exp070 is not a valid guard for exp063 reproducibility. Keep the outputs only as a runtime/reference audit.

Earlier runs remain invalid or superseded: train v1 completed but is invalid because `well` was read without `dtype=str`; v2 was manually stopped; v3 had the dtype fix but still used the combined GPU+CPU package and was manually stopped to avoid wasting weekly GPU quota.

## Evaluation

The corrected train runs used exp063 saved tracker/PF/Beam output train features, 3,783,989 rows, 773 wells, 65 features, and 5-fold GroupKFold by `well`.

Corrected pooled RMSE:

| mode | model | RMSE | prediction SHA |
| --- | --- | ---: | --- |
| `gpu_repro_guard_dp_threads8` | `lgb_mean` | 9.731506199 | `09ccb9edd59cd50057da0ee7738229749996219708f36e6c45f870d0efd026a5` |
| `cpu_deterministic_threads8` | `lgb_mean` | 9.764917679 | `e09344e4fe0a8158150c60e018cb12867107b870e1bbb262dc9b46f0e8a3d557` |

The source exp063 tracker feature SHA was `4ebf8f4fec0be09fba5c9c585d3699a78fbc6511b16b066098a7ca65362c5f90`.

Runtime:

| run | elapsed seconds | elapsed h:mm:ss |
| --- | ---: | --- |
| GPU train v4 | 6,900.889 | 1:55:00.889 |
| CPU train v2 | 6,308.689 | 1:45:08.689 |

Inference v1 regenerated exp063 public replay PF/Beam/likelihood-PF features from current raw test files. Feature generation took 94.557 sec; total inference took 122.117 sec. It produced 14,151 predicted rows, 0 fallback rows, and submission SHA `9d26b8b80df859b0e137e14e9fc3dba4acaf68252ebc2e87dc40153541be291b`.

Submit-check for `/tmp/kaggle-output/exp070_gpu_reproducibility_guard_for_exp063/infer_v1/submission.csv` passed: no duplicate IDs, no empty/NaN/Inf-like values, 14,151 rows, 2 columns, sample-compatible header and row count.

## Interpretation

The corrected train runs are internally consistent for the 65-feature compact tracker surface, but they do not answer the original exp063 reproducibility question. GPU train v4 is not a valid CV anchor for exp063 full replay. CPU train v2 is only a runtime/control reference. In this deterministic setup CPU was 592.200 sec faster than GPU, while GPU produced the better CV by 0.033411480 RMSE on the invalid-for-purpose surface.

## LB Reinterpretation

2026-06-14 follow-up: the user reported that exp070 updated the Public LB record. The Kaggle submissions list at that time showed two recent completed scores, `8.548` (`ref=53669416`) and `8.515` (`ref=53669453`), either of which beats exp027 `8.781` and exp063 `8.811`. The exact ref-to-artifact attribution still needs to be verified because the submissions had empty descriptions.

This changes the interpretation of exp070, not the original reproducibility verdict:

- Exp070 remains invalid as `gpu_reproducibility_guard_for_exp063`, because it used the 65-feature compact tracker surface instead of the intended 196-feature full replay surface.
- Exp070 is now a valid LB candidate / feature-surface signal. The compact tracker surface plus LightGBM retraining and raw-test PF/Beam regeneration may generalize better to the public hidden set than the fuller exp063 feature surface.
- The CV/LB relationship is weak on this branch: exp070 GPU CV `9.731506199` is worse than exp063 CV `9.630105`, while the reported LB is better. Treat small CV differences on exp063-derived replay surfaces cautiously until a more robust holdout or repeated attribution is available.
- Next action is to verify the submission ref and SHA, then reclassify exp070 as `invalid_as_repro_guard_valid_as_lb_candidate` rather than a pure discard.

## Next

Do not use exp070 as the exp063 reproducibility candidate. First verify which exp070 submission produced the new LB record, then record it as a compact-surface candidate. Separately, use exp072's train-side full replay feature cache to implement a corrected full-replay LightGBM GPU reproducibility guard with the 196-feature surface.
