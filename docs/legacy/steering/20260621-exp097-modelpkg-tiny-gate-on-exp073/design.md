# Design: exp097_modelpkg_tiny_gate_on_exp073

## Inputs

- exp073 inference prediction:
  - `exp063_full_replay_repro_guard_inference_test_predictions.csv.gz`
  - selected mode: `gpu_repro_guard_dp_threads8`
  - selected model: `lgb_mean`
- model-package prediction:
  - `submission_model_package_only.csv`
  - produced by Pilkwang reference notebook or compatible rerun
- sample submission:
  - `data/raw/sample_submission.csv` or Kaggle competition input

## Algorithm

1. Load exp073 inference rows and filter to selected mode/model.
2. Load model-package-only submission and validate `id,tvt`.
3. Align both predictions to sample submission order.
4. For each grid point:

```text
diff = modelpkg_tvt - base_tvt
g = gmax / (1 + (abs(diff) / scale)^2)
pred_tvt = base_tvt + g * diff
```

5. Save variant metrics and selected candidate predictions.
6. For inference, write `submission.csv` only if selected candidate passes:
   - raw model-package diff p95 <= 35 ft
   - correction p95 <= 0.10 ft
   - correction max <= 1.0 ft

## Risk Controls

- No true labels are used.
- No direct model-package replacement.
- No public-output tuning loop.
- Guard failure creates summary JSON but no submission file.
- SHA values are stored for exp073 input, model-package input, selected prediction, and final submission.

## Limitations

This is not hidden-compatible model-package regeneration yet. It consumes a precomputed `submission_model_package_only.csv`, so any submit decision must separately review whether that source is acceptable for the target Kaggle execution.
