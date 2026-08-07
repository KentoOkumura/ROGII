# exp059_pf_model_diff_foldsafe_surface_shrink result

## Summary

Status: submitted complete.

Kaggle train version 3 completed on
`kentookumura/exp059-pf-model-diff-train`. The run used 1,782,279 rows and
773 wells from the exp029 cutoff-0.65 public sel15 PF/Beam pseudo-test feature
artifact. Version 1 failed because source configs were not packaged; version 2
failed because the copied seed-bagging helper passed member metadata to a local
summary function that did not accept it. Version 3 fixed both issues.

## Metrics

| Candidate | Original-fold RMSE | Well-hash RMSE | Delta vs confidence raw original | Delta vs confidence raw well-hash |
| --- | ---: | ---: | ---: | ---: |
| `lgbm_capacity_pf_model_diff_foldsafe_raw` | 15.037567 | 14.735200 | -0.908111 | -0.834016 |
| `lgbm_capacity_pf_model_diff_foldsafe_foldout_bucket_shrink` | 15.092041 | 14.770612 | -0.853637 | -0.798604 |
| `lgbm_capacity_pf_model_diff_foldsafe_confidence_foldout_bucket_shrink` | 15.098501 | 14.797292 | -0.847177 | -0.771924 |
| `pf090_hold010` | 15.089532 | 15.089532 | -0.856146 | -0.479684 |
| `public_pf_selector` | 15.172636 | 15.172636 | -0.773042 | -0.396580 |
| `exp054_foldout_control` | 15.368749 | 15.583832 | -0.576929 | +0.014617 |
| `exp052_foldout_control` | 15.506899 | 15.684333 | -0.438779 | +0.115118 |
| `lgbm_capacity_pf_confidence_only_raw` | 15.945678 | 15.569216 | 0.000000 | 0.000000 |

## Interpretation

The fold-safe PF/Beam-vs-exp052/054 model-diff features are a strong positive
result on the exp029 pseudo-test surface. The selected raw candidate beats the
exp058 confidence-only raw reference by -0.908111 original-fold and -0.834016
well-hash. It also beats the direct PF controls overall on both audits:
`pf090_hold010` by -0.051965 / -0.354332 and `public_pf_selector` by
-0.135069 / -0.437436.

Surface-specific fold-out shrink did not help. The raw candidate is best;
fold-out bucket shrink and confidence-conditioned shrink both made RMSE worse.
Distance bucket metrics are mixed: raw model-diff improves near and mid buckets
strongly, but in `rows_2500_plus` it remains worse than `pf090_hold010` and
`public_pf_selector`.

## Decision

This is the first exp029 pseudo-test surface ML candidate in this branch that
beats both direct PF controls overall on original-fold and well-hash. It is
supported as a train-side candidate. Do not adopt the shrink variants.

Inference version 1 completed as
`kentookumura/exp059-pf-model-diff-infer`. The port used only
`lgbm_capacity_pf_model_diff_foldsafe_raw`; shrink variants were excluded.
`submission.csv` passed local submit-check. Public sample output has SHA256
`2b86386f19279e79e7184096f353ccf2b97785de67b268caa56aa5f85405a815`, identical
to exp027 public replay and exp058 inference output.

Public sample changed rows are 0 because all three public sample wells used the
`physical_visible` branch. Therefore the run confirms packaging and format, but
does not provide hidden-branch output evidence. A code submission would be a
medium-risk hidden-branch test, not a public-output-improved candidate.

The code submission completed as ref `53549815` with Public LB 11.878. This is
a large improvement over exp058 PF-confidence-only Public LB 12.778, but it is
still worse than exp054 seed-bag pseudo-tail Public LB 11.856 by +0.022 and
worse than exp039 Public LB 11.740 by +0.138. Do not promote exp059 to the
current ML-route LB anchor.
