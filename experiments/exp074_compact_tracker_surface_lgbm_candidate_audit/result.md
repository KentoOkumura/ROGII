# exp074_compact_tracker_surface_lgbm_candidate_audit Result

## Status

Kaggle train v1 and inference v1 completed. Public LB submit is still optional and has not been run.

## Evaluation

Train v1:

| mode | model | CV RMSE | prediction SHA |
| --- | --- | ---: | --- |
| `gpu_repro_guard_dp_threads8` | `lgb_mean` | 9.731506199 | `09ccb9edd59cd50057da0ee7738229749996219708f36e6c45f870d0efd026a5` |

Train evidence:

- kernel: `kentookumura/exp074-compact-tracker-lgbm-audit-train` v1
- rows / wells / features: 3,783,989 / 773 / 65
- elapsed: 6,925.739 sec
- feature source SHA: `4ebf8f4fec0be09fba5c9c585d3699a78fbc6511b16b066098a7ca65362c5f90`
- model manifest SHA: `e379b078b4fdfaceb39c25fcc8246cab221ab16038e3dac4f8c2b74360197ece`

Inference v1:

- kernel: `kentookumura/exp074-compact-tracker-lgbm-audit-infer` v1
- raw-test regenerated feature SHA: `723d2d29bd4701f05fc7ee7337a6911368dcc4dec651237679b739260a74e5d7`
- rows / fallback: 14,151 / 0
- prediction SHA: `0a4a5c4010217624f8eb73f191ecbbedeaefba369db40542c171c078d5c84a9f`
- submission SHA: `22f9eb3710ccec7741ce8006bee02ed69ed25829439114411cdba0038dcde0bc`
- submit-check: PASS

## Interpretation

exp074 reproduces the exp070 compact-surface CV exactly under a clean candidate-audit experiment name. This supports treating the 65-feature compact tracker surface as a real LB candidate, separate from the exp063/exp073 full-replay reproducibility line.

It is still not a deterministic anchor: inference regenerates PF/Beam/likelihood-PF features with the inherited exp070 public replay implementation, and no Public LB ref has been submitted or attributed for exp074.

## Next

Optional next step is a code submission from inference v1 if a fresh Public LB attribution is needed for this exact exp074 kernel version and submission SHA.
