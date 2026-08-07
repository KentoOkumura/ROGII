---
name: kaggle-oof-readout
description: "Create repeatable Kaggle OOF error readouts for experiments by joining out-of-fold predictions, feature importance, feature caches, by-well metrics, and bucket summaries. Use when analyzing where a Kaggle model is good or bad, turning OOF diagnostics into follow-up backlog items, or implementing experiments like exp086_oof_feature_importance_error_readout."
---

# Kaggle OOF Readout

Use this skill to turn existing OOF predictions and feature artifacts into a repeatable diagnostic experiment. It is for analysis and backlog steering, not for anchor updates or direct submissions.

## Standard Workflow

1. Use `kaggle-review-exp` first for normal experiment lifecycle rules.
2. Create a new exp when the readout has a new hypothesis or analysis surface. Do not reuse the source model exp id.
3. Create `.steering/YYYYMMDD-expXXX-title/` before editing.
4. Put all code/config/notebooks under `experiments/expXXX_title/`.
5. Keep `experiment.route: ml_model` unless the readout primarily audits PF/Beam generation itself.
6. Run the first full readout on Kaggle. Local smoke is optional only when the needed artifacts are locally complete.
7. Record run evidence in `SESSION_NOTES.md`, `result.md`, `metrics.json`, `experiment_summary.md`, and update/remove the corresponding `KAGGLE_DIRECTION.md` backlog item.
8. When the completed readout provides a reusable OOF interpretation, model explanation, feature/failure analysis, or cross-experiment comparison, update an existing metadata-indexed report in `docs/surveys/` or create one. `docs/surveys/README.md` is the discovery entry point.

## Inputs To Prefer

Read existing generated artifacts instead of recomputing model training:

- OOF predictions with `id`, `well`, target, prediction, and policy/model columns.
- Fold/model feature importance from saved boosters or prior train outputs.
- Feature cache rows joined by `id`; read only selected columns with `usecols`.
- By-well metrics, distance/tail bucket metrics, and prior error maps when available.

For exp073-family analyses, common sources are:

- exp073 OOF predictions from `exp063_full_replay_repro_guard_predictions.csv.gz`
- exp077 policy predictions from `exp077_full_replay_postprocess_guard_predictions.csv.gz`
- exp077 feature importance mean from `exp063_full_replay_repro_guard_feature_importance_mean.csv`
- exp072 full replay feature cache from `exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz`

Always support both local `/tmp/kaggle-output/...` paths and Kaggle `/kaggle/input/notebooks/...` source paths in config. Ignore 0-byte local placeholders.

## Readout Scope

Use OOF readouts to answer:

- Which wells, distance buckets, tail lengths, or prefix-length buckets dominate error?
- Which high-importance feature value buckets have elevated MAE/RMSE?
- Which feature values correlate with absolute or squared error?
- Did a postprocess/guard help globally while hurting specific wells or buckets?
- Which next action is justified: `diagnostic only`, `confidence feature`, `sample weight`, `postprocess`, or `add-only feature`?

Do not use feature importance alone to approve routers, direct PF/Beam replacement, or submissions. Convert findings into a small fold-safe follow-up experiment.

## Implementation Pattern

When building a new readout experiment:

- Copy from a close diagnostic exp only if it reduces setup work, then delete unrelated training/inference code.
- Use a dedicated module such as `oof_feature_importance_error_readout.py`.
- Make train notebook readable: setup, config display, readout execution, artifact display.
- Make inference notebook diagnostic-only if no submission should be produced.
- Save at least:
  - policy/model metrics
  - feature summary
  - feature quantile metrics
  - well summary
  - plots for error lift and error correlation
  - summary JSON with input paths and SHA values

Use `scripts/oof_readout.py` as a template or standalone helper when the source artifacts match the expected schema.

## Recording Rules

In `result.md`, separate score-like diagnostics from true CV/LB:

- OOF baseline and compare RMSE are diagnostic if no new model was trained.
- Do not update route anchors from a readout.
- State the next concrete backlog item and why.

In `docs/surveys/`:

- Keep the human-readable completed analysis: question, evidence boundary, experiment/model structure needed to interpret it, OOF EDA, findings, non-use constraints, and links to the experiment result and study outputs.
- Search `docs/surveys/README.md` first and update the same thematic report instead of creating one file per rerun or plot version.
- Use `oof_analysis` plus any relevant `experiment_review`, `model_explanation`, `feature_analysis`, or `comparison` metadata types.
- Run `task update-survey-index` and `task validate-surveys` before considering the readout documentation complete.

In `KAGGLE_DIRECTION.md`:

- Remove the completed readout backlog item.
- Add a short judgment memo with the source artifacts, main findings, and non-use constraints.
- Update downstream backlog rows with specific feature/bucket candidates and risk notes.

## Validation

Before Kaggle push:

```bash
uv run python -m py_compile experiments/EXP/*.py
uv run python -m json.tool experiments/EXP/EXP_train.ipynb
uv run ruff check experiments/EXP/*.py
uv run python scripts/validate_experiment.py --experiment EXP
uv run python scripts/prepare_kaggle_notebooks.py --experiment EXP --notebook train --run-on-push --strict
```

After Kaggle output:

- Confirm `KernelWorkerStatus.COMPLETE` or equivalent output presence.
- Download output with `kaggle kernels output`.
- Record output path, rows/wells, elapsed time, source SHAs, and artifact SHAs.
- If logs are initially empty, keep the same kernel id and avoid duplicate pushes.
