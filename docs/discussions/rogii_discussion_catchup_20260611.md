# ROGII Discussion Catch-up 2026-06-11

- archived topics: `docs/discussions/rogii-wellbore-geology-prediction-*.md`
- topic listing: `docs/discussions/rogii-wellbore-geology-prediction_topics_recent_20260611.csv`
- source: Kaggle competition discussions, `recent` sort, pages 1-5, 86 topics

## Read First

1. `rogii-wellbore-geology-prediction-707613-pf-baseline-got-lb-8-863-any-ideas-to-make-it-learnable-with-nn.md`
   - PF baseline with public LB 8.863; useful comments point toward measuring whether PF generates near-truth candidates, then learning a candidate scorer / classifier rather than direct regression first.
2. `rogii-wellbore-geology-prediction-705210-png-files-don-t-match-the-data.md`
   - PNG/PPTX visual references do not reliably match raw CSV typewell data; treat images as explanatory material, not as a source of truth for feature engineering.
3. `rogii-wellbore-geology-prediction-699853-multi-trajectory-prediction-mtp-with-deep-cnn-for-welllog-inversion.md`
   - Long MTP/CNN thread. The strongest direction is multi-mode trajectory generation plus ranking/verifier, with repeated warnings that GR matching is ill-conditioned.
4. `rogii-wellbore-geology-prediction-703883-practical-notes-from-reproducing-and-stress-testing-notebooks.md`
   - Practical replay checklist: rebuild predictions from competition inputs, write root `submission.csv`, merge by id, check pairwise submission distances and per-well continuity.
5. `rogii-wellbore-geology-prediction-704273-how-much-should-we-trust-the-lb-score.md` and `rogii-wellbore-geology-prediction-701995-is-the-public-lb-test-set-26-fixed.md`
   - Useful for LB/CV caution. Do not tune only against public LB before hidden-compatible replay checks pass.

## Signals For Next Experiments

- PF/beam remains the most credible public route, but the discussion emphasis has shifted from hand-coded PF to candidate generation plus learned selection.
- New notebook ecosystem around `ridge-sp`, `SP45`, `fle3n`, and blend pipelines should be audited before more local-only self tuning.
- Typewell/PNG mismatch discussion reinforces using `data/raw` CSVs as the source of truth and avoiding image-derived assumptions.
- Replay audits should flag static visible-test CSV writers, positional row writing, and post-processing that hides per-well discontinuities.

