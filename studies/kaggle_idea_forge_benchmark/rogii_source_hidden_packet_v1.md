# Source-hidden task packet: wellbore trajectory prediction

## Evaluation boundary

- Evidence cutoff: 2026-07-12.
- This packet is the complete allowed source bundle for the blind run.
- Do not use Web search, Kaggle discussions, solution writeups, later experiment files, survey reports, backlog files, or outputs from other benchmark runs.
- Treat every claim not stated here as an assumption.

## Task and deployment contract

- Code competition. Predict `tvt` for every row in the unknown suffix of each horizontal well.
- Metric: row-pooled RMSE. A small number of large, persistent misses can dominate the score.
- Train: 773 wells and 3,783,989 scored rows.
- Each horizontal-well table contains measured-depth/trajectory coordinates, a gamma-ray log (`GR`), and `TVT_input`. `TVT_input` is known in a prefix and missing in the evaluation suffix.
- Each well has a paired vertical reference/typewell table containing depth-indexed reference information including GR.
- Some formation-related columns exist only in train and are unavailable at inference.
- Public test is a three-well example. Hidden test is expected to contain roughly 200 wells and replaces the example files and `sample_submission.csv` during notebook rerun.
- The inference notebook must enumerate wells dynamically, align predictions one-to-one to the runtime sample IDs, run offline, and finish within nine hours.
- CPU and T4 GPU are available. Peak memory and deterministic regeneration matter.
- Trusted validation uses whole-well GroupKFold. Any learned selection or blending must be trained out-of-fold and reproducible for an unseen well.

## Current anchors

### E01 — reproducible ML anchor

- A deterministic ML pipeline has CV RMSE 9.5264 and Public LB 8.780.
- It combines row features with several precomputed physical/path predictions.
- It is reproducible and shippable, but still makes long correlated errors in some wells.

### E02 — physical/path baselines

- A likelihood particle-filter prediction has full-OOF RMSE 11.5949.
- An exact HMM/likelihood-PF blend reaches RMSE 10.2697.
- Five existing PF/beam-style candidate paths have full-grid oracle-union RMSE 7.4340. This oracle uses truth to choose the closest candidate and is not deployable.
- Candidate selection without truth is an unresolved bottleneck.

## Pre-cutoff experiment evidence

### E03 — local learned heatmap signal

- A five-channel CNN heatmap probe was trained on local windows.
- One-fold top-3 coverage within 10 ft: real GR 0.4492, shuffled GR 0.2324, no-GR 0.0625.
- Five-fold/773-well top-3 coverage: real GR 0.5000, shuffled GR 0.2185, no-GR 0.0714.
- Real-GR top-10 coverage is 0.8089; top-10 oracle RMSE is 13.2963.
- Adding one tested geometry channel changed top-3 coverage from 0.5000 to 0.4877, while its top-10 oracle improved from 13.2963 to 11.9954.
- A larger tested window reduced top-3 coverage to 0.4175.
- Worst-well top-3 coverage remains zero.
- Conclusion at cutoff: the heatmap uses genuine GR signal, but the tested local output is not strong enough for direct replacement.

### E04 — candidate-set headroom

- Existing five-candidate union on covered rows: within-10 0.9496, oracle RMSE 5.0687.
- Ten heatmap candidates alone: within-10 0.8089, oracle RMSE 13.3526.
- Existing five plus heatmap ten: within-10 0.9870, oracle RMSE 2.7455.
- New-best candidate rate is 0.2525 overall and 0.3174 in the 1000-ft-plus suffix bucket.
- All 668 affected wells improve in oracle or remain tied; none worsen because this is a truth-aware union diagnostic.
- No target-free selector has demonstrated that it can identify the useful added candidate.

### E05 — attempts to turn local heatmaps into full-well point paths

- Overlapping local paths were stitched into full-well candidates.
- On covered rows, stitched-only top-5 oracle RMSE is 46.9589; adding them to existing candidates improves oracle 5.1394 to 4.4077.
- A full-grid fill achieved row coverage 1.0, but 56.99% of rows were right-end extrapolations. Stitched-only top-5 oracle RMSE is 50.0852; union oracle improves 7.4340 to 5.9415.
- A learned full-tail multi-trajectory head removed extrapolation fallback and covered every row. Its top-5-only oracle RMSE is 32.3331 and probability-weighted point path RMSE is 59.2721. Adding its five paths improves existing-union oracle 7.4340 to 5.1137.
- Dense-window diagnostics show cases where rank 1 is about 97 ft wrong while rank 2 is about 1.2 ft wrong. Candidate ranking/calibration is weak.
- Conclusion at cutoff: direct or probability-weighted learned point paths are rejected; the candidate set still has large oracle headroom.

### E06 — same-well prefix GR as an auxiliary observation

- A weak likelihood term derived from the same horizontal well's known prefix was added to an HMM that already used typewell GR.
- Best variant improves RMSE 11.5949 to 11.3500 and improves all distance buckets.
- A stronger weight mostly removes the gain.
- 461 wells improve and 312 worsen; worst-well regression is +46.95 ft.
- The run takes about 10 h 50 m on the tested CPU path, above the submission wall.
- Earlier direct candidate, hard switch, and dense-gate uses of this signal were rejected.
- Conclusion at cutoff: the tested additive likelihood is diagnostic only; the signal is not approved for raw inference.

### E07 — fixed-lag particle smoothing

- A 500-particle, 128-seed, lag-64 ancestor smoother was evaluated on all 773 wells.
- It worsens RMSE 11.5949 to 13.4954 and within-10 from 0.7728 to 0.6738.
- Runtime is 14.88 CPU hours across four deterministic shards.
- The experiment used a different seed namespace from the frozen control. Even the last 64 unsmoothed rows are 1.7669 RMSE worse than control, so it is not a clean paired causal ablation.
- Lag 128/256 and a seed-paired rerun were not executed.
- Conclusion at cutoff: this lag-64 implementation is closed; the effect of smoothing alone is not identified.

## Reusable assets

- Five whole-well existing candidate paths and their OOF predictions.
- Local heatmap logits, ranks, entropy/margins, and ten candidate windows across five folds.
- Full-grid five-path learned artifact with coverage flags and path scores.
- HMM/PF predictions, uncertainty summaries, and by-well/distance metrics.
- Whole-well fold assignments and a deterministic ML training/inference pipeline.

## Required deliverable

Generate a diverse portfolio of 10–14 next-experiment ideas. The top five must be implementable as staged tests, not just broad model names. For each idea state:

- mechanism and hypothesis;
- which evidence supports and contradicts it;
- exact difference from the closest failed attempt;
- cheapest informative test, full validation, and kill criterion;
- hidden inference/runtime contract;
- whether candidate coverage and target-free selectability are separate concerns.

Do not implement, train, browse, or submit anything.
