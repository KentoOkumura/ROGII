# exp070_gpu_reproducibility_guard_for_exp063 Tasklist

- [x] Create steering docs.
- [x] Create experiment from exp063.
- [x] Replace train path with exp063-output feature loader.
- [x] Add LightGBM reproducibility modes.
- [x] Save metrics, OOF prediction, feature schema, model manifest, and hashes.
- [x] Replace train notebook with reproducibility guard notebook.
- [x] Mark inference notebook as not applicable by default.
- [x] Prepare Kaggle train notebook.
- [ ] Run the same corrected GPU-only Kaggle train package twice. v1 invalid due well dtype bug; v2/v3 manually stopped; GPU-only config prepared.
- [ ] Compare metrics, OOF SHA, model hashes, and runtime.
- [ ] Record result in `SESSION_NOTES.md`, `result.md`, `metrics.json`, and `experiment_summary.md`.
