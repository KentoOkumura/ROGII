# exp070_gpu_reproducibility_guard_for_exp063 Requirements

## Goal

Fix the reproducibility boundary for exp063-derived LightGBM retraining before using exp063 CV deltas for fine-grained decisions.

## Requirements

- Use `exp063_ravaghi_vs_pixiux_lgbm_feature_parity_audit` as the parent experiment.
- Do not regenerate PF/Beam or likelihood-PF features.
- Read PF/Beam/likelihood-PF inputs from exp063 output `ravaghi_vs_pixiux_public_replay_tracker_features_train.csv.gz`.
- Change only LightGBM training hyperparameters and execution mode.
- Evaluate with GroupKFold grouped by exp063 `well`.
- Save mode-level metrics, OOF predictions, feature schema, model manifest, model hashes, and OOF content SHA.
- Treat the result as a train-side reliability guard, not as direct Public LB evidence.

## Non-Goals

- No inference submission by default.
- No CatBoost, Ridge stack, final public notebook blend, static visible override, or projection postprocess.
- No new PF/Beam feature generation.
