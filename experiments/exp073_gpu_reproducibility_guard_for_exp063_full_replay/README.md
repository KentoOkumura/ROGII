# exp073_gpu_reproducibility_guard_for_exp063_full_replay

## Status

Implemented, not run. Kaggle push is intentionally blocked until exp072 feature cache output is available.

## Hypothesis

exp063 の GPU train-side CV が bitwise 再現しなかった原因を、feature generation ではなく LightGBM 実行 mode の差分として切り分けられる可能性がある。exp072 の raw-train-only 196-feature cache を固定入力にすれば、GPU double precision / deterministic flags / fixed threads の再現境界を SHA 付きで監査できる。

## Purpose

Corrected reproducibility guard for the exp063 full public replay LightGBM surface.

exp070 used a 65-feature compact tracker frame, so it is invalid for the original exp063 full replay reproducibility question. This experiment uses the exp072 train-only cache, which should contain the full Pixiux replay surface with 196 features.

## Scope

- Train reads `exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz` from exp072 output.
- Train reruns only the exp063 Pixiux public LightGBM 3-config family.
- Default mode is `gpu_repro_guard_dp_threads8`.
- Inference regenerates current raw test PF/Beam/likelihood-PF features and applies exp073 saved boosters.
- No static visible override, final public blend, CatBoost, Ridge stack, pretrained public booster, or projection postprocess is used.

## Validation Strategy

Train は exp072 cache を読み、feature count 196 を assert する。GroupKFold by `well`、target `TVT - last_known_tvt`、exp063 Pixiux LightGBM 3 configs を固定し、`gpu_repro_guard_dp_threads8` の pooled RMSE、OOF prediction SHA、model SHA、runtime を保存する。

Inference は train-side CV と分け、raw test files から exp063 public replay test features を再生成して exp073 saved boosters を適用する。

## Findings

No run yet. exp072 がまだ実行中のため、この実験は notebook package 準備まで完了した状態。

## Expected Outputs

- `exp063_full_replay_repro_guard_metrics.csv`
- `exp063_full_replay_repro_guard_by_well.csv`
- `exp063_full_replay_repro_guard_predictions.csv.gz`
- `exp063_full_replay_repro_guard_feature_schema.csv`
- `exp063_full_replay_repro_guard_summary.json`
- `exp063_full_replay_repro_guard_lgb_models/manifest.json`
- `exp063_full_replay_repro_guard_inference_metrics.csv`
- `exp063_full_replay_repro_guard_inference_summary.json`
- `submission.csv`
