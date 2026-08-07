# exp076_exp039_cv_reassessment

`exp073_gpu_reproducibility_guard_for_exp063_full_replay` の deterministic full replay LightGBM family を、`exp039` / `exp038` 系の CV surface で再評価する実験。

ユーザー指定の backlog 名は `exp073_exp039_cv_reassessment` だが、既存最大番号が `exp075` のため、実験フォルダは正しく increment して `exp076_exp039_cv_reassessment` とした。

## Status

実装済み。Kaggle train / inference は未実行。

## Hypothesis

exp073 deterministic full replay LightGBM family は、exp039 CV surface 上でも旧 exp039 single-LGBM branch より強い可能性がある。ただし exp073 native CV と exp039 CV は評価面が違うため、結果は anchor 更新ではなく比較監査として扱う。

## Scope

- Route: `ml_model`
- Parent: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- CV surface: `exp029_public_sel15_pf_oof_feature_generation/features/public_sel15_pf_oof_features.csv.gz`
- Train feature cache: `exp072_exp063_full_replay_feature_cache` の 196-feature full replay cache
- Train responsibility: LightGBM reproducibility
- Inference responsibility: PF/Beam raw-test regeneration reproducibility

## Validation Strategy

`leave_one_original_fold_out` と `well_hash_holdout` の 2 audit を使う。train は固定 full replay feature cache、deterministic LightGBM settings、model SHA、OOF prediction SHA で再現性を確認する。inference は exp073 の stable per-well seed policy で raw-test PF/Beam features を再生成し、decompressed content SHA、prediction SHA、submission SHA を記録する。

## Train

Train notebook は固定済み exp072/exp073 full replay train feature cache と exp039 CV surface を `id` join し、`leave_one_original_fold_out` と `well_hash_holdout` で exp073 LightGBM family を評価する。

証跡として feature content SHA、OOF prediction SHA、fold model SHA、model manifest を保存する。

## Inference

Inference notebook は train output の saved boosters を読み、raw test から exp073 full replay PF/Beam/likelihood-PF features を再生成して予測する。PF/Beam の再現性は exp073 で確立済みの stable per-well seed policy と generated feature content SHA で担保し、重複生成による照合は行わない。

## Findings

未実行。実装と静的検証後に Kaggle train / inference 結果を追記する。
