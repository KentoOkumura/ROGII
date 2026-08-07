# 要件

## 依頼

`exp073_exp039_cv_reassessment` を実装する。実験名の接頭辞は既存最大番号から正しく increment し、実際の experiment は `exp076_exp039_cv_reassessment` とする。

## 制約

- Route: `ml_model`
- 対象は破棄済み `exp068` の置き換えで、`exp063` ではなく `exp073_gpu_reproducibility_guard_for_exp063_full_replay` を exp039 CV surface で評価する。
- train は LightGBM の再現性を担保する。
- inference は PF/Beam の再現性を担保する。
- PF/Beam の再現性手順は exp073 で確立済みなので、2 回生成による一致確認はしない。
- static public-sample prediction artifact は使わない。

## 受け入れ基準

- `experiments/exp076_exp039_cv_reassessment/` に train / inference notebook、config、実装モジュール、記録ファイルがある。
- train notebook は exp039 CV surface と exp072/exp073 full replay train cache を `id` align し、LightGBM model SHA と OOF prediction SHA を保存する。
- inference notebook は saved booster を使い、raw test から PF/Beam/likelihood-PF features を再生成し、feature content SHA、prediction SHA、submission SHA を保存する。
- `task validate-exp EXP=exp076_exp039_cv_reassessment` が通る。
