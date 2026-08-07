# exp076_exp039_cv_reassessment 結果

## Status

Kaggle GPU train v3、inference v1、提出まで完了。Public LB は 8.799。

## Train Evidence

- Kaggle kernel: `kentookumura/exp076-exp039-cv-reassessment-train`
- Kaggle version: 3
- output取得先: `/tmp/kaggle-output/exp076-train-v3-complete`
- rows: 3,783,989
- wells: 773
- features: 196
- fold assignment: `exp039_fold_by_well_id`
- target audit overlap rows: 1,781,963
- `cache_target_vs_exp039_target_delta_abs_max`: 0.0
- `cache_target_matches_exp039_target_delta`: true

Pooled CV:

| audit | model | RMSE |
| --- | --- | ---: |
| leave_one_original_fold_out | lgb0 | 9.966644975 |
| leave_one_original_fold_out | lgb1 | 9.651043444 |
| leave_one_original_fold_out | lgb2 | 9.705470297 |
| leave_one_original_fold_out | lgb_mean | 9.696040174 |
| well_hash_holdout | lgb0 | 9.799962732 |
| well_hash_holdout | lgb1 | 9.551209424 |
| well_hash_holdout | lgb2 | 9.539948111 |
| well_hash_holdout | lgb_mean | 9.553554167 |

Key SHA:

- feature cache content SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- exp039 surface content SHA: `b96046cd452abca92ed7200188e3b628745a3f5ff5ccb70679da1dc97a79b5a3`
- model manifest SHA: `d68aac6d5ec5a34f8ead30a0b119daace4823e4f1207428201ba8e47b54db37f`
- OOF predictions decompressed content SHA: `2fe04bd2980505ee4fd7dbd0acd61b307e429d45a6baa71a2f11edc3effb21ef`
- selected `leave_one_original_fold_out/lgb_mean` prediction SHA: `afcf0bd9325b7ee7060d299285dfa8aeaf035405902839deb47e2ece8d04c783`
- selected `well_hash_holdout/lgb_mean` prediction SHA: `9ee31704447a57f2262a689d4b66300b3eb21f14b1de1d0217342c068f0972c2`

## Interpretation

`leave_one_original_fold_out/lgb_mean` は 9.696040174 で、exp073 native CV 9.526374749 より悪い。`well_hash_holdout/lgb_mean` は 9.553554167 で exp073 native CV に近いが、評価面が違うため deterministic anchor 更新根拠としては扱わない。

exp039 surface は exact id では sparse なので、v3 は exp073 cache 全行に exp039 fold を well 単位で付与した。exact id overlap では target 差分が 0.0 で、target 変更ではなく CV surface reassessment として成立している。

## Inference And Submission

- Kaggle kernel: `kentookumura/exp076-exp039-cv-reassessment-infer`
- Kaggle version: 1
- output取得先: `/tmp/kaggle-output/exp076-infer-v1-complete`
- selected mode/model: `gpu_repro_guard_dp_threads8__leave_one_original_fold_out` / `lgb_mean`
- loaded models: 15
- test rows / submission rows: 14,151 / 14,151
- fallback rows: 0
- prediction range: 11594.158203125 - 12241.3212890625
- prediction SHA: `bfb21114e7deb98e5880ebd0a1f0b33dfb129531dd4e8fa0908e4c65e01e4938`
- submission SHA: `6afd2296208449ef372e4aef49c41de7636aadc266e09fd5ee41a2a4d36623c1`
- regenerated test feature content SHA: `e3567a64807a16c3c4d80fe6bca2611ba3fe8d13b4b20be4540e8d1ac354965c`
- submit-check: PASS
- submission ref: `53757190`
- Public LB: 8.799

## Next Action

Public LB は exp027 8.781 / exp073 周辺の 8.780 より悪いため、exp076 は採用しない。exp039 CV surface reassessment は完了として閉じる。
