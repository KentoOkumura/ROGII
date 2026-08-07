# exp191_typewell_late_range_continuity_selector_on_exp176

## 状態

Kaggle train v1 完了。train-side continuity audit として supported、direct inference / submit は未選定。

## 狙い

exp176 の `lgb_candidate_error_ranker` は OOF RMSE 10.641296 で exp157/158 を上回ったが、row-wise max path switch は 330.842 / 1000 rows と高い。exp176 の positive signal を direct selector として使わず、exp158 と同じ Viterbi continuity guard で path を平滑化する。

## 仮説

exp176 の candidate-error surface を Viterbi continuity selector で平滑化すれば、row-wise の RMSE 改善を大きく失わずに path switch と短い segment を抑えられる。

## 検証方針

exp099 v2 cache、exp072 dense cache、raw train typewell context、exp176 v3 saved boosters から OOF score surface を復元する。候補集合は exp176 と同じ 8 候補に固定し、`likpf_mean_single`、`exp176_error_ranker_rowwise`、Viterbi variants、oracle を比較する。

## 所見

Best Viterbi は `viterbi_sw400_bias000_jw050_jf025_d075_std999999_md0000_seg012`。RMSE 10.598006880 で、exp176 row-wise 10.641296371 から -0.043289491、exp158 best Viterbi 10.789163253 から -0.191156373 改善した。path switch は exp176 row-wise の 261,391 から 3,620 へ大きく減少した。

ただし near / mid distance bucket は小幅に悪化し、by-well では 356 wells が exp176 row-wise より悪化した。selected TVT を直接推論・提出に使うのではなく、後続の exp148 系 confidence / segment-stability feature surface として扱う。

## 実行コスト

- Runtime: CPU
- 新規 LightGBM booster: 0
- exp176 saved booster inference: 15
- Viterbi variants: 180
- control / parent 再学習: なし

## 主な生成物

Kaggle output: `kaggle/output/train_v1/artifacts/`

- `exp191_typewell_late_range_continuity_selector_on_exp176_metrics.csv`
- `exp191_typewell_late_range_continuity_selector_on_exp176_oof_predictions.csv.gz`
- `exp191_typewell_late_range_continuity_selector_on_exp176_selection_distribution.csv`
- `exp191_typewell_late_range_continuity_selector_on_exp176_by_well.csv`
- `exp191_typewell_late_range_continuity_selector_on_exp176_bucket_metrics.csv`
- `exp191_typewell_late_range_continuity_selector_on_exp176_viterbi_params.csv`
- `exp191_typewell_late_range_continuity_selector_on_exp176_score_summary.csv`
- `exp191_typewell_late_range_continuity_selector_on_exp176_summary.json`
