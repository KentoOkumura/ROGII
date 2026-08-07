# exp023_pseudo_tail_distance_augmentation 結果

## 状態

完了。Kaggle train notebook version 1 で full CV を実行した。

## 評価

- Raw clean CV 基準: `exp013 lightgbm_no_gr` 13.549257
- Distance-weighted training reference: `exp020 near_down_far_up_lightgbm` 13.470015
- Distance-weighted postprocess reference: `exp021 weighted_distance_bucket_shrink` 13.415799
- Best training variant: `pseudo_tail_3_cutoffs_distance_balanced` 12.942938
- Other variants:
  - `pseudo_tail_1_cutoff`: 12.971839
  - `pseudo_tail_3_cutoffs`: 13.012302
  - `distance_balanced_sampling`: 13.441648
  - `control_lightgbm_no_gr`: 13.494554

## 解釈

`pseudo_tail_3_cutoffs_distance_balanced` は raw 基準 から -0.606319、`exp021` weighted bucket shrink reference から -0.472861 改善した。pseudo-tail 系は fold 1-3 で特に大きく改善し、fold 4 でも control 15.279820 に対して 14.993852 まで改善した。

距離 bucket 平均では best variant が rows 0-49、50-249、250-999、1000-2499、2500+ の全域で control より改善している。特に near / mid の改善が大きく、far も control より下がっているため、単なる near-row overfit ではない。

この実験は train-side CV のみで、inference submission は作っていない。次は best variant を inference 化し、raw pseudo-tail prediction と exp014/exp021 型の bucket shrink を同一 OOF で監査してから提出候補を決める。
