# exp020_distance_weighted_training_audit 結果

## 状態

完了。Kaggle train notebook v1 で full CV を実行した。

## 評価

- Raw clean CV 基準: `exp013 lightgbm_no_gr` 13.549257
- Held-out postprocess reference: `exp014` 13.535596
- Best training variant: `near_down_far_up_lightgbm` 13.470015
- Other variants:
  - `control_lightgbm_no_gr`: 13.549257
  - `far_upweight_lightgbm`: 13.550841
  - `near_downweight_lightgbm`: 13.580536
  - `near_mid_far_segmented_lightgbm`: 13.655239

## 解釈

`near_down_far_up_lightgbm` は raw 基準 から -0.079242、`exp014` held-out postprocess から -0.065581 改善した。rows 0-249 は悪化したが、rows 1000-2499 と 2500+ の改善が全体を押し下げたため、near-row を捨てるだけの過剰最適化ではない。

ただし今回の実験は train-side CV だけで、inference submission は作っていない。次は selected weight profile を inference notebook に反映し、raw / exp014 bucket shrink / selected weighted model / weighted+postprocess の提出候補を同一設定で作る。
