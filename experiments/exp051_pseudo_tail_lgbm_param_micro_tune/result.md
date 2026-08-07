# exp051_pseudo_tail_lgbm_param_micro_tune 結果

## 状態

完了。Kaggle train notebook version 1 で full CV を実行した。

## 評価

- 主評価: well-level GroupKFold、valid well の本来の `TVT_input` NaN evaluation zone の RMSE。
- `lgbm_capacity_leaves47_minchild60`: 12.706752
- `lgbm_capacity_leaves47_minchild60_exp014_bucket_shrink_params`: 12.634392
- same-run control `lgbm_control_pseudo_tail_3_cutoffs_distance_balanced_exp014_bucket_shrink_params`: 12.784540
- 比較基準:
  - `exp023` raw pseudo-tail LightGBM CV: 12.942938
  - `exp025/026` fixed bucket-shrink CV: 12.870780
  - `exp049` XGBoost fixed bucket-shrink CV: 12.779452
  - `exp026` Public LB: 12.102
  - `exp050` XGBoost pseudo-tail Public LB: 12.083

## Variant Summary

- `lgbm_capacity_leaves47_minchild60_exp014_bucket_shrink_params`: 12.634392
- `lgbm_capacity_leaves47_minchild60`: 12.706752
- `lgbm_control_pseudo_tail_3_cutoffs_distance_balanced_exp014_bucket_shrink_params`: 12.784540
- `lgbm_rowcap700_perwell_exp014_bucket_shrink_params`: 12.827228
- `lgbm_subsample080_colsample085_exp014_bucket_shrink_params`: 12.853042
- `lgbm_control_pseudo_tail_3_cutoffs_distance_balanced`: 12.858849
- `lgbm_rowcap700_perwell`: 12.899172
- `lgbm_subsample080_colsample085`: 12.923176
- `lgbm_rowcap900_perwell_exp014_bucket_shrink_params`: 12.938629
- `lgbm_reglambda050_exp014_bucket_shrink_params`: 12.951170
- `lgbm_rowcap900_perwell`: 13.005018
- `lgbm_reglambda050`: 13.016951
- `lgbm_regularized_leaves23_minchild120_exp014_bucket_shrink_params`: 13.022204
- `lgbm_regularized_leaves23_minchild120`: 13.096797

## Fold

- raw capacity: fold 0 12.645012、fold 1 11.992765、fold 2 11.377316、fold 3 12.344114、fold 4 14.937566
- fixed bucket-shrink capacity: fold 0 12.661040、fold 1 11.911381、fold 2 11.206008、fold 3 12.266242、fold 4 14.876054
- fixed bucket-shrink control: fold 0 12.394506、fold 1 12.159905、fold 2 11.462292、fold 3 12.726376、fold 4 14.959103

## Distance Bucket

Pooled fixed bucket-shrink capacity RMSE:

- rows 0-49: 0.832508
- rows 50-249: 2.964540
- rows 250-999: 6.078927
- rows 1000-2499: 10.819565
- rows 2500+: 15.492962

Same-run fixed bucket-shrink control RMSE:

- rows 0-49: 0.815346
- rows 50-249: 2.945267
- rows 250-999: 6.223707
- rows 1000-2499: 11.051794
- rows 2500+: 15.623918

## 解釈

`num_leaves=47` / `min_child_samples=60` の capacity 増加が最良で、fixed bucket-shrink 後 12.634392。same-run control 12.784540 から -0.150148、従来の exp026 fixed bucket-shrink 12.870780 から -0.236388、exp049 XGBoost fixed bucket-shrink 12.779452 から -0.145060 改善した。

改善は fold 1-4 で control より良く、fold 0 では control より悪い。距離 bucket では rows 250-999、1000-2499、2500+ が control より改善し、rows 0-49 と 50-249 はわずかに悪化した。全体改善は中距離から遠距離の改善が支えている。

same-run control 自体も過去 exp026 より良いため、絶対値だけでなく同一実行内 control との差分を重視する。Public LB は未確認で、exp021/026/050 と同様に CV 改善が LB へ転移しない可能性がある。次は選択候補だけを inference port し、submit-check、予測範囲、exp026/exp050 submission との差分を確認する。
