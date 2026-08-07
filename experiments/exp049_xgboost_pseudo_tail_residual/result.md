# exp049_xgboost_pseudo_tail_residual 結果

## 状態

完了。Kaggle train notebook version 1 で full CV を実行した。

## 評価

- 主評価: well-level GroupKFold、valid well の本来の `TVT_input` NaN evaluation zone の RMSE。
- `xgboost_pseudo_tail_3_cutoffs_distance_balanced`: 12.839225
- `xgboost_pseudo_tail_3_cutoffs_distance_balanced_exp014_bucket_shrink_params`: 12.779452
- 比較基準:
  - `exp023` raw pseudo-tail LightGBM CV: 12.942938
  - `exp025/026` fixed bucket-shrink CV: 12.870780
  - `exp026` Public LB: 12.102

## Fold

- raw XGBoost: fold 0 12.342875、fold 1 12.534229、fold 2 12.049358、fold 3 12.573381、fold 4 14.572776
- fixed bucket-shrink: fold 0 12.343176、fold 1 12.511784、fold 2 11.934379、fold 3 12.513074、fold 4 14.473851

## Distance Bucket

Pooled fixed bucket-shrink RMSE:

- rows 0-49: 0.858545
- rows 50-249: 2.950028
- rows 250-999: 6.250212
- rows 1000-2499: 10.963121
- rows 2500+: 15.650705

## 解釈

XGBoost raw は LightGBM pseudo-tail raw 12.942938 から -0.103713 改善し、固定 bucket-shrink 後は `exp026` の 12.870780 から -0.091328 改善した。固定 bucket-shrink は raw XGBoost からさらに -0.059773 改善し、fold 1-4 では raw より良い。fold 0 はほぼ同等で、悪化は +0.000301 に留まる。

距離 bucket では固定 bucket-shrink により rows 0-49、250-999、1000-2499、2500+ が raw より改善し、rows 50-249 は 2.682015 から 2.950028 に悪化した。全体改善は主に near 0-49 と 1000+ rows の改善で支えられている。

結論として、`xgboost_pseudo_tail_3_cutoffs_distance_balanced_exp014_bucket_shrink_params` を ML route の通常 CV 基準に更新する。ただし Public LB は未確認で、exp021/026 と同じく CV 改善が LB に転移しない可能性がある。次は exp044 補助 fold / distance bucket の破壊的悪化確認、または同じ XGBoost 構成の inference port と submit-check へ進む。
