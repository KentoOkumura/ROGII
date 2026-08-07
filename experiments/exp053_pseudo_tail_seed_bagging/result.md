# exp053_pseudo_tail_seed_bagging 結果

## 状態

完了。Kaggle train notebook version 1 で full CV を実行した。

## 評価

- 主評価: well-level GroupKFold、valid well の本来の `TVT_input` NaN evaluation zone の RMSE。
- `lgbm_capacity_seed_bag3_exp014_bucket_shrink_params`: 12.633797
- `lgbm_capacity_seed_bag3`: 12.715910
- same-run `lgbm_capacity_single_seed_control_exp014_bucket_shrink_params`: 12.734551
- same-run `lgbm_capacity_single_seed_control`: 12.800238
- 比較基準:
  - `exp051` best fixed bucket-shrink CV: 12.634392
  - `exp052` pseudo-tail 自前系 Public LB: 12.076

## Fold

- fixed seed bag3: fold 0 12.394983、fold 1 11.966605、fold 2 11.431497、fold 3 12.370181、fold 4 14.793445
- fixed single seed control: fold 0 12.512253、fold 1 12.064434、fold 2 11.370228、fold 3 12.578912、fold 4 14.920249

## Distance Bucket

Pooled fixed seed bag3 RMSE:

- rows 0-49: 0.818646
- rows 50-249: 2.917066
- rows 250-999: 6.023023
- rows 1000-2499: 10.820044
- rows 2500+: 15.499329

Pooled fixed single seed control RMSE:

- rows 0-49: 0.827229
- rows 50-249: 2.931081
- rows 250-999: 6.077660
- rows 1000-2499: 10.853090
- rows 2500+: 15.645229

## 解釈

3-seed bagging は same-run single seed control に対して全体で -0.100754 改善し、fold 0/1/3/4 と全 distance bucket で改善した。一方、直近の exp051 best 12.634392 に対する改善は -0.000595 とごく小さい。exp053 は「single seed の不安定さを平均化できる」証拠にはなるが、新しい通常 CV 基準としては exp051 と実質同等と扱う。

推論 port は、提出枠や実行時間に余裕がある場合の低優先候補に留める。次に通常 CV を伸ばすなら、seed 数を増やすよりも cutoff distribution / distance balancing / target scaling のような構造的変更を優先する。
