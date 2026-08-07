# exp050_xgboost_pseudo_tail_inference_submit 結果

## 状態

Kaggle inference version 1 完了。`submission.csv` を生成し、submit-check は PASS。code submit ref `53521999` は complete、Public LB は 12.083。

## 設定

- 学習: exp049 XGBoost pseudo-tail + distance-balanced residual model
- 後処理: exp025-selected fixed `exp014_bucket_shrink_params`
- CV reference: exp049 12.779452
- Public LB reference: exp026 12.102
- 全体 / PF route reference: exp027 8.781

## Output

- Kaggle kernel: `kentookumura/exp050-xgb-pseudotail-infer`
- output: `/tmp/kaggle-output/exp050_xgboost_pseudo_tail_inference_submit/inference_v1`
- submission ref: `53521999`
- Public LB: 12.083
- submission rows: 14,151
- missing values: 0
- duplicate ids: 0
- prediction range: 11587.960181 - 12234.905349
- prediction mean/std: 11907.057784 / 278.953551
- submit-check: PASS

## exp026 差分

Against `/tmp/kaggle-output/exp026_pseudo_tail_bucket_shrink_inference_submit/inference/submission.csv`:

- changed rows: 14,151
- diff min/max: -4.050819 / 3.799731
- diff mean: -0.244825
- diff absolute mean: 1.100087
- diff RMSE: 1.431860
- correlation: 0.999991802

## 解釈

XGBoost inference output は exp026 LightGBM fixed bucket-shrink と高相関だが、全行が変化し、差分 RMSE は 1.431860 と十分に実質的な変更になっている。予測範囲は exp026 の 11590.725143 - 12237.368348 よりやや低い側へ広がり、最大側はやや低い。

形式面は submit-check PASS。Public LB 12.083 は exp026 fixed bucket-shrink 12.102 から -0.019 改善し、旧自前 pseudo-tail 系では最良を更新した。一方、ML route Public LB 基準の exp039 11.740 には +0.343、全体 / PF route 基準の exp027 8.781 には +3.302 届いていないため、全体基準は更新しない。
