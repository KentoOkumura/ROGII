# exp052_lgbm_capacity_pseudotail_inference_submit 結果

## 状態

Kaggle inference version 1 完了。`submission.csv` を生成し、submit-check は PASS。code submit ref `53524340` は complete、Public LB は 12.076。

## 評価

- 学習: exp051 selected LightGBM capacity pseudo-tail + distance-balanced residual model
- 後処理: exp025 selected fixed `exp014_bucket_shrink_params`
- CV reference: exp051 12.634392
- Kaggle kernel: `kentookumura/exp052-lgbm-cap-pseudotail-infer`
- output: `/tmp/kaggle-output/exp052_lgbm_capacity_pseudotail_inference_submit/inference_v1`
- submission ref: `53524340`
- Public LB: 12.076

## Submission Diagnostics

- rows: 14,151
- missing values: 0
- duplicate ids: 0
- SHA256: `657ca475d9ff8abfa7a1f482473b47815a2c9001803ae4b2b63c5074585d992b`
- prediction range: 11587.429983 - 12236.572595
- prediction mean/std: 11905.842985 / 279.169431

Diff vs exp026 LightGBM fixed bucket-shrink submission:

- min: -6.363842
- max: 2.009570
- mean: -1.459623
- abs mean: 1.724878
- RMSE: 2.225454
- corr: 0.999984451265

Diff vs exp050 XGBoost fixed bucket-shrink submission:

- min: -5.605740
- max: 3.099844
- mean: -1.214799
- abs mean: 1.656530
- RMSE: 2.011376
- corr: 0.999983797988

## 解釈

exp052 は pseudo-tail 自前系の Public LB を exp050 12.083 から 12.076 へ -0.007 改善した。exp026 12.102 からは -0.026 改善。CV では exp051 が exp049 から -0.145060 改善していたが、Public LB への転移は小さい。

予測は exp026/exp050 と高相関だが、差分 RMSE は exp026 比 2.225454、exp050 比 2.011376 で実質的な変更になっている。予測範囲は exp050 よりやや広く、mean は低い。

結論として、pseudo-tail 自前系の Public LB 基準は exp052 の 12.076 に更新する。ただし ML route 全体基準 exp039 11.740 と、全体 / PF route 基準 exp027 8.781 には届かないため、全体基準は更新しない。
