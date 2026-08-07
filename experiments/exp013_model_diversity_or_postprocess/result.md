# exp013_model_diversity_or_postprocess 結果

## 結論

Kaggle full CV は完了。raw `lightgbm_no_gr` は exp012 基準 と同じ CV 13.549257 を再現した。

OOF postprocess の最良は `distance_bucket_shrink_fit` で OOF-fit score 13.501824。raw から 0.047433 改善したため、`config.yaml` の `postprocess.selected_method` を `distance_bucket_shrink` に固定した。ただし bucket alpha は同じ OOF rows の正解残差で fit しているため、13.501824 は nested / held-out CV ではない。clean CV 基準は raw `lightgbm_no_gr` 13.549257 として扱う。

HGB/LightGBM blend は 0.90/0.80/0.70 の全候補で raw より悪化したため採用しない。

Inference / submit-check / submit も完了。Public LB は 12.271 で、従来 基準 `exp012` の 12.320 を更新した。

## Kaggle Train

- Kernel: `kentookumura/exp013-model-diversity-or-postprocess-train`
- Version: 1
- Status: COMPLETE
- Runtime from log: about 680 seconds to metrics write
- Output: `/tmp/kaggle-output/exp013_model_diversity_or_postprocess/train`

## Kaggle Inference / Submission

- Kernel: `kentookumura/exp013-distance-bucket-shrink-inference`
- Version: 1
- Status: COMPLETE
- Submit-check: PASS
- Submission ref: `53363702`
- Public LB: 12.271
- Private LB: -
- Archived submission: `data/external/kaggle-output/exp013_model_diversity_or_postprocess/inference/submission.csv`
- SHA256: `689a1cefccddeb1ed0407c461695bcec7e81bc3c760cab1b2a80c97391b95843`

## CV / OOF-Fit Postprocess

| Candidate | Method | Score | Delta vs raw |
| --- | --- | ---: | ---: |
| `distance_bucket_shrink_fit` | `distance_bucket_shrink` | 13.501824 | -0.047433 |
| `near_基準_damping_100_0.30` | `near_基準_damping` | 13.545208 | -0.004049 |
| `near_基準_damping_50_0.20` | `near_基準_damping` | 13.546807 | -0.002450 |
| `sg_smooth_w31_b0.50` | `sg_smooth` | 13.547621 | -0.001636 |
| `sg_smooth_w15_b0.50` | `sg_smooth` | 13.547809 | -0.001448 |
| `raw_lightgbm_no_gr` | `raw` | 13.549257 | 0.000000 |

## Distance Buckets

| Bucket | Rows | Alpha | Last 基準 RMSE | Raw RMSE | Bucket shrink RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| rows_0_49 | 38,650 | 0.200000 | 0.960110 | 3.231596 | 0.941663 |
| rows_50_249 | 154,600 | 0.591718 | 4.184865 | 3.829747 | 3.475713 |
| rows_250_999 | 579,068 | 1.150000 | 10.046803 | 7.197536 | 7.127516 |
| rows_1000_2499 | 1,151,925 | 1.150000 | 14.967594 | 11.870592 | 11.752972 |
| rows_2500_plus | 1,859,746 | 1.049077 | 18.529919 | 16.391606 | 16.386609 |

## Artifacts

- `artifacts/ablation_metrics.csv`
- `artifacts/fold_metrics.csv`
- `artifacts/fold_model_training.csv`
- `artifacts/model_group_summary.csv`
- `artifacts/postprocess_metrics.csv`
- `artifacts/postprocess_distance_bucket_summary.csv`
- `artifacts/postprocess_selected_params.json`
- `artifacts/well_metrics.csv`
- `artifacts/exp013-model-diversity-or-postprocess-train.log`
- `artifacts/inference_well_summaries.csv`
- `artifacts/exp013-distance-bucket-shrink-inference.log`
- `metrics.json`

`row_oof_predictions.csv` は 1.1GB のため実験ディレクトリには常設せず、`data/external/kaggle-output/exp013_model_diversity_or_postprocess/train/artifacts/row_oof_predictions.csv` に残した。
