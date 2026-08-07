# exp012_single_catboost_lightgbm_residual 結果

## 結論

Kaggle full CV では `lightgbm_no_gr` が best。CV は 13.549257 で、`control_hgb_no_gr` の 13.882944 から 0.333687 改善した。`exp002` all-GR public 基準 control からは 0.575312 改善。

Inference / submit-check / submit も完了。Public LB は 12.320 で、従来の Public LB 基準 `exp002` 12.533 を更新した。

## Kaggle Train

- Kernel: `kentookumura/exp012-single-catboost-lightgbm-residual-train`
- Version: 1
- Status: COMPLETE
- Runtime from log: about 1,559 seconds
- Output: `/tmp/kaggle-output/exp012_single_catboost_lightgbm_residual/train`

## Kaggle Inference / Submission

- Kernel: `kentookumura/exp012-lightgbm-no-gr-inference`
- Version: 1
- Status: COMPLETE
- Submit-check: PASS
- Submission ref: `53330920`
- Public LB: 12.320
- Private LB: -
- Archived submission: `data/external/kaggle-output/exp012_single_catboost_lightgbm_residual/inference/submission.csv`

## CV

| Variant | Estimator | Feature set | CV | delta vs exp003 |
| --- | --- | --- | ---: | ---: |
| `lightgbm_no_gr` | `LGBMRegressor` | `no_gr_signal` | 13.549257 | -0.333687 |
| `lightgbm_all` | `LGBMRegressor` | `all` | 13.747339 | -0.135605 |
| `catboost_no_gr` | `CatBoostRegressor` | `no_gr_signal` | 13.850921 | -0.032023 |
| `control_hgb_no_gr` | `HistGradientBoostingRegressor` | `no_gr_signal` | 13.882944 | 0.000000 |
| `catboost_all` | `CatBoostRegressor` | `all` | 14.094769 | +0.211825 |
| `control_hgb_all` | `HistGradientBoostingRegressor` | `all` | 14.124569 | +0.241625 |

## Group Notes

`model_group_summary.csv` shows `lightgbm_no_gr` improves the all-well mean RMSE versus `control_hgb_no_gr` from 11.144215 to 10.901288. It also improves high-GR-missing and long-eval groups. The public-like keep-all-GR group remains worse than `control_hgb_all` but improves versus `control_hgb_no_gr` from 11.552606 to 11.120797.

## Artifacts

- `artifacts/ablation_metrics.csv`
- `artifacts/fold_metrics.csv`
- `artifacts/fold_model_training.csv`
- `artifacts/model_group_summary.csv`
- `artifacts/well_metrics.csv`
- `artifacts/exp012-single-catboost-lightgbm-residual-train.log`
- `artifacts/inference_well_summaries.csv`
- `artifacts/exp012-lightgbm-no-gr-inference.log`
