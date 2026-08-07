# exp039_single_xgboost_swap 結果

## 状態

Kaggle train version 1 completed.

## 実行

- Train audit: completed on Kaggle
- Inference: not applicable
- Submission: not submitted
- Static checks: `ruff` PASS, `py_compile` PASS, `validate_experiment` PASS
- Smoke: HGB override only, `/tmp/exp039_xgb_swap_smoke`, 5 wells / 11,207 rows
- Kaggle package: `experiments/exp039_single_xgboost_swap/kaggle/train`
- Kaggle kernel: `kentookumura/exp039-single-xgb-swap-train` v1
- Kaggle URL: `https://www.kaggle.com/code/kentookumura/exp039-single-xgb-swap-train`
- Kaggle id_no: `122273860`
- Output: `/tmp/kaggle-output/exp039_single_xgboost_swap/train_v1`
- Synced artifacts: `experiments/exp039_single_xgboost_swap/artifacts/single_xgboost_*`

## 結果

| Audit | Best overall | RMSE |
| --- | --- | ---: |
| original-fold | `pf090_hold010` | 15.089532 |
| well-hash | `pf090_hold010` | 15.089532 |

| Candidate | original-fold RMSE | well-hash RMSE | 解釈 |
| --- | ---: | ---: | --- |
| `base_geometry_bucket_shrink` | 19.075357 | 19.138113 | single-XGBoost base reference |
| `base_plus_pf_beam_diagnostics_bucket_shrink` | 16.029777 | 16.028160 | selected XGBoost feature candidate |
| `base_plus_pf_prediction_bucket_shrink` | 16.129598 | 16.095024 | PF prediction only; diagnostics 追加に負ける |
| `exp026_regenerated_bucket_shrink` | 16.483627 | 16.429613 | exp026-style regenerated 基準 on exp029 surface |
| `public_pf_selector` | 15.172636 | 15.172636 | XGBoost feature candidate より強い public PF control |
| `pf090_hold010` | 15.089532 | 15.089532 | overall best control |

Comparison:

- selected XGBoost vs `base_geometry_bucket_shrink`: -3.045580 original-fold / -3.109953 well-hash
- selected XGBoost vs exp038 selected single-LGBM `base_plus_pf_prediction_bucket_shrink`: +0.179630 original-fold / +0.207310 well-hash
- selected XGBoost vs `public_pf_selector`: +0.857141 original-fold / +0.855524 well-hash
- selected XGBoost vs `pf090_hold010`: +0.940245 original-fold / +0.938628 well-hash

## 評価予定

| Audit | 必須比較 |
| --- | --- |
| original-fold | `base_geometry_bucket_shrink`、`public_pf_selector`、`pf090_hold010`、`exp026_regenerated_bucket_shrink` |
| well-hash | `base_geometry_bucket_shrink`、`public_pf_selector`、`pf090_hold010`、`exp026_regenerated_bucket_shrink` |

## 解釈

XGBoost 化は base single-XGBoost reference からは大きく改善したが、exp038 の single-LGBM selected candidate と public PF controls を下回った。exp050 の pseudo-tail residual estimator swap で見えた小幅改善は Ravaghi/public sel15 feature surface には転移しなかった。

したがって `exp039_single_xgboost_swap` は inference port / submit に進めない。ML route Public LB 基準は `exp039_ravaghi_single_lgbm_inference_submit` 11.740 のまま維持する。
