# exp014_postprocess_cv_audit 結果

## 結論

`exp013` の distance bucket shrink は、同一 OOF fit の 13.501824 ほどではないが、
held-out alpha でも改善が残った。

- raw `lightgbm_no_gr`: 13.549257
- same-OOF bucket fit / exp013 fixed alpha: 13.501824
- leave-one-original-fold-out bucket fit: 13.535596
- well-bucket holdout fit: 13.510690

よって、`exp013` の Public LB 12.271 は引き続き有効な LB 基準 とする。
ただし今後の CV 記録では same-OOF 13.501824 を clean CV と呼ばず、
監査済み postprocess CV は 13.535596 として扱う。

## 設定

- 親: `exp013_model_diversity_or_postprocess`
- 入力: `data/external/kaggle-output/exp013_model_diversity_or_postprocess/train/artifacts/row_oof_predictions.csv`
- 対象 variant: `lightgbm_no_gr`
- 検証: original fold leave-one-out、stable well hash holdout
- メトリック: RMSE
- シード: 42 相当の exp013 OOF split。well hash holdout は deterministic hash。

## 結果

| Candidate | Score | Delta vs raw |
| --- | ---: | ---: |
| raw_lightgbm_no_gr | 13.549257 | 0.000000 |
| last_anchor | 15.909853 | 2.360595 |
| exp013_fixed_bucket_alphas | 13.501824 | -0.047434 |
| in_sample_bucket_refit | 13.501824 | -0.047434 |
| leave_one_original_fold_out_bucket_fit | 13.535596 | -0.013661 |
| well_bucket_holdout_fit | 13.510690 | -0.038567 |

## 解釈

近傍 bucket は 基準 が非常に強く、alpha 0.20 への shrink が安定して効く。
250 rows 以降は alpha 1.15 に張り付く bucket が多く、raw residual をやや強める方向が効いている。
ただし fold 4 の far bucket などでは raw より悪化するため、過度に自由な bucket tuning は避ける。

## 次

`public_pf_beam_scale_selector_features` に進み、PF / beam / hold blend route を fold-safe OOF feature として再現する。
