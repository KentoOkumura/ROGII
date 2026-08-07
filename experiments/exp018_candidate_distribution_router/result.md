# exp018_candidate_distribution_router 結果

## 仮説

PF/beam と DTW/DWT は add-only features では悪化したが、候補予測としては
距離帯や候補間 disagreement が良い行だけに限定すれば raw
`lightgbm_no_gr` を補える可能性がある。

## 設定

- 親: `exp013_model_diversity_or_postprocess`
- 検証: 既存 OOF の same-OOF 比較、leave-one-original-fold-out selection audit、stable well-hash holdout selection audit
- メトリック: RMSE
- raw 基準: `lightgbm_no_gr` 13.549257

## 結果

| メトリック | 値 |
| --- | --- |
| Raw clean CV | 13.549257 |
| Best same-OOF router | 13.537122 (`disagreement_damped_raw`) |
| Bucket oracle same-OOF | 13.545073 |
| Leave-one-original-fold-out selection | 13.644470 |
| Well-hash holdout selection | 13.646503 |
| Public LB | - |
| Private LB | - |

## 解釈

Same-OOF では `disagreement_damped_raw` が 62,757 rows を damping し、
raw 13.549257 から 13.537122 へ改善した。ただし fold 外 router selection は
13.644470 / 13.646503 で raw より大きく悪い。特に
`dtw_lightgbm_equal_blend` が fold 外で大きく外れるため、候補 routing の選択は
安定していない。

Bucket oracle も 13.545073 で改善は 0.004184 に留まり、250 rows 以降は raw
LightGBM を選ぶ。候補分布には同一 OOF の小さい改善余地はあるが、clean CV で
提出実装へ進めるだけの根拠はない。

## 次

この router は提出実装へ進めない。PF/beam row OOF が復元された場合だけ、
候補品質監査として同じ script に含めて再評価する。
