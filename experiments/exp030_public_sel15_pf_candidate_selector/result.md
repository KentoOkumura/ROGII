# exp030_public_sel15_pf_candidate_selector 結果

## 仮説

`exp029` の public sel15 PF/Beam OOF-like artifact には、public PF、beam、hold、保守的 blend の間で選択できる confidence signal が残っている可能性がある。

## 設定

- 親: `exp029_public_sel15_pf_oof_feature_generation`
- 検証: same-OOF diagnostics、leave-one-original-fold-out selection、stable well-hash holdout selection
- メトリック: RMSE
- シード: 42

## 結果

| メトリック | 値 |
| --- | --- |
| Raw public PF selector | 15.172636 |
| Best same-OOF candidate | 15.089532 (`pf090_hold010`) |
| Leave-one-original-fold-out candidate selection | 15.141132 |
| Well-hash holdout candidate selection | 15.131490 |
| Leave-one-original-fold-out bucket selection | 15.157679 |
| Well-hash holdout bucket selection | 15.183372 |
| Public LB | - |
| Private LB | - |

## 解釈

`pf_pred` 90% + `last_anchor_tvt` 10% の固定 blend が same-OOF で最良になり、raw public PF 15.172636 から 15.089532 へ改善した。fold 外の global candidate selection でも original-fold 15.141132、well-hash 15.131490 と raw を上回った。

一方、confidence fallback 系は悪化し、bucket-wise selection は original-fold では 15.157679 と小改善したが well-hash では 15.183372 で raw より悪化した。したがって hard row/bucket selector はまだ不安定で、次に inference 化するなら固定 90/10 hold blend か、ごく保守的な blend のみを候補にする。

## 次

`exp026_oof` は source artifact で未接続なので後続に回す。次は `pf090_hold010` を public sel15 inference flow に移植できるか確認し、Public LB 8.781 基準 に対して差分 range / SHA / submit-check を見てから提出判断する。
