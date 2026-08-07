# exp433_rsd_sparse_anchor_direct_oof_readout 結果

## 状態

Kaggle private CPU version 3で完了。technical gateは全PASSしたが、
scientific gateは全9条件をFAILしたため、疎なRSD anchor branchを
same-OOF救済なしで閉鎖した。

## 実行

- kernel:
  `kentookumura/exp433-rsd-sparse-anchor-direct-oof-readout-train`
- version / id_no:
  `3 / 128939253`
- route:
  `pf_beam`
- runtime / peak RSS:
  `122.701148 sec / 2.708778 GB`
- GPU / internet:
  無効 / 無効
- 実行量:
  1 decoder / 1 diagnostic / 773 wells / 5 reporting folds
- model / booster / HMM / PF / Beam / GPU / parent再生成:
  すべて0

version 1はCSV保存前floatのproducer logical SHAを再読込後に照合した
非round-trip-safeな契約で停止し、version 2は予約scope `fold`の
metrics routingで停止した。いずれも科学予測の失敗とは区別して修正し、
version 3で全処理を完走した。

## primary結果

| メトリック | exp226 base | exp433 primary | gain |
| --- | ---: | ---: | ---: |
| pooled OOF RMSE | 9.427110 | 9.692148 | -0.265039 ft |
| distance 1000+ | 10.331435 | 10.629970 | -0.298535 ft |
| persistent episode | 19.588114 | 19.860192 | -0.272078 ft |
| raw GR missing | 8.858973 | 9.183398 | -0.324424 ft |
| hidden-like spatial | 9.399891 | 9.666254 | -0.266363 ft |
| hidden-like typewell-purged | 9.266907 | 9.525655 | -0.258748 ft |

全5 foldsで悪化した。

| fold | base RMSE | primary RMSE | gain |
| ---: | ---: | ---: | ---: |
| 0 | 9.456630 | 9.644632 | -0.188002 |
| 1 | 9.131959 | 9.537400 | -0.405441 |
| 2 | 10.284827 | 10.635823 | -0.350996 |
| 3 | 9.187307 | 9.443437 | -0.256130 |
| 4 | 9.046482 | 9.161810 | -0.115329 |

near領域だけは0--50 / 50--100 / 100--500でそれぞれ
`+0.010300 / +0.026029 / +0.015655 ft`改善したが、500+の悪化が上回った。

## mechanism / tail

- supported blocks:
  `1,993 / 7,787 = 25.593939%`
- supported wells:
  `690 / 773`
- persistent episode SSE reduction:
  `-2.797279%`
- persistent episode wells improved:
  `160 / 449 = 35.634744%`
- corrected total SSEに対するnew episode SSE:
  `5.228087%`（上限5%をFAIL）
- improved wells:
  `281 / 773 = 36.351876%`
- by-well delta RMSE p95:
  `+3.282839 ft`
- worst well:
  `fb3848a1`, `+15.926322 ft`
- best well:
  `5bd25f59`, `-7.394790 ft`

report-only blockwise top-1はRMSE `18.367839`で、exact `25.288510%`、
top-3 coverage `47.466131%`だった。supported nonzero-oracle blockでの
direction accuracyはblockwise `51.840299%`、Viterbi `44.479102%`。

## gate

technical checksはすべてPASSした。

- input SHA / inventory
- parent RMSE parity
- prediction freezeとindependent full / probe rerun SHA
- truth / hidden / episodeのfreeze前read 0
- correction slope
- runtime / peak memory

scientific checksは9件すべてFAILした。

- pooled gain
- improvement folds
- distance 1000+ gain
- persistent episode SSE reduction
- persistent episode well improvement fraction
- guarded scope non-regression
- new episode SSE fraction
- by-well p95
- worst-well regression

## 再現性

- prediction logical SHA:
  `c461a14708ffc951060a77e0016a7947f7e2cae1abeb28b539465c0289100377`
- datum path logical SHA:
  `e3b4f9afbe0f431c5f80add93f11abb15af44dbae64fd9511be579e2d8bef96e`
- fixed probe prediction SHA:
  `639fb28ff2397123b24d44fe3aaaa56570aa0840412f541072260a9f7af46b9a`
- independent full / probe rerun:
  一致
- truth read:
  freeze前0行、freeze後3,783,989行

## 結論

exp426のcoverage FAILを解除して実OOFへ直接適用しても、固定Viterbiは
exp226の累積offsetを改善せず、pooled、全fold、long-tail、persistent episode、
hidden-like、by-well tailを一貫して悪化させた。疎なRSD scoreの
unsupported carryだけではabsolute re-anchorとして不十分である。

事前登録どおりdecoder、transition、support、activation、clip、blend、
well gateを同じOOFで救済しない。exp426のtechnical FAILも維持し、
inference / submissionへ進まない。
