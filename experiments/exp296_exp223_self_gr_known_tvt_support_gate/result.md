# exp296_exp223_self_gr_known_tvt_support_gate 結果

## 状態

Kaggle private CPU version 3（id_no `127897387`）を完了した。technical guardは全PASS、performance guardはFAILで、事前登録どおりbranchを閉じた。inference、submissionは行っていない。

## 実行契約

- parent: `exp223_joint_typewell_self_gr_hmm_likelihood_probe`
- control: saved `hmm_selfgr_boost_only_a070_c100`
- variant: `hmm_selfgr_boost_only_a070_c100_known_tvt_support_gate`
- rows / wells: `3,783,989 / 773`
- active variant / HMM well-runs: `1 / 773`
- LightGBM config / trained fold / booster: `0 / 0 / 0`
- parent-control retraining / GPU: `0 / 0`
- runtime: `16,667.265 sec = 4.630 hours`

## Overall

| metric | exp223 saved control | exp296 strict gate | delta |
| --- | ---: | ---: | ---: |
| RMSE | 11.349942946 | 12.159749140 | +0.809806194 |
| MAE | 6.471268657 | 7.041758436 | +0.570489779 |
| within10 | 0.794840577 | 0.767777602 | -0.027062975 |

保存済みcontrolをtruth-late join後に再集計したRMSEは11.349942946であり、configに事前記録した11.349950650との差は約`7.7e-6 ft`である。判定には同一join上の再集計値を使った。

## Fold

| fold | control RMSE | exp296 RMSE | delta |
| ---: | ---: | ---: | ---: |
| 0 | 10.001164 | 10.966358 | +0.965195 |
| 1 | 11.479555 | 12.010424 | +0.530868 |
| 2 | 13.754857 | 14.392580 | +0.637723 |
| 3 | 12.294011 | 12.035704 | -0.258307 |
| 4 | 9.148242 | 11.437966 | +2.289724 |

改善は1/5 foldsだった。

## Scopeとtail safety

| scope | rows | control RMSE | exp296 RMSE | delta |
| --- | ---: | ---: | ---: | ---: |
| true TVT inside known range | 2,324,458 | 10.044092 | 9.472290 | -0.571802 |
| true TVT outside known range | 1,459,531 | 13.164897 | 15.506322 | +2.341425 |
| distance 1000+ | 3,012,442 | 12.455457 | 13.352948 | +0.897491 |
| hidden-like spatial | 972,463 | 12.463402 | 13.574215 | +1.110813 |
| hidden-like typewell-purged | 976,449 | 12.266317 | 13.384951 | +1.118634 |

近距離を含む全distance bucketも悪化し、deltaは0-50 / 50-100 / 100-250 / 250-500 / 500-1000で`+0.011925 / +0.020665 / +0.034828 / +0.104991 / +0.410558 ft`だった。

302 wellsが改善、471 wellsが悪化した。by-well p95 deltaは`+1.728087 ft`。worst `2364716c`はRMSE `4.661378 -> 44.349169`、delta `+39.687791 ft`だった。best `028d7b28`は`-19.815744 ft`であり、効果は強く不均一だった。

## Hard gate

technical 12/12はPASSした。入力773 wells、finite coverage 1.0、saved control row identity/SHA、support外contribution exact 0、support内boost delta exact 0、base/self-GR config parity、truth-before-freeze 0、control再学習0、LightGBM/fold/booster 0を確認した。

performanceは10項目中2項目のみPASSした。PASSはinside-range delta `-0.571802 <= +0.02`とstep p99 delta `-0.000266 <= 0`。pooled、4/5 folds、outside-range、1000+、hidden-like 2面、by-well p95、worst-wellの8項目はFAILした。

## 再現性とartifact監査

- canonical kernel: `kentookumura/exp296-exp223-self-gr-known-tvt-support-gate-train`
- saved control decompressed SHA: `0eb48b5516276b0ab7b2191a52a39ebb89d9997363cc7839ede519c7863baa0c`
- prediction decompressed SHA: `e87f1c64a870991b65f310891b316e2854f6c717947df923d60ab2f73c5ac99a`
- prediction raw gzip SHA: `e8aabd98ed0d7b675b2d8f20d793129b02ada3c7a571a8706a72099b2bb07261`
- support manifest SHA: `b537eb37a81155da031f46b8472a848e4bfed257e845a745b0f4890e8383b209`
- output summary download SHA: `74701c7642c86d3e9b019f46b11491c12d00eacec74b7a3a42b013fa13ffc4fc`
- kernel log SHA: `8efb5e7a8065e0caafaa648fb41d50784d0110ee880805943a63f77031c581a0`

Kaggle outputからmetrics/manifest/schemaの小規模13 artifactだけを取得し、summaryに記録された13 SHAと全件一致した。大容量prediction freeze / OOF readoutは、summary内のraw/decompressed SHAを記録し、ローカルには取得していない。

## 結論

strict known-TVT support gateは棄却する。range内rowでは有効だが、range外・1000+・hidden-like・worst-wellへの損失が大きく、candidate stateがprefix known range外という条件だけでself-GRを参照しない設計は安全でない。

契約どおりsupport padding、soft/hole-aware gate、alpha/clip/window/top-k/threshold救済へ進まず、inference/submissionも行わない。target-freeなself-GR quality/support-riskをadd-only ML featureとして使う案は、hard gateと分離した既存低優先backlogへ証拠だけを反映する。
