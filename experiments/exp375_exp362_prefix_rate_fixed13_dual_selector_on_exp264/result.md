# exp375_exp362_prefix_rate_fixed13_dual_selector_on_exp264 結果

## 結論

`prefix_rate_exact_hmm`にはoracle上の補完性があるが、現行dual selectorでは
安全に局在化できなかった。technical / leakage / selector score guardは通過した一方、
parent fixed12に対してpooled、5/5 folds、near、1000+、well tailを悪化させたため、
decisionは`FAIL_CLOSE_FIXED13_SELECTOR_BRANCH`とする。

downstream TVT、current-test候補生成、inference、submissionへ進めない。

## 設定

- 親: `exp264_exp263_candidate_confidence_dual_selector`
- 候補元: `exp362_segment_local_donor_slope_exact_hmm`
- 追加候補: `prefix_rate_exact_hmm`
- 検証: exp263 selector outer 5 × inner 4 nested stacking
- 学習量: 1 variant / 2 objectives / 40 CPU selector models
- control再学習 / GPU / downstream TVT: 0 / 0 / 0
- Kaggle kernel: `kentookumura/exp375-exp362-prefix-fixed13-selector-train`
- version / id_no: `1 / 128436686`
- runtime: `6978.658914 sec`

## 主結果

| 指標 | fixed13 | parent fixed12 | 差分 fixed13 - parent |
| --- | ---: | ---: | ---: |
| pooled hard OOF RMSE | 8.787855710 | 8.652531956 | +0.135323754 |
| near 0--250 RMSE | 1.700791894 | 1.663644827 | +0.037147066 |
| distance 1000+ RMSE | 9.652429166 | 9.503798844 | +0.148630322 |
| hidden-like spatial | 9.456282303 | 9.536496454 | -0.080214151 |
| hidden-like typewell-purged | 9.340994256 | 9.412065207 | -0.071070951 |

fixed fallbackは変更されず`8.238331546`、error parity最大差は`0.0 ft`だった。

### Fold別 hard OOF

| Fold | fixed13 | parent fixed12 | 差分 |
| ---: | ---: | ---: | ---: |
| 0 | 9.443650200 | 8.991720025 | +0.451930174 |
| 1 | 8.540884371 | 8.426634925 | +0.114249446 |
| 2 | 8.963983300 | 8.900503341 | +0.063479959 |
| 3 | 8.426945436 | 8.426473003 | +0.000472433 |
| 4 | 8.521858836 | 8.499558672 | +0.022300164 |

改善foldは`0/5`。

### Selectorとwell safety

- 追加候補top1: `436,138 / 3,783,989 rows = 11.525879%`
- positive usage fold: `5/5`
- improved / worsened wells: `393 / 380`
- by-well delta p95: `+1.047744567 ft`
- worst well: `b19b0395`, `+28.995116411 ft`
- worst wellでの追加候補top1率: `0.258042%`
- usage-delta Pearson / Spearman: `-0.024414 / -0.041055`

追加候補は十分使われたが、使用率と安全性の関係はほぼなかった。worst wellでは
追加候補自体のtop1率が低いのに親から大きく悪化しており、追加候補の直接選択だけでなく、
13候補でselectorを再学習したことによる既存候補のreranking不安定性が疑われる。
同じ`b19b0395`は独立したexp373 fixed13でも`+29.062586652 ft`のworstだった。

## Selector score / leakage

- 40 models、25 compact partitions、18,919,945 compact rowsを完了。
- outer-valid candidate-score rows: `49,191,857`
- expected-error MAE: `3.807291329` vs prior `5.813038460`
- within10 logloss: `0.356357163` vs prior `0.508900257`
- within10 Brier: `0.110892820` vs prior `0.164483282`
- 3 score指標はpooledと5/5 foldsでpriorを改善。
- outer-valid wellのinner assignment除外、inner train/valid well非重複、
  outer-train inner OOF、outer-valid 4-inner ensembleをすべて確認。
- exp362 truth/error列のfeature-freeze前loadは0、global key join missingは0、
  source foldはmodel featureに未使用、native confidence finite率は1.0。

したがって結果は実装失敗やleakageではなく、score calibrationが改善してもhard pathの
parent比較とwell safetyを保証しないscientific negative resultとして扱う。

## Post-freeze add-one novelty診断

この診断はselector prediction freeze後に計算し、学習とscientific gateへ戻していない。

| 粒度 | fixed12 oracle | fixed13 add-one oracle | headroom | strict unique-best |
| --- | ---: | ---: | ---: | ---: |
| H512 | 3.700319996 | 3.537643353 | 0.162676643 ft | 1236 / 7787 = 15.872608% |
| whole well | 4.801786361 | 4.678573028 | 0.123213333 ft | 130 / 773 = 16.817594% |

候補の補完性は実在する。しかしoracle headroomはdeployable selector性能ではなく、
今回のhard OOF悪化を上書きする根拠にはしない。

## Scientific gate

PASS:

- selector score guard
- 追加候補pooled usage
- 追加候補5-fold usage
- hidden-like 2面

FAIL:

- pooled parent非悪化
- parent比改善fold数
- near非悪化
- 1000+非悪化
- by-well p95非悪化
- worst-well非悪化

総合gateはFAIL。

## 再現性

- deterministic anchor: false
- 理由: exp362 source候補にrerun parityがなく、exp375もrerunしていない
- exp362 decompressed SHA:
  `e1d672ff9743b92c33a40bec8d4cf3b0a8c29cdbbb37948992f0809522e3e7ef`
- exp362 post-read prediction content SHA:
  `fa23301c5b3da1a9846630009e327016b5f131dc1ac370e0c2fa94f9b0561095`
- feature schema SHA:
  `465eee3b936bca4acafbe4c9010e6f744d0b5219c2d0ebeafb6199bdc11c4faf`
- selector model manifest SHA:
  `03e0277c62c2d315fe5000c9538095449ec73eff4dba71d6a4c311201b1cfbba`
- compact manifest SHA:
  `06a8fbd204c9df14499d0f47f63902faa8c241d1171e71b88cf3b1fd35f36f62`
- outer-valid candidate score SHA:
  `6086a1f1de211a43712cae893669dabbf1958b01edc8a5047d81724db725d67a`
- summary SHA:
  `1e503ff962e7707662baf27d3dab19924ca31fbb24bb5c969dd7170fcdf3f318`
- submission SHA: 対象外

## 次

同一OOF上のweight、threshold、domain、gate救済は行わず、このfixed13 branchを閉じる。
原因確認が必要になった場合だけ、exp264 / exp371 / exp373 / exp375の保存済み
candidate scoreを使う0-boosterの`fixed13_selector_incumbent_reranking_instability_readout`
を低優先で検討する。追加候補がtop1でない行に限定した既存候補choiceの変化を
truth join前にfreezeし、同じworst-tailが候補横断で再現するかだけを診断する。
