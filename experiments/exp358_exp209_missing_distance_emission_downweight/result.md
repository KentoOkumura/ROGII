# exp358 結果

## 状態

Kaggle private CPU train version 2（id_no `128528105`）でStage 1を完了した。
事前登録したscientific gateはFAILし、decisionは
`missing_distance_exp209_failed_close_without_rescue`。
inferenceとsubmissionは実装・実行せず、このbranchを閉じる。

## 仮説と固定設計

exp209のGR interpolation、known-prefix zero-fill sigma、absolute TVT grid、
41 rate states、transition、prior、posterior meanを固定し、raw-missing rowだけ
Gaussian log-emissionを
`max(0.25, 2^(-nearest_finite_row_distance/8))`
で1回弱めた。

- scientific variant: 1
- reporting folds: 5
- exact-HMM well-runs: 773
- model / LightGBM config / trained fold / booster: 0 / 0 / 0 / 0
- PF / Beam / parent-control rerun: 0 / 0 / 0
- Kaggle: private CPU、GPU/TPU/internet off
- runtime: `17,475.557881 sec`

saved exp209 HMM、saved exp072 LikPF、exp226 fold/truth、exp115 hidden-like
assignmentはSHA固定のread-only inputとして使い、親controlは再実行していない。
candidate predictionとraw-GR emission contractをtruth結合前にfreezeした。

## Stage 1結果

### Overallとfold

`improvement_ft = control RMSE - candidate RMSE`で、正値が改善を表す。

| scope | candidate RMSE | exp209 control RMSE | improvement |
| --- | ---: | ---: | ---: |
| overall | 12.012570 | 11.938287 | -0.074283 |
| fold 0 | 11.195489 | 10.923776 | -0.271713 |
| fold 1 | 12.333944 | 12.302481 | -0.031463 |
| fold 2 | 11.599101 | 11.570050 | -0.029050 |
| fold 3 | 12.731649 | 12.723861 | -0.007788 |
| fold 4 | 12.121979 | 12.067702 | -0.054276 |

最低`+0.05 ft`改善に対して`-0.074283 ft`、改善foldは必要4/5に対して
0/5だった。

### 重要scope

| scope | candidate RMSE | control RMSE | improvement |
| --- | ---: | ---: | ---: |
| raw GR observed | 11.982124 | 11.933740 | -0.048384 |
| raw GR missing | 12.077802 | 11.948064 | -0.129738 |
| gap 1--3 | 11.969646 | 11.845475 | -0.124171 |
| gap 4--15 | 12.314455 | 12.171910 | -0.142545 |
| gap 16+ | 12.038180 | 11.926330 | -0.111850 |
| missing fraction low | 11.617131 | 11.604489 | -0.012642 |
| missing fraction mid | 12.450871 | 12.489036 | +0.038165 |
| missing fraction high | 11.954713 | 11.792411 | -0.162302 |
| MD since 0--250 | 2.212989 | 2.213935 | +0.000946 |
| MD since 250--1000 | 5.691362 | 5.666805 | -0.024557 |
| MD since 1000+ | 13.218208 | 13.135431 | -0.082776 |
| hidden-like spatial | 12.789461 | 12.564491 | -0.224970 |
| hidden-like typewell-purged | 12.596831 | 12.367244 | -0.229587 |

mid missing-fractionと直近250 ftだけは小さく改善したが、仮説の主対象である
raw-missing、全gap bucket、高missing-fraction、1000+、hidden-like 2面は
すべて悪化した。

### by-wellとfixed blend

- improved / regressed wells: `358 / 415`
- median delta RMSE（candidate - control）: `+0.000786 ft`
- p95 delta RMSE: `+0.469370 ft`（gateは`<= 0`）
- worst well: `f5859199`
- worst delta RMSE: `+6.630365 ft`（gate上限`+0.25 ft`）
- fixed LikPF 50:50 candidate / control:
  `10.306673 / 10.269693`
- fixed blend delta: `+0.036981 ft`で非劣化guard FAIL

## Technical gateの事後監査

Kaggleが記録したStage 1 technical gateは
`missing_weight_formula_exact=false`の1項目によりformal FAILだった。
一方、次はすべてPASSしている。

- 3,783,989 rows / 773 wells / 773 HMM runs
- input SHA、row/well/ID、saved control metric parity
- finite coverage 1.0
- posterior normalization max error `3.997e-15`
- observed weight exact 1
- missing weight min / max / unique:
  `0.25 / 0.917004043 / 16`
- Gaussian emission clip 600、weight application count exact 1
- truth rows accessed before prediction freeze: 0

原因切り分けのため50,553,974 byteのfrozen raw-GR emission contractだけを取得した。
1,200,837 missing rows中753 rowsで、gzip CSV再読込後のweightが再計算値と
bit-exact一致しなかったが、最大絶対差は
`5.551115123125783e-17`で、`rtol=0, atol=1e-16`では全件一致した。
したがって生成・HMM適用時の式逸脱ではなく、post-CSV float parseに対する
過剰なbit-exact guardである。実行後にgate定義は変更せずformal FAILを記録する。
scientific gateはこのtechnical表示に関係なく明確にFAILしているため、再実行しない。

## 再現性

- canonical / package / remote Notebook: 22 cells
- package / remote cell-source SHA:
  `fbe348f196f8a5ddcd74938a82480ca969da3e32d4020053ef666958f2c18356`
- scientific contract SHA:
  `90e02546e56e9b0b3c1d58f944fa6f3fe82fc63c6467bfde8e4b94882399ab65`
- input/control manifest SHA:
  `c07b2ee089457d07257661d452c8511452f9dd03ca41827fa3e3eaa480af00f0`
- prediction decompressed SHA:
  `5d5c1cb9a0682d5f352e56dd19fffd44574816ce26b0a0e85cfc49d16cc14742`
- raw-GR emission contract decompressed / raw gzip SHA:
  `36499c6d7d81eb90e98f9181f427db5023eda3857dc3877503e64ac1bdfb7e14` /
  `eebf8350458c0ee46e0924ff5827fd9db124dbdb7e95282e12b619bfdc425d85`
- observation audit decompressed SHA:
  `bf88a2e069e0b8eb906b6a90fdf11e4e4374cf5153a0f3e7dc40b9ea4949e4d4`

logs、small metrics/gate/audit artifactsと、technical切り分けに必要なraw-GR
emission contractだけを取得した。86,367,760 byteのpredictionを含むoutput archive
全体はダウンロードしていない。

## 解釈

距離が離れた補間GRを一律に弱めると、wrong evidenceを抑える利益より、
exp209が利用していた有効な観測情報を失う害が大きい。悪化はmissing rowだけでなく
observed rowにも伝播し、HMM smoothingを通じてwell全体のpathを変えている。
gap長にかかわらず全bucketが悪化し、hidden-likeとtailも悪化したため、
half-life/floor調整やhard maskで救済する根拠はない。

## 結論

事前契約どおりhalf-life/floor grid、hard mask、sigma/transition/prior変更、
temperature/clip search、blend rescue、same-OOF rescueを行わず閉じる。
inferenceとsubmissionへ進まない。
