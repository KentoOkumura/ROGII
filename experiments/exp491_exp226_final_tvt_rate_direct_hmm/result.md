# exp491 exp226最終rate直接入力HMM 結果

## 結論

Kaggle private CPU Stage 0 version 2を完了し、`stage0_fail_closed`と判定した。
technical gateは全件PASSしたが、mechanism gateは7件中1件だけPASS、
6件FAILだった。fixed32は機構確認でありCVやpromotion evidenceではないが、
事前登録した停止条件に従いStage 1、PF救済、inference、submissionへ進まない。

## 実行契約

- candidate: `exp226_final_rate_direct_transition` 1本
- Stage 0: fixed32、156,088 suffix rows、32 candidate HMM well-runs
- control再実行 / ML config / trained fold / booster / PF / Beam / GPU: すべて0
- version 1: 32/32 HMM wells後、truth-late readout前のgzip直列化で技術失敗
- version 2: 同じ科学的契約で完了
- 累計attempted HMM well-runs: 64

## 主要結果

| scope | exp226 final RMSE | candidate RMSE | candidate - exp226 |
| --- | ---: | ---: | ---: |
| all32 | 7.976057 ft | 12.290251 ft | +4.314194 ft |
| matched control 16 | 7.081195 ft | 6.101789 ft | -0.979406 ft |
| persistent 16 | 8.757067 ft | 16.169236 ft | +7.412169 ft |
| raw GR observed | 8.277210 ft | 12.363989 ft | +4.086780 ft |
| raw GR missing | 7.197283 ft | 12.110385 ft | +4.913102 ft |

改善foldは3/5だった。fold 0は`+9.406244 ft`、fold 1は`+8.942345 ft`
悪化し、fold 2–4はそれぞれ`-0.507348 / -3.346939 / -1.005852 ft`
改善した。pooled matched controlだけは改善したが、persistent episode SSEは
`-3.142300`（314.23%悪化）、paired by-well delta p95は`+22.805439 ft`、
worst well deltaは`+24.277444 ft`だった。

## Gate判定

technical gateは、156,088 rows / 32 wells / 5 folds、重複・欠損0、
finite coverage、正のdelta MD、strict allowlist、truth/role/episode-late freeze、
first-difference parity、TVT/U-rate恒等式、transition/posterior normalization、
gzip logical SHA readback、runtime、RSSをすべてPASSした。

mechanism gate:

| gate | 判定 |
| --- | --- |
| all32でexp226 final比0.10 ft以上改善 | FAIL |
| matched-control悪化0.02 ft以下 | PASS |
| persistentで0.10 ft以上改善 | FAIL |
| 改善fold 4/5以上 | FAIL（3/5） |
| persistent episode SSE 5%以上削減 | FAIL |
| by-well delta p95 0.25 ft以下 | FAIL |
| worst-well delta 2.0 ft以下 | FAIL |

## 解釈

exp226 finalの局所増分を毎行そのままtransition centerへ与えるだけでは、
exp226のabsolute offsetをHMM位置posteriorとGR emissionで安全に補正できなかった。
matched controlのpooled改善はある一方、persistent wells、fold 0/1、
GR observed/missingの両方、well tailが大幅に悪化しており、改善は安定していない。
この結果は「persistent rate状態を除いた直接rate入力」がexp226 finalを
一般に改善するという仮説を支持しない。

事前登録どおり、rate smoothing / clipping / scale、emission / grid変更、
blend / selector、same-OOF調整、PFによる救済は行わない。条件付きPF後続案も
先行条件を満たさなかったため閉鎖する。

## Runtime・再現性

- Kaggle kernel: version 2、id_no `129213586`、private CPU
- elapsed: `35.917593 s`
- candidate HMM: `10.073264 s`
- full 773-well projection: `243.332278 s`
- peak RSS: `0.449200 GiB`
- scientific contract SHA:
  `b89a89e14cc0ff628aaa8e49814f4fa66f68eab7d5e0a732b7d1565e4fb841c4`
- prediction decompressed / logical / readback SHA:
  `8b137edc5ee9cf578f0c6f17d6ab7fd3c34f5b86d3c3a6cbc29f432a7d15ee56`
- schedule manifest content SHA:
  `0dc5b8eb152ad70bda55c33917530eb737f4f3dfac83ddd3e5e6b165e4ba68d0`
- diagnostic manifest SHA:
  `b6ff40e23788d0213fc52eaf0b064d8fc64341980d998c0e9a3ce7503110b019`
- summary artifact SHA:
  `1a434e951a62e80aaaaa4a3a2b4e9b2cb7731c0de67c26f6829f4b891b0d57d6`
- deterministic anchor: false（同一成功runの再現確認は未実施）

## 最終判断

exp491はStage 0で完了・fail-closedとする。Stage 1、PF後続、inference、
submissionはすべてブロックする。
