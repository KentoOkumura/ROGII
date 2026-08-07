# exp357 結果

## 状態

Kaggle private CPU version 1でStage 0、version 2で実際のStage 1 exact-HMMを完了した。
Stage 1のHuber HMMは保存済みGaussian HMMを全体・4/5 folds・全required scopeで
改善したが、by-well p95、worst well、exp226 direct ceilingを満たさなかった。
固定gateに従い`stage_1_failed_close_without_rescue`で終了する。

- kernel: `kentookumura/exp357-huber-emission-independent-audit-train`
- version / id_no: `2 / 128448451`
- Stage 0 / Stage 1 runtime: `319.617349 / 9597.242200 sec`
- runtime: CPU、GPU/TPU/internet off

## 仮説

固定`delta=1.345` Huberが、exp280のSHA固定Gaussian shift scoreに対し、
center付近の識別力を保ったままextreme residualの影響を抑えられるかを独立監査した。

## Stage 0結果

| Gate | 値 | 判定 |
| --- | ---: | --- |
| score finite / row identity / Gaussian rank parity | `1.0 / 1.0 / 1.0` | PASS |
| pooled MRR gain | `+0.0000416`（必要`+0.01`） | **FAIL** |
| pooled top3 gain | `+0.0001284`（必要`+0.01`） | **FAIL** |
| MRR改善fold | `2/5`（必要`4/5`） | **FAIL** |
| top3改善fold | `2/5`（必要`4/5`） | **FAIL** |
| stress MRR / top3非悪化 | `false / false` | **FAIL** |
| real-vs-circular gap非悪化 | `true / true` | PASS |
| extreme residual top3 gain | `+0.0114943` | PASS |
| extreme residual regret delta | `-0.652270 ft` | PASS |

Huber / Gaussianのpooled MRRは`0.3896675 / 0.3896260`、
top3は`0.4525491 / 0.4524207`だった。Huberのtop1 margin平均は
`0.0172596`でGaussianの`0.0183269`を下回り、flattening signalも検出された。

## Stage 0実行契約

- fixed Huber scientific score / saved Gaussian control: `1 / 1`
- shift candidates / reporting folds: `13 / 5`
- HMM well-run / model config / trained fold / booster: `0 / 0 / 0 / 0`
- parent/control再学習: `0`
- inference / submission: `0 / 0`

3,783,989 rows、773 wells、7,787 blocksを評価した。

## Stage 1実HMM結果

| Gate | 値 | 判定 |
| --- | ---: | --- |
| overall RMSE: Gaussian → Huber | `9.827420 → 9.737195`（`-0.090225 ft`） | PASS |
| 改善fold | `4/5` | PASS |
| long-tail 1000+ | `10.834743 → 10.746975`（`-0.087768 ft`） | PASS |
| hidden-like spatial | `10.556607 → 10.347981`（`-0.208626 ft`） | PASS |
| hidden-like typewell-purged | `10.304139 → 10.302894`（`-0.001245 ft`） | PASS |
| by-well p95 delta | `+0.003365 ft`（必要`<=0`） | **FAIL** |
| worst well `4a8ecc0b` | `+1.403715 ft`（必要`<=+0.25`） | **FAIL** |
| exp226 direct ceiling | `9.737195 > 9.427110`（`+0.310086 ft`） | **FAIL** |
| finite / row identity | `1.0 / 1.0` | PASS |

fixed Huber `delta=1.345`の実HMMを1 variant、773 wellsで実行した。
model config、trained fold、booster、parent Gaussian HMM再実行はすべて0。
Stage 1の科学gate全体とdirect promotionはFAILで、inference / submissionは実施しない。

## 再現性

- target-free score content SHA:
  `832552dded42940ac57cc9aac425d1ea1be7b0d6b6e17950d792f7e1c9a95902`
- target-free score decompressed SHA:
  `65403a1c56e666782a9c7e423d2b8c42b27977fb43d24c57cdda5b1750f4029a`
- block readout content SHA:
  `7a296bf21ada0c3366007cbeb77d9f231679e4ad36366cb218d85cf06a6c6276`
- block readout decompressed SHA:
  `4e6870fbc4e597a1cfc677a768cdd46760b9207fce7c0d61a5715d336c4f231a`
- gate SHA:
  `b79190de4e4d737a71f37373e2efd73a86b8e68712db952e21d245eb418e4dd7`
- Stage 1 candidate content SHA:
  `784885201e8faf5ecb9d4a91d8722d3477572b2b9e837fb5d76cf72e0b50c4a6`
- Stage 1 prediction decompressed SHA:
  `1a8e94c2d54227c1afe49e89982d97f636fbc1bf814500765a0efb4cf26221b8`
- Stage 1 gate SHA:
  `52b6f23e3d46e99ddc5940653edb80f723eb356c7022d313b116ec7700249829`

metrics、gate、主要artifact SHAはKaggle kernelログから記録した。
実ファイル確認が不要なため、output archive全体はダウンロードしていない。

## 解釈

Stage 0 proxyは実HMMの平均改善を予測できなかった。実HMMではGaussian比
`-0.090225 ft`、4/5 folds、全required scopeで改善しており、Huber tail自体には
平均的な効果がある。一方、well-level tailは一貫せず、最悪井戸は`+1.403715 ft`悪化し、
直接基準exp226にも`+0.310086 ft`劣る。したがって採用可能な改善ではない。
delta、scale、sigma、tempering、blendの事後調整は行わない。

## Stage 1実行契約

- Huber exact-HMM variant: 1
- HMM well-runs / reporting folds: `773 / 5`
- model config / trained fold / booster / parent-control再実行: `0 / 0 / 0 / 0`
- Stage 0再実行、inference、submission: 0
- exp281 Gaussian emissionだけをfixed Huber `delta=1.345`へ置換する。
- 実行結果は固定tail-safety / direct ceilingをFAILしたため、救済なしで閉じる。
