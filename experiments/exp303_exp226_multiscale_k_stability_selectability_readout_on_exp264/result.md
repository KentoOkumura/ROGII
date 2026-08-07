# exp303_exp226_multiscale_k_stability_selectability_readout_on_exp264 結果

## 状態

Kaggle private CPU version 1で固定readoutを完了した。technical checksは全PASSしたが、
事前登録したscientific checksは全FAILしたため、`technical_pass_scientific_fail`としてbranchを閉じる。
selector学習、prediction変更、inference、submissionは行わない。

## 実行

- Kernel: `kentookumura/exp303-k-stability-readout-train` version 1、id_no `128080983`
- Runtime: 約142.125秒（Kaggle log終端時刻）
- 実行量: 1 fixed readout、5 evaluation folds、0 model、0 trained fold、0 booster、candidate再生成0
- Runtime contract: private CPU、GPU/TPU/internet無効
- 評価単位: 7,787 H512 blocks / 773 wells / positive 2,596 blocks

## 仮説

K12/K16/K24のtarget-free scale instabilityが、corrected exp264 Stage C v6によるK16 misrankingを識別する。

## 結果

| メトリック | 事前条件 | 結果 | 判定 |
| --- | ---: | ---: | --- |
| pooled H512 AUC | `>=0.65` | `0.488805` | FAIL |
| AUC `>0.5`のfold数 | `>=4/5` | `1/5` | FAIL |
| top/bottom positive-rate lift | `>=1.5x` | `0.916190x` | FAIL |
| top-bottom mean K16 benefit差 | `>=0.25 ft` | `-1.205532 ft` | FAIL |
| 1000+ / hidden-like方向 | 全面PASS | `0/3` | FAIL |
| Public/Private LB | 対象外 | `- / -` | - |

pooledでは高instability側ほどpositive率が低く、K16 benefitも悪かった。primary scoreは期待方向に
識別できず、ランダム近傍よりわずかに逆方向だった。

### fold別

| fold | blocks | AUC | positive-rate lift | mean benefit差 (ft) | 方向 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 1,558 | 0.495293 | 0.989474 | -1.442062 | FAIL |
| 1 | 1,558 | 0.449110 | 0.666667 | -3.240628 | FAIL |
| 2 | 1,556 | 0.496653 | 0.940594 | -1.246143 | FAIL |
| 3 | 1,558 | 0.477773 | 0.968421 | -0.627297 | FAIL |
| 4 | 1,557 | 0.520549 | 1.117117 | +0.646817 | PASS |

### stress scope

| scope | blocks | AUC | positive-rate lift | mean benefit差 (ft) | 方向 |
| --- | ---: | ---: | ---: | ---: | --- |
| distance_1000_plus | 6,242 | 0.488712 | 0.933333 | -0.954359 | FAIL |
| hidden_like_spatial | 2,001 | 0.479843 | 0.941176 | -1.196517 | FAIL |
| hidden_like_typewell_purged | 2,009 | 0.490046 | 0.993056 | -0.914468 | FAIL |

## Technical checks

feature coverage、block重複0、truth-before-freeze 0、exp302 prediction SHA、corrected exp264
candidate-score SHA、score再計算（最大絶対差0.0）はすべてPASSした。したがってscientific FAILは
入力不整合やfreeze違反ではなく、固定したK-scale instability scoreに識別力がなかった結果として扱う。

## 再現性

- input manifest SHA: `9089d8ded32a7c30ae1504a345993d654a8630ce600ba335a17b1cee99c840ba`
- feature schema SHA: `ca56361d0aef8a8ffe127418ceadd1cf666dcdeaafd13246a7be19ddfe0e69a7`
- feature content SHA: `964da0fa966cf24f1f2d3755cf365767f19c37a62c7fab46be819528196a38ea`
- H512 block content SHA: `55fd6db94a7b4120cde515238e743a273c047aef26f772b32972a4e4cf851267`
- post-freeze truth content SHA: `c3db553e89c7495cbcc01d99a38ab2b301bacf3e6fc8fa4dffe930eb93b35982`
- summary JSON SHA: `e7b76c68191273d2798968d5b87e6f1a6ef4cca1401b981e4c74143f1399649e`
- score recompute max abs: `0.0`
- deterministic prediction anchor: false（診断readoutでありpredictionを生成していない）

全metricsと生成物SHAはKaggle logに表示されたため、大きなoutput archiveは取得していない。

## 解釈

K12/K16/K24間のlevel、局所slope、固定segment-boundary jumpの大きさは、corrected exp264が
K16を過小評価する場所の代理にはならなかった。5 fold中4 foldと全stress scopeで方向が逆であり、
単なる閾値不足ではない。K値間の変動は「K16を選ぶべき不確実性」より、候補全体が不安定で
K16 benefitも低い領域を表している可能性が高い。

## 次

事前登録どおり、同じOOFでscore方向反転、feature weight、H128/H512、boundary幅、thresholdを
救済しない。K-scale instability selector familyは閉じ、新規救済expを追加しない。
次の独立候補は既に設計・実装済みのexp305 exact-HMM emission auditを優先する。
