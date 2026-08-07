# exp340 結果

## 状態

Kaggle private CPU kernel version 1を完了した。technical gateはPASS、固定scientific
AND gateは7/7 family FAIL。判定は
`close_depth_alias_confidence_branch_without_rescue`であり、補正、推論、提出は行わない。

- Kernel: `kentookumura/exp340-exp226-alias-readout-on-exp264-train`
- Kaggle id_no: `128356047`
- runtime: `26.400168 sec`
- 評価量: 3,783,989 rows / 7,787 blocks / 773 wells / 5 folds
- 実行量: 7 family / circular control 1 / model 0 / trained fold 0 /
  booster 0 / HMM well-run 0 / 親control再学習0
- feature content SHA:
  `70748900630ac7b67fb5d489c8410a1200f0ded547b3df13502e7bb21626e437`
- quantile content SHA:
  `e207e5598542030747fbbdfdbc6c21ea2d8b686ce81b95fd681d34df00cbee98`

## Pooled readout

| family | Q4-Q1 mean block RMSE (ft) | bad10 AUC | 正方向fold | real > circular fold | 判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| top1-top2 margin | -0.346353 | 0.486936 | 2/5 | 対象外 | FAIL |
| softmax entropy | -0.350594 | 0.497847 | 1/5 | 対象外 | FAIL |
| likelihood-weighted shift std | +0.142634 | 0.512446 | 3/5 | 対象外 | FAIL |
| zero-shift rank | +0.905341 | 0.541894 | 5/5 | 対象外 | FAIL |
| absolute top1 shift | +1.359545 | 0.544737 | 5/5 | 対象外 | FAIL |
| prior-block top1 jump | +2.253795 | 0.574392 | 5/5 | 3/5 | FAIL |
| 3-block sign inconsistency | +1.513854 | 0.548155 | 5/5 | 5/5 | FAIL |

## 判定

coverage、fold別Q1/Q4分離、1000+とhidden-like 2面は上位4 familyで概ね成立し、
zero-shift rank以降はQ4-Q1 RMSE差と4/5 foldも満たした。しかし全familyで必須AUC
`>=0.60`を満たさず、最良はprior-block jumpの`0.574392`だった。同familyはさらに
circular controlを上回ったfoldが3/5で、必須4/5に届かなかった。3-block sign
inconsistencyはcircular guardを満たしたがAUC `0.548155`に留まった。

これは「大きなshiftや連続block間の不整合は平均誤差の層別化には使えるが、10 ft以上の
失敗を安定して識別するconfidenceとしては弱い」ことを示す。結果を見たthreshold探索、
family blend、補正、selector化は行わず、本branchを閉じる。

## 次

同familyの救済実験は追加しない。独立仮説として実装済みのexp342 Student-t Stage 0と、
design-onlyのexp343 ACF安定性監査を既存P3候補として残す。どちらも実行は別承認とする。
