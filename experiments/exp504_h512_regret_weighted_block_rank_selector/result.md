# 結果

## 状態

`completed_train_side_rejected_no_inference_no_submit`

Kaggle private CPU version 1（`id_no=129488458`）は完走し、technical gateを全PASSした。
pooled OOFはanchorを改善したが、fold、hidden-like scope、by-well tailの固定gateをFAILしたため、
`FAIL_TERMINAL_CLOSE_WITHOUT_HORIZON_LOSS_WEIGHT_OR_THRESHOLD_RESCUE`として閉じる。

## 実行契約

- fixed12 / H512 non-overlap / exp293 grouped outer 5 folds
- corrected exp264 88 row features、固定9統計、1,986 pair features
- regret-weighted pairwise logistic rank 1 config
- Borda top-1 + fixed 0.5 anchor guard
- 実績: 1 scientific variant / 1 config / 5 CPU models / 5 boosters
- 親control再学習、candidate再生成、PF/HMM/Beam、GPU、inference、submission: 0

## OOF

| fold | selector RMSE | anchor RMSE | delta |
| ---: | ---: | ---: | ---: |
| 0 | 7.838884 | 8.191667 | -0.352783 |
| 1 | 8.210832 | 8.486255 | -0.275423 |
| 2 | 7.623981 | 7.859006 | -0.235025 |
| 3 | 8.481513 | 8.357930 | +0.123583 |
| 4 | 8.383404 | 8.283040 | +0.100364 |
| pooled | **8.114277** | **8.238332** | **-0.124055** |

pooled gain `0.124055 ft`は必要な`0.05 ft`を満たしたが、非劣化foldは`3/5`で、必要な
`4/5`に届かなかった。

## 固定scopeとby-well tail

| scope | selector RMSE | anchor RMSE | delta |
| --- | ---: | ---: | ---: |
| MD since 0--250 | 1.571563 | 1.574294 | -0.002730 |
| MD since 250--1000 | 4.060865 | 4.162777 | -0.101912 |
| MD since 1000+ | 8.909382 | 9.042324 | -0.132942 |
| hidden-like spatial | 9.033867 | 8.748108 | +0.285759 |
| hidden-like typewell-purged | 8.963965 | 8.694132 | +0.269833 |

- by-well改善 / 悪化: `446 / 319`
- delta p50 / p90 / p95: `-0.154410 / +1.683951 / +2.963656 ft`
- worst: `81bf5923`, `+16.799044 ft`

hidden-like 2面は固定上限`+0.02 ft`を大きく超え、by-well p95 / worstも固定上限
`+0.25 ft`をFAILした。

## rank mechanism readout

- H512 exact top-1 accuracy: `0.112624`
- weighted / unweighted pair accuracy: `0.741908 / 0.691638`
- NDCG@1: `0.682537`
- top-3 oracle coverage: `0.286503`
- anchor選択: `2,942 / 7,787 blocks = 0.377809`
- anchor guard fallback: `747 blocks = 0.095929`
- inter-block switches: `4,427`

pairwise局所判別は成立し、pooled平均も改善した。一方、fold 3/4とhidden-like 2面で反転し、
well-tailを抑えられなかった。平均的なcandidate disagreementを捉えるだけでは、分布外寄りの
wellでblock hard choiceを安全にするには不足したと解釈する。

## technical / 再現性

- rows / wells / blocks: `3,783,989 / 773 / 7,787`
- technical checks: 全PASS
- truth前target-free freezeとouter-valid prediction freeze: PASS
- runtime: `5,422.758 sec`、peak RSS `12.884 GiB`
- block feature content SHA:
  `f333d097d8bdd369b2b6786328dee050d6bb5ba4114d810e26e60be976fd56c8`
- model manifest logical SHA:
  `60696a0574de0c62f8c413c2344a664f40a40634f202ec3a02a754bd2ef3de25`
- OOF prediction content SHA:
  `1dd09844b70536ec7eae26d6656efb70a00bdc3488a57aca188fb6dfc3b2504f`

根拠はKaggle logsと、必要な小型生成物だけを取得した`kaggle/output/train_v1/artifacts/`に置く。

## 判断と次

scientific promotionはFAIL。凍結契約どおり、exp504内でH128/H256、loss、weight、threshold、
model、guard、smooth/blend/gateを救済せず、inference / submissionへ進めない。

将来原因確認が必要な場合だけ、保存済みOOF・block selection・pair/rank readoutを入力にした
0-model / 0-predictionのblock-rank tail attributionを低優先P4として別途設計する。現行の
P1/P2/P3候補を追い越さない。
