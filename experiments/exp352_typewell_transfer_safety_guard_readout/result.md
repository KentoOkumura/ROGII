# exp352_typewell_transfer_safety_guard_readout 結果

## 状態

Kaggle private CPU version 1を完了し、固定Stage 0 gateはFAILした。
8 checks中7 PASSで、worst-well safetyだけがFAILした。branchは救済調整、再実行、
inference、submissionなしで閉じる。

- kernel: `kentookumura/exp352-typewell-transfer-guard-readout-train`
- version / id_no: `1 / 128360039`
- diagnostic runtime: `12.973409 sec`
- runtime: CPU、GPU/internet off

## 仮説

固定したtarget-free availability/support/fallback guardが、exp311のsame-group平均gainを保ちながら
未知群・purged wellへのnegative transferとworst-well悪化を防げるかを監査する。

## 判定予約

- coverage `>=0.90`
- identity parity `<=1e-10`
- same-group gain `>=0.05 horizontal GR API`、改善fold `>=4/5`
- leave-group-out / spatial+typewell-purged negative transfer
  `<=0.00 horizontal GR API`
- worst-well regression `<=+0.25 horizontal GR API`

exp311はTVT予測を生成していないため、事前登録した数値は維持し、
誤っていた`ft`表記だけを保存scoreと同じhorizontal GR APIへ訂正した。

## 変更点

design-only scaffoldへcompact self-contained train/inference候補、exp311 SHA preflight、
target-free availability/fallback manifest、late suffix scoring、3面gate、contract testsを
追加し、compact trainを正規Notebookへ採用してStage 0を実行した。予測、inference、
submissionは生成していない。

## Stage 0結果

| Gate | 値 | 判定 |
| --- | ---: | --- |
| fold-safe exact coverage | 0.972833 | PASS |
| identity row parity max abs | 0.0 | PASS |
| truth rows before manifest freeze | 0 | PASS |
| same-group gain | +0.381540 GR API | PASS |
| same-group improved folds | 5/5 | PASS |
| leave-group-out negative transfer | -0.164862 GR API | PASS |
| spatial+typewell-purged negative transfer | -0.496752 GR API | PASS |
| worst-well regression | +12.914716 GR API | **FAIL** |

- same-group: 773 wells。exact 752、global fallback 21、identity 0。
- leave-one-group-out: 773 wells。global fallback 773。
- spatial+typewell-purged: 200 wells。exact 138、global fallback 62。
- worst wellは`d07aed8f`。same-groupでidentity RMSE `5.587119`から
  guarded RMSE `18.501835`へ悪化した。

## 再現性

- deterministic anchor: いいえ。RNGなしの固定readoutだがrerun parityは未確認。
- seed policy: RNGなし、保存済みfold/group/well順固定。
- availability manifest freeze SHA:
  `2cbc04ebc4badbdd0d4d482f6bf9447ef085b8b98ab3dd77d7095f0ed93331a3`
- availability content SHA:
  `6648b769c2f2eb15a0166670ca82019c3141f5c9e65582d6cfd93c4b36303be0`
- score content SHA:
  `b56498246fb401ca18a1d6e94b09f8dd82f047497f279504c3c834c12243c4d3`
- model / prediction / submission SHA: 非該当。

## 解釈

固定availabilityは平均gainと未知群/purged面のpooled non-regressionを保ったが、
peer 2 / support 64を満たすexact group内にも`d07aed8f`のようなcatastrophic transferが残った。
support availabilityだけでは個別well safetyを判定できず、exp311のworst-tail failureを
解消できない。旧exp311/312のFAIL判断と旧exp314--320の閉鎖を維持する。

## 次

同じscoreでpeer/support threshold、fallback順、global重みを救済調整しない。
既存の独立候補`exp353_typewell_group_quality_feature_preflight`は、direct priorではなく
soft quality featureの事前診断としてのみ残し、本結果から自動実装・昇格しない。
