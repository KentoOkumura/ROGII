# exp436_sparse_global_stratigraphic_potential 結果

## 状態

`stage0_fail_closed`。canonical private Kaggle CPU kernel version 2
（id_no `129058940`）が`COMPLETE`し、Stage 0は科学契約どおりFAILしてbranchを
閉じた。Stage 1、Stage 2、推論、提出は実行していない。

## 仮説

outer-train contactから6つのglobal `U_k(X,Y)` surfacesを疎に正則化し、
最後の既知点からの固定等重みpotential差でsuffixを進めれば、exp226の局所donor
mismatch蓄積を避け、単一`P(X,Y)`では失われる地層準位も保持できる。

## 固定設定

- 親/control: exp226、保存OOF CV `9.427109596582213`、再生成0
- Route: `pf_beam`
- 1 primary candidate / 6 formation report-only paths
- 6 surfaces ×5 folds=`30` global field fits
- target raw formation / GR使用なし
- local surface、HMM、PF、Beam、ML、selector、blendなし

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |

Stage 0 target-free resource/integrity:

| 項目 | 観測値 | 判定 |
| --- | ---: | --- |
| rows / wells / folds | 3,783,989 / 773 / 5 | PASS |
| source-valid overlap | 0 | PASS |
| target formation / GR / suffix truth reads | 0 / 0 / 0 | PASS |
| duplicate contact/node/edge keys | 0 | PASS |
| finite source coverage | 1.0 | PASS |
| source wells / formation 最小 | 4（閾値32） | FAIL |
| fold 0 surface solve成功率 | 5/6 = 0.833333 | FAIL |
| query coverage / supported formations / query sources | 0 / 0 / 0 | FAIL（query未実行） |
| reported projected runtime（query未実行） | 175.435738 sec | PASS・参考値 |
| projected peak RSS | 0.549873 GB | PASS |
| preflight sparse solves | 30 | contract内 |

source contact wellsは、`ANCC 595–598`、`ASTNU 614–616`、`ASTNL 616–618`、
`EGFDU 592–596`、`EGFDL 555–561`に対し、`BUDA`だけが各fold
`5 / 4 / 4 / 5 / 6`だった。fold 0では前5面のHuber IRLS + LSQRが成功し、
BUDAは`5 < 32`でsolveを開始しなかった。6面すべてが揃わないため、固定16 wellsの
queryは意図どおり開始していない。

したがってruntime値はcensusと5面solveを主に反映し、full query込みの投影ではない。
Stage 0全体はsupport/query gateでFAILしているため、このruntime PASSをpromotion根拠には
使わない。

version 1は同じBUDA不足を例外として終了した。gateを変更せず、data support不足を
solver manifestとStage 0 decisionへ保存して正常終了するfail-close修正をversion 2で
検証した。

## 再現性

- deterministic anchor: false
- seed policy: RNGなし、固定順
- kernel version / id_no: `2 / 129058940`
- artifact bundle SHA256:
  `5b1a20bda01409bb562c241611f023568ab811f9be8683ff7456f212a56da6d2`
- Stage 0 decision SHA256:
  `af50b420c899c58854a6721b1fdf011b05472746eff86871739f54af5275fbd8`
- query / prediction: 6面contract不成立のため0 rows
- model / submission SHA: 対象外

## 解釈

Stage 0はleakage、入力identity、finite、重複、runtime、RSSを満たした。一方、
BUDA first contactのfold-safe supportが4–6 wellsしかなく、固定6面global
potentialを同定できない。formation除外やcontact定義変更で同じexpを救済しないという
事前契約に従い、exp436をnegative resultとして閉じる。exp381のcontact-TVT FAIL、
exp383のresource FAIL、exp273のprefix 2D gradient negativeは維持する。

## 次

exp436では再実行しない。BUDAをtarget-free source supportだけで事前除外した固定5面
contractを検討する場合は、exp436のFAILを再分類せず、別実験・別承認で扱う。
