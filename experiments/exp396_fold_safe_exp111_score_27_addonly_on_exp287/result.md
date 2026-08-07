# exp396_fold_safe_exp111_score_27_addonly_on_exp287 結果

## 状態

Stage A全gate PASS。Stage BはKaggle T4で固定15/15 GPU boostersを完走したが、
固定promotion gateは1/6項目だけPASSしてFAIL。branchは推論・提出なしで閉鎖した。

## 仮説

exp111 score系27列をdownstream outer foldに対してstrict nested生成すれば、旧fold0 model全train適用の
non-OOF問題を除去し、exp287の421特徴に追加情報を与えられる。

## 固定設定

- 親: `exp287_fold_safe_formation_74_addonly_on_exp264`
- clean tail control: `exp264_exp263_candidate_confidence_dual_selector`
- Stage A: outer 5 × inner 4 × 2目的 = 40 CPU boosters
- Stage B: 421 + 27 = 448特徴、1 variant × 3 configs × 5 folds = 15 GPU boosters
- control再学習: 0 boosters
- seed: 42 + stable SHA256 local subsample seed

## 変更点

exp287の421特徴とdownstream TVT設定は固定し、strict nested生成するexp111 score系27列だけを
add-onlyする設計である。旧non-OOF 27列と依存GRWR 6列は入力にしない。

## 実装結果

- strict nested scorer、target-free 48特徴、model固有median、stable sample、10 core→27列を実装した。
- targetは特徴量ビルダーへ渡す前に分離し、outer/inner well境界と実行承認をfail-closedにした。
- 明示承認後にtrain候補を正規train notebookへ採用した。inferenceはfail-closed候補のままである。
- 専用testは10件PASS。py_compile、Ruff、Jupytext round-tripもPASSした。
- Kaggle private CPU version 1でStage A 0-booster preflightを実行し、16/16 checksをPASSした。
- version 2で固定40 CPU scorerをすべて学習し、40 model、40 model固有median、
  10 score-core partitionsを保存した。
- Stage Bはprivate T4 version 1で15/15 boostersを完走し、15 modelとOOFを保存した。
- inference、submissionは実行していない。

## Preflight結果

- Kernel: [`kentookumura/exp396-foldsafe-exp111-score27-exp287-train`](https://www.kaggle.com/code/kentookumura/exp396-foldsafe-exp111-score27-exp287-train)
- version / id_no: `1` / `128540844`
- status: `KernelWorkerStatus.COMPLETE`
- rows / wells: `3,783,989` / `773`
- nested fold roles: `20`、outer/inner well overlap合計 `0`
- row / well / full coverage: `1.0 / 1.0 / 1.0`、duplicate ID `0`
- fixed schema: scorer input `48`、score core `10`、derived features `27`
- runtime / peak RSS: `277.133756 sec` / `5.168308 GB`
- fitted boosters / predictions / submission / control retrain: `0 / 0 / 0 / 0`
- 旧exp111 model prediction、旧27列、依存GRWR 6列は未使用

入力SHA、行整合、fold/well境界、schema、固定実行量を含む16項目はすべてtrueだった。
これはtechnical preflightのPASSであり、40 booster実行時のscorer品質とresource gateのPASSではない。

## Stage A本学習結果

- Kernel version / id_no: `2` / `128540844`
- status: `KernelWorkerStatus.COMPLETE`
- completed CPU boosters: `40 / 40`
- model / median / schema records: `40 / 40 / 40`
- score-core partitions: `10`、duplicate ID合計 `0`
- nested fold roles: `20`、outer/inner well overlap合計 `0`
- sampled rows: 全20 roleで`350,000`
- runtime / peak RSS: `3662.974058 sec` / `8.762432 GB`
- prediction / submission / control retrain: `0 / 0 / 0`
- technical checks: `22 / 22 PASS`
- scorer-quality checks: `6 / 6 PASS`

| 品質指標 | learned | outer-train candidate prior | delta | 改善fold |
| --- | ---: | ---: | ---: | ---: |
| expected-error MAE | 5.908305 | 39.055075 | -33.146770 | 5/5 |
| within10 logloss | 0.345125 | 0.535689 | -0.190564 | 5/5 |
| within10 Brier | 0.112135 | 0.177406 | -0.065271 | 5/5 |

3指標とも固定gateの「pooled改善かつ4/5 folds以上改善」を満たした。これは27列を作るscorerの
品質結果であり、exp287へ27列を追加したdownstream TVT RMSEの改善結果ではない。

## 科学的結果

| メトリック | 値 |
| --- | --- |
| Stage A technical preflight | 16/16 PASS |
| Stage A technical gate | 22/22 PASS |
| Stage A scorer quality | 6/6 PASS、3指標とも5/5 folds改善 |
| Stage A full-run resource gate | 3662.97秒 / 8.762 GB、PASS |
| Stage B OOF RMSE | `8.134294735` |
| exp287 OOF RMSE | `8.136708220` |
| Stage B delta vs exp287 | `-0.002413486 ft`、必要値`<= -0.02`でFAIL |
| nonworse folds vs exp287 | `2/5`、必要値`>= 4/5`でFAIL |
| maximum scope delta vs exp287 | `+0.026155871 ft`、上限`+0.02`でFAIL |
| by-well delta p95 vs exp287 | `+0.342926545 ft`、上限`0.0`でFAIL |
| worst-well delta vs corrected exp264 | `+7.802733095 ft`、上限`+0.25`でFAIL |
| +1/+3/+5 ft worsened wells vs exp264 | `68 / 16 / 5`、上限`135 / 39 / 14`でPASS |
| Stage B promotion gate | `1/6 PASS`、総合FAIL |
| Public LB | 未提出 |
| Private LB | 未提出 |

fold別deltaは`+0.015713 / -0.033683 / -0.011692 / +0.000726 / +0.016578 ft`。
scope別deltaはnear `+0.003766`、mid `+0.026000`、1000+ `-0.005071`、
hidden-like spatial `+0.021101`、hidden-like typewell-purged `+0.026156 ft`だった。
27列のfeature importanceはgainの`1.1025%`、splitの`2.4936%`を占めたが、
小さいpooled gainをfold/scope/by-well tailへ安定転移できなかった。

## 判定

Stage Bはpooled OOFを`0.002413 ft`だけ改善したが、事前固定した実用改善量、fold再現性、
scope、by-well p95、worst-wellの5 gateを満たさなかった。scorer単体品質のStage A PASSは、
downstream TVTでの安定した価値へつながらなかった。`fold_safe_exp111_score_27_addonly_on_exp287`
はnegative resultとして閉鎖し、exp287をtrain-side parent anchorに維持する。
subset/grid、same-OOF rescue、gate緩和、再学習、inference、submissionへは進まない。

## 再現性

- deterministic anchor: false
- seed policy: fixed fold seed 42 + stable SHA256 local subsample seed
- kernel version / id_no: `2` / `128540844`
- fold assignment SHA: `e8c3dd328af6e5b295940467679fb36b07614f6d6f0b1120f1bc373bec85f92f`
- nested manifest logical SHA: `f8104336a64369cccdeddbf1c29c349872f6c91529677391cdb9946892035`
- synthetic 27-feature logical SHA: `2c5e71073a8f70825b95e1f825789a00d79fff275b27b90271bffe589df8a61c`
- preflight manifest SHA: `2661b808dedc5e463d3813af501e6e61f397cac36f044e9cf709b8d6ccb02749`
- nested fold manifest SHA: `3d42173760625993e51a76d91eb2b7d7dc9e9fbe1a16dbba71c2382c749d41c0`
- Stage A summary SHA: `e5fa67ed62266a8cfed03121463692d1abec1e1e85616e3af0611912cf1a383e`
- gate SHA: `35314fb017234cd0ed3fe8a2eb7f998e7ceee9b837a20c4c207d54718391ff49`
- model manifest SHA: `b42757edd5dbb061ccc2903595e7c1676a8e21bfcc87837202a3f9387f4bd028`
- score partition manifest SHA: `055d2c1f5583b6b44b0ea3413bfd66592b485adb764cec69522e7b33d571aba5`
- quality SHA: `40abadfb5e9e385a9b15fbc53ae4d829cde948badbd0e2be66b64ea7f213b679`
- reproducibility manifest SHA: `84cc8703868eea2ebddbbfb176e861505e1f3b6281c725b02e772a31518338f0`
- log SHA: `a8e96ce402e6e44972121ea79034431225bb6345b7de64b4088cb14bded1353b`
- Stage B kernel version / id_no: `1` / `128570498`
- Stage B feature schema SHA:
  `07f6c2b51d166f210bae18720c32fae638aead255b24adda4ac598eaac517630`
- Stage B model manifest SHA:
  `85059c057895365c53158e75a5f18246414d591123c5b52a1627501af63d75c1`
- Stage B OOF prediction SHA:
  `ebf4a12896c80435ab12f16e8bcb3297874edef81b7630f42dea7c53713a81c3`
- Stage B metrics SHA:
  `d80395f71279e7a8c1597f91902303c471ecfaf22e13fddca61193f7fcdc5146`
- Stage B reproducibility manifest SHA:
  `fd7434ce4dd39be3996d1a10c403d1ed00d4d2a4ce67d34c266b265e1031c8a9`
- Stage B log SHA:
  `0a28d6b1a83453e27f3ceeb4b5121fc0ae6cbca30564077974377de593989b46`
- submission SHA: なし
- rerun result: なし

## 次

exp396は閉鎖する。27列の救済実験は追加しない。保存済み生成物だけでStage A scorer-qualityから
Stage B downstream TVTへの転移失敗を分解する0-booster readoutを低・P4に置き、
新しい独立した必要性と承認がない限り着手しない。
