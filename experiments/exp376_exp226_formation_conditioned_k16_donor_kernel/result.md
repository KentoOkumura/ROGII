# exp376_exp226_formation_conditioned_k16_donor_kernel 結果

## 状態

`kaggle_cpu_v2_completed_technical_pass_direct_fail_novelty_fail`。
Kaggle CPU v2はtechnicalとtarget-free Stage 0をPASSしたが、
Stage 1 directとStage 2 candidate noveltyをFAILした。
decisionは`close_formation_conditioned_donor_branch_without_rescue_grid`。
推論・提出は実施していない。

## 仮説

exp226の正解TVT由来K=16 donor slopeを固定し、既存XY donor weightだけを
outer-fold-safeな地層相対座標でsoft reweightすると、supportを保持したまま
地層的に不整合なdonorの影響を弱められる。

## 設定

- 親: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- 検証: exp226 outer 5-fold well split、Stage 0 target-free、
  Stage 1 direct、Stage 2 fixed12 add-one novelty
- メトリック: RMSE
- 予定量: 1 variant / 0 model config / 5 reporting folds /
  0 trained fold / 0 booster / parent control再実行0
- stochastic処理: なし
- 実装: 4,007行・9章・19セル、同一exp helper importなし
- v1 kernel: `kentookumura/exp376-formation-k16-donor-train` version 1、
  id_no `128436621`
- v1実行config SHA256:
  `a67d6f65307434fd3b8f1df8e0a7dca7ed9f0db51c5dad18dfee8fd2157b86e6`
- v1実行source SHA256:
  `625f05620785687a9cc8d6154d8eb63a30a4350b83ab7ad44a57c92915a5d8e5`
- v2 kernel: 同じkernelのversion 2、`COMPLETE`
- v2 runtime: metrics完了`1574.961秒`、log最終`1585.333秒`
- v2実行config SHA256:
  `b6ea3eaa05a9179d259e66a2bbf3b4ceda460d0647af9610bb7a6653bec77358`
- v2実行source SHA256:
  `aaaa75a4378b858d6dc180010e9f7cdaf2a72107e644287ae6ccdbcb0a822bea`
- 修正後検証: 専用test 4件、`py_compile`、ruff F821、
  Jupytext round-trip PASS

## 結果

| メトリック | 値 |
| --- | --- |
| v2 kernel status | COMPLETE |
| Technical / Stage 0 | PASS / PASS |
| Stage 1 direct / Stage 2 novelty | FAIL / FAIL |
| exp226 control RMSE | 9.427109597 |
| formation-conditioned K16 RMSE | 9.443257190 |
| direct delta | +0.016147593 ft（悪化） |
| 改善fold | 1/5 |
| by-well p95 / worst delta | +0.376679 / +1.891560 ft |
| H512 add-one oracle改善 | +0.019403532 ft |
| whole-well add-one oracle改善 | +0.015542019 ft |
| H512 strict unique-best率 | 9.387441% |
| novelty改善fold | 5/5 |
| Public LB | 未実行 |
| Private LB | 未実行 |

Stage 0は3,783,989 rows / 773 wells / 12,368 segmentsで全チェックをPASS。
formation factorは`0.511501--1.0`、finite coverage 1.0、fallback 0、
ESS比p05 `0.927173`、freeze前valid reference/truth/formation readはいずれも0。

direct scopeはnear 0--250 ftだけ`-0.010633 ft`改善し、mid / 1000+ /
hidden-like 2面は`+0.009194 / +0.017799 / +0.018538 / +0.019237 ft`悪化した。
well単位は376改善 / 397悪化、worstは`a3518960`の`+1.891560 ft`。

candidate noveltyはstrict unique-best率と5/5 foldsをPASSしたが、
H512 / whole-well改善は事前閾値`0.05 ft`に対して
`0.019404 / 0.015542 ft`に留まりFAILした。

## 再現性

- deterministic anchor: いいえ。v2は成功したが、成功run間のSHA一致は未確認。
- kernel version: v1 ERROR / v2 COMPLETE、id_no `128436621`。
- target-free prediction logical SHA:
  `5205c67f6cad8d549863f122ab989bf2874587c574494b59b639a1bc5d66fb25`
- prediction decompressed SHA:
  `49621fb7838bb5234553d507d9e6fe38a55127b25e82621ee655091fe6b340a0`
- support / reference logical SHA:
  `1b14e696817dded0ec81d357c9529fdee263b6a7dc6fa432c8427ad57dc19258` /
  `92dd0be43ac74c1210a5babdea256eec271ee430cf5c2659bb8ce69b923137a8`
- input manifest / formation schema file SHA:
  `a4a4a99ce3edab30aa52c41079aebdd47d7bc9f2fb93c9c44b84b98a5e81fbad` /
  `1f9dec7ac0025213629f45685d772e27fa067169cf3ee610a0c59cd04bcc1470`
- truth logical SHA:
  `e9067327058431278a0fd994e8e6005b76ab99acbd3942118974599afb69a8d0`
- submission SHA: submission未実施。
- model SHA / manifest SHA: 学習モデルがないため非該当。
- gzip生成物はdecompressed content SHAを主証拠として記録した。

## 解釈

地層条件付き候補はexp226と相関`0.999999782`で、既存K16のごく小さい変形だった。
一定のadd-one headroomは5/5 foldsで存在するが、directでは平均・tailとも悪化し、
fixed12への増分も採用閾値の半分未満だった。今回の固定soft factorは既存supportを
安全に維持できた一方、地層的に不整合なdonorを有効に識別する強さが不足した。

weight強度、surface、signature、donor数、bandwidthの救済gridは事前禁止事項なので
行わない。exp362のlocal donor support branchと本branchのnegative evidenceを維持し、
同じK16 donor kernel上の追加条件付け案を新規backlogへ増やさない。

## 次

branchを閉じる。current-test生成、selector組み込み、inference、submission、
救済versionは行わない。独立した既存PF/Beam候補の優先順位は変更しない。
