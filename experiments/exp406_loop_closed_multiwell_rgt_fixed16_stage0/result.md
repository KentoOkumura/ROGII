# exp406_loop_closed_multiwell_rgt_fixed16_stage0 結果

## 状態

Kaggle private CPU version 1（id_no `128637170`）を完了した。
固定Stage 0のtechnical gateは15項目中12項目PASS、3項目FAILで、
decisionは`close_exp406_without_parameter_rescue`。full OOF Stage 1、
current-test、inference、submissionへ進まずbranchを閉じる。

## 固定仮説

current-testで観測可能なhorizontal GRからpairwise TVT対応を作り、
全well graphをTVT ft単位でloop-closeすれば、target RGT supportと
prefix予測signalが16 wellsで非退化になる、という仮説を検証した。
fixed16、H256/H128、12 donors、top4 edge、±55/5 ft、Huber IRLS 10回、
circular control、gateは実行後に変更していない。

## Stage 0結果

| gate | 観測値 | 閾値 | 判定 |
| --- | ---: | ---: | --- |
| graph query coverage | 0.451157 | >=0.90 | FAIL |
| connected target coverage | 1.000000 | 1.00 | PASS |
| finite loop-closed row coverage | 0.755026 | >=0.95 | FAIL |
| fundamental cycles | 9,272 | >=30 | PASS |
| raw / solved cycle residual p95 | 70.0 / 7.1e-15 ft | solved <=5 ft | PASS |
| cycle p95 reduction | 1.000000 | >=0.50 | PASS |
| real minus circular NCC | 0.874148 | >=0.10 | PASS |
| real better than circular folds | 5/5 | >=4/5 | PASS |
| projected 773-well runtime | 65,543.109 sec | <=30,600 sec | FAIL |
| peak RSS | 0.544994 GB | <=25 GB | PASS |

fixed16実測はtarget-free `1,356.649 sec`、全diagnostic `1,356.666 sec`。
full投影は約18.21時間で上限の2.14倍だった。

query coverageは16/16 wellsで0.90未満、finite row coverageが0.95以上だったのは
5/16 wellsだけだった。target側rejectionでは
`nonpositive_local_tvt_progress`が43.97%、NCC閾値未満が19.79%、
finite pair不足が16.24%を占め、retainedは0.63%だった。
GR signal自体はcircular controlを5/5 foldsで明瞭に上回り、
loop solverも9,272 cyclesを数値的に閉じたため、主失敗は
「GR対応が無信号」や「solver不安定」ではなく、
固定pairwise構築のsupport不足と計算量である。

target-free technical gateで停止したため、prefix512 truth joinと
exp226 K16 geometry replayは0回。prefix科学性能は未評価であり、
prefix FAILとは解釈しない。

## Safety・再現性

- source-target overlap、suffix truth、target Formation、hidden roleの
  freeze前read: `0 / 0 / 0 / 0`
- unknown-suffix prediction: 生成・保存なし
- loop gaugeに`solved_tvt`や禁止Formation/truth列なし
- model / booster / PF / HMM / Beam: `0 / 0 / 0 / 0 / 0`
- 生成物manifestのfile SHA: `8/8`一致
- summary SHA256:
  `e9332c3166c875ee663c41d3f7bec0d17c68f3902aa9eb372996d77920575413`
- gate SHA256:
  `677613e1d8d856a93e2a5d1fd65840cfd8dc80064facd503713454a042f0afb8`
- SHA manifest SHA256:
  `92f53159a550dfba0a37c4a657e7cb98660357d9279e56783062201ae2315321`

## 判断

固定契約どおりexp406をtechnical FAILで閉じる。fixed16再選択、
donor/window/shift/edge/NCC/Huber/gate調整、prefixだけの救済run、
full OOF、current-test、inference、submissionは行わない。

次候補はexp406のparameter rescueではない。Formation-derived exp386が
高いRGT coverageを持ちながらrouteを全棄却した段階だけを分解する
`rgt_edge_cycle_path_rejection_readout`を低優先度P4の独立候補として残す。
着手には別steeringとユーザー承認が必要。
