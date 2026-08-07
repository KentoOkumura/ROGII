# exp371_exp333_fixed13_dual_selector_on_exp264 結果

## 状態

Kaggle CPU version 3でStage Aとnested Stage Cを完了した。technical / selector
score / leakage guardはPASSしたが、事前固定したwell-level safety guardをFAILした。
Stage C decisionは`FAIL_CLOSE_FIXED13_SELECTOR_BRANCH`のまま維持する。

その後、ユーザー明示例外により、pooled改善を下流TVTで確認するStage Dを実行した。
`clean273 + fixed13 compact77 = 350`特徴、3 configs × 5 folds = 15/15 GPU
boostersをKaggle T4 version 1で完走し、保存済みexp264 Stage D v3と比較した。
平均RMSEと全集計scopeは改善したが、by-well p95とworst-wellの固定安全条件をFAILした。
最終decisionは`STAGE_D_MEAN_IMPROVED_TAIL_GATE_FAILED_CLOSE_NO_INFERENCE`である。

## 仮説

exp361でfixed12へのadd-one oracle headroomを示したexp333は、corrected exp264と同じ
target-free dual selectorへ13本目として追加すれば、有効区間をcandidate value・shape・
bank disagreementから識別できる。

## 固定設定

- 親: corrected `exp264` Stage C v6
- candidate: exp263 fixed12 + `exp333_segment_offset`
- outer / inner: 5 / 4、well単位
- objectives: `pred_abs_error`, `p_within10`
- compact: 74 -> 77
- train: 40 CPU boosters
- parent/control retraining: 0
- downstream TVT: 15 GPU boosters
- inference / submission: 0 / 0

## 変更点

親fixed12の候補順序、fixed fallback 7本、selector objective、sampling、
LightGBM設定は固定し、`exp333_segment_offset`だけを13本目としてprimary domainへ追加した。
exp333の保存source foldはprovenance-onlyとし、global key join後は親と同じexp263
selector foldでnested学習・評価した。

## Stage C 結果

| 指標 | fixed13 | 親fixed12 | 差分 |
| --- | ---: | ---: | ---: |
| pooled hard OOF RMSE | 8.419997 | 8.652532 | -0.232535 |
| near 0--250 ft | 1.640529 | 1.663645 | -0.023116 |
| 1000 ft以上 | 9.240731 | 9.503799 | -0.263068 |
| hidden-like spatial | 9.129719 | 9.536496 | -0.406777 |
| hidden-like typewell-purged | 9.008795 | 9.412065 | -0.403270 |

- parent比fold改善は4/5。fold 3だけ`+0.116619 ft`悪化した。
- exp333のprimary top1使用率はpooled `6.267989%`、全5 foldsで正だった。
- fixed fallback RMSE `8.238332`に対しては`+0.181666 ft`悪い。
- well単位では400/773改善、373/773悪化、中央値差は`-0.006019 ft`。
- by-well delta p95は`+0.861529 ft`で、事前上限`+0.25 ft`をFAILした。
- worst well `a48640d9`は`3.349108 -> 14.107105`、`+10.757997 ft`悪化した。
  同wellのexp333 top1率は`42.961165%`だった。
- ただし全wellのexp333使用率とRMSE差のPearson相関は`-0.070004`であり、
  単純な使用率thresholdだけを安全gateとする根拠はない。

## Stage C Gate

| Gate | 判定 |
| --- | --- |
| 40 models / 25 partitions / row数・SHA | PASS |
| exp333 global key join / selector-fold repartition | PASS |
| exp333 truth/error pre-freeze load 0 / source fold feature利用0 | PASS |
| selector 2目的のprior改善 | PASS |
| leakage audit | PASS |
| pooled / 4 folds / near / 1000+ / hidden-like | PASS |
| by-well p95 / worst well | FAIL |
| scientific integration gate | **FAIL** |

平均改善とexp333利用は実在するが、well間のtail regressionを抑えられなかった。
exp286、exp335、exp287と同じく、global CV改善とwell-level safetyが分離する失敗である。
同一OOFでcandidate weight、使用率threshold、domain、gateを調整せず、Stage Cの
fixed13 selector FAILは再分類しない。

## Stage D 結果

保存済みfixed13 compact 77列をclean 273列へadd-onlyし、親の保存済みfixed12
compact add-only OOFと同一行で比較した。

| 指標 | fixed13 + compact | 親fixed12 + compact | 差分 |
| --- | ---: | ---: | ---: |
| pooled OOF RMSE | 8.369996 | 8.460811 | -0.090815 |
| near 0--250 ft | 1.521613 | 1.583151 | -0.061537 |
| mid 250--1000 ft | 4.043133 | 4.099686 | -0.056553 |
| 1000 ft以上 | 9.203975 | 9.302283 | -0.098308 |
| hidden-like spatial | 9.116400 | 9.420315 | -0.303915 |
| hidden-like typewell-purged | 9.037914 | 9.341391 | -0.303477 |

- fold改善は3/5。fold 0/1は`+0.077893 / +0.047410 ft`悪化し、
  fold 2/3/4は`-0.295061 / -0.091289 / -0.197859 ft`改善した。
- well単位では389/773改善、384/773悪化、中央値差は`-0.002906 ft`。
- by-well delta p95は`+1.179312 ft`で、固定上限`+0.25 ft`をFAILした。
- worst well `e25f1537`は`4.706827 -> 9.344426`、
  `+4.637599 ft`悪化し、固定上限`+0.25 ft`をFAILした。

| Stage D gate | 判定 |
| --- | --- |
| pooled改善 | PASS |
| 改善fold 3/5以上 | PASS |
| near / 1000+ 非悪化 | PASS |
| hidden-like 2面非悪化 | PASS |
| by-well p95 `<=+0.25 ft` | FAIL |
| worst well `<=+0.25 ft` | FAIL |
| 総合AND gate | **FAIL** |

候補パス改善の信号は下流モデルでも平均値として再現したが、well間tail regressionを
抑えられなかった。Stage CとStage Dの両方で同じ安全上の問題が残ったため、
current-test inferenceとsubmissionへ進めず、このbranchを閉じる。

## 再現性

- seed: 42
- kernel: `kentookumura/exp371-exp333-fixed13-selector-train` version 3、
  id_no `128372803`
- runtime: `6761.965850 sec`
- feature schema SHA:
  `4665ca7317ddcb993326e66ee19aa908f4aeff5fe88b2b16bac3db12c35b665f`
- nested selector model manifest SHA:
  `c5b70f32d698056336fe98eddd87f9f5fb041adea3797b56ede16b756d44396d`
- nested compact manifest SHA:
  `534e8278ad0e0dddc04a94236e949e9a5680342138bdebfdbf22fd8fb4f08956`
- outer-valid candidate score SHA:
  `5601b369704d36b4e8e8fba342a153ce09a8016f465a1a0bc88bb8beccecd9df`
- exp333 OOF decompressed SHA:
  `f2ebc6f6ea243b45fdb785342b8815b3b04947f96d787d3017e5e2be7ff92e5a`
- parent exp264 score SHA:
  `a10b7848127f01bef522f4b17dfd1640c9784956892dc24fc1159e3869500abc`
- Stage D kernel:
  `kentookumura/exp371-exp333-fixed13-selector-tvt-train` version 1、
  id_no `128524177`
- Stage D runtime: `13619.488220 sec`
- Stage D model manifest SHA:
  `151d7a06eeb0dcce157faf36f4613b3f3704a34d167c6e39f3eaa267426ba8e1`
- Stage D OOF prediction SHA:
  `272325effac930cab0ff944ec9ed493a3ff2dceb4ae2a4844d482f99fc20ad3c`
- Stage D metrics SHA:
  `b1c9335ac0c4f5558ee43741fc14c8f2bc5e7921ba6803624494d4380d82b0de`
- Stage D reproducibility manifest SHA:
  `e2f39868333c3c17d81ab82c20541ad004f4971dc12b3e1d151d98c8b76a4238`
- Public / Private LB: 未実行

## 次

fixed13 selector単体のFAILとStage Dのtail gate FAILを保持する。同一OOFでの
weight、threshold、feature、gate救済は行わない。exp333固有またはfixed13再学習による
incumbent rerankingの原因を再訪する場合は、既存の0-booster attribution案を
独立承認して実施する。inferenceとsubmissionは行わない。
