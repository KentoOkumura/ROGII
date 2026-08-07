# exp407_fold_safe_inverse_rmse_weighted_dual_selector_on_exp264 結果

## 状態

Stage B Kaggle CPU version 1は技術的に完走した。technical gateは全PASSしたが、
scientific全AND gateはFAILしたため、`fail_close_exp407_without_rescue`で閉鎖する。
Stage C/D、inference、submissionへ進めない。

## 仮説

exp264の12候補共有dual selectorで、各fit partition内の候補別TVT RMSEに反比例する
穏やかなtask weightを掛けると、候補を削除せずに共有modelのnegative transferを減らせる。

## 設定

- 親: `exp264_exp263_candidate_confidence_dual_selector`
- Route: `ml_model`
- 変更: training sample weightのみ
- 固定: 12候補、88特徴、legal domain、objectives、fold、sampling、LightGBM params
- v1実行範囲: 1 variant × 2 objectives × 5 folds = 10 CPU boosters
- control再学習: 0
- シード: 42

## 結果

| メトリック | 値 |
| --- | --- |
| Kaggle train | v1 COMPLETE、1,531.430秒 |
| technical gate | PASS |
| scientific gate | FAIL |
| hard-primary OOF RMSE | 8.668141 |
| Public LB | 未実行 |
| Private LB | 未実行 |

### 親exp264 Stage B v5との比較

| 指標 | exp407 | 親 | exp407 - 親 | fold条件 |
| --- | ---: | ---: | ---: | --- |
| expected-error MAE | 3.798670 | 3.795801 | +0.002869 | 改善2/5、FAIL |
| within10 logloss | 0.360461 | 0.359972 | +0.000489 | nonworse 1/5、FAIL |
| within10 Brier | 0.112648 | 0.112451 | +0.000197 | nonworse 2/5、FAIL |
| hard-primary RMSE | 8.668141 | 8.587004 | +0.081137 | nonworse 1/5、FAIL |

pooled toleranceだけならwithin10 logloss/BrierはPASSしたが、fold再現性を満たさない。
expected-error MAEとhard-primary RMSEはpooled条件もFAILした。

### tail / robustness

| scope | exp407 - 親 RMSE | 判定 |
| --- | ---: | --- |
| near 0–250 | +0.005087 | PASS |
| 1000+ | +0.091228 | FAIL |
| hidden-like spatial | +0.103759 | FAIL |
| hidden-like typewell-purged | +0.079052 | FAIL |
| worst well `52f1e77a` | +16.226863 | FAIL |

## 実装確認

- fit-partition限定inverse-RMSE weightと最終`[0.5, 1.5]` range fail-closedを実装
- 両objectiveのtraining rowsだけへ同一weightを渡し、validation / metricはunweighted
- 5 foldのweight / sampling / truth-read / feature content SHA監査を実装
- 保存済みparent v5とのfold / near / 1000+ / hidden-like / worst-well全AND gateを実装
- 専用synthetic test 9件、親selectorとの関連回帰を含む26件、py_compile、Ruff、
  Jupytext round-trip、strict validationをPASS
- 全体testは1,159件PASS・7件skip・4件既存領域FAIL。FAILは未変更のexp293
  contract SHA 2件とexp296 stale status/approval assertion 2件で、exp407由来は0
- 実行時technical gateは、3,783,989 base rows、45,407,868 candidate-long rows、
  88特徴、12候補、10 models / SHA 10-of-10、candidate order、legal domainを全PASS
- weightはfold別平均1、全fold範囲`0.646822–1.262666`、forbidden truth read 0、
  fit-valid well overlap 0、validation/metric weightなし

## 再現性

- deterministic anchor: false
- seed policy: 親のseed 42とdeterministic sampled row IDsを継承
- kernel: `kentookumura/exp407-inverse-rmse-dual-selector-exp264-train` v1
- kernel id_no: `128636600`
- runtime: Kaggle CPU、internet off、private
- package config SHA: `0d7252823acb5d97cb5cb8782fb174f11d5debfdbc037616755143085b4493d9`
- feature schema SHA:
  `aaef4ffdd90667893b099b76a52f1957b22197aea9cee5e5b57bc81048ddd3a4`
- fold weight table SHA:
  `ecf3e93b161e2a173ed3cadbf69cc369d367f38d939d8463be1624e4c851922b`
- weight manifest SHA:
  `7234351edcebbd852ea8d4b771258e676bbb5295939105226fb61c92033be067`
- model manifest SHA:
  `1fce3716fc7f545e0ea883e8ee71b05174d141212334f7c01913b32ef38adfd4`
- candidate-score OOF SHA:
  `d993b806d92c2462c1509f110669b272b27d48806c0280a2cf54e87c7f32f1e8`
- compact-meta OOF SHA:
  `a88503f506985ae1b25391234abc753e39bd1d81b52b98e827486fa6102b9672`
- gate SHA:
  `2ae8cb3eaafd3f11558035f27c4afdf0be70468e8edacb0c8703c6b949a99962`
- prediction SHA: 対象外
- submission SHA: 対象外
- rerun result: 未実行

## 解釈

fold-safe inverse-RMSE weight自体は設計どおり生成・適用できた。しかし、候補単体の
global qualityを共有modelのtask importanceへ変換しても、row-localな候補順位の学習は
改善しなかった。nearだけは非劣化だった一方、1000+、hidden-like両面、worst wellが
悪化しており、global candidate qualityによる一様downweightはtailで必要な弱候補の
局所signalまで弱めた可能性がある。これは推測であり、今回のgate判定自体は保存済み親との
事前固定比較だけに基づく。

## 次

exp407はweight強度変更、inverse-square、clip/exponent grid、候補削除、same-OOF rescueを
行わず閉鎖する。exp264 corrected Stage B v5をselector anchorとして維持する。
別候補として、保存済み親/exp407 OOFだけを用いる0-boosterのcandidate-switch tail attribution
readoutを低優先度でbacklogへ置き、将来のtask-weight familyを再開する前の原因診断に限定する。
