# タスクリスト

## 完了

- [x] `exp396_fold_safe_exp111_score_27_addonly_on_exp287` を採番した。
- [x] 親をexp287、clean tail controlをcorrected exp264、Routeを `ml_model` に固定した。
- [x] exp111 score系27列の名前・順序・派生契約を固定した。
- [x] outer 5 × inner 4 × 2目的 = 40 CPU boostersのstrict nested契約を固定した。
- [x] model固有48-column median、stable row sample、current-test outer-model再利用契約を固定した。
- [x] Stage A technical / score / resource gateを固定した。
- [x] Stage Bの448特徴、15 GPU boosters、control再学習0、promotion gateを固定した。
- [x] backlog、experiment summary、steering、experiment scaffoldをdesign-onlyで記録した。
- [x] Stage A実装の明示指示を受け、実装承認と実行承認を分離してconfigへ記録した。
- [x] Jupytext percent形式のcompact self-contained train候補とfail-closed inference候補を実装した。
- [x] target-free 48特徴、nested fold、stable sample、model固有median、10 core→27列、
  Stage A gate、実行量guardの専用testを追加した。
- [x] 既存正規`.ipynb`を上書きせず、candidate notebookへ変換して静的検証した。
- [x] 明示承認後に正規train notebookを採用し、Kaggle private CPUへpackage/pushした。
- [x] 0-booster preflightで入力SHA、fold/well非重複、27列schema、実行量を確認し、
  16/16 checksをPASSした。
- [x] preflight manifest、nested fold manifest、logsをSHA固定して記録した。
- [x] 承認済み40 CPU boostersを同一canonical kernelのversion 2で実行した。
- [x] Stage Aのtechnical 22/22、scorer quality 6/6、runtime / memory gateを全PASSした。
- [x] 40 model、40 median、10 score partitionsと主要生成物SHAを記録した。
- [x] Stage B実装と15 GPU booster T4実行の別承認を得た。
- [x] 保存済みformation/score core再利用、448列matrix、15 model OOF、
  fixed promotion gate、artifact SHAを実装した。
- [x] 正規train NotebookへStage Bを採用し、専用11 testsと静的検証をPASSした。
- [x] Stage B private T4 version 1で固定15/15 GPU boostersを完走した。
- [x] OOF、fold、scope、by-well、clean-tail gateと主要生成物SHAを記録した。
- [x] 固定promotion gate 1/6 PASSを根拠にbranchをfail-closedで閉鎖した。

## 未着手・別承認が必要

- [x] 40 CPU boostersのKaggle private CPU実行を別途承認した。
- [x] Stage Aが全PASSした場合だけStage B実装の承認を得る。
- [x] Kaggle GPU push前に1 variant × 3 configs × 5 folds = 15 boosters、
  control再学習0を再提示して明示承認を得る。
- [x] Stage BのOOF、fold、scope、by-well、clean-tail gateとartifact SHAを記録する。
- [x] promotion gate FAILのためinferenceを実装・実行しないと確定する。

## 閉鎖

- Stage Bは15/15 boostersを完走したが、pooled/fold/scope/by-well p95/worst-well gateをFAILした。
- inference / submission、same-OOF rescue、gate緩和、再学習へ進まない。

## 次のアクション

exp287をtrain-side parent anchorに維持する。保存済み生成物だけを使う0-booster転移失敗原因readoutは
低・P4とし、新しい独立した必要性と承認がない限りexp396を再開しない。
