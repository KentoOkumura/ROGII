# タスクリスト

## 変更点

設計済みscaffoldへcompact self-contained notebook、fold-safe readout、negative control、SHA、テストを追加する。

## 実行

- [x] compact trainをcanonical notebookへ採用した。
- [x] Kaggle private CPU version 1を実行し、ログと必要artifactを検証した。
- [x] metrics、result、summary、backlog、後続dependencyを更新した。

## 完了

- [x] 仮説、route、親、群定義、fold、truth境界を固定した。
- [x] 1 variant / 5 folds / 0 model / 0 booster / 0 decoderを固定した。
- [x] 実装・inference・submissionをdisabledにした。
- [x] compact self-contained Jupytext train/inference sourceと別名notebookを作成した。
- [x] fold-safe pair/group table、well等重みHuber affine、identity shrinkageを実装した。
- [x] leakage assertion、SHA manifest、3 stress surface、2 negative control、promotion gateを実装した。
- [x] inference/submissionをfail-closedのまま維持した。
- [x] unit test、Jupytext round-trip、構文/F821、experiment/template validationを実施した。

## 結果と次

Kaggle version 1は正常完了したが、fit-RMSE R²とworst-well safetyが固定gateを失敗した。branchを閉じ、exp312〜320は停止する。inference/submissionは行わない。
