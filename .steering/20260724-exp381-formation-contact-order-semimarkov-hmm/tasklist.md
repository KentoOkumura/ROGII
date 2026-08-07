# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし。Stage 0不合格のためStage 1 branchをfail closedした。

## 完了

- route、地層順、交差定義、target補正、7 state、duration prior、gateを確定。
- 実験ディレクトリとsteeringを作成。
- outer-train `FormationPlaneKNN(k=10)`、constant baseline、first-crossing、prefix単一offset、truth-late-join契約を固定。
- compact self-contained Stage 0 train / fail-closed inference候補を実装。
- 専用test 10件、Ruff、py_compile、Jupytext round-trip、strict実験validationをPASS。
- Kaggle version 1のsource formation全有限guard ERRORを診断し、
  formation別finite outer-train donor固定k=10へ修正。
- Kaggle private CPU version 2を完了し、期待15 artifactとSHAを検証。
- coverage、MD位置、順序、5/5 folds、constant比はPASSしたが、
  contact-TVT RMSE `44.7701 ft > 15 ft`で固定AND gateをFAIL。
- Stage 1、HMM/PF/Beam、inference、submissionを実行せずbranchを閉じた。
