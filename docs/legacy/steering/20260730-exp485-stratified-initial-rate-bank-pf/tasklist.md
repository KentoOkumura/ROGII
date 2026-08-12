# タスクリスト

## 未着手

- なし。

## 進行中

- なし。

## ブロック中

- なし。

## 完了

- exp268と現行exp404初期化を確認した。
- 5 center equal-strataの単一PF候補、gate、実行量、再現性を固定した。
- backlog、steering、design-only scaffoldを作成した。
- compact self-contained Stage 0 train実装と正規Notebookを作成した。
- 5×100 allocation、interleave、duplicate/fallback、stable seed、
  duplicate-center exp404 bitwise parityの契約test 7件を作成した。
- strict experiment validationとrun-on-push実行用Kaggle package準備を完了した。
- Kaggle private CPU version 1でfixed32 Stage 0を完走した。
- 14 checks中13 PASS、full runtime projectionだけが
  `30,894.444 > 30,600 sec`でFAILした。
- 事前登録どおりparameter/gate救済なしでbranchをfail-closedとした。
- 元のruntime gate FAILを保持したままStage 1へ進むユーザー例外承認を記録した。
- runtime例外を明記したStage 1 self-contained train実装を追加した。
- canonical Kaggle kernel version 2で全773 wells、98,944 seed-well、
  49,472,000 particle startsを完了し、target-free成果物をfreezeした。
- version 2がtruth-late readout中の保存HMM integrity checkで停止したこと、
  freeze前のtruth/control/fold/hidden-like readが0であることを確認した。
- version 2凍結成果物をSHA固定したprivate Datasetへ保存し、Kaggle自動展開版を
  含むresume loaderの全件検証を通した。
- SHA固定したversion 2 target-free成果物からtruth-late評価を再開した。
- canonical Kaggle kernel version 3をCOMPLETEまで監視し、CVとgateを記録した。
- technical 19/19 PASS、candidate `11.092618091`、保存control
  `10.914522073`、scientific gate FAILを確認した。
- 事前登録どおりinference/submissionと同一OOF救済を行わずbranchを閉じた。
