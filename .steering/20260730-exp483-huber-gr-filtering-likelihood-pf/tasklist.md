# タスクリスト

## 未着手

- なし。

## 進行中

- なし。

## ブロック中

- なし。科学ゲートFAILによりbranchを閉鎖した。

## 完了

- exp389/exp430/exp404/exp417の結果を確認した。
- Huber filtering尤度、固定値、実行量、gate、再現性契約を確定した。
- backlog、steering、design-only実験scaffoldを作成した。
- compact self-contained Stage 0候補をJupytext percent形式で実装した。
- formula/no-op toy PF、truth-late、stable seed、SHA readbackの専用testを実装した。
- 2026-07-30の追加依頼で正規train Notebook採用、Kaggle package/push、
  fixed32 Stage 0実行が承認された。
- 正規train Notebookを採用し、canonical private CPU kernel version 1を実行した。
- Stage 0は156,088 rows / 32 wellsを処理し、10 technical gateを全PASSした。
- fixed32 report-onlyはcandidate `9.811671590`、保存control `9.616740808`、
  差`+0.194930782 ft`、improved wells `18/32`だった。
- Stage 0後の時点ではStage 1、inference、submissionを実行していなかった。
- 2026-07-30の追加依頼で全773 wells Stage 1が別承認された。
- Stage 1 truth-late CV、保存control/HMM、5 folds、stress scope、
  by-well tail、fixed HMM/PF 50:50の全AND gateを実装した。
- canonical private CPU kernel version 2へStage 1をpushした。
- version 2は全773 wells / 3,783,989 rowsを処理して`COMPLETE`した。
- Stage 1 technical gateを全PASSし、実行量、truth-late、parity、SHA、
  runtime、RSSを確認した。
- candidate / 保存controlは`11.095404595 / 10.914522073 ft`で、
  `0.180882522 ft`悪化、改善foldは`3 / 5`だった。
- scope、by-well p95/worst、fixed HMM/PF 50:50を含む科学AND gateをFAILした。
- `terminal_close_without_huber_or_pf_rescue`を記録し、backlogから削除した。
- inference、submissionを実行していない。
