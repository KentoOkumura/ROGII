# タスクリスト

## 未着手

- なし。

## 進行中

- なし。

## ブロック中

- なし。Stage 1はStage 0 scientific gate FAILにより不適格となり、分岐を閉じた。

## 完了

- backlog、実験scaffold、steeringを作成した。
- PF state、quota、Stage 0/1 gate、seed policy、実行量を確定した。
- 2026-07-25、Stage 0のcompact self-contained train source / notebookと正規train
  notebookを実装した。
- truth-late freeze、decompressed SHA、3固定軌道、circular control、GroupKFold、
  1000+ / hidden-like 2面、technical / scientific gateを実装した。
- 構文、ruff、9件の専用test、Jupytext round-trip、`validate-exp`を通過した。
- inference placeholderはsample submissionを作らないfail-closed notebookへ置換した。
- Stage 1 PF、inference、submissionは実装していない。
- 2026-07-25、Kaggle private CPU version 1（id_no `128528103`）を
  `267.914282461 sec`で完了した。
- technical gateは全PASSした。
- overall top1 `0.469591`、MRR gain `+0.276771`、1000+ / hidden-like 2面の
  RMSE方向はPASSした。
- real-circular top1差は`+0.005576 < 0.03`、passing foldsは`2/5 < 4/5`で
  scientific gateをFAILした。
- decisionを`stage_0_failed_close_without_rescue`とし、Stage 1 PF、inference、
  submission、parameter rescueなしで完了・閉鎖した。
