# タスクリスト

## 未着手

- なし

## 進行中

- なし

## ブロック中

- Stage 1 PF、inference、submissionはStage 0 scientific gate FAILにより不適格。

## 完了

- backlog、実験scaffold、steeringを作成した。
- marginal state、Stage 0/1 gate、seed policy、実行量を確定した。
- 2026-07-25、Stage 0 compact self-contained train source / Notebookを実装し、
  placeholderの正規train Notebookを置換した。
- known-prefix 128/64 predictive NLL、saved exp072 `likpf_mean` suffix weak posterior、
  truth-late freeze、circular control、fold / hidden-like / AND gateを実装した。
- inference placeholderをsubmission非生成のfail-closed Notebookへ置換した。
- Stage 1 PF helper、Kaggle package、推論、submissionは実装していない。
- 2026-07-25、canonical private CPU kernel version 1でStage 0を実行した。
- technical gateはPASS、known-prefix NLL gainとweak massのscientific gateを
  FAILし、`stage_0_failed_close_without_rescue`でbranchを閉じた。
- Stage 1 PF、inference、submission、同一OOFでの救済再実行を行わないと確定した。
