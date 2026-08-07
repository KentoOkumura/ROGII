# タスクリスト

## 未着手

- なし

## 進行中

- なし

## ブロック中

- なし。Stage 1はStage 0 scientific FAILにより不適格。

## 完了

- 既存未採番backlogをexp370として採番した。
- 実験scaffold、steering、trigger/atlas/rejuvenation設計を確定した。
- Stage 0/1 gate、seed policy、実行量を固定した。
- 初期設計時はNotebook / PF helper / Kaggle packageへ着手せず、実装承認を待った。
- 2026-07-25、Stage 0のcompact self-contained train source / notebookと、
  sample submissionを作らないfail-closed inference source / notebookを実装した。
- 500 particles × 1 seedのexp072互換diagnostic PF、pre-resampling ESS、
  known-prefix q99.5 GR change AND ESS/N <= 0.20 trigger、refractory 512を実装した。
- fold-safe atlas prototype、top3 / 10 ft separation、saved exp072 base control、
  truth-late freeze、AUC / circular / coverage / 5-fold / hidden-like gateを実装した。
- 構文、ruff、12件の専用test、Jupytext train/inference round-tripを通過した。
- 2026-07-25、ユーザーの `実行してください` により、Stage 0正規train Notebook採用と
  private Kaggle CPU package / push / runの承認を得た。
- inference placeholderは上書きせず、Stage 1 / inference / submissionは無効のまま。
- Kaggle version 1はcompetition mount resolverの欠陥で科学計算前にfail-closed。
- resolverを修正し、回帰testを追加してversion 2を実行した。
- version 2は773 diagnostic PF seed-well runsを完了し、technical gateをPASSした。
- trigger rate `3.527e-6`、AUC `0.499998`、atlas top3 coverage `0.076923`、
  saved base比gain `-0.769231`、passing folds `0/5`でscientific gateをFAILした。
- Stage 1、inference、submissionを実行せず、branchを閉じた。
