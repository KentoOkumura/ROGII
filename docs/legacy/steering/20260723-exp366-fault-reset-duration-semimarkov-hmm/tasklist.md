# タスクリスト

## 未着手

- なし。

## 進行中

- なし。

## ブロック中

- Stage 0 scientific FAILのためStage 1、inference、submissionは不適格。

## 完了

- backlog、実験scaffold、steeringを作成した。
- trigger、reset state、Stage 0/1 gate、禁止事項、再現性、実行量を確定した。
- 2026-07-25、Stage 0 compact self-contained train候補とfail-closed inference候補を実装した。
- 13 branchの固定512行preflight、truth前freeze、late truth / hidden-like readout、
  gate判定、専用testを実装した。
- 2026-07-25、正規train Notebookを採用し、Kaggle private CPU version 1をpush。
  raw identity SHA adapter不一致で科学処理前にfail-closed。
- raw identity専用adapterだけを親契約へ合わせ、専用12 tests、Ruff、py_compile、
  Jupytext、strict validation、package manifestを再検証した。
- Kaggle private CPU version 2（id_no `128543224`）を完了。
  technical PASS / scientific FAIL、passing folds `0/5`、
  decision=`stage0_failed_close_without_semimarkov_hmm`。
- 小さい必要成果物だけを選択取得し、manifest SHAを検証した。
- Stage 1を未実装のまま閉じ、inference / submissionを生成していない。
