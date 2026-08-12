# タスクリスト

## TODO

- なし。branchはclosed。

## 進行中

- なし。

## ブロック中

- inference、submissionは未承認。
- rescue、再実行は禁止し、branchをclosedとして維持する。

## 完了

- 2026-07-23: exp358として採番し、exp307依存を削除してexp209へ直結した。
- 2026-07-23: sigmaをexp209固定とし、missing-distance weightだけへ変更を限定した。
- 2026-07-23: Stage 0 technical gate、Stage 1科学gate、実行量を固定した。
- 2026-07-25: Stage 0 compact self-contained train候補とfail-closed inference候補を
  Jupytext percent形式で実装した。正規Notebookは未変更。
- 2026-07-25: leading/internal/trailing gap、all-missing fallback、式の厳密一致、
  truth-free suffix surface、technical gate、inference停止境界の専用test 9件を追加した。
- 2026-07-25: ユーザー実行承認によりStage 0候補を正規train Notebookへ採用し、
  canonical Kaggle CPU packageを生成・検証した。
- 2026-07-25: 初回pushのKaggle title 50文字制約を特定し、科学条件を変えず
  canonical id/titleを49文字へ短縮した。
- 2026-07-25: Kaggle private CPU version 1（id_no `128528105`）を完了し、
  3,783,989 rows / 773 wells、truth read・HMM・model・booster・control rerun各0、
  Stage 0 technical gate 23/23 PASSと生成物8件を記録した。
- 2026-07-25: ユーザーがStage 1への進行を別途承認した。実行量を
  1 fixed variant / 5 reporting folds / 773 exact-HMM well-runs、
  model・booster・parent-control再実行各0に固定した。
- 2026-07-25: Stage 1 compact self-contained trainを実装し、正規train
  Notebookへ採用した。dedicated 14件、common込み18件、Jupytext round-trip、
  Ruff、`validate-exp`、package/bootstrap/remote SHA一致をPASSした。
- 2026-07-25: canonical private CPU version 2（id_no `128528105`）を
  `17475.557881 sec`で完了した。direct RMSE
  `12.012570`はexp209 `11.938287`より`0.074283 ft`悪化し、
  改善fold 0/5、required scope、by-well p95/worst、fixed LikPF 50:50をFAILした。
- 2026-07-25: formal technical gateの唯一のfalseはpost-CSV bit-exact weight
  guardで、753 / 1,200,837 missing rowsの最大差`5.551e-17`と切り分けた。
  scientific FAILは独立して明確なため再実行せず、
  `missing_distance_exp209_failed_close_without_rescue`で閉じた。
