# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- 2026-07-23: exp359として採番し、exp323 dependencyを削除してexp281へ直結した。
- 2026-07-23: fixed window/score/lambda、Stage 0 rank gate、absolute ceilingを固定した。
- 2026-07-23: exp305/343をnegative referenceに限定し、救済gridを禁止した。
- 2026-07-25: ユーザー依頼によりStage 0 compact self-contained train候補と
  fail-closed inference候補を実装した。
- 2026-07-25: exp226 window profile/score、13-shift normalization、
  softmax posterior-SD lambda、stable SHA permutation、saved exp280 center-block lookup、
  truth late-join、固定gateの専用test 8件を通過した。
- 2026-07-25: Jupytext変換・round-trip、py_compile、Ruff、strict experiment
  validationを通過した。正規Notebookは未変更。
- 2026-07-25: ユーザーの「実行してください」により、正規train Notebook採用と
  Kaggle private CPU Stage 0の1回実行を承認した。
- 2026-07-25: canonical private CPU version 1（id_no `128528648`）を完了した。
  技術gateはPASSしたが、window scoreはsaved exp280 control比でpooled MRR
  `-0.022264`、top3 `-0.033496`、改善fold各`0/5`、stress 3面も負方向だった。
- 2026-07-25: 固定Stage 0 gate FAILにより、Stage 1、inference、submissionへ
  進まず、救済なしで閉じた。
