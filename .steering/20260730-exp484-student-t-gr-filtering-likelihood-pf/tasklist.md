# タスクリスト

## 未着手

- なし。

## 進行中

- なし。

## ブロック中

- inference、submissionはStage 1 scientific gate FAILにより禁止。

## 完了

- exp374/exp404/exp417の根拠を確認した。
- fixed df=4 filtering尤度、実行量、gate、再現性を確定した。
- backlog、steering、design-only scaffoldを作成した。
- ユーザー依頼`exp484を実装してください`をimplementation承認として記録した。
- Jupytext percent形式のcompact self-contained Stage 0 train候補と正規train
  Notebook、fail-closed inference guardを実装した。
- stable-hash fixed32 manifestをtarget-freeに固定した。
- Student-t formula、中心2次近似、finite weight、exp404 input parity、
  stable seed、truth-late、SHA、Notebook contractの専用testを作成した。
- 専用test `10 passed`、Jupytext roundtrip、py_compile、ruffをPASSした。
- Kaggle CPU kernel version 2でStage 0を完了し、16/16 technical gateを
  PASSした。
- 追加依頼`Stage1へ進んでください`を全773 wells Stage 1の実装、
  canonical package、push/run承認として記録した。
- 全773 wellsのStage 1 truth-late CVと固定promotion gateを実装した。
- 専用test `13 passed`、Jupytext、py_compile、Ruff、strict exp/template
  validationをPASSした。
- 同じcanonical Kaggle CPU kernelへStage 1 version 3をpushし、
  `RUNNING`を確認した。
- canonical kernel version 3で全773 wellsのStage 1を完了した。
- 18/18 technical checksはPASSしたが、pooled gain、fold数、
  raw observed、typewell-purged、by-well p95/worstをFAILした。
- `terminal_close_without_student_t_or_pf_rescue`を適用し、
  inference / submissionを実行せずbranchを閉じた。
