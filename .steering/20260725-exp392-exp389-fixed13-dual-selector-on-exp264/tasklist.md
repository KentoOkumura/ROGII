# タスクリスト

## TODO

- なし。

## 進行中

- なし。

## ブロック中

- downstream TVT、current-test candidate生成、inference、submissionは対象外。

## 完了

- 2026-07-25: exp391まで使用済みを確認し、exp392を採番した。
- 2026-07-25: `docs/06_reproducibility.md`を確認した。
- 2026-07-25: exp388 fixed13契約を構成参照にし、Student-tを併用しない
  Huber単独add-one設計を固定した。
- 2026-07-25: ユーザー指示をimplementationと40 CPU booster runの承認として
  記録した。
- 2026-07-25: compact train/inference、exp389 fixed13 helper、contracts、
  専用test、正規Notebookを完成した。
- 2026-07-25: Jupytext、py_compile、Ruff、strict validation、専用・共通
  55 testsをPASSした。
- 2026-07-25: 1 variant / 2 objectives / outer 5 / inner 4 /
  40 CPU boosters / parent control retraining 0を再確認し、packageの
  metadata/bootstrap/kernel source/SHAを監査した。
- 2026-07-25: canonical kernel version 1をpushし、RUNNINGを確認した。
- 2026-07-25: version 1を`3666.541645 sec`で完了し、40/40 selector models、
  technical / leakage / score guard PASSを確認した。
- 2026-07-25: fixed13 RMSE `8.769791682`、親比`+0.117259726 ft`、
  改善2/5 folds、by-well p95`+0.774302299 ft`で科学gate FAILと判定した。
- 2026-07-25: decision `FAIL_CLOSE_FIXED13_SELECTOR_BRANCH`を記録し、
  downstream TVT、inference、submissionなしでbranchを閉じた。
