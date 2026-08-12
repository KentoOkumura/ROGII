# タスクリスト

## TODO

- なし。

## 実行中

- なし。

## 完了

- backlog項目を作成した。
- 実験scaffoldを作成した。
- requirements、design、tasklistを作成した。
- 12候補semantic replacement contractを固定した。
- 実行量とcontrol非再学習を固定した。
- 再現性、leakage、scientific gateを固定した。
- canonical notebookをmarkdown-only placeholderにした。
- 2026-07-31: ユーザーの実装承認を記録した。
- Jupytext percent形式のcompact self-contained train候補とipynbを別名で実装した。
- exp374 allowlist / SHA、12候補順、4 changed / 8 unchanged、formula、
  88/74列schemaの自動guardを実装した。
- truth-late、global key join、source-fold非利用、Stage A schema自然生成、
  scientific gateのcontract testを実装した。
- exp492 / exp493 sibling test 17件、Jupytext test、構文、import-only確認を通した。
- canonical notebook、Kaggle package/push/runは変更・実施していない。
- 2026-07-31: ユーザーの実行依頼をcanonical採用・package・run承認として記録した。
- package前に1 variant / 2 objectives / outer 5 / inner 4 /
  40 CPU booster、control再学習0、GPU/downstream/inference/submission 0を再確認した。
- Kaggle version 1は親config未解決で学習前停止し、SHA固定親configをbootstrapへ
  同梱した。
- Kaggle version 2は全40 boosterと科学readout後のimportance集計で停止し、
  long-form schemaに合わせて修正した。
- ユーザー承認を受け、version 3を追加40・累計80 CPU boosterで再実行した。
- version 3は40/40 booster、technical / leakage / selector score guardをPASSし、
  hard RMSE `8.616237400`を得た。
- 3/5 folds、by-well p95 `+0.540095855 ft`、worst
  `f6d009f4 +10.472288433 ft`によりscientific gateをFAILした。
- decision `FAIL_CLOSE_FIXED12_STUDENT_T_REPLACEMENT_SELECTOR`を記録し、
  weight / threshold / domain / gate救済、downstream、inference、submissionなしで
  branchを閉じた。
- 必要な小型成果物だけを選択取得し、主要SHAを照合した。
- `experiment_summary.md`、`KAGGLE_DIRECTION.md`、実験記録を更新した。
