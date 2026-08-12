# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- [x] 2026-07-26: backlogへ設計確定・未実装候補を追加した。
- [x] 2026-07-26: steeringと実験scaffoldを作成した。
- [x] 2026-07-26: routeを`ml_model`、親をexp264、変更変数をsample weightだけに固定した。
- [x] 2026-07-26: fold-safe weight式、clip、Stage B gate、FAIL時no-rescueを固定した。
- [x] 2026-07-26: Stage B/C/Dのbooster数と承認境界を固定した。
- [x] 2026-07-26: ユーザーからStage B implementation-onlyの明示承認を得た。
- [x] 2026-07-26: Jupytext percent形式のcompact self-contained train候補を別名で作成した。
- [x] 2026-07-26: 親exp264のcandidate contract、88列schema、fold、sampling、
  LightGBM設定を固定して取り込んだ。
- [x] 2026-07-26: fit partition限定の候補別RMSE、inverse-RMSE weight、
  truth-read ledger、sampling / feature content SHAを実装した。
- [x] 2026-07-26: 同じweightを両objectiveのtraining rowsだけへ適用し、
  validation / metricをunweightedのままにした。
- [x] 2026-07-26: leakage、normalization、range fail-closed、candidate order、
  sampling contract、model count、Notebook境界のsynthetic testを追加した。
- [x] 2026-07-26: 正規Notebook採用とStage B Kaggle実行の明示承認を得た。
- [x] 2026-07-26: 1 variant、2 objectives、5 folds、10 CPU boosters、
  control再学習0を再確認した。
- [x] 2026-07-26: package/source config SHA、private/CPU/internet off metadata、
  bootstrap configを確認した。
- [x] 2026-07-26: canonical kernelへversion 1をpushした。
- [x] 2026-07-26: v1の`COMPLETE`、実行時間、logsを確認した。
- [x] 2026-07-26: 必要な小容量artifactだけを取得し、weight table、
  model manifest、OOF content SHAを監査した。
- [x] 2026-07-26: technical gate PASS、scientific全AND gate FAILを確定した。
- [x] 2026-07-26: `result.md`、`metrics.json`、`SESSION_NOTES.md`、
  `experiment_summary.md`、`KAGGLE_DIRECTION.md`へ結果を記録した。
- [x] 2026-07-26: Stage C/D、inference、submissionを閉じ、
  `fail_close_exp407_without_rescue`で完了した。
