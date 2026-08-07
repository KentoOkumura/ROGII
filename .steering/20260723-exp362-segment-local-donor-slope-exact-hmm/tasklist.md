# タスクリスト

## TODO

- なし

## 進行中

- なし

## ブロック中

- なし

## 完了

- 2026-07-23: `exp362_segment_local_donor_slope_exact_hmm` の steering と実験 scaffold を作成した。
- 2026-07-23: exp226 予測非依存、direct HMM、K16 local-gradient prior、fold-safe donor、固定 fallback、比較条件、成功条件、再現性契約を確定した。
- 2026-07-23: 実装、Kaggle push、実行、推論、提出を未着手のまま保持した。
- 2026-07-23: compact self-contained train 候補へ stable 5-fold、fold-safe K16 donor ledger、
  target local-gradient schedule、residual-rate exact HMM、prediction freeze、late truth/control join、
  support/fallback・fold・distance・1000+・hidden-like 2 面・by-well・SHA 出力を実装した。
- 2026-07-23: inference は sample submission を作らない fail-closed compact 候補として実装した。
- 2026-07-23: 専用テスト 10 件、exp209 kernel bitwise parity、Jupytext train/inference、
  `py_compile`、Ruff F821、`make validate-exp` strict を PASS した。
- 2026-07-23: compact train候補を正規notebookへ採用し、1 variant / 5 reporting folds /
  773 HMM runs / 0 booster / control再実行0を確認してKaggle CPU version 1をpushした。
- 2026-07-24: version 1（id_no `128368310`）が`19777.653141 sec`でCOMPLETE。
  notebook technical gateはPASS、scientific gateは3/5 foldsとworst `+52.741426 ft`でFAILした。
- 2026-07-24: post-run target prior監査でlocal gradient採用0/12,368、全segmentがprefix rateへ
  退化していたことと、保存fallback flagの同名field上書きバグを確認した。branchをfail closedとし、
  parameter/fallback/HMM/blend救済、再実行、inference、submissionを行わない。
