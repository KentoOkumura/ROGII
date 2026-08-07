# タスクリスト

## 未着手

- なし。

## ブロック中

- Stage 1、inference、submission。
- grid/support/variance/noise/rate/emission/gate救済。

## 完了

- 2026-07-29: exp443へ採番し、P3 representation候補とした。
- 2026-07-29: trapezoidal meanとlattice variance floor式を固定した。
- 2026-07-29: exp439との仮説差、fixed32 gates、truth-late、SHAを固定した。
- 2026-07-29: backlog、steering、実験scaffoldとdesign-only記録を作成した。
- 2026-07-29: 固定5-cell mean/variance-floor projectionを実装した。
- 2026-07-29: exp439 failure edge、joint exhaustive reference、truth-late、
  deterministic SHA、fail-closed execution/inference contractの専用testを追加した。
- 2026-07-29: compact self-contained train/inference候補を別名で生成し、
  正規Notebook scaffoldを保持した。
- 2026-07-29: 専用12 tests、exp439/443関連24 tests、Jupytext round-trip、
  py_compile、Ruff F821、strict experiment validationをPASSした。
- 2026-07-30: 正規train Notebook採用、Kaggle package、Stage 0 fixed32の
  実行承認を得た。Stage 1、inference、submissionは未承認のまま維持した。
- 2026-07-30: Kaggle private CPU version 1（id_no `129095370`）で32/32
  HMM wellsを完走した。数値contractはPASSしたが、runtime projectionと
  mechanism 4/6項目がFAILしたため`stage0_fail_closed`で終了した。
- 2026-07-30: 成果物を`artifacts/kaggle_v1`へ取得し、metrics / gate /
  prediction / moment audit / rate readout SHAを記録した。Stage 1、rerun、
  inference、submission、same-fixed32 rescueへ進まない。
