# exp459 タスクリスト

## 未着手

- なし。

## 進行中

- なし。

## ブロック中

- なし。Stage 1、inference、submissionは未承認待ちではなく
  Stage 0 mechanism FAILにより禁止。

## 完了

- exp404/417のPF契約と保存controlを確認した。
- exp444のacceleration state、transition、runtime negative contextを確認した。
- exp367のsigned-curvature PF negative contextを確認した。
- `docs/06_reproducibility.md`のPF seed / raw-test regeneration / SHA契約を確認した。
- `(TVT, U-rate, U-acceleration)`の状態、更新順、初期分布、resampling方針を固定した。
- Stage 0/1の実行量、technical/mechanism/scientific gate、禁止事項を固定した。
- backlog、steering、design-only実験scaffoldを作成した。
- Jupytext percent形式のcompact self-contained Stage 0候補を実装した。
- 3値state、transition row-sum、境界fold、更新順、`-delta_Z` identity、
  stable seed、独立RNG、zero-acceleration exp404 bitwise parityのtestを実装した。
- candidate prediction、acceleration ledger、runtime ledger、input/scientific
  contract、全SHAをtruth attachment前にfreezeする実装を追加した。
- truth-late direction / persistent episode / matched-control readoutと
  fail-closed gateを実装した。
- 正規train Notebookをself-contained 23セル構成へ採用した。
- `pytest` 10件、Jupytext roundtrip、`py_compile`、ruff F821/F401/E9をPASSした。
- canonical Kaggle package / pushとfixed32 Stage 0実行の承認を得た。
- Stage 0の1 variant、32 PF well-runs、4,096 seed-well、2,048,000 particle
  starts、sentinel 4 wells、control/model/HMM/Beam/GPU rerun 0をpush前に再確認した。
- canonical private CPU kernel version 1をpushし、id_no `129167965`、metadata、
  terminal status `COMPLETE`を確認した。
- technical gate全PASS、mechanism gate FAILをterminal logから判定した。
- `stage0_fail_closed`としてStage 1 eligibleをfalseにし、再実行、inference、
  submissionを無効化した。
- metrics、result、SESSION_NOTES、SHA、runtime、no-rescue decisionを記録した。

## 次のアクション

exp459はterminal close。Stage 1、inference、submission、parameter / gate救済を
行わず、既存の独立した非acceleration候補を優先する。
