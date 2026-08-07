# タスクリスト

## 未着手

- なし

## 進行中

- なし

## ブロック中

- Stage 0 mechanism gate FAILのため、Stage 1、inference、submissionへ進まない。

## 完了

- exp408のforward hysteresis / rate under-response証拠を確認した。
- exp268、exp338、exp370の失敗と重複しない単一変更に限定した。
- CUSUM、transition変更、duration、refractory、edge処理を固定した。
- Stage 0 / Stage 1の対象、実行量、gate、禁止事項を固定した。
- `docs/06_reproducibility.md`を読み再現性設計へ反映した。
- steering、実験scaffold、backlogをdesign-onlyとして作成した。
- ユーザーの実装指示を記録した。
- fixed32 sample manifestをtarget-free matchingで生成しSHAを固定した。
- compact self-contained train候補を実装した。
- fail-closed inference候補を実装した。
- 専用contract tests 13件を実装しPASSした。
- Jupytext `--test`、py_compile、Ruff F821を通した。
- Make fallbackのstrict experiment validationを通した。
- exp408回帰込み21 testsを通した。
- exp408 compact train 2,483行に対しexp411 train候補2,227行・9章で、
  route固有の上位ロジックと生成物保存をself-containedに保持した。
- ユーザーの2026-07-27の実行指示を、正規Notebook採用とStage 0 Kaggle CPU実行の
  承認として記録した。
- push対象を1 treatment × 32 HMM well-runs、parent rerun / LightGBM / booster /
  PF / Beam / GPUすべて0と再確認した。
- 正規Notebookを採用し、canonical kernel Version 1–4をfail-closed実行した。
- Version 4で32/32 HMM well-runsを約1,009.6秒、peak RSS約1.02GBで完走した。
- 親SHA、親cache row-index、raw id schema、float round-trip SHAの実行時契約を修正し、
  exp408回帰込み26 tests、Jupytext、py_compile、Ruff、strict validationを通した。
- Version 5 strict packageを生成した。
- canonical kernel Version 5を`COMPLETE`まで実行し、32 / 32 wells、
  `1,133.133秒`、peak RSS `1.020561 GB`を確認した。
- Kaggle outputのmetrics / summary / readout / prediction / scheduleを取得し、
  行数、raw / decompressed / logical SHAをログと照合した。
- technical gate 13 / 13 PASS、mechanism gate 2 / 6 PASS、
  `promotion_eligible: false`、`stage0_fail_closed`を確定した。
- Stage 1、inference、submissionを実行せずbranchを閉じた。
