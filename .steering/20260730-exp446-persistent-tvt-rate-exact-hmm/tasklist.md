# タスクリスト

## 未着手

- なし。

## ブロック中

- Stage 1、rerun、inference、blend/selector、submission。
- rate span、momentum、noise、grid、emission、prior、gateのsame-OOF rescue。

## 完了

- 2026-07-30: exp446へ採番し、低-中・P3のscientific HMM候補とした。
- 2026-07-30: 親 `(TVT,U-rate)` とcandidate `(TVT,TVT-rate)` の差分を固定した。
- 2026-07-30: prefix rate、41-state grid、transition、position meanを固定した。
- 2026-07-30: constant-Z sentinel、fixed32、Stage 1、truth-late、SHA、
  runtime、fail-close契約を固定した。
- 2026-07-30: steeringを実験scaffoldより先に作成した。
- 2026-07-30: 親コードをコピーせず、テンプレートからdesign-only実験scaffoldを
  作成し、configと記録を確定した。
- 2026-07-30: `KAGGLE_DIRECTION.md`の未着手backlogへP3として追加し、
  `experiment_summary.md`へexp446を反映した。
- 2026-07-30: project template / strict config / strict experiment validationと
  design consistency assertionをPASSした。
- 2026-07-30: 別名compact self-contained train候補とinference guardを実装した。
- 2026-07-30: TVT-rate prefix初期化、41-state grid、exp209 local rate kernel、
  `q_destination*delta_MD` position kernelを実装した。
- 2026-07-30: constant-Z parent parity、small-state dense reference、
  fixed32 truth-late ledger、rate/grid/transition/posterior/prediction/diagnostic
  SHAとreadback contractを実装した。
- 2026-07-30: 専用12 tests、Jupytext round-trip、py_compile、Ruff F821、
  project/strict config/strict exp validationをPASSした。
- 2026-07-30: ユーザー依頼「実行してください」により、正規Notebook採用、
  Kaggle package、固定Stage 0の1回実行が承認された。
- 2026-07-30: push前実行量をcandidate 1本×32 wells、parent rerun 0、
  model/booster/PF/Beam/GPU 0として`SESSION_NOTES.md`へ再確認した。
- 2026-07-30: compact self-contained train/inferenceを正規Notebookへ採用し、
  private CPU・internet off・run-on-pushのKaggle packageを作成した。
- 2026-07-30: 初回SaveKernel 400後、未使用`src/`を除く同一科学契約・
  canonical slugのpackageへ縮小し、Kaggle version 1をpushした。
- 2026-07-30: version 1（id_no `129106260`）で32/32 wells、
  156,088 suffix rowsを完走した。
- 2026-07-30: technical `17/18`、mechanism `0/7`。runtime projection、
  under-response、forward/persistent SSE、well/fold、matched-control安全性を
  FAILし、`stage0_fail_closed`でbranchを閉じた。
- 2026-07-30: `result.md`、`metrics.json`、`experiment_summary.md`、
  `KAGGLE_DIRECTION.md`を更新し、未着手backlogからexp446を削除した。
