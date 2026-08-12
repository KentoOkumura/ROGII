# exp433 タスクリスト

## 未着手（別承認後）

- compact候補のレビュー後、別承認を得て正規train Notebookへ採用する。
- Kaggle private CPU packageのmetadataとbootstrapを検証する。
- push前に1 primary decoder、1 diagnostic、773 wells、5 reporting folds、
  model / booster / HMM / PF / Beam / GPU各0を再確認する。
- 別承認後にKaggle trainを1回実行し、固定gateを判定する。

## 進行中

- なし

## ブロック中

- 正規Notebook編集、Kaggle package / push / runは別承認待ち。
- inference / submissionは全train-side gate PASS後も別設計・別承認が必要。

## 完了

- exp426のtechnical FAILと、疎なabsolute anchor仮説を分離した。
- exp426 version 1 score / manifest SHAを固定入力として記録した。
- primaryを固定Viterbi 1個、blockwise top-1を診断だけに固定した。
- coverageを必須report、実OOF RMSEをprimary outcomeに固定した。
- technical / scientific gate、truth-late順序、fail-close、禁止事項を固定した。
- 実行量を1 decoder、1 diagnostic、773 wells、5 folds、
  0 model / booster / HMM / PF / Beam / GPUに固定した。
- `docs/06_reproducibility.md`を確認し、SHA / rerun / bootstrap方針を反映した。
- steeringとdesign-only実験scaffoldを作成した。
- 2026-07-28: ユーザーの実装依頼を受け、compact self-contained Jupytext
  train候補と対応する未実行Notebook候補を作成した。
- 2026-07-28: frozen score / manifest SHA、score非再生成、support診断、
  fixed Viterbi、固定補間、truth-late ledger、independent full rerun SHA、
  scope / fold / by-well / persistent episode gateを実装した。
- 2026-07-28: contract tests 9件、py_compile、ruff、Jupytext parityをPASSした。
- 正規train / inference Notebookはplaceholderのまま維持した。
