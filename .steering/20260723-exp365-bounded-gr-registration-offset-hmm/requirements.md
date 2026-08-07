# 要件

## 依頼

HMM 状態変数案として bounded GR registration offset HMM の backlog、実験ディレクトリ、
steeringを作り、実装前の設計を確定する。

2026-07-25のユーザー依頼`exp365を実装してください`により、設計済みStage 0だけを
実装対象へ変更した。Stage 1、Kaggle実行、推論、提出には承認を拡張しない。

## 制約

- Route: `pf_beam`
- `docs/06_reproducibility.md` に従う。
- Stage 0をcompact self-contained Notebookとして実装する。
- Kaggle package / push / run、Stage 1、推論、提出は行わない。
- physical positionとemission lookup offsetを混同しない。
- rate予測、DTW、GR affine、sigma変更を同時に加えない。

## 受け入れ基準

- delta grid、prior、transition、output/emissionの役割が一意である。
- Stage 0はknown prefixだけで完結する。
- Stage 1 gate、実行量、fail policyが固定されている。
- Stage 0実装とfail-closed inferenceが専用test・静的検証を通る。
- backlog と summary にStage 0実装済み・未実行として登録する。

## 実装結果

- compact self-contained trainとfail-closed inferenceを実装した。
- 正規Notebookへ採用し、専用9 tests、py_compile、Ruff、Jupytext、
  strict experiment validationを通した。
- Kaggle package / push / run、Stage 1、推論、提出は実行していない。

## 生成物

実行時にrolling ledger、delta posterior、safe input manifest、resource projection、
freeze manifest、fold metrics、gate report、summaryを保存する。現時点では実行生成物はない。
