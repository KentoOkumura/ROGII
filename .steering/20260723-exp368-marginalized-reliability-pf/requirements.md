# 要件

## 依頼

PF 状態変数案として marginalized reliability PF の backlog、実験ディレクトリ、
steeringを作り、実装前の設計を確定する。

2026-07-25 の追加依頼 `exp368を実装してください` により、別承認を必要としていた
Stage 0 の実装と placeholder Notebook の置換を承認済みとする。

2026-07-25 の追加依頼 `実行してください` により、固定Stage 0をcanonical Kaggle
private CPU kernelへpushして完了まで監視することを承認済みとする。

## 制約

- Route: `pf_beam`
- `docs/06_reproducibility.md` のPF seed / SHA規約に従う。
- 初回scaffold作成時は実装しない。追加依頼後はStage 0だけ実装する。
- Kaggle package / push / runは固定Stage 0だけ行う。Stage 1 PF、推論、提出は行わない。
- reliabilityはsampleせず粒子ごとに厳密周辺化する。
- exp072 dynamics、particle/seed数を固定する。

## 受け入れ基準

- q遷移、weak emission、marginal update、Stage 0/1 gateが一意である。
- stable RNG、train/test別生成、truth-late-joinを記録する。
- 1 variant / 98,944 seed-well runs / control replay 0を記録する。
- backlog と summary に低優先・設計確定・未実装として登録する。
- Stage 0は1 diagnostic / 5 reporting folds / PF seed-well runs 0 /
  model・booster 0としてcompact self-contained train Notebookに実装する。
- inference Notebookはsubmissionを生成しないfail-closed構成にする。
