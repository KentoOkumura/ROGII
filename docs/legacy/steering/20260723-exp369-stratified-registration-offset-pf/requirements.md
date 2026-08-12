# 要件

## 依頼

PF 状態変数案として stratified registration offset PF の backlog、実験ディレクトリ、
steeringを作り、実装前の設計を確定する。

## 制約

- Route: `pf_beam`
- `docs/06_reproducibility.md` のPF seed / SHA規約に従う。
- 実装、Notebook置換、Kaggle実行、推論、提出は行わない。
- physical positionとGR lookup offsetを分離する。
- particle総数500、seed数128、exp072 dynamicsを固定する。

## 受け入れ基準

- delta grid、transition、初期count、quota、Stage 0/1 gateが一意である。
- Stage 0はknown prefixだけで完結する。
- 1 variant / 98,944 seed-well runs / control replay 0を記録する。
- backlog と summary に設計確定・未実装として登録する。
