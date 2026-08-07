# 要件

## 依頼

PF 状態変数案として stratified signed-curvature PF の backlog、実験ディレクトリ、
steeringを作り、実装前の設計を確定する。

## 制約

- Route: `pf_beam`
- `docs/06_reproducibility.md` のPF seed / SHA規約に従う。
- 実装、Notebook置換、Kaggle実行、推論、提出は行わない。
- particle総数500、seed数128、exp072 likelihood/dynamicsを固定する。
- current rateをprefix / geometryから予測しない。

## 受け入れ基準

- curvature state、transition、quota、Stage 0/1 gateが一意である。
- stable per-well/per-seed RNGとtrain/test別生成を明記する。
- 1 variant / 98,944 seed-well runs / control replay 0を記録する。
- backlog と summary に設計確定・未実装として登録する。

## 2026-07-25 実装承認

- ユーザーの `exp367を実装してください` を Stage 0 実装の承認として記録する。
- 実装対象は 0-PF の Stage 0 に限定する。
- Stage 1 PF 実装は Stage 0 の全 gate PASS と別承認が必要。
- Kaggle package push、Stage 0 実行、推論、提出は引き続き未承認。
- Stage 0 は 512 行の完全 block、stride 256、同一 well 内で1 block循環した GR
  negative control、score同点時 `0/-1/+1`、scope正方向は selected path の
  block RMSE gain vs zero path として一意化する。

## 2026-07-25 Stage 0実行承認

- ユーザーの `実行してください。` をcanonical Kaggle private CPU Stage 0の
  prepare / push / run / completion monitoring承認として記録する。
- 実行量は固定3 path、5 reporting folds、PF / control replay / LightGBM / booster各0。
- Stage 1 PF、inference、submissionは承認対象外。
