# exp459 要件

## 依頼

Particle Filterの状態を`(TVT, rate)`から
`(TVT, rate, acceleration)`へ拡張する実験について、バックログ、steering、
実験ディレクトリを作成して設計を確定する。

2026-07-30の追加依頼`exp459を実装してください`により、compact
self-contained Stage 0候補、contract test、正規train Notebook採用までを
追加承認する。同日の追加依頼`実行してください`により、canonical Kaggle
train package / pushとfixed32 Stage 0実行を追加承認する。Stage 1実行、
inference、submissionは引き続き未承認とする。

## 根拠

- exp444の3値persistent acceleration exact HMMは数値contractをPASSしたが、
  full runtime投影`144,232.851 sec`で事前上限をFAILし、科学評価に進めなかった。
- PFでは500粒子内にaccelerationを保持でき、exact HMMの
  position×rate×acceleration格子全列挙を避けられる。
- exp367のfixed signed-curvature PF path preflightはreal-minus-circular識別と
  fold再現性をFAILしており、acceleration仮説には強いnegative contextがある。

## 制約

- Routeは`pf_beam`。
- 科学的親はexp417、PF実装参照と保存controlはexp404 x1.0 / scale-5とする。
- acceleration値とtransitionはexp444から固定する。
  - values: `[-0.0005, 0, +0.0005]`
  - interior transition: `[0.08, 0.84, 0.08]`
  - boundary外向きmassはboundary stayへ加える。
- state更新順は`acceleration -> rate -> TVT -> GR weight`。
- `rate=d(TVT+Z)/dMD`を維持し、TVT更新には`-delta_Z`を含める。
- 500 particles、128 seeds、Gaussian GR emission x1.0、temperature 5、
  process noise、resampling、position/rate rougheningを変更しない。
- 初期accelerationは全粒子zero。resampling時はaccelerationを複製し、
  acceleration rougheningは行わない。
- exp444 / exp367を再開またはpositive evidenceへ再分類しない。
- Stage 0はexp411 fixed32のtechnical / mechanism preflightで、CVと呼ばない。
- Stage 1はStage 0全PASSと別承認がある場合だけ773 wellsを実行する。
- 保存exp404 controlを使い、control PFは再実行しない。
- `docs/06_reproducibility.md`に従い、PFのper-well stable seed、raw train/test別生成、
  content SHA、Kaggle bootstrapを設計する。
- implementation、test、正規train Notebook採用、canonical Kaggle package /
  push、fixed32 Stage 0実行は承認済み。
- Stage 1、inference、submissionは未承認。

## 受け入れ基準

- `(TVT, U-rate, U-acceleration)`の定義、初期分布、遷移式、更新順が一意である。
- acceleration値、3×3 transition matrix、境界処理が固定されている。
- base PF乱数streamとacceleration乱数streamを分離し、zero-acceleration sentinelで
  exp404 bitwise parityを要求している。
- Stage 0/1のvariant、PF well-run、seed-well、particle start、control rerun、
  model / booster / HMM / Beam / GPU数が記録されている。
- prediction freeze前のtruth / error / fold / episode / hidden-like role readが0である。
- Stage 0にacceleration非退化、方向一致、persistent改善、matched-control安全性、
  runtime/RSSの全AND gateがある。
- Stage 1にoverall、fold、GR observed/missing、1000+、hidden-like 2面、
  by-well tail、固定HMM/PF blendの全AND gateがある。
- acceleration / transition / particle / seed / temperature / emission /
  noise / resampling / roughening / gateのsame-OOF救済を禁止している。
- compact self-contained Stage 0実装、contract test、正規train Notebookが
  存在する。
- Stage 0完了後は再実行flagをfalseへ戻し、Stage 1 / inference / submissionも
  falseのままである。
- Stage 0のtechnical / mechanism gate、runtime、SHA、terminal decisionを
  `metrics.json`、`SESSION_NOTES.md`、`result.md`へ記録する。

## 実行結果

Kaggle private CPU version 1（id_no `129167965`）でtechnical gateは全PASSしたが、
direction agreement `0.501086`、persistent episode SSE reduction `-11.6190%`、
matched-control pooled delta `+0.435213 ft`によりmechanism gateをFAILした。
Stage 0はCVではなく、route anchorを更新しない。

## 次のアクション

`stage0_fail_closed`としてbranchを閉じる。Stage 1、inference、submission、
acceleration / transition / noise / particle / seed / temperature / emission /
gate / blend / selectorによる救済を行わない。
