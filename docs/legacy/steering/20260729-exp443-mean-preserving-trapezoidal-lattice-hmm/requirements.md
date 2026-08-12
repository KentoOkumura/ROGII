# 要件

## 依頼

rate追従に関する第3案として、台形積分を格子上で実行可能な別表現にする
`exp443_mean_preserving_trapezoidal_lattice_hmm`を確定する。当初は
design-onlyだったが、2026-07-29のユーザー依頼でcompact self-contained候補と
専用testの実装、2026-07-30のユーザー依頼で正規train Notebook採用、
Kaggle package、Stage 0 fixed32実行まで承認された。Stage 1、inference、
submissionは引き続き別承認とする。

## 仮説

exp209のdestination-rate Euler更新は区間内rate変化を積分しない。source/destination
rateの台形平均を使い、その平均を格子上で厳密保存すれば、signed position biasを
減らせる。

Assumption: persistent誤差の一部がrate kernelの追従だけでなく、position積分と
格子投影の平均biasに由来する。

## 制約

- Route `pf_beam`、親/controlはexp209。
- exp209 rate marginal、grid、state、emission、prior、readoutを固定。
- meanは`0.5*(r_source+r_destination)*dMD-dZ`。
- varianceは`max(parent target variance, lattice minimum variance)`。
- supportは固定5 cells、非負maximum-entropy projection。
- 格子由来variance inflationを隠さず保存・報告する。
- exp439のtarget variance厳密保存契約を再試行しない。
- grid/support/noise/rate/emission/gate救済をしない。
- 実装、正規train Notebook採用、Stage 0 runは承認済み。
  Stage 1、inference、submissionは未承認。

## 受け入れ基準

- mean、minimum variance、effective variance、projectionが一意である。
- exp439 failure edgeが実行可能になる理由をtarget-freeに検証できる。
- fixed32、technical/mechanism gates、truth-late、SHA、実行量が固定されている。
- exp439 failure edgeが固定5-cell projectionで実行可能になるcontract testがある。
- Stage 0 terminal結果、fail-close、Stage 1禁止状態が全文書で一致する。

## 実行結果

2026-07-30にKaggle private CPU version 1を完走した。数値contractは成立したが、
runtime projectionとmechanism 4/6項目がFAILしたため、本要件のfail actionを適用して
branchを閉じた。実装、正規train Notebook、Stage 0は完了。Stage 1、inference、
submissionは実施しない。
