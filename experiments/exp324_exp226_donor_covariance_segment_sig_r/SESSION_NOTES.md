# exp324 セッションノート

## 目的

exp226 donor共分散を、最終予測ではなくsegment別HMM rate diffusionとして利用する。

## 現在の状態

- 2026-07-21: steering、scaffold、設計を確定。
- terminal closed / 未実装 / 未実行。
- Stage 0: 1 diagnostic、HMM 0。Stage 1最大: 1 variant、773 HMM runs。model・booster・control再実行0。

## 固定事項

- weighted MAD、`n_eff/(n_eff+50)` log shrink、clip `[0.001,0.004]`、minimum effective donors 10。
- exp323 prior mean、GR emission、momentum、position kernel、grid、posterior meanは固定。
- schedule freeze後だけtruth residualを読む。

## 再現性

RNGなし。fold/well/segment/donor順、donor covariance、sigma schedule、fallback/clip manifest、predictionのcontent SHAを記録する。

## 2026-07-22 閉鎖

親exp323のterminal closeにより本実験も閉鎖した。reparentや実装再開は行わない。exp338 PASS後の新exp323相当がさらにPASSした場合だけ、新番号で新exp324相当を設計する。

## 次

なし。
