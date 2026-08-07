# exp412_beta_filter_rate_disagreement_two_pass_reset 結果

## 状態

Kaggle private CPU Stage 0 Version 3を完了し、`stage0_fail_closed`。
`promotion_eligible: false`のため、Stage 1、inference、submissionへ進まない。

## 実行

- kernel:
  `kentookumura/exp412-beta-filter-rate-gap-two-pass-reset-train`
- kernel id no / version: `128917257 / 3`
- Stage 0: baseline 32 + treatment 32 = 64 HMM well-runs
- retryを含む累計: 65 HMM well-runs
- elapsed: `2,142.435153秒`
- peak RSS: `0.991058 GB`
- model / booster / PF / Beam / GPU: 0
- scientific contract SHA:
  `9f216b213907c42009ce1a7dbcb48f8dc27d7e8861696df3d4fb5432a7da3993`

## 結果

| 項目 | 結果 | Gate |
| --- | ---: | --- |
| finite coverage | `1.0` | PASS |
| saved exp209 baseline parity | `0.0 ft` | PASS |
| active rows / fraction | `5,902 / 0.038752` | PASS |
| active wells | `21 / 32` | PASS |
| beta方向一致 | `0.776347` | PASS |
| 方向一致fold | `4 / 5` | PASS |
| backward active-row coverage | `0.093159` | PASS |
| backward cause SSE reduction | `-0.069575` | **FAIL** |
| forward cause SSE regression | `+0.013257` | PASS |
| control active-row fraction | `0.029605` | PASS |
| control RMSE delta | `+0.005836 ft` | PASS |
| full runtime projection | `51,753.199秒` | **FAIL** |

technical gateは12 / 13、mechanism gateは5 / 6をPASSした。fold別方向一致は
`0.997364 / 0.834123 / 0.726087 / 0.366008 / 0.874887`で、fold 3だけFAIL。

## 解釈

beta-filter disagreementは、exp411のcausal innovation scheduleより方向性と選択性が
大幅に良く、方向一致、4 / 5 folds、forward / control安全性を満たした。ただし主目的の
backward causeではSSEが`6.96%`悪化した。改善は一部wellへ集中し、
`fae0c593 -1.856824 ft`、`57f05c51 -3.080338 ft`の一方、
`a9c9b150 +3.160318 ft`、`c9e980e8 +2.260255 ft`と大きく悪化した。

したがって、future betaのrate方向は多くのactive rowで正しいが、固定10%の
transition de-stickをそのまま適用しても、wrong position basinを安定して修復できない。
同じOOF上でthreshold、window、transfer量、well/row gateを救済探索しない。

runtime projectionも固定上限8.5時間を超えるため、mechanismを満たしたとしても現行の
773-well two-pass fullは実行対象外だった。

## Artifact監査

- activation schedule: 152,303 rows、
  raw `b5064975...8fcdaa`、decompressed `18cdb7da...7d85b`
- predictions: 152,303 rows、
  raw `2852a250...728fc`、decompressed `29f3148c...fccd6`
- well metrics: 32 rows、`28fbddbf...a1d82`
- direction truth-late readout: 5,902 rows、`5e4bab61...d3a958`
- cause episode readout: 20 rows、`b2bbcf21...d27519`
- summary: 6,997 bytes、`621decde...a167379`
- input manifest: 1,414 bytes、`bfcd1852...0c8e1b5`

Kaggle outputの`metrics.json`、summary、7 artifactについて、gate/result一致、
raw/decompressed SHA、CSV row数を実ファイルで照合した。

## 結論

現行仮説はnegative resultとして信頼できる。exp412を閉じ、Stage 1、inference、
submissionは実行しない。次の候補は、beta方向triggerのsame-OOF救済ではなく、
独立した単一因子仮説として管理する。
