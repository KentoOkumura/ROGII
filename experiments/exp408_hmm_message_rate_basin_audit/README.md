# exp408 HMM message / rate basin audit

## 状態

Kaggle private CPU version 3で450-well message auditを完了しました。
全technical gateをPASSし、結果と再現可能なchunk readoutを保存済みです。

## 仮説

長いoffsetは一律の「GRが別modeに一致した」現象ではなく、predictive prior、
GR emission、backward beta、sum-product readout、hidden rate supportの異なる段階で
形成される複数経路を含むと仮定します。

## 親実験と変更点

親は`exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`です。
exp209のgrid、rate state、transition、GR emission、posterior mean readoutは一切変えず、
exp270で事前固定されたpersistent 450 wellsだけを再decodeしました。予測を変更する実験では
なく、既存artifactに無かったpredictive / filtered / backward messageとrate massを
stream保存する診断だけを追加しています。

## 検証方針

exp209 exact HMMのpersistent offset 450 wellsをcurrent設定のまま再decodeし、
predictive・filtered・smoothed message、hidden rate mass、position transport momentを
直接観測するtrain-side原因診断です。

- Route: `pf_beam`
- HMM: 1 current variant / 450 well-runs
- model / booster / PF / Beam / GPU: 0
- 実行先: Kaggle private CPU
- prediction候補、inference、submission: なし

truthとepisode境界はwellごとのprediction/message SHA freeze後だけ診断maskへ使います。
HMM decoderはtruth引数を持ちません。詳細は
`.steering/20260726-exp408-hmm-message-rate-basin-audit/`を参照してください。

## 所見

排他的分類ではforward transition / prior hysteresisがepisode SSEの
`59.3978%`、backward smoothing reversalが`23.0444%`、
sum-product multiplicityが`9.0396%`を占めました。raw/imputed GRが事前閾値で
支配的だったepisodeは0件です。

current rowのGR emissionはpredictive priorをほとんど変えず、persistent rowの
`70.3493%`ではemission前からtruth-vs-mean logitが`-ln(3)`未満でした。
一方、betaは`67.5874%`の行でtruth oddsを強く下げ、rate massを回復しながら
truth position massを減らすtranslation-lock行が`43.3341%`ありました。

詳細は`result.md`、Kaggle出力の小型manifestは`artifacts/kaggle_v3/`、
row-level再集計は`artifacts/readout_v3/`を参照してください。

## 次の扱い

mode ID保持、GR weight変更、position sigma / exact-mean、Viterbi / MAP置換は優先しません。
次に介入する場合は、rate-changeへの追従遅れを抑えつつstable区間を壊さない単一の
transition / reset仮説をknown-prefixの0-HMM preflightから別実験として設計します。
exp408の再実行、inference、submissionは行いません。
