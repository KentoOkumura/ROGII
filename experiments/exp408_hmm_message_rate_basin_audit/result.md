# exp408_hmm_message_rate_basin_audit 結果

## 目的と仮説

exp209 persistent offsetがpredictive prior、current GR emission、backward beta、
sum-product multiplicity、state supportのどの段階で形成されるかを、親HMMを変更せず
内部messageから排他的に分離する。current-row GRによる即時wrong-mode移行が全体主因なら
raw / imputed GR aliasがdominantになるはずであり、そうでなければforward historyや
backward / multiplicityへ帰属する、という反証可能な診断である。

## 結論

exp209 exact HMMのpersistent offsetを支配するのは、current rowのGR emissionが
その場で別modeを選ぶ現象ではない。主経路は、過去から持ち越したposition-rate
posteriorとsticky/coarse rate dynamicsにより、current emissionを入れる前の
predictive priorが既にwrong datum basinへ偏っている
`forward transition / prior hysteresis`である。

事前固定した排他的分類では、638 episodesのうち452件、episode SSEの
`59.3978%`がこの経路だった。forward conditionを他原因との重複を許して数えると
469件、SSE `65.7812%`で、5 foldsすべてに再現した。

第二経路はfuture betaによる`backward smoothing reversal`で、86件、SSE
`23.0444%`。第三の広い増幅器はsum-product path multiplicityであり、排他的には
37件、SSE `9.0396%`だが、重複ありでは276件、SSE `72.0915%`に存在した。
truth position/rate support不足は18件、SSE `6.3949%`に限定された。

raw GR / imputationが`ln(3)`以上のlog-odds効果でepisode行の50%以上を
wrong basinへ押す事前定義のepisodeは、いずれも`0 / 638`だった。
したがって、過去の「GR matchingで違うmodeに入った」という説明は、
相関した弱いGR evidenceがhistoryやfuture betaに蓄積する一部群の説明としては残るが、
全体のroot causeとしては否定された。

より正確には、

> rate posteriorの0方向under-responseが誤差方向のposition displacementを積み上げ、
> absolute datumを途中で再anchorしないtranslation-gauge構造によりoffsetが残る。
> current rowのGR更新はほぼpredictive priorを変えず、future transition / GRを含む
> betaとsum-product multiplicityがwrong datum massをさらに強める。GR aliasは一部で
> historyのseedまたはlock増幅器になるが、一般的な即時mode-switch原因ではない。

という機構が、今回の内部message直接観測と既存監査の双方に整合する。

## Kaggle実行とtechnical gate

- kernel: `kentookumura/exp408-hmm-message-rate-basin-audit-train`
- version / id_no / status: `3 / 128636642 / COMPLETE`
- runtime / peak RSS: `15,930.997 sec`（4.4253 h）/
  `3.587807 GB`
- scope: `450 wells / 2,264,135 suffix rows /
  638 episodes / 807,710 episode rows`
- current HMM: `1 variant / 450 well-runs`
- model / LightGBM config / trained fold / booster / PF / Beam / GPU:
  `0 / 0 / 0 / 0 / 0 / 0 / 0`
- posterior mean parity max abs diff: `0.0 ft`
- message normalization max abs error: `5.33755e-8`
- truth / episode rows read before each well freeze: `0 / 0`
- technical gate: `11 / 11 PASS`
- HMM runtime合計 / median per well:
  `11,721.966 sec / 25.164 sec`
- well全処理runtime合計 / median:
  `15,836.306 sec / 34.263 sec`

version 1はraw horizontalへ存在しない`id`を要求してHMM前に停止した。
version 2はbackward position marginalをNumPy `sum`へ変えた加算順差により、
先頭wellで保存exp270とのparityが最大`0.0546875 ft`ずれてfail-closeした。
version 3はexp270と同じ`position -> rate`のfloat64累積順へ戻し、
toleranceを緩めず全450 wellsで完全parityを達成した。

## 排他的な原因分類

| 原因 | episodes | wells | rows | episode比 | SSE比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| forward transition / prior hysteresis | 452 | 350 | 557,692 | 70.8464% | **59.3978%** |
| backward smoothing reversal | 86 | 72 | 129,308 | 13.4796% | **23.0444%** |
| sum-product path multiplicity | 37 | 35 | 60,641 | 5.7994% | **9.0396%** |
| state support shortage | 18 | 18 | 23,113 | 2.8213% | **6.3949%** |
| mixed / unresolved | 45 | 37 | 36,956 | 7.0533% | 2.1233% |
| raw GR alias | 0 | 0 | 0 | 0% | 0% |
| imputation alias | 0 | 0 | 0 | 0% | 0% |

forward causeのfold別SSE比は`49.44--66.86%`、backward causeは
`18.33--26.12%`であり、特定foldだけの現象ではない。

## Message段階の直接観測

### Forward prior

- predictive truth-vs-mean logitが`-ln(3)`未満:
  `70.3493% rows / 69.1486% SSE`
- filteredでも同条件:
  `70.3556% / 69.1492%`
- predictive truth-basin mass median:
  `0.012404`
- filtered truth-basin mass median:
  `0.012351`

current emission前後がほぼ同じなので、offset区間ではcurrent rowに到着した時点で
truth basin massの大半が既に失われている。

### Current-row GR emission

- posterior basin logitを`-ln(3)`以上悪化:
  `0.2533% rows / 0.9238% SSE`
- current emissionがtruth優勢からwrong優勢へ新規反転:
  `9 / 807,710 rows`
- raw観測行のdirect GR比較:
  candidate側`47.6818%`、truth側`52.3182%`
- direct `|log likelihood差| > ln(3)`:
  candidate側`0.5486%`、truth側`0.9248%` of observed rows

effectを`0.1`まで弱めても、current emissionがepisode行の25%以上を悪化させるのは
28件、SSE `3.3491%`、50%以上では6件、SSE `0.0744%`だった。
effect `ln(1.5)`では25%以上が2件、SSE `0.0156%`、50%以上は0件。
したがってGR非支配という結論は`ln(3)`・50%だけの閾値依存ではない。

一方、episode全体で弱いdirect GR evidenceを累積した既存分類では、
observed candidate-strongが180件、SSE `33.4158%`存在する。
これはGR alias群が無いという意味ではない。相関した弱いevidenceが長時間累積し、
既に形成されたhistory priorやfuture betaをlockする一部経路は残る。
ただしそのcandidate-strong群でも、排他的SSEの`74.3055%`はforward cause、
`22.5465%`はbackward causeだった。

### Backward betaとtranslation lock

- betaがtruth oddsを`-ln(3)`以上悪化:
  `67.5874% rows / 66.9676% SSE`
- filtered truth優勢をsmoothed wrongへ反転:
  `18.4187% rows / 21.4334% SSE`
- smoothed truth-vs-mean logitが`-ln(3)`未満:
  `89.0818% rows / 89.8347% SSE`
- beta truth-vs-mean logit delta:
  mean `-5.8963`、SSE加重mean `-7.7271`

backward messageはtruth position massを平均`-0.10190`減らす一方、
truth近傍rate massを平均`+0.04167`増やした。実際、rate massが回復しながら
truth position massが減る行は`43.3341%`、SSE `38.3313%`に達する。
これはlocal rateを再同期してもabsolute datumだけが戻らない
translation-gauge lockの直接証拠である。

ただしbetaはfuture transitionとfuture emissionの合成であり、
backward再帰という演算単体を原因とは呼ばない。既存の正しいemissionを持つ合成制御では
backwardはlagを修復している。実データのstructured future evidenceとhistory priorの
相互作用が増幅器である。

### Hidden rateとtransition displacement

- filtered rateがtruthと同方向だが絶対値の小さい0方向under-response:
  `70.9074% rows / 70.3580% SSE`
- 同条件がepisode行の50%以上:
  `511 / 638 episodes / 85.1158% SSE`
- current expected displacement errorがoffsetと同方向:
  `63.3981% rows / 67.0193% SSE`
- episode平均同士の符号一致:
  `74.1379% episodes / 90.2246% SSE`
- episode平均のSpearman:
  `0.56928`

一方、current position-kernel quantization bias自体がoffsetと同方向なのは
`32.9686% rows / 28.2397% SSE`、episode平均の符号一致は
`29.7806% / SSE 20.8492%`、Spearmanは`-0.38923`だった。
exact-mean transportへ置換した局所誤差の方がoffset方向に揃うため、
position quantizationは正しいrate stateならdriftを生成できるものの、
実データの誤ったrate posteriorに対しては平均的に暴走を弱めるregularizerでもある。

## Sum-productとmode ID

- row-wise multiplicity条件:
  `73.6082% rows / 79.4500% SSE`
- episode行の50%以上で同条件:
  `426 episodes / 80.9748% SSE`
- 独立episode条件:
  `276 episodes / 72.0915% SSE`
- exclusive multiplicity:
  `37 episodes / 9.0396% SSE`

多経路質量は広い増幅器だが、forward 197件、backward 41件と重複する。
exp270ではpersistent episodesの`67.40%`、SSE `73.29%`が
global Viterbi rate-mode zero-switch well上にあった。exp391はprefix mode IDを
保持するno-switch conditional meanを試したが、16-well段階でparity、normalization、
runtime gateをFAILし、active candidate 0で閉じた。

したがって「mode IDを保持する」だけでは、同一mode内の連続datum drift、
translation lock、sum-product massを解決できない。

## Artifactと再現性

- prediction manifest:
  `1fb7b95e998de57cdf3308798d03ff83787e35adde7a54a66ac0027542da323d`
- message manifest:
  `c866df1a1e9fb9c6ef16d6fc9713a304bfe5679cb72eeffab362c2482121e6e6`
- row ledger raw SHA:
  `97c86f4907ec2a65200a8f83dce239cf180c33ee33c15974b0d5bf2a0a7bdde7`
- row ledger decompressed SHA:
  `74bb3c6b5593c3e01065b9feb81d4f76ee5133eef67a8e8972df22eb61ad2ffb`
- episode summary SHA:
  `b230ffc759e6ee4891f22809b3f3c8a8796681fb461ec0b7215b94a352bf0ab0`
- cause summary SHA:
  `53d0033f3b940585d1ffbfaac7fb1d0d56219e8bb92288a639ad7c353160d1f7`
- well manifest SHA:
  `5cd80f0e6732eafbc7edd4de45db702c5d673217e18e5fdce0db15a7079bdc3a`

小さいKaggle生成物は`artifacts/kaggle_v3/`、chunk readoutは
`artifacts/readout_v3/`へ保存した。860,095,821 bytesのrow ledger本体は
Git対象にせず`/tmp/exp408_row_ledger_v3.csv.gz`へ置き、raw/decompressed SHAを
双方照合した。

## 次

offset対策を再開する場合、優先対象はhardなmode ID固定や全well一律GR downweightではなく、
次の順になる。

1. local rateとabsolute datumを分離し、rate再同期後にdatum massを戻せる
   geometry-awareな再anchor / residual-datum state。
2. forward/filterとfuture betaが強く不一致な区間だけをtruth-freeに検出し、
   betaまたはhistory massを再配分する仕組み。
3. multiplicityが大きくViterbi headroomもある区間だけを対象にした
   uncertainty-aware readout。global Viterbi全行置換は既存結果で悪化しているため行わない。

同じOOFを見たGR閾値、beta温度、mode境界のparameter rescueは行わず、
新しい介入は別の事前固定実験として評価する。
