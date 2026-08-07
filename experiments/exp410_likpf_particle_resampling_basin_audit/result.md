# exp410_likpf_particle_resampling_basin_audit 結果

## 仮説

exp072 likelihood-PFのpersistent offsetが、HMMと同じtransition / prior
hysteresisなのか、PF固有のinitial support、GR emission、resampling extinction、
within-seed particle平均、128 seed平均、support / clampのどこで形成されるかを、
保存予測と完全一致する内部粒子replayで特定する。

## 設定

- route / parent: `pf_beam` / `exp072_exp063_full_replay_feature_cache`
- 対象: PF固有に`abs(error) > 10 ft`が128行以上連続する496 wells /
  839 episodes / 819,288 episode rows
- PF: 500 particles ×128 deterministic seeds、momentum `0.998`、
  velocity noise `0.002`、position noise `0.005`、ESS threshold `0.5`、
  systematic resampling、roughening `0.1 / 0.001`
- 評価: target-late原因診断のみ。prediction candidate、CV、LB、model、
  inference、submissionはない

## 結論

PFのオフセットは、HMMと「同じ難しい区間で同じ方向へ外れる」ことは多いが、
支配的な内部原因は同じではない。

PFでは、真値近傍の粒子supportが薄い・失われることと、複数の粒子basin /
seed basinを算術平均することが主因である。排他的分類では
`support_or_clamp_shortage`がSSE `36.4701%`、128 seed間の
`across_seed_aggregation_multiplicity`が`36.2441%`、
seed内のparticle平均multiplicityが`10.8561%`だった。重複を許すと
particle support shortageは390 episodes、SSE `83.6379%`に存在し、
across-seed multiplicityは419 episodes、`42.8304%`に存在した。
hard clamp外の真値は0行なので、`support_or_clamp`の実体はclampではなく
有限粒子supportである。

一方、HMMの主因だったforward transition / prior hysteresisはPFでは排他的
`10.7177%`、重複あり`10.9814%`に留まった。GR updateは全episode行合計で
SSEを`207,765.8`だけ改善し、平均移動量も`0.0317 ft`である。resampling直後の
平均位置変化は`0.000245 ft`、SSE差`-5.77`、majority-seed extinctionは0行で、
通常の各行resamplingが即座にoffsetを作るという説明は支持されない。

ただし、これは「resamplingは将来のparticle genealogyに影響しない」という意味では
ない。観察分類の即時効果と、以後の確率経路まで変える無効化介入は別物なので、
固定12 sentinel wells ×12 variantsのpaired counterfactualもKaggle CPUで完了した。
そこでroughening 10倍がepisode SSEを`0.7530倍`へ改善したことから、通常の
resampling直後の絶滅ではなく、resampling時の多様性付与とその後のgenealogyが、
一部の大誤差区間を増幅・回復させる因果レバーであることは確認できた。ただし
10 / 16 episodes、8 / 12 wellsの改善に留まり、符号検定も有意ではない。これは
target-late sentinel内の機構確認であり、全OOFで有効なprediction候補ではない。

## 496-well観察監査

### 排他的な主原因

| 原因 | episodes | wells | SSE比 |
| --- | ---: | ---: | ---: |
| particle support不足 | 122 | 110 | 36.4701% |
| 128 seed間の算術平均multiplicity | 314 | 212 | 36.2441% |
| seed内particle平均multiplicity | 313 | 222 | 10.8561% |
| transition propagation escape | 39 | 36 | 10.7177% |
| observed GR emission alias | 15 | 15 | 3.6664% |
| mixed / unresolved | 31 | 30 | 1.2880% |
| resampling extinction | 5 | 5 | 0.7577% |

この表は一つのepisodeを一原因へ割り当てるため、因果鎖の重なりを隠す。事前固定した
非排他的flagでは、support不足単独がSSE `36.4701%`、support不足＋across-seedが
`32.2008%`、support不足＋transitionが`4.9058%`だった。support不足と
across-seedを合わせたfamilyは、誤差が負の区間でも正の区間でも、全5 outer foldsでも
支配的である。

### 粒子・seed support

- 真値がmajority seedの粒子support外: 526,682 rows（64.2853%）/
  SSE 83.0651%
- 真値がhard clamp外: 0 rows
- 128 seed中の最良seedが算術平均より5 ft以上よい:
  649,903 rows（79.3253%）/ SSE 87.1065%
- 同10 ft以上: 511,162 rows（62.3910%）/ SSE 77.6469%
- 最良seedが真値5 ft以内: 449,051 rows（54.8099%）
- truth近傍seedが1%以上存在: 389,637 rows（47.5580%）
- 1%閾値でtruth basinとfixed-candidate basinが共存:
  約68.2% rows

算術平均のRMSEは`23.5181 ft`。target-freeな単純seed medianはSSEを
`+10.2190%`悪化させた一方、真値を使うrow-wise oracle best seedはSSEを
`21.9936%`へ減らし、RMSE `11.0294 ft`だった。これはheadroomの証拠であり、
実運用可能な選択器の成績ではない。

### GR emission、transition、resampling

- predictive mean（GR前）はfixed mean比SSE `+0.0459%`
- GR updateは55.6733%の行で真値方向へ動き、合計SSEを`0.0458%`改善
- filtered meanとpost-resample meanはfixed outputと実質同一
- mean ESS: `371.59 / 500`
- resampled seed fraction: `2.3811%`
- unique ancestor fraction: `98.9813%`
- severe ancestor concentration / majority-seed extinction: ともに0行
- 81 threshold combinationsでpersistent resampling主因は0
- transition主因の最大SSE比は`2.8431%`、emissionは`9.6595%`、
  事前固定primary thresholdでは両方0

排他的なtransition / emission labelは、長いepisodeを継続的に支配する割合ではなく、
onset前後の疎なtriggerを表す。最初のevent時刻中央値はtransitionがonsetの27行前、
emissionが34行前、resamplingが6行前だった。

### 消失と再捕捉

- predictive particle mass消失: 319 / 839 episodes
- 消失後に監査区間内で再捕捉: 259 / 319（81.1912%）
- 再捕捉なし60 episodesのSSE比: 21.9886%
- truth-close seed fraction消失: 639 / 839
- 再捕捉率: 64.1628%
- fixed outputが5 ft以内へ戻ったepisode: 416 / 839
- 最後まで戻らない423 episodesのSSE比: 85.4226%

粒子supportは一度失われても多くは再捕捉するが、算術平均出力の回復は半数未満で、
回復しない長いepisodeがSSEを支配する。

## HMMとの比較

PF 839 episodesとexp408 HMM 638 episodesをinterval overlapで比較した。

- offset wells: PF 496、HMM 450、共通363、PFのみ133、HMMのみ87
- PF episodeの53.8737%がHMM episodeと重なり、PF SSEの78.9383%を占める
- 重なる区間の誤差方向一致: 90.2655%
- ただし内部mechanism family一致: episode 8.4071% /
  overlap SSE 10.0206%

HMMの排他的原因はforward transition / prior hysteresis `59.3978%`、
backward smoothing `23.0444%`、sum-product multiplicity `9.0396%`、
state support不足`6.3949%`だった。したがって、両者は同じGR/geometry曖昧区間で
同方向へ外れやすいが、HMMはsticky transitionとfuture beta、PFは有限粒子supportと
粒子・seed平均という異なる内部経路でoffsetを維持している。

## Counterfactual

固定12 wells / 16 episodesに対し、baseline、momentum、process noise、
initial spread、GR sigma、resampling threshold / disable、roughening、clampの
12 variantsをKaggle CPU 4 shardで実行し、144 well-runsをstrict mergeした。
baseline persisted prediction parityは12 / 12 wellsで最大差`0.0 ft`、12 variants /
well、16 episodes、55,104 episode rows、実装SHA、重複なし、4 shard coverageの
全guardをPASSした。baselineのepisode SSEは`113,223,940.62`、RMSEは
`45.329 ft`だった。

| 介入 / 読出し | episode SSE比 | 改善episode | 改善well | 解釈 |
| --- | ---: | ---: | ---: | --- |
| truth-best seed oracle | 0.2812 | 16 / 16 | 12 / 12 | 非運用oracle。seed basinに大きなheadroom |
| roughening 10倍 | 0.7530 | 10 / 16 | 8 / 12 | 最良のtarget-free介入だが不均一 |
| suffix全体likelihood最良seed | 0.8290 | 10 / 16 | 8 / 12 | target-freeだがoffline |
| likelihood重み付きseed平均 | 0.8310 | 10 / 16 | 8 / 12 | target-freeだがoffline |
| process noise 3倍 | 0.8917 | 11 / 16 | 8 / 12 | support回復候補だがgain集中 |
| initial spread 3倍 | 0.9270 | 10 / 16 | 7 / 12 | 弱く不均一 |
| resampling threshold 0.1 | 0.9629 | 9 / 16 | 6 / 12 | median wellではほぼ中立 |
| clamp margin 2倍 | 1.0000 | 0 / 16 | 0 / 12 | 全episode同値。hard clamp原因を否定 |
| particle modeのseed平均 | 1.0006 | 5 / 16 | 4 / 12 | within-seed平均の置換では救済せず |
| seed median | 1.0027 | 6 / 16 | 5 / 12 | 単純集約置換では救済せず |
| momentum 1.0 | 1.0255 | 10 / 16 | 8 / 12 | momentum decay主因を支持せず |
| GR sigma 1.3倍 | 1.1401 | 7 / 16 | 5 / 12 | exp400/404同様、global緩和は悪化 |
| resampling無効 | 3.4809 | 8 / 16 | 6 / 12 | catastrophic outlierがあり安全でない |
| process noise 0 | 6.2659 | 4 / 16 | 3 / 12 | transition noiseは通常必要 |
| GRほぼ無効 | 8.8351 | 2 / 16 | 1 / 12 | 14 / 16 episodesで悪化、GRは通常必要 |

GRほぼ無効はepisode符号検定`p=0.00418`、well符号検定`p=0.00635`で一貫して
悪化した。したがってGR emissionは一部でalias triggerになっても、全体としては
offset原因ではなく主に修正力である。GR sigma 3倍は11 / 16 episodes、9 / 12 wellsで
改善した一方、pooled SSEは`1.0237倍`、全suffixでは`1.3093倍`へ悪化し、少数の
catastrophic outlierを作った。global sigma relaxationを再開する根拠にはならない。

roughening 10倍は全12 leave-one-well-out pooled集計で改善
（SSE比`0.6026–0.8262`）したが、episode / well符号検定は
`p=0.4545 / 0.3877`で、原因別にはacross-seed `0.0835倍`、
transition `0.842倍`、support不足`0.872倍`を改善する一方、within-seed平均
`1.836倍`、明示的resampling extinction `1.920倍`へ悪化した。
process noise 3倍も全leave-one-well-out pooled集計では改善したが、最大gainのwellへ
強く集中した。よって両者は「有限粒子support / basin維持が因果的」という証拠であり、
固定ハイパーパラメータとしての一般化証拠ではない。

resampling無効化は、以後のsystematic drawとroughening由来RNG消費も変える複合介入で、
直接のresampling extinctionだけを測らない。観察監査の即時効果がほぼ0で、明示的
resampling cause episodeでもthreshold 0.1、無効化、roughening 10倍がすべて悪化した
ため、「resampling extinctionがPF offsetの主因」という単純な説明は棄却する。

## 再現性

- fixed prediction logical SHA:
  `b3aa657aeb24be33e710098823bd52ddf95c8484eefdda7160edee9c26198c5f`
- episode / target-well SHA:
  `0bffbc6bd6d89fdbfa11aa86419df8dbd84b1819af717413f0ae3bfa82799804` /
  `ae9d4fd7429ad2b51f8d620a0413e74cb55efccefbb30cce0e2031230548f30e`
- full code / config SHA:
  `750ddbdd20f1e0cd46f1e0a6fcc3ff630524d781dac144a0ac636390a4b6dce1` /
  `15cb5ef655a2b499a87ab1ced9a8c4017c73fb3a4434d2051c6be9473bb8d10c`
- 496 / 496 wellsで保存予測との最大差: `0.0 ft`
- strict coverage、重複なし、4 shard row-ledger SHA guard: PASS
- full shard elapsed:
  `6261.156 / 5890.934 / 5764.288 / 5481.023 sec`
- max peak RSS: `1.984 GB`
- counterfactual sentinel SHA:
  `7e8491d4e1cde59caaed12c638451615b8f113c42811dd7f70f356afe0cf9a04`
- counterfactual baseline-kernel / config / runner / variant-contract SHA:
  `0fb334969a47cd375889590c758292f6f3f2566154174e0bb4bbd97518298050` /
  `4a10a0479c42f10c9d59f35ad73c5fd4f4072091981488ab33ca38964f8b6a65` /
  `ad45d435252e6b62cd384ca3be9bca11305fcb838e1b5f4285430d5e710002e4` /
  `4ef9ab14aa75ba23ce0b8d1d7457b1359b76e22df45c8c667993500269d89be9`
- counterfactual shard elapsed:
  `1880.797 / 2067.560 / 1356.260 / 2118.808 sec`
- counterfactual max peak RSS: `1.983 GB`
- model / prediction candidate / inference / submission: なし

## 次

本実験は原因診断として完了する。PF offsetに対して単純seed median / particle mode、
global GR sigma緩和、resampling無効化、hard clamp拡張は追わない。roughening 10倍と
process noise 3倍は機構上のleadだが、target-late 12 wellsで選ばれた結果なので、
prediction候補として検証する場合は、保存済みexp072 controlを再学習せず、
単一固定variantを全OOFで別承認・別実験として評価する。
