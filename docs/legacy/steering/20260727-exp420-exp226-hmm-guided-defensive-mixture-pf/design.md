# 設計

## 仮説

exp226の空間geometry rateとHMMのtarget-well rate innovationを、元transitionを残す
importance-corrected PF proposalへだけ使えば、absolute datumを引き継がずに
finite-particle support不足と持続的なrate追従遅れを同時に減らせる。

## 結論

`exp420_exp226_hmm_guided_defensive_mixture_pf` を、3者の prediction blendではなく
単一の importance-corrected likelihood-PF として設計する。route は `pf_beam`。

- exp226の強みである空間地質 geometryは、GR補正前の局所surface rateとして
  particle proposalへ入れる。
- HMMの強みである全stateのforward evidenceは、predictive-to-filtered rate
  innovationの方向検知だけに使う。
- PFはcontinuous state、GR likelihood、resamplingを担い、元transition 50%を残す
  defensive mixtureで有限粒子supportを維持する。
- 128 seedの単純平均は使わず、exp404で固定したtemperature-5 full-suffix
  log-evidence weightingを使う。

HMM posterior mean、backward smoother、exp226 absolute pathは一切平均しない。
したがって最終予測を生成する本体は1本のPFだけであり、MLとのensembleではない。

## 根拠

`physical_model_summary.md` の監査では、persistent offsetの主因は次のように異なる。

1. HMM: forward transition / prior hysteresisがepisode SSEの`59.3978%`。
   rate under-responseを積分した後、absolute datumを再anchorできずtranslation-gauge
   lockへ入る。
2. PF: finite particle support不足が`36.4701%`、across-seed算術平均が
   `36.2441%`。hard clampやresampling直後の全消失ではない。
3. exp226: donorとtargetの`0.02--0.04 ft/row`程度のrate mismatchを、一度だけの
   boundary anchorから長いsuffixへ積分する。

よってabsolute path同士の平均では共通のdatum driftを残し得る。3者から絶対位置を
引き継がず、exp226 rateで空間的な探索先を増やし、HMM innovationでtarget well自身の
変化方向を追加し、PFのimportance correctionでtarget posteriorを固定する。

## 状態と固定PF

exp404 x1.0 scale5 PFと同じsurface stateを使う。

```text
s_t = TVT_t + Z_t
r_t = MDあたりsurface rate
delta_md_t = max(MD_t - MD_(t-1), 1)
mu0_t = 0.998 * r_(t-1)
p0(r_t | r_(t-1)) = Normal(mu0_t, 0.002^2)
s_t | r_t = Normal(s_(t-1) + r_t * delta_md_t, 0.005^2)
```

PFは500 particles ×128 stable seeds、ESS threshold 0.5、systematic resampling、
rough position / rate `0.10 / 0.001`、raw Gaussian GR x1.0を固定する。

## exp226 geometry rate

同じouter foldで保存されたGR補正前geometryだけを使う。

```text
g_t = exp226.tvt_geop_t + Z_t
r_geo_t = (g_t - g_(t-1)) / delta_md_t
```

最初のsuffix rowの`g_(t-1)`は最後の既知`TVT_input + Z`とする。
exp226 final、`gr_delta`、U projection、truth、errorはproposalへ渡さない。

## untreated HMM innovation schedule

exp209互換のforward filterを変更せずに1回実行する。position grid、41 rate states、
rate step `0.005`、`sig_r=0.002`、`sig_p=0.02`、momentum `0.998`、Gaussian GR
emission、priorは固定する。backward passとposterior-mean predictionは使わない。

各suffix rowでemission前後のrate meanから、

```text
u_t = (mu_filtered_t - mu_predictive_t) / 0.005
```

を計算する。exp411の固定two-sided CUSUMをそのままschedule生成に使う。

- drift allowance: `0.01 rate cell`
- threshold: `1.0 rate cell`
- trigger後のactivation: 次の`32 transitions`
- refractory: `128 rows`
- trigger時にCUSUM reset
- active中の再trigger / direction flip: なし
- scheduleはPF開始前にfreezeしSHAを確定

scheduleから使うのは`active`と`direction in {-1, 0, +1}`だけである。HMMのabsolute
position、smoothed rate、posterior mean、truth-best stateは使用しない。

## Defensive mixture proposal

inactive rowではexp419 proposalを維持する。

```text
q_inactive(r_t) =
    0.5  * Normal(mu0_t, sigma_r^2)
  + 1/6 * Normal(r_geo_t, (1  * sigma_r)^2)
  + 1/6 * Normal(r_geo_t, (4  * sigma_r)^2)
  + 1/6 * Normal(r_geo_t, (16 * sigma_r)^2)
```

active rowではgeometry側の半分をHMM方向proposalへ移す。

```text
r_hmm_t = mu0_t + direction_t * 0.005

q_active(r_t) =
    0.5  * Normal(mu0_t, sigma_r^2)
  + 1/12 * Normal(r_geo_t, (1  * sigma_r)^2)
  + 1/12 * Normal(r_geo_t, (4  * sigma_r)^2)
  + 1/12 * Normal(r_geo_t, (16 * sigma_r)^2)
  + 1/12 * Normal(r_hmm_t, (1  * sigma_r)^2)
  + 1/12 * Normal(r_hmm_t, (4  * sigma_r)^2)
  + 1/12 * Normal(r_hmm_t, (16 * sigma_r)^2)
```

`sigma_r=0.002`。元transitionを常に0.5含むため`q >= 0.5*p0`であり、
importance ratioは構成上`p0/q <= 2`となる。position conditionalはtargetと同じなので、
GR update前のweightへrate密度比だけを掛ける。

```text
weight_t *= p0(r_t | r_(t-1)) / q(r_t) * raw_gr_gaussian_likelihood_t
```

importance clipは行わない。incremental normalizer、log evidence、ESSは補正後weightから
計算する。HMM innovationはraw GRを利用するがproposal correctionを通すため、
無限粒子極限のPF target densityへHMM evidenceを追加するものではない。

## Seed aggregation

exp404 x1.0のtemperature-5 full-suffix evidence weightingを固定する。

```text
seed_weight_k =
  softmax((seed_total_log_marginal_likelihood_k - max_log_likelihood) / 5)
prediction_t = sum_k seed_weight_k * seed_prediction_(k,t)
```

これは128 seedの算術平均multiplicityを緩和する。full-suffix GRを使うためbatch /
non-causal predictionと明記し、online filterとは呼ばない。

## アプローチ

### Stage 0: fixed44 mechanism preflight

同じexp420内で、次の固定assetのwell列だけをunionする。

- exp411 fixed32: persistent 16 + matched control 16
- exp410 fixed12 PF counterfactual sentinel
- expected unique wells: 44
- expected overlap: 0

selection role / cause / episode / truthはscheduleとcandidate predictionのfreeze後だけ読む。
Stage 0 pooled RMSEはselection-biasedなのでpromotion gateに使わない。

1. 44 wellsのuntreated HMM scheduleをfreezeする。
2. 同じ44 wellsをcandidate PF 1 variantで実行する。
3. all-guidance-zeroでexp404、HMM-weight-zeroでexp419 proposalとのparityを確認する。
4. fixed32でdirection / lead / control activation、fixed12でsupport / episode SSEを
   mechanism gateとして判定する。
5. 全gate PASS時だけ、別承認後にfull OOFへ進む。FAIL時は同じOOFで設定を救済せず閉じる。

Stage 0予定量はHMM signal 44 well-runs、PF 44 well-runs、5,632 seed-well
trajectories、2,816,000 particle starts、model / booster / GPU 0。

### Stage 1: full OOF

別承認後だけ全773 wellsをKaggle CPU 4 shardで実行する。

1. untreated HMM schedule、candidate prediction、proposal / importance / ESS /
   per-seed evidence / predictive support diagnosticsをtruth前にfreezeする。
2. 保存済みexp404 scale5、exp226 final、exp263 fixed physical blendをcontrol /
   referenceとしてlate joinする。controlは再実行しない。
3. freeze後にtruth、fold、GR scope、distance scope、hidden-like role、
   exp408 / exp410 fixed episodeを結合する。
4. technical、mechanism、standalone、physical-anchor gateの順に判定する。

full予定量はHMM signal 773 well-runs、candidate PF 773 well-runs、
98,944 seed-well trajectories、49,472,000 particle starts、reporting folds 5、
LightGBM config / trained fold / booster / model / GPU 0。

## 実験範囲

- 対象実験: `exp420_exp226_hmm_guided_defensive_mixture_pf`
- Route: `pf_beam`
- 実装lineage parent: `exp419_exp226_guided_defensive_mixture_pf`
- scientific PF parent: `exp404_scale5_sigma_gr_likelihood_pf_ablation`
- geometry parent:
  `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- HMM kernel parent:
  `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- HMM schedule reference:
  `exp411_predictive_filtered_rate_innovation_destick`
- mechanism evidence:
  `exp408_hmm_message_rate_basin_audit`、
  `exp410_likpf_particle_resampling_basin_audit`
- 変更する変数:
  exp419のrate proposalに、固定CUSUM active rowだけHMM方向3成分を加える。
- 固定する変数:
  PF target transition、position conditional、GR emission、initialization、
  resampling、roughening、particles、seeds、temperature-5 aggregation。

最終predictionは`exp226_hmm_guided_defensive_mixture_scale5` 1 variantだけである。
HMM / exp226 / PF prediction blend、selector、ML modelは実験範囲外。

## 事前固定gate

### Stage 0 technical

- fixed32 / fixed12 asset SHA一致、unique 44、overlap 0、fold 0--4
- schedule / candidate / diagnostic freeze前のtruth / error / role / cause read 0
- untreated HMM no-trigger parity、schedule rerun logical parity
- all-guidance-zero exp404 parity、HMM-weight-zero exp419 proposal parity
- inactive / active mixture weight sum 1、positive finite density
- `p0/q <= 2 + 1e-12`、importance clip 0、finite candidate coverage 1.0
- active-row fraction `0.001--0.25`、persistent active wells `>=8/16`
- planned execution count、runtime、peak RSS内

### Stage 0 mechanism

全条件をANDで要求する。

- HMM future-rate direction agreement `>=0.60`
- direction agreement `>0.50`のfold `>=4/5`
- onset 32-row前のtrigger coverage `>=0.50`、eligible episodes `>=8`
- matched control active-row fraction `<=0.10`
- persistent minus control active-well fraction `>=0.20`
- fixed12 exp410 episode SSEを保存exp404 scale5比`>=10%`削減
- fixed12 majority-seed predictive support外率を`>=5 percentage points`削減
- fixed12 worst-well RMSE regression `<=2.0 ft`

### Full mechanism

- 保存exp404 scale5 `10.914522073423171`比pooled gain `>=0.10 ft`
- scale5比改善fold `>=4/5`
- raw-GR observed gain `>=0.10 ft`
- raw-GR missing / high-missing / 1000+ / hidden-like 2面のregression各`<=0.02 ft`
- exp410 scopeのmajority-seed support外率を`>=5 percentage points`削減
- exp410 fixed episode SSEを`>=10%`削減
- exp408 HMM fixed episode scopeのSSEを`>=5%`削減
- by-well delta RMSE p95 `<=0.25 ft`
- worst-well regression `<=2.0 ft`

### Standalone / physical anchor

- standalone:
  exp226 final `9.427109596582213`比gain `>=0.03 ft`、改善fold`>=3/5`
- physical anchor:
  exp263 fixed physical blend
  `0.50 * exp226_final + 0.25 * (last_known_tvt + exp072_likpf_mean_d)
  + 0.25 * exp209_hmm_mean_tvt`（保存済みRMSE `8.238331`）比gain
  `>=0.03 ft`、改善fold`>=3/5`

mechanismだけPASSなら機構支持のみ。standaloneまでPASSならPF単体候補、
physical-anchorまでPASSならPF route anchor候補とする。inference資格はいずれも
自動では与えず、同じexp420内の別設計・別承認を必要とする。

## 再現性設計

- seed policy:
  `sha256("likpf::train::<well_id>") % 2147483647 + 1 + seed_index`
- stochastic処理:
  mixture component draw、particle initialization / propagation、
  systematic resampling、roughening。
- PF:
  likelihood-PF 500 particles ×128 seeds。HMM scheduleはRNGなし。
- 並列:
  well内single worker。well / split / seed index由来のlocal RNGを使い、shard番号、
  well実行順、thread schedulingをseedへ含めない。
- parity RNG:
  guidance weight 0 modeでは追加component drawを消費せず、親PFのRNG順を保つ。
- runtime:
  Kaggle private CPU、GPU off、internet off、4 deterministic well shards。
- SHA:
  raw / Type Well、exp226 geop、exp404 control、fixed manifests、code、config、
  scientific contract、HMM schedule、candidate prediction、proposal diagnostics、
  well manifestを記録する。gzipはdecompressed content SHAを主証拠にする。
- model / submission:
  生成しないためSHAは非該当。inference設計後に必要となった時点で追記する。
- rerun:
  fixed probe wellでschedule、prediction、diagnostic logical parityを確認する。
- Kaggle bootstrap:
  package後にembedded ZIP内のconfig、source、run stage、kernel source、asset SHAを
  正本と照合する。
- deterministic anchor:
  full coverage、全SHA、fixed-probe rerun、raw-test regenerationが揃うまでは
  deterministic anchorと呼ばない。

## リスク

- リークリスク:
  exp226はfold-safe `tvt_geop`だけをallowlistで読む。fixed manifestのrole /
  cause、truth、error、episode、fold、hidden-likeはcandidate freeze後だけ読む。
- proposal variance:
  geometry / HMM方向が外れると有限粒子を浪費し得る。元transition 50%と
  `p0/q <=2`はcatastrophic importance weightを抑えるが、改善は保証しない。
- HMM誤検知:
  raw GR aliasやsticky prior由来のinnovationを拾い得るため、32 transitionsと
  refractory 128を固定し、active割合 / control発火 / future方向をStage 0で検証する。
- CV/LB不一致:
  Stage 0は原因enriched fixed44でありpooled RMSEへ一般化できない。full 773-well
  GroupKFold surfaceとhidden-like 2面を通るまで候補化しない。
- ランタイム/メモリ:
  PFに加えuntreated HMM forwardが必要。fullは4 shard、各保守7.5時間、
  hard stop9時間、peak RSS 25GBを上限とする。
- 再現性:
  component drawとresamplingがある。global RNGを禁止し、schedule / prediction /
  diagnosticsのlogical SHAとprobe rerunを要求する。
- interpretation:
  proposal correctionにより無限粒子targetは元PFと同じ。改善はfinite-support配置と
  seed aggregationの改善として解釈し、新しいabsolute datum観測とは解釈しない。

## 禁止する救済

- HMM posterior mean / backward message / fixed offset / MAP / Viterbiの混合
- exp226 final / GR correction / U projection / absolute pathの混合
- prediction blend、selector、ML meta feature / model
- CUSUM threshold / drift / duration / refractory grid
- proposal weight / width、importance clip、GR sigma、process noise、roughening、
  particle / seed数のgrid
- donor-distance / well / row hard gate
- parent control rerun、same-OOF rescue
- full gate通過前のinference / submission

## 実装状態

2026-07-27の実装承認により、compact self-contained train候補、untreated HMM
schedule、inactive / active proposal、fixed44 / full orchestration、truth-late
readout、fail-close gate、専用contract testsを実装した。科学契約、実行量、
gate、禁止救済は本設計から変更していない。正規Notebook採用、Kaggle package /
push / run、inference、submissionは未承認のまま。
