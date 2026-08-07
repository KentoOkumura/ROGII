# 設計

## 仮説

exp226のfold-safe geometry rateを有限粒子のproposalにだけ使い、元transitionを含む
defensive mixtureへ`p0/q`補正を適用すれば、target posteriorを変えずにexp410で確認した
particle support不足を減らせる。

## 結論

exp226のabsolute pathをPFのpriorやoffset stateへ入れる案は採らない。exp226は最後の
既知TVTからdonor由来の相対増分を累積するため、局所rate mismatchを長いsuffixへ
vertical offsetとして持ち越す。一方、exp410ではexp072 likelihood-PFのpersistent
offsetの主因がfinite particle support不足とwithin/across-seed平均であり、
HMMのsticky transitionとは異なると確認された。

そこで、exp226 geometryをPFのtarget modelではなく有限粒子を配置するproposalにだけ
使う。通常proposalを必ず50%残すdefensive mixtureと、厳密なimportance correctionにより、
exp226が外れたwellでもPF本来のtransition posteriorを変更しない。これはexp281の
`TVT = exp226 + delta`固定grid HMMとは異なり、continuous particles、proposal sampling、
importance weighting、ESS resamplingを使ってfinite-support問題を直接検証する。

## 状態と記号

exp072 PFと同じsurface stateを使う。

- `s_t = TVT_t + Z_t`: particle position
- `r_t`: MDあたりsurface rate
- `delta_md_t = max(MD_t - MD_(t-1), 1)`
- `m = 0.998`
- `sigma_r = 0.002`
- `sigma_p = 0.005`

通常rate transitionを次とする。

```text
mu0_t = m * r_(t-1)
p0(r_t | r_(t-1)) = Normal(mu0_t, sigma_r^2)
s_t | r_t = Normal(s_(t-1) + r_t * delta_md_t, sigma_p^2)
```

exp226側はGR correction前のfold-safe geometryだけを使う。

```text
g_t = exp226.tvt_geop_t + Z_t
r_geo_t = (g_t - g_(t-1)) / delta_md_t
```

最初のsuffix rowでは`g_(t-1)`を最後の既知`TVT_input + Z`とする。
exp226 final、`gr_delta`、U projection、truth、errorは使用しない。

## Defensive mixture proposal

rate proposalを次に固定する。

```text
q(r_t) =
    0.5 * Normal(mu0_t, sigma_r^2)
  + 1/6 * Normal(r_geo_t, (1  * sigma_r)^2)
  + 1/6 * Normal(r_geo_t, (4  * sigma_r)^2)
  + 1/6 * Normal(r_geo_t, (16 * sigma_r)^2)
```

position conditionalは通常transitionと同じにするため、importance ratioではrate密度だけが
残る。GR update前のparticle weightへ次を掛ける。

```text
importance_t = p0(r_t | r_(t-1)) / q(r_t | r_(t-1), r_geo_t)
weight_t *= importance_t * raw_gr_gaussian_likelihood_t
```

`q >= 0.5 * p0`なので`importance_t <= 2`である。importance clipは行わない。
normalizer、log marginal likelihood、ESSはimportance-corrected weightから計算する。
systematic resamplingとrougheningはexp072から変更しない。

## seed aggregation

128 seedの算術平均は使わず、exp404 x1.0で固定済みのtemperature-5 evidence weightingを
そのまま使う。

```text
seed_weight_k =
    softmax((seed_total_log_marginal_likelihood_k - max_log_likelihood) / 5)
prediction_t = sum_k seed_weight_k * seed_prediction_(k,t)
```

log marginal likelihoodはimportance-corrected incremental normalizerから積算する。
suffix TVTは使わない。全suffix GRを使うため、candidateはKaggle batch predictionであり
causal online predictionとは呼ばない。

## アプローチ

### Stage 0: implementation-time technical preflight

train-side実装承認により、次をコードとcontract testへ実装した。実artifactを使う
Kaggle fixed-probe実行は別承認である。

1. exp226 fold-safe OOF、exp404 frozen scale5 control、raw train / Type Wellの
   raw / decompressed / logical / schema SHAを検証する。
2. synthetic系列と固定probe wellでproposal density、mixture weights、
   `p/q <= 2`、finite weight、geometry weight 0 parityを確認する。
3. proposal入力allowlistを`MD / Z / raw GR / Type Well GR / last-known state /
   exp226 geop`へ固定し、truth / error / fold / hidden-like / exp226 final /
   `gr_delta`を拒否する。
4. candidate 1 variant以外のrun flagをfalseにする。

### Stage 1: full OOF single variant

別承認後だけ、全773 wellsをKaggle CPU 4 shardで実行する。

1. 各wellを同じimmutable key由来の128 seedsで生成する。
2. candidate row prediction、particle support diagnostics、importance diagnostics、
   seed evidence、well manifestをtruth前にfreezeし、SHAを確定する。
3. exp404 frozen scale5 controlとexp226 final OOFをID一対一で結合する。
4. freeze後にtruth、reporting fold、raw-GR scope、hidden-like role、
   exp410固定496-well / episode assetを読む。
5. technical、mechanism、standalone adoptionの順にgateを判定する。

### 条件付きStage 2: auxiliary allocation

Stage 1 mechanism gate PASSかつstandalone adoption gate FAILの場合でも、自動では進めない。
別のユーザー判断がある場合だけ、current-row predictive GR likelihoodでproposal componentへ
particle数を事前配分するauxiliary PFを別実験として設計する。Stage 1と同じOOFで
mixture weightや幅を調整しない。

## 実験範囲

- 対象実験: `exp419_exp226_guided_defensive_mixture_pf`
- Route: `pf_beam`
- scientific PF parent:
  `exp404_scale5_sigma_gr_likelihood_pf_ablation`
- geometry parent:
  `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- mechanism evidence:
  `exp410_likpf_particle_resampling_basin_audit`
- negative reference:
  `exp281_exp226_residual_offset_exact_hmm_transition_probe`
- control:
  saved `likpf_scale_5_x1p0`
- standalone reference:
  saved exp226 final OOF
- candidate:
  `exp226_guided_defensive_mixture_scale5`

変更する変数はrate proposalだけである。PF target transition、position conditional、
GR emission、resampling、rougheningとseed aggregationは固定する。

## 実行予定量

- active scientific PF variants: 1
- candidate PF well-runs: 773
- control PF well-runs: 0
- exp226 / HMM / Beam reruns: 0
- seed-well trajectories: `773 * 128 = 98,944`
- particle starts: `98,944 * 500 = 49,472,000`
- reporting folds: 5（学習foldではない）
- LightGBM configs / trained folds / boosters: `0 / 0 / 0`
- GPU: 0
- Kaggle CPU shards: 4
- 保守的runtime: 各shard6時間、hard stop各9時間

これは設計値であり、実行承認ではない。

## 評価と固定gate

### Technical gate

1. input / row / well / fold / finite coverage
2. proposal allowlist、mixture weight sum、density positivity
3. `importance <= 2 + 1e-12`
4. geometry weight 0 fixed-probe parity
5. saved control RMSE parity
6. variant / PF / model / booster実行量

### Mechanism gate

全条件をANDで要求する。

1. saved scale5 `10.914522073423171`比pooled gain `>=0.10 ft`
2. scale5比改善fold `>=4/5`
3. raw-GR observed gain `>=0.10 ft`
4. raw-GR missing、high-missing、1000+、hidden-like 2面のregression
   各`<=0.02 ft`
5. exp410 496-well scopeのmajority-seed predictive support外row率を
   `>=5 percentage points`減らす
6. exp410固定episode scopeのSSEをexp404 scale5比`>=10%`減らす
7. by-well delta RMSE p95 `<=0.25 ft`
8. worst-well regression `<=2.0 ft`

### Standalone adoption gate

1. exp226 final `9.427109596582213`比gain `>=0.03 ft`
2. exp226比改善fold `>=3/5`

mechanism gateだけPASSした場合はproposal機構を支持するが、PF route candidateや
inference資格は与えない。両gate PASS時だけ、同じexp419内のraw-test inference設計を
別承認で開始できる。

## 実装時のNotebook契約

Jupytext percent形式の`*_compact_selfcontained_train.py`候補と変換後のcompact
Notebookを作成した。正規Notebookは採用承認までtemplate stubのままとする。

Notebook上で少なくとも次を追える構成にする。

1. Imports
2. Runtime / stable-seed / SHA helpers
3. Raw, exp226 geometry, and saved-control input checks
4. Exact exp072 target transition and scale5 aggregation
5. Defensive mixture proposal and importance correction
6. Technical preflight and proposal diagnostics
7. Shard orchestration and candidate freeze
8. Late truth join, fold/scope/support/tail metrics, manifests, and gate

実装では、majority-seed predictive supportをtruth-lateで厳密評価するため、各shardが
全suffix row ×128 seedのpre-GR predictive TVT support min/maxをfloat32 NPYとして
predictionと同時にSHA freezeする。truth結合後にだけsupport包含率を計算し、exp410の
固定839 episode row ledgerと比較する。

## 再現性設計

- seed policy:
  `sha256("likpf::train::<well_id>") % 2147483647 + 1 + seed_index`
- stochastic処理:
  mixture-component draw、rate / position noise、systematic resampling、
  roughening。
- RNG:
  well / split / seed indexから事前生成し、global RNGやthread schedulingへ依存させない。
  geometry-weight-0 parity modeではcomponent drawを消費せず、exp404 kernelのRNG順を保つ。
- 並列:
  well内single worker。shard番号と実行順をseedへ含めない。
- CPU/GPU:
  Kaggle CPU、GPU off、internet off。
- SHA:
  exp226 OOF、exp404 frozen control、raw / Type Well、episode asset、
  code、config、proposal contract、well manifest、candidate raw /
  decompressed / logical predictionを記録する。
- gzip:
  raw gzip SHAとdecompressed CSV SHAを分ける。
- rerun:
  固定probe wellを別runで再生成し、candidate predictionとdiagnosticのlogical parityを確認する。
- Kaggle bootstrap:
  package生成後にembedded ZIP内のconfig、source、run flag、kernel sourceを正本と照合する。
- deterministic anchor:
  full coverage、全SHA、probe rerun、raw-test regenerationが揃うまでは主張しない。

## リスク

- proposal variance:
  exp226 rateが外れるとgeometry成分のweightが小さくなる。defensive 50%と`p/q <=2`で
  catastrophic importance weightを防ぐが、有効粒子数を浪費する可能性がある。
- geometry leakage:
  exp226 OOFはvalidation wellをdonor fieldとkappa fitから除外した同fold生成物に限定する。
  final / GR / truth / error列をallowlistで拒否する。
- GR二重利用:
  proposalにはGR未使用`geop`だけを使い、GRはPF emissionに一度だけ使う。
- finite-particle interpretation:
  importance correctionにより無限粒子極限のtarget posteriorはexp072と同じである。
  改善はmodel変更ではなく有限粒子近似とsupport配置の改善として解釈する。
- runtime:
  3 Gaussian mixture densityとimportance計算がexp404より重い。4 shardと9時間hard stopを
  事前固定し、controlを再実行しない。
- CV/LB:
  exp226 CV 9.427 / LB 9.837、exp263 fixed blend CV 8.238 / LB 7.800と
  絶対差がある。train-side guard PASS前にinference / submissionへ進めない。
- selection:
  mixture weight、幅、gateを同じOOFで変更しない。

## 対象外

- exp226 / PF prediction blend
- exp226 absolute TVT、offset、segment IDをPF stateやemissionへ追加
- HMM / Viterbi / forward-backward / fixed offset grid
- donor distanceによるadaptive mixture weight
- current-row GRによるauxiliary particle allocation
- mixture weight / width / importance clipのgrid
- process noise、roughening、GR sigma、ESS threshold、particle / seed数の変更
- ML / selector / meta feature
- raw-test inference / submission
