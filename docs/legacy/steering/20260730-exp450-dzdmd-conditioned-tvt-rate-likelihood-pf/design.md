# 設計

## 1. 科学差分

親exp404は粒子状態を
`(U=TVT+Z, r=dU/dMD, weight)`として、`r_t=0.998*r_(t-1)+noise`
を使う。exp450は座標を`(TVT, q=dTVT/dMD, weight)`へ厳密変換した上で、
visible prefixから学習した`dZ/dMD`条件付き平均の残差を持続させる。

wellのvisible prefixで、連続する行の

```text
delta_MD > 0
g_k = delta_Z / delta_MD
q_k = delta_TVT_input / delta_MD
```

がすべてfiniteなstepだけを使う。valid stepが10以上なら既存`PF_Z`と同じ
unweighted OLS `q_k = beta*g_k + intercept`を`np.linalg.lstsq`で1回だけ
推定する。10未満または係数がnonfiniteなら`beta=-1, intercept=0`へfallback
する。係数のclip、shrink、regularization、grid searchは行わない。

suffix row `t`では親と同じ`delta_MD_t=max(raw_delta_MD_t, 1.0)`を使い、
最初のrowのprevious Zはvisible prefix最終行のZとする。

```text
g_t       = (Z_t - Z_(t-1)) / delta_MD_t
mu_t      = beta*g_t + intercept
q_t       = mu_t + 0.998*(q_(t-1) - mu_(t-1)) + 0.002*epsilon_t
TVT_t     = TVT_(t-1) + q_t*delta_MD_t + 0.005*eta_t
```

`mu_(t-1)`の初期値には最後のfinite visible-prefix `g`を使う。finite `g`が
なければ0とする。初期position drawは親U-position drawからlast Zを引いた
`last_TVT_input + 4.5*N(0,1)`。初期rate drawは親のtail30
`median((delta_TVT_input+delta_Z)/delta_MD)`から`g_prev`を引き、
同じ`0.01*N(0,1)`を加える。fallback規則は親と同じ、valid step 3未満で
parent initial U-rateを0とする。

positionは親と同じtypewell範囲`[vmin-100, tmax+100]`へTVT座標でclipする。
GR emission、weight normalization、ESS、systematic resampling、roughening、
seed別log-likelihood、temperature-5 seed aggregationは変更しない。

`beta=-1, intercept=0`なら

```text
q_t = 0.998*(q_(t-1)+g_(t-1)) - g_t + noise
```

となり、親U-rate PFの厳密な座標変換になる。これをStage 0Aの技術sentinelに
使い、学習型候補とは数えない。

## 2. 単一変更の境界

- 対象実験:
  `exp450_dzdmd_conditioned_tvt_rate_likelihood_pf`
- Route: `pf_beam`
- 親endpoint: `exp417_scale5_seed_aggregation_promotion_audit`
- 実装参照・保存control:
  `exp404_scale5_sigma_gr_likelihood_pf_ablation`
- OLS参照: exp072内の`PF_Z`
- negative mechanism参照:
  `exp446_persistent_tvt_rate_exact_hmm`
- 変更する変数:
  rate座標と、well別prefix OLSで定めたtime-varying transition center。
- 固定する変数:
  particles 500、seeds 128、temperature 5、GR scale x1.0、momentum 0.998、
  rate noise 0.002、position noise 0.005、initial position spread 4.5、
  initial rate spread 0.01、rough position/rate 0.1/0.001、
  resample threshold 0.5、typewell grid、GR補間、output readout。
- 明示的に入れないもの:
  PF_Zのrate likelihood、smoothed-GR mixture、`zsig`由来noise、
  coefficient clip/shrink、追加candidate、gate/router。

## 3. Prefix-only mechanism readout

unknown suffix予測前に、wellごとに次をfreezeする。

- valid step数、fallback、beta、intercept。
- `g/q/mu`のmin/max/mean/stdとprefix fitted residual SSE。
- visible prefix末尾20 valid stepsをholdoutし、それより前に10 steps以上
  あるwellだけで行うtarget-free backtest。
- backtestでは学習型centerとexact transform center
  `mu_exact=-g`のq SSEを比較する。これは係数やwellを選ぶために使わない。

Stage 0Bで、fixed32全予測とprefix-fit ledgerのfreeze後だけexp446と同じ
persistent/control role、episode、suffix truthをattachする。candidateと
保存exp404 predictionの双方から、last visible TVTを含む有限差分
`delta_prediction/delta_MD`を作り、zero-directed under-response、
forward-cause、persistent episodeを同じ座標で比較する。

## 4. 段階と実行量

Stage 0A（technical parity、exp410 sentinel12）:

- scientific variant 0、technical exact-transform variant 1。
- parent U-rate 12 + exact transformed TVT-rate 12 = 24 PF well-runs。
- 24 ×128 = 3,072 seed-well trajectories。
- 3,072 ×500 = 1,536,000 particle starts。
- unknown suffix truth、fold、role、episode、errorは読まない。

Stage 0B（mechanism preflight、exp411/exp446 fixed32）:

- scientific variant 1 ×32 wells = 32 candidate PF well-runs。
- 32 ×128 = 4,096 seed-well trajectories。
- 4,096 ×500 = 2,048,000 particle starts。
- 保存exp404 control rerun 0。fixed32はCV/promotion evidenceではない。

Stage 1（全OOF、Stage 0A/0B全PASS・別承認時のみ）:

- scientific variant 1 ×773 wells = 773 candidate PF well-runs。
- 773 ×128 = 98,944 seed-well trajectories。
- 98,944 ×500 = 49,472,000 particle starts。
- 4 CPU shardsを想定。保存exp404 control rerun 0。

全stageでLightGBM config、trained fold、booster、fitted model、HMM、Beam、
GPUは0。実装・package・runは現在0である。

## 5. Gate

Stage 0A technical:

- same seed/particle/random-draw順でparent U-rateと
  `beta=-1, intercept=0` candidateのseed prediction、weight、
  log-likelihood、temperature-5 predictionが最大絶対差`<=1e-10`。
- rate/position update、clip、resampling、roughening、finite coverage、
  execution count、seed identity、artifact readback SHAがすべてPASS。
- freeze前のtruth/error/fold/role/episode読取0。

Stage 0B target-free / mechanism:

- OLS/fallback contract、全係数と全予測finite、prefix-fit/prediction SHA、
  32 unique wells、persistent/control各16、5 reporting foldsを確認する。
- eligible prefix backtestの学習型q SSEがexact `-g`よりpooledで非悪化し、
  4/5 reporting folds以上で非悪化。
- zero-directed under-response SSE shareを保存exp404より絶対`0.05`以上削減。
- forward-cause / persistent episode SSEをそれぞれ`10% / 5%`以上削減。
- persistent改善well `>=10/16`、persistent改善fold `>=4/5`。
- matched control pooled RMSE delta `<=+0.02 ft`、
  by-well delta p95 `<=+0.25 ft`。
- full runtime投影`<=30,600 sec`、peak RSS`<=25 GB`。

Stage 1 scientific:

- 保存exp404 scale-5 x1.0 RMSE `10.914522073`から`0.05 ft`以上改善。
- 4/5 folds以上改善。
- raw GR observedは`0.05 ft`以上改善。
- raw GR missing、高missing、1000+、hidden-like spatial、
  hidden-like typewell-purgedは各RMSE regression `<=0.0 ft`。
- by-well delta p95 `<=0.0 ft`、worst regression `<=0.25 ft`。
- exp209との固定50:50 HMM/PF blend `10.084909680`より非悪化。
- 全項目AND。1項目でもFAILならterminal closeする。

FAIL時はbeta/intercept、minimum support、prefix window、holdout、
momentum/noise/roughening、particle/seed、temperature、GR scale、rate likelihood、
well/row gate、blend/selectorで救済しない。

## 6. 再現性と承認境界

- seed:
  `sha256_first16("likpf::train::<well_id>") mod 2147483647 + 1 + seed_index`
  のexp404 policyを継承する。
- global RNGをthread間で共有せず、well/seedごとに独立streamを作る。
- stable orderはwell id、元row、seed index、particle index。
- Stage 0Aのpaired pathは同じ初期乱数、process noise、resampling uniform、
  roughening drawを同じ順序で消費する。
- prefix-fit ledger、scientific config、prediction schema/logical content、
  diagnostics、truth-late ledger、Kaggle package/kernel versionのSHAを記録する。
- gzipはdecompressed logical content SHAを主証拠にする。
- train/testは別にraw inputから生成し、同じprefix-fit contractを適用する。
- 初回成功runはdeterministic anchorとしない。独立rerunのprediction content
  SHA一致までanchor化しない。
- Kaggle bootstrap時にpackage内configとrepository configの一致をfail-fastする。
- model/submissionは対象外。inference/submissionは別承認がない限り作らない。

## 7. リスク

- prefix OLSはin-sampleで、unknown suffixへbeta/intercept関係が持続しない。
- gのrangeが狭いwellではOLS係数が不安定になり、tail driftを増やし得る。
- per-well prefix fitはtarget-freeだが、suffix truthより良いwellだけを選ぶと
  leakageになるため、fallback以外のwell選択を禁止する。
- exp417は平均改善してもwell-tail gateをFAILした。primary平均だけでなく
  p95/worstとhidden-likeをAND gateにする。
- 500×128の全773 wellsはCPU負荷が高い。fixed12 parityとfixed32 mechanismで
  fail-fastし、fullは別承認にする。
- Numba/thread scheduleで乱数消費が変わる危険があるため、per-well/seed streamと
  paired draw parityを必須にする。

## 8. Version 2 technical parity amendment

Version 1のStage 0Aでは、11 wellsはresampling分岐なしで最大`1e-9`級、
`5f4d2a52`は57 resampling分岐を伴う大きな内部差になった。しかし全12 wellsの
temperature-5集約予測差は最大`4.836692824e-09 ft`であり、科学candidateが
比較する親の最終readoutは実用上同一だった。

ユーザー承認により、version 2のtechnical gateは最終readout parityを正とする。

```text
max_abs(parent_temperature5 - exact_temperature5) <= 1e-6 ft
```

`1e-6 ft`は約`0.305 micrometer`で、出力精度やRMSE判定に影響しない
工学的ゼロとして事前固定する。seed prediction、weight、log-likelihood、
position/rate、resamplingは`diagnostic_checks`へ分離し、値とmismatch countを
削除せず保存する。clip mismatch、finite coverage、well/run/seed/particle count、
truth-free input、artifact readbackは引き続きhard gateとする。

artifact readbackはscientific dataを変えず、CSVを`%.17g`で保存し、
`float_precision="round_trip"`で再読込することでbinary64のlogical contentを
検証する。Stage 0BのOLS、candidate PF、保存control、truth-late、全mechanism
gateは禁止事項を含めて変更しない。
