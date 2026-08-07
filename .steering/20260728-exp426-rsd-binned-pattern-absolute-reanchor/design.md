# 設計

## 結論

`exp426_rsd_binned_pattern_absolute_reanchor`を、論文scoreをHMM emissionへ
直接置換する実験ではなく、absolute datum観測の識別性から順に反証する
3-stage gate実験として設計する。

1. Stage A: RSD-binned pattern scoreにabsolute offsetの識別力があるか。
2. Stage B: 同じscoreでexp226のlocal pathを壊さずcoarse datumだけを再アンカー
   できるか。
3. Stage C: top-3 datum basinをimportance-corrected PF proposalへ使い、
   finite-particle supportを改善できるか。

Stage A/Bはdeterministicな0-model replay、Stage Cだけがstochastic PFである。
前段がFAILした場合、後段は実装しない。

## 根拠

### HMM

exp408のpersistent offset 638 episodesでは、排他的SSEの`59.3978%`が
forward transition / prior hysteresis、`23.0444%`がbackward smoothing reversal
だった。current emissionがtruth優勢からwrong優勢へ新規反転したのは
`9 / 807,710 rows`だけである。

一方、filtered rateがtruthと同方向でも0方向へunder-responseする行は
`70.9074% rows / 70.3580% SSE`で、rateが再同期してもabsolute positionだけが
戻らないtranslation-gauge lockが確認されている。したがってpointwise Gaussianを
相関へ置換するだけではなく、local rateとcoarse absolute datumを分離して再アンカー
できるかを先に検証する必要がある。

### exp226

保存OOF RMSEは`9.427109596582213`。global biasが説明するMSEは`0.101%`だけだが、
K16 segment mean-offset oracleは`9.4271 -> 1.1306`で`98.5617%`のMSEを説明した。
persistent episodesは645件で、全rowの約19%がSSEの`82.0073%`を占める。
0--50 ft RMSE `1.741`に対し2000+は`11.151`であり、一度だけのprefix anchorから
局所rate mismatchを積分し、absolute suffix offsetを戻せないことが主因である。

### PF

exp410のPF persistent offsetでは、排他的SSEの`36.4701%`がparticle support不足、
`36.2441%`がacross-seed arithmetic aggregation、
`10.8561%`がwithin-seed particle meanだった。truthがmajority particle support外の
行は`64.2853%`、SSE `83.0651%`である。

GRをほぼ無効化するとsentinel episode SSEが`8.835072倍`へ悪化したため、
GR全体を弱めるのではなく、元proposalと全anchor supportを残したまま
scoreで有限粒子の配置だけを改善する。

### 論文から採用する部分

採用するのは、candidate earth modelでMD logをRSDへ投影した後、
0.5 ft RSD binごとにGRを平均し、Type Wellとcorrelationで比較する観測設計である。
Pearson / Spearman / CosineのうちPearsonをprimaryに固定し、論文のFisher変換後の
signed scoreを使う。

論文のdip / inclination priorはconstant-offset candidateでは全candidateに共通であり、
rankingへ情報を追加しない。local geometry priorはexp226 pathそのものとして固定する。
SAMC、posterior credible interval、複数chain平均はこの実験では再現しない。

## 実験範囲

- 対象:
  `exp426_rsd_binned_pattern_absolute_reanchor`
- Route: `pf_beam`
- absolute path parent:
  `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- PF parent:
  `exp404_scale5_sigma_gr_likelihood_pf_ablation`
- HMM mechanism evidence:
  `exp408_hmm_message_rate_basin_audit`
- PF mechanism evidence:
  `exp410_likpf_particle_resampling_basin_audit`
- matched pattern controls:
  `exp280_exp226_shift_likelihood_separability_readout`、
  `exp360_typewell_reference_shift_zncc_confidence_readout`
- Stage A/B実行量:
  model / LightGBM config / trained fold / booster / HMM / PF / Beam / GPU =
  `0 / 0 / 0 / 0 / 0 / 0 / 0 / 0`
- inference / submission:
  design時点では無効

## 共通candidate contract

### 入力

candidate freeze前に読める列をallowlistで固定する。

- exp226 OOF:
  `well_id, row_idx, suffix_offset, fold, tvt_pred`
- horizontal:
  `MD, GR, TVT_input`
- Type Well:
  `TVT, GR`
- geometry:
  Stage Cのsurface state変換に必要な`Z`

`tvt_true`、`TVT` target、`error`、`abs_error`、oracle offset、
persistent episode / cause、hidden-like roleはfreeze後だけ読む。

### blockとoffset

- suffix順のnon-overlapping block: 512 rows
- offset states:
  `D = [-80,-40,-20,-10,-5,-2,0,2,5,10,20,40,80] ft`
- candidate:
  `TVT_candidate(t,k) = exp226_tvt_pred(t) + D[k]`
- tie:
  `higher score -> zero -> smaller |offset| -> negative -> positive`

exp280/exp360と同じblock / offset bankを使い、過去controlとの比較を可能にする。

### RSD-binned primary score

各block / candidateで次を行う。

1. `RSD = TVT_candidate`とする。
2. global origin 0 ft、幅`0.5 ft`で
   `bin_id = floor(RSD / 0.5)`を計算する。
3. raw finite horizontal GRをbin内で平均する。GRをforward-fillしない。
4. finite Type Well GRだけをTVT軸で線形補間し、bin centerへ対応させる。
   finite range外はextrapolate / endpoint holdしない。
5. raw finite horizontal points `>=32`、occupied paired bins `>=16`、
   両系列の標準偏差`>1e-6`をsupportedとする。
6. paired bin mean間のPearson `r`を計算し、
   `z = atanh(clip(r, -1+1e-6, 1-1e-6))`、
   `U = 0.5 * sign(z) * z^2`をprimary scoreとする。

`U`は論文のsimilarity cost項に対応し、block内candidate rankingはPearsonと同じである。
Cosine / Spearmanはdescriptive、raw pointwise Pearsonとexp280互換raw Gaussianは
matched controlとする。13 score labelのstable SHA256 permutationをnegative control
にする。

## Stage A: absolute-datum identifiability

### 手順

1. 全773 OOF wellsのtarget-free score bank、support mask、rank、top-3をfreezeする。
2. input / config / schema / logical score content SHAを確定する。
3. freeze後だけtruthをjoinし、各blockのdiscrete SSE oracle offsetを求める。
4. primary / raw Pearson / raw Gaussian / permutationで同じidentifiability指標を
   集計する。
5. 全technical / scientific gateをAND判定する。

continuous block oracleをcandidate stateへ丸めるのではなく、
`argmin_k SSE(exp226 + D[k], truth)`をdiscrete oracleとする。tie policyは
target-free rankと同じ順序に固定する。

### technical gate

- expected 773 wells / 5 folds / 3,783,989 inventoryとの整合。
- score freeze前のtruth / error / oracle / episode / hidden role read=`0`。
- supported block fraction `>=0.95`。
- supported well fraction `>=0.98`。
- score、rank、top-3、control、manifestのfinite / duplicate / order検査。
- fixed probeの独立rerunでlogical score / rank SHA一致。
- CPU runtime `<=1,800 sec`、peak RSS `<=25 GB`。

### scientific gate

全条件をANDで要求する。

- primary top-1 discrete-oracle exact match `>=0.25`。
- primary top-3 oracle coverage `>=0.55`。
- nonzero-oracle direction accuracy `>=0.60`。
- direction accuracy `>0.50`のfold `>=4/5`。
- top-1 exact matchがmatched raw Pearsonより`>=0.03`高く、改善fold`>=4/5`。
- top-1 exact matchがmatched raw Gaussianより`>=0.03`高く、改善fold`>=4/5`。
- top-1 exact matchがstable permutationより`>=0.10`高く、改善fold`>=4/5`。
- blockwise independent top-1 replayがexp226よりRMSEを`>=0.10 ft`改善し、
  改善fold`>=4/5`。
- 1000+、hidden-like spatial、hidden-like typewell-purgedでdirection accuracy
  `>0.50`かつtop-1 replay RMSE regression `<=0.02 ft`。

Stage A FAIL時はscore family、bin幅、block、offset、support、metricを救済せず閉じる。

## Stage B: exp226 coarse absolute reanchor

Stage A全gate PASSと別実装承認後だけ実装する。

### Coarse datum path

block `j`、offset state `d_j in D`のMAP pathをViterbiで求める。

```text
objective =
  sum_j U_j(d_j)
  - 0.5 * (d_0 / 5 ft)^2
  - 0.5 * sum_j ((d_j - d_(j-1)) / 10 ft)^2
```

- 初期状態はsuffix boundaryの既知datum `d=0`へ接続する。
- 初回blockは`|d_0| <=20 ft`、隣接blockは
  `|d_j-d_(j-1)| <=40 ft`のhard supportを持つ。
- 全candidate invalid blockではemissionを全state 0とし、transitionだけでcarryする。
- partial invalid candidateだけを`-inf`とする。
- score scale、transition sigma、hard supportは固定し探索しない。

row correctionはsuffix boundaryの`0`と各block centerのMAP `d_j`を線形補間し、
最終center以降だけlast valueを保持する。

```text
TVT_reanchored(t) = exp226_tvt_pred(t) + datum_correction(t)
```

exp226のlocal rate / shape、GR correction、U projectionを再計算しない。
batch full-suffix readoutであり、real-time causal filterとは呼ばない。

### promotion gate

- exp226 `9.427109596582213`比RMSE gain `>=0.10 ft`。
- 改善fold `>=4/5`。
- exp226 fixed persistent episode SSEを`>=10%`削減。
- persistent episode wellsの`>=60%`を改善。
- 1000+ RMSEを`>=0.20 ft`改善。
- 0--50 / 50--100 ft、raw-GR missing、hidden-like 2面のregressionを各
  `<=0.02 ft`。
- corrected pathで新規検出されるpersistent episode SSEを、
  corrected全SSEの`<=5%`とする。
- by-well RMSE delta p95 `<=+0.25 ft`、worst delta `<=+5.0 ft`。
- row correction slopeは構成上`<=40/512=0.078125 ft/row`、
  nonfinite / duplicate / coverage violation 0。

FAIL時はtop-1、top-3平均、transition penalty、block、interpolation、
clip、blend、well gateを同じOOFで救済しない。

## Stage C: PF absolute-anchor proposal

Stage A/B全gate PASS、Stage B結果レビュー、ユーザーの別承認後だけ実装する。

### Stateとtarget

exp404 x1.0 / temperature-5 likelihood-PFのrate transition、500 particles、
128 seeds、raw Gaussian GR、resampling、rougheningを固定する。

surface stateを`s=TVT+Z`とし、通常rowでは親PFと同じtransitionを使う。
各512-row blockの先頭だけ、position targetを次へ置換する。

```text
p_aug(s_t) =
  0.90 * p0_continuation(s_t)
  + 0.10 * Uniform13Anchor(s_t)

anchor_k =
  Normal(
    exp226_tvt_pred(t) + Z(t) + D[k],
    sigma_anchor=1.0 ft
  )
```

rate transitionはanchor componentでも親PFと同じにし、absolute positionだけを
再アンカーする。

### Uniform controlとpattern-guided proposal

同じaugmented targetに対して2 proposalを比較する。

```text
q_uniform = p_aug

q_guided =
  0.90 * p0_continuation
  + 0.05 * Uniform13Anchor
  + 0.05 * EqualWeightTop3PatternAnchor
```

top-3はStage Aでfreezeしたrankを使う。unsupported blockは`q_uniform`へfallbackする。
`q_guided >= 0.5 * p_aug`なので`p_aug / q_guided <=2`を構成上保証する。
importance clipは使わない。

```text
weight_t *= p_aug(state_t) / q(state_t) * raw_gr_likelihood_t
```

pattern scoreはproposal allocationにだけ使われ、target densityやraw GR emissionへ
追加されない。これにより同じGRの二重尤度化を避ける。seed aggregationは
exp404のtemperature-5 full-suffix log-evidence weightingを固定する。

### Stage C0 sentinel

exp410のSHA固定12 sentinel wellsを使う。

- active PF variants: uniform 1 / guided 1
- wells: 12
- PF well-runs: `24`
- seed-well trajectories: `3,072`
- particle starts: `1,536,000`
- zero-anchor parity probe: 1 well、promotion variantには数えない
- model / LightGBM config / fold training / booster / HMM / Beam / GPU:
  `0 / 0 / 0 / 0 / 0 / 0 / 0`

mechanism gate:

- guidedのexp410 fixed episode SSEがsaved exp404比`>=10%`減少。
- majority-seed truth-support外率がsaved exp404比`>=5 percentage points`減少。
- truth-near seed fractionがsaved exp404比`>=5 percentage points`増加。
- episode onset後128 rows以内のtruth-basin recapture率が`>=0.10`増加。
- guidedがuniformよりepisode SSEを`>=5%`、support外率を`>=2 points`改善。
- guidedが`>=8/12` wellsでsaved exp404よりRMSE改善。
- sentinel worst-well regression `<=+2.0 ft`。

technical gate:

- zero-anchor probeでexp404 prediction / RNG logical parity。
- mixture weight sum、finite density、`p_aug/q<=2+1e-12`、importance clip 0。
-全13 anchorにproposal supportがあり、top-3 occupancyが非退化。
- proposal / prediction / support diagnostics freeze前truth read 0。
- per-well stable seed、common seed bank、logical SHA、duplicate / coverage guard。
- full 4-shard projectionの各最大runtime `<=27,000 sec`、peak RSS `<=25 GB`。

### Stage C1 full 773-well

C0全gate PASSとさらに別承認後だけ、guided 1 variantを4 CPU shardで実行する。
uniform controlとexp404 controlは再実行しない。

- guided PF well-runs: 773
- seed-well trajectories: 98,944
- particle starts: 49,472,000
- reporting folds: 5
- model / booster / HMM / Beam / GPU: 0

PF physical-model contribution gate:

- saved exp404 scale5 `10.914522073423171`比RMSE gain `>=0.10 ft`。
- 改善fold `>=4/5`。
- exp410 episode SSE `>=10%`削減、majority-seed support外率
  `>=5 percentage points`削減。
- raw-GR observed gain `>=0.10 ft`。
- raw-GR missing / high-missing、1000+、hidden-like 2面のregression各
  `<=0.02 ft`。
- by-well delta p95 `<=+0.25 ft`、worst delta `<=+2.0 ft`。

追加分類として、exp226比`>=0.03 ft`かつ3/5 foldsならPF standalone候補、
exp263 fixed physical blend `8.238331`比`>=0.03 ft`かつ3/5 foldsなら
PF route anchor候補と記録する。これはPF contribution gateとは分け、
未達でもexp404比の機構改善を再分類しない。

## HMMへの読み替え境界

Stage Aが検証するのは、RSD-binned scoreがcoarse absolute datumを観測できるという
共通必要条件である。Stage B/CがPASSしても、exact HMMのforward-backward、
state grid、posterior meanを変更した結果ではない。

HMMへ転用する場合は別実験で、local position-rate stateとcoarse residual-datum
stateを分離し、同じscore bankをdatum emissionへ使う。current pointwise emissionの
全面置換、Viterbi化、HMM/PF/exp226 prediction blendはこの実験では行わない。

## 再現性設計

- seed policy:
  Stage A/BはRNGなし。Stage Cは
  `sha256("exp426::<split>::<well_id>::<variant>")`由来のbase seedと
  seed indexを使う。
- stochastic components:
  Stage Cのparticle initialization / propagation、mixture draw、
  systematic resampling、roughening。
- parallel RNG:
  global RNGを禁止し、well / variant / seed indexのlocal RNGを使う。
  shard番号、well順、thread schedulingをseedへ含めない。
- runtime:
  Kaggle private CPU、GPU / internet無効、well内single worker。
- SHA:
  raw / Type Well、exp226、exp404、fixed manifest、config、code、score bank、
  rank/top-3、reanchored prediction、PF prediction、support diagnosticsを記録する。
  gzipはdecompressed content SHAを主証拠にする。
- model / submission SHA:
  modelとsubmissionを作らないため非該当。inference設計時に追記する。
- deterministic anchor:
  implementation-ready時点ではfalse。Stage A/Bは独立rerun logical SHA、
  Stage Cはfixed probe parityとfull raw-test regenerationが揃うまで
  deterministic anchorと呼ばない。
- Kaggle bootstrap:
  実装後の各push前にloose / embedded config、source、run stage、
  input asset SHAを照合する。

## リスク

- 識別性:
  exp280のraw-GR shift top-1 truth-nearestは約`18.95%`、
  exp360 ZNCC bad10 AUCは`0.505164`でnegative。RSD binningだけで十分な
  absolute offset signalが得られない可能性が高い。
- HMM誤帰属:
  exp408ではcurrent emissionはroot causeでない。Stage Aを通過しても
  HMM改善を自動的に主張しない。
- exp226 path依存:
  scoreはexp226 final pathのlocal shapeが概ね正しいことを前提とする。
  local shape自体が誤るblockではconstant-offset bankは救済できない。
- scoreの未来利用:
  512-row全体を使うbatch scoreでありreal-time causal geosteeringではない。
  コンペのfull suffix推論に限定して解釈する。
- PF target変更:
  Stage Cはproposalだけでなく10% uniform absolute-anchorをtarget transitionへ
  加える物理モデル変更である。uniform / guided比較で寄与を分離する。
- double counting:
  pattern scoreはqだけに使い、`p/q`補正する。scoreを追加likelihoodへ入れない。
- mode averaging:
  Stage Cはtop-3を別anchor componentとして保持し、offset値やPF predictionを
  事前平均しない。
- runtime:
  full PFは49,472,000 particle startsを伴うため、C0 sentinelと別承認を必須にする。
- same-OOF rescue:
  Pearson / Cosine / Spearman、bin / block / offset、Viterbi prior、
  anchor weight / sigma、top-K、PF temperatureの結果後差替えを禁止する。

## 禁止する救済

- Stage A FAIL後のbin幅、block、offset、support、score family、threshold grid
- Cosine / Spearmanを見てprimaryへ差し替えること
- truth/error/oracleによるblock / well activation
- exp226 correctionのclip、blend、well gate、top-K平均への差替え
- HMM emission全面置換、Viterbi全行置換、HMM/PF/exp226 prediction blend
- PF anchor weight、sigma、top-K、particle / seed、GR sigma、rougheningのgrid
- guided proposalでtop-3以外のanchor supportを0にすること
- `p/q` clip、pattern scoreの追加likelihood化
- parent control再実行、full gate前のinference / submission

## Design-only境界

この設計セッションで作るのはsteering、実験scaffold、config、README、
SESSION_NOTES、result、metrics、backlog / summary記録だけである。
compact source、正規Notebook編集、tests、生成物、Kaggle package / push / run、
inference、submissionは作成しない。

## Stage A実装境界

2026-07-28の追加依頼では、正規Notebookを上書きせず、
`exp426_rsd_binned_pattern_absolute_reanchor_compact_selfcontained_train.py`
と対応する`.ipynb`候補を実装する。

- exp226 final `tvt_pred`だけをcandidate base pathとして読む。
- RSD-binned signed Fisher-Pearson、raw pointwise Pearson、
  exp280互換raw Gaussian、stable score-label permutationを同一block /
  offset bankで生成する。
- score / support / rank / top-3 / manifest / logical SHAと固定probe再実行を
  freezeした後だけ`tvt_true`とhidden-like roleを読む。
- technical FAIL時はtruthを読まずfail-close生成物を保存する。
- unsupported blockのreplayはoffset 0へfallbackし、RMSEはscope全row、
  identifiabilityはsupported blockだけで評価する。
- primary / controlの識別率比較はprimary RSD supportの共通block集合で行い、
  control固有invalidはoffset 0 fallbackとして数える。
- Stage B / C、正規Notebook採用、Kaggle package / push / run、
  inference、submissionは今回も行わない。
