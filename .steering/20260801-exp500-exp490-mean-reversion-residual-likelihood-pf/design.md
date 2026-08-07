# 設計

## 根拠

exp490は、exp357 residual-offset exact HMMへK16区間1つのhalf-lifeを持つgeometry平均回帰を
追加し、full OOF RMSEを`9.737195 -> 8.480155 ft`、persistent episode SSEを
`41.4100%`改善した。一方、by-well delta RMSE p95は`+7.257814 ft`、worstは
`+49.602560 ft`で、固定強度の全well適用はfail-closedとなった。

exp486のPF variant Bは同じ`TVT=tvt_geop+offset`座標を持つが、遷移は
`offset_t=offset_(t-1)+rate_t*dMD+noise`であり、offsetそのものをgeometryへ戻す
`rho_t`を持たない。full OOFは`11.139812 ft`でexp404 control `10.914522 ft`より悪化したが、
これはexp490型平均回帰をPFで検証した結果ではない。

exp498はexp490のtail悪化を事前のtarget-free物理regimeで分離できなかった。このため、
本実験ではadaptive gateを作らず、exp490の固定1 variantをPFへ忠実に移植して独立に反証する。

## 状態と遷移

score row `t`について次を定義する。

- `G_t`: group-safeなexp226 OOF `tvt_geop`
- `delta_t = TVT_t - G_t`: geometryからのresidual offset [ft]
- `q_t`: residual offset-rate [ft / MD-ft]
- `dMD_t = MD_t - MD_(t-1)`: 正のMD差
- `s(t)`: destination row `t`が属するexp226と同じunknown-suffix K16区間
- `L_k`: 区間`k`へ入る全transitionの`dMD`合計。`L_k > 0`を必須とする

平均回帰係数はexp490と同じ1点へ固定する。

```text
rho_t = 2 ** (-dMD_t / L_s(t))
```

各完全K16区間の`rho_t`累積積は`0.5`になる。PFの更新順序は次で固定する。

```text
q_t = 0.998 * rho_t * q_(t-1) + 0.002 * Normal(0, 1)
delta_t = rho_t * delta_(t-1) + q_t * dMD_t + 0.005 * Normal(0, 1)
TVT_t = G_t + delta_t
```

区間境界ではdestination rowの区間を使う。最初のunknown rowでは最後のknown-prefix rowからの
`dMD`を使う。process noiseの分布・draw順序はexp486を維持し、`rho_t`はnoiseを加える前の
transition centerだけへ作用させる。resampling後のrougheningはexp486と同じ順序で行う。

## 初期化と固定PF契約

- initial offset center:
  `last_known_TVT - tvt_geop_at_first_score_row`
- initial offset spread: `4.5 ft`
- initial offset-rate center / spread: `0.0 / 0.01`
- particles / seeds: `500 / 128`
- momentum / rate noise / position noise: `0.998 / 0.002 / 0.005`
- rough position / rate: `0.1 / 0.001`
- resample threshold: particle数の`0.5`
- GR emission: exp486 / exp404のcapped Gaussian、scale multiplier `1.0`
- GR base scale clip: `[10, 60]`
- missing GR: both-direction linear interpolation後にtypewell mean
- seed aggregation: full-suffix log evidenceのtemperature `5.0`
- output dtype: `float32`

temperature-5はfull suffixのGR evidenceを使うため、最終出力はbatch / non-causal predictionであり、
online filterとは呼ばない。exp490のHuber emission、exact forward-backward、state grid、
posterior meanは移植しない。

## 実験範囲

- 対象実験: `exp500_exp490_mean_reversion_residual_likelihood_pf`
- Route: `pf_beam`
- 科学的PF親: `exp486_exp226_geometry_residual_likelihood_pf`
- 平均回帰機構親: `exp490_geometry_centered_mean_reverting_offset_hmm`
- PF実装参照: `exp486` compact self-contained train
- geometry親: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- control: 保存済みexp404 `likpf_scale_5_x1p0`、保存済みexp486 residual PF
- mechanism readout: exp408 HMM persistent episode、exp410 PF persistent episode
- 変更する変数: residual rateとoffsetのtransition centerへ`rho_t`を追加
- 固定する変数: PFの初期化、noise、emission、resampling、roughening、particle / seed、
  temperature、GR欠損処理、出力dtype、fold、scope、評価指標
- scientific variant: `k16_half_life_mean_reverting_residual_likpf` 1件

これはPF target transitionの変更である。元targetを維持するimportance-corrected proposal-only案、
parent / mean-reversion mixture、absolute geometry unaryとの併用は別仮説として本実験に含めない。

## Stage 0: fixed44 mechanism preflight

次の固定assetのwell列だけをcandidate生成前にunionする。

- exp411 fixed32: persistent 16 + matched control 16
- exp410 PF counterfactual sentinel: 12
- expected unique wells: 44
- expected overlap: 0

role、cause、episode、truth、foldはcandidate prediction、target-free PF ledger、scientific contract、
prediction content SHAをfreezeした後だけ読む。Stage 0 pooled RMSEはselection-biasedなので、
CVやfull promotion metricとして扱わない。

### Stage 0実行量

- candidate PF: `1 variant x 44 wells = 44 PF well-runs`
- seed-well trajectories: `44 x 128 = 5,632`
- particle starts: `5,632 x 500 = 2,816,000`
- saved exp404 / exp486 control PF rerun: `0`
- HMM / Beam / model / LightGBM config / trained fold / booster / GPU: `0`

### Technical gate

全項目ANDとする。

- fixed32 / sentinel SHA一致、32 + 12、重複0、unique 44 wells
- exp226 geometry inputのdecompressed SHA、allowlist、row identity、coverage `1.0`
- K16 segmentが全score rowsを重複なく覆い、全`dMD`と`L_k`が正
- `0 < rho_t <= 1`、finite coverage `1.0`
- 各完全K16区間の`rho_t`累積積が`0.5`、絶対誤差`<=1e-10`
- zero offset / zero residual-rate / zero noiseでTVTがgeometryに一致
- `rho=1` synthetic fixtureでexp486 residual transitionとbitwiseまたはfloat32 exact parity
- prediction、particle weight、ESS、log evidence、offset / rate ledgerがfinite
- candidate freeze前のtruth / error / fold / role / episode / hidden-like readが0
- stable per-well/per-seed RNG、execution count、固定順序が契約どおり
- full runtime投影`<=30,600 sec`、peak RSS`<=25 GiB`

### Mechanism / safety gate

全項目ANDとする。

- fixed32 persistent episode SSEを保存exp486 residual PFから`>=5%`削減
- persistent 16 wells中、保存exp486 residual PFから改善well `>=10`
- persistent側の改善fold `>=4/5`
- matched-control pooled RMSEの保存exp404比regression `<=0.02 ft`
- matched-control by-well delta RMSE p95の保存exp404比`<=0.25 ft`
- exp408 episode数を保存exp404比で増やさず、recovery @256 / @512を悪化させない
- PF sentinelのexp410 episode SSEを保存exp404比`>=5%`削減
- PF sentinel worst-well RMSE regressionの保存exp404比`<=2.0 ft`
- resampling count、minimum ESS fraction、offset edge/support diagnosticsにnon-finiteまたは
  全行collapseがない

1項目でもFAILなら同じfixed44でhalf-life、noise、temperature、particle / seed、gate、mixture、
proposalを救済せずbranchを閉じる。全PASS時だけ、Stage 1実装・Kaggle実行を別承認で求める。

## Stage 1: full group-safe OOF

Stage 0全PASSと別承認の後だけ、固定1 variantを773 wellsで生成する。必要ならwell identityと
suffix rowsだけで決まるdeterministic 4 CPU shardを使う。candidateとdiagnosticのSHAを
全shardでfreezeした後にtruth、fold、scope、episode、保存controlをlate joinする。

2026-08-02のユーザー明示overrideにより、Stage 0全PASSの先行条件だけを例外化する。
Stage 0の3 safety FAILとterminal-close記録は変更しない。科学式、PF設定、shard式、full gate、
禁止救済は変更せず、4 target-free CPU shardsを生成し、全shardのprediction / residual /
seed-evidence / K16-rho / well-manifest SHAをstrict mergeで再検証した後だけtruthと保存controlを読む。

### Stage 1実行量

- candidate PF: `1 variant x 773 wells = 773 PF well-runs`
- seed-well trajectories: `773 x 128 = 98,944`
- particle starts: `98,944 x 500 = 49,472,000`
- reporting folds: `5`
- saved exp404 / exp486 control PF rerun: `0`
- HMM / Beam / model / LightGBM config / trained fold / booster / GPU: `0`

### Scientific promotion gate

全項目ANDとする。

- 保存exp404 `10.914522073`からRMSEを`>=0.05 ft`改善
- 保存exp486 residual `11.139812021`からRMSEを`>=0.05 ft`改善
- exp404比の改善fold `>=4/5`
- raw GR observedでexp404から`>=0.05 ft`改善
- raw GR missing、高missing、MD 1000+、hidden-like spatial、hidden-like typewell-purgedを
  exp404から悪化させない
- by-well delta RMSE p95のexp404比`<=0.0 ft`
- worst-well RMSE regressionのexp404比`<=0.25 ft`
- exp408 persistent episode SSEをexp404比`>=5%`削減し、episode数とrecovery @256 / @512を
  悪化させない
- exp410 PF persistent episode SSEをexp404比`>=5%`削減
- exp209との固定50:50 readoutが保存基準`10.084909680`より悪化しない
- exp226 final `9.427109597`を`>=0.02 ft`上回る
  （candidate RMSE `<=9.407109597`）
- row identity、finite coverage、weight normalizationがすべて`1.0`

exp490とのprediction correlation、by-well gain correlation、exp490 catastrophic 51 wellsとの
悪化重複はreport-onlyとし、候補選択やgate調整には使わない。全gate PASSでもinference / raw-test
regeneration / submissionは同じexp500内の別設計追記と別承認を必要とする。

## 禁止事項

- half-life、K16数、rho式、momentum、noise、initial spread、roughening、resample thresholdの探索
- particle数、seed数、temperature、GR sigma / clip / emissionの変更
- Huber / Student-t emission、exact-HMM smoothing、exp490 prediction blend
- exp486 absolute geometry unaryとの併用
- parent transitionとのmixture、importance-corrected proposal-onlyへの変更
- GR confidence、geometry disagreement、early offset、truth/errorによるwell / row gate
- same-fixed44 / same-OOF winner selection、blend、selector、fallback、postprocess rescue
- control PF / HMM / Beamの再実行
- 2026-08-02の明示override以外のStage 1再実行、inference、submission

## 再現性設計

- seed policy:
  exp486実装と同じ
  `int(sha256("likpf::train::<well_id>").hexdigest()[0:16],16) % 2147483647 + 1 + seed_index(0..127)`。
  variant名をseedへ含めず、exp486 residual PFと同じimmutable labelを使う。
- stochastic components:
  particle initialization、rate / offset process noise、systematic resampling、roughening。
- global RNGを使わず、well / seedごとの局所RNGを渡す。外側well worker数やshard順序で
  各wellの乱数列が変わらないことをcontract testにする。
- stable order: well id、row、seed、particleの順を固定する。
- Stage 1 shard policy:
  `little_endian_uint64(sha256("exp500::full_pf_shard::" + well)[0:8]) mod 4`。
  shardは実行分割だけで、particle state、seed、prediction値へ影響させない。
- runtime: Kaggle private CPU、GPU off、internet offを正とする。Stage 0の各wellで
  `elapsed_seconds / suffix_rows`を記録し、そのp95と固定shard別suffix row数の積を
  shard runtime投影とする。4 shardの最大投影が`30,600 sec`以下であることを要求する。
- input SHA: raw identity、exp226 OOF geometry、fixed32、PF sentinel、episode、hidden-like、
  saved controlのraw / decompressed / logical SHAを記録する。
- contract SHA: K16境界、`L_k`、`rho_t`、transition更新順、PF固定parameter、seed式を含む。
- prediction SHA: raw gzip SHAとdecompressed content SHAを分け、logical row/value SHAも保存する。
- diagnostic SHA: per-well runtime、ESS、resampling、offset / rate、rho / K16 ledgerを保存する。
- model manifest SHA: 学習モデルがないため非該当。decoder / PF scientific contract SHAで代替する。
- deterministic anchor: design / Stage 0 / full 1回だけでは主張しない。固定probe well rerun、
  full prediction coverage、将来のraw-test regeneration再実行一致を満たした場合だけ再評価する。
- Kaggle package作成時は、bootstrap ZIP内config / sourceと正のconfig / sourceのSHA一致を
  push前に検証する。

## リスク

- 科学リスク: 平均回帰が正しい長期offset粒子を消す。exp490のp95 / worst FAILをhard gateで継承する。
- particle impoverishment: HMMと異なりPFはresampling後に失った非ゼロoffset modeを復元しにくい。
  ESS、resampling、offset supportをtruth前にfreezeして監査する。
- リークリスク: fixed32 / sentinel / episodeは過去truth由来の選択を含む。candidate生成前はwell identity
  だけを使い、role / outcomeはprediction freeze後に結合する。
- CV/LB不一致: Stage 0は原因enrichedでCVではない。full GroupKFoldとhidden-like 2面を通るまで
  direct候補にしない。full PASS後もraw-test regenerationとsubmissionは別承認とする。
- ランタイム: 49,472,000 particle startsを要する。Stage 0実測投影が上限を超えたら、
  parameter削減で救済せず閉じる。
- 再現性: stochastic PFとthread / shard順序で予測が変わり得る。stable seed、固定順序、probe rerun、
  decompressed content SHAを必須にする。

## 現在の状態

2026-08-01の後続ユーザー承認により、Stage 0 fixed44専用のJupytext source、candidate
Notebook、contract test 7件を実装し、candidateを正規train Notebookへ採用してKaggle private
CPU version 2を完走した。technical checksは13/13 PASS、persistent subsetは13/16 wells・
5/5 folds改善したが、matched-control pooled / by-well p95とPF sentinel worst-wellの3 safety
gateをFAILした。事前登録どおり`stage0_fail_closed`で終端閉鎖する。Stage 0はCVではなく、
2026-08-02のユーザー明示overrideによりStage 1だけを実装・実行する。Stage 0 fail-closeは
再分類せず、inference、submission、same-fixed44 / same-OOF rescueは引き続き存在しない。
