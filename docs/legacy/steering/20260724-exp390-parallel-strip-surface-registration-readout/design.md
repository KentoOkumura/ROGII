# 設計

## 根拠

`exp114` の773-well geometry summaryをquery wellごとに再集計すると、最近傍wellとの
axial angle差は中央値`0.411°`、`5°`未満`95.34%`、`10°`未満`97.80%`だった。
最近傍centroid距離中央値は`482.57 ft`、query軸に対するcross-track距離中央値は
`383.90 ft`、projected along-track overlap中央値は`0.99936`である。
well tortuosityは`98.58%`が`<=1.01`、全wellが`<=1.05`だった。

一方、既存結果は well-level の近傍集約だけでは不十分である。

- `exp114`: trajectory-shape K8 correctionはlikPFを`11.594898→11.151818`へ改善したが、
  XY-onlyも`11.157376`でほぼ同等、357 wells悪化、worst`+6.508121 ft`。
- `exp201`: XY近傍 residual-profile correlation中央値`0.0082`、
  bias sign一致率`0.4945`。近傍残差の直接コピーは支持されない。
- `exp119`: same-typewell horizontal GR transferはlikPFより悪化。
- `exp226`: K16 segment / XY local-linear donor fieldはCV`9.427109597`まで成立したが、
  explicitなparallel-strip registrationではなく、Public LB`9.837`。
- `exp273`: genericな2D gradient direct candidateはscalar controlより悪化。
- `exp383`: all-TVT absolute/vector fieldを実行中だが、6 Formation signatureを使う
  generic 2D fieldであり、同じalong-track位置を明示対応する本仮説とは分ける。

したがって、平行性をwell similarity featureとして追加するのではなく、XYを
「進行方向` s `」「横方向` n `」へ変換し、同じ`s`にあるouter-train surfaceを
横方向補間する座標系として使う。

## アプローチ

### 1. Query-centric parallel-strip座標

各query wellの全`X/Y`軌跡に2次元PCAを適用し、第一主成分をaxial unit vector
`e_s`とする。符号はabsolute component最大の成分が正になるようcanonicalizeし、
`e_n=(-e_sy,e_sx)`をcross-track unit vectorとする。

query row位置`p_q`に対して、

```text
s = dot(p, e_s)
n = dot(p - p_q(s), e_n)
```

と定義する。query centerlineは各query nodeで`n=0`である。global XYの回転・平行移動に
依存しないよう、距離、角度、overlapとcanonical axisだけを使用する。

### 2. Pair eligibilityとdonor固定

donor候補はouter-train wellsだけとし、queryとのpairが次をすべて満たす場合だけ使う。

- axial angle mismatch: `<=5°`（方向はmodulo π）
- query score rangeに対するprojected along-track overlap: `>=0.80`
- median absolute cross-track distance: `<=2000 ft`
- donorのprojected `s` stepがcanonical方向へ進む割合: `>=0.99`
- queryと同一wellではない

eligible donorをmedian cross-track distance、well idの順に並べ、最大16 unique wellsに固定する。
strip fitには4 unique wells以上、かつqueryの`n=0`に対して正側・負側を最低1 wellずつ要求する。
片側だけの行、support不足行、non-monotone donor、range外挿行はstripを使用しない。

### 3. Same-s donor surface

outer-train donorの全row truthから`S=TVT+Z`を計算する。query nodeはtarget MDの64 ft間隔で
固定し、各query nodeのglobal `s`へ、各donorの`S(s)`と`X/Y(s)`を線形補間する。
range外への線形外挿は禁止する。

各nodeでdonor sample `(n_d,S_d)`へ次のweighted robust local linearをfitする。

```text
S_d = a(s) + b(s) * n_d
w_d = exp(-n_d^2 / (2 * 1000^2))
```

- Huber delta: `1.345`
- IRLS iterations: `5`
- slope ridge: weighted normal matrix traceの`1e-6`
- stable donor order: `(abs(n), well_id)`
- `a(s)` / `b(s)`のalong-track smoothing: 5-node centered median

query surfaceは`S_strip(s)=a(s)`である。64 ft node間をquery rowへ線形補間する。

### 4. Known-prefix gauge calibration

targetの既知prefixだけで

```text
r_prefix = TVT_input + Z - S_strip
```

を作り、deterministic Huber intercept（delta`1.345`、5 iterations）を1個fitする。
prefix finite rowが64未満ならstrip candidate全体を無効とする。slope/scale/warpはfitせず、
vertical gauge offsetだけを`S_strip`へ加える。

### 5. 単一candidateとfallback

科学candidateは1本だけとする。

```text
parallel_strip_two_sided_fallback_exp226
```

two-sided stripが有効な行では`TVT_pred=S_strip_calibrated-Z`、それ以外は保存済みexp226 OOFを
exact fallbackする。alpha、clip、soft blend、one-sided shrink、candidate平均は作らない。
row/block/well oracleは診断集計だけに使い、oracle predictionは保存しない。

## 実験範囲

- 対象実験: `exp390_parallel_strip_surface_registration_readout`
- Route: `pf_beam`
- 親・control: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
  保存済みouter-5-fold OOF、CV`9.427109596582213`、decompressed SHA
  `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`
- 参照: exp114 / exp119 / exp201 / exp226 / exp273 / exp383。
- 変更する変数: query-centric `(s,n)` registrationとsame-s two-sided
  outer-train `S=TVT+Z` cross-track interpolation。
- 固定する変数: exp226 fold identity、score rows、保存control、pair thresholds、
  donor数、64 ft node、1000 ft bandwidth、Huber、median smoothing、prefix intercept、
  fallback、report scopes。
- fitted model / LightGBM / HMM / PF / Beam: なし。
- 実行量: 1 candidate / 5 reporting folds / 773 query strip solves /
  model config・trained fold・booster各0 / exp226 control再生成0。

## 段階と停止条件

### Stage 0: target-free geometry/support/resource

16 wellsのresource preflightを先に行い、次をAND判定する。

- rows / wells / fold identityがexp226 controlと一致
- outer-valid donor overlap `0`
- freeze前のtarget suffix truth / raw Formation / GR read `0`
- finite fallback prediction coverage `1.0`
- two-sided strip eligible row coverage `>=0.50`
- two-sided strip eligible well coverage `>=0.75`
- eligible nodeのunique donor p05 `>=4`
- eligible pair angle p95 `<=5°`
- eligible pair overlap p05 `>=0.80`
- projected full runtime `<=7200 sec`
- projected peak RSS `<=16 GB`

1つでもFAILなら、pair threshold、donor数、one-sided rule、bandwidthを救済せず閉じる。

### Stage 1: known-prefix rolling-origin

known prefixが1024 rows以上あるwellで末尾512 rowsを隠し、prefix calibrationはそれ以前だけでfitする。
比較baselineはcut rowで`S=TVT_input+Z`を固定する`vertical_only_anchor`とする。
exp226保存OOFはunknown suffix専用なので、このStageはstripが有効なheldout prefix行だけを
real/circular/vertical-onlyの共通scopeでscoreし、exp226 fallbackは使用しない。

- heldout prefix RMSE gain `>=0.25 ft`
- positive folds `>=4/5`
- eligible heldout coverage `>=0.50`
- circularly shifted donor-s controlよりreal donor-sが`>=0.10 ft`良い

全PASS後だけstrip prediction、pair、fit、calibrationのlogical SHAをfreezeしてStage 2へ進む。

### Stage 2: truth-late suffix score

freeze後だけouter-valid suffix truthをjoinし、保存済みexp226 OOFと比較する。

Scientific-support gate:

- pooled RMSE gain vs exp226 `>=0.25 ft`
- positive folds `>=4/5`
- strip-eligible rows RMSE gain `>=0.50 ft`
- `1000+` RMSE gain `>=0.25 ft`
- near `0--250 ft` regression `<=0.05 ft`
- hidden-like spatial / typewell-purged regression `<=0.0 ft`

Promotion-safety gate:

- improved-or-equal wells `>=` worse wells
- by-well delta p95 `<=0.0 ft`
- worst-well delta `<=+0.25 ft`

scientific gateだけPASSしてpromotion gateをFAILした場合、parallel-strip signalは
別実験のcandidate/feature候補として記録できるが、本実験のinference/submissionは開かない。
両gate PASSでもcurrent-test generation、inference、submissionは別承認とする。

Report-only:

- row / H128 / H256 / H512 / whole-wellで
  `min(exp226, strip_candidate)`のadd-one oracle gain
- one-sided / no-overlap / low-support / edge-of-family層別
- cross-track distance、angle、overlap、prefix reconstruction error層別

oracle prediction、oracle gate、truth-derived selectorは保存しない。

## exp383との順序

exp390はno-formation / explicit-registrationであり科学的には独立だが、
exp383が既にall-TVT absolute/vector fieldを実行中である。

- exp383がStage 0/1を全PASSしexp226を`>=1.0 ft`改善し、hidden-likeとpromotion safetyも
  満たした場合、exp390は重複候補として実装前にdemote/closeを再検討する。
- exp383がvector condition、donor dispersion、surface support、direct scoreのいずれかでFAILした場合、
  exp390はそのparameter救済ではなく、parallel degeneracyをstrip coordinateへ変換する独立P1として残す。
- exp383の結果をexp390の固定threshold変更には使用しない。

## 再現性設計

- seed policy: RNGなし。fold、query well、donor well、node、pair、fit、rowをimmutable keyでstable sortする。
- stochastic処理の有無: なし。
- PF/Beam / likelihood-PF / seed baggingの有無: なし。
- 並列処理と乱数の関係: RNGなし。parallel workerを使う場合もwell単位結果をstable keyで再sortする。
- CPU/GPU runtimeとdeterministic flags: CPUのみ、GPUなし、internet off。
- train cache / test regeneration SHA:
  exp226 control、fold manifest、geometry summary、eligible pair table、node donor table、
  fit diagnostics、calibration、predictionのschema/logical/decompressed content SHAを記録する。
- model manifest / prediction / submission SHA:
  fitted modelなし。solver contract manifestとOOF prediction SHAを保存し、submission SHAは未対象。
- Kaggle package bootstrap:
  将来の別承認後にprivate / CPU / internet off、canonical id/title、config/source/bootstrap SHAを照合する。
- deterministic anchor:
  初回成功runでは主張せず、同じ設定の成功rerunでpair/node/fit/prediction logical SHA一致を要求する。

## リスク

- リークリスク:
  outer-valid TVTがdonor surface、pair eligibility、support threshold、prefix calibrationへ入ること。
  role-read ledger、source/valid overlap 0、freeze前truth read 0をhard gateにする。
- 地質リスク:
  wellboreが平行でも地層が連続・平行とは限らない。fault、landing zone差、cross-track外挿で破綻し得る。
  two-sided supportとprefix rolling-originを必須とし、片側外挿を禁止する。
- CV/LB不一致リスク:
  trainのspatial densityがhidden testと異なる可能性がある。Public 3 wellsで設定を選ばず、
  geometry coverageとhidden-like 2面を分離評価する。
- ランタイム/メモリリスク:
  4M rowsの全pair展開が大きい。64 ft query node、max16 donor、batch処理、
  16-well projectionでfull前に止める。
- 再現性リスク:
  PCA axis符号、equal-distance donor、Huber convergence、median edge処理。
  canonical axis、stable tie、固定iteration、logical SHAで管理する。
- 同一OOF救済リスク:
  thresholdやbandwidthの小変更で改善を作りやすい。全値を設計時点で固定し、
  FAIL後のgrid、soft blend、clip、selectorを禁止する。
