# 設計

## 1. 結論

`exp428_similar_well_gr_registration_map_transfer_readout`を、similar-well間の
GR registration転送性だけを測る0-model Stage 0として固定する。

donor wellごとに、true TVTを物理座標の基準として

```text
Horizontal_GR(i) ≈ TypeWell_GR(true_TVT(i) + delta_w(i))
```

を最もよく満たす`delta_w(i)`をft単位で求める。queryとHorizontal GR波形が最も似た
outer-train donorを選び、donorのglobal shiftをqueryのregistration shift予測として
転送する。query true TVTから作る参照shiftは、予測とartifactをfreezeした後だけ作る。

## 2. exp423との違い

exp423はqueryとdonorのHorizontal GRをDTW対応させ、
`donor_true_TVT(progress) - donor_true_TVT(start)`という正解TVT pathをqueryへ転写した。
exp428ではHorizontal-to-Horizontal DTWはdonor順位にしか使わない。

転写対象は次だけである。

- global registration shift: donor全体で代表する`何ftずらすか`
- local registration map: progressごとの`delta_w(s)`
- mapping shape readout: global shift、線形drift、局所warp振幅

donorのTVT増分、絶対TVT、geometry、rateはquery出力へ入れない。したがってexp423の
truth-warp transfer failureとは異なる仮説であり、exp423を救済・再分類するものでもない。

## 3. 実験範囲

- 対象実験: `exp428_similar_well_gr_registration_map_transfer_readout`
- Route: `pf_beam`
- 親実験: `exp423_same_typewell_gr_dtw_truth_warp_transfer_readout`
- 主参照:
  - `exp065`: native-overlap Type Well groupとpairwise軸対応
  - `exp288`: true/known TVTでType Well GRをHorizontal GRへsampleする座標定義
  - `exp311`: same-Type-Well群のGR関係に平均的転送性がある一方、worst-well safetyは未解決
  - `exp360` / `exp365`: 同一well内のshift/registration evidenceは弱く、
    predictionへ直結させるべきでないというnegative evidence
- reporting folds: 5
- 学習対象: なし
- 実行対象外: ML、HMM、PF、Beam、test inference、submission

## 4. fold、query、donor

### 4.1 foldとquery

- `exp423` / `exp109`と同じsorted-well、seed 42の5-fold identityを使う。
- queryはfold内のouter-valid wellとする。
- query regionは既存pseudo-unknown suffixとする。
- suffix GRはraw testでも観測できるため、donor検索に使用可能とする。
- suffix true TVTはfreeze前に読まない。

### 4.2 donor pool

donorは次をすべて満たすwellとする。

1. query foldのouter-train well。
2. queryと同じ`exp065 native_overlap=1` group。
3. Horizontal suffix GRのfinite率70%以上。
4. queryとの共通resampled supportが70%以上。
5. donor registration mapにidentifiable blockが3個以上ある。

eligible donorが1本以上ならsupportedとする。最大5本までGR距離順に保持し、
primaryはrank 1だけを使う。query/donor集合の交差はfoldごとに0をassertする。

## 5. Type Well軸の対応

`row_lag_b_minus_a`と`row_lag_ft_equivalent`は、Type Well CSVの先頭・末尾trim差や
sampling間隔を表し得るため、そのままregistration shiftへ足さない。

`exp065 typewell_native_overlap_pairs.csv`のうち、`exact_match_rate=1.0`かつ
`overlap_fraction_shorter>=0.80`のedgeだけを使う。設計時のtarget-free確認では、
このexact edgeは10,697件あり、TVT軸差は全件0 ftだった。一方、10,656件はrow lagが
非ゼロであり、row lagとTVT軸差が別物であることを確認した。

exact edgeについて、

```text
axis_offset_b - axis_offset_a
  = median(TypeWell_TVT_b - TypeWell_TVT_a)
```

をedgeとするgroup graphを作る。group representativeを0 ftとして各wellの
`axis_offset_w`を決める。各edgeのmin/max差とgraph cycle residualは`1e-6 ft`以内を
必須とし、超えたgroupはinvalidとする。

donor shiftをqueryのType Well軸へ移す式は次に固定する。

```text
delta_donor_to_query
  = delta_donor + axis_offset_query - axis_offset_donor
```

現在のnative-overlap pairでTVT軸差が0 ftでも、この変換を省略しない。

## 6. donor registration mapの作成

### 6.1 blockとshift

- donor pseudo suffixを512 row、stride 512 rowの非重複blockに分ける。
- shift候補は
  `[-80,-40,-20,-10,-5,-2,0,2,5,10,20,40,80] ft`に固定する。
- 各shiftで
  `TypeWell_GR(true_TVT + delta)`をlinear interpolation、no extrapolationで得る。
- Horizontal GRとのraw-finite ZNCCを、finite pair 32以上のときだけ計算する。
- best shiftは`(ZNCC desc, |shift| asc, shift asc)`で一意に決める。

positive shiftは「同じHorizontal rowのtrue TVTより深いType Well TVT位置をsampleすると
GRが合う」を意味する。

### 6.2 identifiable blockとmap summary

次を両方満たすblockだけをidentifiableとする。

- best ZNCC `>= 0.30`
- bestとsecond bestのZNCC差 `>= 0.05`

donor mapは以下で表現する。

- `local_offset(s)`: identifiable blockのbest shift。block間は線形補間、端は最寄値保持。
- `global_shift`: identifiable block shiftのmedian。
- `stretch_ft_per_suffix`: normalized progressに対するlocal offsetの最小二乗傾き。
- `local_warp_mad_ft`: shiftと線形傾きを除いた残差のMAD。

primaryは`global_shift`だけを使う。local mapとshape summaryは、どの対応付け型が
転送可能かを分解する追加readoutである。

## 7. GR波形が似たwellの選択

query/donorのHorizontal suffix GRだけを使い、exp423と同じ固定処理を行う。

1. centered rolling median 5 rows
2. normalized suffix progress上の256点へ線形補間
3. median / MADによるrobust z-score
4. forward-only constrained DTW
5. Sakoe–Chiba band 32 points、axis run上限4
6. `(normalized cost asc, donor well_id asc)`で順位付け

Type Well groupはeligibility guardであり、Horizontal GR DTWが「似たwell」の順位である。
GR距離、support、well ID以外のdonor truthやquery truthを順位に使わない。

## 8. candidateとcontrol

### 8.1 primary

`selected_top1_global_shift`を唯一のprimaryとする。rank-1 donorのglobal shiftを
5節のType Well軸式でquery軸へ変換する。

### 8.2 controlsとdiagnostics

- `zero_shift`: 0 ft。転送自体の必要性を測る。
- `stable_random_same_group`: eligible donorをquery well IDのSHA256で1本選ぶ。
  GR類似順位の追加価値を測る。
- `same_group_median_global_shift`: outer-train donorのglobal shift median。
  個別analog選択がgroup平均より有用かを測る。
- `top5_oracle_global_shift`: freeze後、上位5 donorからquery global reference shiftに
  最も近い1本を選ぶ。transferability headroom専用でdeployしない。
- `selected_top1_local_map`: rank-1 donorのlocal mapをHorizontal-to-Horizontal DTW pathで
  query progressへ写す。global shiftに対する局所shapeの追加価値だけを測る。

run後にlocal、oracle、group medianをprimaryへ差し替えない。

## 9. query referenceとlate truth join

query truthを読む前に次をfreezeする。

- fold / row inventory
- Type Well group / axis offsets
- outer-train donor registration maps
- query-donor GR距離と順位
- primary、controls、local diagnostic map
- support / fallback
- input、config、schema、logical/decompressed content SHA

freeze後だけquery true TVTを読み、6節と同じblock、shift grid、ZNCC、support、tie-breakで
query reference registration mapを作る。query reference global shiftはidentifiable
block shiftのmedianとする。

unsupported queryのartifact値は0 ftとするが、supported-only primary metricへ混ぜない。
coverage自体をtechnical gateで判定する。

## 10. 評価

primary metricはsupported query wellのequal-well
`global_registration_shift_MAE_ft`とする。追加で次を記録する。

- fold別MAE、within 2/5 ft、shift sign accuracy
- primary / zero / random / group median / oracle
- GR-DTW costとdonor-to-query shift absolute errorのSpearman
- primary shiftをquery true TVTへ加えた参照GRのZNCC gain対zero shift
- supported well/block、group/donor数、fallback
- Type Well axis offsetとgraph consistency
- local mapのblock shift MAE
- shift、stretch、local warp MADのquery/donor対応
- hidden-like spatial / typewell-purged
- by-well primary-minus-zero absolute-error deltaの分位点

registration shiftはTVT correction量としてscoreしない。TVT RMSE candidateは作らない。

## 11. 固定gate

### 11.1 technical gate

すべて必須とする。

- 5 folds完備
- donor/query intersection 0
- query truth read before freeze 0
- Type Well axis graph conflict 0
- supported query well fraction `>= 0.70`
- identifiable query block fraction `>= 0.50`
- supported prediction finite fraction `= 1.00`
- 同一input/configの独立rerunでlogical content SHA一致

### 11.2 scientific gate

すべて必須とする。

1. top-5 per-well oracleがzero shiftよりglobal-shift MAEを`>=2.0 ft`改善し、4/5 folds non-worse。
2. primaryがzero shiftより`>=1.0 ft`改善し、4/5 folds non-worse。
3. primaryがstable randomより`>=1.0 ft`改善し、4/5 folds non-worse。
4. primaryがsame-group medianより`>=0.5 ft`改善し、4/5 folds non-worse。
5. GR-DTW costとshift absolute errorのSpearmanがpooled `>=0.15`、4/5 foldsで正。
6. query ZNCC gain対zero shiftが平均`>=0.01`、4/5 foldsで正。
7. hidden-like 2面のshift MAE deltaがともに`<=0.0 ft`。
8. primary-minus-zero by-well absolute-error delta p90が`<=+5.0 ft`。

local mapはprimaryを救済しない。global gate通過後にlocal mapがglobal shiftよりblock MAEを
`>=1.0 ft`、4/5 foldsで改善し、hidden-like 2面non-worseなら
`global_and_local_registration_shape_transfer`、そうでなければ
`global_shift_transfer_only`と分類する。

## 12. 分岐規則

- technical FAIL: invalidまたはsupport不足。科学結論へ進まない。
- oracle FAIL: cross-well registration-map transferを閉じる。
- oracle PASS / primary FAIL: map headroomはあるがGR類似donor選択が失敗。
- global PASS / local FAIL: constant global shiftだけを支持。
- global / local PASS: globalと局所shape転送を支持。
- PASS後のHMM/PF/Beam observation offset prior、candidate統合、test parityは別実験、
  新steering、ユーザー承認を必須とする。

shift/block/group/similarity/support/gateのsame-OOF rescueは行わない。

## 13. 再現性設計

- seed policy: real pathは乱数なし。random controlだけstable SHA256 per query well。
- stochastic処理: なし。
- PF/Beam / likelihood-PF / seed bagging: 新規実行なし。
- 並列処理: 初回はsingle process。tie-breakとwell/orderを辞書順に固定する。
- runtime: CPU-only、GPU/AMP/internet off。
- SHA: raw input、config、fold/row inventory、Type Well axis graph、schema、
  target-free logical content、gzipはdecompressed content SHAを記録する。
- model / prediction / submission SHA: model、TVT prediction、submissionを作らないため
  not applicable。

## 14. 実装境界

2026-07-28の追加実装依頼により、compact self-contained Jupytext train source、正規train
notebook、専用testまでを実装する。科学設計、primary、control、gateは変更しない。
その後の追加実行依頼によりKaggle CPU package / push / Stage 0 runを承認した。
inference、submissionは別承認のままとする。
- deterministic anchor: 独立rerunのlogical content SHA一致後に限りreadout artifactを
  deterministicと扱う。submission anchorにはしない。
- Kaggle package: 実装・runが承認された場合のみ作成し、metadataとbootstrap内
  `config.yaml`の一致をpush前に確認する。

## 14. リスク

- donor registration label自体がGR aliasや振幅差で不安定な可能性があるため、
  identifiability coverageを独立gateにする。
- same-Type-Well groupでもHorizontal GR生成過程が異なり、global shiftが転送しない
  可能性がある。
- full suffix GRによるdonor検索はtestで利用可能だが、train pseudo-tailとhidden testの
  suffix長・欠損分布が異なる可能性がある。
- donor truthから作るregistration mapはfold-safe supervisionであり、current testへ
  持ち出す際は全train donor mapのartifact/SHAとraw-test parityが別途必要になる。
- registration offsetの符号をTVT error correctionへ流用すると物理的意味を取り違える。
