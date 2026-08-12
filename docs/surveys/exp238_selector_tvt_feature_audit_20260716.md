---
title: exp238 selector / TVT feature audit
date: 2026-07-16
types:
  - experiment_review
  - model_explanation
  - oof_analysis
  - feature_analysis
  - comparison
experiments:
  - exp148
  - exp218
  - exp226
  - exp237
  - exp238
  - exp243
  - exp251
  - exp257
topics:
  - selector
  - candidate_path
  - feature_importance
  - confidence
  - lightgbm
status: final
summary: "exp238の候補bank、nested selector、415列TVTモデル、OOF安全性と特徴量重複を統合して説明する。"
---

# exp238 selector / TVT feature audit

- 対応する上位仮説: なし

作成日: 2026-07-16

## 結論

- ML予測をHMMのcenter / emissionに使う`exp221`、`exp234`、`exp240`はスタッキング寄りとして当面スコープ外にする。selector候補bankは11本をそのまま増やさず、`exp226 K16`、`blend_likpf_hmm_w500`、`self-GR HMM`、`likpf_mean`、`pf_ancc`をcore/fallbackにする。primitive bankへ切り替える場合だけw500をexact HMMへ分解し、親子を同時にselectableにしない。
- `beam_mean`は単体RMSE 15.774で直線的なpathが多いが、exp237 selectorが選んだ行ではlikPFより良い。一方、exp226との50/50は11.189112、cross-fitも9.432876でexp226単体より+0.005766だった。平均部品ではなく1本だけselector reserveに残し、Beam variantは追加しない。
- `likpf_mean`は単純算術平均ではなくseed likelihood-weighted mean。非平均化したexp243 K8 medoidはdirect replacementには弱いが、base8 unionのwhole-well oracleを6.5924から5.4996へ改善するため、優先する実験候補へ引き上げる。
- self-GRは既に検討済み。exp091/093のraw self-GR 5本はoracle headroomがあるがdirect pathと旧scorerは失敗。exp223の弱いSelf-GR likelihoodを加えたHMMは改善し、exp237の現行11候補bankに入っている。
- HMM+LGB除外後の全横断平均監査では、`exp226 + w500`固定50/50がRMSE 8.238331、exp226比-1.188778、5/5 fold改善。raw-test生成可能な`exp226 + likPF + exact HMM`のheld-out-well cross-fitは8.231651、5/5 fold改善だった。`last_anchor_tvt`はnear候補、exp103 `xy_likpf_scale_12`はselector diversity候補に留める。
- selectorはhard TVTを主出力にせず、outer-fold別の候補誤差分布とrank-slot confidenceを出す。direct pathはdiagnosticに限定する。
- TVT LightGBMは`TVT-last_known_tvt`残差を維持し、selectorはadd-onlyにする。exp257のreplacement-onlyはsame-fold exp238よりRMSE +0.164641で悪化した。
- exp238の415列には、exp148 lineageから継承したhigh-confidence重複17列が残る。さらにformation末尾重視12列は相関0.999990〜0.999993。最初はexact 17だけのdrop ablation、その後にformation 12を別ablationにする。
- 415列全体のtrain-row相関matrixは未保存。既存の数値相関は前半294列の600,000-row auditと、11 candidate pathの全3,783,989-row residual相関。GRWR 86列とnsel 35列は生成式上の従属関係まで監査したが、全組合せの実測相関は別のno-training readoutが必要。

## Evidence boundary

- exp238 final model: 380 base + 35 nested selector = 415 features、3 LightGBM configs × 5 folds。OOF `lgb_mean` 7.936690、Public LB 7.775。
- 現行11候補rankerの参照はexp237。raw-test-safe化した最新exp251の295列 expected-error fixed Viterbiは8.502212で、overall、1000+、worst-well guardが不通過。採用済みhard selectorはまだない。
- importanceはLightGBM `feature_importances_`のsplit回数でありgainではない。各configの総split数で正規化してから3 config平均した値で順位付けした。
- exp238のhistorical exp218との差はouter-fold assignmentが一致しないため、selector特徴だけの因果差ではない。

## Candidate paths

| candidate | 内容 | RMSE | unique-best | exp237 OOF選択率 | 推奨 |
| --- | --- | ---: | ---: | ---: | --- |
| `v6_k16_geometry_gr_u_projection` | exp226 K16 geometry/GR/U-projection | 9.427 | 14.73% | 17.12% | core keep |
| `blend_likpf_hmm_w500` | likPFとexact HMMの50/50平均 | 10.270 | 7.47% | 8.61% | scope内固定blendの主成分。primitive bankではexact HMMと二者択一 |
| `hmm_selfgr_boost_only_a070_c100` | self-GR HMM | 11.350 | 22.27% | 25.76% | core keep |
| `likpf_mean` | likelihood-weighted PF平均 | 11.595 | 12.76% | 13.47% | core fallback |
| `pf_ancc` | ANCC粒子フィルタ | 14.493 | 14.36% | 22.26% | core diversity |
| `beam_mean` | 複数Beam path平均 | 15.774 | 8.36% | 2.10% | reserve keep; do not add Beam variants |
| `tvt_dense50` | dense spatial ANCC（prefix末尾50 bias） | 19.995 | 5.73% | 3.18% | preferred late-bias dense path |
| `tvt_densew` | dense spatial ANCC（prefix加重bias） | 20.103 | 4.93% | 3.97% | family-exclusion audit; corr 0.99935 with dense50 |
| `tvt_dense` | dense spatial ANCC（full-prefix bias） | 23.470 | 6.63% | 2.79% | keep one broad dense path |
| `hyb` | Beam/NCC hybrid | 117.249 | 1.40% | 0.58% | exclude with sc_ens unless an outer-safe subgroup proves value |
| `sc_ens` | multi-scale NCC ensemble | 193.646 | 1.36% | 0.16% | exclude first; weak and redundant with hyb |

主な候補残差相関:

| left | right | residual correlation |
| --- | --- | ---: |
| `tvt_densew` | `tvt_dense50` | 0.999350 |
| `sc_ens` | `hyb` | 0.997180 |
| `tvt_dense` | `tvt_densew` | 0.899553 |
| `tvt_dense` | `tvt_dense50` | 0.894514 |
| `likpf_mean` | `blend_likpf_hmm_w500` | 0.867464 |
| `blend_likpf_hmm_w500` | `hmm_selfgr_boost_only_a070_c100` | 0.864206 |
| `pf_ancc` | `likpf_mean` | 0.628019 |
| `pf_ancc` | `blend_likpf_hmm_w500` | 0.582059 |
| `pf_ancc` | `beam_mean` | 0.567231 |
| `likpf_mean` | `hmm_selfgr_boost_only_a070_c100` | 0.553062 |

解釈:

- `tvt_densew` / `tvt_dense50`は0.999350で、同時にselectableにする価値が小さい。
- `sc_ens` / `hyb`は0.997180、Viterbi選択率もほぼ0なので最初のcandidate exclusion候補。
- `blend_likpf_hmm_w500`はlikPFとHMMの決定的50/50平均で、新しいgeneratorではない。ただしHMM+LGB除外後はexp226との50/50が8.238331となり、scope内最良のtarget-free固定平均である。fixed blendではw500を残し、primitive selectorではexact HMMへ分解する。

### 全候補パスの平均・凸結合監査

81 candidate pathsと13 selector/TVT model outputsを同じ3,783,989行へ揃え、全4,371ペア、24本shortlistの均等平均190,026組、上位20本の全1,140 tripleを監査した。詳細は[`candidate_path_blend_audit_20260716.md`](candidate_path_blend_audit_20260716.md)を正とする。

| 組合せ | 評価方法 | RMSE | best member差 | 改善fold | 判断 |
| --- | --- | ---: | ---: | ---: | --- |
| exp226 K16 + w500 | fixed 50/50 | **8.238331** | **-1.188778** | **5/5** | target-free固定の主候補 |
| exp226 + likPF + exact HMM | well cross-fit凸結合 | 8.231651 | -1.195458 | 5/5 | 固定案との差-0.00668。複雑化の便益小 |
| exp226 + self-GR HMM | fixed 50/50 | 8.532715 | -0.894395 | 5/5 | primitive 2本の最小案 |
| exp226 + self-GR HMM | well cross-fit凸結合 | 8.401770 | -1.025340 | 5/5 | 重みはexp226平均63.9% |
| exp226 + exp192 likPF + exact HMM | well cross-fit凸結合 | 8.209225 | -1.217885 | 5/5 | 数値最良だがexp192 train cache only |

fixed exp226+w500は475/773 wellsを改善し、全distance bucket、hidden-like 2群、5/5 foldsを改善した一方、worst-well回帰+15.396。cross-fit主案も489 wells改善 / 284悪化、worst +13.808なので、平均だけでwell safetyが成立したとは扱わない。

### `beam_mean`は候補に残すか

exp237 OOF 3,783,989行を再集計した。見た目の直線性だけでは除外せず、oracle contributionと実際にselectorが選んだ領域のcounterfactualを優先する。

| surface | Beam rows | rate | Beam RMSE on rows | likPF RMSE on same rows | Beam beats likPF | all-row RMSE | BeamをlikPFへ強制置換 | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle | 316,191 | 8.36% | 2.7123 | 11.4776 | 100.00% | 2.883510 | 4.32530 | +1.44180 |
| rowwise | 79,508 | 2.10% | 11.4543 | 15.8032 | 66.28% | 8.545228 | 8.68974 | +0.14452 |
| viterbi | 66,111 | 1.75% | 12.5730 | 17.6732 | 68.68% | 8.545093 | 8.70137 | +0.15628 |

`rowwise`のoracle-Beam recallは4.43%、Viterbiは3.95%に留まる一方、Beamを選んだ行のoracle precisionは17.62% / 18.87%。つまりBeam自体が不要なのではなく、Beamが勝つ領域の識別が弱い。強制likPF置換は、selector再学習や次善候補を使わないため削除損失の上限寄りの診断である。判断は**1本だけreserveに残す、Beam top-K/posterior variantは増やさない**。exp173/177のBeam top-K posterior・gap/entropy gateはnegativeでclosedのままとする。
Beam残差相関はself-GR HMM 0.4134、blend 0.4728、likPF 0.4876、exp226 0.5481、pfANCC 0.5672で、current coreに対して一定の多様性もある。ただしexp226との50/50は11.189112、cross-fitも9.432876でexp226単体を更新しなかった。よってreserve判断はselector用途だけに限定する。

### PFを単純平均しない候補

`likpf_mean`は500 particles × 128 seedsの各seed予測をseed likelihoodで重み付けした平均で、元から単純算術平均ではない。exp243はさらに平均前の実在seed trajectoryをcluster medoidとして保持した。

- 最良direct medoidはRMSE 12.296667で`likpf_mean` 11.594898より+0.701770。直接置換は不採用。
- base8 + K8 medoid oracleはrow 4.564605→3.216218、block128 4.805040→3.399936、whole-well 6.592426→5.499587。K8 unique-bestは43.88%、374/773 wellsを改善。
- K3/K5/K8全部を入れてもK8単独からwhole-well -0.006406だけなので、候補化するならK8だけ。
- exp252ではK8内の`cluster_likelihood_mass` / likelihood rank / gapがwhole-well AUC 0.6752 / 0.6551 / 0.6542。ただしbank gateは最良でも0.5606、固定top1はbest base8比+3.1949 ft。
- raw生成はexp243で約10時間18分/773 wells、hidden約200 wellsの単純比例は約2時間40分。したがってbase8 fallback付きのfold-safe selectorへ**高優先の実験候補**として追加するが、current coreやdirect pathにはまだしない。

### 過去実験から再検討する候補

| 候補 | 根拠 | 判断 |
| --- | --- | --- |
| `last_anchor_tvt` | RMSE 15.910だがexp093 oracle best 370,631行（9.79%）。raw-test-safeで非常に安い | **高**。まず現行pruned bankへのrow/block/well oracle追加量とnear選択を監査 |
| `recent_linear` / prefix末尾slope | exp019の0–49行で0.7966、50–249行で3.6155。exp238の`slp_b_d_50`は重要度2位 | **中**のnear専用expert。exp001 full OOF 41.022のため、global候補ではなく250 ft以内・fallback必須 |
| exp221 HMM+LGB | OOF 8.327728。全候補監査では強かったが、ML予測をHMMへ入れる構造 | **スタッキング寄りとして当面scope外**。比較履歴のみ保持 |
| exp082 fle3n final ensemble | Public LB 7.601でroute anchorだが、public/pretrained branchを含むfinal blend | aligned outer-fold OOFと候補固有confidence契約がないため、selector候補にはしない。LB anchorとしてのみ保持 |
| exp243 K8 PF medoids | row/block/wellすべて大きいoracle headroom、K8内confidence signalあり。全横断convex portfolioでは強い3本に対して重み0近傍 | selector候補としては保持。固定平均部品としては低優先、高コストかつbase8 fallback必須 |
| exp091/093 raw self-GR (`self_gr_sc8/15/25`, `best`, `ens`) | baseline oracle 7.4340→6.9589、within10 0.9065→0.9225。`sc25`はoracle best 175,030行 | **direct再追加はしない**。best単体250.162、ens 191.216、旧rankerはsc25/best/sc15を0行選択。Self-GRのscore/gap/qualityだけをconfidenceへ使う |
| exp223 Self-GR likelihood HMM | likPF 11.59490→11.34995。全距離/hidden-like改善だがworst well +46.95 | **現行core**。raw self-GR pathとは別物で、exp237の`hmm_selfgr_boost_only_a070_c100`として既に候補化済み |
| exp225 state-known TVT Self-GR HMM | RMSE 14.21295、likPF比+2.618、long-tail/hidden/worst-wellが悪化 | **再候補化しない**。state-known emission設計はnegative |
| exp103 `xy_likpf_scale_12` | 単体13.916。exp226との50/50 9.949でexp226を悪化、cross-fitも約9.12 | selector diversityに限定。固定平均部品にはしない |
| exp106 `pf_z_ms_scale_3` | 単体16.146。exp226との50/50は約10.95、cross-fit約9.31 | 低優先のselector候補。固定平均部品にはしない |
| exp142 trajectory-aware PF | global 23.132で不採用だが0–50/50–100/100–250 ftは0.551/1.267/2.630でlikPFより良い | **低**のnear-only。`last_anchor`/recent slopeの方が安く安全なので後順位 |
| exp202–215 heatmap topK/path | union oracleは大きいが生成path単体32–50 ft級、selector/feature follow-upも親を更新せず | ユーザー判断の**closed/rejectedを維持**。path候補として再開しない |

その他、exp128/134のSelf-GR hard switch/gate、Beam top-K（exp173/177）、MAP/dominant HMM（exp236）、adaptive/robust PF（exp232/233/241/242）、quantile/DTW/atlas HMM（exp229–231）、alt typewell path（exp187）はnegativeまたはclosed。exp129 spatial priorとexp176 typewell late-rangeはpath置換ではなくconfidence/context featureとしてのみpositiveで、current candidate pathにはしない。`exp218_centered_residual_diverse_hmm`もHMM+LGB scopeに含めて保留する。

## Selector features

推奨入力はcandidate-long形式にする。row contextとcandidate-specific featureを分離し、candidate indexを連続量として扱わない。

exp237は184列のcontext schemaからcandidate-long展開後320列を学習に使用した。split importance上位は次のとおり。`candidate_index`が2位なのは候補固有biasを拾う一方、任意の候補順序に閾値構造を入れるため、次版ではcategorical/one-hotへ置き換える。

| rank | exp237 selector feature | mean split importance |
| ---: | --- | ---: |
| 1 | `candidate_minus_last` | 1903.2 |
| 2 | `candidate_index` | 1155.4 |
| 3 | `v6_k16_geometry_gr_u_projection_minus_last` | 979.6 |
| 4 | `candidate_tvt` | 838.8 |
| 5 | `hmm_selfgr_boost_only_a070_c100_minus_last` | 774.6 |
| 6 | `copcf_spatial_xy_plus_trajectory_shape_k8_minus_candidate` | 704.2 |
| 7 | `tvt_densew_vs_tvt_dense50_abs` | 703.8 |
| 8 | `copcf_nearest_other_cluster_dist` | 691.4 |
| 9 | `hmm_exact_std` | 648.2 |
| 10 | `eval_len` | 566.4 |
| 11 | `copcf_own_cluster_dist` | 562.2 |
| 12 | `hmm_selfgr_boost_only_a070_c100_vs_v6_k16_geometry_gr_u_projection_abs` | 558.8 |

全320列の説明・順位・mean±std split importanceと、exp251 v4の全295列×2目的の重要度は[`selector_feature_catalog_20260716.md`](selector_feature_catalog_20260716.md)に分離した。exp237/251の共通282列におけるexpected-error重要度順位相関は0.952、exp251内のexpected-error/within10順位相関は0.981。高相関でも2目的のrowwise RMSE優劣はversion間で反転しており、同じ出力とは扱わない。

### 候補パスの信頼度を入力できるか

できる。HMMのσに相当する`hmm_exact_std`と`hmm_selfgr_std`はexp237/251ですでに入力済みで、split importanceも上位だった。
現状はrow contextとして全candidate行へ反復される。次版では、対応するHMM candidate行の`candidate_sigma_tvt`へ写し、`candidate_has_sigma`を付ける方がcandidate固有confidenceとして明確になる。元のglobal HMM stdも、他familyからHMMへ切り替える判断用contextとして残す。

| signal | exp237 importance rank | mean split importance | 証拠と扱い |
| --- | ---: | ---: | --- |
| `hmm_exact_std` | 9 | 648.2 | exp205でHMM absolute errorとの相関0.3995。粗いrisk signal |
| `hmm_selfgr_std` | 14 | 536.4 | self-GR HMMのposterior TVT幅 |
| `hmm_exact_loglik` | 17 | 487.2 | 行数で正規化して使う |
| `hmm_selfgr_loglik` | 27 | 370.2 | stdと別のfit quality |
| `pf_ancc_std` | 99 | 91.0 | PF粒子の行別spread。単独では弱い |
| `crfe_dense_candidate_std` | 65 | 177.4 | dense候補集合のspread |

ただしσをそのまま「小さいほど正しい」とは扱わない。exp221ではposterior std最低binのRMSEが8.986、中央binは7.66–7.80、最高binは9.997で非単調だった。exp223もlow-std bin RMSE 9.365を記録している。

#### 「候補別・outer-fold内で予測誤差へ校正」の正確な意味

ユーザーの理解どおり、**現行exp237/251ではσはselectorの入力特徴量の1つ**である。別のσ専用calibratorを直列に置いた、という意味ではない。selector LightGBMが`candidate identity + σ + loglik + 距離 + 候補間差 + その他context`から`|candidate_tvt-true_tvt|`または`P(error≤10)`を学ぶため、これは多変量モデル内での暗黙のcalibrationである。前の「校正する」という表現は別モデルを示すように読めて曖昧だった。

outer-fold内とは、outer-valid wellの正解をσ→error対応の学習に使わないというleakage guardを指す。outer fold fについて、outer-train側はinner OOF selector score、outer-valid側はouter-train wellsだけで学習したselector scoreを作る。exp238はouter 5 × inner 4 = 20モデルでこの契約を実装した。単変量の`σ→expected error`曲線を別途fitする案はdiagnosticにはなるが、σと誤差が非単調なので現時点の主案ではない。

candidate-longの共通confidence schemaは次を推奨する。family固有のraw値を保持し、該当しないfamilyは0埋めせず`confidence_valid`とmissing indicatorを付ける。

- 共通: `candidate_family` one-hot、`confidence_valid`、`confidence_source`、`sigma_tvt`、`loglik_per_row`、`score_margin`、`entropy`、`support_count`、`candidate_tvt-anchor`、candidate間distance/disagreement。
- HMM: posterior TVT std、GR observationの`hmm_prefix_sigma`、loglik/row、posterior entropy、top1/top2 mass gap、grid-edge mass、bimodal/mean-in-valley flag。固定hyperparameterのLGB emission σはconfidenceにしない。
- PF/K8: particle std、ESS fraction、resampling/collapse rate、seed prediction std、seed likelihood dispersion、cluster mass、likelihood mass/rank/gap、assignment distance、cluster entropy。
- Beam: retained top-K posterior/gap gateはexp173/177でnegativeなので再利用しない。保持する`beam_mean`にはlikPF/HMM/exp226とのdisagreement、local slope/curvature、直線度、anchorからのdriftだけを使う。
- dense/spatial/exp226: neighbor weighted std、distance、neighbor count、coverage/fallback、geometry gap、GR delta、donor agreement。
- 変換: `log1p(sigma)`、well内percentile、local rolling p50/p90/max、σの変化量、候補間σ比、`sigma × disagreement`。すべてouter-train fit、outer-valid applyにする。

候補ごとに同じ意味のσが必ず存在するわけではない。そのため、raw confidence proxyはfamily別に計算し、最終的な共通尺度をselectorの`pred_abs_error` / `p_within10`にする。

| candidate family | target-free raw confidence | 現状 |
| --- | --- | --- |
| `pf_ancc` | particle TVT std、ESS、resampling/collapse率、likelihood entropy/margin、multiobs score | `pf_ancc_std`とmultiobsは入力済み。ESS等は追加可能 |
| `likpf_mean` | seed予測のlikelihood加重std、seed weight entropy/max、effective seed数、PF ESS、multiobs | std/entropyの完全なcandidate固有化は未実装。再生成可能 |
| `beam_mean` | Beam間spread、直線度、local slope/curvature、boundary/clip率、他core候補とのdisagreement | disagreementは入力済み。exp173/177のtop-K posterior gateはnegativeなので主信頼度にしない |
| `sc_ens` / `hyb` | scale別NCC、1位−2位gap、scale間TVT spread、coverage/trust、Beamとのagreement | multiobs/NCCとSelf-GR系scoreを転用可。ただしpath自体は除外優先 |
| dense 3本 | neighbor weighted std、距離、count/coverage、prefix bias RMSE、3本のspread | `dense_std/dist`、dense pair差、CRFEは入力済み |
| exact / Self-GR HMM | posterior std、loglik/row、entropy、top-mass gap、grid-edge mass、prefix σ、Self-GR quality/valid | std/loglik/Self-GR qualityは入力済み。entropy/edge massは追加可能 |
| likPF-HMM blend | 両親のconfidence、両親のTVT差、blend位置、親ごとの予測誤差 | 固有σはない。親confidenceとdisagreementから作る |
| exp226 geometry | donor/neighbor support、KNN距離、geometry projection gap、GR delta、condition、family disagreement | gapは入力済み。`exp226_gr_delta/geop_tvt`はexp251 v4 raw-test契約外なのでparity実装後のみ復帰 |
| K8 PF medoids | cluster likelihood mass/rank/gap、assignment distance、cluster entropy、seed std、ESS | exp252でmass/rank/gapのwhole-well AUC 0.675/0.655/0.654。候補bank全体のgateは未成立 |
| `last_anchor` / recent slope | prefix slope fit残差、窓間slope分散、外挿距離、曲率/step、anchor quality | 安価に算出可能。near専用expertとして監査前 |
| xy-likPF / trajectory PF | seed std、XY slope fit残差/condition、PF stats、core候補との差 | 過去artifactから追加可能。K8とのfamily重複を先に監査 |
| raw self-GR path | NCC peak gap、scale間spread、match coverage、Self-GR/typewell agreement | exp091/093から算出可能。ただしraw pathは弱いのでconfidence featureだけを再利用 |
| exp221 HMM+LGB | HMM posterior診断、LGB fold/ensemble spread、quantile幅、HMM centerとの差 | 算出可能だがスタッキング寄りとしてcurrent scope外。比較履歴のみ |

- Row context: `md_since`、tail位置、`eval_len`、anchor geometry、GR missing/prefix coverage、DWT/FFT、candidate spread。
- Candidate-specific: `candidate_tvt-anchor`、family、上記confidence、各observation likelihood、exp226 geometry gap、candidate-vs-family disagreement。
- Safety: outer-train wellsだけで作るwell/segment risk、near-row flag、fallback候補、raw-test parity flag。
- 避けるもの: target由来gate、same-fold true error、ordinal `candidate_index` / `top1_code`の連続値扱い、全欠損列の黙示0補完。

## Selector output format

### exp238の35 rank-slotとexp251のdual outputはどちらがよいか

両者は同じ層ではないため、排他的に選ばない。**canonical selector出力はexp251形式**、**現行TVTモデルへ渡すadapterはexp238形式**がよい。

| 形式 | 情報 | 長所 | 短所 / evidence | 判断 |
| --- | --- | --- | --- | --- |
| exp238 35 rank-slot | 11候補の予測誤差、top1/top2候補値・誤差・margin、one-hot、spread | exp238 add-only OOF 7.936690、Public LB 7.775。TVTモデルがsoftに再利用できる | `p_within10`を持たず、一部summaryは決定的重複。historical exp218とはfold不一致 | **現行downstream adapterとして維持** |
| exp251 dual raw output | 各候補の`p_within10`と`pred_abs_error` | 情報を落とさず、rowwise/Viterbi/TVT adapterへ派生可能。確率校正も監査できる | v4 fixed Viterbi 8.502212でguard不通過。v2/v4でどちらの目的が良いか反転 | **canonical出力契約として採用、hard selectorは不採用** |
| exp257 replacement-only | selector出力を既存29 slotへ圧縮し、`nsel_*`を追加しない | schemaを380列に維持 | same-fold exp238比+0.164641悪化 | **不採用** |

数値も一方的ではない。exp251 v2ではprobability rowwise 8.682860、expected-error rowwise 8.464866、expected-error Viterbi 8.402086。v4では8.479603 / 8.548425 / 8.502212となり、probabilityとexpected-errorの優劣が反転した。よって片方を捨てず、各候補2 scoreを保存する。ただし現時点でexp251 scoreをexp238 TVTモデルへadd-onlyしたsame-fold比較はなく、exp238の35列を直ちに置換する根拠はない。

正規出力は2層にする。

1. canonical long artifact: `id, well, row, outer_fold, role, candidate, candidate_family, candidate_tvt, confidence_valid, pred_abs_error, p_within10`。`pred_sq_error`や`p_abs_gt25`は将来objectiveとして検証できた場合だけ追加する。raw confidence列は同じschema manifestに持つ。
2. downstream wide artifact: `pred_error__<candidate>`、`p_within10__<candidate>`と、top1/top2 delta、margin、ratio、score mean/std、candidate spread、top1 one-hotからなるfold別rank-slot features。現行exp238は予測誤差側35列だけなので、確率側はsame-fold add-only ablation後に採否を決める。

推論ではouter foldごとに4 inner selectorを平均し、そのouter fold用compact列を同じouter foldのTVT LightGBMへ渡す。model/schema/SHA manifest、candidate order、missing/nonfinite countsを必須にし、public-test row artifactは入力にしない。hard top1 TVTは監査列にだけ残す。exp255ではassertive bounded correctionがglobal -0.058700でもworst-well +3.151245でguard不通過だった。

### exp263 cacheを入力にするexp264 selector

旧7候補backlogはユーザー訂正を受けて廃止し、`exp264_exp263_candidate_confidence_dual_selector`として設計を確定した。候補source of truthはexp263であり、score対象はStage 1でcurrent-test生成可能な6 primitive、5 fixed 50/50 pair、固定`exp226_w500_50_50`の合計12 surfaceである。exp263の33 inventory、12 core primitive、virtual/namedを含む23 unique train/diagnostic surfaceは用途が違い、deployable score bankへ全て入れる意味ではない。

12候補は次のとおり。

- primitive: `exp226_k16`, `selfgr_hmm_a070`, `likpf_mean`, `exact_hmm`, `pf_ancc`, `beam_mean`
- pair: `exp226_k16__selfgr_hmm_a070`, `exp226_k16__exact_hmm`, `exp226_k16__likpf_mean`, `selfgr_hmm_a070__likpf_mean`, `likpf_mean__exact_hmm`
- fixed: `exp226_w500_50_50`

`blend_likpf_hmm_w500`は`likpf_mean__exact_hmm`のaliasなので別IDにしない。pairとnamed fixedを同一hard-selectable closureにしないexp263 guardを維持するため、12本はすべてscoreする一方、top1/top2/marginはprimitive+pair 11本domainとprimitive+fixed 7本domainで分ける。この12本はすべてexp263 Stage 1でcurrent-test生成済みである。Stage 0 OOFにはあるが現行Stage 1出力に未収録の追加6 primitiveとouter-fold fitted formula 2本は、Stage 1拡張とparity確認前には混ぜない。

2026-07-16のexp264 Stage A v1では600,000 candidate-long rowsを監査し、162特徴候補から100列を採用したが、このfeature contractは後日**無効化**した。採用100列にhidden/current testには存在しないtraining-only formation 6列（`ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA`）のraw値とlast-known差分、計12列が混入していた。outer-validにはこれらの実値があるため、well-group OOFでも本番のfeature availabilityを再現できない。全欠損41、定数5、完全重複16、高相関35組、schema SHA、74列compactという集計は無効実行の再現用にだけ残す。confidence 21列のexp263 Stage 1 parity自体は別契約なので、候補値とともにraw-test parityを再監査してから再利用する。

source-native confidenceは全候補に同じ1 scalarがあるわけではない。primitiveはexp263のsigma/loglik/entropy/margin/support/ESS/fallbackをNaN+validityで保持し、formulaは親別namespaced confidence、valid数、component range/stdを使う。全候補にavailability、shape、anchor距離、bank/formula disagreementのuniversal proxyを必ず持たせ、共通のlearned confidenceを`pred_abs_error`と`p_within10`にする。

特徴はexp251 v4の295 candidate-long列を分類seedとして参照し、最終allowlistとして無条件継承しない。`ctx/cand/conf/bank/formula/id`へ再編し、旧`sc_ens/hyb/tvt_dense*`固有列とordinal `candidate_index`を除去、candidate-relative特徴をexp263 bankへ再計算する。初回実装ではraw horizontal/typewellからtrain/current-test共通contextを生成し、`copcf_*`は同一cross-fit/current-test generatorを接続するまでdeferする。candidate IDは12 one-hot。constant/all-missing/exact duplicateをStage Aで除き、|Pearson|/|Spearman| 0.999以上は報告する。全特徴の説明、provenance、raw-test status、欠損率、objective×fold重要度をcatalog化する。

Stage Aは0 booster。Stage Bはstandard LightGBM固定で1 variant × 2 objectives × outer 5 = 10 CPU boosters。canonical audit出力は候補別dual scoreのcandidate-long Parquetだが、推論はscore CSVを保存・再読込せず同じprocess内でcompact化する。exp264ではViterbiを作らず、hard readoutはrowwise diagnosticだけにする。

2026-07-17にStage B v2を10 CPU boostersで完了したが、上記12列のfeature availability leakageによりexpected-error MAE、within10 logloss/Brier、5/5 fold改善、score guard、hard top1、hidden-like readout、feature importanceをすべて無効化した。これは狭義のtarget leakageではなく、validation時だけ利用できる入力を使ったtrain-test schema leakageである。候補別scoreをcompactへ残す判断も撤回する。

同日のStage C v3も40 CPU boosters、40 models、25 partitions、18,919,945 rowsの74列compactを生成したところまでは実行履歴だが、score、guard、nested leakage PASS、hard top1、confidence重要度、`sigma_tvt`重要度を含む全性能readoutを無効化した。inner/outer分離が正しくても、両側にhiddenでは得られない12列が存在するため修復されない。

TVT add-onlyは無効なStage C compactを入力にしたため、Stage Dも無効である。2 variants × 3 configs × 5 folds = 30 GPU boostersを完走した事実だけを履歴として残し、matched control `8.545568`、add-only `7.805644`、fold/bucket/hidden-like/worst-well比較、特徴重要度を性能・診断・negative resultのいずれにも使用しない。matched control自体もexp218 raw-test feature contractを独立に再監査するまで比較anchorにしない。compact inference、hard selector、Viterbi、submissionへ進まない。詳細の正は[`exp264 candidate contract`](../../experiments/exp264_exp263_candidate_confidence_dual_selector/candidate_contract.yaml)、[`feature contract`](../../experiments/exp264_exp263_candidate_confidence_dual_selector/feature_contract.yaml)、[`output contract`](../../experiments/exp264_exp263_candidate_confidence_dual_selector/output_contract.md)、[`Stage D importance readout`](../../experiments/exp264_exp263_candidate_confidence_dual_selector/stage_d_feature_importance_readout.md)である。

## TVT prediction model features

推奨順序:

1. exp238のanchor residual targetとadd-only selector構成をbaselineに固定。
2. high-confidence exact duplicate 17列だけをdropしたsame-fold ablation。exp198ではexp148 CVを-0.043358改善したがPublic LB 7.930で、exp238への転移は未検証。
3. formation `bw50/tvtF50` 12列を別ablationでdrop。weighted側を残す。
4. nselはreplacementしない。まずordinal codeとdeterministic summariesを相関監査し、slim化は別ablation。
5. GRWRはall-zero列とfilter/window間高相関をno-training auditしてからgroup単位で削る。importanceだけで一括削除しない。

## Importance family summary

| family | features | normalized split share |
| --- | ---: | ---: |
| `base_replay` | 196 | 56.961% |
| `u_projection` | 44 | 17.089% |
| `nested_selector` | 35 | 9.573% |
| `gr_wavelet_rotation` | 86 | 9.367% |
| `learned_likelihood` | 54 | 7.010% |

### Top 40

| rank | feature | family | share | mean rank | lgb0/lgb1/lgb2 split | 説明 |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 1 | `nsel_top1_minus_anchor` | `nested_selector` | 1.6228% | 1.7 | 3547.6/2209.6/2358.0 | selector top1候補TVT − last_known_tvt。 |
| 2 | `slp_b_d_50` | `base_replay` | 1.5241% | 2.3 | 3167.8/2276.4/2145.0 | 末尾50行 slope外挿TVT − anchor。 |
| 3 | `grwr_fft_rotation_ratio_x_log1p_md_since` | `gr_wavelet_rotation` | 1.5002% | 2.7 | 3027.0/2245.2/2176.4 | FFT rotation-band energy比 × log1p(md_since)。 |
| 4 | `dz` | `base_replay` | 1.4109% | 4.7 | 2859.6/2111.2/2037.2 | 予測行Z − anchor行Z。 |
| 5 | `dx` | `base_replay` | 1.3364% | 6.3 | 2757.4/2031.0/1862.6 | 予測行X − anchor行X。 |
| 6 | `dy` | `base_replay` | 1.3252% | 7.0 | 2682.2/2022.6/1878.2 | 予測行Y − anchor行Y。 |
| 7 | `frac` | `base_replay` | 1.3182% | 6.3 | 2976.4/1847.2/1792.8 | 予測tail内の0〜1正規化行位置。 |
| 8 | `spatial_knn_dist` | `base_replay` | 1.2981% | 8.3 | 3352.2/1634.4/1623.6 | formation spatial KNNで使う最短正規化XY距離。 |
| 9 | `nsel_top2_minus_anchor` | `nested_selector` | 1.2707% | 8.0 | 2823.0/1768.6/1775.0 | selector top2候補TVT − last_known_tvt。 |
| 10 | `dense_dist` | `base_replay` | 1.2245% | 9.3 | 2587.0/1813.8/1706.2 | dense spatial KNNの最短正規化XY距離。 |
| 11 | `dense_std` | `base_replay` | 1.1899% | 11.3 | 2713.2/1609.2/1654.4 | dense spatial KNN近傍ANCCの重み付き標準偏差。 |
| 12 | `slp_b_d_all` | `base_replay` | 1.1080% | 14.7 | 2478.6/1578.6/1499.4 | 全prefix slope外挿TVT − anchor。 |
| 13 | `uproj_likpf_mean_resid_mad` | `u_projection` | 1.0117% | 14.7 | 1812.2/1632.0/1528.0 | likpf_mean U-projection: 上記|residual|のwell内中央値。 |
| 14 | `dense_nb_std` | `base_replay` | 1.0045% | 14.0 | 1835.8/1657.0/1454.2 | 既知prefixのdense KNN近傍標準偏差平均（well定数）。 |
| 15 | `grwr_gr_missing_rate` | `gr_wavelet_rotation` | 0.9959% | 16.0 | 1773.0/1638.0/1482.2 | 水平井GRのwell内欠損率。 |
| 16 | `uproj_pf_ancc_resid_mad` | `u_projection` | 0.9847% | 17.3 | 1683.6/1581.6/1555.0 | pf_ancc U-projection: 上記|residual|のwell内中央値。 |
| 17 | `form_mean_d` | `base_replay` | 0.9832% | 18.7 | 2027.2/1442.4/1421.2 | 6 formation由来TVT候補の平均 − anchor。 |
| 18 | `slp_50` | `base_replay` | 0.9650% | 18.7 | 1593.8/1613.0/1505.6 | 既知prefix末尾50行のTVT/MD robust slope。 |
| 19 | `grwr_fft_dominant_frequency_norm` | `gr_wavelet_rotation` | 0.9452% | 20.7 | 1645.4/1509.2/1478.6 | detrend後GR FFTのdominant frequency（正規化）。 |
| 20 | `form_rng_d` | `base_replay` | 0.9347% | 22.7 | 1767.6/1503.4/1345.0 | 6 formation由来TVT候補の最大−最小。 |
| 21 | `form_std_d` | `base_replay` | 0.9337% | 22.0 | 1738.4/1442.4/1421.6 | 6 formation由来TVT候補の標準偏差。 |
| 22 | `slp_z` | `base_replay` | 0.9184% | 23.0 | 1563.0/1527.8/1404.8 | 既知prefixのTVT/Z robust slope。 |
| 23 | `tvt_dense_d` | `base_replay` | 0.9115% | 25.3 | 1814.2/1389.8/1316.4 | dense ANCC + full-prefix bias TVT候補 − anchor。 |
| 24 | `tvt_dense50_d` | `base_replay` | 0.9033% | 25.7 | 1637.0/1417.0/1388.4 | dense ANCC + prefix末尾50行bias TVT候補 − anchor。 |
| 25 | `pfx_rmse` | `base_replay` | 0.8976% | 25.3 | 1549.2/1497.4/1352.6 | 既知prefix GRとTVT対応typewell GRのRMSE。 |
| 26 | `beam_vcons_d` | `base_replay` | 0.8960% | 29.0 | 1809.8/1357.4/1282.2 | very-conservative Beam候補 − last_known_tvt。 |
| 27 | `cal_a` | `base_replay` | 0.8917% | 26.3 | 1537.2/1440.4/1390.4 | 既知prefix GRをtypewell GRへaffine fitした傾き。 |
| 28 | `uproj_pf_z_resid_mad` | `u_projection` | 0.8735% | 29.7 | 1445.8/1471.2/1349.4 | pf_z U-projection: 上記|residual|のwell内中央値。 |
| 29 | `z` | `base_replay` | 0.8634% | 30.7 | 1461.6/1435.0/1327.8 | 予測行のZ座標。 |
| 30 | `nsel_pred_error_hmm_selfgr_boost_only_a070_c100_mean_tvt` | `nested_selector` | 0.8549% | 31.3 | 1513.0/1396.0/1288.6 | nested selectorが予測したself-GR HMMの絶対誤差。 |
| 31 | `beam_vloose_d` | `base_replay` | 0.8467% | 36.0 | 1808.4/1242.2/1176.2 | very-loose Beam候補 − last_known_tvt。 |
| 32 | `dense_ancc` | `base_replay` | 0.8454% | 34.3 | 1443.8/1409.2/1286.4 | dense spatial KNNで補完したANCC surface値。 |
| 33 | `eval_len` | `base_replay` | 0.8452% | 35.0 | 1449.8/1348.2/1340.2 | 予測tailの行数。 |
| 34 | `spatial_vs_dense` | `base_replay` | 0.8438% | 35.0 | 1821.4/1216.4/1178.4 | ANCC formation-spatial候補 − dense ANCC候補。 |
| 35 | `grwr_fft_high_frequency_ratio` | `gr_wavelet_rotation` | 0.8324% | 38.0 | 1443.4/1360.0/1276.8 | 正規化周波数0.35超のGR FFT energy比率。 |
| 36 | `tw_gr_mean` | `base_replay` | 0.8315% | 37.0 | 1489.2/1349.2/1248.4 | typewell GR平均（well定数）。 |
| 37 | `grwr_fft_dominant_energy_ratio` | `gr_wavelet_rotation` | 0.8235% | 39.7 | 1378.2/1369.6/1277.8 | detrend後GR FFTの最大周波数bin energy比率。 |
| 38 | `beam_vs_spatial` | `base_replay` | 0.8178% | 38.0 | 1660.0/1224.8/1177.8 | conservative Beam候補 − ANCC formation-spatial候補。 |
| 39 | `cal_b` | `base_replay` | 0.8149% | 40.7 | 1350.8/1351.4/1277.8 | 既知prefix GRをtypewell GRへaffine fitした切片。 |
| 40 | `uproj_beam_mean_resid_mad` | `u_projection` | 0.8140% | 41.0 | 1335.6/1379.8/1258.0 | beam_mean U-projection: 上記|residual|のwell内中央値。 |

14列は3 configすべてsplit importance 0: `gr_d1`, `gr_d2`, `grwr_dwt_approx_best_is_default_candidate`, `grwr_dwt_approx_minus_raw_default_candidate_cost`, `grwr_dwt_approx_minus_raw_default_candidate_ncc`, `grwr_dwt_minus_raw_ncc_gap_x_candidate_range`, `grwr_dwt_minus_raw_ncc_gap_x_dwt_energy_ratio_w065`, `grwr_raw_best_is_default_candidate`, `grwr_rolling_median_11_best_is_default_candidate`, `grwr_savgol_31_p2_best_is_default_candidate`, `grwr_typewell_gr_missing_rate`, `ll_candidate_tvt_likpf_mean_minus_likpf_mean_tvt`, `ll_multiobs_ncc_hyb`, `sc_trust`。

## Duplicate and correlation audit

### High-confidence exact / functional duplicates inherited into exp238

| drop candidate | keep | relation | correlation |
| --- | --- | --- | ---: |
| `sc_trust` | - | `constant` | constant |
| `ll_candidate_tvt_likpf_mean_minus_likpf_mean_tvt` | - | `constant_zero` | constant |
| `ll_candidate_tvt_beam_mean_minus_last_known_tvt` | `beam_mean_d` | `existing_delta_duplicate` | 0.9999999962819376 |
| `ll_candidate_tvt_beam_mean_minus_likpf_mean_tvt` | `uproj_diff_beam_mean_minus_likpf_mean` | `existing_disagreement_duplicate` | 1.0 |
| `ll_candidate_tvt_hyb_minus_last_known_tvt` | `hyb_d` | `existing_delta_duplicate` | 1.0 |
| `ll_candidate_tvt_likpf_mean_minus_last_known_tvt` | `likpf_mean_d` | `existing_delta_duplicate` | 1.0 |
| `ll_candidate_tvt_pf_ancc_minus_last_known_tvt` | `pf_ancc_delta` | `existing_delta_duplicate` | 1.0 |
| `ll_candidate_tvt_pf_ancc_minus_likpf_mean_tvt` | `uproj_diff_pf_ancc_minus_likpf_mean` | `existing_disagreement_duplicate` | 1.0 |
| `ll_candidate_tvt_sc_ens_minus_last_known_tvt` | `sc_ens_d` | `existing_delta_duplicate` | 1.0 |
| `tda0` | `gr_vs_tw_anc` | `near_exact_public_replay_duplicate` | 0.9999999707804362 |
| `dense_bias` | `dense_rmse` | `near_exact_public_replay_duplicate` | 0.9999999994694734 |
| `uproj_beam_mean_resid` | `uproj_beam_mean_corr` | `sign_flip_duplicate` | -1.0 |
| `uproj_beam_med_resid` | `uproj_beam_med_corr` | `sign_flip_duplicate` | -1.0 |
| `uproj_diff_pf_ancc_minus_pf_z` | `pf_vs_z` | `existing_disagreement_duplicate` | 1.0 |
| `uproj_likpf_mean_resid` | `uproj_likpf_mean_corr` | `sign_flip_duplicate` | -1.0 |
| `uproj_pf_ancc_resid` | `uproj_pf_ancc_corr` | `sign_flip_duplicate` | -1.0 |
| `uproj_pf_z_resid` | `uproj_pf_z_corr` | `sign_flip_duplicate` | -1.0 |

### Near duplicates / high correlations from the 600k-row exp148-lineage audit

- formation weighted vs last50 12 pairs: |r| 0.999990〜0.999993。
- `md_since` / `dxy`: 0.999997950。意味は異なるため即dropではない。
- `gr_nrg` / `grm21`: 0.999921773。
- `tvt_dense50_d` / `tvt_densew_d`: 0.999539184。
- `form_rng_d` / `form_std_d`: 0.996395546。
- `ll_candidate_tvt_std` / `ll_candidate_tvt_range`: 0.999463033。
- `ll_candidate_tvt_sc_ens_minus_likpf_mean_tvt` / `ll_candidate_tvt_hyb_minus_likpf_mean_tvt`: 0.997646402。
- `uproj_beam_mean_resid_mad` / `uproj_beam_med_resid_mad`: 0.979712902。
- `uproj_source_u_std` / `uproj_source_u_range`: 0.986839609。
- `uproj_corr_std` / `uproj_corr_range`: 0.985400052。

nselの生成式上の従属関係:

- `top2_minus_top1 = top2_minus_anchor - top1_minus_anchor`。
- `error_margin = error_top2 - error_top1`、`error_ratio = error_top1 / max(error_top2,1e-3)`。
- 11個の`top1_is_*`の和は1、`top1_code`はone-hotの線形結合。
- `score_mean/std`は11個の`pred_error_*`から、`candidate_std/range`は11 candidate TVTから決定される。

これらは木モデルで即バグではないが、importanceを分散し、ordinal codeに不自然な閾値順序を与える。slim化は`top1/top2 delta + per-candidate error + one-hot`をcoreとし、code/派生summaryを別ablationで落とすのが安全。

## All 415 features in normalized importance order

| rank | feature | family | share | mean rank | lgb0/lgb1/lgb2 split | duplicate note | 説明 |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | `nsel_top1_minus_anchor` | `nested_selector` | 1.6228% | 1.7 | 3547.6/2209.6/2358.0 |  | selector top1候補TVT − last_known_tvt。 |
| 2 | `slp_b_d_50` | `base_replay` | 1.5241% | 2.3 | 3167.8/2276.4/2145.0 |  | 末尾50行 slope外挿TVT − anchor。 |
| 3 | `grwr_fft_rotation_ratio_x_log1p_md_since` | `gr_wavelet_rotation` | 1.5002% | 2.7 | 3027.0/2245.2/2176.4 |  | FFT rotation-band energy比 × log1p(md_since)。 |
| 4 | `dz` | `base_replay` | 1.4109% | 4.7 | 2859.6/2111.2/2037.2 |  | 予測行Z − anchor行Z。 |
| 5 | `dx` | `base_replay` | 1.3364% | 6.3 | 2757.4/2031.0/1862.6 |  | 予測行X − anchor行X。 |
| 6 | `dy` | `base_replay` | 1.3252% | 7.0 | 2682.2/2022.6/1878.2 |  | 予測行Y − anchor行Y。 |
| 7 | `frac` | `base_replay` | 1.3182% | 6.3 | 2976.4/1847.2/1792.8 |  | 予測tail内の0〜1正規化行位置。 |
| 8 | `spatial_knn_dist` | `base_replay` | 1.2981% | 8.3 | 3352.2/1634.4/1623.6 |  | formation spatial KNNで使う最短正規化XY距離。 |
| 9 | `nsel_top2_minus_anchor` | `nested_selector` | 1.2707% | 8.0 | 2823.0/1768.6/1775.0 |  | selector top2候補TVT − last_known_tvt。 |
| 10 | `dense_dist` | `base_replay` | 1.2245% | 9.3 | 2587.0/1813.8/1706.2 |  | dense spatial KNNの最短正規化XY距離。 |
| 11 | `dense_std` | `base_replay` | 1.1899% | 11.3 | 2713.2/1609.2/1654.4 |  | dense spatial KNN近傍ANCCの重み付き標準偏差。 |
| 12 | `slp_b_d_all` | `base_replay` | 1.1080% | 14.7 | 2478.6/1578.6/1499.4 |  | 全prefix slope外挿TVT − anchor。 |
| 13 | `uproj_likpf_mean_resid_mad` | `u_projection` | 1.0117% | 14.7 | 1812.2/1632.0/1528.0 | u_projection_family_slim_review | likpf_mean U-projection: 上記|residual|のwell内中央値。 |
| 14 | `dense_nb_std` | `base_replay` | 1.0045% | 14.0 | 1835.8/1657.0/1454.2 |  | 既知prefixのdense KNN近傍標準偏差平均（well定数）。 |
| 15 | `grwr_gr_missing_rate` | `gr_wavelet_rotation` | 0.9959% | 16.0 | 1773.0/1638.0/1482.2 |  | 水平井GRのwell内欠損率。 |
| 16 | `uproj_pf_ancc_resid_mad` | `u_projection` | 0.9847% | 17.3 | 1683.6/1581.6/1555.0 | u_projection_family_slim_review | pf_ancc U-projection: 上記|residual|のwell内中央値。 |
| 17 | `form_mean_d` | `base_replay` | 0.9832% | 18.7 | 2027.2/1442.4/1421.2 |  | 6 formation由来TVT候補の平均 − anchor。 |
| 18 | `slp_50` | `base_replay` | 0.9650% | 18.7 | 1593.8/1613.0/1505.6 |  | 既知prefix末尾50行のTVT/MD robust slope。 |
| 19 | `grwr_fft_dominant_frequency_norm` | `gr_wavelet_rotation` | 0.9452% | 20.7 | 1645.4/1509.2/1478.6 |  | detrend後GR FFTのdominant frequency（正規化）。 |
| 20 | `form_rng_d` | `base_replay` | 0.9347% | 22.7 | 1767.6/1503.4/1345.0 |  | 6 formation由来TVT候補の最大−最小。 |
| 21 | `form_std_d` | `base_replay` | 0.9337% | 22.0 | 1738.4/1442.4/1421.6 |  | 6 formation由来TVT候補の標準偏差。 |
| 22 | `slp_z` | `base_replay` | 0.9184% | 23.0 | 1563.0/1527.8/1404.8 |  | 既知prefixのTVT/Z robust slope。 |
| 23 | `tvt_dense_d` | `base_replay` | 0.9115% | 25.3 | 1814.2/1389.8/1316.4 |  | dense ANCC + full-prefix bias TVT候補 − anchor。 |
| 24 | `tvt_dense50_d` | `base_replay` | 0.9033% | 25.7 | 1637.0/1417.0/1388.4 |  | dense ANCC + prefix末尾50行bias TVT候補 − anchor。 |
| 25 | `pfx_rmse` | `base_replay` | 0.8976% | 25.3 | 1549.2/1497.4/1352.6 |  | 既知prefix GRとTVT対応typewell GRのRMSE。 |
| 26 | `beam_vcons_d` | `base_replay` | 0.8960% | 29.0 | 1809.8/1357.4/1282.2 |  | very-conservative Beam候補 − last_known_tvt。 |
| 27 | `cal_a` | `base_replay` | 0.8917% | 26.3 | 1537.2/1440.4/1390.4 |  | 既知prefix GRをtypewell GRへaffine fitした傾き。 |
| 28 | `uproj_pf_z_resid_mad` | `u_projection` | 0.8735% | 29.7 | 1445.8/1471.2/1349.4 | u_projection_family_slim_review | pf_z U-projection: 上記|residual|のwell内中央値。 |
| 29 | `z` | `base_replay` | 0.8634% | 30.7 | 1461.6/1435.0/1327.8 |  | 予測行のZ座標。 |
| 30 | `nsel_pred_error_hmm_selfgr_boost_only_a070_c100_mean_tvt` | `nested_selector` | 0.8549% | 31.3 | 1513.0/1396.0/1288.6 |  | nested selectorが予測したself-GR HMMの絶対誤差。 |
| 31 | `beam_vloose_d` | `base_replay` | 0.8467% | 36.0 | 1808.4/1242.2/1176.2 |  | very-loose Beam候補 − last_known_tvt。 |
| 32 | `dense_ancc` | `base_replay` | 0.8454% | 34.3 | 1443.8/1409.2/1286.4 |  | dense spatial KNNで補完したANCC surface値。 |
| 33 | `eval_len` | `base_replay` | 0.8452% | 35.0 | 1449.8/1348.2/1340.2 |  | 予測tailの行数。 |
| 34 | `spatial_vs_dense` | `base_replay` | 0.8438% | 35.0 | 1821.4/1216.4/1178.4 |  | ANCC formation-spatial候補 − dense ANCC候補。 |
| 35 | `grwr_fft_high_frequency_ratio` | `gr_wavelet_rotation` | 0.8324% | 38.0 | 1443.4/1360.0/1276.8 |  | 正規化周波数0.35超のGR FFT energy比率。 |
| 36 | `tw_gr_mean` | `base_replay` | 0.8315% | 37.0 | 1489.2/1349.2/1248.4 |  | typewell GR平均（well定数）。 |
| 37 | `grwr_fft_dominant_energy_ratio` | `gr_wavelet_rotation` | 0.8235% | 39.7 | 1378.2/1369.6/1277.8 |  | detrend後GR FFTの最大周波数bin energy比率。 |
| 38 | `beam_vs_spatial` | `base_replay` | 0.8178% | 38.0 | 1660.0/1224.8/1177.8 |  | conservative Beam候補 − ANCC formation-spatial候補。 |
| 39 | `cal_b` | `base_replay` | 0.8149% | 40.7 | 1350.8/1351.4/1277.8 |  | 既知prefix GRをtypewell GRへaffine fitした切片。 |
| 40 | `uproj_beam_mean_resid_mad` | `u_projection` | 0.8140% | 41.0 | 1335.6/1379.8/1258.0 | u_projection_family_slim_review | beam_mean U-projection: 上記|residual|のwell内中央値。 |
| 41 | `spatial_ancc_d` | `base_replay` | 0.8107% | 39.3 | 1452.6/1305.4/1226.2 |  | spatial KNN ANCC surface − anchor位置のtypewell GR値。 |
| 42 | `grwr_fft_rotation_energy_ratio` | `gr_wavelet_rotation` | 0.8057% | 41.7 | 1236.0/1377.4/1299.4 |  | 正規化周波数0.06〜0.35のrotation-band energy比率。 |
| 43 | `tvt_densew_d` | `base_replay` | 0.7770% | 45.7 | 1386.6/1205.2/1224.2 |  | dense ANCC + prefix指数加重bias TVT候補 − anchor。 |
| 44 | `md_since` | `base_replay` | 0.7663% | 44.3 | 1477.4/1144.0/1166.6 |  | anchor行からのMD距離。 |
| 45 | `pf_vs_spatial` | `base_replay` | 0.7643% | 43.3 | 1580.6/1146.0/1077.6 |  | ANCC PF候補 − ANCC formation-spatial候補。 |
| 46 | `known_len` | `base_replay` | 0.7565% | 49.0 | 1278.2/1272.4/1150.6 |  | 既知TVT_input prefixの行数。 |
| 47 | `uproj_beam_med_resid_mad` | `u_projection` | 0.7563% | 49.0 | 1172.8/1273.4/1228.8 | u_projection_family_slim_review | beam_med U-projection: 上記|residual|のwell内中央値。 |
| 48 | `grwr_known_prefix_fraction` | `gr_wavelet_rotation` | 0.7445% | 52.0 | 1229.6/1240.2/1165.6 |  | 全horizontal行に占める既知prefix比率。 |
| 49 | `grwr_fft_notch_residual_energy_ratio` | `gr_wavelet_rotation` | 0.7430% | 51.0 | 1145.0/1268.8/1195.8 |  | 上位3周波数を除いたGR FFT residual energy比率。 |
| 50 | `dxy` | `base_replay` | 0.7281% | 51.3 | 1341.2/1126.0/1118.2 |  | anchorからのXY平面距離。 |
| 51 | `beam_cons_d` | `base_replay` | 0.7210% | 51.0 | 1422.8/1117.2/1033.4 |  | conservative Beam候補 − last_known_tvt。 |
| 52 | `ll_learned_pred_abs_error_beam_mean` | `learned_likelihood` | 0.7207% | 49.3 | 1527.6/1041.0/1025.8 |  | exp111 L1 modelが推定した複数Beam path平均の絶対誤差。 |
| 53 | `beam_stiff_d` | `base_replay` | 0.7205% | 51.7 | 1382.0/1121.6/1057.8 |  | stiff Beam候補 − last_known_tvt。 |
| 54 | `dxdmd` | `base_replay` | 0.7173% | 48.7 | 1490.0/1071.2/1010.4 |  | 行差分 dX/dMD。 |
| 55 | `slp_all` | `base_replay` | 0.7054% | 55.0 | 1224.4/1153.0/1080.6 |  | 既知prefix全体のTVT/MD robust slope。 |
| 56 | `nsel_pred_error_exp226_v6_k16_geometry_gr_u_projection` | `nested_selector` | 0.6931% | 54.3 | 1408.6/1048.4/987.0 |  | nested selectorが予測したexp226 K16 geometry/GR/U-projectionの絶対誤差。 |
| 57 | `pf_vs_dense` | `base_replay` | 0.6828% | 56.7 | 1349.0/1058.6/977.0 |  | ANCC PF候補 − dense ANCC候補。 |
| 58 | `beam_loose_d` | `base_replay` | 0.6710% | 57.3 | 1391.6/990.6/958.0 |  | loose Beam候補 − last_known_tvt。 |
| 59 | `beam_mid_d` | `base_replay` | 0.6708% | 58.0 | 1277.0/1059.6/977.6 |  | middle Beam候補 − last_known_tvt。 |
| 60 | `nsel_pred_error_beam_mean` | `nested_selector` | 0.6572% | 58.3 | 1417.0/896.6/968.4 |  | nested selectorが予測した複数Beam path平均の絶対誤差。 |
| 61 | `beam_std_d` | `base_replay` | 0.6506% | 57.3 | 1447.6/912.4/900.2 |  | 7種類のBeam候補deltaの行別標準偏差。 |
| 62 | `beam_sm5_d` | `base_replay` | 0.6395% | 62.7 | 1254.4/982.6/930.2 |  | smoothed Beam (r=5)候補 − last_known_tvt。 |
| 63 | `ll_learned_prob_likpf_mean` | `learned_likelihood` | 0.6306% | 63.3 | 1022.4/1045.8/1006.2 |  | exp111 classifierが推定したlikelihood-weighted PF平均のP(|error|≤10ft)。 |
| 64 | `pf_z_delta` | `base_replay` | 0.6094% | 65.7 | 1264.8/891.6/876.8 |  | Z-aware PF候補 − last_known_tvt。 |
| 65 | `beam_mean_d` | `base_replay` | 0.6030% | 65.7 | 1279.6/869.0/859.4 |  | 7種類のBeam候補deltaの行別平均。 |
| 66 | `frac2` | `base_replay` | 0.6028% | 72.7 | 627.6/1117.6/1114.0 |  | fracの二乗。 |
| 67 | `likpf_mean_d` | `base_replay` | 0.5930% | 66.7 | 1308.8/812.8/847.0 |  | likelihood-weighted PF平均候補 − anchor。 |
| 68 | `nsel_pred_error_blend_likpf_hmm_w500` | `nested_selector` | 0.5927% | 67.0 | 968.4/962.0/960.6 |  | nested selectorが予測したlikPFとexact HMMの50/50平均の絶対誤差。 |
| 69 | `uproj_diff_pf_z_minus_likpf_mean` | `u_projection` | 0.5804% | 69.7 | 1231.0/831.0/832.8 |  | U-spaceのpf_z − likpf_mean。Z/anchor共通項は相殺。 |
| 70 | `uproj_diff_beam_mean_minus_beam_med` | `u_projection` | 0.5709% | 70.3 | 1266.8/812.4/781.6 |  | U-spaceのbeam_mean − beam_med。Z/anchor共通項は相殺。 |
| 71 | `dydmd` | `base_replay` | 0.5607% | 72.0 | 1073.6/876.8/820.8 |  | 行差分 dY/dMD。 |
| 72 | `tw_range` | `base_replay` | 0.5558% | 69.7 | 879.2/963.4/863.4 |  | typewell TVT軸のrange（well定数）。 |
| 73 | `pf_vs_z` | `base_replay` | 0.5284% | 75.3 | 1177.6/743.8/727.0 |  | ANCC PF候補 − Z-aware PF候補。 |
| 74 | `frm_rmse_ANCC` | `base_replay` | 0.5247% | 76.0 | 811.0/898.4/840.2 |  | ANCC formation surface TVT式の既知prefix RMSE。 |
| 75 | `ktvt_std` | `base_replay` | 0.5164% | 78.0 | 686.6/920.0/877.4 |  | 既知prefix TVT_inputの標準偏差。 |
| 76 | `ll_learned_prob_top1_value` | `learned_likelihood` | 0.5039% | 78.0 | 839.4/792.2/829.0 |  | exp111 within10確率面のtop-k、rank、margin、entropyまたは確率加重TVT要約。 |
| 77 | `ktvt_range` | `base_replay` | 0.4908% | 82.0 | 736.8/837.6/805.4 |  | 既知prefix TVT_inputのrange。 |
| 78 | `uproj_diff_pf_ancc_minus_likpf_mean` | `u_projection` | 0.4894% | 83.7 | 1115.0/694.6/649.6 |  | U-spaceのpf_ancc − likpf_mean。Z/anchor共通項は相殺。 |
| 79 | `pf_ancc_delta` | `base_replay` | 0.4869% | 82.3 | 1063.6/689.8/682.0 |  | ANCC PF候補 − last_known_tvt。 |
| 80 | `uproj_likpf_mean_corr` | `u_projection` | 0.4685% | 81.0 | 871.2/719.0/718.6 |  | likpf_mean U-projection: well内robust polynomial U-trend − source U。 |
| 81 | `uproj_diff_pf_z_minus_beam_mean` | `u_projection` | 0.4657% | 82.0 | 850.4/702.6/737.8 |  | U-spaceのpf_z − beam_mean。Z/anchor共通項は相殺。 |
| 82 | `frm_rmse_ASTNU` | `base_replay` | 0.4632% | 85.3 | 698.0/776.2/771.8 |  | ASTNU formation surface TVT式の既知prefix RMSE。 |
| 83 | `nsel_top2_minus_top1` | `nested_selector` | 0.4608% | 99.0 | 1824.2/325.2/340.4 |  | selector top2候補TVT − top1候補TVT。 |
| 84 | `uproj_likpf_mean_resid` | `u_projection` | 0.4587% | 83.7 | 840.8/741.0/677.2 | sign_flip_duplicate → keep `uproj_likpf_mean_corr` (r=-1.0) | likpf_mean U-projection: source U − well内robust polynomial U-trend。 |
| 85 | `beam_med_d` | `base_replay` | 0.4542% | 83.7 | 841.8/725.0/672.0 |  | 7種類のBeam候補deltaの行別中央値。 |
| 86 | `ll_learned_prob_beam_mean` | `learned_likelihood` | 0.4483% | 87.3 | 952.0/629.2/654.4 |  | exp111 classifierが推定した複数Beam path平均のP(|error|≤10ft)。 |
| 87 | `uproj_diff_pf_z_minus_beam_med` | `u_projection` | 0.4430% | 87.0 | 779.6/702.8/691.0 |  | U-spaceのpf_z − beam_med。Z/anchor共通項は相殺。 |
| 88 | `uproj_absdiff_pf_z_beam_mean` | `u_projection` | 0.4133% | 93.3 | 816.6/645.0/587.0 |  | U-spaceの|pf_z − beam_mean|。 |
| 89 | `frm_rmse_EGFDU` | `base_replay` | 0.4114% | 99.7 | 511.8/764.0/695.8 |  | EGFDU formation surface TVT式の既知prefix RMSE。 |
| 90 | `nsel_pred_error_tvt_dense` | `nested_selector` | 0.4111% | 94.0 | 878.8/576.2/596.6 |  | nested selectorが予測したdense spatial ANCC（full-prefix bias）の絶対誤差。 |
| 91 | `frm_rmse_ASTNL` | `base_replay` | 0.4026% | 101.7 | 507.2/708.4/713.6 |  | ASTNL formation surface TVT式の既知prefix RMSE。 |
| 92 | `ll_learned_error_top1_value` | `learned_likelihood` | 0.4024% | 95.7 | 658.2/704.4/602.0 |  | exp111 predicted-absolute-error面のtop-k、rank、marginまたは誤差逆重みTVT要約。 |
| 93 | `dzdmd` | `base_replay` | 0.4023% | 100.3 | 1129.2/478.8/461.2 |  | 行差分 dZ/dMD。 |
| 94 | `nsel_pred_error_likpf_mean` | `nested_selector` | 0.3990% | 95.7 | 750.2/620.8/597.8 |  | nested selectorが予測したlikelihood-weighted PF平均の絶対誤差。 |
| 95 | `ll_learned_prob_weighted_tvt_minus_last_known_tvt` | `learned_likelihood` | 0.3982% | 98.3 | 649.8/624.6/666.8 |  | exp111 within10確率面のtop-k、rank、margin、entropyまたは確率加重TVT要約。 |
| 96 | `uproj_absdiff_pf_ancc_pf_z` | `u_projection` | 0.3857% | 100.7 | 830.4/540.6/555.4 |  | U-spaceの|pf_ancc − pf_z|。 |
| 97 | `uproj_absdiff_pf_z_beam_med` | `u_projection` | 0.3851% | 99.7 | 771.0/582.2/557.6 |  | U-spaceの|pf_z − beam_med|。 |
| 98 | `uproj_diff_beam_mean_minus_likpf_mean` | `u_projection` | 0.3849% | 100.7 | 747.8/615.2/542.6 |  | U-spaceのbeam_mean − likpf_mean。Z/anchor共通項は相殺。 |
| 99 | `ll_learned_pred_abs_error_likpf_mean` | `learned_likelihood` | 0.3827% | 100.7 | 637.8/630.4/601.6 |  | exp111 L1 modelが推定したlikelihood-weighted PF平均の絶対誤差。 |
| 100 | `uproj_absdiff_pf_z_likpf_mean` | `u_projection` | 0.3747% | 103.3 | 838.0/531.2/509.6 |  | U-spaceの|pf_z − likpf_mean|。 |
| 101 | `uproj_source_u_range` | `u_projection` | 0.3722% | 104.0 | 708.6/580.6/549.4 | u_projection_family_slim_review | 5候補のU値の行別range。 |
| 102 | `uproj_source_u_std` | `u_projection` | 0.3695% | 105.3 | 744.2/552.4/537.8 | u_projection_family_slim_review | 5候補のU値の行別標準偏差。 |
| 103 | `uproj_diff_beam_med_minus_likpf_mean` | `u_projection` | 0.3663% | 104.0 | 677.6/559.0/567.8 |  | U-spaceのbeam_med − likpf_mean。Z/anchor共通項は相殺。 |
| 104 | `nsel_pred_error_tvt_densew` | `nested_selector` | 0.3648% | 106.7 | 656.6/590.2/547.0 |  | nested selectorが予測したdense spatial ANCC（prefix加重bias）の絶対誤差。 |
| 105 | `ll_candidate_tvt_beam_mean_minus_last_known_tvt` | `learned_likelihood` | 0.3635% | 109.3 | 429.6/662.4/644.2 | existing_delta_duplicate → keep `beam_mean_d` (r=0.9999999962819376) | 元の複数Beam path平均 TVT − last_known_tvt。learned予測値ではない。 |
| 106 | `nsel_pred_error_tvt_dense50` | `nested_selector` | 0.3634% | 104.3 | 664.4/554.0/569.6 |  | nested selectorが予測したdense spatial ANCC（prefix末尾50 bias）の絶対誤差。 |
| 107 | `nsel_error_top1` | `nested_selector` | 0.3622% | 107.0 | 604.6/606.2/559.4 |  | selectorが予測したtop1候補の絶対誤差。 |
| 108 | `uproj_diff_pf_ancc_minus_beam_mean` | `u_projection` | 0.3618% | 106.0 | 660.0/569.6/550.8 |  | U-spaceのpf_ancc − beam_mean。Z/anchor共通項は相殺。 |
| 109 | `frm_rmse_EGFDL` | `base_replay` | 0.3612% | 109.7 | 408.2/654.6/657.6 |  | EGFDL formation surface TVT式の既知prefix RMSE。 |
| 110 | `grwr_known_prefix_rows_log1p` | `gr_wavelet_rotation` | 0.3600% | 112.3 | 223.2/752.2/698.6 |  | 既知TVT_input prefix行数のlog1p。 |
| 111 | `nsel_error_top2` | `nested_selector` | 0.3569% | 110.3 | 534.4/598.4/596.8 |  | selectorが予測したtop2候補の絶対誤差。 |
| 112 | `uproj_absdiff_beam_mean_beam_med` | `u_projection` | 0.3483% | 107.7 | 816.4/470.6/467.6 |  | U-spaceの|beam_mean − beam_med|。 |
| 113 | `uproj_diff_pf_ancc_minus_beam_med` | `u_projection` | 0.3392% | 112.7 | 636.6/536.8/500.2 |  | U-spaceのpf_ancc − beam_med。Z/anchor共通項は相殺。 |
| 114 | `sqrt_frac` | `base_replay` | 0.3362% | 117.0 | 231.6/673.0/663.4 |  | fracの平方根。 |
| 115 | `uproj_absdiff_pf_ancc_likpf_mean` | `u_projection` | 0.3315% | 111.3 | 740.8/460.8/460.4 |  | U-spaceの|pf_ancc − likpf_mean|。 |
| 116 | `frm_rmse_BUDA` | `base_replay` | 0.3244% | 117.7 | 369.2/611.4/565.8 |  | BUDA formation surface TVT式の既知prefix RMSE。 |
| 117 | `nsel_pred_error_pf_ancc` | `nested_selector` | 0.3225% | 113.3 | 656.6/470.8/475.0 |  | nested selectorが予測したANCC粒子フィルタの絶対誤差。 |
| 118 | `uproj_pf_ancc_corr` | `u_projection` | 0.3187% | 114.7 | 841.0/395.8/390.4 |  | pf_ancc U-projection: well内robust polynomial U-trend − source U。 |
| 119 | `uproj_pf_ancc_resid` | `u_projection` | 0.3043% | 118.7 | 730.8/386.8/418.8 | sign_flip_duplicate → keep `uproj_pf_ancc_corr` (r=-1.0) | pf_ancc U-projection: source U − well内robust polynomial U-trend。 |
| 120 | `pf_z` | `base_replay` | 0.2973% | 122.0 | 538.0/481.4/443.0 |  | Z-aware particle filterの絶対TVT候補。 |
| 121 | `uproj_beam_mean_corr` | `u_projection` | 0.2933% | 121.0 | 659.8/380.0/430.8 |  | beam_mean U-projection: well内robust polynomial U-trend − source U。 |
| 122 | `pf_ancc_std` | `base_replay` | 0.2894% | 123.0 | 919.2/285.6/307.2 |  | ANCC particle filter粒子の行別TVT標準偏差。 |
| 123 | `uproj_absdiff_pf_ancc_beam_mean` | `u_projection` | 0.2891% | 121.7 | 609.0/429.4/403.2 |  | U-spaceの|pf_ancc − beam_mean|。 |
| 124 | `ll_candidate_tvt_likpf_mean_minus_last_known_tvt` | `learned_likelihood` | 0.2860% | 127.3 | 282.2/530.4/540.4 | existing_delta_duplicate → keep `likpf_mean_d` (r=1.0) | 元のlikelihood-weighted PF平均 TVT − last_known_tvt。learned予測値ではない。 |
| 125 | `ll_learned_prob_pf_ancc` | `learned_likelihood` | 0.2809% | 127.3 | 511.0/423.0/447.4 |  | exp111 classifierが推定したANCC粒子フィルタのP(|error|≤10ft)。 |
| 126 | `ll_learned_error_weighted_tvt_minus_last_known_tvt` | `learned_likelihood` | 0.2805% | 127.3 | 529.4/408.4/445.6 |  | exp111 predicted-absolute-error面のtop-k、rank、marginまたは誤差逆重みTVT要約。 |
| 127 | `uproj_absdiff_pf_ancc_beam_med` | `u_projection` | 0.2756% | 126.7 | 574.8/424.2/374.2 |  | U-spaceの|pf_ancc − beam_med|。 |
| 128 | `uproj_beam_mean_resid` | `u_projection` | 0.2751% | 127.7 | 602.6/375.8/397.4 | sign_flip_duplicate → keep `uproj_beam_mean_corr` (r=-1.0) | beam_mean U-projection: source U − well内robust polynomial U-trend。 |
| 129 | `uproj_pf_z_corr` | `u_projection` | 0.2745% | 125.7 | 769.8/297.2/343.6 |  | pf_z U-projection: well内robust polynomial U-trend − source U。 |
| 130 | `ll_learned_prob_top2_value` | `learned_likelihood` | 0.2735% | 127.3 | 434.2/475.8/421.8 |  | exp111 within10確率面のtop-k、rank、margin、entropyまたは確率加重TVT要約。 |
| 131 | `pf_ancc` | `base_replay` | 0.2593% | 132.3 | 475.6/430.8/371.2 |  | ANCC particle filterの絶対TVT候補。 |
| 132 | `uproj_absdiff_beam_mean_likpf_mean` | `u_projection` | 0.2576% | 133.0 | 574.6/353.6/362.8 |  | U-spaceの|beam_mean − likpf_mean|。 |
| 133 | `uproj_absdiff_beam_med_likpf_mean` | `u_projection` | 0.2512% | 134.0 | 557.0/357.0/344.4 |  | U-spaceの|beam_med − likpf_mean|。 |
| 134 | `tda80` | `base_replay` | 0.2510% | 134.0 | 408.0/423.0/393.4 |  | raw GR − typewell GR(anchor TVT +80 ft)。 |
| 135 | `ll_learned_error_top2_value` | `learned_likelihood` | 0.2506% | 134.7 | 428.4/407.4/391.0 |  | exp111 predicted-absolute-error面のtop-k、rank、marginまたは誤差逆重みTVT要約。 |
| 136 | `uproj_beam_med_corr` | `u_projection` | 0.2472% | 135.0 | 562.8/314.2/363.6 |  | beam_med U-projection: well内robust polynomial U-trend − source U。 |
| 137 | `uproj_pf_z_resid` | `u_projection` | 0.2471% | 134.7 | 626.2/305.4/323.4 | sign_flip_duplicate → keep `uproj_pf_z_corr` (r=-1.0) | pf_z U-projection: source U − well内robust polynomial U-trend。 |
| 138 | `ll_learned_pred_abs_error_pf_ancc` | `learned_likelihood` | 0.2469% | 135.7 | 451.8/382.2/381.2 |  | exp111 L1 modelが推定したANCC粒子フィルタの絶対誤差。 |
| 139 | `uproj_beam_med_resid` | `u_projection` | 0.2414% | 137.3 | 546.0/324.4/341.0 | sign_flip_duplicate → keep `uproj_beam_med_corr` (r=-1.0) | beam_med U-projection: source U − well内robust polynomial U-trend。 |
| 140 | `tvtF_ASTNU` | `base_replay` | 0.2401% | 137.7 | 405.0/391.2/378.0 |  | ASTNU formation surface: spatial KNN surfaceとfull-prefix median biasから作るTVT候補。 |
| 141 | `uproj_diff_pf_ancc_minus_pf_z` | `u_projection` | 0.2363% | 139.0 | 189.8/470.8/447.8 | existing_disagreement_duplicate → keep `pf_vs_z` (r=1.0) | U-spaceのpf_ancc − pf_z。Z/anchor共通項は相殺。 |
| 142 | `grm101` | `base_replay` | 0.2245% | 139.3 | 641.0/252.2/263.6 |  | raw GRのcentered rolling-101平均。 |
| 143 | `nsel_top1_code` | `nested_selector` | 0.2222% | 141.0 | 591.6/262.8/280.6 |  | 予測絶対誤差が最小の候補index（数値code）。 |
| 144 | `grwr_dwt_detail_energy_ratio_w129` | `gr_wavelet_rotation` | 0.2200% | 143.0 | 739.2/185.8/233.2 |  | db4 level-3 DWT detailのdetail/(raw-local+detail) energy比（window 129）。 |
| 145 | `ll_candidate_tvt_pf_ancc_minus_last_known_tvt` | `learned_likelihood` | 0.2198% | 142.0 | 227.0/409.2/406.0 | existing_delta_duplicate → keep `pf_ancc_delta` (r=1.0) | 元のANCC粒子フィルタ TVT − last_known_tvt。learned予測値ではない。 |
| 146 | `ll_candidate_tvt_pf_ancc_minus_likpf_mean_tvt` | `learned_likelihood` | 0.2182% | 143.3 | 212.6/416.6/402.6 | existing_disagreement_duplicate → keep `uproj_diff_pf_ancc_minus_likpf_mean` (r=1.0) | 元のANCC粒子フィルタ TVT − likpf_mean_tvt。learned予測値ではない。 |
| 147 | `last_known_tvt` | `base_replay` | 0.2043% | 145.0 | 294.6/361.4/332.2 |  | 既知prefix末尾のTVT。全残差予測のanchor。 |
| 148 | `uproj_beam_mean_abs_resid` | `u_projection` | 0.1898% | 153.0 | 603.4/184.0/204.2 | u_projection_family_slim_review | beam_mean U-projection: 上記residualの絶対値。 |
| 149 | `uproj_corr_std` | `u_projection` | 0.1890% | 152.7 | 552.6/197.0/226.8 | u_projection_family_slim_review | 5候補のpolynomial correction値の行別標準偏差。 |
| 150 | `tvtFw_ASTNU` | `base_replay` | 0.1873% | 149.3 | 272.4/331.4/302.8 |  | ASTNU formation surface: spatial KNN surfaceとprefix指数加重biasから作るTVT候補。 |
| 151 | `tvtF_ANCC` | `base_replay` | 0.1846% | 150.0 | 333.6/282.2/291.6 |  | ANCC formation surface: spatial KNN surfaceとfull-prefix median biasから作るTVT候補。 |
| 152 | `grwr_raw_std_w129` | `gr_wavelet_rotation` | 0.1813% | 159.0 | 599.2/160.2/192.8 |  | raw GRのlocal rolling標準偏差（window 129）。 |
| 153 | `uproj_likpf_mean_abs_resid` | `u_projection` | 0.1792% | 157.3 | 565.2/166.6/203.2 | u_projection_family_slim_review | likpf_mean U-projection: 上記residualの絶対値。 |
| 154 | `ll_candidate_tvt_beam_mean_minus_likpf_mean_tvt` | `learned_likelihood` | 0.1760% | 160.7 | 134.8/352.4/337.0 | existing_disagreement_duplicate → keep `uproj_diff_beam_mean_minus_likpf_mean` (r=1.0) | 元の複数Beam path平均 TVT − likpf_mean_tvt。learned予測値ではない。 |
| 155 | `uproj_corr_range` | `u_projection` | 0.1745% | 160.0 | 537.6/173.6/196.8 | u_projection_family_slim_review | 5候補のpolynomial correction値の行別range。 |
| 156 | `uproj_pf_z_abs_resid` | `u_projection` | 0.1693% | 163.7 | 525.8/163.2/192.6 | u_projection_family_slim_review | pf_z U-projection: 上記residualの絶対値。 |
| 157 | `tvtF50_ASTNU` | `base_replay` | 0.1683% | 155.7 | 242.0/284.2/287.0 | formation_weighted_vs_last50_near_duplicate → keep `tvtFw_ASTNU` (r=0.9999904079094132) | ASTNU formation surface: spatial KNN surfaceとprefix末尾50行biasから作るTVT候補。 |
| 158 | `dense_rmse` | `base_replay` | 0.1661% | 156.0 | 248.0/296.0/261.4 |  | 既知prefixでのdense ANCC TVT式のRMSE（well定数）。 |
| 159 | `uproj_pf_ancc_abs_resid` | `u_projection` | 0.1655% | 167.7 | 552.0/146.6/171.8 | u_projection_family_slim_review | pf_ancc U-projection: 上記residualの絶対値。 |
| 160 | `uproj_beam_med_abs_resid` | `u_projection` | 0.1614% | 168.3 | 514.4/152.8/176.2 | u_projection_family_slim_review | beam_med U-projection: 上記residualの絶対値。 |
| 161 | `tvtF_ASTNL` | `base_replay` | 0.1611% | 156.7 | 255.2/262.6/266.2 |  | ASTNL formation surface: spatial KNN surfaceとfull-prefix median biasから作るTVT候補。 |
| 162 | `tvtF_EGFDL` | `base_replay` | 0.1608% | 156.0 | 251.2/264.0/266.6 |  | EGFDL formation surface: spatial KNN surfaceとfull-prefix median biasから作るTVT候補。 |
| 163 | `tvtF_EGFDU` | `base_replay` | 0.1587% | 157.3 | 248.2/266.6/257.0 |  | EGFDU formation surface: spatial KNN surfaceとfull-prefix median biasから作るTVT候補。 |
| 164 | `nsel_top2_code` | `nested_selector` | 0.1582% | 170.7 | 524.6/147.2/159.8 |  | 予測絶対誤差が2番目に小さい候補index（数値code）。 |
| 165 | `ll_learned_prob_weighted_tvt_minus_likpf_mean_tvt` | `learned_likelihood` | 0.1489% | 160.3 | 269.8/239.8/222.8 |  | exp111 within10確率面のtop-k、rank、margin、entropyまたは確率加重TVT要約。 |
| 166 | `tvtFw_ANCC` | `base_replay` | 0.1415% | 164.0 | 226.0/244.2/219.0 |  | ANCC formation surface: spatial KNN surfaceとprefix指数加重biasから作るTVT候補。 |
| 167 | `tvtF_BUDA` | `base_replay` | 0.1412% | 164.0 | 195.4/240.0/245.0 |  | BUDA formation surface: spatial KNN surfaceとfull-prefix median biasから作るTVT候補。 |
| 168 | `ll_learned_error_weighted_tvt_minus_likpf_mean_tvt` | `learned_likelihood` | 0.1373% | 164.7 | 277.6/205.0/199.0 |  | exp111 predicted-absolute-error面のtop-k、rank、marginまたは誤差逆重みTVT要約。 |
| 169 | `tvtF50_ANCC` | `base_replay` | 0.1365% | 165.3 | 202.2/233.8/225.0 | formation_weighted_vs_last50_near_duplicate → keep `tvtFw_ANCC` (r=0.999990304638189) | ANCC formation surface: spatial KNN surfaceとprefix末尾50行biasから作るTVT候補。 |
| 170 | `bw_early_ANCC` | `base_replay` | 0.1348% | 166.7 | 187.4/242.0/221.0 |  | ANCC formation surface: prefix前1/3のTVT+Z−formation median bias。 |
| 171 | `bw_early_ASTNU` | `base_replay` | 0.1338% | 168.0 | 178.6/249.6/215.8 |  | ASTNU formation surface: prefix前1/3のTVT+Z−formation median bias。 |
| 172 | `grwr_raw_minus_rolling_absmean_w129` | `gr_wavelet_rotation` | 0.1300% | 179.0 | 404.2/130.6/142.4 |  | |raw GR − rolling denoised GR|のlocal平均（window 129）。 |
| 173 | `grwr_raw_minus_savgol_absmean_w129` | `gr_wavelet_rotation` | 0.1284% | 180.3 | 416.2/115.4/141.0 |  | |raw GR − savgol denoised GR|のlocal平均（window 129）。 |
| 174 | `tvtFw_ASTNL` | `base_replay` | 0.1254% | 171.0 | 175.0/217.8/212.2 |  | ASTNL formation surface: spatial KNN surfaceとprefix指数加重biasから作るTVT候補。 |
| 175 | `tvtFw_EGFDL` | `base_replay` | 0.1246% | 171.7 | 180.4/226.6/196.0 |  | EGFDL formation surface: spatial KNN surfaceとprefix指数加重biasから作るTVT候補。 |
| 176 | `bw_ANCC` | `base_replay` | 0.1202% | 175.0 | 179.2/217.0/186.8 |  | ANCC formation surface: full-prefixで推定したTVT+Z−formationのmedian bias。 |
| 177 | `grwr_dwt_detail_energy_w129` | `gr_wavelet_rotation` | 0.1196% | 185.7 | 399.6/103.4/126.2 |  | db4 level-3 DWT detailのdetail二乗平均（window 129）。 |
| 178 | `tvtF50_EGFDL` | `base_replay` | 0.1192% | 174.3 | 163.4/214.0/197.0 | formation_weighted_vs_last50_near_duplicate → keep `tvtFw_EGFDL` (r=0.9999924985201734) | EGFDL formation surface: spatial KNN surfaceとprefix末尾50行biasから作るTVT候補。 |
| 179 | `tdbc40` | `base_replay` | 0.1186% | 177.3 | 284.2/158.0/156.8 |  | raw GR − typewell GR(Beam reference TVT +40 ft)。 |
| 180 | `tvtF50_ASTNL` | `base_replay` | 0.1176% | 176.7 | 168.2/212.2/188.2 | formation_weighted_vs_last50_near_duplicate → keep `tvtFw_ASTNL` (r=0.999992160518398) | ASTNL formation surface: spatial KNN surfaceとprefix末尾50行biasから作るTVT候補。 |
| 181 | `tvtFw_EGFDU` | `base_replay` | 0.1140% | 178.3 | 154.8/207.6/187.0 |  | EGFDU formation surface: spatial KNN surfaceとprefix指数加重biasから作るTVT候補。 |
| 182 | `tvtFw_BUDA` | `base_replay` | 0.1127% | 179.7 | 142.6/205.8/192.2 |  | BUDA formation surface: spatial KNN surfaceとprefix指数加重biasから作るTVT候補。 |
| 183 | `tda40` | `base_replay` | 0.1120% | 180.7 | 244.6/163.8/151.8 |  | raw GR − typewell GR(anchor TVT +40 ft)。 |
| 184 | `tvtF50_BUDA` | `base_replay` | 0.1106% | 182.0 | 144.8/202.2/184.8 | formation_weighted_vs_last50_near_duplicate → keep `tvtFw_BUDA` (r=0.9999921650393682) | BUDA formation surface: spatial KNN surfaceとprefix末尾50行biasから作るTVT候補。 |
| 185 | `bw_early_EGFDU` | `base_replay` | 0.1104% | 180.7 | 129.8/204.2/193.0 |  | EGFDU formation surface: prefix前1/3のTVT+Z−formation median bias。 |
| 186 | `bw_ASTNU` | `base_replay` | 0.1091% | 182.7 | 135.6/203.2/184.0 |  | ASTNU formation surface: full-prefixで推定したTVT+Z−formationのmedian bias。 |
| 187 | `tvtF50_EGFDU` | `base_replay` | 0.1077% | 183.3 | 152.4/198.0/170.2 | formation_weighted_vs_last50_near_duplicate → keep `tvtFw_EGFDU` (r=0.9999924884500538) | EGFDU formation surface: spatial KNN surfaceとprefix末尾50行biasから作るTVT候補。 |
| 188 | `grwr_raw_rolling_corr_w129` | `gr_wavelet_rotation` | 0.1072% | 190.0 | 350.6/89.2/122.0 |  | raw GRとrolling denoised GRのlocal相関（window 129）。 |
| 189 | `bww_ANCC` | `base_replay` | 0.1000% | 187.3 | 129.2/178.6/172.2 |  | ANCC formation surface: prefix後半を重くしたTVT+Z−formationの指数加重bias。 |
| 190 | `bw_early_ASTNL` | `base_replay` | 0.0952% | 191.7 | 107.8/173.8/171.8 |  | ASTNL formation surface: prefix前1/3のTVT+Z−formation median bias。 |
| 191 | `bww_ASTNU` | `base_replay` | 0.0946% | 191.7 | 116.0/180.2/157.0 |  | ASTNU formation surface: prefix後半を重くしたTVT+Z−formationの指数加重bias。 |
| 192 | `grwr_dwt_detail_absmean_w129` | `gr_wavelet_rotation` | 0.0900% | 195.3 | 274.0/89.0/104.4 |  | db4 level-3 DWT detailのdetail絶対値平均（window 129）。 |
| 193 | `bw_early_EGFDL` | `base_replay` | 0.0862% | 203.3 | 75.4/167.8/162.6 |  | EGFDL formation surface: prefix前1/3のTVT+Z−formation median bias。 |
| 194 | `grwr_raw_savgol_corr_w129` | `gr_wavelet_rotation` | 0.0861% | 199.0 | 292.2/64.6/96.8 |  | raw GRとsavgol denoised GRのlocal相関（window 129）。 |
| 195 | `dense_bias` | `base_replay` | 0.0849% | 210.0 | 52.0/189.6/153.6 | near_exact_public_replay_duplicate → keep `dense_rmse` (r=0.9999999994694734) | 既知prefixでのdense ANCC TVT式の平均bias（well定数）。 |
| 196 | `bw_mid_ASTNU` | `base_replay` | 0.0841% | 200.3 | 95.2/158.4/147.4 |  | ASTNU formation surface: prefix中1/3のTVT+Z−formation median bias。 |
| 197 | `grs101` | `base_replay` | 0.0837% | 200.7 | 268.6/77.4/92.0 |  | raw GRのcentered rolling-101標準偏差。 |
| 198 | `nsel_error_margin` | `nested_selector` | 0.0811% | 202.3 | 262.0/79.2/83.8 |  | predicted error top2 − top1。 |
| 199 | `ll_learned_error_margin_top2_top1` | `learned_likelihood` | 0.0799% | 198.7 | 168.8/110.8/118.6 |  | exp111 predicted-absolute-error面のtop-k、rank、marginまたは誤差逆重みTVT要約。 |
| 200 | `bw_EGFDU` | `base_replay` | 0.0786% | 204.7 | 87.0/153.2/134.0 |  | EGFDU formation surface: full-prefixで推定したTVT+Z−formationのmedian bias。 |
| 201 | `grwr_dwt_detail_energy_ratio_w065` | `gr_wavelet_rotation` | 0.0752% | 212.3 | 306.0/45.2/57.0 |  | db4 level-3 DWT detailのdetail/(raw-local+detail) energy比（window 65）。 |
| 202 | `ll_learned_prob_margin_top1_top2` | `learned_likelihood` | 0.0751% | 201.7 | 190.0/95.4/96.2 |  | exp111 within10確率面のtop-k、rank、margin、entropyまたは確率加重TVT要約。 |
| 203 | `bww_ASTNL` | `base_replay` | 0.0740% | 211.0 | 73.6/147.2/129.6 |  | ASTNL formation surface: prefix後半を重くしたTVT+Z−formationの指数加重bias。 |
| 204 | `grwr_raw_dwt_corr_w129` | `gr_wavelet_rotation` | 0.0723% | 208.0 | 231.4/60.0/86.6 |  | raw GRとdwt denoised GRのlocal相関（window 129）。 |
| 205 | `bw50_ASTNU` | `base_replay` | 0.0722% | 210.7 | 78.0/131.6/133.4 | formation_weighted_vs_last50_near_duplicate → keep `bww_ASTNU` (r=0.999991055011816) | ASTNU formation surface: prefix末尾50行のTVT+Z−formation median bias。 |
| 206 | `tda-40` | `base_replay` | 0.0711% | 205.0 | 127.6/119.0/103.2 |  | raw GR − typewell GR(anchor TVT -40 ft)。 |
| 207 | `bw_mid_ANCC` | `base_replay` | 0.0693% | 213.0 | 76.4/127.2/126.0 |  | ANCC formation surface: prefix中1/3のTVT+Z−formation median bias。 |
| 208 | `grm51` | `base_replay` | 0.0684% | 209.0 | 184.4/90.4/75.6 |  | raw GRのcentered rolling-51平均。 |
| 209 | `bw50_ANCC` | `base_replay` | 0.0682% | 218.3 | 59.0/139.6/122.4 | formation_weighted_vs_last50_near_duplicate → keep `bww_ANCC` (r=0.9999911224154846) | ANCC formation surface: prefix末尾50行のTVT+Z−formation median bias。 |
| 210 | `bw_ASTNL` | `base_replay` | 0.0679% | 216.7 | 67.0/130.2/124.2 |  | ASTNL formation surface: full-prefixで推定したTVT+Z−formationのmedian bias。 |
| 211 | `bww_EGFDU` | `base_replay` | 0.0675% | 215.3 | 73.2/134.8/113.0 |  | EGFDU formation surface: prefix後半を重くしたTVT+Z−formationの指数加重bias。 |
| 212 | `ll_candidate_tvt_sc_ens_minus_likpf_mean_tvt` | `learned_likelihood` | 0.0659% | 209.0 | 106.6/116.8/98.0 | high_corr_or_redundancy_review | 元のmulti-scale NCC ensemble TVT − likpf_mean_tvt。learned予測値ではない。 |
| 213 | `tdbc-20` | `base_replay` | 0.0655% | 208.7 | 149.2/91.0/88.8 |  | raw GR − typewell GR(Beam reference TVT -20 ft)。 |
| 214 | `tda-80` | `base_replay` | 0.0653% | 211.0 | 170.2/78.8/83.8 |  | raw GR − typewell GR(anchor TVT -80 ft)。 |
| 215 | `bw_early_BUDA` | `base_replay` | 0.0634% | 221.0 | 60.6/126.6/112.6 |  | BUDA formation surface: prefix前1/3のTVT+Z−formation median bias。 |
| 216 | `bw_mid_ASTNL` | `base_replay` | 0.0609% | 223.3 | 57.0/120.6/110.2 |  | ASTNL formation surface: prefix中1/3のTVT+Z−formation median bias。 |
| 217 | `tdbc-40` | `base_replay` | 0.0605% | 214.7 | 132.8/86.8/83.4 |  | raw GR − typewell GR(Beam reference TVT -40 ft)。 |
| 218 | `ll_candidate_tvt_hyb_minus_likpf_mean_tvt` | `learned_likelihood` | 0.0579% | 220.0 | 86.2/106.6/88.0 | high_corr_or_redundancy_review | 元のBeam/NCC hybrid TVT − likpf_mean_tvt。learned予測値ではない。 |
| 219 | `bw50_ASTNL` | `base_replay` | 0.0575% | 230.3 | 44.2/110.2/114.8 | formation_weighted_vs_last50_near_duplicate → keep `bww_ASTNL` (r=0.9999923004739684) | ASTNL formation surface: prefix末尾50行のTVT+Z−formation median bias。 |
| 220 | `nsel_error_ratio` | `nested_selector` | 0.0575% | 218.7 | 188.4/47.6/65.6 |  | predicted error top1 / max(top2, 1e-3)。 |
| 221 | `ll_learned_prob_entropy` | `learned_likelihood` | 0.0559% | 218.3 | 145.0/61.4/78.0 |  | exp111 within10確率面のtop-k、rank、margin、entropyまたは確率加重TVT要約。 |
| 222 | `bw_mid_EGFDU` | `base_replay` | 0.0541% | 232.0 | 46.8/101.8/105.6 |  | EGFDU formation surface: prefix中1/3のTVT+Z−formation median bias。 |
| 223 | `bw50_EGFDU` | `base_replay` | 0.0533% | 234.3 | 39.8/114.6/95.4 | formation_weighted_vs_last50_near_duplicate → keep `bww_EGFDU` (r=0.9999925776098378) | EGFDU formation surface: prefix末尾50行のTVT+Z−formation median bias。 |
| 224 | `bw_mid_EGFDL` | `base_replay` | 0.0529% | 233.7 | 46.2/107.6/95.4 |  | EGFDL formation surface: prefix中1/3のTVT+Z−formation median bias。 |
| 225 | `bww_EGFDL` | `base_replay` | 0.0511% | 232.7 | 51.2/103.2/88.0 |  | EGFDL formation surface: prefix後半を重くしたTVT+Z−formationの指数加重bias。 |
| 226 | `ll_learned_prob_likpf_rank` | `learned_likelihood` | 0.0493% | 230.7 | 74.2/86.4/78.4 |  | exp111 within10確率面のtop-k、rank、margin、entropyまたは確率加重TVT要約。 |
| 227 | `tda-20` | `base_replay` | 0.0490% | 225.7 | 97.8/72.8/72.4 |  | raw GR − typewell GR(anchor TVT -20 ft)。 |
| 228 | `bw_BUDA` | `base_replay` | 0.0484% | 239.3 | 38.4/104.0/84.8 |  | BUDA formation surface: full-prefixで推定したTVT+Z−formationのmedian bias。 |
| 229 | `bw_EGFDL` | `base_replay` | 0.0484% | 236.3 | 48.8/97.0/83.6 |  | EGFDL formation surface: full-prefixで推定したTVT+Z−formationのmedian bias。 |
| 230 | `tdpf30` | `base_replay` | 0.0476% | 226.7 | 158.6/44.8/47.2 |  | raw GR − typewell GR(ANCC PF TVT +30 ft)。 |
| 231 | `tdbc20` | `base_replay` | 0.0464% | 226.3 | 106.0/67.8/59.4 |  | raw GR − typewell GR(Beam reference TVT +20 ft)。 |
| 232 | `nsel_top1_is_blend_likpf_hmm_w500` | `nested_selector` | 0.0459% | 228.0 | 127.6/43.8/64.0 |  | selector top1がlikPFとexact HMMの50/50平均であるone-hot flag。 |
| 233 | `nsel_top1_is_likpf_mean` | `nested_selector` | 0.0437% | 229.3 | 113.0/50.6/58.6 |  | selector top1がlikelihood-weighted PF平均であるone-hot flag。 |
| 234 | `tdpf15` | `base_replay` | 0.0419% | 233.0 | 90.2/64.2/55.0 |  | raw GR − typewell GR(ANCC PF TVT +15 ft)。 |
| 235 | `bw_mid_BUDA` | `base_replay` | 0.0416% | 248.7 | 32.8/83.2/79.0 |  | BUDA formation surface: prefix中1/3のTVT+Z−formation median bias。 |
| 236 | `tdbc10` | `base_replay` | 0.0411% | 232.3 | 95.2/53.2/58.4 |  | raw GR − typewell GR(Beam reference TVT +10 ft)。 |
| 237 | `tda20` | `base_replay` | 0.0404% | 232.7 | 97.8/51.4/54.8 |  | raw GR − typewell GR(anchor TVT +20 ft)。 |
| 238 | `bw50_EGFDL` | `base_replay` | 0.0393% | 250.7 | 33.2/76.0/75.4 | formation_weighted_vs_last50_near_duplicate → keep `bww_EGFDL` (r=0.9999925774984988) | EGFDL formation surface: prefix末尾50行のTVT+Z−formation median bias。 |
| 239 | `gr_env` | `base_replay` | 0.0392% | 235.7 | 120.8/43.8/39.6 |  | raw GRのcentered rolling-21最大値。 |
| 240 | `tdpf-30` | `base_replay` | 0.0389% | 234.3 | 102.4/50.6/45.6 |  | raw GR − typewell GR(ANCC PF TVT -30 ft)。 |
| 241 | `bww_BUDA` | `base_replay` | 0.0385% | 251.0 | 35.0/72.0/74.4 |  | BUDA formation surface: prefix後半を重くしたTVT+Z−formationの指数加重bias。 |
| 242 | `ll_multiobs_mae_beam_mean` | `learned_likelihood` | 0.0368% | 239.3 | 122.2/31.4/39.6 |  | 複数Beam path平均のmulti-observation GR MAE。 |
| 243 | `tda5` | `base_replay` | 0.0361% | 239.0 | 89.4/48.6/45.0 |  | raw GR − typewell GR(anchor TVT +5 ft)。 |
| 244 | `tda10` | `base_replay` | 0.0347% | 242.3 | 83.4/49.0/43.0 |  | raw GR − typewell GR(anchor TVT +10 ft)。 |
| 245 | `grwr_raw_minus_dwt_absmean_w129` | `gr_wavelet_rotation` | 0.0342% | 247.0 | 50.4/52.2/62.8 |  | |raw GR − dwt denoised GR|のlocal平均（window 129）。 |
| 246 | `tdbc-3` | `base_replay` | 0.0342% | 242.7 | 89.4/41.2/43.6 |  | raw GR − typewell GR(Beam reference TVT -3 ft)。 |
| 247 | `grwr_raw_std_w065_x_log1p_md_since` | `gr_wavelet_rotation` | 0.0334% | 244.7 | 127.0/23.0/29.2 |  | raw GR local std(w65) × log1p(md_since)。 |
| 248 | `ll_multiobs_mae_likpf_mean` | `learned_likelihood` | 0.0331% | 244.7 | 94.4/37.0/39.2 |  | likelihood-weighted PF平均のmulti-observation GR MAE。 |
| 249 | `ll_multiobs_mae_pf_ancc` | `learned_likelihood` | 0.0331% | 243.7 | 95.4/39.4/36.0 |  | ANCC粒子フィルタのmulti-observation GR MAE。 |
| 250 | `tda-10` | `base_replay` | 0.0328% | 245.7 | 69.4/47.8/46.2 |  | raw GR − typewell GR(anchor TVT -10 ft)。 |
| 251 | `bw50_BUDA` | `base_replay` | 0.0323% | 258.0 | 27.4/65.2/59.2 | formation_weighted_vs_last50_near_duplicate → keep `bww_BUDA` (r=0.9999925897079838) | BUDA formation surface: prefix末尾50行のTVT+Z−formation median bias。 |
| 252 | `grwr_ll_entropy_x_dwt_energy_ratio_w065` | `gr_wavelet_rotation` | 0.0316% | 247.3 | 113.4/21.8/32.8 |  | learned-likelihood entropy × DWT detail energy比(w65)。 |
| 253 | `tdbc0` | `base_replay` | 0.0307% | 249.7 | 67.8/39.4/46.4 |  | raw GR − typewell GR(Beam reference TVT +0 ft)。 |
| 254 | `tdpf-2` | `base_replay` | 0.0298% | 250.0 | 79.4/35.8/37.0 |  | raw GR − typewell GR(ANCC PF TVT -2 ft)。 |
| 255 | `tdbc5` | `base_replay` | 0.0295% | 249.3 | 88.6/30.2/34.2 |  | raw GR − typewell GR(Beam reference TVT +5 ft)。 |
| 256 | `ll_learned_error_likpf_rank` | `learned_likelihood` | 0.0294% | 254.7 | 40.8/51.2/49.8 |  | exp111 predicted-absolute-error面のtop-k、rank、marginまたは誤差逆重みTVT要約。 |
| 257 | `nsel_top1_is_hmm_selfgr_boost_only_a070_c100_mean_tvt` | `nested_selector` | 0.0289% | 250.7 | 85.8/22.8/40.8 |  | selector top1がself-GR HMMであるone-hot flag。 |
| 258 | `grwr_raw_minus_savgol_absmean_w065` | `gr_wavelet_rotation` | 0.0282% | 255.0 | 118.0/15.0/20.6 |  | |raw GR − savgol denoised GR|のlocal平均（window 65）。 |
| 259 | `tdbc3` | `base_replay` | 0.0282% | 252.7 | 74.6/34.6/34.6 |  | raw GR − typewell GR(Beam reference TVT +3 ft)。 |
| 260 | `grwr_raw_std_w065` | `gr_wavelet_rotation` | 0.0277% | 252.7 | 102.8/18.4/27.0 |  | raw GRのlocal rolling標準偏差（window 65）。 |
| 261 | `tdbc-10` | `base_replay` | 0.0256% | 257.0 | 71.0/27.4/33.0 |  | raw GR − typewell GR(Beam reference TVT -10 ft)。 |
| 262 | `nsel_score_std` | `nested_selector` | 0.0255% | 256.0 | 95.6/18.8/22.2 |  | 11候補predicted errorの行別標準偏差。 |
| 263 | `grwr_raw_minus_rolling_absmean_w065` | `gr_wavelet_rotation` | 0.0248% | 258.3 | 94.8/14.4/23.8 |  | |raw GR − rolling denoised GR|のlocal平均（window 65）。 |
| 264 | `nsel_top1_is_hyb` | `nested_selector` | 0.0243% | 278.0 | 125.8/4.4/7.6 |  | selector top1がBeam/NCC hybridであるone-hot flag。 |
| 265 | `nsel_top1_is_exp226_v6_k16_geometry_gr_u_projection` | `nested_selector` | 0.0239% | 268.0 | 27.8/42.8/43.6 |  | selector top1がexp226 K16 geometry/GR/U-projectionであるone-hot flag。 |
| 266 | `gr_vs_slp_all` | `base_replay` | 0.0239% | 259.7 | 66.0/27.8/28.8 |  | raw GR − 全prefix slope外挿TVTでのtypewell GR。 |
| 267 | `tdbc-5` | `base_replay` | 0.0235% | 260.0 | 80.0/21.4/22.6 |  | raw GR − typewell GR(Beam reference TVT -5 ft)。 |
| 268 | `grwr_dwt_detail_energy_w065` | `gr_wavelet_rotation` | 0.0234% | 263.7 | 102.2/9.8/16.6 |  | db4 level-3 DWT detailのdetail二乗平均（window 65）。 |
| 269 | `tdpf8` | `base_replay` | 0.0231% | 261.3 | 62.8/29.2/26.6 |  | raw GR − typewell GR(ANCC PF TVT +8 ft)。 |
| 270 | `tda-5` | `base_replay` | 0.0228% | 261.0 | 64.0/25.4/28.0 |  | raw GR − typewell GR(anchor TVT -5 ft)。 |
| 271 | `ll_learned_error_top1_index` | `learned_likelihood` | 0.0226% | 268.7 | 32.2/40.0/37.0 |  | exp111 predicted-absolute-error面のtop-k、rank、marginまたは誤差逆重みTVT要約。 |
| 272 | `tdpf-15` | `base_replay` | 0.0219% | 264.3 | 85.0/15.8/17.0 |  | raw GR − typewell GR(ANCC PF TVT -15 ft)。 |
| 273 | `ll_learned_prob_top1_index` | `learned_likelihood` | 0.0207% | 270.0 | 36.0/31.6/34.0 |  | exp111 within10確率面のtop-k、rank、margin、entropyまたは確率加重TVT要約。 |
| 274 | `nsel_top1_is_beam_mean` | `nested_selector` | 0.0204% | 268.3 | 81.4/13.0/16.0 |  | selector top1が複数Beam path平均であるone-hot flag。 |
| 275 | `nsel_top1_is_tvt_dense` | `nested_selector` | 0.0204% | 269.3 | 86.4/13.2/12.0 |  | selector top1がdense spatial ANCC（full-prefix bias）であるone-hot flag。 |
| 276 | `gr_nrg` | `base_replay` | 0.0200% | 266.3 | 52.0/23.0/26.8 |  | raw GR二乗のrolling-21平均平方根。 |
| 277 | `tdpf-4` | `base_replay` | 0.0196% | 268.3 | 60.6/19.4/22.2 |  | raw GR − typewell GR(ANCC PF TVT -4 ft)。 |
| 278 | `gr_vs_tw_anc` | `base_replay` | 0.0177% | 273.3 | 47.8/21.2/21.8 |  | raw GR − anchor TVTでのtypewell GR。 |
| 279 | `tdpf2` | `base_replay` | 0.0173% | 273.0 | 52.0/19.2/18.6 |  | raw GR − typewell GR(ANCC PF TVT +2 ft)。 |
| 280 | `grm21` | `base_replay` | 0.0171% | 273.3 | 41.8/22.0/22.6 |  | raw GRのcentered rolling-21平均。 |
| 281 | `grwr_raw_rolling_corr_w065` | `gr_wavelet_rotation` | 0.0171% | 284.3 | 78.6/5.6/10.4 |  | raw GRとrolling denoised GRのlocal相関（window 65）。 |
| 282 | `tdpf4` | `base_replay` | 0.0170% | 273.7 | 54.6/17.0/17.2 |  | raw GR − typewell GR(ANCC PF TVT +4 ft)。 |
| 283 | `tdpf-8` | `base_replay` | 0.0168% | 275.3 | 61.0/14.0/14.6 |  | raw GR − typewell GR(ANCC PF TVT -8 ft)。 |
| 284 | `glead30` | `base_replay` | 0.0163% | 280.0 | 69.2/8.8/11.0 |  | raw GRを30行leadした値。 |
| 285 | `grwr_dwt_detail_absmean_w065` | `gr_wavelet_rotation` | 0.0160% | 280.7 | 67.8/9.4/10.4 |  | db4 level-3 DWT detailのdetail絶対値平均（window 65）。 |
| 286 | `sig_mean_d` | `base_replay` | 0.0157% | 278.0 | 55.6/12.6/15.2 |  | PF・Beam・NCC・formation・dense候補群の平均 − anchor。 |
| 287 | `ll_learned_prob_sc_ens` | `learned_likelihood` | 0.0149% | 281.3 | 29.4/23.4/21.0 |  | exp111 classifierが推定したmulti-scale NCC ensembleのP(|error|≤10ft)。 |
| 288 | `grs51` | `base_replay` | 0.0140% | 287.0 | 59.6/6.4/10.6 |  | raw GRのcentered rolling-51標準偏差。 |
| 289 | `nsel_pred_error_hyb` | `nested_selector` | 0.0135% | 283.3 | 41.6/14.2/14.6 |  | nested selectorが予測したBeam/NCC hybridの絶対誤差。 |
| 290 | `nsel_top1_is_tvt_densew` | `nested_selector` | 0.0135% | 285.0 | 46.8/11.2/13.4 |  | selector top1がdense spatial ANCC（prefix加重bias）であるone-hot flag。 |
| 291 | `glag30` | `base_replay` | 0.0132% | 285.3 | 47.6/11.2/11.6 |  | raw GRを30行lagした値。 |
| 292 | `ll_learned_prob_hyb` | `learned_likelihood` | 0.0132% | 284.3 | 35.0/15.2/17.2 |  | exp111 classifierが推定したBeam/NCC hybridのP(|error|≤10ft)。 |
| 293 | `tda0` | `base_replay` | 0.0128% | 287.7 | 28.2/21.8/14.0 | near_exact_public_replay_duplicate → keep `gr_vs_tw_anc` (r=0.9999999707804362) | raw GR − typewell GR(anchor TVT +0 ft)。 |
| 294 | `hyb_d` | `base_replay` | 0.0127% | 286.0 | 38.0/12.0/16.0 |  | Beam/NCC hybrid候補 − anchor。 |
| 295 | `sig_std` | `base_replay` | 0.0117% | 293.3 | 44.4/9.8/8.4 |  | PF・Beam・NCC・formation・dense候補群の行別標準偏差。 |
| 296 | `sc_vs_beam` | `base_replay` | 0.0116% | 296.7 | 48.2/6.0/9.0 |  | NCC ensemble候補 − conservative Beam候補。 |
| 297 | `grwr_dwt_approx_default_candidate_cost` | `gr_wavelet_rotation` | 0.0114% | 291.0 | 39.0/8.8/12.2 |  | dwt_approx GR面: default likPFのlocal GR observation cost。 |
| 298 | `nsel_pred_error_sc_ens` | `nested_selector` | 0.0111% | 292.3 | 38.6/9.2/11.0 |  | nested selectorが予測したmulti-scale NCC ensembleの絶対誤差。 |
| 299 | `nsel_top1_is_pf_ancc` | `nested_selector` | 0.0110% | 292.0 | 35.4/10.2/12.0 |  | selector top1がANCC粒子フィルタであるone-hot flag。 |
| 300 | `grwr_raw_savgol_corr_w065` | `gr_wavelet_rotation` | 0.0109% | 302.7 | 50.8/3.6/6.0 |  | raw GRとsavgol denoised GRのlocal相関（window 65）。 |
| 301 | `grwr_raw_dwt_corr_w065` | `gr_wavelet_rotation` | 0.0108% | 301.3 | 46.6/5.8/6.8 |  | raw GRとdwt denoised GRのlocal相関（window 65）。 |
| 302 | `nsel_top1_is_sc_ens` | `nested_selector` | 0.0102% | 327.7 | 58.8/0.4/0.2 |  | selector top1がmulti-scale NCC ensembleであるone-hot flag。 |
| 303 | `nsel_score_mean` | `nested_selector` | 0.0097% | 297.3 | 24.8/10.8/13.8 |  | 11候補predicted errorの行別平均。 |
| 304 | `grwr_fft_rotation_ratio_x_candidate_range` | `gr_wavelet_rotation` | 0.0089% | 302.7 | 33.2/6.8/7.8 |  | FFT rotation-band energy比 × 8候補TVT range。 |
| 305 | `grwr_rolling_median_11_default_candidate_cost` | `gr_wavelet_rotation` | 0.0085% | 304.3 | 29.8/6.6/8.8 |  | rolling_median_11 GR面: default likPFのlocal GR observation cost。 |
| 306 | `grwr_candidate_tvt_std` | `gr_wavelet_rotation` | 0.0084% | 303.3 | 25.2/8.2/10.2 |  | 8候補TVTの行別標準偏差。 |
| 307 | `grwr_savgol_31_p2_default_candidate_cost` | `gr_wavelet_rotation` | 0.0083% | 307.0 | 31.0/6.2/7.2 |  | savgol_31_p2 GR面: default likPFのlocal GR observation cost。 |
| 308 | `grwr_dwt_detail_energy_ratio_w033` | `gr_wavelet_rotation` | 0.0081% | 313.3 | 37.8/2.6/4.6 |  | db4 level-3 DWT detailのdetail/(raw-local+detail) energy比（window 33）。 |
| 309 | `nsel_candidate_range` | `nested_selector` | 0.0077% | 308.7 | 23.4/6.4/10.2 |  | 11候補TVTの行別range。 |
| 310 | `sc_ens_d` | `base_replay` | 0.0076% | 309.0 | 24.6/6.4/9.0 |  | multi-scale NCC ensemble候補 − anchor。 |
| 311 | `grwr_dwt_energy_ratio_w065_x_candidate_std` | `gr_wavelet_rotation` | 0.0075% | 316.3 | 36.0/2.4/3.4 |  | DWT detail energy比(w65) × 8候補TVT標準偏差。 |
| 312 | `grwr_candidate_tvt_range` | `gr_wavelet_rotation` | 0.0074% | 309.0 | 26.8/7.0/5.8 |  | 8候補TVTの行別range。 |
| 313 | `grwr_raw_std_w033` | `gr_wavelet_rotation` | 0.0072% | 323.3 | 37.4/1.0/2.6 |  | raw GRのlocal rolling標準偏差（window 33）。 |
| 314 | `sc_cons_d` | `base_replay` | 0.0071% | 312.3 | 23.6/6.0/7.8 |  | sc8/sc15/sc25候補の平均 − anchor。 |
| 315 | `ll_candidate_tvt_range` | `learned_likelihood` | 0.0071% | 311.7 | 25.2/6.6/5.8 | high_corr_or_redundancy_review | exp111の5候補TVTの行別range。 |
| 316 | `nsel_top1_is_tvt_dense50` | `nested_selector` | 0.0064% | 322.0 | 30.8/2.2/2.8 |  | selector top1がdense spatial ANCC（prefix末尾50 bias）であるone-hot flag。 |
| 317 | `grwr_raw_minus_savgol_absmean_w033` | `gr_wavelet_rotation` | 0.0064% | 323.0 | 30.6/1.4/3.6 |  | |raw GR − savgol denoised GR|のlocal平均（window 33）。 |
| 318 | `glag15` | `base_replay` | 0.0059% | 320.0 | 23.8/3.8/4.2 |  | raw GRを15行lagした値。 |
| 319 | `ll_learned_pred_abs_error_sc_ens` | `learned_likelihood` | 0.0058% | 316.7 | 19.6/6.4/4.8 |  | exp111 L1 modelが推定したmulti-scale NCC ensembleの絶対誤差。 |
| 320 | `glead15` | `base_replay` | 0.0058% | 318.3 | 20.4/4.0/6.4 |  | raw GRを15行leadした値。 |
| 321 | `ll_candidate_tvt_hyb_minus_last_known_tvt` | `learned_likelihood` | 0.0058% | 318.3 | 13.8/5.6/9.6 | existing_delta_duplicate → keep `hyb_d` (r=1.0) | 元のBeam/NCC hybrid TVT − last_known_tvt。learned予測値ではない。 |
| 322 | `ll_candidate_tvt_std` | `learned_likelihood` | 0.0057% | 323.0 | 26.0/2.8/2.6 | high_corr_or_redundancy_review | exp111の5候補TVTの行別標準偏差。 |
| 323 | `grwr_raw_default_candidate_cost` | `gr_wavelet_rotation` | 0.0055% | 318.3 | 11.6/5.6/10.2 |  | raw GR面: default likPFのlocal GR observation cost。 |
| 324 | `grm5` | `base_replay` | 0.0053% | 317.0 | 12.8/7.0/7.0 |  | raw GRのcentered rolling-5平均。 |
| 325 | `ll_candidate_tvt_sc_ens_minus_last_known_tvt` | `learned_likelihood` | 0.0053% | 319.0 | 9.8/6.2/10.0 | existing_delta_duplicate → keep `sc_ens_d` (r=1.0) | 元のmulti-scale NCC ensemble TVT − last_known_tvt。learned予測値ではない。 |
| 326 | `grwr_dwt_detail_energy_w033` | `gr_wavelet_rotation` | 0.0051% | 330.7 | 26.2/1.4/1.2 |  | db4 level-3 DWT detailのdetail二乗平均（window 33）。 |
| 327 | `nsel_candidate_std` | `nested_selector` | 0.0050% | 322.3 | 15.4/5.6/5.2 |  | 11候補TVTの行別標準偏差。 |
| 328 | `grwr_savgol_31_p2_zero_minus_best_cost` | `gr_wavelet_rotation` | 0.0048% | 327.3 | 21.0/3.6/1.6 |  | savgol_31_p2 GR面: anchor固定候補cost − 全候補最良cost。 |
| 329 | `ll_learned_prob_top3_contains_likpf` | `learned_likelihood` | 0.0047% | 321.3 | 9.4/8.8/5.0 |  | exp111 within10確率面のtop-k、rank、margin、entropyまたは確率加重TVT要約。 |
| 330 | `grwr_raw_minus_rolling_absmean_w033` | `gr_wavelet_rotation` | 0.0046% | 329.3 | 21.4/2.0/2.0 |  | |raw GR − rolling denoised GR|のlocal平均（window 33）。 |
| 331 | `grwr_dwt_approx_zero_minus_best_cost` | `gr_wavelet_rotation` | 0.0045% | 326.7 | 18.0/2.2/4.2 |  | dwt_approx GR面: anchor固定候補cost − 全候補最良cost。 |
| 332 | `ll_learned_pred_abs_error_hyb` | `learned_likelihood` | 0.0044% | 328.0 | 17.4/2.2/4.2 |  | exp111 L1 modelが推定したBeam/NCC hybridの絶対誤差。 |
| 333 | `grwr_raw_minus_dwt_absmean_w065` | `gr_wavelet_rotation` | 0.0044% | 325.0 | 11.8/3.8/6.8 |  | |raw GR − dwt denoised GR|のlocal平均（window 65）。 |
| 334 | `ll_multiobs_score_pf_ancc` | `learned_likelihood` | 0.0040% | 328.3 | 13.4/2.8/5.0 |  | ANCC粒子フィルタのmulti-observation一致score。 |
| 335 | `grwr_rolling_median_11_zero_minus_best_cost` | `gr_wavelet_rotation` | 0.0039% | 333.0 | 16.6/1.4/3.4 |  | rolling_median_11 GR面: anchor固定候補cost − 全候補最良cost。 |
| 336 | `tdpf0` | `base_replay` | 0.0039% | 338.0 | 19.2/1.4/1.2 |  | raw GR − typewell GR(ANCC PF TVT +0 ft)。 |
| 337 | `ll_multiobs_score_beam_mean` | `learned_likelihood` | 0.0038% | 332.7 | 16.6/2.6/1.6 |  | 複数Beam path平均のmulti-observation一致score。 |
| 338 | `grwr_savgol_31_p2_default_candidate_ncc` | `gr_wavelet_rotation` | 0.0036% | 344.0 | 19.2/0.2/1.2 |  | savgol_31_p2 GR面: default likPFのlocal GR NCC。 |
| 339 | `grwr_dwt_detail_absmean_w033` | `gr_wavelet_rotation` | 0.0034% | 338.7 | 17.0/1.4/0.8 |  | db4 level-3 DWT detailのdetail絶対値平均（window 33）。 |
| 340 | `ll_multiobs_score_likpf_mean` | `learned_likelihood` | 0.0034% | 336.3 | 14.6/1.4/2.6 |  | likelihood-weighted PF平均のmulti-observation一致score。 |
| 341 | `grwr_raw_rolling_corr_w033` | `gr_wavelet_rotation` | 0.0033% | 346.7 | 18.0/0.2/0.8 |  | raw GRとrolling denoised GRのlocal相関（window 33）。 |
| 342 | `grwr_rolling_median_11_default_candidate_ncc` | `gr_wavelet_rotation` | 0.0029% | 342.0 | 14.2/0.8/1.2 |  | rolling_median_11 GR面: default likPFのlocal GR NCC。 |
| 343 | `sc25_d` | `base_replay` | 0.0028% | 344.3 | 14.0/0.8/1.0 |  | half-window 25のmulti-scale NCC候補 − anchor。 |
| 344 | `grwr_raw_dwt_corr_w033` | `gr_wavelet_rotation` | 0.0025% | 349.0 | 14.0/0.4/0.2 |  | raw GRとdwt denoised GRのlocal相関（window 33）。 |
| 345 | `grwr_raw_savgol_corr_w033` | `gr_wavelet_rotation` | 0.0024% | 357.0 | 13.4/0.0/0.6 |  | raw GRとsavgol denoised GRのlocal相関（window 33）。 |
| 346 | `glead5` | `base_replay` | 0.0022% | 343.3 | 8.6/0.6/2.8 |  | raw GRを5行leadした値。 |
| 347 | `grwr_raw_zero_minus_best_cost` | `gr_wavelet_rotation` | 0.0022% | 344.0 | 9.2/0.8/2.0 |  | raw GR面: anchor固定候補cost − 全候補最良cost。 |
| 348 | `grwr_savgol_31_p2_best_minus_default_cost` | `gr_wavelet_rotation` | 0.0022% | 350.3 | 11.0/1.2/0.2 |  | savgol_31_p2 GR面: 最良候補cost − default候補cost。 |
| 349 | `ll_learned_error_top3_contains_likpf` | `learned_likelihood` | 0.0020% | 337.3 | 3.8/2.8/3.4 |  | exp111 predicted-absolute-error面のtop-k、rank、marginまたは誤差逆重みTVT要約。 |
| 350 | `grwr_dwt_approx_best_minus_default_cost` | `gr_wavelet_rotation` | 0.0019% | 348.7 | 9.2/0.2/1.4 |  | dwt_approx GR面: 最良候補cost − default候補cost。 |
| 351 | `grwr_dwt_approx_default_candidate_ncc` | `gr_wavelet_rotation` | 0.0019% | 350.3 | 10.0/0.6/0.4 |  | dwt_approx GR面: default likPFのlocal GR NCC。 |
| 352 | `grs21` | `base_replay` | 0.0019% | 356.0 | 11.0/0.0/0.2 |  | raw GRのcentered rolling-21標準偏差。 |
| 353 | `glag5` | `base_replay` | 0.0019% | 345.3 | 8.2/1.4/0.8 |  | raw GRを5行lagした値。 |
| 354 | `grwr_rolling_median_11_best_minus_default_cost` | `gr_wavelet_rotation` | 0.0018% | 345.0 | 7.0/1.4/1.4 |  | rolling_median_11 GR面: 最良候補cost − default候補cost。 |
| 355 | `sc25_sc` | `base_replay` | 0.0014% | 370.0 | 7.4/0.4/0.0 |  | half-window 25 NCC matching score。 |
| 356 | `gr` | `base_replay` | 0.0012% | 344.3 | 2.8/1.6/1.6 |  | 水平井の補間済みraw GR。 |
| 357 | `grwr_raw_minus_dwt_absmean_w033` | `gr_wavelet_rotation` | 0.0011% | 352.7 | 5.0/0.4/0.8 |  | |raw GR − dwt denoised GR|のlocal平均（window 33）。 |
| 358 | `sc15_d` | `base_replay` | 0.0009% | 370.3 | 4.2/0.0/0.8 |  | half-window 15のmulti-scale NCC候補 − anchor。 |
| 359 | `grwr_raw_best_minus_default_cost` | `gr_wavelet_rotation` | 0.0009% | 357.0 | 4.2/0.2/0.4 |  | raw GR面: 最良候補cost − default候補cost。 |
| 360 | `glag1` | `base_replay` | 0.0008% | 349.3 | 1.4/1.2/1.4 |  | raw GRを1行lagした値。 |
| 361 | `glead1` | `base_replay` | 0.0008% | 349.0 | 1.2/1.6/1.0 |  | raw GRを1行leadした値。 |
| 362 | `ll_multiobs_ncc_pf_ancc` | `learned_likelihood` | 0.0007% | 373.0 | 4.0/0.0/0.2 |  | ANCC粒子フィルタのmulti-observation NCC。 |
| 363 | `grwr_raw_default_candidate_ncc` | `gr_wavelet_rotation` | 0.0005% | 372.7 | 3.2/0.0/0.0 |  | raw GR面: default likPFのlocal GR NCC。 |
| 364 | `ll_multiobs_ncc_beam_mean` | `learned_likelihood` | 0.0005% | 383.7 | 3.2/0.0/0.0 |  | 複数Beam path平均のmulti-observation NCC。 |
| 365 | `grwr_savgol_31_p2_zero_rank_norm` | `gr_wavelet_rotation` | 0.0004% | 369.7 | 2.0/0.0/0.4 |  | savgol_31_p2 GR面: anchor固定候補のcost順位（正規化）。 |
| 366 | `grwr_rolling_median_11_zero_rank_norm` | `gr_wavelet_rotation` | 0.0004% | 368.0 | 1.6/0.4/0.0 |  | rolling_median_11 GR面: anchor固定候補のcost順位（正規化）。 |
| 367 | `grwr_dwt_approx_candidate_cost_std` | `gr_wavelet_rotation` | 0.0002% | 365.7 | 1.2/0.2/0.0 |  | dwt_approx GR面: 候補costの標準偏差。 |
| 368 | `tdsc30` | `base_replay` | 0.0002% | 382.0 | 1.2/0.2/0.0 |  | raw GR − typewell GR(NCC ensemble TVT +30 ft)。 |
| 369 | `grwr_raw_candidate_cost_std` | `gr_wavelet_rotation` | 0.0002% | 374.0 | 1.4/0.0/0.0 |  | raw GR面: 候補costの標準偏差。 |
| 370 | `sc8_d` | `base_replay` | 0.0002% | 381.0 | 1.0/0.0/0.2 |  | half-window 8のmulti-scale NCC候補 − anchor。 |
| 371 | `ll_multiobs_ncc_likpf_mean` | `learned_likelihood` | 0.0002% | 387.7 | 1.2/0.0/0.0 |  | likelihood-weighted PF平均のmulti-observation NCC。 |
| 372 | `grwr_raw_zero_rank_norm` | `gr_wavelet_rotation` | 0.0002% | 368.3 | 0.6/0.2/0.2 |  | raw GR面: anchor固定候補のcost順位（正規化）。 |
| 373 | `grwr_rolling_median_11_candidate_cost_entropy` | `gr_wavelet_rotation` | 0.0002% | 378.0 | 1.0/0.0/0.0 |  | rolling_median_11 GR面: 候補cost分布のentropy。 |
| 374 | `ll_multiobs_mae_hyb` | `learned_likelihood` | 0.0002% | 386.0 | 1.0/0.0/0.0 |  | Beam/NCC hybridのmulti-observation GR MAE。 |
| 375 | `tdsc-8` | `base_replay` | 0.0002% | 397.7 | 1.0/0.0/0.0 |  | raw GR − typewell GR(NCC ensemble TVT -8 ft)。 |
| 376 | `tdsc2` | `base_replay` | 0.0002% | 400.0 | 1.0/0.0/0.0 |  | raw GR − typewell GR(NCC ensemble TVT +2 ft)。 |
| 377 | `grwr_dwt_approx_candidate_cost_entropy` | `gr_wavelet_rotation` | 0.0001% | 370.0 | 0.6/0.0/0.2 |  | dwt_approx GR面: 候補cost分布のentropy。 |
| 378 | `sc15_sc` | `base_replay` | 0.0001% | 385.0 | 0.6/0.0/0.2 |  | half-window 15 NCC matching score。 |
| 379 | `grs5` | `base_replay` | 0.0001% | 371.0 | 0.8/0.0/0.0 |  | raw GRのcentered rolling-5標準偏差。 |
| 380 | `grwr_savgol_31_p2_minus_raw_default_candidate_cost` | `gr_wavelet_rotation` | 0.0001% | 384.7 | 0.8/0.0/0.0 |  | savgol_31_p2面とraw面のdefault likPF cost差。 |
| 381 | `ll_multiobs_mae_sc_ens` | `learned_likelihood` | 0.0001% | 388.7 | 0.8/0.0/0.0 |  | multi-scale NCC ensembleのmulti-observation GR MAE。 |
| 382 | `tdsc-15` | `base_replay` | 0.0001% | 396.7 | 0.8/0.0/0.0 |  | raw GR − typewell GR(NCC ensemble TVT -15 ft)。 |
| 383 | `tdsc-4` | `base_replay` | 0.0001% | 399.0 | 0.8/0.0/0.0 |  | raw GR − typewell GR(NCC ensemble TVT -4 ft)。 |
| 384 | `tdsc15` | `base_replay` | 0.0001% | 401.3 | 0.8/0.0/0.0 |  | raw GR − typewell GR(NCC ensemble TVT +15 ft)。 |
| 385 | `grwr_savgol_31_p2_candidate_cost_entropy` | `gr_wavelet_rotation` | 0.0001% | 379.7 | 0.4/0.0/0.2 |  | savgol_31_p2 GR面: 候補cost分布のentropy。 |
| 386 | `grwr_dwt_approx_zero_rank_norm` | `gr_wavelet_rotation` | 0.0001% | 374.3 | 0.4/0.2/0.0 |  | dwt_approx GR面: anchor固定候補のcost順位（正規化）。 |
| 387 | `grwr_rolling_median_11_minus_raw_default_candidate_cost` | `gr_wavelet_rotation` | 0.0001% | 383.7 | 0.6/0.0/0.0 |  | rolling_median_11面とraw面のdefault likPF cost差。 |
| 388 | `grwr_rolling_median_11_minus_raw_default_candidate_ncc` | `gr_wavelet_rotation` | 0.0001% | 384.7 | 0.6/0.0/0.0 |  | rolling_median_11面とraw面のdefault likPF ncc差。 |
| 389 | `ll_multiobs_score_hyb` | `learned_likelihood` | 0.0001% | 395.0 | 0.6/0.0/0.0 |  | Beam/NCC hybridのmulti-observation一致score。 |
| 390 | `sc8_sc` | `base_replay` | 0.0001% | 398.3 | 0.6/0.0/0.0 |  | half-window 8 NCC matching score。 |
| 391 | `tdsc-30` | `base_replay` | 0.0001% | 401.3 | 0.6/0.0/0.0 |  | raw GR − typewell GR(NCC ensemble TVT -30 ft)。 |
| 392 | `grwr_savgol_31_p2_minus_raw_default_candidate_ncc` | `gr_wavelet_rotation` | 0.0001% | 390.3 | 0.4/0.0/0.0 |  | savgol_31_p2面とraw面のdefault likPF ncc差。 |
| 393 | `ll_multiobs_ncc_sc_ens` | `learned_likelihood` | 0.0001% | 396.7 | 0.4/0.0/0.0 |  | multi-scale NCC ensembleのmulti-observation NCC。 |
| 394 | `tdsc-2` | `base_replay` | 0.0001% | 402.3 | 0.4/0.0/0.0 |  | raw GR − typewell GR(NCC ensemble TVT -2 ft)。 |
| 395 | `tdsc0` | `base_replay` | 0.0001% | 405.3 | 0.4/0.0/0.0 |  | raw GR − typewell GR(NCC ensemble TVT +0 ft)。 |
| 396 | `grwr_raw_candidate_cost_entropy` | `gr_wavelet_rotation` | 0.0000% | 383.3 | 0.2/0.0/0.0 |  | raw GR面: 候補cost分布のentropy。 |
| 397 | `grwr_rolling_median_11_candidate_cost_std` | `gr_wavelet_rotation` | 0.0000% | 387.3 | 0.2/0.0/0.0 |  | rolling_median_11 GR面: 候補costの標準偏差。 |
| 398 | `grwr_savgol_31_p2_candidate_cost_std` | `gr_wavelet_rotation` | 0.0000% | 391.0 | 0.2/0.0/0.0 |  | savgol_31_p2 GR面: 候補costの標準偏差。 |
| 399 | `ll_multiobs_score_sc_ens` | `learned_likelihood` | 0.0000% | 400.0 | 0.2/0.0/0.0 |  | multi-scale NCC ensembleのmulti-observation一致score。 |
| 400 | `tdsc4` | `base_replay` | 0.0000% | 409.3 | 0.2/0.0/0.0 |  | raw GR − typewell GR(NCC ensemble TVT +4 ft)。 |
| 401 | `tdsc8` | `base_replay` | 0.0000% | 410.3 | 0.2/0.0/0.0 |  | raw GR − typewell GR(NCC ensemble TVT +8 ft)。 |
| 402 | `gr_d1` | `base_replay` | 0.0000% | 378.0 | 0.0/0.0/0.0 | all models zero split | raw GRの1階行差分。 |
| 403 | `gr_d2` | `base_replay` | 0.0000% | 379.0 | 0.0/0.0/0.0 | all models zero split | raw GRの2階行差分。 |
| 404 | `grwr_dwt_approx_best_is_default_candidate` | `gr_wavelet_rotation` | 0.0000% | 381.0 | 0.0/0.0/0.0 | all models zero split | dwt_approx GR面: 最良cost候補がdefault likPFかのflag。 |
| 405 | `grwr_dwt_approx_minus_raw_default_candidate_cost` | `gr_wavelet_rotation` | 0.0000% | 382.7 | 0.0/0.0/0.0 | all models zero split | dwt_approx面とraw面のdefault likPF cost差。 |
| 406 | `grwr_dwt_approx_minus_raw_default_candidate_ncc` | `gr_wavelet_rotation` | 0.0000% | 383.7 | 0.0/0.0/0.0 | all models zero split | dwt_approx面とraw面のdefault likPF ncc差。 |
| 407 | `grwr_dwt_minus_raw_ncc_gap_x_candidate_range` | `gr_wavelet_rotation` | 0.0000% | 385.0 | 0.0/0.0/0.0 | all models zero split | DWT-vs-raw default NCC差 × 候補TVT range。 |
| 408 | `grwr_dwt_minus_raw_ncc_gap_x_dwt_energy_ratio_w065` | `gr_wavelet_rotation` | 0.0000% | 386.0 | 0.0/0.0/0.0 | all models zero split | DWT-vs-raw default NCC差 × DWT detail energy比(w65)。 |
| 409 | `grwr_raw_best_is_default_candidate` | `gr_wavelet_rotation` | 0.0000% | 387.0 | 0.0/0.0/0.0 | all models zero split | raw GR面: 最良cost候補がdefault likPFかのflag。 |
| 410 | `grwr_rolling_median_11_best_is_default_candidate` | `gr_wavelet_rotation` | 0.0000% | 390.3 | 0.0/0.0/0.0 | all models zero split | rolling_median_11 GR面: 最良cost候補がdefault likPFかのflag。 |
| 411 | `grwr_savgol_31_p2_best_is_default_candidate` | `gr_wavelet_rotation` | 0.0000% | 394.3 | 0.0/0.0/0.0 | all models zero split | savgol_31_p2 GR面: 最良cost候補がdefault likPFかのflag。 |
| 412 | `grwr_typewell_gr_missing_rate` | `gr_wavelet_rotation` | 0.0000% | 398.0 | 0.0/0.0/0.0 | all models zero split | typewell GRのwell内欠損率。 |
| 413 | `ll_candidate_tvt_likpf_mean_minus_likpf_mean_tvt` | `learned_likelihood` | 0.0000% | 399.0 | 0.0/0.0/0.0 | constant_zero; all models zero split | 元のlikelihood-weighted PF平均 TVT − likpf_mean_tvt。learned予測値ではない。 |
| 414 | `ll_multiobs_ncc_hyb` | `learned_likelihood` | 0.0000% | 402.0 | 0.0/0.0/0.0 | all models zero split | Beam/NCC hybridのmulti-observation NCC。 |
| 415 | `sc_trust` | `base_replay` | 0.0000% | 408.0 | 0.0/0.0/0.0 | constant; all models zero split | 既知prefix長から作るNCC trust。exp238 train面では定数。 |

## Sources

- `experiments/exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218/{config.yaml,result.md,metrics.json}`
- Kaggle output `kentookumura/exp238-nested-rank-slot-exp218-train` v5 feature importance artifact。
- Kaggle output `kentookumura/exp237-hmm-exp226-candidate-selector-exp183-train` v1 candidate readout / residual correlation。
- `docs/surveys/exp148_exp092_feature_replacement_audit_20260704.md`と`studies/feature_replacement_audit/outputs/`。
- `experiments/exp198_exact_replacement_prune_on_exp148/result.md`。
- `experiments/exp252_pf_seed_medoid_selectability_audit/result.md`。
- `experiments/exp019_pf_beam_candidate_quality_audit/result.md`、`exp091_self_gr_likelihood_pf_beam_probe/result.md`、`exp093_pf_candidate_coverage_then_ranker_audit/result.md`。
- `experiments/exp128_trajectory_local_typewell_self_gr_switch_audit/result.md`、`exp134_self_gr_multiscale_longtail_gate/result.md`。
- `experiments/exp103_pf_z_xy_likpf_ensemble_parity/result.md`、`exp106_strict_exp072_pf_z_multiseed_scale_cache/result.md`。
- `experiments/exp142_trajectory_aware_pf_transition_prior/result.md`、`exp173_beam_topk_path_posterior_audit/result.md`、`exp177_beam_topk_bimodal_gate_posthoc_audit/result.md`。
- `experiments/exp205_exact_hmm_smoother_exp072_compatible_cache_audit/result.md`、`exp221_lgb_oof_gaussian_emission_hmm_on_exp148/result.md`、`exp223_joint_typewell_self_gr_hmm_likelihood_probe/result.md`、`exp225_state_known_tvt_self_gr_hmm_emission/result.md`。
- `experiments/exp243_pf_seed_medoids/result.md`、`experiments/exp251_raw_test_safe_dual_objective_candidate_ranker/result.md`。
- `experiment_summary.md`のexp202–215 heatmap route close記録。
- exp237 OOF chunked Beam marginal readout。source gzip SHA256 `c5d94361c2582f3f2e419ff70e8f87c1e4d3613b4cc21981e11f009f956d66c9`。
- `experiments/exp255_nested_selector_gated_bounded_direct_readout_on_exp238/result.md`。
- `experiments/exp257_nested_selector_output_replacement_only_on_exp218/result.md`。
