# exp300 exp264 vs exp274 well / selector readout 結果

## 仮説

exp264のexp274比悪化はlong-tail深部の一部wellへ集中し、target-free well品質、予測不一致、またはselectorのcandidate family / margin / switchに偏りがある。

## 結論

exp264 corrected Stage D v3は、前のsubmitted ML anchor exp274 raw CatBoost OOFよりpooled RMSEが`+0.277308 ft`悪く、387/773 wellで悪化した。悪化はlong-tail深部と少数wellへ集中する。

依頼への直接回答は、**主因はStage C selectorの候補ranking失敗であり、Stage Dの過剰反応ではない**、である。`>3 ft`悪化73 wellでは候補集合内のoracle最良候補RMSEが`5.2626`なのに、selectorが選んだhard候補は`17.1663`まで悪化した。Stage D finalは`16.6532`へ緩和している。73 well中67 wellでselected hard候補が既にexp274より悪く、集約MSE分解でもselector ranking regret `+266.9864`が支配項、Stage D効果は`-17.3527`で改善側だった。

前節で調べた時系列の候補切替は主因ではない。問題は「候補が切り替わった瞬間」ではなく、正しい候補が候補集合内に存在するのに、selectorが別候補を継続的に高くrankしたことである。

ただしexp264の最終予測はhard selectorではない。corrected Stage C v6のscore/rank/marginを74 compact特徴へ縮約し、corrected Stage D v3 downstream LightGBMへadd-onlyした予測である。以下のdominant候補・margin・switchはscore landscapeの診断であり、hard routingの因果効果ではない。

## 比較契約

- source: `exp264_exp263_candidate_confidence_dual_selector`
- comparison: `exp274_catboost_final_regressor_swap_on_exp238`
- row: 3,783,989
- well: 773
- metric: tail rows上のRMSE
- exp264 surface: `corrected_stage_d_v3_selector_compact_addonly_lgb_mean`
- selector surface: `corrected_stage_c_v6_strict_nested_outer_valid`
- candidate-long source: `nested_outer_valid_candidate_score.parquet`、45,407,868 rows、SHA `a10b7848...abc`
- exp274 surface: raw CatBoost OOF
- outer fold: 710 well一致、63 well不一致
- model / candidate generation / inference / submission: 0 / 0 / 0 / 0

## OOF比較

| 指標 | exp274 | exp264 | 差（exp264 - exp274） |
| --- | ---: | ---: | ---: |
| pooled RMSE | 8.183504 | 8.460811 | +0.277308 |
| matched-fold 710 wells | 8.285073 | 8.564503 | +0.279430 |
| mismatched-fold 63 wells | 6.897718 | 7.149798 | +0.252080 |

悪化well数は以下。

| 条件 | well数 |
| --- | ---: |
| `delta > 0` | 387 |
| `delta > 0.25 ft` | 330 |
| `delta > 1 ft` | 194 |
| `delta > 3 ft` | 73 |
| `delta > 5 ft` | 40 |
| 改善 | 386 |

worst 12 wellは以下。

| well | exp274 | exp264 | delta |
| --- | ---: | ---: | ---: |
| 81bf5923 | 22.2747 | 39.5183 | +17.2436 |
| ee0300f7 | 11.7840 | 25.2406 | +13.4566 |
| add9c322 | 4.8205 | 17.4047 | +12.5842 |
| 4c2208f5 | 6.8924 | 18.6312 | +11.7389 |
| 4caa7289 | 15.4426 | 26.8896 | +11.4471 |
| 57f05c51 | 11.6012 | 21.4253 | +9.8241 |
| 11d0f5ac | 10.4079 | 19.9913 | +9.5834 |
| c8d9680c | 9.7871 | 19.2015 | +9.4144 |
| 70925e23 | 16.9679 | 26.3084 | +9.3405 |
| 91db7070 | 32.7389 | 41.5974 | +8.8586 |
| ef8e3ed0 | 10.8642 | 19.7168 | +8.8526 |
| b3388334 | 18.4246 | 26.9088 | +8.4841 |

正のSSE増分は上位7 / 22 / 40 wellで`20.1% / 44.9% / 60.1%`を占め、少数wellへの集中がある。

## 悪化wellの特徴

### 1. long-tail深部

distance `0--1000 ft`の全bucketはexp264が改善し、`1000+`だけが悪化した。

| distance | delta RMSE |
| --- | ---: |
| 0--50 | -0.263582 |
| 50--100 | -0.214435 |
| 100--250 | -0.113385 |
| 250--500 | -0.045444 |
| 500--1000 | -0.123097 |
| 1000+ | +0.327833 |

relative-tail q9/q10は`+0.598962 / +0.710069 ft`で、全体悪化の中心は末尾20%にある。well長そのものはmaterial regressionを分けず、`>5 ft`群のtail rows中央値は他群とほぼ同じだった。

### 2. GR欠損ではない

GR observed / missingのdeltaは`+0.327715 / +0.164605 ft`で、欠損rowだけが原因ではない。tail GR missing fraction、変動、lag correlationも`>3 ft`群を有意に分離しなかった。

### 3. target-free raw特徴は弱い

raw trajectory、GR、typewell、formation距離を使った補助logistic AUCは、`delta > 3 ft`で`0.495675`。`>3 ft`群のdeployable特徴は多重検定後に有意なものがなく、単純な事前well routerは支持されない。

oracleでは`>3 ft`群のtail TVT range中央値が`32.81 ft`、その他が`25.79 ft`で差が出た。これは真値を使う原因診断であり、推論時routerには使えない。

### 4. exp264とexp274の予測不一致は強いposthoc信号

well内prediction-pair RMSE中央値は`>3 ft`群`7.709`、その他`2.272`。単変量AUCは`0.946888`、`>5 ft`では`0.973874`だった。予測自体から計算可能なtarget-free信号だが、同じOOFで発見したposthoc結果なので、outer-trainだけでruleを固定する再検証なしにgateへ使わない。

## selectorでの選ばれ方

### 直接回答: 誤候補選択かStage D応答か

primary 11候補について、`pred_abs_error`最小をselector選択、`actual_abs_error`最小をoracle選択とし、同じrowで次の恒等式へ分解した。

`Stage D − exp274 = (oracle候補 − exp274) + (selector選択 − oracle候補) + (Stage D − selector選択)`

| `>3 ft`悪化73 well | RMSE | exp274比または前段比MSE |
| --- | ---: | ---: |
| exp274 | 10.7964 | — |
| oracle最良候補 | 5.2626 | `-88.8676` vs exp274 |
| selector選択hard候補 | 17.1663 | `+266.9864` selection regret vs oracle |
| Stage D final | 16.6532 | `-17.3527` vs selected hard |
| Stage D finalのexp274比 | — | `+160.7660` |

したがって、候補集合にはexp274より大幅に良い候補が存在したが、selectorがそれを選べずranking regretを生んだ。Stage Dは集約ではselected hard pathを少し改善しており、全体悪化を作った主因ではない。

tieを許容したoracle正解率は`>3 ft`群で`7.92%`、その他で`12.98%`だった。特に`1000+ ft`の悪化群では`6.56%`まで低下し、平均absolute regretは`11.22 ft`だった。悪化73 wellの内訳は以下。

| 判定 | well数 |
| --- | ---: |
| selected hardがexp274より悪い | 67 |
| そのうちStage Dが緩和 | 41 |
| そのうちStage Dも追加悪化 | 26 |
| selected hardはexp274より良いがStage Dが悪化させた | 6 |

selection regretの`52.3%`は、oracle最良候補が`exp226_k16`なのに別候補を選んだrowから発生した。最大の誤ranking pairは`Self-GR/LikPF`選択に対してoracleが`exp226_k16`で、悪化群のselection-regret SSE `16.97M`だった。つまりBeam/LikPF系が多いという先の相関より具体的に、深部でK16が正解のrowをSelf-GR/LikPF、LikPF、PF ANCC、Beamなどへ誤rankingすることが中心だった。

`Self-GR/LikPF`を選び、oracleがK16だったpairは、全体で42,302行、`>3 ft`悪化群で13,331行だった。悪化群ではSelf-GR/LikPFのMAE / RMSEが`34.146 / 36.891 ft`、同じrowのK16が`6.082 / 9.363 ft`で、MAE差`+28.064 ft`、selection-regret MSE `+1273.287`だった。全体でもSelf-GR/LikPF `16.601 / 22.714 ft`に対してK16 `3.427 / 6.331 ft`である。

Beamは`>3 ft`群360,510行のうち19,095行で選ばれ、18,947行がoracle非Beam、正解は148行だけだった。Beam誤選択率は全row比`5.256%`で、その他群の`2.523%`の`2.083x`。その他群と同率なら期待9,096行なので、母数調整後のexcessは約9,851行だった。Beam選択内の誤選択率も`99.22%`で、その他群`90.34%`より`+8.88 percentage points`高かった。

個別には`81bf5923`はselected hard `29.744`がexp274 `22.275`より既に悪く、Stage Dがさらに`39.518`へ悪化させたため両方が原因だった。一方`57f05c51`など6 wellではselected hardはexp274より良く、Stage Dだけが最終悪化を作った。この例外はselector ranking失敗とは分けて扱う。

### 直接検証: 候補切替が悪化を生んだか

`switch`はcorrected Stage C v6 primary `pred_abs_error` hard top1が同じwellの直前rowから変わったrowと定義した。exp274にはselectorがないため、「exp274から候補が変わった」とは定義できない。ここではexp264内の時系列切替と、Stage D finalのexp274比悪化の重なりを測った。

| 対象 | 切替近傍 | row share | 正のSSE悪化share | net SSE悪化share |
| --- | ---: | ---: | ---: | ---: |
| 全773 well | 切替rowのみ | 5.42% | 3.19% | 5.41% |
| 全773 well | ±5 row | 26.26% | 16.25% | 27.84% |
| `>3 ft`悪化73 well | 切替rowのみ | 3.95% | 2.75% | 2.75% |
| `>3 ft`悪化73 well | ±5 row | 19.85% | 14.26% | 14.20% |

`>3 ft`群では切替±5行の外側がrowの`80.15%`、正のSSE悪化の`85.74%`を占めた。したがって切替境界で予測が壊れる現象は主因ではなく、悪い区間は候補が安定した後も持続している。

切替後の全runについて、新しく選ばれたhard候補と「直前runの候補を現在run全体で維持」のactual-TVT SSEを比較した。

| 対象 | switch run | 新候補が悪いrun | 新候補hard RMSE | 前候補維持RMSE | 新−前 MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全体 | 205,264 | 94,164 (45.87%) | 8.6663 | 8.9287 | -4.6181 |
| `>3 ft`群のwell集約 | 14,249 | 36/73 wellでnet悪化 | — | — | 中央値 -0.0749 |

globalでは切替後の新候補の方が前候補維持より良く、候補切替を一律に抑えるとhard pathは悪化する。`>3 ft`群でも、hard path上で有害だったswitch-runはrowの`46.56%`・Stage D finalの正の悪化量の`49.18%`、有益だったswitch-runはrowの`52.84%`・正の悪化量の`50.80%`だった。Stage D finalの悪化は両側へほぼ均等に存在し、hard候補の切替成否と連動していない。

個別wellでは例外がある。`91db7070`、`4ed93db6`、`368131f9`、`67847288`の4 wellだけが、(1) 前候補維持より新候補のhard pathが悪い、かつ (2) 切替±5行の正の悪化shareがrow shareを上回る、の両方を満たした。これは切替寄与と整合する候補であり、とくに`91db7070`はdelta `+8.8586 ft`、新−前MSE `+111.59`、±5行のrow/悪化share `17.60% / 22.06%`だった。一方worstの`81bf5923`は新候補が前候補維持よりMSE `-129.18`改善し、±5行の悪化shareも`3.37%`に過ぎず、切替原因説と反対だった。

### global row-level hard top1分布

primary `pred_abs_error` domainの上位はSelf-GR HMM `17.82%`、PF ANCC `17.31%`、K16/LikPF `11.20%`、Exact HMM `9.22%`、K16/Exact-HMM `8.35%`。Beamは全体では`3.03%`と少ない。

### well dominant候補別の悪化

`dominant`は各wellでhard top1になったrowが最も多い候補を指す。

| dominant候補 | wells | `>3 ft`率 | well delta平均 |
| --- | ---: | ---: | ---: |
| Beam mean | 20 | 25.0% | +2.0143 |
| Likelihood PF mean | 36 | 19.4% | +1.2828 |
| K16 / Self-GR | 31 | 19.4% | +0.9874 |
| Self-GR / LikPF | 33 | 24.2% | +0.2954 |
| PF ANCC | 169 | 8.9% | +0.2365 |
| K16 / LikPF | 109 | 4.6% | +0.1272 |
| Self-GR HMM | 200 | 4.0% | -0.0815 |
| K16 / Exact-HMM | 55 | 7.3% | -0.3762 |

`>3 ft`悪化wellでのdominant候補prevalence liftはBeam `3.20x`、Self-GR/LikPF `3.07x`、LikPF `2.31x`、K16/Self-GR `2.30x`。反対にSelf-GR単独`0.40x`、K16/LikPF `0.46x`だった。これはwell-level prevalenceであり、悪化群row-level selection shareではない。

### material regressionは低confidenceではない

| selector指標 | `>3 ft`群中央値 | その他中央値 | p値 |
| --- | ---: | ---: | ---: |
| primary error margin mean | 0.480424 | 0.200106 | 6.62e-12 |
| probability margin mean | 0.016270 | 0.003688 | 4.03e-16 |
| dominant share | 0.380502 | 0.347855 | 1.66e-3 |
| switches / 1000 rows | 35.846954 | 53.003510 | 2.23e-7 |

悪化wellはmarginが大きく、dominant shareが高く、switchが少ない。これは「候補間で頻繁に迷って切り替わるwell」が悪化したという説明と反対である。直接検証でも切替近傍への悪化集中はなく、主パターンは候補が安定した区間でStage D finalのexp274比悪化が持続する形だった。

worst例では`81bf5923`がSelf-GR/LikPF dominant share `62.7%`、`add9c322`がBeam `84.9%`、`4caa7289`がLikPF `72.8%`。一方`ee0300f7`はSelf-GR dominantだがswitch `121.5/1000`で、全worst wellが同じ形ではない。

### Stage Dはhard top1をそのまま使わない

globalではhard primary top1 `8.652532`に対しStage D final `8.460811`で、downstreamは平均的にはhard top1より良い。`>3--5 / >5 ft`群でもhard-minus-final RMSEは`+0.7489 / +0.4103 ft`でfinalが緩和しているが、exp274まで戻せていない。個別にはfinalがhard top1より大幅に悪いwellもあり、candidate選択だけでなくcompact特徴に対するdownstream応答もriskの一部である。

## 解釈

exp264の悪化の主因は、long-tail深部の一部wellでStage C selectorが候補集合内のoracle良候補をrankできず、誤候補を高confidenceかつ持続的に選んだことだった。時系列switchは主因ではなく、Stage Dも集約上はselected hard pathを緩和した。ただしStage Dが追加悪化させた32/73 well、特にselected hard自体はexp274より良かった6 wellは別のdownstream failureである。oracle rankingはtruthを使う診断なので、そのままcandidate routingへ利用せず、fold-safeなselector score改善または既存risk監査で再検証する必要がある。

## non-use contractと次

- exp274はsubmitted anchorでもtrain-side rejected。比較結果を新CV anchorにしない。
- exp300はposthoc診断。switch suppression、candidate除外、hard fallback、threshold/weight grid、guard緩和を承認しない。
- oracle候補はactual TVTを使う候補集合内上限であり、そのIDや誤ranking labelをdeployable routingへ使わない。
- target/oracle特徴や悪化labelをdeployable featureへ戻さない。
- 次は既存の`exp276_corrected_exp264_parent_revalidation`で、事前固定済みtarget-free risk familyが今回のregimeをouter-foldで再現するかだけを0-boosterで監査する。
