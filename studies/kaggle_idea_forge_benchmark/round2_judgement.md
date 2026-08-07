# Blind benchmark round 2 判定

- 判定日: 2026-08-07
- 対象: `treatment_v2_run1.json`、`treatment_v2_run2.json`
- 正解側資料: 指定された ROGII 1st / 6th / 7th / 9th / 10th / 11th / 13th / 14th place の一次 archive 8件
- 主採点: 各 run の `portfolio` に入った top 5
- 副採点: 各 run の `idea_cards` 全12件

## 結論

両 run とも、一次 archive を見せない packet だけから、上位解法で実際に効いた機構を実験可能な形で再発見した出力と判定する。主採点は両 run とも **15/16**、副採点は両 run とも **16/16** で同点である。

round 1 の最大欠落だった G7 は、両 run の top 5 にある `I03` で明確に埋まった。これは単なる input noise や residual stacking ではない。実 OOF の誤差振幅・自己相関・bias・欠損・rank/mode inversion を測り、それに似せた壊れた first-pass conditioning を生成し、pretrain、real OOF-like conditioning で短く fine-tune、未見 group で評価する一連の案になっている。一次 archive の 7th place にある「fabricated mistakes で trust を先に学習させる second-pass refiner」と同じ中心機構であり、両 run とも G7 は 2 点である。

top 5 で唯一満点でない G6 は、`I03` が conditioning corruption 中に row order・座標・prefix・raw observation を保存しているため部分一致の 1 点とした。ただし、その主目的は G7 であり、一次 archive の Z-shift、path-relief / thickness warp、`U = TVT + dZ` donor synthesis のように domain invariant を使って target・trajectory・入力を整合的に再生成する独立案ではない。全12では run1 `I12`、run2 `I08` がこの不足を具体的に補い、G6 は2点になる。

## 採点方法

各 G は、対象範囲内にある最も強い単一 card を基準に 0 / 1 / 2 点とした。

- 0: 該当機構がない。
- 1: 抽象的な言及、機構の一部分、または近接機構にとどまる。
- 2: input、target/objective、output/decode または downstream role、反証可能な test / kill criterion が一つの card 内で接続されている。

別 card の断片をつないで2点にはしていない。同じ card が複数の G を満たす場合も、それぞれについて閉じた因果案になっている場合だけ対応を認めた。oracle coverage と truth-free selectability は別物として扱った。

## 一次 archive から固定した機構

| G | 上位解法で確認した中心機構 |
|---|---|
| G1 | 1st / 9th / 13th place は、well 全体を candidate TVT × MD の確率場・cost volume として扱い、expected path、whole-well trajectory bank、または posterior-normalized coherent decode に落とした。rowwise point regressionの言い換えではない。 |
| G2 | 1st place は PF の2D probability heatmapを U-Net channelとして使い、9th place は HMM、beam、U-Net の posterior を point 化前に積で融合した。弱い確率出力を最終 path として直出しせず、次段の evidence にした。 |
| G3 | 11th place は既知 prefix の GR–TVT を same-well reference として matching image / PF に使い、その TVT coverage 外では typewell に戻した。6th / 9th place にも selfGR / public heel reference と coverage 補完がある。 |
| G4 | 6th place は typewell、self、neighbor、GR-neighbor、self-graft、physical prior の有無、PF dynamics / smoothing の違いから error の向きが異なる候補を作り、91 trajectory を broad soft average した。seed違いだけの多様性や hard top-1 ではない。 |
| G5 | 7th place は2 refiner の disagreement を gate にし、平均では無価値な neighbour 等の signal を不確実な箇所だけで使った。10th place にも model disagreement による per-row blend gate がある。 |
| G6 | 1st place は `TVT + Z` を保つ Z-shift と GR transform、9th place は path-relief / monotone thickness warp 後に cost channel を再計算、13th place は donor の `U = TVT + dZ` だけを借りて TVT を再構成する physically consistent synthetic pretraining を使った。 |
| G7 | 7th place は first-pass を extra channel にする second-pass U-Net に対し、実誤差分布に似た low-pass random-walk error を持つ conditioning を synthetic wells 上で捏造して pretrainし、短い real fine-tune 後に refiner として使った。copy shortcut を壊して trust を学ぶ機構である。 |
| G8 | 6th place は PF の GPU 化で約200倍高速化し、それまで非現実的だった多数 trial と long-lag / full smoothing を解禁した。7th place は hidden が3-well stubより大きいことを前提に seconds-per-well を管理し、exact-parity numba HMM、CPU/GPU overlap、chunking等で9時間 wallを通した。 |

## スコア

| 対象 | G1 | G2 | G3 | G4 | G5 | G6 | G7 | G8 | 合計 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| run1 top 5 | 2 | 2 | 2 | 2 | 2 | 1 | 2 | 2 | **15/16** |
| run1 全12 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | **16/16** |
| run2 top 5 | 2 | 2 | 2 | 2 | 2 | 1 | 2 | 2 | **15/16** |
| run2 全12 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | **16/16** |

## run1 の card 対応

| G | top 5 での対応 | 全12での補強 | 判定理由 |
|---|---|---|---|
| G1 | `I01 Whole-well increment posterior with joint decoding` | — | whole-well sequenceから increment/residual bins と transition scores を出し、prefix境界付きDPで一つの coherent path に decode する。posterior support、entropy、tail/worst-well testまである。 |
| G2 | `I04 Abstaining soft posterior over diverse candidate union` | `I07 Heatmap logits as local factors in a global decoder` | top 5でも heatmap candidates、logits/ranks/entropy を direct path ではなく mixture evidence として使う。`I07` はさらに明確で、full logit surface を bounded local factor として global HMM/DP に入れ、非coverage行では neutral factor にする。 |
| G3 | `I02 Coverage-aware same-well calibration reference` | — | prefix GR/TVTを entity-specific calibration reference と reliability に変え、coverage内だけ correction、短prefix・範囲外・曖昧matchでは global referenceへ戻す。 |
| G4 | `I04` | `I11 Exchangeable diverse trajectory set predictor` | 既存whole-well、heatmap、HMM/PF、ML anchorという異なるsourceの候補を、disagreementを保ったsoft posteriorで融合する。`I11` は observation / reference / decoder 軸の set generation と別scorerを追加する。 |
| G5 | `I04` | `I08 Persistent-error risk calibrator for conditional anchor blending` | `I04` は posterior spread / disagreement で anchor へ abstain するため、弱いcandidate unionを不確実性条件付きで使う閉じた案である。`I08` は persistent anchor failure risk のときだけ physical posteriorへ bounded blendする、より直接的な案。 |
| G6 | `I03`（部分） | `I12 Invariant-preserving observation-shift augmentation` | `I03` は raw input invariantsを保存した conditioning corruptionなので1点。`I12` は GR amplitude/offset/dropout/mismatchだけを壊し、TVT・座標・row order・prefix境界を保ち、派生 feature を再計算する明示的augmentationなので全12では2点。 |
| G7 | `I03 OOF-shaped corrupted-candidate refiner` | — | OOF誤差過程を測定し、clean pseudo-candidateを joint にcorruptしてpretrain、real OOF-like bundleでfine-tune、outer held groupでcopy-through・corruption mismatch・runtimeをkillする。5要素が一枚に閉じている。 |
| G8 | `I05 Paired-parity batched state-space engine` | — | exact recurrenceのemission cache、ragged batching、deterministic reductionで E06 の10h50m pathを高速化し、paired control parityと、解禁される `I02` の end-to-end OOF accuracyを採用条件にしている。単なる速度 benchmark ではない。 |

## run2 の card 対応

| G | top 5 での対応 | 全12での補強 | 判定理由 |
|---|---|---|---|
| G1 | `I01 Joint tail increment distribution with constrained decode` | `I05 Low-rank spline residual field around the ML anchor` | mode weights と increment paths を complete suffix object として出し、expected accumulated pathへ coherent decodeする。input mask、accumulation consistency、unsupported-length fallback、tail testまで具体化されている。 |
| G2 | `I04 Nested soft posterior over heterogeneous candidate support` | `I06 Global K-best path decode with heatmaps as observation potentials` | top 5でも heatmap logits/uncertaintyを heterogeneous support の evidence として扱う。`I06` は local logitsを full-tail latticeのobservation potentialにし、K-best global pathsまで point化を遅らせるため一次 archiveとの一致がさらに強い。 |
| G3 | `I02 Prefix-conditioned entity-specific typewell calibration` | — | prefix-only calibration domainとreliabilityを作り、support内でだけ calibrated reference、外では immutable global typewell fallbackを使う。 |
| G4 | `I04` | `I12 Role-diverse leave-one-mechanism-out candidate factory` | 既存PF/beam、heatmap、ML anchor、HMM/PF uncertaintyをsoft conditional meanとabstentionで融合する。`I12` は observation / reference / representation / decoderごとの候補を作り、leave-one-mechanism-outで diversity を監査する。 |
| G5 | `I04` | `I11 Candidate disagreement as an abstention signal for anchor correction` | top 5では disagreement、coverage、score temperatureから abstentionを出す。`I11` は candidateを直接出力せず、cross-family disagreementが成立する範囲だけ E01 の clipped correctionを許すため、7th / 10th placeのgateにより直接対応する。 |
| G6 | `I03`（部分） | `I08 EDA-gated invariant-preserving trajectory corruption` | `I03` のraw invariant保持はG7付随なので1点。`I08` はまず座標increment・TVT increment・prefix boundaryの関係をEDAで反証し、支持された関係だけを保存しながら geometry/GR/availabilityを変え、全derived targetを再計算するため2点。 |
| G7 | `I03 OOF-error-matched corrupted-conditioning refiner` | — | outer-train OOFから amplitude、lag correlation、bias、missingness、rank inversionを推定し、corruption-pretrain、disjoint real-OOF fine-tune、outer held evaluationを行う。clean-candidate対照とdeliberate inversionのno-copy testまである。 |
| G8 | `I10 Parity-locked batched observation cache and K-best engine` | — | repeated observation evaluationをcacheし、bounded-memory exact K-best DPへ変え、path/score parity、3倍速度、hidden-unit scaling、`I06` または `I02` のend-to-end accuracy runを採用条件にしている。 |

## G7 が本当に埋まったか

**埋まった。両 run とも2点。** 判定に必要な閉ループは次の通りである。

1. deployment時の first-pass errorを OOF から測る。
2. clean / truth-derived intermediate をその error processで壊す。
3. 壊れた conditioning から真値を復元する refiner をpretrainする。
4. real OOF-like conditioningで短くfine-tuneする。
5. 完全に未見のwell groupで評価し、copy-through、corruption mismatch、tail悪化、runtimeをkill条件にする。

run1 `I03` は全5段を満たす。run2 `I03` も全5段を満たし、outer-trainだけからcorruption統計を作ること、real OOF fine-tuneをdisjoint subsetにすること、意図的rank inversionに対するcopy率まで明記しており、leakage防止はrun2の方がわずかに明文化が強い。ただし mechanism recall の点数差を付けるほどではない。

一次 archive の7th placeは synthetic well 上で low-pass random-walk errorを first-pass conditioningへ足し、60 epoch pretrain後に8 epoch real fine-tuneした。両候補はその固有数値や実装をコピーしていない一方、必要な因果機構は再現している。したがって「corruption」という単語一致ではなく、source-hiddenでの独立な mechanism recovery と判定する。

## 安全性・leakage監査

### top 5

両 run とも明白な unsafe / leakage card はない。

- selectors、reliability、fusion、refinerは whole-well outer / nested OOFを要求している。
- held well の truth を oracleとしてdeployment decisionへ使っていない。
- formation-only列、public 3-well ID、保存済みpublic predictionをhidden inferenceへ持ち込まない。
- dynamic well enumeration、runtime sample IDの1対1整列、candidate/context欠損時fallback、offline determinismが明記されている。
- G7 corruption統計は outer-train OOF だけから推定し、held groupを汚さない設計である。
- G8 は数値/path parityを先に要求し、速度だけで accuracy mechanismを採用しない。

### 全12で残る実装時の注意

| run | card | リスク | 必要な固定 |
|---|---|---|---|
| run1 | `I11` | candidate set / diversity objectiveを全OOFのoracle改善を見て調整し、その同じOOFで別scorerのgainを報告すると meta-selection optimism が入る。 | generator family・K・diversity thresholdの選択も inner foldsに閉じ、未使用outer wellsで generator coverage と selector gainを一度だけ判定する。 |
| run2 | `I06` | cached logits上でK-best設計を truth coverage に合わせて繰り返し直すと、そのfoldのoracleへ過適合し得る。 | lattice、K、transition、calibrationをouter-trainでfreezeし、outer-heldではcoverageとselectabilityを評価するだけにする。 |
| run2 | `I12` | leave-one-mechanism-out oracle screenでfamilyを全OOFから選んだ後、同じwell群でfusionの実現gainを測ると候補family選択の楽観が残る。 | candidate-family screen自体をinner loopへ入れ、outer-heldでは選択済みsetのnative coverage、fallback、fusion gainだけを測る。 |

これらは card を unsafe として0点にする欠陥ではなく、実装時に nested selection を徹底すべき箇所である。run2 `I03`、run1 `I04` / run2 `I04` のtop5案はこの点をすでに明示している。

## parameter-only監査

両 run とも `is_parameter_only: false` の自己申告だけでなく、実際に top 5 が representation、signal role、training distribution、fusion decision、compute graph を変更している。既存failureとの差がweight、lag、particle count、window sizeだけの案はない。

また、両 run とも次を明示的に reject している。

- raw same-well prefix likelihoodのweight再調整
- seed非pairedのままlag / smoothingを増やすこと
- heatmap rank-1 / probability-weighted learned pathの直出し
- truth-aware candidate oracleをdeployable selectorとみなすこと

このため parameter-only 監査は両 run ともPASSである。

## coverage / selectability 分離

両 run ともPASSである。全 card に `coverage_test` と `selectability_test` が別にあり、特に `I04` は以下を分離している。

- coverage: within-10、truth bracketing、oracle RMSE、new-best rate、candidate availability
- selectability: nested OOFのtruth-free posterior、selector regret、calibration、score shuffle、abstention後の実RMSE

全候補欠損時のanchor fallbackを native candidate coverage に算入していない点も適切である。compute / validation cardでselectabilityが非該当の場合も、速度やcoverageだけをaccuracyの証拠にしないことを明記している。

## source boundary監査

候補が宣言した allowed source は両 run とも `rogii_source_hidden_packet_v1.md` のみであり、相対path / 絶対pathの表記差だけで実体は同じである。出力中の次の数値・事実はすべてpacketにある。

- E01–E07のCV、oracle、coverage、worst regression、runtime
- 773 train wells、約200 hidden wells、3-well public example、9時間wall
- heatmap logits / ranks / entropy、既存5候補、full-grid artifact、HMM/PF uncertainty等の再利用可能asset
- train-only formation列の禁止、whole-well GroupKFold、dynamic ID alignment

一方、archive固有の author / rank、Private結果、`91` candidates、GPU約200倍、7th placeの約6000 synthetic wells・60/8 epoch、1st placeの具体的Z-shift実装、13th placeの `U = TVT + dZ` donor synthesis といった後日情報は候補出力に現れない。G7の手順が7th placeと強く一致すること自体は、idea-forgeの imperfect-intermediate probe とpacketの ranking failure / OOF error evidence から導ける範囲で、archive固有fingerprintとは判定しない。

静的な出力比較だけで unauthorized read が絶対になかったことまでは証明できないが、内容上のsource-boundary違反の兆候は両 run とも検出しなかった。

## 最終判定

| run | source-hidden上位機構の生成 | G7回復 | safety | parameter-only | coverage/selectability | 判定 |
|---|---|---|---|---|---|---|
| run1 | top 5で15/16、全12で16/16 | 2/2 | PASS（全12の`I11`はnested generator selectionを要確認） | PASS | PASS | **上位解法級ideaがsource-hiddenで生成されたと言える** |
| run2 | top 5で15/16、全12で16/16 | 2/2 | PASS（全12の`I06`/`I12`はouter-held freezeを要確認） | PASS | PASS | **上位解法級ideaがsource-hiddenで生成されたと言える** |

ここでいう「上位解法級」は、一次 archive の重要機構をsource-hiddenで再発見し、反証可能な実験へ落としたという意味である。実際のROGIIスコア改善やPrivate上位を保証する判定ではない。両 run に優劣は付けない。run2はG7のfold isolationをやや厳密に書き、run1は全12の`I07`でheatmap-as-factorを特に明快に書いているが、主・副採点とも同点が妥当である。
