---
title: 候補パス平均・凸結合の全横断監査
date: 2026-07-16
types:
  - oof_analysis
  - comparison
experiments:
  - exp072
  - exp103
  - exp104
  - exp192
  - exp209
  - exp221
  - exp223
  - exp226
  - exp232
  - exp233
  - exp234
  - exp240
  - exp243
topics:
  - candidate_path
  - blend
  - pf_beam
  - hmm
  - selector
status: final
summary: "保存OOF上の候補パスを横断比較し、exp226とlikPF・exact HMM系の固定結合を主要候補として整理した。"
---

# 候補パス平均・凸結合の全横断監査

作成日: 2026-07-16

## 結論

- ユーザー判断により、ML予測をHMMのcenter / emissionに使う`exp221`、`exp234`、`exp240`はスタッキング寄りとして当面スコープ外にする。selector/TVTモデル最終出力13本も同じく候補パス平均の主評価から外す。
- 除外後の最良単体は`exp226 K16`のRMSE 9.427110。
- 最も単純で強い固定結合は、`exp226 K16`と既存`blend_likpf_hmm_w500`の50/50平均である。実体は`0.50 × exp226 + 0.25 × likpf_mean + 0.25 × exact HMM`で、target-free固定RMSE **8.238331**、exp226比**-1.188778**、5/5 fold改善だった。
- outer-train wellsだけで重みをfitするraw-test生成可能な最良tripleは`exp226 + likpf_mean + exact HMM`で、cross-fit RMSE **8.231651**、exp226比**-1.195458**、5/5 fold改善。fold重みはexp226 55–60%、likPF 17–19%、exact HMM 22–27%で安定しており、固定exp226+w500との差は0.00668だけである。
- train cacheまで許す診断最良は`exp226 + exp192 hard-window likPF + exact HMM`のcross-fit 8.209225。ただしexp192 componentはraw-test portがないため主案にしない。
- primitive 2本の単純50/50では`exp226 + self-GR HMM`が最良でRMSE 8.532715、5/5 fold改善。cross-fitは8.401770。w500という既存派生候補を使いたくない場合の最小案になる。
- `beam_mean + exp226`の50/50は11.189112へ悪化し、cross-fitも9.432876でexp226単体より+0.005766。Beamは平均部品ではなくselector reserveだけに残す。
- K8 medoid、xy-likPF、multiseed PF、robust/mixture PFも監査したが、raw生成系の上位結合を更新しなかった。selector diversityの材料にはできるが固定平均部品としては低優先。

## Evidence boundary

- 対象は3,783,989行、773 wellsの保存済みtrain-side OOF / pseudo-tail予測。
- ID、well、well内rowをcanonical exp072 cacheへ揃え、重複、欠損、coverage不足があれば停止する。94予測すべてcoverage 100%。
- 内訳は候補パス81本、selector/TVTモデル出力13本。
- current primary scopeは81候補パスから`hmm_lgb_exp148`、`hmm_exp218_shrink_a050`、`hmm_exp218_residual_scale`を除いた78本。全94予測の結果は比較履歴として保持する。
- 全94予測の4,371ペアを評価。24本shortlistではサイズ2–6の均等平均190,026組を全列挙し、上位20本では1,140 tripleをcross-fitした。
- 固定50/50・均等平均はtarget-free。同じOOF全体で最適化した重みは楽観的なdiagnosticに限定した。
- 主評価のcross-fitは、outer-valid wellを除く4 foldで非負・総和1の重みをfitし、held-out foldへ適用した。5 foldの予測を結合してRMSEを計算した。
- cross-fitもこの候補bankを見て組合せを選んだ後の結果であり、完全に独立した最終CVではない。新実験ではbank、重みfit、fallback、guardを先に固定する。
- prediction差RMSE 0.25以下の完全に近い候補ペアは0組。ただしHMM系は残差cosineが高く、同一familyの追加は限界効用が小さい。

再現スクリプトと全表は[`studies/candidate_path_blend_audit`](../../studies/candidate_path_blend_audit/)に置く。

## `last_anchor`より良いknown 33候補

HMM+LGB系とmodel/selector outputsを除いたcurrent scopeで、`last_anchor` RMSE 15.909866未満と確認できた候補は次の33本。RMSE昇順。最初の監査にロードした28本に加え、後から漏れを確認したexp104 PF-Z seedbag 5本をcatalog-onlyのsuperseded referenceとして追記した。

| candidate path | source | family | RMSE | raw-test status |
| --- | --- | --- | ---: | --- |
| `exp226_k16` | exp226 | geometry | 9.427110 | inference実績あり |
| `selfgr_hmm_a070` | exp223 | HMM/self-GR | 11.349943 | 再生成経路あり |
| `exp192_likpf` | exp192 | hard-window PF | 11.544812 | train cacheのみ |
| `selfgr_hmm_a150` | exp223 | HMM/self-GR | 11.559584 | diagnosticのみ |
| `hmm_peer_atlas` | exp231 | HMM/atlas | 11.569942 | rejected、raw-testなし |
| `likpf_mean` | exp072 | PF | 11.594898 | 既存raw-test cacheあり |
| `exact_hmm` | exp209 | HMM | 11.938287 | 再生成経路あり |
| `pf_medoid_k8_m0` | exp243 | PF medoid | 12.499353 | candidate-only、raw-testなし |
| `pf_medoid_k8_m1` | exp243 | PF medoid | 12.852916 | candidate-only、raw-testなし |
| `pf_mix_e02` | exp233 | mixture PF | 13.519963 | rejected、raw-testなし |
| `pf_medoid_k8_m2` | exp243 | PF medoid | 13.527396 | candidate-only、raw-testなし |
| `pf_temp_t2` | exp232 | robust PF | 13.529887 | rejected、raw-testなし |
| `pf_temp_t4` | exp232 | robust PF | 13.532730 | rejected、raw-testなし |
| `pf_mix_e05` | exp233 | mixture PF | 13.550173 | rejected、raw-testなし |
| `exp192_pf_ancc` | exp192 | hard-window PF | 13.821165 | train cacheのみ |
| `exp103_xy_likpf_scale_12` | exp103 | xy-likPF | 13.916271 | train candidateのみ |
| `pf_medoid_k8_m4` | exp243 | PF medoid | 13.922659 | candidate-only、raw-testなし |
| `pf_medoid_k8_m3` | exp243 | PF medoid | 13.936275 | candidate-only、raw-testなし |
| `exp103_xy_likpf_scale_8` | exp103 | xy-likPF | 13.961015 | train candidateのみ |
| `exp103_xy_likpf_scale_5` | exp103 | xy-likPF | 14.030092 | train candidateのみ |
| `exp103_xy_likpf_scale_3` | exp103 | xy-likPF | 14.092584 | train candidateのみ |
| `exp104_pf_z_seedbag_scale_12` | exp104 | PF-Z seedbag | 14.145856 | superseded、train cacheのみ |
| `exp104_pf_z_seedbag_scale_8` | exp104 | PF-Z seedbag | 14.171680 | superseded、train cacheのみ |
| `exp104_pf_z_seedbag_scale_5` | exp104 | PF-Z seedbag | 14.178127 | superseded、train cacheのみ |
| `hmm_state_selfgr` | exp225 | HMM/self-GR | 14.212951 | rejected、raw-testなし |
| `exp104_pf_z_seedbag_scale_3` | exp104 | PF-Z seedbag | 14.215698 | superseded、train cacheのみ |
| `pf_medoid_k8_m5` | exp243 | PF medoid | 14.482404 | candidate-only、raw-testなし |
| `pf_ancc` | exp072 | PF | 14.493051 | 既存raw-test cacheあり |
| `exp103_xy_likpf_mean` | exp103 | xy-likPF | 14.580554 | train candidateのみ |
| `exp104_pf_z_seedbag_mean` | exp104 | PF-Z seedbag | 14.587060 | superseded、train cacheのみ |
| `exp192_beam_mean` | exp192 | hard-window Beam | 15.677016 | train cacheのみ |
| `pf_medoid_k8_m6` | exp243 | PF medoid | 15.721456 | candidate-only、raw-testなし |
| `beam_mean` | exp072 | Beam | 15.774327 | 既存raw-test cacheあり。平均部品は不採用、selector reserve |

### cache用core 12候補

known 33本はreference inventoryとして残すが、後続cacheへ値・confidenceを複製するのは次の12本に縮約する。

- non-PF: `exp226_k16`、`selfgr_hmm_a070`、`selfgr_hmm_a150`、`hmm_peer_atlas`、`exact_hmm`、`hmm_state_selfgr`。
- PF/Beam代表: `likpf_mean`、`exp192_likpf`、`pf_ancc`、`beam_mean`、`pf_medoid_k8_m0`、`exp103_xy_likpf_scale_12`。

PF/Beam系は27本から6本へ削減した。robust/mix 4本はfamily内残差cosine 0.9967–0.9978でほぼ重複し全てraw-testなし、xy-likPFは単体最良のscale 12だけ、hard-windowは最良`exp192_likpf`だけを残す。K8 medoidは互いに多様だが、exp252でbank gateと固定top1が不成立かつraw再生成が重いため、coreは単体最良m0だけとし、m1–m6は既存exp243/252 artifactへのexternal diagnostic referenceに下げる。`pf_ancc`と`beam_mean`は平均部品では弱いが、raw-test生成可能な別observation / selector reserveとして残す。

plain exp072 `pf_z`は存在するがRMSE 17.788171でanchorより悪く、exp192 hard-window版は19.705106、strict exp106 multiseed最良も16.145943だったためreference inventory外。exp104 PF-Z seedbag 5本はanchorより良いのに当初監査sourceへ未登録だった。best scale12は14.145856だが、同じXY-slope branchの後続exp103 `xy_likpf_scale_12` 13.916271に劣り、全distance bucketでlikPFを下回り、raw-test portもないためcoreへは戻さない。exp141のZ-driven gateもlikPF 11.594898から最良11.633719へ悪化している。

スコープ外の参考値は`exp221` 8.327728、`exp240` 8.336854、`exp234` 8.427220。これらは削除せず比較履歴に残すが、current rankingには使わない。scope内では単体が弱いself-GR HMM、likPF、exact HMMがexp226と低い残差cosineを持つため平均で効いた。

## ペア平均

### 絶対RMSEで有力な候補パスペア

| A | B | residual cosine | 50/50 RMSE | better親との差 | cross-fit RMSE | cross-fit差 | 改善fold |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| self-GR HMM | exp226 K16 | 0.343641 | **8.532715** | -0.894395 | **8.401770** | -1.025340 | 5/5 |
| peer-atlas HMM | exp226 K16 | 0.337492 | 8.607484 | -0.819626 | 8.448157 | -0.978952 | 5/5 |
| exact HMM | exp226 K16 | 0.297062 | 8.635074 | -0.792035 | 8.419751 | -1.007359 | 5/5 |
| exp192 likPF | exp226 K16 | 0.379097 | 8.727406 | -0.699704 | 8.530865 | -0.896245 | 5/5 |
| likPF | exp226 K16 | 0.399893 | 8.813822 | -0.613288 | 8.610567 | -0.816543 | 5/5 |

primitive 2本だけなら`exp226 + self-GR HMM`が最も安全な固定案。ただし、既存派生候補w500を1本として扱うと`exp226 + w500`の50/50が8.238331まで改善し、primitive pairを大きく上回る。

### RMSE改善幅だけが大きい弱いペア

`tvt_dense50 + exp192_pf_z`は19.705106より14.804796へ-4.900310、`tvt_dense50 + K8 medoid m7`は17.043353より13.614995へ-3.428358と大きく改善した。これは低相関平均の原理を確認するが、絶対RMSEが強い単体・3本結合に遠い。selector diversityの証拠にはなるが、提出blendの候補にはしない。

## 3本以上の結合

### well cross-fit凸結合

raw-test生成可能なscope内最良tripleは次の3本だった。

| component | full-OOF diagnostic weight | fold別cross-fit weight range |
| --- | ---: | ---: |
| exp226 `exp226_k16` | 0.574682 | 0.553755–0.596412 |
| exp072 `likpf_mean` | 0.180944 | 0.172173–0.190047 |
| exp209 `exact_hmm` | 0.244374 | 0.215025–0.270200 |

- full-OOF最適化診断: 8.194600。採用スコアには使わない。
- held-out-well cross-fit: **8.231651**、exp226比**-1.195458**、5/5 fold改善。
- 489 wells改善 / 284悪化、well RMSE差中央値-0.723352、worst-well +13.807803。
- exp192 hard-window likPFへ置換するとcross-fit 8.209225だが、exp192はtrain cache only。
- self-GR HMMをexact HMMの代わりに入れた`exp226 + self-GR HMM + likPF`は8.240050で僅差。496 wells改善 / 277悪化だがhidden-likeは主案より弱い。

### 固定平均

| members / weights | fixed RMSE | exp226との差 | fold | raw-test判断 |
| --- | ---: | ---: | ---: | --- |
| exp226 50% + w500 50% | **8.238331** | **-1.188778** | **5/5** | 全成分生成可能、target-free固定 |
| exp226 50% + self-GR HMM 50% | 8.532715 | -0.894395 | 5/5 | primitive 2本の最小案 |
| exp226 + self-GR HMM + exp192 likPF、各1/3 | 8.533442 | -0.893668 | - | exp192 train cache only |
| exp226 + likPF + exact HMM、各1/3 | 8.596018 | -0.831091 | - | 全成分生成可能だがw500利用案より弱い |

`exp226 + w500`は新しいweight searchではなく、既存候補2本の50/50である。実体はexp226 50%、likPF 25%、exact HMM 25%。cross-fit最適重みとの差が0.00668しかないため、複雑な重み学習を避けるcurrent first choiceになる。

## scope内推奨結合の頑健性

| prediction | overall | 0–50 | 50–100 | 100–250 | 250–500 | 500–1000 | 1000+ | hidden spatial | hidden typewell |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exp226 K16 | 9.427110 | 1.742865 | 1.670616 | 2.184186 | 3.541102 | 5.469033 | 10.331435 | 9.399891 | 9.413374 |
| fixed exp226 + w500 | **8.238331** | **0.999067** | **1.176485** | **1.823929** | **3.068759** | **4.613953** | 9.042324 | 8.748108 | 8.694132 |
| cross-fit exp226 + likPF + exact | 8.231651 | 1.096470 | 1.217659 | 1.831957 | 3.074033 | 4.649590 | **9.031948** | **8.574173** | **8.531535** |
| cross-fit exp226 + self-GR + likPF | 8.240050 | 1.079783 | 1.208895 | 1.842871 | 3.106345 | 4.691346 | 9.038046 | 8.699737 | 8.654100 |

fixed exp226+w500は5/5 fold、475 wells改善 / 298悪化、well RMSE差中央値-0.766671。cross-fit主案は5/5 fold、489/284 wells、中央値-0.723352。いずれも全distance bucketとhidden-like 2群でexp226を改善するが、worst-wellは固定+15.395782、cross-fit+13.807803である。平均だけでwell safetyは成立しない。

1. 最初はtarget-freeな`0.5 × exp226 + 0.5 × w500`を固定する。
2. convex weight学習は固定案との差0.00668に追加価値があるかを別比較にし、同時採用しない。
3. overall、全距離帯、hidden-like、5 fold、by-wellを保存する。worst-well +15.4 ftを許容したまま自動採用しない。

## 過去family別の平均適性

| family / 過去実験 | 最良の確認結果 | 判断 |
| --- | --- | --- |
| exp221 / exp234 / exp240 HMM+LGB系 | 全候補監査では7.69台まで改善 | **スタッキング寄りとしてscope外**。比較履歴のみ |
| exp226 K16 | scope内単体9.427110。w500との固定50/50 8.238331 | **主成分** |
| exact HMM / exp209 | likPFとの50/50がw500 10.269696。exp226との3本cross-fitで約24% | **弱い単体だが主blend成分** |
| self-GR HMM / exp223 | exp226との固定50/50 8.532715、cross-fit 8.401770 | primitive 2本案。3本主案とは僅差でないためfallback |
| peer-atlas HMM / exp231 | exp226とのcross-fit 8.448157 | diagnostic positiveだがrejected/no raw-test |
| hard-window PF / exp192 | exp226 + exp192 likPF + exact HMM cross-fit 8.209225 | scope内diagnostic最良だがtrain cache only |
| K8 medoids / exp243 | exp226 + exact HMM + m0 cross-fit 8.227618 | 数値は有望だがraw-test未生成・高コスト。固定平均では主案を更新しない |
| xy-likPF / exp103 | scale12 + exp226は50/50 9.948985、cross-fit約9.12 | 50/50でexp226を悪化。平均部品として不採用 |
| multiseed PF-z / exp106 | exp226との50/50は約10.95、cross-fit約9.31 | 不採用 |
| robust/mixture PF / exp232/233 | exp226との50/50約9.93、PF同士の残差cosine約0.997 | family内でほぼ冗長。重み0になり不採用 |
| Beam | exp226との50/50 11.189112、cross-fit 9.432876でexp226比+0.005766 | 平均部品は不採用。selector reserveのみ |
| state-known self-GR HMM / exp225 | exp226との50/50 9.559でexp226単体より悪い | 不採用 |

## `blend_likpf_hmm_w500`の扱い

再構成した`likpf_mean + exact_hmm`の50/50はRMSE 10.269696で、既存w500の値を再現する。

- likPF単体: 11.594898。
- exact HMM単体: 11.938287。
- 50/50: 10.269696、5/5 fold改善、539 wells改善 / 234悪化。
- cross-fit重みはexact HMM平均0.4697、std 0.0194、RMSE 10.285676。
- ただしworst-wellはbest parent比+23.036613。

したがって「既存候補を平均しているだけ」という理解で正しい。一方、HMM+LGBを除外すると、w500とexp226の50/50がRMSE 8.238331となり、scope内で最も強いtarget-free固定平均になった。この用途ではw500を残す意味がある。

- 固定blendとして使う場合: `exp226`と`w500`の2出力だけを50/50にし、selector bankへ派生候補を増やさない。
- primitive selectorを作る場合: likPF、exact HMM、exp226を独立に保持し、w500を同時にselectableにはしない。

## selector/TVTモデル出力の平均

候補パス以外も参考に全ペアへ含めたが、HMM+LGBと同じ理由でcurrent primary scopeから外す。

| outputs | fixed RMSE | cross-fit RMSE | 判断 |
| --- | ---: | ---: | --- |
| exp193 LGB + exp237 row selector | 7.883523 | 7.915260 | 5/5だがexp237 raw-test parity fail |
| exp251 probability + exp255 assertive | 7.892442 | 7.815428 | 5/5だが両方guard fail/no inference |
| exp251 probability + exp238 add-only | 7.886650 | 7.834299 | fixedは改善、exp251 guard fail |

これは「selector score」ではなく、selectorが選んだpathやTVT回帰モデルの最終予測を平均した結果である。候補パスselectorのbank設計とは別層で管理する。raw-test contractを満たしていない出力は、CVが良くても提出blendへ昇格させない。

## 次の判断

no-training監査としては、次の優先順位になる。

1. **高・基盤**: `last_anchor_better_candidate_confidence_pair_cache` を作成し、`last_anchor`より良いknown 33候補をreference inventoryとして固定する。新cacheはfamily圧縮したcore 12候補（PF/Beamは27本から6代表）、有望8 pair、3 named tripleへ縮約する。全378 pairは当初監査28本の履歴に残し、superseded exp104 5本のpair sweepは追加しない。
2. **中・cache後**: `exp226 + blend_likpf_hmm_w500`の固定50/50。target-free、RMSE 8.238331、5/5 foldsで、追加weight fitが不要。cache/virtual pair adapter の parity target にする。
3. **中**: `exp226 + likPF + exact HMM`のouter-well convex blend。8.231651だが固定案との差は-0.006680だけなので複雑化の便益は小さい。
4. **最小primitive案**: `exp226 + self-GR HMM`の固定50/50、8.532715。w500を派生候補として使わない場合の代替。
5. **診断のみ**: exp192 hard-window likPF、peer atlas、K8 medoidを含む組合せ。改善はあるがraw-test portまたは生成コストに問題がある。
6. **scope外**: exp221/234/240 HMM+LGB系、selector/TVT最終出力。比較履歴は保持するがcurrent優先順位へ入れない。
7. **bank整理**: fixed blendならexp226とw500だけ、primitive selectorならexp226・likPF・exact HMMだけにし、親子派生候補を同時に増やさない。

新実験の作成、重み契約、raw-test生成・提出は複数の妥当な選択肢があるため、この監査では着手していない。
