# 設計

## 結論

`exp391_prefix_anchored_mode_persistence_hmm_readout`を、原因切り分けと
prefix-anchor mode persistenceの二段階readoutとして固定する。

前rowの「top1」を参照するだけではmode遷移を防げない。top1はposterior massの順位であり、
2つのmodeのmassが交差すると、同じ物理modeでもtop1/top2の名前が入れ替わる。また、
1 rowごとのjumpが小さくても、小さい遷移を累積して別modeへ移れる。このため本設計では
mass rankから独立したmode identityを持ち、前rowからのtransition overlapで追跡する。
cross-mode edgeを一度でも通ったpathは、step sizeに関係なく最終候補から除外する。

## 既存証拠と切り分けが必要な理由

- exp236ではexp221 posteriorの二峰rowは35,399 / 3,783,989、138 wells、317 segments。
  mode mass switch / track breakは17 / 17だった。
- exp236のposterior mean RMSEは8.327728、marginal MAPは8.365160、
  dominant-mode conditional meanは8.331754で、直接decoder置換はnegativeだった。
- exp270ではexp209 posterior mean 11.938287に対し、marginal MAP 12.592479、
  global Viterbi 15.551665で、direct replacementはnegativeだった。
- exp236はexp221のLGB-centered HMM、exp270はexp209のraw absolute-TVT HMMであり、
  top-2 massとViterbiをartifact間で直接joinするとposterior contractが混ざる。
- exp226 final K16は`preprojection = tvt_geop + gr_delta`へdegree-4 U projectionを適用して
  `tvt_pred`を作る。したがってraw HMMがjumpしていても、K16 projectionまたはその後の
  fixed blendがrampを作る可能性が残る。

既存のMAP / Viterbi直接採用が悪かった事実は維持し、exp391はその再提案ではなく、
観測されたramp-to-offsetの生成箇所を同一well上で切り分ける。

## 実験範囲

- 対象実験: `exp391_prefix_anchored_mode_persistence_hmm_readout`
- Route: `pf_beam`
- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 保存control:
  exp209 posterior mean、decompressed SHA
  `8e2f42367b7b8b28e73094eae642c57c75dc8a7ebcfbc3826b0f2067b37f7ae5`
- 参照:
  exp236、exp270、exp226、exp263、exp115。
- 変更する変数:
  posterior shape readout、stable mode identity、cross-mode edge ledger、
  prefix-anchor no-switch conditional decoder。
- 固定する変数:
  exp209 absolute-TVT HMMのgrid、rate state、transition、emission、sigma、prior、
  typewell GR、missing-GR handling、posterior-mean定義。
- fitted model / LightGBM / PF / Beam / booster / GPU: すべてなし。
- inference / submission: 無効。

## Artifact contract

### exp270: exp209 mean / MAP / Viterbi

- aggregate candidate:
  `exp270_exact_hmm_posterior_mode_candidate_audit_candidates.csv.gz`
- fixed columns:
  `posterior_mean`、`marginal_map`、`topk_path_1`
- `topk_path_1`をglobal Viterbi aliasとする。
- aggregate artifactを優先し、なければ次の2 shardをstrict joinする。
  - shard0: 1,792,363 rows / 363 wells /
    decompressed SHA
    `93e0aeac70b1e84d139aab05f9a1d6abd577d2a07388367cf2dac362e8f68b6d`
  - shard1: 1,991,626 rows / 410 wells /
    decompressed SHA
    `831cbfb5adfe09f98059f4e2a192d7913331f6c57c437fadc989f01e3c91aee5`

### exp226: K16 projection前後とreporting fold

- source:
  `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction_train_oof_predictions.csv.gz`
- `k16_preprojection = tvt_geop + gr_delta`
- `k16_postprojection = tvt_pred`
- `well_id`、`row_idx`、`suffix_offset`、`fold`をrow identity / reporting scopeに使う。
- exp226 prediction列は原因分類とscore controlにだけ使い、exp209 mode生成へ入力しない。

### exp263: fixed physical candidate

- candidate: `exp226_w500_50_50`
- 定義: exp226 50% + likelihood-PF 25% + exact HMM 25%
- 保存OOF CV: 8.238331
- Stage 0 manifest SHA:
  `85e60ac10b50197fa44ea29faffcbba81bd0746114bc53bae0f5cc537a26bb9e`
- fixed candidate値を再生成せず、保存cacheからjoinする。
- cacheの`outer_fold`はexp072由来のstorage partitionであり、exp226 reporting
  foldとは別契約である。両fold labelの数値一致は要求しない。exp226 `fold`だけを
  reporting scopeに使い、exp263は`id` / `well` / `well_row_idx` / `md_since`と
  partition内component identityをstrict検証する。

### exp236: threshold reference only

exp236のrow summary値をexp209のposterior値として使用しない。次のpeak/basin判定定数だけを
事前登録値として再利用する。

- minimum peak height: 0.02
- minimum top2 mass: 0.10
- minimum top2 / top1 mass ratio: 0.25
- minimum peak separation: 6.0 ft
- minimum valley depth: 0.30
- mean / valley density ratio maximum: 1.20
- mode tracking allowance: 6.0 ft

## Stage A0: 保存artifactによるtarget-free event census

HMMを再実行する前に、exp270 / exp226 / exp263を同一row identityでstrict joinする。
truth、error、hidden-like roleはまだjoinしない。

### eventの事前定義

各path `x_t`についてstep `d_t = x_t - x_{t-1}`を計算する。exp270の
`marginal_map`または`topk_path_1`と`posterior_mean`の差が6.0 ft以上で、かつその差が
連続32 rows以上維持される区間を`decoder_separation_event`とする。event同士の間隔が
32 rows未満なら同一eventへ結合する。

同じeventで以下を保存する。

- posterior mean / MAP / Viterbiのstart、end、最大step、差の符号、持続row数
- K16 preprojection / postprojectionのstart、end、最大step、線形ramp slope
- exp263 fixed candidateのstart、end、最大step、線形ramp slope
- `MAP - mean`、`Viterbi - mean`、`postprojection - preprojection`、
  `fixed_candidate - postprojection`
- well、row、fold、suffix-distanceだけからなるevent manifest

16-well preflightはtruth-free event severityで選ぶ。各foldから
`max(abs(MAP-mean), abs(Viterbi-mean))`上位3 wellsを選び、重複を除いた後、
全体中央値に最も近いwellを1件加える。不足時はwell id昇順で補完する。
選択manifestをfreezeしてlogical SHAを保存する。

### Stage A0 technical gate

- joined rows 3,783,989、wells 773、folds 0--4
- key duplicate 0、missing join 0、各artifact内well-fold inconsistency 0、
  exp270 / exp263 `id` / `md_since` mismatch 0
- exp270 posterior meanとexp209保存controlのmax abs差 `<=1e-5 ft`
- finite path coverage 1.0
- selected preflight wells 16、各foldのwellが最低1
- truth / error / hidden-like role read count 0
- artifact identity / decompressed SHA / event manifest logical SHAを記録

1つでもFAILなら実装値やjoin toleranceを救済せず停止する。

## Stage A1: 同一exp209 posteriorの16-well切り分け

固定16 wellsだけexp209 exact HMMを`return_post=true`で再生する。同じpassから
posterior mean、marginal MAP、global Viterbi、top-2 basin TVT / mass /
conditional meanを抽出する。exp236 row summaryをjoinしない。

### peakとbasin

1. TVT marginal posteriorをTVT grid上で計算する。
2. local maximumをTVT昇順で列挙し、exp236固定thresholdを適用する。
3. 隣接peak間の最小posterior位置をbasin境界とする。
4. basin mass、peak TVT、basin conditional meanを計算する。
5. `top1` / `top2`は表示用のmass rankであり、mode IDには使わない。

### 原因ラベル

原因は排他的に決めず、各eventへ複数のboolean flagを付ける。

- `posterior_averaging_supported`:
  top-2 peak separationが6 ft以上、mass rankが交差し、MAPまたはViterbiが一方のbasinに
  留まる間にposterior meanが2 basin間を連続的に移動する。
- `transition_kernel_supported`:
  marginal MAP自身が同じ方向へ32 rows以上rampし、stable mode IDをまたぐ。
- `k16_projection_supported`:
  raw HMM MAP/Viterbiには不連続なbasin切替があるが、
  K16 preprojectionからpostprojectionへの差でrampが新たに生じる。
- `fixed_blend_supported`:
  K16 postprojectionにはないrampがexp263 fixed candidateで生じる。
- `unresolved`:
  上記のどれも満たさない、またはstable top-2 basinを構成できない。

### Stage A1 gate

- 16 wells全件でexp209 posterior mean / MAP / Viterbiがexp270保存値と
  max abs `<=1e-5 ft`
- posterior normalization error `<=1e-8`
- mode ledgerのkey重複、nonfinite、identity collisionが0
- peak/basin tie-breakを含む再実行可能なdecoder manifestを保存
- decoder separation eventが最低10件
- `posterior_averaging_supported`または`transition_kernel_supported`が
  eligible eventの60%以上かつ4/5 foldsで1件以上
- 16-well実測から773 wells projected runtime `<=30,600 sec`、
  projected peak RSS `<=25 GB`

HMM内原因が60%未満なら、Stage Bへ進まずK16 projection / fixed blend側の原因として閉じる。
technical / mechanism / resource gate全PASSでも、full runは別承認とする。

## Stage B: prefix-anchor mode persistence

Stage A1全PASS後だけ、1 variant / 773 wellsのexact HMM passで実行する。
Stage A1の16 wellsはfull runに含め、別variantとして再計上しない。

### stable mode identity

mode IDはmass rankでなくlineageとして管理する。

1. suffix開始時のjoint start priorを`anchor_lineage`とする。
2. row `t-1`の各mode basinに属するjoint posterior massをexp209 transitionでrow`t`へ運ぶ。
3. transported massとrow`t` basinのoverlapを計算し、最大weight matchingでmode IDを継承する。
4. overlap tieは`previous mode id`、`current basin center TVT`の順でstableに解決する。
5. matching allowance 6.0 ftを超える、またはbasinがmerge / splitする場合は
   ancestry ledgerに明記する。mass rankの交差だけではmode switchとしない。
6. path edgeのsource / destination mode IDが異なる場合を`cross_mode_edge=1`とする。
7. `mode_switch_count`はcross-mode edgeの累積和とし、step deltaの大きさには依存しない。

### candidate

保存する科学candidateは1本だけである。

```text
prefix_anchor_no_switch_conditional_mean
```

- `anchor_mode_id`: suffix開始priorから継承したlineage ID
- 各rowでは`anchor_mode_id`かつ`mode_switch_count=0`のjoint posteriorだけを再正規化し、
  TVT conditional meanを出力する。
- alternate mode、jump-used、switch-used laneはdiagnostic列だけに保存する。
- no-switch massが0、identityが未解決、normalizationが壊れたwellでは、
  row単位blendせずwell全体を保存exp209 posterior meanへfail closedする。
- no-switch candidateとparent meanをsoft averageしない。
- candidate生成とlogical SHA freeze後だけtruthをjoinする。

この設計により、大きな1-step jumpだけでなく、小さなstepを積み上げて別modeへ移るpathも
`mode_switch_count > 0`となり、最終candidateから除外される。一方、同じmode basin自体が
物理的に移動することは許容する。

## Stage B scientific gate

主比較はexp209 posterior meanであり、exp226 / exp263は下流影響のreporting controlとする。

- pooled RMSE gain vs exp209 posterior mean `>=0.25 ft`
- positive reporting folds `>=4/5`
- Stage AのHMM-supported eventsでRMSE gain `>=0.50 ft`
- 1000+ distance bucket regression `<=0.0 ft`
- hidden-like spatial regression `<=0.0 ft`
- hidden-like typewell-purged regression `<=0.0 ft`
- by-well RMSE delta p95 `<=0.0 ft`
- worst-well regression `<=+0.25 ft`
- fail-closed well率 `<=0.10`
- exp263 fixed candidateへ25% HMM componentだけを置換したreport-only fixed formulaが
  元のexp263 fixed candidateを悪化させない

全AND gateとする。FAIL時はmode identity、threshold、matching、fallback、weightを救済せず閉じる。
PASSしてもPF / Beam移植、K16変更、inference、submissionは別実験・別承認とする。

## モデル別の後続判断

exp391内では次を実装しない。Stage Bが科学gateをPASSした場合だけ、別実験として設計する。

| モデル | 将来のmode保持 | final lane |
| --- | --- | --- |
| HMM | `anchor_mode_id`と`mode_switch_count`をjoint state ledgerへ追加 | `mode_switch_count=0`だけを再正規化 |
| PF | particle ancestryへ固定mode IDとcross-mode countを保持 | no-switch particlesだけを再正規化 |
| Beam | edgeへcross-mode flag、pathへcountを保持 | no-switch beamを別枠で保持 |
| K16 | latent mode pathがないため直接移植しない | projection前後の原因診断だけ |

PFのrejuvenationやBeamのjump edgeを先に追加すると、原因がposterior averagingなのか
downstream smoothingなのか不明なまま探索空間だけが増えるため、本実験の結果前には進めない。

## 実行量と承認境界

| Stage | 内容 | HMM well runs | model / booster | 承認 |
| --- | --- | ---: | ---: | --- |
| A0 | 保存artifact join / event census | 0 | 0 / 0 | 実装承認が必要 |
| A1 | 固定16-well same-posterior preflight | 16 | 0 / 0 | 実行前に別承認 |
| B | 1 variant full readout | 773 | 0 / 0 | A1全PASS後に別承認 |

parent control再学習・別variant再実行は0。Stage Bの773 runsはposteriorとcandidateを同一passで
作る。Kaggle train push前に`1 variant / 773 HMM runs / LightGBM 0 configs /
fold 0 / booster 0 / PF 0 / Beam 0`をSESSION_NOTESへ再記録する。

## 再現性設計

- seed policy:
  RNGなし。well、row、state、peak、basin、mode ID、eventをimmutable keyでstable sortする。
- stochastic処理:
  なし。
- PF / Beam / likelihood-PF / seed bagging:
  なし。exp263値は保存artifactのreporting inputだけ。
- 並列処理:
  outer worker 1を既定とする。Numba thread数をmanifestへ記録し、row reduction orderを固定する。
- CPU/GPU:
  private CPU、GPUなし、internet off。full run上限30,600 sec / 25 GB。
- input SHA:
  exp209 control、exp270 shard、exp226 OOF、exp263 cache、exp115 role assignmentについて
  raw SHAとdecompressed/logical SHAを記録し、後者を主証拠とする。
- output SHA:
  event manifest、posterior row summary、mode ledger、cause labels、candidate、
  score tableのschema SHAとlogical content SHAを保存する。
- model SHA:
  fitted modelなし。exp209 HMM、peak、basin、matching、decoderの全定数を含む
  decoder contract manifest SHAをmodel manifest相当として保存する。
- prediction / submission SHA:
  train-side candidate SHAだけを保存。submissionは生成しない。
- deterministic anchor:
  初回runでは主張しない。同じ設定の成功rerunでevent manifest、mode ledger、
  candidate logical SHAが一致した場合だけ主張できる。
- Kaggle bootstrap:
  実装承認後にcanonical id/title、private、CPU、internet off、config/source/bootstrap SHAを照合する。

## リスク

- mode label switching:
  top1/top2 rankをIDにするとmass crossoverで偽switchになる。transition-overlap lineageで防ぐ。
- gradual cross-mode drift:
  `jump_used`だけでは検出できない。全edgeのmode ID変化を累積する。
- mode merge / split:
  identityが一意でないwellは推測で補わず、well全体をparentへfail closedする。
- posterior contract混同:
  exp236とexp270は別HMM。exp236値はthreshold reference以外に使わない。
- leakage:
  event選択、16-well選択、mode tracking、candidate生成前のtruth/error/role readを0にする。
- selection bias:
  16 wellsはtarget-free severityとfold quotaで固定する。科学結論はfull 773 runでのみ確定する。
- CV/LB不一致:
  exp209 raw HMMは最終提出anchorではない。Stage B PASSでもinferenceへ直結しない。
- runtime / memory:
  full posterior tensorはprocess-localとし、row summaryとledgerだけstream保存する。
- direct decoder regression:
  exp236 / exp270のnegative resultから、同一OOFでのMAP、Viterbi、threshold、blend救済を禁止する。

## 設計時点の禁止事項

- notebook / helper / testの実装
- Kaggle package生成、push、train、inference、submission
- exp236 row summaryとexp270 Viterbiを同一posteriorとして結合
- top1/top2 rankをstable mode IDとして使用
- truth/error/oracleによるevent、well、mode、threshold、fallback選択
- jump penalty、mode数、peak threshold、matching allowance、blend weightのgrid
- marginal MAP / Viterbi / alternate modeの直接final採用
- row単位oracle fallback、softmax平均、cross-mode path平均
- PF / Beam / K16 / exp263 generatorの変更
