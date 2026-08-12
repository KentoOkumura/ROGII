# 設計

## アプローチ

exp293の12 pathをH256 blockごとのcandidate stateとして扱う。連続する同一candidateを
1 segmentとし、segment長は最低2 blocks、上限はwell末尾まで、最終short segmentは
right-censoredとしてexact log-space semi-Markov forward-backwardで周辺化する。
Viterbi pathは予測に使わず、block marginalをblock center間で線形補間し、
12 pathの凸結合を最終TVTとする。

各candidate/blockの観測scoreは、candidate TVTの局所`[-55,+55] ft`を5 ft刻みで
ずらしたType Well GRとhorizontal GRの形状一致から作る。両系列をblock内で
median/MAD標準化し、raw / rolling-21 / rolling-101のcapped squared residualを
`0.50 / 0.25 / 0.25`で固定合成する。shiftにはLaplace scale 20 ftのpriorを置いて
log-sum-exp周辺化し、shift自体はTVTへ足さない。64 finite pair未満またはcoverage
0.50未満はneutral emissionとする。20%のcandidate共通neutral mixtureを常に残し、
GR不一致だけで強制mode移動させない。

rolling-21 / rolling-101は各H256 block内だけのcentered rolling meanとし、
full windowが揃う位置だけをfiniteにする。block境界をまたぐ値は読まない。
block-order controlは同じ長さのfull H256 blockだけをSHA256順で並べ替え、
最終short blockは固定する。fixed16 resource preflightはouter foldごとの
SHA256 rankをround-robinし、fold 0から順に合計16 wellsを事前決定する。

segment開始priorは`exp226_w500_50_50=0.45`、
`exp226_k16=0.10`、残り10候補を各0.045に固定する。現在stateと異なるsegmentへ
切り替えるときだけ`log(9)`の固定penaltyを加える。現在stateがgeometryでない場合、
新segmentの`exp226_k16`確率を再正規化後も0.10以上に保つ。
このgeometry再注入はdockingと独立であり、exp399の復帰不能構造を持たない。

## 実験範囲

- 対象実験: `exp405_geometry_reinjected_interval_semimarkov_fusion`
- Route: `pf_beam`
- 親実験: `exp293_physics_only_candidate_bank_headroom_contract`
- 比較基準: `exp263_last_anchor_better_candidate_confidence_pair_cache`
- 変更する変数: candidateを固定したまま、局所形状GR emission、
  explicit-duration posterior、docking-independent geometry floorを追加する。
- 固定する変数: 12 candidate値・順序・fold・row identity・H256/H512 assignment、
  ±55 ft / 5 ft grid、emission weight、duration、prior、penalty、全gate。
- scientific endpoint: candidate posterior mean TVTを1本だけ生成する。
- negative controls: stable within-well circular GRとstable block-order GR permutation。
- 実行量: real 1 endpoint + 2 diagnostic controls / 5 reporting folds /
  model・LightGBM・booster・PF・HMM・Beam・parent rerun各0。

## 段階と停止条件

1. fixed16で入力parity、role-read、posterior正規化、resourceだけをpreflightする。
2. technical PASS時だけ773 wellsのsaved-OOF score/posterior/predictionをtruth-freeでfreezeする。
3. truth-lateにduration-constrained oracle、primary RMSE、fold/scope/tail、
   negative control、geometry massを評価する。
4. 全scientific gate PASS時だけcurrent-test実装を同じexp405内で解禁する。
5. 技術的に有効なscientific FAIL時はexp405を閉じ、exp406だけを解禁する。
   technical ERRORは修正後に同じ契約を再実行し、exp406の解禁理由にしない。

## current-test契約

- exp405 PASS前は実装しない。
- PASS後もcandidate familyとsemi-Markov設定を変えず、exp263でraw-test再生成可能と
  確認済みの6 primitiveから5 pairと固定formulaを同じfloat演算順で作る。
- candidate生成器、Type Well/local GR score、semi-Markov posteriorの
  train/inference parityとcontent SHAを記録する。
- current-test実装、Kaggle inference、submissionはそれぞれ明示承認を得る。
- current-testで利用できない入力が1つでもあればfail closedし、代替candidateや
  exp263 fallback submissionへ自動差替えしない。

## 再現性設計

- seed policy: real endpointはRNGなし。negative controlの回転量とblock permutationは
  `SHA256("exp405::<control>::<well_id>")`から決定する。
- stochastic 処理の有無: negative controlの見かけ上のrandomizationだけ。local RNGも不要。
- PF/Beam / likelihood-PF / seed baggingの有無: 保存pathを読むため0。
- 並列処理と乱数の関係: well単位2 threadsまで。全結果を
  `(fold, well_id, block_id, candidate_order, shift)`でstable sortする。
- CPU/GPU runtime: Kaggle CPU、GPU/AMP/internet off。
- train cache SHA: exp293 candidate content、bank manifest、block assignmentの
  raw/decompressed/logical SHAを照合し、score/posterior/prediction/schema SHAを保存する。
- current-test regeneration SHA: PASS後の実装時にraw input、6 primitive、
  derived 12 bank、posterior、predictionのlogical content SHAを追加する。
- model / submission SHA: fitted modelなし。trainではprediction SHAのみ、
  submission SHAはcurrent-test提出時まで対象外。
- Kaggle bootstrap: package作成後にembedded config/source SHAとmetadataを照合する。
- deterministic anchor: 初回成功runでは主張せず、同じinput SHAの成功rerunで
  score/posterior/prediction SHA一致を要求する。

## リスク

- リークリスク: truthでcandidate、shift、duration、prior、gateを選ぶこと。
  score/prediction freeze前truth read 0をhard gateにする。
- 識別リスク: exp297と同様にGRがtruth-good pathを識別できない可能性がある。
  real対2 controlsと5/5 foldを必須にし、FAIL時はweight/gridを救済しない。
- geometry collapseリスク: likelihoodがgeometry floorを圧倒する可能性がある。
  posterior mass guardをpromotion条件へ含める。
- CV/LB不一致リスク: OOF pathは保存済みでもcurrent-test pathはraw再生成となる。
  current-test実装前に全generator parityと入力可用性を別途監査する。
- ランタイム/メモリリスク: 3.78M rows × 12 candidates × 23 shifts。
  fixed16 projection、chunk処理、2-well並列、25 GB guardを固定する。
- 再現性リスク: equal posteriorやfloating reduction順。stable orderとfloat64
  log-space reduction、tie policyを固定する。
