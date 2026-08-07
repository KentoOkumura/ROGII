# 設計

## アプローチ

outer-train donorとfixed16 targetのhorizontal GRを、H256 window / H128 strideで
block化する。targetのexp226 geometry pathを中心にType Well TVT
`[-55,+55] ft`を5 ft刻みで走査し、same-Type-Wellを優先した近傍12 donor wellsの
GR blockとraw / rolling-21 / rolling-101 morphology NCCを比較する。
各target blockは上位4 edgeだけを、score、finite coverage、candidate/edge keyの
固定順で保持する。

edgeが表すrelative TVT offsetを`o_ij(u) ≈ g_j(u)-g_i(u)`として、
fundamental cycle basis上の和が0へ近づくようHuber delta 5 ft、
10 iterationsのdeterministic sparse IRLSで全well/block gaugeを同時に解く。
outer-train donor真値TVTとtargetのvisible prefixはgauge anchorに使う。
このStage 0ではunknown suffix TVT pathを出力せず、edge support、connectedness、
cycle residual、prefix pseudo-cut、resourceだけを保存する。

prefix rolling-originの比較controlは、保存済みexp226 OOFがofficial suffix行しか
持たずprefix512を直接覆わないため、ユーザー承認済みの推奨案を使う。
target-free graphをSHA freezeした後、各foldのouter-trainだけからexp226 original
K16 raw/smoothed donor field、near-strike ANCC local-theta、adaptive Kappaを
再構築し、fixed16 pseudo-cutの`tvt_geop`相当だけを生成する。GR correctionと
U-projectionは使わず、official OOFや全well predictionを再生成しない。

## 実験範囲

- 対象実験: `exp406_loop_closed_multiwell_rgt_fixed16_stage0`
- Route: `pf_beam`
- workflow parent / unlock dependency:
  `exp405_geometry_reinjected_interval_semimarkov_fusion`
- scientific family: `conditional_independent_gr_first_loop_closed_rgt_family`
- negative reference: `exp386_cycle_consistent_rgt_scenario_bank`
- 変更する変数: Formation-derived RGTとroute enumerationを、
  horizontal-GR pairwise correspondenceとglobal loop closureへ置換する。
- 固定する変数: fixed16 selector、exp226 folds/control、H256/H128、
  ±55/5 ft、12 donors、top4 edges、multiscale score、IRLS、全gate。
- 実行量: 1 Stage 0 diagnostic / 16 target wells / 5 donor graph contexts /
  exp226 fixed16 prefix geometry fold replay最大5 /
  model・LightGBM・booster・HMM・PF・Beam各0。

## Stage 0の評価順

1. exp405 decision manifestとfixed16 identityを検証する。
2. truth-freeにreal/circular pairwise edge、graph、cycle basis、solved gaugeをfreezeする。
3. coverage、connectedness、cycle residual、negative-control separationを判定する。
4. technical PASS時だけvisible prefix末尾512 rowsをpseudo-unknownとして評価する。
5. resource projectionを含む全gate PASS時だけfull-OOF Stage 1の設計資格を得る。
6. FAIL時はdonor数、window、shift、NCC、edge数、Huber、cycle gateを救済せず閉じる。

## exp386との差

- target GRを観測入力として明示利用する。Formation 6列は使わない。
- pairwise対応edgeを先に作り、loop closureをedge offsetの単位`ft`で直接解く。
- k-shortest routeや8--32 scenariosをStage 0で列挙しない。
- exp386のgraph query 0を閾値緩和で救済せず、edge生成式とRGT sourceを置換する。
- exp386 fixed16 selectorだけを比較可能性のため再利用する。

## 再現性設計

- seed policy: RNGなし。circular controlのrotationは
  `SHA256("exp406::circular::<well_id>::<block_id>")`から決める。
- stochastic処理: なし。controlもdeterministic。
- PF/Beam / likelihood-PF / seed bagging: 0。
- 並列処理: target well単位2 threadsまで。node/edge/cycle/resultをimmutable keyで再sortする。
- CPU/GPU: Kaggle CPU、GPU/AMP/internet off。
- input SHA: exp405 decision、exp226 OOF/fold、raw well identity、Type Well、
  fixed16 manifestを記録する。
- feature SHA: pairwise edge、cycle basis、solved gauge、role-read ledger、
  prefix readoutのschema/logical/decompressed SHAを記録する。
- model/prediction/submission SHA: fitted modelとunknown suffix predictionを
  作らないためStage 0では対象外。
- Kaggle bootstrap: package作成後にembedded config/source/metadata SHAを照合する。
- deterministic anchor: 初回PASSでは主張せず、同じinput SHAのrerunで
  edge/cycle/gauge SHA一致を要求する。

## リスク

- リークリスク: donor graphへのouter-valid混入、suffix truth、target Formation、
  hidden-like roleの早期read。role-read ledgerをhard gateにする。
- 科学リスク: GRの反復模様が偽edgeを作り、閉路整合だけでは正しい対応を識別できない。
  circular controlとprefix rolling-originをANDで要求する。
- loop closureリスク: cycle residualの単位不整合でexp386と同様に全routeを失う。
  raw/solved residualをTVT ftへ統一し、edge rejection funnelを保存する。
- CV/LB不一致リスク: Stage 0は16 wellsのprefix evidenceであり、deployable
  suffix精度ではない。PASSしてもLB候補やcurrent-test候補と呼ばない。
- ランタイム/メモリリスク: donor pair × window × shift探索。
  12 donors、top4 edge、2-well並列とfull投影gateを固定する。
- control計算リスク: exp226 geometry controlは5 outer foldsのfield/Kappa再構築を
  含む。fixed16 pseudo-cut以外を予測せず、Stage 0 diagnosticの合計runtimeを別記録する。
  773-well Stage 1投影はfull実行対象になるtarget-free graph時間だけを線形外挿し、
  prefix control時間を混ぜない。
- 再現性リスク: equal NCC、sparse solve、connected-component gauge。
  stable tieと固定IRLS iteration、component最小key anchorを使う。
