# 設計

## アプローチ

各outer-train rowで`S=TVT+Z`を計算し、固定地層順
`ANCC→ASTNU→ASTNL→EGFDU→EGFDL→BUDA`のどの区間にあるかと、
隣接面の間のfractionから連続RGTを定義する。地層面を6本別々にtargetへ補間するのではなく、
64 ft window / 32 ft strideでRGT進行、地層間厚さ、構造位置、坑井方向をgraph nodeへ集約する。

outer-train間だけに近傍24 unique wellsのedge候補を作り、positive RGT progress、
formation interval identity、trajectory direction、outer-train LOOのq05--q95 stretchを
固定制約とする。fundamental cycle basis上の閉路残差を同時に抑え、坑井対ごとの局所対応が
全体で矛盾しないgraphを作る。

outer-valid/testは`MD/X/Y/Z/TVT_input`だけでgraphへ接続し、known prefixをgauge anchorとする。
target GRは使用しない。deterministic k-shortest pathで総cost順の候補を列挙し、
path RMS差0.5 ft未満を重複として除外する。最低8、最大32 scenarioを保存する。
候補をtruthで選ばず、graph cost、cycle cost、prefix misfit、stretch costから作るpriorだけを添付する。

## 実験範囲

- 対象実験: `exp386_cycle_consistent_rgt_scenario_bank`
- Route: `pf_beam`
- 親実験: なし。`independent_topology_first_rgt_physics_family`
- 比較対象: exp226 direct path、exp293 fixed candidate bank、exp301 potential、
  exp377 formation-relative、exp383 vector field。
- 変更する変数: TVT補間をglobal RGT correspondence graphと複数scenario生成へ置換する。
- 固定する変数: exp226 outer 5-fold、地層順、node/edge/stretch/scenario/diversity設定、評価scope。
- 出力: deployable predictionではなく、固定scenario bank、prior cost、donor mapping、
  exp387用outer-train reference-GR template。
- 実行量: 1 scientific variant / 5 graph solves / 773 target-well path solves /
  scenario 8--32 / model・HMM・PF・Beam・booster各0。

## 段階と停止条件

1. Stage 0 target-freeでrole read、coverage、cycle残差、scenario数、resourceを監査する。
2. 全PASS時だけknown prefixの末尾512行を隠すrolling-originを評価する。
3. rolling-origin PASS時だけscenario bankをSHA freezeし、suffix truthをlate joinする。
4. scenario oracle `<=5.50 ft`と全fold/coverage/diversity gateを満たした場合だけexp387を開く。
5. いずれかFAILならgraph/edge/stretch/count/diversityを同じOOFで救済せず閉じる。

## 再現性設計

- seed policy: RNGなし。fold/well/node/edge/cycle/pathをimmutable keyでstable sortする。
- stochastic処理の有無: なし。
- PF/Beam / likelihood-PF / seed baggingの有無: なし。k-shortest pathは決定論的。
- 並列処理と乱数の関係: RNGなし。並列化する場合も結果をgraph/path keyで再sortする。
- CPU/GPU runtimeとdeterministic flags: CPUのみ、GPUなし。
- train cache / test feature regenerationのSHA記録方針:
  fold、node、edge、cycle、scenario path、reference-GR template、schema、logical content SHAを記録する。
- model manifest / prediction / submission SHA記録方針:
  fitted modelなし。scenario-bank manifestとpath prediction SHAを保存し、submission SHAは対象外。
- Kaggle package bootstrap確認方針:
  将来pushする場合はprivate / CPU / internet off、config/source/bootstrap SHAを照合する。
- deterministic anchor:
  初回成功runでは主張せず、成功rerunのgraph/scenario/prediction SHA一致を要求する。

## 実装時に固定した細部

- RGT は各rowで formation surfaceを `TVT + Formation` の構造座標へ写し、
  `S=TVT+Z` を固定順の隣接面間で線形補間する。面反転は補正せずunavailableとする。
- graph edgeはwell centroidの近傍24 unique wellsから作り、well pairごとに同一formation
  intervalのRGT対応を最大4 nodeでmedian集約する。
- well-level offsetをformation interval単位へ正規化し、Huber `delta=0.10`、固定5 iterationの
  sparse IRLSで同時に解く。stable Kruskal forestのnon-tree edgeをfundamental cycleとする。
- target scenarioは近傍source-well graphへvirtual start/endを接続し、単純pathを
  `(cost, immutable path key)` 順で最大128本列挙する。各routeの局所
  `dS/dMD` / `dRGT/dMD`をtarget control nodeへ積分し、最後のfinite prefixをhard anchorにする。
- prior 4成分は単一固定weight 1.0で加算し、prefix misfitだけ25 ftで無次元化する。
- Stage 2 oracleはexp293との比較契約とStage 1のheldout長に合わせ、512-row block単位に固定した。
- 正規Notebookは上書きせず、別名compact候補だけを生成する。実行時はconfigの
  authorization guardがfail-closedする。

## リスク

- リークリスク: target GRや生Formation、suffix truthがscenario costへ入ることが最大のリスク。
  role read ledgerとfreeze前0-readをhard gateにする。
- CV/LB不一致リスク: scenario oracleはdeployable scoreではない。exp387のtarget-free GR posteriorが
  成立するまでLB候補と呼ばない。
- ランタイム/メモリリスク: 全坑井pair graphと32 pathsの組合せ爆発。
  edge候補96、24 unique wells、16-well projectionを固定してfull前に止める。
- 再現性リスク: equal-cost pathとgraph traversal順。全tieをimmutable keyとSHAで解消する。
- 物理リスク: targetにformation面を与えないためgraphから複数解を十分生成できない可能性がある。
  oracle `<=5.50 ft`を厳格な反証条件とする。
