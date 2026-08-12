# 設計

## 仮説

trainでは利用可能なformation 6面のwithin-well変化が`Z + TVT`の変化とほぼ一致する。
したがって未知suffixのTVT予測は、GRから多数のTVT曲線を選ぶ問題ではなく、共通formation surface
`F(X,Y)`のanchor-relative変位を復元する問題として解ける。

exp226の主な損失は少数wellのpersistentなwhole-well biasへ集中している。通常のKNN/平滑surfaceは
断層を跨いだdonor平均でこのtailを作るため、空間edgeを連続層内edgeとfault-crossing edgeへrobustに
分けるpiecewise-smooth scalar potentialを全hidden-like well同時に推定すれば、平均的な局所精度ではなく
catastrophic block-offsetを直接減らせる、という仮説を検証する。

## 数理モデル

canonical潜在surfaceを`F(X,Y)`、well固有のvertical datumを`c_w`とする。

```text
ANCC_w(r) = F(X_w(r), Y_w(r))                         [outer-train observation]
Z_w(r) + TVT_input_w(r) + c_w = F(X_w(r), Y_w(r))   [hidden-like known prefix]
TVT_hat_w(r) = TVT_input_w(a)
               + F_hat(X_w(r),Y_w(r)) - F_hat(X_w(a),Y_w(a))
               - (Z_w(r)-Z_w(a))
```

`a`は最後のfinite `TVT_input` row。`c_w`はprefix data termのrobust locationとしてjoint推定し、
予測式ではanchor差分により相殺される。

MAP objectiveは一つとする。

```text
outer-train ANCC data loss
+ hidden-like prefix (Z + TVT_input + c_w) data loss
+ along-well first/second-difference smoothness
+ spatial-edge truncated-quadratic gradient-consistency loss
+ connected fault-cut total-variation penalty
```

観測/data lossはStudent-t、spatial edgeはouter-train residual MADで正規化したtruncated quadraticとする。
大残差edgeは重みが0へ近づきfault cutとして働き、近接edgeのcut weightにはTV penaltyを置いて孤立cutを
抑える。binary candidateや複数surfaceを出力せず、deterministic alternating IRLSの最終MAP解一つだけを返す。

## graph契約

- node: 各wellをMD順に固定16行strideでsamplingし、最後の既知row、最終row、trajectory turning pointを必ず追加する。
- outer-train node observation: `ANCC`。他の5 formationはStage 0監査のみ。
- hidden-like node observation: known prefixの`Z + TVT_input`とlatent `c_w`。suffixのformation/TVTは存在しないものとして扱う。
- along-well edge: 連続sample node間。切断せず、first/second difference priorを持つ。
- spatial edge: XY標準化空間のdeterministic k=12近傍。self-well edgeを除き、distance、source well id、row idでstable tie-breakする。
- transductive scope: 各outer foldのvalidation wellを全件同時にgraphへ含め、hidden約200 wellsのbatch推論を模倣する。
- scale: XY、edge residual、data loss scaleはouter-trainだけのmedian/MADからfold別に推定する。
- optimization: fixed 8 outer iterations、各iterationでfault weight更新後に疎線形系をCPU float64で解く。収束判定はrelative objective `1e-6`、未収束はfail-closedでfallbackしない。
- row復元: graph nodeの`F_hat`を各well MD軸でpiecewise linear interpolationし、全evaluation rowへ一意に写す。

数値定数は`config.yaml`に置き、初回run後の同一OOF tuningは禁止する。実装時に数値的不可能性が判明し
scientific contractを変える場合は、実行前にsteering/configを更新してユーザー確認を得る。

## 段階実験

### Stage 0: fault topology association readout

1. 既存5-fold well splitを固定し、outer-validのformation 6列とprediction-target true TVTをdropする。
2. outer-train `ANCC`だけでspatial graphとrobust edge residualを作る。outer-validはXY trajectoryとknown prefixだけを追加する。
3. truthを結合する前に、wellごとのmax/mean/p90 fault-cut weight、cut crossing count、anchor以後初回cut距離をfreezeしcontent SHAを取る。
4. freeze後に保存済みexp226 by-well bias/RMSEを結合し、`abs(bias)>=10` AUC、Spearman、fold方向を読む。
5. 成功条件を一つでも外した場合はStage 1へ進まない。

これはoracle候補品質を測る処理ではなく、fault topologyが既知のcatastrophic bias故障をtarget-freeに説明するかの
反証可能な先行監査である。

### Stage 1: fault-aware transductive MAP、GRなし

Stage 0と同じfold/graph契約で、一つのMAP surfaceからdirect OOFを生成する。比較は保存済みexp226 OOFのみ。
exp226、HMM、PF、Beam、ML予測をsolverに入れない。metricsはoverall/fold/distance/hidden-like/by-well、bias、
well p90/p95/p99/worst、固定exp226 worst-66集合のMSE share、fault-risk bucketとする。

Stage 2事前guardを通過しても自動でGR実装へ進めず、結果とruntimeを記録して別承認を求める。
inference候補化guardを直接通過した場合も、raw-test 200-well shadow runtimeと再現性監査を別承認で行う。

### Stage 2: 条件付きjoint GR factor

Stage 1事前guard通過時だけ、known prefixから校正したType Well GR event factorを同じobjectiveへ追加する。
raw pointwise GR Gaussian、NCC hard match、shift candidate bankは使わない。CWT/局所極値で得たordered event列を
semi-Markov likelihoodとして弱く加え、amplitude、blur、noiseは各wellのknown prefixだけで推定する。
GR factor weightはouter-train prefix maskingだけで事前固定し、official outer-valid suffixで調整しない。

## 実験範囲

- 対象実験: `exp289_fault_aware_transductive_geological_potential`
- Route: `pf_beam`
- 親実験: なし。新規standalone physics family。比較・根拠は`exp226`、`exp138`、`exp280`、`exp282`、`exp285`。
- 変更する変数: fixed-direction/local donor fieldから、fault-cutを持つintegrable 2D scalar surfaceのjoint MAPへモデルクラスを変更する。
- 固定する変数: official 5-fold well split、evaluation mask、RMSE、raw coordinate/known-prefix contract、保存済みexp226比較OOF。
- 初回設計phaseで作成したもの: steering、backlog、experiment scaffold、config/README/SESSION_NOTES/result/metrics、summary記録。
- 追加実装phaseで作成するもの: Stage 0 Jupytext self-contained source/notebook、disabled inference source/notebook、leakage/freeze/graph専用tests。
- 追加実装phaseでも作成しないもの: Stage 1/2 solver、Kaggle package、train/inference output、submission。

## oracle禁止契約

- row/segment/block/well best-of-Nを計算しない。
- 候補数増加によるoracle改善を成功条件にしない。
- true TVT、true error、exp226 errorをgraph、fault weight、threshold、scale、stop iterationへ使わない。
- Stage 0 riskとStage 1 predictionをSHA freezeした後にだけtruth/errorをjoinする。
- diagnosticとしてのMSE分解はモデル性能や達成可能下限とは解釈しない。

## 再現性設計

- seed policy: canonical pathはRNGなし。fold、well、row、edgeをstable辞書順で処理する。
- stochastic処理の有無: なし。random sampling、PF、seed bagging、GPU boosterを使わない。
- PF/Beam / likelihood-PF / seed baggingの有無: すべてなし。route名はrepository上のphysical solver分類として`pf_beam`を使う。
- 並列処理と乱数の関係: canonical runはsingle process。将来parallel化する場合もgraph構築順と疎行列tripletをstable sortし、parity確認前はcanonicalにしない。
- CPU/GPU runtimeとdeterministic flags: CPU float64、GPU/internet off、BLAS thread数固定。疎solver library/versionをmanifestへ記録する。
- train cache / test regeneration SHA: fold map、input file、node table、edge table、outer-train observation、hidden-prefix observation、fault-risk、fault weight、surface node solutionのschema/content SHAをfold別に記録する。
- model manifest / prediction / submission SHA: solver config/objective/iteration historyをmodel manifest相当としてSHA化し、OOF/test prediction content SHAとsubmission SHAを分ける。Stage 0ではprediction/submissionなしを明記する。
- gzip: raw gzip SHAとdecompressed logical content SHAを分け、logical content SHAを主証拠にする。
- Kaggle package bootstrap: prepare後にembedded `config.yaml`、solver source、project config、CPU/GPU/internet metadataのSHA parityを確認する。今回はprepareしない。
- deterministic anchor: rerun parityとhidden raw-test regeneration parityが確認されるまでfalse。

## リスク

- リークリスク: train-only formationをouter-validへ残すと完全なlabel proxyになる。foldごとのwell除外、forbidden-column scan、truth-after-freezeをhard guardにする。
- CV/LB不一致リスク: transductive foldの約154 wellsとhidden約200 wellsで密度・空間分布が異なる。spatial/typewell-purged hidden-like readoutと200-well shadow runtimeを必須にする。
- モデルリスク: faultではなく滑らかなlong-range trend不足がcatastrophic biasの主因ならStage 0が失敗する。その場合は救済gridをせず仮説を棄却する。
- 識別リスク: well方位がほぼ平行で2D gradientが弱識別になる。absolute outer-train ANCC data termとprefix datumを使い、gradient-only inversionにはしない。
- 過平滑化リスク: ordinary KNN/GPと同様にfaultを跨ぐ。truncated-quadratic edgeとcut continuityを必須とする。
- 過切断リスク: edgeを切りすぎるとhidden suffixがunanchoredになる。connected-componentがouter-train observationまたはknown prefixを持たない場合はfail-closedとする。
- ランタイム/メモリリスク: 約30万node規模の疎graphを想定する。Stage 0でnode/edge数、peak RSS、solver見積もりを記録し、Kaggle 9時間へ1.5時間以上の余裕を要求する。
- 再現性リスク: kNN tie、疎行列構築順、BLAS thread、solver toleranceで差が出る。stable sort、single process、thread固定、float64、iteration log SHAで管理する。
