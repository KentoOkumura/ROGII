# 設計

## 仮説

各horizontal well `w`、formation `f`、row `r`について、

\[
U_w(r)=TVT_w(r)+Z_w(r),\qquad
U_w(r)=T_{w,f}+S_{w,f}(r)+c_w
\]

と書けるなら、同一wellの2点`a,b`の差分ではformation top `T`とwell datum `c_w`が消え、

\[
U_w(b)-U_w(a)=S_{w,f}(b)-S_{w,f}(a)
\]

になる。6 formationの差分中央値を観測し、それらを最もよく積分する共通2D scalar field
`Phi(X,Y)`を求めれば、known prefix末尾`a`から未知suffix`r`を

\[
\widehat{TVT}_w(r)=TVT_{input,w}(a)+Z_w(a)
 +\Phi(X_w(r),Y_w(r))-\Phi(X_w(a),Y_w(a))-Z_w(r)
\]

で直接予測できる。絶対formation surfaceを合わせる従来法ではなく、gauge-invariantなedge observationを
積分することが中核である。

## exp289との差分

- exp289はANCC絶対面、latent well datum、known prefix observation、fault-risk / fault cutを主仮説にした。
- exp301は6 formationの井戸内差分だけを使い、datumを変数として持たない。
- exp301 Stage 1はfault detection/cut、GR、existing predictionを一切使わないordinary smooth integrable fieldである。
- exp289 Stage 0 FAILのthreshold、risk feature、ANCC surfaceを変更する救済ではない。

## Stage 0: 物理恒等式とsupportの監査

solver実装前に、outer-trainだけを使って次をfold別に監査する。

1. rowを`MD`昇順、元row indexでstable sortする。
2. stride 16でnodeを取り、各wellの最終rowも必ず含める。
3. 隣接sample node間について各formationの`delta S_f`とtrue `delta U`を計算する。
4. formation別と6面中央値について`delta U-delta S`のRMSE、MAE、bias、MAD、finite coverageを出す。
5. outer-validはformation/TVTをdropしたsafe loaderでgeometry supportだけを監査する。

Technical PASSは、各foldで禁止列hit 0、solver fit前のouter-valid truth access 0、prediction suffix identity
3,783,989 rows / 773 wellsを満たすことに加え、各formationと6面中央値のedge identity RMSEが全fold`<=0.02 ft`、
有効edgeの`>=99.5%`で3 formation以上がfinite、全query rowにbilinear basisが定義され、donor constraintを含む
active grid componentへ属すること。
1項目でもFAILならStage 1 solverは実装・実行せずbranchを閉じる。

`0.02 ft`はpreliminary local readoutで約0.007 ftだった観測を再現確認するためのtechnical toleranceであり、
そのpreliminary値自体は実験結果として扱わない。

## Stage 1: gauge-invariant multiformation edge potential

### Observation

- outer-train wellのみをdonorにする。
- row strideは16、最終rowを必ず含め、隣接sample nodeをedgeとする。
- edge responseはfiniteな6 formation差分のmedian。3面未満しかfiniteでないedgeは除外してmanifestへ記録する。
- edge scaleは`max(1.4826 * MAD_f(delta S_f), 0.02 ft)`。
- formation絶対値、typewell top、well datumはsolverへ渡さない。

### Field discretization

- outer-train donorとquery geometryのXY unionを250 ft正方gridへ写像する。queryはgeometryだけを使うtransductive scopeとする。
- trajectoryからChebyshev距離1 cell以内をactive nodeとし、active gridを4-neighbor componentへ分ける。
- edge endpointと全query rowの`Phi`は同じbilinear basisで評価する。
- 各componentは少なくとも1 donor edgeを含むことを必須とし、fallback predictionは作らない。
- componentごとにpotential平均0を課す。予測はpotential差だけなのでgauge固定値は結果に影響しない。

### Objective

grid node potential `phi`を次で求める。

\[
\min_\phi
\sum_e \rho_{1.345}\!\left(\frac{(B_b-B_a)\phi-d_e}{s_e}\right)
+\lambda\lVert D_2\phi\rVert_2^2
\]

`B`はbilinear basis、`d_e`は6 formation差分中央値、`D2`はactive grid上の`xx, yy, xy`二階差分。
二階差分を使うため定数面と平面を不当に0へ縮めない。Huber IRLSは最大20反復、relative objective change
`1e-6`で停止、stable sparse ordering、float64、single process CPUで決定論的に解く。

`lambda`はouter-valid TVTを使わず、outer-train wellのSHA256 stable 3-way inner splitに対するheld-out formation-edge
Huber lossで`[1e-3, 1e-2, 1e-1]`から選ぶ。同値は大きいlambdaを選ぶ。grid 250 ft、stride 16、Huber 1.345、
inner split数、lambda候補は固定し、同じOOF上で変更しない。

## CVとreadout

- outer foldはexp226保存OOFのwell-fold identityを再利用する。
- outer-valid fold全wellをdonor、robust scale、inner selectionから完全除外する。
- outer-valid/test own formation columnsとGRはsafe loaderでdropし、アクセスattemptも0を記録する。
- prediction content SHAをfreezeした後にだけtrue TVTとexp226比較値をlate joinする。
- Stage 1 technical guardとしてrow identity 3,783,989 / well identity 773、duplicate 0、finite prediction coverage 1.0を要求する。
- primaryはpooled direct RMSE。fold、distance bucket、1000+、hidden-like spatial、hidden-like typewell-purged、
  by-well p50/p95/worst、well biasを必須reportにする。
- exp293 deployable12はpost-freeze diagnosticにだけ使い、exp301を1候補追加したH512 oracle headroomを読む。
  bank候補値は変更せず、oracle predictionは保存しない。

## 事前登録した成功条件

Stage 1 PASSはdirect qualityとcandidate noveltyの両方を満たす場合だけとする。

Direct quality:

- pooled RMSE `<=9.2271095966 ft`（saved exp226 9.4271095966より0.20 ft以上改善）。
- exp226より5/5 outer foldsで改善。
- 1000+、hidden-like spatial、hidden-like typewell-purged、by-well p95が全てexp226非悪化。
- worst wellのexp226比delta `<=+0.25 ft`。

Candidate novelty:

- exp293 fixed deployable12へexp301だけを加えた非重複H512 oracle RMSEがfixed12より`>=0.10 ft`改善。
- H512 oracleが4/5 folds以上で改善。
- exp301がstrict unique-bestになるH512 block比率`>=2.0%`。

一方だけPASSしてもexp301 inference、案2、案3へ自動進行しない。grid/stride/lambda/formation subset/Huber/fallback/blendを
同じOOF truthに合わせて救済せず、negative resultとして閉じる。

## 実験範囲

- 対象実験: `exp301_gauge_invariant_multiformation_edge_potential`
- Route: `pf_beam`
- 親実験: なし。比較anchorはexp226、独立性比較はexp289、candidate novelty比較はexp293。
- 変更する変数: formation絶対面から6面edge差分への観測変更、2D integrable scalar field、direct candidate。
- 固定する変数: outer fold、score rows、known-prefix anchor、6 formation、grid/stride/objective/inner lambda contract、success gate。
- 2026-07-20設計セッションの範囲: 設計とfail-closed scaffoldだけ。
- 2026-07-20実装依頼の範囲: solver、audit、OOF prediction、事後readoutを実装する。ただし実行はStage 0 PASS時だけStage 1へ進む。Kaggle実行、inference、submission、案2/案3は別承認まで行わない。

## 後続分岐契約（案2 / 案3）

正本は`experiments/exp301_gauge_invariant_multiformation_edge_potential/reserved_followup_contract.md`とする。
案2はexp301 PASS後にだけ、exp301 posterior energyでexp293 fixed deployable12のH512候補を選ぶ物理evidence実験である。
案3はexp301 PASSかつexp295自身のStage B promotion PASS後にだけ、exp295の学習済みunary/transitionを固定して
exp301 priorを加え再decodeするensemble実験である。番号、実装、実行は未予約であり、この2案をexp301へ混ぜない。

## 再現性設計

- seed policy: RNGなし。well/row/edge/grid/component/sparse indexをstable sortし、inner splitはSHA256で固定する。
- stochastic処理: なし。
- PF/Beam / likelihood-PF / seed bagging: なし。route分類だけ`pf_beam`。
- 並列処理: 初回はsingle process。thread scheduling依存を避け、sparse reduction順を固定する。
- runtime: Kaggle private CPU、GPU/internet off。実装前にnode/edge/nnz/推定runtime/memoryのguardを追加する。
- input SHA: raw file、fold map、exp226 OOF、exp293 bank manifest、hidden-like assignmentを記録する。
- content SHA: safe-loader schema、donor edge logical content、active grid、component、lambda selection、predictionを記録する。
- gzip: raw file SHAに加えdecompressed content SHAを必須にする。
- model SHA: learned modelはない。solver config、sparse structure、solution vectorをmodel-equivalent manifestとしてhashする。
- prediction/submission: OOF prediction SHAを記録する。inference/submissionは別承認まで生成しない。
- Kaggle bootstrap: loose/package/bootstrap config SHA一致をpush前に確認する。
- deterministic anchor: rerun parityと全SHA一致を確認するまではfalse。

## リスク

- identifiability: difference observationだけではcomponent定数が不定だが、anchor-relative predictionでは相殺される。
- support: donorのないquery componentは外挿不能。既存prediction fallbackで隠さずtechnical FAILにする。
- oversmoothing: faultを跨ぐ可能性はあるが、exp301ではfault cutを追加しない。tail guardで反証する。
- discretization: 250 ft grid以下のvariationを失う可能性がある。outer-valid truthでgridを救済しない。
- transductive scope: query XYは使うがformation/GR/TVTは使わない。schema allowlistとSHAで固定する。
- CV/LB: public testは3 wellsのためCV-LB差でgateを緩めない。
- runtime/memory: inner 3-way×3 lambdaと5 outer foldsのsparse IRLSが重い。実装時に事前nnz guardを置くが契約を変えない。
