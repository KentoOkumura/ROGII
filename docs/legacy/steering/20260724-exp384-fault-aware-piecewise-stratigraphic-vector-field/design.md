# 設計

## アプローチ

exp383のsmooth stratigraphic vector fieldは、faultを跨ぐdonorを同じlocal planeへ入れると
両側を平均する。本実験ではexp383の256 ft primary donor nodesをgraph化し、
6地層surface geometryと正解`S=TVT+Z` residualが同時に不連続なedgeを切る。

得られたpiecewise domainごとにexp383と同じabsolute/vector fieldをfitし、
targetではsurface signatureと既知prefix misfitからdomain posteriorを作る。
出力はhard domainではなく、smooth base fieldを必ず残したposterior平均とする。

### 1. 先行条件と固定入力

- exp383 Stage 0/1の全gate PASSを必須とする。
- exp383のfold manifest、surface prediction、全TVT donor catalog、
  29次元signature、smooth field、prefix calibration、exp226 fallback、
  path solverのcontent SHAを固定入力にする。
- exp383を再実行せず保存済み生成物を使う。

### 2. outer-train fault graph

- graph nodeはexp383の256 ft primary donor nodesだけに固定する。
- 各nodeからXY最近傍12 unique-well nodesへ無向edgeを張る。
- 最大edge距離は4,000 ft、同一source well間edgeは禁止する。
- outer-trainだけで次をmedian/MAD標準化する。
  - 6 surface値の局所plane residual差
  - 5 adjacent thickness差
  - 12 surface gradient差
  - exp383 smooth absolute-field residual`S_true-S_smooth`
  - exp383 smooth vector-rate residual
- edgeを切る条件は次の固定ANDとする。

```text
formation_jump = mean(clip(abs(z_surface_thickness_gradient), 0, 6)^2) >= 9
structural_jump = max(abs(z_absolute_residual), abs(z_rate_residual)) >= 3
```

- cut後のconnected componentでunique wellsが8未満のものは独立domainにせず
  `small_component`としてsmooth baseだけへ戻す。
- unique wellsが8以上でもcut fault edgeに1本も接しないcomponentは
  `no_fault_component`としてpiecewise候補にせず、smooth baseへexact fallbackする。
- component IDは最小`(well_id, MD)`順で安定採番する。
- fault graphにouter-valid truth/formationを入れない。

### 3. domain field

- 各eligible componentでexp383と同じ6-surface relative absolute/vector fieldをfitする。
- queryごとのcomponent候補は最近傍8 componentsまで、各component最低12 unique wellsとする。
- component fieldのsurface/field/ridge/window/path設定はexp383と同一。
- component固有の追加学習パラメータは持たない。

### 4. target domain posterior

query nodeごとに、outer-train componentの29次元signature center/covariance、
XY distance、surface uncertaintyからtarget-free log weightを作る。

```text
logw_domain = -0.5 * d_signature^2 - 0.5 * d_xy^2 / h_component^2
```

- posterior temperatureは1.0に固定する。
- `h_component`はcomponent nodeのXY centerからの距離p75を
  exp383 surface bandwidth範囲`[500, 4000] ft`へclipした値に固定する。
- surface uncertaintyはcomponent field varianceをcomponent内surface variance p50で割った
  `-0.5 * log1p(ratio)`としてlog weightへ加える。
- smooth base fieldへ最低posterior mass 0.25を固定する。
- remaining 0.75を最大8 componentへsoftmax配分する。
- target prefixがあるcomponent pathについては、既知prefix上の
  `S_input-S_component` Huber RMSEを追加likelihoodとして全prefixで計算する。
- prefix likelihood scaleはouter-train leave-one-well-out prefix RMSE p50に固定する。
- suffix truthやGR/typewellをposteriorへ使わない。
- eligible componentがなければposterior base=1.0としてexp383へexact fallbackする。

### 5. path

absolute field、vector rate、uncertaintyをposterior平均してexp383と同じprefix校正、
exp226 shrink、banded path solverへ渡す。solver、curvature、query gridは変更しない。

## 実験範囲

- 対象実験: `exp384_fault_aware_piecewise_stratigraphic_vector_field`
- Route: `pf_beam`
- 親: exp383
- 変更する変数: fault graph、piecewise component field、domain posterior。
- 固定する変数: exp383の全入力、surface/catalog/field/prefix/fallback/path契約。
- 予定量: 1 piecewise candidate / 5 reporting folds / fitted ML model・HMM・PF・Beam・booster各0。
- parent control再実行0。
- Runtime: Kaggle CPU。実装と実行はexp383 PASS後の別承認。
- 2026-07-24の直接指示でコード実装と正規Notebook採用だけを先行した。
  exp383 manifest/SHA pin、Kaggle package/push/run、科学scoreは引き続き未承認。

## 検証段階

### Stage 0: target-free graph/domain integrity

- exp383 input SHA一致。
- outer-valid graph node/edge/reference count 0。
- valid生Formation/suffix truth read 0。
- graph/component/posterior finite coverage 1.0。
- eligible query coverage`>=0.80`。
- eligible component unique donor wells p05`>=12`。
- base posterior mass`>=0.25`。
- posterior row sum誤差`<=1e-12`。
- ineligible/no-fault rowのexp383 absolute/rate/path parity max abs`<=1e-8 ft`。
- 16-well projected runtime`<=30,600 sec`、peak RSS`<=25 GB`。

FAILならlate truth scoring、edge/threshold/component救済を行わない。

### Stage 1: direct piecewise path

- exp383比pooled RMSE gain`>=0.50 ft`。
- positive folds`>=4/5`。
- 1000+ gain`>=0.50 ft`。
- hidden-like spatial/typewell-purged gainが各`>=0.25 ft`。
- near 0--250 delta`<=+0.05 ft`。
- eligible rowsだけでexp383比gain`>=0.75 ft`。
- by-well tailとfault-domain別deltaは必須報告、初回signal gateではreport-only。

全PASSでexp385をunlockする。PASSしてもinference、submission、exp385実行は別承認。

## 再現性設計

- seed policy: RNGなし。node/edge/component/domain順をstable sort。
- stochastic処理: なし。
- PF/Beam: なし。
- CPU/GPU: Kaggle CPU / GPU off / internet off。
- SHA: exp383 input manifest、graph nodes/edges、cut flags、components、
  component fields、posterior、prefix likelihood、path、OOF predictionのlogical content SHA。
- parallelism: component並列後にcomponent ID/query MD順へ再整列する。
- deterministic anchor: 初回runでは主張しない。rerun graph/component/prediction SHA一致後に再評価。
- Kaggle bootstrap: 実装承認後にconfig/source/input SHAとrun flagsを照合する。

## リスク

- leakage: fault graphへouter-valid residual/formationを入れないrole guardが必須。
- target transfer: train truthで見えるfaultがtarget surface signatureから識別できない可能性がある。
- fragmentation: small componentをsmooth baseへ戻し、posterior base floorを0.25とする。
- CV/LB: fault分布がhiddenと異なる可能性があるためhidden-likeとdomain coverageを分離報告する。
- runtime/memory: 256 ft nodesだけでgraphを作り、全scale全row graphは作らない。
- post-hoc: edge閾値、k、distance、min wells、temperature、base floorを救済しない。
