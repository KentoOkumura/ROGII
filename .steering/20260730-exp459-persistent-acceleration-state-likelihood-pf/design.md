# exp459 設計

## 1. 検証する問い

exp404/417 likelihood-PFの各粒子を

```text
(TVT, U-rate)
```

から

```text
(TVT, U-rate, persistent U-acceleration)
```

へ拡張すると、exact HMMの3次元格子を使わずにrate trendを曖昧GR区間へ持続できるかを
検証する。

```text
U = TVT + Z
U-rate = d(TVT + Z) / dMD
U-acceleration = d(U-rate) / dMD
```

exp444は同じ科学仮説をexact HMMで表現したがruntime gateで停止した。exp459は
solver高速化ではなく、有限粒子近似という異なる推論器で同じ状態仮説を直接検証する。

## 2. 固定状態と遷移式

acceleration状態は3値とする。

```text
a = [-0.0005, 0, +0.0005] rate/MD-ft
```

遷移行列は次で固定する。

```text
P =
  [[0.92, 0.08, 0.00],
   [0.08, 0.84, 0.08],
   [0.00, 0.08, 0.92]]
```

境界から外向きの`0.08`はboundary stayへ加える。更新順は次のとおり。

```text
delta_MD = max(MD_t - MD_{t-1}, 1.0)
delta_Z  = Z_t - Z_{t-1}

a_t      ~ P(a_t | a_{t-1})
rate_t   = 0.998 * rate_{t-1}
           + a_t * delta_MD
           + 0.002 * Normal_rate(0, 1)
TVT_t    = TVT_{t-1}
           + rate_t * delta_MD
           - delta_Z
           + 0.005 * Normal_position(0, 1)
```

これは内部で`U_t=TVT_t+Z_t`を持ち

```text
U_t = U_{t-1} + rate_t * delta_MD + position_noise
```

と更新する実装と同値である。

初期TVT / rateはexp404を固定する。accelerationは全粒子zeroから開始する。
resamplingでは選択粒子のacceleration stateを複製し、position/rateだけを既存量で
roughenする。

## 3. 固定するPF契約

- 500 particles、128 stable seeds。
- initial TVT spread `4.5 ft`、initial rate spread `0.01`。
- momentum `0.998`、rate noise `0.002`、position noise `0.005`。
- ESS threshold `0.5`、rough position/rate `0.1 / 0.001`。
- exp404 x1.0 Gaussian GR emission、GR scale clip `[10,60]`。
- missing GR補間、Type Well grid/pad、float32 output。
- primary readoutはtemperature `5.0`のfull-suffix seed evidence aggregation。
- arithmetic meanはprimary、fallback、promotion candidateとして使わない。
- 保存exp404 scale-5 x1.0をcontrolとし、control PFは再実行しない。

変更する科学変数はpersistent acceleration stateの追加だけである。

## 4. 乱数streamとzero-acceleration parity

acceleration drawを既存PFの逐次乱数streamへ追加すると、rate/position noiseや
resampling uniformの消費順が変わり、zero accelerationでも親と比較できなくなる。
したがって次の2 streamを分離する。

- base PF stream:
  `sha256_first16("likpf::train::<well_id>") + seed_index`
- acceleration stream:
  `sha256_first16("exp459::acceleration::<split>::<well_id>::<seed_index>")`

well / row / seed / particle順を固定し、thread schedulingに依存させない。
zero-acceleration sentinelでは3 acceleration値をすべて0へ置くが、base streamへ
一切影響させず、exp404 transitionとpredictionのbitwise parityを要求する。

## 5. 段階設計と実行量

### Stage 0

exp411 fixed32を使うtechnical / mechanism preflightであり、CVではない。

- scientific variants: 1
- candidate PF well-runs: 32
- seed-well trajectories: `32 × 128 = 4,096`
- particle starts: `4,096 × 500 = 2,048,000`
- zero-acceleration sentinel: 4 wells、科学候補外
- saved control PF rerun: 0
- LightGBM config / trained fold / booster / HMM / Beam / GPU: 0

prediction、acceleration ledger、runtime ledger、全content SHAをfreezeするまで、
suffix truth、error、fold、episode、hidden-like roleを読まない。

Technical AND gate:

- acceleration state / transition / row-sum / boundary contract。
- update orderと`-delta_Z` identity。
- zero-acceleration exp404 bitwise parity。
- base / acceleration stream分離とstable seed identity。
- finite prediction coverage `1.0`。
- full runtime投影`<=30,600 sec`、peak RSS`<=25 GB`。

Mechanism AND gate:

- 平均nonzero acceleration mass `0.01--0.80`。
- posterior accelerationとfuture rate-curvature方向一致`>=0.60`、4/5 folds。
- persistent episode SSE削減`>=5%`。
- persistent改善`>=10/16 wells`、`>=4/5 folds`。
- matched control pooled regression`<=+0.02 ft`。
- matched control by-well delta p95`<=+0.25 ft`。

### Stage 1

Stage 0全PASSと別承認がある場合だけ同じ実験内で実装・実行する。

- scientific variants: 1
- candidate PF well-runs: 773
- seed-well trajectories: `98,944`
- particle starts: `49,472,000`
- planned CPU shards: 4
- saved control PF rerun / model / booster / HMM / Beam / GPU: 0

## 6. Stage 1 promotion gate

全条件を満たす場合だけ後続候補とする。

- exp404/417 scale-5 x1.0 RMSE `10.914522073`から`0.05 ft`以上改善。
- 4/5 folds以上で改善。
- raw-GR observedで`0.05 ft`以上改善。
- persistent episode SSEを`5%`以上削減。
- raw-GR missing、高missing wells、1000+、hidden-like spatial、
  hidden-like typewell-purgedの各scopeでregression `<=0.0 ft`。
- by-well delta p95 `<=0.0 ft`、worst-well regression `<=0.25 ft`。
- exp209 HMMとの固定50:50 blendが保存control `10.084909680`より非悪化。

一つでもFAILならbranchを閉じ、acceleration値、transition、particle / seed、
temperature、GR emission、noise、resampling、roughening、gate、blend、selector、
MLで救済しない。

## 7. 再現性設計

- stochastic処理:
  particle initialization、rate/position noise、discrete acceleration transition、
  systematic resampling、position/rate roughening。
- 並列処理:
  per-well / per-seed明示streamを渡し、global RNGとthread順序へ依存させない。
- device:
  CPU固定。GPUは使用しない。
- raw train / raw test:
  別生成し、row/well count、schema、logical content SHAをそれぞれ保存する。
- 記録:
  input、scientific contract、acceleration ledger、prediction、well audit、
  runtime ledgerのschema/content SHA。gzipはdecompressed content SHAを主証拠とする。
- deterministic anchor:
  初回runでは主張しない。独立rerunでpredictionとledger SHAが一致した場合だけ検討する。
- Kaggle bootstrap:
  push前にpackage notebookの埋め込みconfigと正の`config.yaml`が一致することを確認する。

## 8. リスク

- リークリスク:
  exp411 fixed32はerror mechanism選択sampleなのでStage 0をCVと呼ばない。
- 科学リスク:
  wrong acceleration trendを持続させ、PF mode slipを悪化させ得る。
- 識別リスク:
  exp367ではfixed curvature pathとcircular controlのGR識別差が不足した。
- particle degeneracy:
  resamplingでacceleration minority stateが消える可能性がある。
- runtime:
  state格子の3倍化はないが、acceleration drawとledger保存でCPU/RSSが増える。
- 再現性:
  acceleration drawがbase streamを進める実装はzero-parityを破壊するため禁止する。

## 9. 承認境界

2026-07-30の最初の依頼でbacklog、steering、実験scaffold、design freezeまでを
承認済みとする。同日の追加依頼`exp459を実装してください`により、PFコード、
Jupytext source、contract test、正規train Notebook採用までを承認済みとする。
同日の追加依頼`実行してください`により、canonical Kaggle package / pushと
fixed32 Stage 0実行までを承認済みとする。Stage 1、inference、submissionは
未承認であり、Stage 0全PASSでも自動実行しない。
