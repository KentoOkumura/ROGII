# exp459_persistent_acceleration_state_likelihood_pf

## 状態

- Route: `pf_beam`
- 状態: `stage0_fail_closed`
- 優先度: P3
- CV / Public LB / Private LB: なし
- 親: exp417
- PF実装参照・保存control: exp404
- acceleration科学参照: exp444

## 仮説

各粒子に`TVT`、`U-rate=d(TVT+Z)/dMD`、persistentな3値
`U-acceleration`を持たせると、GRが局所的に曖昧な区間でもrateの上昇・一定・
下降trendを維持できる。exact HMMのようにposition×rate×acceleration格子を
全列挙せず、500粒子内で近似するため、exp444の状態数3倍runtimeを回避できる。

## 固定状態遷移

```text
acceleration = [-0.0005, 0, +0.0005] rate/MD-ft
transition =
  [0.92, 0.08, 0.00]
  [0.08, 0.84, 0.08]
  [0.00, 0.08, 0.92]

a_t   ~ P(a_t | a_{t-1})
rate_t = 0.998 * rate_{t-1} + a_t * delta_MD + 0.002 * Normal(0, 1)
TVT_t  = TVT_{t-1} + rate_t * delta_MD - delta_Z
         + 0.005 * Normal(0, 1)
```

初期accelerationは全粒子`0`。resampling時は選ばれたacceleration状態をそのまま
複製し、acceleration rougheningは行わない。GR尤度、500 particles、128 seeds、
position/rate roughening、ESS threshold、temperature-5 seed集約はexp404/417から
変更しない。

## exp444 / exp367との関係

- exp444の3値accelerationと遷移確率を一要因比較のため固定する。
- exp444はexact-HMM runtime FAILのまま再分類せず、solver救済もしない。
- exp367はfixed signed-curvature pathのGR識別gate FAILのまま保持する。
- exp367のpath、score、triggerは本PFへ入力しない。

したがって本実験は、閉鎖済みbranchの再開ではなく、同じpersistent acceleration
仮説を有限粒子近似で直接検証する独立実験である。

## 検証方針

Stage 0はexp411 fixed32を使うtechnical / mechanism preflightでありCVではない。
Kaggle private CPU version 1（id_no `129167965`）で完了した。

- candidate: 1 variant × 32 PF well-runs
- 128 seeds × 500 particles
- 保存exp404 controlの再実行: 0
- LightGBM / model / booster / HMM / Beam / GPU: 0
- zero-acceleration sentinel: 4 wells、科学候補外

acceleration mass、将来rate curvatureとの方向一致、persistent episode SSE、
matched-control安全性、runtime/RSSを全AND判定した。technical gateは全PASSしたが、
方向一致`0.501086`、persistent SSE reduction `-11.6190%`、matched-control
pooled delta `+0.435213 ft`でmechanism gateをFAILした。Stage 1 eligibleはfalse。

## 再現性

既存PFのbase乱数streamとacceleration遷移streamを分離する。acceleration drawが
rate/position/resamplingの乱数消費をずらさないため、zero-acceleration sentinelで
exp404とのbitwise parityを要求できる。

raw train / raw testは別生成として扱い、入力、scientific contract、
acceleration ledger、prediction、well audit、runtime ledgerのschema/content SHAを
保存する。初回runはdeterministic anchorとしない。

## リスク

- 3値accelerationがwrong trendも持続させる。
- resamplingで少数acceleration modeが消える。
- exp367のnegative control結果から、GRが曲率符号を識別できない可能性がある。
- fixed32はmechanism選択sampleなので、pooled値をCVやroute anchor更新に使えない。

## 成果物

compact self-contained train source、正規train Notebook、contract testを実装済み。
PFコードには3値acceleration遷移、独立RNG、zero-acceleration exp404 parity、
target-free prediction/acceleration/runtime freeze、truth-late mechanism readout、
Stage 0 fail-closed gateを含む。Kaggle Stage 0の生成物はversion 1 outputに保存され、
ログにcontent SHAを記録した。inference、提出物はない。

## 所見

acceleration stateはnonzero mass `0.666245`を維持し、実装・parity・runtimeは
成立した一方、将来curvature符号を識別できず、persistent区間とmatched controlを
悪化させた。したがって有限粒子近似にしてもpersistent acceleration仮説を支持せず、
exp444のexact HMM runtime FAILとexp367のsigned-curvature識別FAILを補強する
negative resultとして扱う。ただしfixed32はCVではなくroute anchorを更新しない。

## 次

branchを閉じ、Stage 1、inference、submissionを実行しない。parameter、transition、
noise、particle、seed、temperature、GR emission、gate、blend、selectorによる
same-fixed32救済を行わない。
