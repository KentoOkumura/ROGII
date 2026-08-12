# 設計

## 1. Variant A: absolute geometry unary

exp404のposition/rate stateとGaussian GR likelihoodを完全に維持し、各particleの
TVTへexp279と同じexp226 geometry potentialを加える。

```text
g_t = group-safe exp226 tvt_geop
z_geo = (TVT_particle - g_t) / 20
log L_total = log L_GR + 0.50 * (-0.5 * min(z_geo^2, 600))
```

exp226 `tvt_pred`や`gr_delta`は使わない。これはproposalではなくfilter targetへ
absolute unaryを追加するvariantである。

## 2. Variant B: slow residual-offset state

particle stateを`(TVT+Z, rate)`から`(offset, offset_rate)`へ変更し、
観測用/出力用TVTを次で復元する。

```text
TVT_t = tvt_geop_t + offset_t
offset_rate_t = 0.998 * offset_rate_{t-1} + 0.002 * Normal(0,1)
offset_t = offset_{t-1} + offset_rate_t * delta_MD + 0.005 * Normal(0,1)
```

- initial offset center:
  `last_known_TVT - tvt_geop_at_first_score_row`。
- initial offset spread: exp404 position spread `4.5 ft`。
- initial offset-rate center: exp281の固定`0.0`。
- initial offset-rate spread: `0.01`。
- GR emissionは復元TVTで評価し、exp404 capped Gaussian/x1.0を維持する。

これは単なる座標変換ではなく、geometry path周りのslow offsetを0へ戻す
exp281由来のdynamics仮説である。

## 3. 二variantの扱い

両variantは同じ保存exp404 controlに対して独立に判定する。二つの予測を同じOOFで
比較してwinnerを採用せず、片方または両方がgateをPASSしても、後続利用は別steeringと
別承認を必須とする。blend、row/well switch、geometry confidence gateは本実験外。

## 4. Stage 0とmechanism readout

- 2 variants ×32 wells = 64 PF well-runs。
- 8,192 seed-well trajectories、4,096,000 particle starts。
- 保存control rerun 0。
- absolute unary: geometry log-weight分布、ESS/resampling差、geometry残差分位。
- residual state: offset/offset-rate分布、edge/finite、geometry deltaとparticle drift、
  exp226 support coverage。
- fixed32はtechnical/mechanism preflightでCVではない。

## 5. Stage 1

全Stage 0 gate PASS・別承認時だけ、
2 variants ×773 wells = 1,546 PF well-runs、197,888 seed-well、
98,944,000 particle startsを実行する。control PF、HMM、Beam、model、booster、GPUは0。

各variantが独立に次を全て満たすことを要求する。

- exp404 scale-5 x1.0 `10.914522073`から`0.05 ft`以上、4/5 folds以上改善。
- raw observed `0.05 ft`以上改善。
- raw missing、高missing、1000+、hidden-like 2面のregression `<=0.0 ft`。
- by-well p95 `<=0.0 ft`、worst `<=0.25 ft`。
- exp209 HMMとの固定50:50 blend `10.084909680`より非悪化。

exp226 direct RMSEはreferenceとして報告するが、本PF auditのcontrolとはしない。

## 6. Leakage・再現性・禁止事項

- exp226 OOFは`usecols`でallowlistだけを読む。foldはprediction freeze後だけattachする。
- trainはgroup-safe OOF `tvt_geop`、testは同じexp226 geometry生成式をraw入力から
  再生成し、train/testを別SHAとする。inference実装自体は未承認。
- exp404 stable seeds、fixed order、T=5、logical/decompressed SHAを継承する。
- geometry sigma/lambda、offset initialization/noise、particle/seed、temperature、
  unary+residual併用、winner selection、blend/gate、same-OOF救済は禁止。
- 2026-07-30の追加依頼`exp486を実装してください`により、compact
  self-contained Stage 0候補、fail-closed inference guard、contract testまでを
  承認済みとする。正規Notebook採用、Kaggle package / push / run、Stage 1、
  raw-test inference、submissionは未承認である。

同日の追加依頼`実行してください`により、compact候補の正規train Notebook採用、
canonical Kaggle package / push、fixed32 Stage 0実行までを承認済みとする。
Stage 1、raw-test inference、submissionは未承認のままとする。

## 7. 実行後の終端

Kaggle private CPU version 1はfixed32を完走したが、事前固定runtime投影
`180,871.020 sec > 30,600 sec`とstrict residual support boundをFAILした。
support超過は最大約`1.1e-15`の浮動小数誤差だが、独立したruntime FAILがある。
終了後にgate式やtoleranceを変更せず、`stage0_fail_closed`でbranchを閉じる。

## 8. ユーザー承認によるStage 1例外

ユーザーがruntimeを明示的に許容してStage 1進行を指示したため、元の
`runtime_projection=false`を再分類せず、例外付きで全well CVを実行する。
support strict-boundは実測min/max
`0.9999999999999988 / 1.0000000000000011`で、物理的なsupport逸脱ではなく
正規化weight和の丸め誤差である。Stage 1では`[-1e-12, 1+1e-12]`だけを
technical readback toleranceとし、Stage 0のoriginal checkはFAILのまま残す。

実行契約:

- 2 variants ×773 wells = 1,546 candidate PF well-runs
- 197,888 seed-well trajectories
- 98,944,000 particle starts
- saved control PF / HMM / Beam / model / booster / GPU rerun 0
- exp226 geometryは両variant freeze前にallowlist 4列だけ読む
- truth、保存exp404、保存exp209、fold、hidden-like roleは両variant freeze後だけ読む
- variant別CV/fold/scope/by-well/fixed HMM-PF 50:50 gateを独立判定する
- winner選択、blendによる候補統合、parameter/gate救済は行わない

## 9. version 2 freeze recovery

version 2は上記98,944,000 particle startsを完了し、両variantのpredictionと
mechanism ledgerをtruth/control/fold/hidden-like read 0の状態でfreezeした。
その後、exp209期待SHAの62文字manifest typoでtruth-late readoutが停止した。

freeze artifactのraw/decompressed/logical SHAを固定したprivate Dataset
`kentookumura/exp486-v2-stage1-frozen-targetfree`をversion 3の入力とする。
version 3はscientific contractを変更せず、current PF rerun 0で全SHAと
1,546 variant-well identityを復元してから同じtruth-late gateだけを再開する。
