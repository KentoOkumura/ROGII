# 設計

## 仮説

exp413で一度予測したTVT pathのK16低周波rateを、同じ予測対象batch内の他well予測から作る
空間合意へごく弱く縮めれば、真のdonor TVTを使わずにcross-well不整合を減らせる。

## アプローチ

保存済み exp413 outer-fold OOF prediction を一度完成した TVT path とみなす。
foldごとに outer-valid wells を同時に処理し、各wellのscore suffixについて
`U_base = pred_tvt + Z` の相対増分を K=16 の連続piecewise-linear basisへ射影する。
この係数をwell trajectoryのsegment midpoint上へ置き、exp226と同じXY local-linear
kernel kNNで、同じfold内の他well予測だけからtarget segmentの周辺合意係数を推定する。

元の exp413 path は置換しない。自井戸の平滑K16係数と周辺合意係数の差だけを
K16 basisで累積し、first score rowを厳密に0とする低周波補正を作る。
補正は固定`alpha=0.05`と固定`±0.25 ft` capで弱く加える。したがって本実験は
「exp226 absolute predictionのblend」でも「近傍井戸の真のbias転写」でもなく、
predicted-only transductive rate consensusによるexp413後処理である。

## 固定数式

well `i` のscore rowsを `r=0..n_i-1` とする。`r=0`を後処理anchorとし、
transition `r=1..n_i-1`について次を作る。

1. `r0_i[r] = pred_i[r] - pred_i[0]`
2. `u_i[r] = cumsum(-(Z_i[r] - Z_i[r-1]))`
3. exp226互換K16 basis `Phi_i` で
   `c_i = argmin ||Phi_i c - (r0_i - u_i)||^2 + rho * ||D c||^2`
4. segment azimuthと`theta0=118.4 deg`から`proj_ij`を計算し、
   `abs(proj_ij) >= 0.3`のdonorだけ`c_ij / proj_ij`をXY fieldへ置く。
5. target segment midpointで他well donorだけにlocal-linear推定を行い、
   `c_neighbor_ij = field_hat_ij * proj_ij`とする。
6. query projection guard、finite、selected 50 segments中のunique donor wellsが8以上を
   全て満たすsegmentだけ`delta_c_ij = c_neighbor_ij - c_ij`、それ以外は0とする。
7. `raw_correction_i = concat([0], Phi_i @ delta_c_i)`
8. `correction_i = clip(0.05 * raw_correction_i, -0.25, 0.25)`
9. `pred_post_i = pred_exp413_i + correction_i`

`pred_exp413`の行別の細かい形状は保持し、K16低周波成分の差だけを加える。
明示的な追加fadeは使わず、zero-intercept累積そのものをfade-inとする。

## 固定parameter

| 項目 | 固定値 |
| --- | --- |
| scientific primary | `transductive_k16_neighbor_rate_a005_cap025` |
| K16 segments | 16 |
| coefficient smoothing `rho` | 10.0 |
| reference direction `theta0` | 118.4 degrees |
| projection guard | `abs(proj) >= 0.3` |
| local-linear neighbor segments | 50 |
| Gaussian bandwidth | 500.0 ft |
| local-linear ridge | 1.0 |
| minimum unique donor wells | 8 |
| self donor | excluded |
| alpha | 0.05 |
| final correction cap | ±0.25 ft |
| explicit fade / reanchor / projection | none / none / none |
| selectable variants | 1 |

## Validation と phase separation

- exp413のouter 5 foldsを固定し、foldごとのouter-valid wellsをhidden test batchの代理とする。
- prediction phaseでは`well,row_idx,fold,pred_tvt,X,Y,Z`だけを読み、truth、error、
  hidden-like role、by-well outcomeを読まない。
- raw exp413 controlとprimary prediction、support ledger、input/prediction content SHAを
  書き出してreadbackし、`prediction_freeze.json`を保存してからtruthを接続する。
- truth接続後にpooled、fold、MD `0-250 / 250-1000 / 1000+`、hidden-like spatial、
  hidden-like typewell-purged、by-well、correction/support/continuityを評価する。
- fold内の他outer-valid prediction利用はtest-time transductive処理の再現であり、
  同foldのtrue suffix、outer-train true TVT donor、exp226 kappaを使わない。

## Promotion gate

すべてANDとする。

- technical: 3,783,989 rows / 773 wells / folds 0-4、duplicate 0、finite 100%、
  exp413 CV/order/fold/input SHA parity、self donor 0、truth-before-freeze read 0。
- pooled gain: exp413 `7.884802794404715`から`>=0.01 ft`改善。
- fold: 4/5 folds以上nonworse。
- fixed scope: MD `0-250 / 250-1000 / 1000+`、hidden-like spatial、
  hidden-like typewell-purgedのcandidate-parent deltaが各`<=+0.02 ft`。
- by-well: delta RMSE p95とworstが各`<=+0.25 ft`。
- continuity: first score row correctionの最大絶対値`<=1e-12 ft`、
  全row最大絶対補正`<=0.250000001 ft`、nonfinite/row-order change 0。
- mechanism report: supported segment rate、unique donor wells、donor distance、
  raw/capped correction、K16 coefficient disagreementを必ず保存する。

PASS時も本実験内で自動的にinference実装・実行・提出へ進まない。同じexp511の
inference portを別承認で行う。FAIL時は
`FAIL_CLOSE_WITHOUT_ALPHA_CLIP_K_BANDWIDTH_RHO_THETA_SUPPORT_FADE_SCOPE_OR_GATE_RESCUE`
として閉じる。

## 実験範囲

- 対象実験: `exp511_exp413_transductive_k16_neighbor_rate_postprocess`
- Route: `ensemble`
- 親実験: `exp413_scale5_likpf_full_replacement_on_exp335`
- 手法参照: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- negative / guard参照: `exp114`、`exp118`、`exp201`、`exp263`、`exp418`
- 変更する変数: exp413 final OOFへpredicted-only K16 neighbor-rate correctionを1本だけ追加する。
- 固定する変数: exp413 prediction/fold/order/score rows、K16/local-linear parameter、
  alpha/cap/support、評価scope、promotion/fail-close gate。
- 実装前状態: design-only。コード、Notebook、package、run、inference、submissionは0。

## 実行inventory

| scientific variant | report-only variant | model config | trained fold | booster | PF/HMM/Beam | GPU | parent retraining |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## 再現性設計

- seed policy: `no_rng_stable_fold_well_row_segment_distance_source_order`
- stochastic 処理の有無: 本実験内なし。上流exp413のOOFをSHA固定入力として扱う。
- PF/Beam / likelihood-PF / seed bagging の有無: 再生成0。exp226のdeterministic
  K16/local-linear式だけを参照する。
- 並列処理と乱数の関係: RNGなし。実装時に並列化する場合もfold、well ID、segment、
  distance、source rowでstable sortし、reduce順を固定する。
- CPU/GPU runtime と deterministic flags: Kaggle private CPU、GPU/internet off、1 processを正とする。
- train cache / test feature regeneration の SHA 記録方針: exp413 OOF/fold manifest、raw geometry
  identity、K16 coefficient field、support ledger、frozen predictionのschema/content SHAを記録する。
- model manifest / prediction / submission SHA 記録方針: 新規modelなし。input manifest、
  prediction logical/content SHA、Kaggle kernel/versionを記録する。train auditではsubmissionなし。
- Kaggle package bootstrap 確認方針: 実装・runが別承認された場合だけ、embedded config/source、
  CPU/internet、kernel sources、support filesをpackageからreadbackする。
- deterministic anchor: 初回train auditではfalse。独立rerunでprediction content SHAが一致した場合だけ再判定する。

## リスク

- リークリスク: 同fold true TVTやtrain-only ANCCをdonorへ混ぜると即時leakになる。
  prediction allowlistとfreeze-before-truthをfail closedで実装する。
- transductive scopeリスク: outer-valid foldは約155 wells、hiddenは約200 wellsだが、
  visible current testは3 wellsでsupport不足となりidentityになる。public 3-well結果で
  hidden向けsupportを調整しない。
- 相関誤差リスク: 同じexp413 modelから出たwell予測は共通biasを持ち、近傍平滑化では
  そのbiasを直せない。改善は低周波分散低減に限定して期待する。
- 地質境界リスク: XY近傍が同じ構造を保証せず、fault跨ぎで誤ったrate consensusを作り得る。
  projection/support guardと±0.25 ft cap、scope/well-tail gateで制限する。
- CV/LB 不一致リスク: exp413 CV/LBとtransductive batch分布が異なる。LB tuningや
  public well/cardinality ruleを禁止し、hidden-like 2面をAND gateへ入れる。
- ランタイム/メモリリスク: 773 wells ×16 segmentsのfold-local fieldはCPUで処理可能と見込むが、
  全row copyを避け、fold/well chunkとfloat64 prediction boundaryを使う。
- 再現性リスク: equal-distance donor順、local-linear solve、parallel reduceで差が出得る。
  stable tie-break、1 process、入力/field/prediction content SHA readbackを必須にする。
