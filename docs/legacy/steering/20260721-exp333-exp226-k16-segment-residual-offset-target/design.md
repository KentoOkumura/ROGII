# 設計

## アプローチ

各wellのunknown suffix長を`n`、1始まりのsuffix rowを`t=1..n`とし、exp226と同じ式で16区間へ割り当てる。

```text
edges = linspace(0, n, 17)
segment_id(t) = searchsorted(edges[1:], t, side="left") clipped to [0, 15]
```

outer fold `f`のsegment `b`について、strict nested exp226 predictionを`p226_nested`としてtargetを次で定義する。

```text
r_t       = TVT_t - p226_nested_t
y_b       = sum(r_t for t in b) / n_b
weight_b  = n_b
p_final_t = p226_nested_t + yhat_b
```

`y_b`はrow-level squared errorに対する区間一定補正の最適値なのでmeanに固定する。モデル出力`yhat_b`はclipやshrinkをせずsegment全行へbroadcastする。slope、隣接segment interpolation、Viterbi、selectorは別仮説として禁止する。

## 実験範囲

- 対象実験: `exp333_exp226_k16_segment_residual_offset_target`
- Route: `ensemble`
- 親実験: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- row-wise比較: `exp228_direct_residual_correction_on_exp226`
- reusable blend比較: `exp263_last_anchor_better_candidate_confidence_pair_cache`
- ML参照: `exp287_fold_safe_formation_74_addonly_on_exp264`
- 変更する変数: residual targetと学習単位をrowからK16 segmentへ変更する。
- 固定する変数: exp226 K16式、親parameter、outer folds、許可feature family、LightGBM 1 config、offset-only broadcast、metric/guard。

## Stage 0: K16 offset-only headroom

保存済みexp226 group-safe OOFだけを読み、12,368 non-empty segmentsでtruth residual meanを計算する。oracle meanを同segmentへbroadcastしたTVTは診断専用で、prediction/feature/target artifactとして後段へ渡さない。

PASS条件:

- row/well/segment coverageが`3,783,989 / 773 / 12,368`で一致する。
- oracle offset-only RMSE gainがexp226 `9.4271095966`比`>=1.00 ft`。
- 5/5 foldsでgain `>=0.50 ft`。
- target、segment assignment、oracle readoutのSHAを保存し、truthをfeatureへ戻さない。

1つでもFAILならStage 1を実装せずbranchを閉じる。同じOOFでsegment数、center、slope、clipを救済しない。

## Stage 1: strict nested segment model

### Fold contract

outer foldは保存済みexp226 source foldを正とする。outer fold `f`ごとに次を行う。

1. outer-valid wellsを除いたouter-train wellsを作る。
2. outer-trainをstable SHA256 round-robinで4 inner foldsへ分ける。
3. 各inner-validに対し、inner-trainだけでexp226 donor field/kappaをfitしてpredictionを作る。
4. full outer-trainだけでouter-valid exp226 predictionを作り、保存済みexp226 OOFとの`1e-8 ft` parityを要求する。
5. inner OOF residualからouter-train segment target、outer-valid residualから評価targetを作る。
6. outer-train segmentだけでLightGBMを学習し、outer-valid segment offsetを予測する。

inner fold seed keyは`sha256("exp333|outer={f}|well={well_id}")`で固定し、hash昇順を4-fold round-robinへ割り当てる。validation well、outer-valid fold、inner-valid foldのtrue suffix TVTはdonor field、kappa、featureへ入れない。

### Segment feature

row featureはexp228/exp218 generatorから次の3 groupだけを許可する。

- `projection_correction`
- `u_disagreement`（`include_lgb_oof_features=false`）
- `gr_wavelet_rotation_confidence`

各数値列はsegment内finite値のfloat64 mean 1列だけへ集約し、全非finiteならNaNを維持する。std/min/max/quantile/first/last/slopeは作らない。追加する構造列は次の7列に固定する。

- `segment_id`
- `segment_position=(segment_id+0.5)/16`
- `segment_row_count`
- `segment_md_span`
- `exp226_pred_mean`
- `exp226_pred_start`
- `exp226_pred_end_minus_start`

target、residual、oracle candidate、exp145 supervised learned-likelihood、exp264/287 selector score、well IDはfeatureへ入れない。

### Model

exp228の単体最良だった`lgb1`を1 configだけ固定する。

```yaml
boosting_type: gbdt
objective: regression
num_leaves: 64
min_child_samples: 40
subsample: 0.474
subsample_freq: 1
colsample_bytree: 0.393
reg_lambda: 95.75
reg_alpha: 10.79
min_child_weight: 0.24
learning_rate: 0.0093
n_estimators: 10000
random_state: 0
deterministic: true
force_col_wise: true
num_threads: 8
early_stopping_rounds: 250
```

sample weightはraw segment row count。1 variant × 1 config × 5 outer folds = 5 CPU boostersで、GPUは使わない。

### 評価

primaryはouter-valid rowへbroadcastした`p226_nested + yhat_segment`のpooled RMSE。secondaryとしてfold、distance、hidden-like、by-well、segment境界±8 rows、segment-target weighted RMSE、offset signを保存する。

科学的PASS条件:

- pooled RMSE `<=8.894085501`（exp228 8.944085501を0.05 ft以上改善）。
- exp226比4/5 outer folds改善。
- distance 0--250、1000+、hidden-like spatial/typewell-purged、segment境界±8 rows、by-well p95がexp226から非悪化。
- worst-well exp226比delta `<=+0.25 ft`。
- segment-target weighted RMSEがzero-offset priorより5/5 folds改善。

推論候補化の追加条件はpooled RMSE `<=8.238331715`（保存済みexp263）と別承認。科学的PASSだけではcurrent-test model、submissionを作らない。

## 実行量

- Stage 0: 1 readout、0 model、0 booster。
- Stage 1: 1 variant、1 config、5 outer folds、5 LightGBM boosters。
- nested exp226: 5 outer-valid fits + 20 inner-valid fits = 25 donor-field/kappa fits。
- nested prediction: 773 outer-valid + 3,092 inner-valid = 3,865 well-runs。
- parent/control再学習: 0。保存済みexp226/exp228/exp263 metricsを比較に使う。
- runtime: Kaggle private CPU、internet off、8.5時間上限。実装後の32-well preflightで全Stage 1外挿`<=8.5 h`を要求する。

## 再現性設計

- seed policy: outer foldはsaved exp226 identity、inner foldはstable SHA256、LightGBM random_state 0。
- stochastic処理: upstream固定PF/Beam feature cacheとCPU LightGBM。新規乱数生成はinner fold hashとLightGBMだけ。
- 並列処理: global RNGを使わず、feature/segmentはfold/well/rowのcanonical sort。LightGBMは固定8 threads。
- CPU/GPU: CPUのみ、`deterministic=true`、`force_col_wise=true`。GPU禁止。
- SHA: raw/input manifest、exp226 saved OOF、fold map、inner fold map、K16 assignment、row feature schema/content、segment feature schema/content、nested exp226 prediction、segment target、model manifest、OOF row predictionを記録する。
- gzipはdecompressed content SHAを主証拠とする。
- inference/submissionは今回無効なのでsubmission SHAは記録対象外。将来別承認時に有効化する。
- Kaggle bootstrapはpackage作成時にconfigとnotebook内configの一致をhard checkする。

## リスク

- リークリスク: 単純なexp226 OOF residual再利用ではouter-train baseがouter-valid truthをdonorに使い得る。strict outer/inner nested exp226で遮断する。
- target集約リスク: segment内に傾きがあればconstant offsetでは残る。今回は原因分離のためslopeを禁止し、失敗後のsame-OOF救済を行わない。
- 境界リスク: hard broadcastでsegment境界にstepが生じる。境界±8 rowsをhard guardにし、taper/interpolationで救済しない。
- 特徴リークリスク: supervised learned-likelihood/selector scoreはnested contractが異なるため除外する。
- CV/LB不一致: exp226 CV 9.427/LB 9.837、exp287 CV 8.137/LB 7.530かつworst-well FAILであり、pooled gainだけで推論へ進めない。
- runtime: nested exp226 25 fitsが主コスト。32-well preflight外挿が8.5時間を超えたらfull Stage 1を実行しない。
- 再現性: upstream fixed cacheのschema/content SHA不一致、outer parity不一致、fold map不一致はfallbackせず停止する。

