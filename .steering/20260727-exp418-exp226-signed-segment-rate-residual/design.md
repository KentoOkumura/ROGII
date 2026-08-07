# 設計

## 結論

exp226 の offset は K16 境界の jump ではなく、小さな signed increment error が
absolute re-anchor なしで累積した結果である。したがって exp333 のように各区間へ
constant offsetをbroadcastせず、K16ごとのsigned residual rateを予測し、既知境界から
連続積分する一因子実験に固定する。

特徴、fold、model は exp333 Stage 1 と同一にし、変更は
`segment mean offset target + constant broadcast` から
`anchored segment-rate target + cumulative integration` への置換だけとする。

## 数学的定義

### 行と区間

各wellのunknown suffix長を`n`、0始まりのsuffix rowを`i=0..n-1`とする。
K16 assignmentはexp226と同じ式に固定する。

```text
edges = linspace(0, n, 17)
segment_id(i) =
    searchsorted(edges[1:], i + 1, side="left") clipped to [0, 15]
```

base predictionを`p_i`、truthを`T_i`、残差を次で定義する。

```text
e_i = T_i - p_i
```

符号は正ならexp226がtruthより低く、最終predictionへ正方向の補正が必要である。

### Anchored cumulative-rate basis

`B`を`n × 16`行列とする。

```text
B[0, j] = 0
B[i, j] = count(k in 1..i where segment_id(k) == j),  i >= 1
```

すなわちfirst unknown rowの補正は必ず0で、row `i-1 -> i` のintervalには
destination row `i` のsegment rateを適用する。

各wellの16 targetは次で一意に定義する。

```text
y_rate = lstsq(B, e, rcond=None)
```

- dtype: float64
- unit: ft / row
- intercept: なし
- ridge / Huber / clipping / sample reweight: なし
- required matrix rank: 16

modelが予測した16個のrateを`yhat_rate`とし、row correctionと最終予測は

```text
c = B @ yhat_rate
p_final = p + c
```

とする。したがって`c[0] == 0`で、segment境界へconstant level stepを挿入しない。

## 実験範囲

- 対象実験: `exp418_exp226_signed_segment_rate_residual`
- Route: `ensemble`
- 一因子比較元: `exp333_exp226_k16_segment_residual_offset_target`
- base prediction親:
  `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- row-wise比較: `exp228_direct_residual_correction_on_exp226`
- 根本原因:
  `docs/analysis/exp226_offset_root_cause_audit_20260727.md`
- 変更する変数:
  - segment targetをmean offset (`ft`)からanchored cumulative rate (`ft/row`)へ変更
  - correctionをsegment constant broadcastからcontinuous cumulative integrationへ変更
- 固定する変数:
  - exp226-compatible K16 assignment
  - exp333 outer/inner fold identity
  - exp333 target-free 136-feature schemaと集約
  - exp333 LightGBM `lgb1` 1 config
  - segment row-count sample weight
  - CPU deterministic runtimeと全評価scope

## 保存済み入力の再利用

exp418はexp333 Stage 1がtruth join前に凍結した次の生成物を再利用する。

- kernel:
  `kentookumura/exp333-k16-segment-residual-stage1-train`
- nested prediction:
  `exp333_exp226_k16_segment_residual_offset_target_stage1_nested_exp226_predictions.csv.gz`
- exp333 feature-freeze SHA:
  `b2c7bff40f9fc994bd60471c03d9085ba48137c30b358402bfbb1cadecc4a078`
- feature schema content SHA:
  `8a6ae01d792e6cf352f22c4519ec7934e5e39ee032a8ae676da9449722684a45`
- expected feature count: `136`
- saved exp226 OOF decompressed SHA:
  `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`

実装時はexp333 SHA manifestからnested fileのraw/decompressed/logical SHAを読み、
一致後だけ使用する。欠損または不一致ならfail closedとし、exp418内で25回の
donor-field/kappa fitを再実行して救済しない。

## Foldとリーク防止

outer foldは保存済みexp226の5-fold identityをそのまま使う。
outer fold `f`について、

- outer-train target: exp333の`role=inner_oof_train` prediction
- outer-valid base/target: exp333の`role=outer_valid` prediction
- inner fold: exp333のstable SHA256 4-fold manifest

を使う。outer-train well自身のbase predictionを、そのwellのtruthを含むexp226 fitから
作らない。feature freezeとrate basis freezeが完了するまでtruth、residual、error、
oracle、hidden-like roleを読まない。

## Feature contract

exp333 Stage 1の136 model featuresを同じ順序で使う。

- row feature group:
  - `projection_correction`
  - `u_disagreement`
  - `gr_wavelet_rotation_confidence`
- row-to-segment aggregation: 各数値列のsegment内finite float64 mean
- structural 7 columns:
  - `segment_id`
  - `segment_position`
  - `segment_row_count`
  - `segment_md_span`
  - `exp226_pred_mean`
  - `exp226_pred_start`
  - `exp226_pred_end_minus_start`

新しいfeature、feature slope、donor-risk score、K12/K24 disagreement、well ID、
truth/error/oracle、supervised learned-likelihood、selector scoreを追加しない。

## Stage 0: continuous-rate oracle headroom

保存済みouter-valid exp226 OOFとK16 assignmentをtruth-freeでfreezeした後だけtruthを
joinし、各wellの`y_rate`と`B @ y_rate`を計算する。oracle predictionは診断専用で、
current-test predictionやfeatureとして保存しない。

technical gate:

1. rows / wells / segments = `3,783,989 / 773 / 12,368`
2. K16 segment id = `0..15`、各well16区間
3. 全wellで`rank(B) == 16`
4. finite target / correction coverage = `1.0`
5. `correction[0] == 0`
6. matrix productと逐次integrationの最大差 `<=1e-12 ft`
7. target/errorをfreeze前に読む回数 = `0`

scientific gate:

1. exp226 `9.4271095966`からoracle RMSE gain `>=1.00 ft`
2. 5/5 foldsでgain `>=0.50 ft`
3. first-row predictionはexp226とexact parity

FAIL時はbasis、intercept、MD-rate、K、ridgeを同じOOFで変更せずbranchを閉じる。

## Stage 1: strict nested signed-rate model

### Model

exp333で使ったexp228 `lgb1`を1 configだけ固定する。

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

- active variants: 1
- configs: 1
- outer folds: 5
- total boosters: 5
- sample weight: `segment_row_count`
- GPU: 0
- exp226/control fit: 0

### 評価

primaryはouter-valid rowで積分した`p_final`のpooled RMSE。

secondary:

- fold RMSE
- distance 0--250 / 250--1000 / 1000+
- hidden-like spatial / typewell-purged
- K16 boundary ±8 rows
- by-well delta分布、p95、worst
- segment-rate weighted RMSE
- rate sign balanced accuracy
- correction first-row / integration continuity
- feature importance

科学的PASSは全条件のANDとする。

1. pooled RMSE `<=8.894085501`（exp228比0.05 ft以上改善）
2. exp333 `9.076676661`比0.05 ft以上改善
3. exp226比改善fold `>=4/5`
4. near 0--250、1000+、hidden-like 2面、boundary ±8、by-well p95がexp226比非悪化
5. worst-well regression `<=+0.25 ft`
6. segment-rate weighted RMSEがzero-rate priorより5/5 folds改善
7. rate sign balanced accuracyが`>0.5`のfold `>=4/5`
8. first-row correction最大絶対値`<=1e-12 ft`
9. matrix/逐次integration最大差`<=1e-10 ft`

FAIL時はfeature追加、rate target変更、clip、shrink、intercept、re-anchor、model config、
sample weightを同じOOFで救済しない。

## 実行量

| 段階 | variant | config | folds | boosters | exp226 fit | GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage 0 | 0 | 0 | reporting 5 | 0 | 0 | 0 |
| Stage 1 | 1 | 1 | training 5 | 5 | 0 | 0 |

- PF/HMM/Beam regeneration: 0
- control再学習: 0
- initial runtime: Kaggle private CPU、internet off
- hard runtime limit: 4 hours

## Notebook契約

2026-07-27の実装承認により、Jupytext percent形式の
`*_compact_selfcontained_train.py`候補と対応する`.ipynb`を作る。正規Notebookは
別承認までplaceholderのままにする。

Notebook上で次を追える構成にする。

1. Imports
2. Runtime/config/SHA helpers
3. Saved exp333 nested prediction and feature-contract checks
4. K16 cumulative-rate basis helpers
5. Truth-free freeze and late target attachment
6. Stage 0 oracle readout
7. Stage 1 fold-safe LightGBM training
8. Continuous integration and row OOF evaluation
9. Scope/tail metrics, importance, manifests, and gate

実装ではexp333保存nested prediction、fold manifest、feature schema、SHA manifestを
読み、exp333と同じtarget-free feature generatorで136列を再構築してrow content SHA
とfeature-freeze SHAを照合する。exp226 donor field / kappa fitは実装へ持ち込まない。
Stage 1入口はStage 0 summaryのfile SHAと`PASS_STAGE0`をconfigへ固定するまで
fail closedとする。

## 再現性設計

- seed policy:
  outer/inner foldはexp333 manifest、LightGBMは`random_state=0`。
- stochastic処理:
  新規feature RNGなし。CPU LightGBMのみ。
- 並列処理:
  global RNGを使わず、fold/well/row/segmentをcanonical mergesortする。
  LightGBMは固定8 threads。
- CPU/GPU:
  CPU、`deterministic=true`、`force_col_wise=true`、GPU禁止。
- SHA:
  exp333 source manifest、nested prediction raw/decompressed/logical、
  fold manifest、feature schema/content、rate basis、target、model manifest、
  segment prediction、row OOF predictionを記録する。
- gzip:
  decompressed content SHAを主証拠とする。
- Kaggle bootstrap:
  package作成時にrepo config、埋め込みconfig、source manifestの一致をhard checkする。
- deterministic anchor:
  current-test inferenceとrerun parityがないため、train OOFが再現しても
  deterministic submission anchorとは呼ばない。

## リスク

- target安定性:
  cumulative basis列は後半rateほど有効行が少なく、target分散が大きくなり得る。
  rank、condition number、segment位置別target分布をreportするがridge救済はしない。
- model identifiability:
  exp333 featureはoffset target向けに弱かった。featureを固定して一因子比較を守るため、
  signed rateに必要な情報が不足してFAILする可能性がある。
- tail:
  小さなrate誤予測も長いsuffixで積分される。pooled改善だけで昇格せず、
  1000+、hidden-like、p95、worstをhard gateにする。
- first-row residual:
  correctionを0固定するため、GR/Uによるanchor直後のlevel errorは補正しない。
  これはabsolute offset targetを混ぜないための意図的制約である。
- CV/LB:
  exp226 CV 9.427 / Public LB 9.837、現行ML reference exp335 Public LB 7.517であり、
  PASSしても競争上のanchor更新を意味しない。
- saved artifact:
  exp333 nested artifactが取得不能またはSHA不一致なら実装を停止し、再生成で救済しない。

## 対象外

- MD単位rate、local independent OLS slope、Huber slope
- K12/K24、H128/H256/H512、可変segment
- segment offset/interceptとのjoint model
- clipping、shrinkage、taper、boundary smoothing、absolute re-anchor
- donor-risk / multiscale-risk / formation feature追加
- LightGBM config、seed、sample-weight grid
- selector、Viterbi、HMM、PF/Beam変更
- current-test inference、submission
