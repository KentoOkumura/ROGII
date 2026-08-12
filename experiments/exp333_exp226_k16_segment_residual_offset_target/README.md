# exp333_exp226_k16_segment_residual_offset_target

## 状態

- Route: `ensemble`
- 状態: direct Stage 1 gate FAIL・branch closed / exp361根拠のcurrent-test candidate inference v2 technical PASS・submissionなし
- 親: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- 比較: row-wise residualのexp228、reusable fixed blendのexp263、ML参照のexp287
- CV / Public LB / Private LB: `9.076676661 / - / -`

## 仮説

exp226 residualをrowごとに学習したexp228はexp226を`0.483024 ft`改善したが、RMSE `8.944086`に留まった。exp226と同じK16 segment内でresidualを平均し、1 segmentを1 sampleとして学習すれば、局所row noiseを落としてpersistent offsetを予測しやすくできる。

## 単一変更

```text
row residual target         TVT_t - exp226_t
              ↓
K16 segment target          mean_t(TVT_t - exp226_t)
final row prediction        exp226_t + predicted_segment_offset
segment sample weight       segment row count
```

K16境界はexp226の`linspace(0,n,17)`と`searchsorted(..., side="left")`を完全再利用する。mean以外のtarget、slope、別horizon、clip、shrink、taper、interpolation、selectorは使わない。

## リークを避けるfold

単純に保存済みexp226 OOF residualを新しいGroupKFoldへ渡すと、outer-train側のexp226 donor fieldがouter-valid truthを使う可能性がある。このためStage 1はstrict nestedにする。

- outer fold: 保存済みexp226 5-fold identity。
- outer-train target: outer-train内4-foldのinner OOF exp226 predictionから作る。
- outer-valid base: full outer-trainだけからexp226 predictionを作る。
- outer-valid baseは保存済みexp226 OOFと最大差`1e-8 ft`以下を要求する。

## Feature

exp228/exp218 generatorのうち、raw testで再生成できるtarget-free groupだけをsegment finite meanへ集約する。

- `projection_correction`
- `u_disagreement`（LGB OOF featureなし）
- `gr_wavelet_rotation_confidence`

追加はK16位置、row数、MD span、exp226 predictionのmean/start/end差だけ。supervised learned-likelihood、selector score、truth/error/oracle、well IDは入れない。

## 検証方針

保存済みexp226 outer foldを正とする5-fold GroupKFoldで評価する。Stage 0では保存済みOOFのoffset-only headroomだけを読み、Stage 1ではouter-train内4-foldのinner OOF exp226 predictionで学習targetを作る。primary metricはsegment予測をrowへbroadcastしたTVTのpooled RMSEで、fold、distance、hidden-like、by-well、segment境界をhard guardにする。

## 段階

### Stage 0

保存済みexp226 OOFでK16 oracle mean-offsetのheadroomだけを0 model / 0 boosterで測る。exp226比`>=1.00 ft`、各fold`>=0.50 ft`を5/5 foldsで満たさなければStage 1を実装しない。

### Stage 1

Stage 0 PASSと別承認後だけ、strict nested exp226を作り、exp228単体最良のLightGBM `lgb1`を1 configだけ学習する。

- 1 variant × 1 config × 5 outer folds = 5 CPU boosters。
- nested exp226は25 donor-field/kappa fits、3,865 prediction well-runs。
- parent/control再学習は0。保存済みexp226/exp228/exp263を比較に使う。
- GPU、internet、追加config、seed baggingは禁止。

## 判定

- pooled RMSE `<=8.894085501`、すなわちexp228を`0.05 ft`以上改善。
- exp226比4/5 folds改善。
- near 0--250、1000+、hidden-like 2面、segment境界±8 rows、by-well p95はexp226から非悪化。
- worst-well delta `<=+0.25 ft`。
- segment-target weighted RMSEはzero-offset priorを5/5 folds改善。

Stage 1がPASSしても、推論候補化にはexp263 `8.238331715`以下と別承認を要求する。

## 実行境界

Stage 0の0-model headroom readoutをcompact self-contained trainとして実装した。保存済みexp226 OOFは`well_id,row_idx,suffix_offset,tvt_pred,fold`だけを先に読み、K16 assignmentとfold/row contractをSHA付きでfreezeしてから`TVT`をlate joinする。oracle segment offset、segment target、oracle predictionはdeployable生成物として保存せず、content SHAと集約metricsだけを残す。

compact self-contained trainを正規train Notebookへ採用し、Kaggle CPU v1でStage 0を実行した。exp226 RMSE`9.427110`に対しK16 oracle RMSEは`1.130603`、改善`8.296507 ft`。fold改善も`8.359821 / 8.002651 / 9.211417 / 8.005800 / 7.879885 ft`で5/5 PASSした。inferenceはfail-closedである。

## 所見

Stage 0 oracle headroomに対し、Stage 1はexp226から一部を回収して全foldとlong-rangeを改善した。しかしexp228を`0.132591 ft`下回り、near/worst safetyを満たさなかった。現在のtarget-free segment mean featureではconstant offsetを安全に適用できないため、固定ルールどおりbranchを閉じる。

## Stage 1実装

`*_stage1_compact_selfcontained_train.py/.ipynb`を別名候補として追加した。既存のStage 0正規train Notebookと同名compact Notebookは上書きしていない。Notebookはsaved exp226 outer fold、outer-train内SHA256 4-fold、25 donor-field/kappa fits、3,865 well prediction runs、outer-valid parent parity `<=1e-8 ft`を固定する。

row featureはexp072 cacheを`target`列なしで読み、raw anchorも`MD/Z/TVT_input`だけから復元する。exp228の固定sourceからprojection/U-disagreement/GRWR生成器だけを使い、許可済み3群のfinite float64 meanと固定7構造列へ集約する。モデルはexp228 `lgb1` 1 config × 5 outer foldsだけで、全promotion gate、model/feature/nested prediction/OOF SHA保存まで実装した。

32-well Kaggle CPU preflight v1を完走した。166,533 feature rows、full-source 25 fits、160 prediction well-runsを`491.885 sec`で実測し、outer-valid parent parity最大差`1.819e-12 ft`は`1e-8 ft` gateをPASSした。full Stage 1外挿はnested`1,142.530 sec` + feature`3,491.907 sec` + 固定reserve`1,800 sec` = `6,434.437 sec = 1.787 h`で、8.5時間gateをPASSした。preflight自身はmodel/booster 0で、この時点ではCV未算出だった。

full Stage 1 Kaggle CPU train v1は1 variant × 1 config × 5 folds = 5 boosters、nested exp226 25 fits / 3,865 prediction well-runs、control再学習0を`1,781.997 sec`で完走した。CVは`9.076676661`でexp226を`0.350432936 ft`改善し全5 foldsを改善したが、固定pooled上限`8.894085501`には未達だった。near 0--250は`+0.057439 ft`、worst wellは`+8.099023 ft`悪化し、pooled/near/worstの3 gateがFAILした。

decisionは`FAIL_CLOSE_BRANCH`。full-run承認は消費済みで、
`selected_stage=stage_1_train_completed_fail_closed`、`stage_1_run_approved=false`、
`kaggle_push_approved=false`へ戻した。追加config、same-OOF救済、
direct inference、提出は実施しない。

## Current-test candidate inference

exp361がfixed12へのadd-one noveltyをPASSしたため、別承認後に同じexp333内で
candidate-only inferenceを実装した。exp072 deterministic raw-test replay、
exp228 target-free U/GRWR generator、exp226 inference v1、保存済みexp333
5 modelを固定SHAで読み、3 wells × K16 = 48 segmentを予測した。

Kaggle CPU version 2（`id_no=128368525`）は`65.258 sec`でCOMPLETEし、
14,151行、129 row / 136 model features、5-model float64 mean、finite/ID/order/
boundary/model/saved-train parityを全PASSした。offset範囲は
`-4.249479～+2.592369 ft`、平均`+0.289689 ft`。candidate artifactの
decompressed SHAは`7571c628...17cd`である。

version 1はraw replayの205列に含まれるexp072除外済みdiagnostic 9列を
train 196列と誤比較して予測前に停止した。version 2はexp072正規allowlistを
適用する最小修正だけで完了した。新規学習、parent/control再学習、clip/shrink/
taper/slope、selector、blend、`submission.csv`、competition submitは0。

## 次

candidate artifact生成までは完了した。fixed bankへの組み込みは、13候補selector
再学習とtarget-free safety gateのどちらを別実験として固定するかの設計判断待ち。
今回の成果物を単独採用、平均blend、提出へ直結させない。

## 参照

- steering: `docs/legacy/steering/20260721-exp333-exp226-k16-segment-residual-offset-target/`
- exp226: CV `9.427109597` / Public LB `9.837`
- exp228: CV `8.944085501` / 未提出
- exp263: CV `8.238331715` / Public LB `7.800`
- exp287: CV `8.136708220` / Public LB `7.530`、ただしworst-well guard FAIL
