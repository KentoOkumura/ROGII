# 設計

## アプローチ

exp413の最終370特徴面とouter foldを唯一のML入力面にし、保存済みLightGBMを
control兼最終fallbackとして固定する。同じ行、列順、dtype、fold、残差targetで
CatBoost / XGBoostだけを新規学習し、保存済み物理候補1本とfamily単位で混ぜる。

最終weightはPublic LBで調整しない。5 outer foldsのうち1 foldを順に保持し、
残り4 foldのOOF family predictionsで固定bound付き二乗誤差を最小化して
保持foldへ適用する。5個のweight vectorの成分中央値を同じbounded simplexへ
Euclidean projectionした値をdeployment weightとする。

このreadoutは追加boosterを必要としない一方、base OOFを作る各outer modelの
学習集合がmeta holdoutと完全には独立でない。10-model上限を守るため
strict nested base refitは行わず、結果を`OOF-level cross-fit`と明記する。
採用にはfold、hidden-like、well-tailの強い外部guardを要求する。

## 実験範囲

- 対象実験: `exp494_exp413_cat_xgb_physics_bounded_stack`
- Route: `ensemble`
- 親実験: `exp413_scale5_likpf_full_replacement_on_exp335`
- 物理入力: exp413 scale5 replacement candidate bank内の
  `exp226_w500_50_50`。`likpf_mean` slotは`likpf_scale_5_x1p0`
- 参考negative:
  `exp274_catboost_final_regressor_swap_on_exp238`、
  `exp275_xgboost_final_regressor_swap_on_exp238`
- 変更する変数: final regressor familyとfamily-level blendだけ
- 固定する変数: row keys、outer fold、370列名・順序・float32 dtype、
  residual target、anchor、候補生成、selector、signed selector、評価scope
- scope外: feature追加、parameter grid、sample weight、fold変更、
  selector再学習、PF/HMM/Beam再生成、Public LB weight選択、提出

## Stage設計

### Stage 0: exp413入力面の凍結

学習を行わず、exp413の保存済みStage 0/C/S/DをSHA検証する。
final 370列をexp413と同じ手順で1回だけmaterializeし、foldごとに
train / validのrow key、列順、float32 matrix content SHA、anchor、
residual target、outer foldをmanifestへ保存する。

必須parity:

- 3,783,989 rows / 773 wells / outer 5 folds
- clean273 + nested74 + signed23 = 370 unique columns
- Stage C / S fold manifest SHA一致
- exp413 Stage D OOF SHA、15-model manifest SHA、RMSE 7.884802794404715一致
- row keyは`id, well, outer_fold`で一意
- score対象は`TVT_input`欠損suffixだけ
- truthはfeature schema / stack weightsの固定後にだけ読む

Stage 0がFAILした場合は学習0本でcloseする。列の補修、drop、順序変更、
欠損列の代替生成はしない。

### Stage 1: CatBoost / XGBoost Stage D差し替え

Stage 0の同じfold別matrixを読み、`target = actual_tvt - last_known_tvt`を学習する。

- CatBoost: Pixiux `cb0`、1 config x 5 folds = 5 models
- XGBoost: Cdeotte v3、1 config x 5 folds = 5 boosters
- exp413 LightGBM: 保存済み15 models、再学習0
- selector: 保存済み40 + 20 models、再学習0
- 合計新規: 2 variants / 2 configs / 5 folds / 10 GPU models

CatBoost GPU / XGBoost GPUはbitwise deterministicと仮定しない。
各model SHA、best iteration / tree count、fold matrix SHA、OOF prediction SHA、
feature importanceを保存する。

### Stage 2: family別監査

`lgb_mean`、`cat_mean`、`xgb_mean`、`physics`を同じ行順で評価する。

- pooled RMSE、fold RMSE
- distance 0--250 / 250--1000 / 1000+
- hidden-like spatial / typewell-purged
- by-well RMSE、delta p50 / p90 / p95 / worst
- prediction correlation
- residual correlation
- error covariance
- family disagreement分位点

exp275の相関0.999995765をnegative referenceにする。XGBoostの
exp413 residual correlationが0.995以上かつbounded stackへのcross-fit寄与が
0.01 ft未満なら、実行結果ではweight 0を許容し、parameter rescueを行わない。

### Stage 3: 物理候補の固定

使用する物理候補は`exp226_w500_50_50` 1本だけとする。

```text
0.50 * exp226_k16 + 0.25 * likpf_mean + 0.25 * exact_hmm
```

exp413 scale5-overlay版の保存OOFは`8.070218793924594`で、対応Public LBは
存在しない。overlay前のexp263同名候補OOF `8.238331` / Public LB `7.800` /
5/5 fold改善は履歴contextだけとし、scale5版の採用根拠へ転用しない。
他候補とのweight / candidate gridは禁止する。train / hidden inferenceとも
exp413 candidate replayから同じID、同じ`likpf_scale_5_x1p0` semanticsを読む。

### Stage 4: cross-fit bounded stacking

family順は`lgb, cat, xgb, physics`で固定する。interceptなし、重み和1、
目的関数はunweighted suffix-row squared errorとする。

```text
0.60 <= w_lgb <= 1.00
0.00 <= w_cat <= 0.25
0.00 <= w_xgb <= 0.20
0.00 <= w_physics <= 0.20
sum(w) = 1
```

solverはSLSQPの凸二次問題とし、固定initial weight
`[0.70, 0.10, 0.05, 0.15]`から開始する。成功status、constraint residual、
同じweightでの再計算parityを必須にする。subset grid、weight grid、
Public LB refit、intercept、negative weightは使わない。

constant stackの採用条件はすべてAND:

- exp413比pooled RMSE gain >= 0.03 ft
- 非悪化fold >= 4/5
- near 0--250、1000+、hidden-like 2面の悪化が各+0.02 ft以内
- by-well delta p95 <= 0.00 ft
- worst-well delta <= +0.25 ft
- finite coverage 100%、row mismatch 0

FAIL時はconfidence gate、inference、submissionへ進まず、exp413を維持する。

### Stage 5: 条件付きsmall confidence gate

Stage 4がPASSした場合だけ、1個の決定的なdisagreement gateを評価する。
ML coreはdeployment weightsからphysicsを除いて再正規化し、
`d = abs(p_physics - p_ml_core)`とする。各meta-train 4 foldsで
targetを使わず`q50` / `q90`を求め、保持foldへ次を適用する。

```text
s(d) = 1.0                                  if d <= q50
s(d) = 1.0 - 0.5 * (d-q50)/(q90-q50)       if q50 < d < q90
s(d) = 0.5                                  if d >= q90
p_gate_raw = p_ml_core + w_physics*s(d)*(p_physics-p_ml_core)
p_gate = p_constant + clip(p_gate_raw-p_constant, -0.25, +0.25)
```

well ID、座標、Public LB、truth/error、モデル学習をgate featureに使わない。
採用にはStage 4の全gateを維持し、さらにconstant stack比
pooled RMSE gain >= 0.01 ft、4/5 folds非悪化を要求する。
FAIL時はconstant stackを採用し、gate rescueは行わない。

### Stage 6: hidden-safe inference

実装と実行はtrain採用後の別承認とする。同じ実験番号内にJupytext percent形式の
compact self-contained inference候補を作り、採用後に正規Notebookへ反映する。

- sample submissionのID、行数、well数をruntime contractにする
- 公開14,151行 / 3 wellsのassert禁止
- exp413 raw-test feature / candidate replayを1回だけ行う
- 保存済み40 + 20 selector、15 LGB、5 Cat、5 XGBだけをloadする
- `exp226_w500_50_50`をcandidate replayから取得する
- inference中の`.fit()`、weight再推定、threshold再推定は0
- ID merge、順序、重複0、finite 100%、fallback rowsを明示する
- 27,000秒soft budgetを超える前に、ensemble開始前のbudget guardで
  exp413保存アンカーへのglobal fallbackを許可する
- fallbackは全行単位だけとし、理由と行数をmetricsへ記録する。silent fallback禁止
- root `submission.csv`だけを提出候補にする

## 再現性設計

- seed policy: CatBoost 7、XGBoost 42、foldはexp413 manifest固定
- stochastic処理: CatBoost GPU、XGBoost GPU
- PF/Beam / likelihood-PF / seed bagging: 新規実行なし。
  exp413 raw-test replayのstable SHA256 per-well seedをそのまま使う
- 並列処理: Python global RNGを使わず、family model内部だけに限定する
- runtime: Kaggle T4、internet off。train / inferenceともsoft 27,000秒、
  hard 32,400秒
- feature SHA: final370の列順logical SHAとfold別float32 matrix content SHA
- model SHA: 5 Cat + 5 XGB、保存15 LGB、40 + 20 selector manifestを記録
- prediction SHA: family OOF、constant stack、conditional gate、hidden predictionを記録
- submission SHA: submit-check対象のroot `submission.csv`を記録
- deterministic anchor: GPU独立rerun一致前はfalse
- Kaggle bootstrap: package時にembedded / loose config byte parity、
  contract SHA、run flag、kernel sources、GPU、internet offを検証する

## リスク

- exp238ではCatBoost / XGBoostともCV悪化し、XGBoostはLGBとほぼ同一だった。
- exp413はlikPF/HMM候補を既に特徴化しており、物理予測の増分多様性が小さい可能性がある。
- exp413自体のby-well p95 / worst tailが悪いため、平均RMSEだけの改善は採用しない。
- OOF-level cross-fitは10-model制約のためstrict nested stackingではない。
- CatBoost / XGBoost GPUはruntime image差でbitwiseに揺れる可能性がある。
- final370再構築はメモリ負荷が大きい。fold別materializeと即時解放を必須にする。
- full matrix SHAは`memoryview`でzero-copy計算する。CatBoostは`Pool`構築後に
  生行列とfold DataFrameを解放してからfitし、XGBoost行列はCatBoost終了後に
  同じfoldを再読込して凍結SHAと一致確認する。両familyの巨大行列を同時保持しない。
- Stage 0 fold処理後は`malloc_trim(0)`で解放済みarenaをOSへ返す。
  378万行物理OOFは250,000-row Parquet row groupsで書き、全行copyを禁止する。
  base matrixは列chunkを先に選んでからfold行を抽出し、finite検証もrow chunk化する。
- Stage 0完了後はclean273の273特徴を一時float32 NPY memmapへ列chunkで書き、
  metadata 5列を除くDataFrameを学習前に解放する。family fold行列はmemmapから
  再構成して直ちにmappingを閉じ、凍結matrix content SHAとの一致を必須にする。
- CatBoost Poolはtrain raw matrixをPool生成直後に解放してからvalid Poolを作り、
  valid raw matrixもfit前に解放する。runtime memmapはfamily学習後に削除する。
- hiddenでのwell数・行数・軌跡長が公開sampleと異なり、runtimeが増える可能性がある。
- confidence gateはconstant stack通過後のみの小補正であり、失敗時の救済枝にしない。

## 2026-07-31 Stage 6参考提出override

train version 5は固定tail gateをFAILしたため、scientific selectionは引き続き
`exp413_lgb`である。その後のユーザー明示指示により、追加の救済や調整をせず、
保存済みconstant stackだけを参考提出する。

実装はexp413 hidden-compatible version 4のraw-test再生成を親とし、同じouter
foldのfinal370 matrixへCatBoost / XGBoostを各1本ずつ適用する。familyごとに
5 fold平均を取り、凍結deployment weightsで4-family blendを生成する。
`exp226_w500_50_50`は同じscale5-overlay candidate replayから読む。

27,000秒soft budgetをensemble開始前に超えた場合だけ、全行単位でexp413
LightGBM anchorへfallbackする。model欠損、SHA不一致、schema不一致、ID不一致、
非finiteはfallbackせずfail closedする。Notebookは`submission.csv`を生成するが、
competition submit APIはhost側のsubmit-check後にのみ呼ぶ。
