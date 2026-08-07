# exp393_exp347_practical_numerical_equivalence_audit

## 状態

- Route: `ensemble`
- 状態: Stage 0 FAILを保持したStage A fold 0もFAILし、branch close
- 親: terminal closedの`exp347_prefix_gr_unary_batched_window_exact_ssm`
- 正規train Notebook: compact self-contained Stage 0 + Stage A実装
- 正規inference Notebook: fail-closed実装

## 仮説

exp347のscalar/batched posterior cell差`1.4662743e-5`は、GPU float32の計算順による局所差であり、最終posterior mean TVTとMAPには実用上無視できる影響しかない。

## 単一変更

exp347のFAIL判定は維持し、別実験でpromotion surfaceだけをposterior cell max errorからpractical TVT/MAP equivalenceへ変更する。モデル、objective、window、boundary、state grammar、production batch-4 pathは変えない。

## Stage 0結果

- Kaggle private T4 version 2で13 gate中10件PASS、3件FAIL。
- FAILはposterior mean TVT RMSE、TVT max abs、posterior row sum。
- exp347とexp393 Stage 0のFAIL判定は変更しない。

## 検証方針

Stage 0の結果は再分類せず保存する。Stage Aではouter-valid predictionをtruth参照前にfreezeし、exp209 baselineとの10個の事前固定gateをAND評価する。結果を見て閾値、dtype、batch、padding、kernel、objectiveを変更しない。

## Stage A

- Kaggle T4 version 4でfold 0 / seed 42 / neural model 1を完了した。
- real GR RMSEは`22.8661 ft`。shuffle `49.0052 ft`、geometry `32.4650 ft`には勝ったが、保存済みexp209 `12.6711 ft`より`10.1951 ft`悪化した。
- well RMSE p95は`43.0175 ft`でexp209 `26.3015 ft`より悪化し、worst-well regressionも`75.2279 ft > 10 ft`。
- Stage A checksは8/11 PASS、3 FAIL。decision=`close_stage_b_without_exp347_rescue_grid`。
- runtime `3.8304 h`、peak GPU memory `7.4954 GB`。LightGBM、booster、PF/Beam、親・control再学習は0。

## 禁止事項

- exp347の再開・再分類。
- 実行後の閾値変更やmetric grid。
- batch/padding/compile/fused kernel/科学契約の同一exp救済。
- Stage B、推論、提出。

## 現在の生成物

Stage 0 reportとStage Aの小型metrics/manifestをローカル保存済み。Stage A model、frozen prediction、validation readoutはKaggle outputにあり、SHAを記録済み。submissionは生成していない。

## 所見

real GR信号はshuffle/geometryより有効だが、既存exp209 exact HMM baselineを置換できる精度ではない。hidden-like 2群とdistance 1000+でも一貫してexp209より悪く、局所的なtail問題だけではない。

## 次

Stage B、推論、提出へ進まずbranchを閉じる。exp347やStage 0のFAILを再分類せず、同familyの閾値・dtype・batch・padding・kernel救済も行わない。
