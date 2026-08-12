# exp418_exp226_signed_segment_rate_residual

## 状態

- Route: `ensemble`
- 状態: Stage 0 version 1 technical FAIL / terminal fail closed
- oracle RMSE: `0.646951416159574`（deployable model CVではない）
- Public LB / Private LB: なし
- 親: `exp333_exp226_k16_segment_residual_offset_target`
- base prediction: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- Stage 1 / inference / submission: 未承認・未実行

## 仮説

exp226のpersistent offsetは、最後の既知TVT以降で小さなsigned rate mismatchを
累積することが主因である。exp333のsegment-constant offsetではなく、K16ごとの
signed residual rateを予測して先頭補正0から連続積分すれば、境界stepを作らず
累積driftを抑えられる。

## 固定した変更

- exp333のK16、strict nested fold、136特徴、LightGBM `lgb1`を維持する。
- targetだけをsegment mean offsetからzero-intercept cumulative rateへ変更する。
- correctionだけをconstant broadcastからcontinuous integrationへ変更する。
- first unknown row correctionは0。
- clip、shrink、intercept、re-anchor、新特徴、parameter gridは使わない。

## 検証方針

- Stage 0: 保存生成物による0-model / 0-booster oracle headroom監査
- Stage 1: 1 variant ×1 config ×5 folds = 5 CPU boosters
- exp226/control再学習、GPU、PF/HMM/Beam再生成: 0
- primary: outer-valid row-level TVT RMSE
- hard gate: pooled、fold、near、1000+、hidden-like、boundary、by-well p95/worst

詳細なtarget、basis、gateは
`docs/legacy/steering/20260727-exp418-exp226-signed-segment-rate-residual/design.md`
を正とする。

## 実行入口

実装済み候補:

- `exp418_exp226_signed_segment_rate_residual_compact_selfcontained_train.py`
- `exp418_exp226_signed_segment_rate_residual_compact_selfcontained_train.ipynb`

compact候補を正規train Notebookへ採用した。inference Notebookはplaceholderのまま。

Stage 0承認時は`execution_contract.selected_stage=stage_0`とする。Stage 1は
Stage 0 summaryのfile SHAとPASSをconfigへ
固定したうえで別承認時だけ`stage_1`を選ぶ。Stage 1は保存exp333 nested predictionを
使い、exp226 fitは0である。

## 所見

Kaggle version 1は3,783,989 rows / 773 wells / 12,368 segmentsを0 boosterで
処理した。exp226 RMSE `9.427110`に対してrate oracle RMSEは`0.646951`、
5/5 foldsで改善した。一方、matrix / sequential integrationの最大差
`6.2954e-12 ft`が事前固定上限`1e-12 ft`を超えたためtechnical FAILとなった。
他のtechnical check 8件はPASSしたが、契約どおりStage 0全体をfail closedとする。

## 次

Stage 1、inference、submissionへ進まず終了する。再開案はexp418のgateを緩める
same-OOF rescueではなく、truth-freeなcross-runtime numerical contractを先に
固定する独立auditとして別実験・別承認で扱う。
