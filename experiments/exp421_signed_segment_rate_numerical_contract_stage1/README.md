# exp421_signed_segment_rate_numerical_contract_stage1

## 状態

- Route: `ensemble`
- 状態: Stage 1 version 5完了、scientific FAILでbranch閉鎖
- 親: `exp418_exp226_signed_segment_rate_residual`
- exp418の判定: `technical_fail` / `FAIL_CLOSE_BRANCH`のまま維持
- inference / submission: 未承認

## 仮説

exp418で唯一FAILしたmatrix / sequential integration差
`6.295408638834488e-12 ft`を、truth-freeに固定した`1.0e-10 ft`の
float64数値契約で扱う。exp418を事後的にPASSへ書き換えず、後継実験として
同一signed K16 rate Stage 1の学習可能性を評価する。

## exp418からの変更

- exp418 Stage 0 summary file SHAを固定する。
- technical failureが`integration_parity`だけ、他8 checksとpooled / 5-fold
  scientific threshold checksが成立したことを検証する。
- fixed synthetic rate vectorによるtruth-free numerical auditを学習前に実行する。
- numerical toleranceを`1.0e-10 ft`へ固定する。

target、K16 basis、continuous integration、exp333 nested fold、target-free
136特徴、LightGBM `lgb1`、sample weight、Stage 1 gateは変更しない。

## 検証方針

- truth-free synthetic numerical auditとexp418 summary eligibilityを学習前に
  fail-closedで判定する。
- v1はこの2検証をPASS後、exp333 row-feature SHA不一致で0 boosterのまま停止した。
- v2は真値、exp226 prediction、LightGBMを読まず、exp072 cache / exp228 source /
  exp333 schemaと再生成summary・row SHAだけを検証する。
- v2はsource/cache/schema/projection/GRWRをすべて完全再現し、train 773井戸の
  canonical row SHA `d8e932...`もexp333 v1ログと一致した。
- v1の期待値`947572...`はexp333の3井戸current-test inference SHAであり、
  train SHAではなかったため、参照範囲だけを訂正した。
- その後、保存exp333 nested predictionによる5-fold OOFだけを実行する。
- pooled/fold/near/tail/hidden/boundary/by-well/rate-signの既存AND gateを使う。

## 実行量

- feature SHA debug v2: 0 variant / 0 config / 0 fold / 0 booster / truth 0列
- active variant: 1
- LightGBM config: 1
- outer folds: 5
- CPU boosters: 5
- exp226 fit / control再学習: 0
- PF/HMM/Beam再生成: 0
- GPU: 0

## Stage 1結果

- CV: `9.405572476`
- exp226比: `-0.021537121 ft`
- 改善fold: 2/5
- decision: `FAIL_CLOSE_BRANCH`
- rate-target / rate-sign / numerical contractはPASSしたが、pooled、比較base、
  fold再現性、1000+、by-well p95、worst-well gateをFAILした。
- inference / submission: 実施しない

## 実行入口

- Jupytext source:
  `exp421_signed_segment_rate_numerical_contract_stage1_compact_selfcontained_train.py`
- canonical train Notebook:
  `exp421_signed_segment_rate_numerical_contract_stage1_train.ipynb`

Stage 1の結果がPASSでも、inferenceとsubmissionは別承認まで停止する。

## 所見

Kaggle v2/v4で保存境界と特徴surfaceの再現性を確認し、v5で5 boosterを完走した。
rate-spaceの学習可能性は確認できたが、累積TVTとtailへ安定転移せずbranchを閉じる。
