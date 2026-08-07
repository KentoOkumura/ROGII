# 要件

## 依頼

`KAGGLE_DIRECTION.md` のアイデアバックログ先頭にある、all-GR / no-GR / guarded prediction を選ぶ hard well router を `exp007_hard_well_router` として実装する。

## 制約

- 親実験は `exp006_hard_well_router_diagnostic` とし、診断で得た inference-safe 条件だけを router 入力に使う。
- router の route 判定には OOF RMSE、target-derived bucket、valid fold の `TVT` を使わない。
- base 予測は既存の `exp002` all-GR、alternate 予測は `exp003` no-GR、guarded 予測は `exp005` strict guarded 相当に限定する。
- 評価は well 単位 GroupKFold とし、score rows は `TVT_input` が NaN の evaluation zone に限定する。
- 初回の full notebook 実行は Kaggle を正とし、ローカル notebook 実行はしない。

## 受け入れ基準

- `config.yaml` に router 条件、閾値、比較 variant、selected variant が明示されている。
- train notebook が fold-safe に control と router variant を比較し、well ごとの route を artifact に残す。
- inference notebook が selected router variant で `submission.csv` と `inference_well_summaries.csv` を生成できる。
- `task validate-exp`、静的チェック、Kaggle notebook package 生成が通る。
