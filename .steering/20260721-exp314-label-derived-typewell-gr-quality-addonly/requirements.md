# 要件

## 依頼

Train suffix TVT truthからouter-fold安全に作ったType Well群GR品質priorを、testで群contentだけから参照できるadd-only ML特徴として評価する。設計のみで15 boostersは学習しない。

## 制約

- Route: `ml_model`。exp311とexp313の全gate PASSが先行条件。
- exp148のfold、base feature、3 LightGBM configを固定し、saved controlを再学習しない。
- 追加はsupport/noise/reliability 6列だけ。calibrated GRやTVT correctionは禁止。
- 1 variant × 3 configs × 5 folds = 15 new boostersは別途明示承認が必要。

## 受け入れ基準

- outer-train group priorをouter-valid/testへtypewell contentだけでjoinする。
- `lgb_mean`を主評価とし、CV +0.03 ft以上、4/5 folds、全距離帯・hidden-like非悪化、worst +0.25 ft以下を要求する。
- supportなしはglobal priorとavailability=0へ固定fallbackする。
- inference/submissionはtrain-side gate PASS後も別判断とする。
