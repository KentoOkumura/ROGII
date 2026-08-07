# 要件

## 依頼

`KAGGLE_DIRECTION.md` のアイデアバックログ先頭 `distance_weighted_inference_postprocess` を実装する。

## 制約

- 親実験は `exp020_distance_weighted_training_audit`。
- `exp020` の best `near_down_far_up_lightgbm` weight profile をそのまま使い、追加の重み探索はしない。
- CV は well 単位 `GroupKFold` のまま、評価対象は `TVT_input.isna()` 行だけ。
- train-only formation columns は使わない。
- postprocess は raw prediction、last-known TVT anchor、row distance だけで再現できるものに限定する。
- same-OOF fit score と held-out score を混同しない。

## 受け入れ基準

- `experiments/exp021_distance_weighted_inference_postprocess/` が作成され、config / settings / notebooks / docs が exp021 名で揃う。
- train notebook が selected weighted model の OOF を作り、weighted raw と weighted + distance bucket shrink の metrics artifact を保存する。
- inference notebook が selected weighted model を train wells 全体で fit し、設定済み postprocess を適用して `submission.csv` を作る。
- `validate_experiment.py` と Kaggle notebook package generation が通る。
