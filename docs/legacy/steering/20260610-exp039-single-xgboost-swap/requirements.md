# 要件

## 依頼

`exp039_single_xgboost_swap` を実装する。`exp039_ravaghi_single_lgbm_inference_submit` の ML route Public LB 基準 11.740 を作った single LightGBM 系の候補について、Ravaghi/public sel15 由来特徴、fixed bucket shrink、評価 surface を固定したまま、単体 residual estimator だけ XGBoost に差し替えて train-side / surrogate 評価できる状態にする。

## 制約

- Route: `ml_model`
- 親評価実験は `exp038_ravaghi_public_sel15_features_single_lgbm` とする。
- 推論提出済み anchor は `exp039_ravaghi_single_lgbm_inference_submit` として参照する。
- `exp029_public_sel15_pf_oof_feature_generation` の疑似 test rows と特徴列をそのまま使う。
- 同時に feature family、PF/Beam branch、bucket-shrink alpha、residual shrink、row cap を変えない。
- `audit.exp026_training` は regenerated control なので LGBM のまま固定する。
- full notebook 実行は Kaggle を正とし、ローカルでは構造検証と必要なら小さい smoke に留める。

## 受け入れ基準

- `.steering` と `experiments/exp039_single_xgboost_swap/` が存在する。
- `config.yaml` の `experiment.route` は `ml_model`、`model.estimator` は `XGBRegressor`。
- train notebook は `ravaghi_single_xgboost_audit.py` を呼び、`single_xgboost_*` 生成物を読む。
- inference notebook は audit-only として停止し、supported candidate が出るまで提出物を作らない。
- `scripts/validate_experiment.py --experiment exp039_single_xgboost_swap` が通る。
