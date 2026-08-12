# 要件

## 依頼

`KAGGLE_DIRECTION.md` のアイデアバックログ先頭にある `exp012_single_catboost_lightgbm_residual` を実装する。

## 制約

- HGB residual model は破棄せず、`exp002` all-GR と `exp003` no-GR の control として再実行する。
- 変更する変数は residual model class に限定し、split、target、row sampling、baseline anchor、feature construction は `exp003` と揃える。
- CatBoost / LightGBM は単体モデルとして比較し、この実験では blend / ensemble をしない。
- no-GR と all-GR の 2 feature set に絞る。
- 初回 notebook 実行は Kaggle を正とし、ローカル notebook 実行はしない。
- LightGBM / CatBoost は Kaggle runtime の optional dependency として扱い、該当 variant 実行時だけ import する。

## 受け入れ基準

- `experiments/exp012_single_catboost_lightgbm_residual/` に train / inference notebook、`baseline.py`、`config.yaml`、記録ファイルがある。
- train notebook が HGB control、LightGBM、CatBoost の variant 別 GroupKFold CV を実行できる。
- train output に `ablation_metrics.csv`、`fold_metrics.csv`、`well_metrics.csv`、`fold_model_training.csv`、`model_group_summary.csv` を書く。
- `model_group_summary.csv` で hard/no-GR、public-like、high GR missing、long eval、steep trajectory group を比較できる。
- `validate_exp` と notebook preparation が通る。
