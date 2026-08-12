# 設計

## アプローチ

`exp003_residual_ablation` を親にし、leak-safe な residual feature frame と fold runner を再利用する。`baseline.py` の `make_drift_model` を config-driven にして、`HistGradientBoostingRegressor`、`LGBMRegressor`、`CatBoostRegressor` を `model.drift_model.estimator` で切り替える。

train notebook は variant ごとに同じ `GroupKFold`、同じ row sampling cap、同じ residual shrink / clipping を使って CV する。OOF well metrics から HGB all-GR / HGB no-GR controls を使った group tag を作り、model class の改善が全体平均だけでなく hard/no-GR、public-like、high GR missing、long eval に偏っていないかを見る。

## 実験範囲

- 対象実験: `exp012_single_catboost_lightgbm_residual`
- 親実験: `exp003_residual_ablation`
- 変更する変数: residual model class (`HistGradientBoostingRegressor` / `LGBMRegressor` / `CatBoostRegressor`) と feature set (`all` / `no_gr_signal`) の比較
- 固定する変数: GroupKFold、seed、last-anchor residual target、row sampling caps、residual shrink、max residual clipping、feature construction

## Variants

- `control_hgb_all`: exp002 all-GR HGB control
- `control_hgb_no_gr`: exp003 no-GR HGB control
- `lightgbm_all`: LightGBM all-GR
- `lightgbm_no_gr`: LightGBM no-GR
- `catboost_all`: CatBoost all-GR
- `catboost_no_gr`: CatBoost no-GR

## リスク

- リークリスク: feature construction は exp003 と同じで、評価 zone の `TVT` は train/CV target 以外に使わない。
- CV/LB 不一致リスク: `exp002` public anchor と `exp003` CV anchor を別々に control として出し、public-like group の悪化を確認してから inference に進む。
- ランタイム/メモリリスク: 6 variants x 5 folds で重い。`EXPERIMENT_VARIANT_LIMIT` と debug well limit で切り分け可能にする。CatBoost / LightGBM は CPU thread 数を 2 に制限する。
- 依存リスク: local `pyproject.toml` には LightGBM / CatBoost を追加せず、Kaggle runtime import を前提にする。import 失敗時は variant が明示的に失敗する。
