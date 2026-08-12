# 要件

## 依頼

`ravaghi_public_sel15_features_single_lgbm` を実装する。

## 制約

- Route: `ml_model`
- `exp026` pseudo-tail LightGBM を single-model anchor として扱う。
- `exp029` の public sel15 PF/Beam OOF-like artifact を fold-safe な feature source として使う。
- Ridge / meta-stack / public replay ではなく、単体 LightGBM の add-only feature ablation として実装する。
- `target_tvt` は label と scoring 以外で使わない。
- `pf_error`、`last_anchor_error`、`beam_error`、`exp026_oof`、exp026 bridge columns は model feature に入れない。

## 受け入れ基準

- `experiments/exp038_ravaghi_public_sel15_features_single_lgbm/` に config、train/inference notebook、audit script がある。
- base single LGBM と PF/Beam feature 追加候補を同一 split / row cap で比較できる。
- 各候補について raw CV と fixed bucket-shrink CV を記録する。
- original-fold と well-hash holdout の両方を出力する。
- `uv run python scripts/validate_experiment.py --experiment exp038_ravaghi_public_sel15_features_single_lgbm` が通る。
