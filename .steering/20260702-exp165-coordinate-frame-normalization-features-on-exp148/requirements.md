# 要件

## 依頼

`coordinate_frame_normalization_features_on_exp148` を実装する。CPU 実行とし、Kaggle timeout 対策のため学習 notebook は `lgb0`、`lgb1`、`lgb2` に分割する。

## 制約

- Route: `ml_model`
- 親実験: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- control 再学習はしない。保存済み exp148 CV / Public LB を比較基準にする。
- 追加 feature は target-free な raw `MD/X/Y/Z`、known-prefix anchor、prefix-tail azimuth、既存候補間 disagreement に限定する。
- validation/test true TVT、oracle best、true-error rank、OOF absolute error を feature source に入れない。
- 座標正規化値を direct TVT candidate、hard correction、row-wise selector、blend、postprocess replacement として使わない。
- `runtime.kaggle.enable_gpu=false`、internet off、CPU deterministic LightGBM を使う。

## 受け入れ基準

- `experiments/exp165_coordinate_frame_normalization_features_on_exp148/` に config、settings、train/inference notebook、補助 `.py`、README、SESSION_NOTES、result、metrics が揃っている。
- `exp165_coordinate_frame_normalization_features_on_exp148_train_lgb0.py` / `train_lgb1.py` / `train_lgb2.py` がそれぞれ 1 LightGBM config だけを学習する。
- active variant は `coordinate_frame_addonly` のみで、`exp148_historical_control` は disabled。
- 追加 feature group は `coordinate_frame_geometry`、`coordinate_frame_direction`、`coordinate_frame_derivative`、`coordinate_frame_interaction`。
- `make validate-exp EXP=exp165_coordinate_frame_normalization_features_on_exp148` が通る。
- Jupytext 変換、構文チェック、`ruff --select F821` が通る。
- Kaggle push 前に 1 variant、3 configs、5 folds、15 boosters、control 再学習なしを `SESSION_NOTES.md` に記録している。
