# 要件

## 依頼

`tvt_dense_addonly_confidence_features_on_exp092` を実装する。

## 制約

- Route: `ml_model`
- 親実験: `exp092_u_projection_correction_disagreement_fullrun`
- Cache: `exp072_exp063_full_replay_feature_cache`
- exp092 control は再学習しない。既存 exp092 `lgb1` CV 9.322479896 / Public LB 8.350 を baseline として参照する。
- `tvt_dense` / `tvt_densew` / `tvt_dense50` を prediction replacement、hard switch、dense-only submission に使わない。
- valid/test true TVT、oracle best、true-error rank、absolute error label、fold label を特徴生成に使わない。
- 再現性: `docs/06_reproducibility.md` に従い、PF/Beam upstream cache、GPU LightGBM、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。

## 受け入れ基準

- `experiments/exp151_tvt_dense_addonly_confidence_features_on_exp092/` に config、train/inference notebook、実装 helper、README、SESSION_NOTES、result、metrics がある。
- `config.yaml` の `experiment.route` が `ml_model`。
- active variant は `tvt_dense_confidence_addonly` のみで、planned train cost は 1 variant x 3 LightGBM configs x 5 folds = 15 boosters。
- 追加特徴は target-free dense confidence features として feature group 化されている。
- `py_compile`、notebook JSON validation、ruff、`make validate-exp` が通る。
