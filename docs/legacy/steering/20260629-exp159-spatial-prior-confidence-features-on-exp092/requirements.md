# 要件

## 依頼

`spatial_prior_confidence_features_on_exp092` を実装し、Colab で実行できる状態にする。

## 制約

- Route: `ml_model`
- 親実験: `exp092_u_projection_correction_disagreement_fullrun`
- spatial prior 入力: `exp114_spatial_neighbor_prior_signal_audit`
- exp092 control 再学習はしない。
- spatial prior TVT を direct correction / hard selector / candidate replacement として使わない。
- validation/test true TVT、oracle best、true-error rank、fold label を特徴生成に使わない。
- 再現性: `docs/06_reproducibility.md` に従い、GPU 学習、Kaggle/Colab bootstrap、SHA 記録の扱いを設計に明記する。

## 受け入れ基準

- `experiments/exp159_spatial_prior_confidence_features_on_exp092/` に config、settings、train/inference notebook、実装、記録ファイルが揃う。
- train notebook は `spatial_prior_confidence_addonly` 1 variant を実行する。
- Colab runner notebook が作成され、Drive layout、large cache copy、background run marker を扱える。
- planned cost が `1 variant x 3 LGBM configs x 5 folds = 15 boosters` と記録されている。
- inference / submission は disabled とし、raw-test/full-train parity なしに提出へ進まない。
