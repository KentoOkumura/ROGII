# 要件

## 依頼

`target_free_gr_quality_features_on_exp092` を実装する。旧 `gr_quality_only_pseudotail` の名前と対象を更新し、exp092 系 ML surface に GR 値そのものではなく GR 品質 / coverage だけを add-only feature として追加する。

## 制約

- Route: `ml_model`
- 親実験: `exp092_u_projection_correction_disagreement_fullrun`
- cache 親: `exp072_exp063_full_replay_feature_cache`
- quality context 親: `exp065_typewell_supertype_cluster_cv_audit`
- 生 GR 値、row-wise NCC/DTW score、GR 由来 candidate TVT、hard switch、direct replacement は入れない。
- validation/test true TVT、oracle candidate、true error、fold label を特徴量生成に使わない。
- 初回実行は Kaggle Notebook を正とし、ローカル notebook 実行はしない。
- 再現性: `docs/06_reproducibility.md` に従い、Kaggle bootstrap、SHA 記録、LightGBM seed / thread policy を記録する。

## 受け入れ基準

- `experiments/exp137_target_free_gr_quality_features_on_exp092/` に実験コード、config、train/inference notebook、README、result、SESSION_NOTES、metrics がある。
- `config.yaml` の `experiment.route` は `ml_model`。
- train notebook は exp072 cache、raw train well、exp065 artifacts の入力確認、variant 実行、metrics / feature importance / SHA 保存を追える構成である。
- `exp092_full_row_control` と `target_free_gr_quality_addonly` を同一 GroupKFold by well で比較できる。
- inference notebook は submission を生成せず、train-side audit only として停止する。
- 静的検証、notebook JSON validation、ruff、`make validate-exp` が通る。
