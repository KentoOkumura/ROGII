# 要件

## 依頼

`normalized_shape_addonly_features_on_exp092` を実装する。exp092 の U-projection correction / disagreement LightGBM surface を親に、well-local normalized U/MD shape 特徴を add-only で追加する。

## 制約

- Route: `ml_model`
- 親実験: `exp092_u_projection_correction_disagreement_fullrun`
- cache 親: `exp072_exp063_full_replay_feature_cache`
- 既存 raw features、exp092 U-projection correction、U disagreement は置換しない。
- PF/Beam diagnostic score、candidate quality score、hard selector、candidate replacement、target 変更は入れない。
- `tail_scale` / `u_scale` は valid/test true TVT を使わず、known prefix、MD/Z range、candidate dispersion から target-free に定義する。
- GPU train push 前に active variant 数、LightGBM config 数、fold 数、合計 booster 数、control 再学習有無を `SESSION_NOTES.md` に記録する。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。

## 受け入れ基準

- `experiments/exp149_normalized_shape_addonly_features_on_exp092/` に config、settings、train/inference notebook、補助 `.py`、README、SESSION_NOTES、result がある。
- train notebook は setup、入力確認、variant/mode 確認、学習、metrics/feature importance 保存をセル単位で追える。
- config の `experiment.route` は `ml_model`。
- 初回 active variant は `normalized_shape_addonly` のみで、`exp092_full_row_control` は disabled。
- 追加 feature group は `normalized_shape_geometry`、`normalized_candidate_shape`、`normalized_shape_disagreement`。
- `py_compile`、notebook JSON validation、ruff、`make validate-exp` が通る。
- deterministic anchor としては扱わず、Kaggle train 完了時に feature content SHA、model SHA、prediction SHA、Kaggle kernel version を記録する。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録する。
