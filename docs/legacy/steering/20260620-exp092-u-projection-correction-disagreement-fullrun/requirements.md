# 要件

## 依頼

`KAGGLE_DIRECTION.md` の高優先 backlog `u_projection_correction_disagreement_fullrun` を `exp092_u_projection_correction_disagreement_fullrun` として実装する。

## 制約

- Route: `ml_model`
- 親実験は `exp085_u_projection_feature_ablation` とし、base surface は exp073 / cache は exp072 を固定する。
- exp085 の全 variant 再実行はしない。`u_projection_correction_plus_disagreement` のみに絞る。
- LightGBM family は `lgb0/lgb1/lgb2` の 3 model を維持する。
- target は `TVT - last_known_tvt` のままとし、U-space target ablation を混ぜない。
- 追加特徴は target-free な PF/Beam/likelihood-PF candidate path、row Z、MD distance、known-prefix anchor から作る。
- LGB OOF U-space feature は nested fold が必要なため、この fullrun でも無効にする。
- 再現性は `docs/06_reproducibility.md` に従い、GPU LightGBM、Kaggle bootstrap、入力 cache SHA、model SHA、prediction SHA の扱いを記録する。
- inference port は fullrun の pooled OOF、worst-well、distance/tail bucket、feature importance、test-side feature parity を確認するまで行わない。

## 受け入れ基準

- `experiments/exp092_u_projection_correction_disagreement_fullrun/` に config、settings、train/inference notebook、補助 `.py`、README、result、metrics、SESSION_NOTES がある。
- `config.yaml` の `experiment.route` は `ml_model`、`model.feature_ablation.active_variants` は `u_projection_correction_plus_disagreement` のみ。
- train notebook が setup、入力確認、single-variant fullrun、metrics/artifacts の構成で実験内容を追える。
- runner が正式 pooled OOF、by-well metrics、distance/tail bucket metrics、feature importance、feature schema、prediction SHA、model manifest を保存できる。
- `task validate-exp EXP=exp092_u_projection_correction_disagreement_fullrun` または同等の validate script が通る。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
