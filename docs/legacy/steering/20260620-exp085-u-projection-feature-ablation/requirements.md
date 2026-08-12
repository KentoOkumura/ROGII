# 要件

## 依頼

`KAGGLE_DIRECTION.md` の高優先 backlog `u_projection_feature_ablation` を `exp085_u_projection_feature_ablation` として実装する。

## 制約

- Route: `ml_model`
- 親実験: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- target は exp073 と同じ `TVT - last_known_tvt` に固定し、exp080 の target ablation と混ぜない。
- base feature surface は exp072 deterministic full replay cache の 196 features に固定する。
- U-space projection feature は PF/Beam/likelihood-PF candidate path、row `Z`、MD distance、known-prefix anchor だけから作る。
- LGB OOF 由来 feature は nested fold が必要なため、初期実装では無効化する。
- 再現性: `docs/06_reproducibility.md` に従い、PF/Beam source cache、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。

## 受け入れ基準

- `experiments/exp085_u_projection_feature_ablation/` が存在し、`config.yaml`、`settings.py`、train/inference notebook、補助 `.py`、記録ファイルが exp085 を指す。
- `config.yaml` に control、projection correction、U-space disagreement、両者の和の feature variants が定義されている。
- train runner が variant 別の metrics、by-well metrics、bucket metrics、OOF predictions、feature schema、projection feature summary、feature importance、model manifest を保存できる。
- train notebook が setup、入力確認、projection feature ablation、metrics/artifacts の構成で実験内容を追える。
- inference notebook は selected variant 未設定なら停止する guard になっている。
- `py_compile`、notebook JSON validation、`ruff check`、`validate_experiment.py` が通る。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
