# 要件

## 依頼

`linear_md_z_prior_residual_target` を実装する。

## 制約

- Route: `ml_model`
- 親実験: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- diagnostic parent: `exp113_linear_md_z_prior_global_search`
- exp072 の deterministic 196-feature train cache と exp073 LightGBM config family を固定する。
- 変更する変数は supervised target 定義だけにする。
- prior は `T0 + a * (MD - MD0) + b * (Z - Z0)`。`T0/MD0/Z0` は known-prefix 最終行から取る。
- `a,b` は config で固定し、validation tail true TVT で fit / select しない。
- 再現性: `docs/06_reproducibility.md` に従い、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。

## 受け入れ基準

- `config.yaml` に route、lineage、target 定義、active targets、leakage policy が明記されている。
- train notebook が setup、cache preview、known-prefix anchor check、target ablation 実行、metrics display に分かれている。
- `linear_md_z_prior_residual_target.py` が `dTVT` control と linear prior residual targets を同じ GroupKFold / feature cache / model config で比較できる。
- inference notebook は selected target がない限り明示的に止まる。
- `py_compile`、notebook JSON validation、ruff、`validate_experiment.py` が通る。
