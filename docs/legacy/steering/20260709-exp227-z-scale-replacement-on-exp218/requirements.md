# 要件

## 依頼

`z_scale_replacement_on_exp218` を `exp227_z_scale_replacement_on_exp218` として実装する。

exp218 の ML submitted anchor に対して、exp224 で実装済みの target-free well-scaled `z` / `dz` / `dzdmd` / `slp_z` feature builder を再利用する。ただし exp224 の add-only ではなく、raw `z` / `dz` / `dzdmd` / `slp_z` 4 列を model feature list から外し、scaled z 系特徴へ置き換える。

## 制約

- Route: `ml_model`
- 親実験: `exp218_gr_wavelet_rotation_confidence_features_on_exp148`
- 比較 baseline: exp218 `lgb_mean` CV 8.475793752 / Public LB 7.843
- exp224 add-only は exp218 比 +0.062893290 悪化済みのため、raw 4 列を残す add-only 再試行にはしない。
- Kaggle train は CPU 実行とし、タイムアウト対策として `train_lgb0` / `train_lgb1` / `train_lgb2` に分割する。
- active variant は `z_scale_replacement` の 1 本のみ。
- 親 exp218 control / baseline は再学習しない。
- target-derived scaler、direct correction、candidate replacement、blend、postprocess、hard selector、sample-weight 変更は禁止。
- likPF p05-p95 range scale は scale feature に限定し、candidate correction には使わない。
- 再現性は `docs/06_reproducibility.md` に従い、Kaggle bootstrap、feature schema、model manifest、prediction SHA を記録対象にする。

## 受け入れ基準

- `experiments/exp227_z_scale_replacement_on_exp218/` が exp224 から派生されている。
- `model.feature_ablation.active_variants` に `z_scale_replacement` だけが enabled で定義され、`drop_base_columns: [z, dz, dzdmd, slp_z]` を持つ。
- train / inference の feature selection が `drop_base_columns` を反映する。
- CPU mode のみ enabled で、GPU は false。
- train notebook は `train_lgb0` / `train_lgb1` / `train_lgb2` に分割され、各 split が 1 LightGBM config x 5 folds = 5 boosters を実行する。
- 合計 booster 数は 15、control retraining はなしとして `SESSION_NOTES.md` に記録されている。
- Jupytext 変換、`py_compile`、`ruff --select F821,F401`、`validate_exp`、Kaggle package prepare が通る。
