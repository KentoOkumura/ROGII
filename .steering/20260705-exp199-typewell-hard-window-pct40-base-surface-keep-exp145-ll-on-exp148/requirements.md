# 要件

## 依頼

`KAGGLE_DIRECTION.md` の backlog `typewell_hard_window_pct40_base_surface_keep_exp145_ll_on_exp148` を `exp199_typewell_hard_window_pct40_base_surface_keep_exp145_ll_on_exp148` として実装する。

## 制約

- Route: `ml_model`
- 親実験は `exp148_learned_likelihood_fulltrain_addonly_on_exp092`。
- base 196 features と projection / U-disagreement は `exp196_typewell_late_range_hard_window_pct40_full_cache_replacement` 由来にする。
- exp145 `ll_*` learned-likelihood features は exp148 と同じ full-train cache から読み、active model に残す。
- これは `exp196 base + exp145/exp072-derived ll_*` の混合 provenance 診断であり、直接 inference / submit 候補にしない。
- control / parent retraining は行わない。
- Kaggle train する場合の計画は 1 active variant x 1 mode x 3 LightGBM configs x 5 folds = 15 boosters。
- 再現性: `docs/06_reproducibility.md` に従い、input cache SHA、feature schema SHA、model manifest SHA、prediction SHA を train 結果記録時に残す。

## 受け入れ基準

- `.steering/20260705-exp199-typewell-hard-window-pct40-base-surface-keep-exp145-ll-on-exp148/` が存在し、requirements / design / tasklist が埋まっている。
- `experiments/exp199_typewell_hard_window_pct40_base_surface_keep_exp145_ll_on_exp148/` が存在する。
- `config.yaml` の `experiment.route` が `ml_model`、`lineage.parent` が exp148、`lineage.base_surface_parent` が exp196、`lineage.learned_likelihood_parent` が exp145 である。
- active variant は `pct40_base_surface_keep_exp145_ll_mixed_provenance` の 1 個で、feature groups は `projection_correction`、`u_disagreement`、`learned_likelihood_confidence`。
- train notebook source は `data.base_surface_train_feature_cache_local` から exp196 pct40 cache を読み、exp145 learned-likelihood cache を join する。
- inference notebook は train-side diagnostic only とし、submission 生成を行わない。
- `py_compile`、`ruff --select F821`、`validate-exp` が通る。
