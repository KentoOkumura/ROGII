# 要件

## 依頼

`exp184_heatmap_selector_compact_addonly_on_exp148` backlog を実装する。exp184 の CNN/SDF/MTP heatmap selector signal を exp148 の ML route anchor に compact add-only feature として追加する。

## 制約

- Route: `ml_model`
- 親は `exp148_learned_likelihood_fulltrain_addonly_on_exp092`。
- exp148 の base 196 features、`projection_correction`、`u_disagreement`、`learned_likelihood_confidence` は維持する。
- active variant は 1 個、LightGBM 3 configs x 5 folds = 15 boosters。
- Kaggle train は CPU 実行にする。
- タイムアウト対策として学習 notebook を `train_lgb0` / `train_lgb1` / `train_lgb2` に分割し、各 notebook は 1 config x 5 folds = 5 boosters だけを担当する。
- parent/control 再学習はしない。
- exp184 selected TVT の direct replacement、blend、postprocess、hard gate、submit はしない。
- exp184 の 126 heatmap path features 全量投入はしない。
- exp184 OOF / exp182 heatmap prediction の `true_tvt`、`abs_error`、`within10`、`target_in_grid`、oracle 系列は特徴量源に使わない。
- 再現性: `docs/06_reproducibility.md` に従い、upstream artifact、CPU LightGBM、Kaggle bootstrap、SHA 記録方針を設計に明記する。

## 受け入れ基準

- `experiments/exp184_heatmap_selector_compact_addonly_on_exp148/` に config、helper、train/inference notebook source、notebook、README、result、metrics placeholder、SESSION_NOTES がある。
- train notebook が exp148 parent、exp184 selector source、exp182 heatmap source、active variant、planned booster 数を表示する。
- `train_lgb0` / `train_lgb1` / `train_lgb2` notebook が `selected_lgb_config_indices=[0|1|2]` で個別に実行できる。
- compact feature block が selected candidate/family、selected vs `likpf_mean`、selected vs exp148 OOF、segment stability、heatmap score/margin/entropy、sparse distance、real-vs-control confidence gap を含む。
- `learned_likelihood_confidence` block を置換しない。
- Jupytext 変換、`py_compile`、`ruff --select F821`、`make validate-exp` が通る。
