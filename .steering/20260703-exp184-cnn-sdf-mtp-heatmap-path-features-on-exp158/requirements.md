# 要件

## 依頼

`cnn_sdf_mtp_heatmap_path_features_on_exp158` backlog を実装する。exp182 の CNN/SDF/MTP heatmap topK path 出力を、exp157/158 PF/Beam candidate selector の add-only confidence feature として使う。

## 制約

- Route: `pf_beam`
- 親実験: `exp158_segment_continuity_selector_on_exp157`
- heatmap source: `exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe` の `base_real_w128_b64_fullfold` validation predictions を主入力にする。
- `base_shuffled_w128_b64_fullfold` と `base_no_gr_w128_b64_fullfold` は confidence gap feature のみに使う。
- geometry channel と 256x96 window は exp182 で支持されていないため初手では使わない。
- Heatmap path center を direct TVT replacement、softmax weighted TVT、PF weight replacement、hard selector、submission candidate として使わない。
- exp182 の `pred_top*_abs_error`、`top*_within10`、true TVT、oracle label、true-error rank は feature source に使わない。
- exp182 validation predictions は sparse sample なので、well 内 row index で補間して exp158 の row-level selector frame に展開する。
- Kaggle train push 前に active variant 1、LightGBM 3 configs、5 folds、15 boosters、control/parent retraining なしを `SESSION_NOTES.md` に記録する。
- 再現性: `docs/06_reproducibility.md` に従い、upstream exp182 GPU model / gzip input / OOF prediction / model manifest / feature schema の SHA 記録方針を残す。

## 受け入れ基準

- `.steering/20260703-exp184-cnn-sdf-mtp-heatmap-path-features-on-exp158/` に要件、設計、タスクが記録されている。
- `experiments/exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158/` に config、補助実装、train/inference notebook source、notebook、README、result、metrics placeholder、SESSION_NOTES がある。
- train notebook は exp182 heatmap feature source、禁止する target-derived columns、Viterbi variant 数、LightGBM booster 数を表示する。
- 実装は exp157 の 8候補、dense enrichment、exp158 Viterbi grid を維持し、heatmap path features を add-only で追加する。
- `py_compile`、`ruff --select F821`、`jupytext --to ipynb --test` が通る。
- ローカルに不足している upstream cache がある場合は、その不足と Kaggle kernel source 依存を記録する。
