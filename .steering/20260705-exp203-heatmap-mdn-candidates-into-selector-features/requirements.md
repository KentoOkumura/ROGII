# 要件

## 依頼

`heatmap_mdn_candidates_into_selector_or_ml_features` の selector 側を実装する。exp202 の heatmap MDN topK candidate は既存 selector の入力特徴量として追加し、selectable candidate にはしない。

## 制約

- Route: `pf_beam`
- 親: `exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158`
- selector 候補集合は exp184 と同じ 8 候補で固定する。
- exp202 topK TVT は direct replacement、softmax average、PF weight replacement、blend、postprocess、submit に使わない。
- `true_center_tvt`、abs-error、within10、oracle 系列、true-error rank は feature にしない。
- 再現性: `docs/06_reproducibility.md` に従い、upstream stochastic artifact として exp182/exp202 を記録する。

## 受け入れ基準

- `hmdn_` feature block が exp184 selector trainer に add-only で入っている。
- `ranker.candidates` は 8 候補のまま変わっていない。
- row-level と candidate-long の両方に heatmap MDN distance/confidence feature がある。
- hmdn confidence / sparse-distance bucket が診断に出る。
- py_compile、ruff F821、Jupytext conversion/test、validate-exp が通る。
