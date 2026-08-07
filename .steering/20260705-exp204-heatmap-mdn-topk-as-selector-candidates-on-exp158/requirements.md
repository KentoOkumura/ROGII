# 要件

## 依頼

`heatmap_mdn_topk_as_selector_candidates_on_exp158` backlog を実装する。exp202 の heatmap MDN top10 path を exp158/184 selector の selectable candidate として追加し、Viterbi/continuity constraint 付きで target-free に選べるかを train-side audit できる状態にする。

## 制約

- Route: `pf_beam`
- 親は exp203 feature-only 実装を使い、exp184 / exp158 selector と比較できる形を保つ。
- selector 候補は既存 8 候補に `hmdn_top1` ... `hmdn_top10` を加えた 18 候補とする。
- heatmap path を direct replacement、softmax weighted TVT、PF weight replacement、postprocess blend、submission には使わない。
- exp202 の `true_center_tvt`、`target_in_grid`、`best_mode`、abs-error、within10、oracle-best、true-error rank 系列を feature に入れない。
- 再現性: `docs/06_reproducibility.md` に従い、upstream GPU heatmap 由来の非 deterministic 性、LightGBM seed、candidate-long subsample、gzip decompressed SHA 記録方針を明記する。

## 受け入れ基準

- exp204 の experiment directory、config、train/inference notebook source、README、SESSION_NOTES、result、metrics が exp204 名で整っている。
- `ranker.candidates` に 18 候補が定義され、`selector.allowed_switch_candidates` に heatmap MDN top10 が含まれる。
- 初期 exp099 cache に存在しない hmdn candidate columns が required column check を壊さない。
- candidate-long feature に heatmap MDN 候補の rank / score / family flag が入る。
- py_compile、ruff F821、Jupytext conversion/test、`make validate-exp` が通る。
- Kaggle train push 前の予定コストとして 1 active selector variant、18 candidates、3 LightGBM configs、5 folds、15 boosters、control retraining なしが `SESSION_NOTES.md` に記録されている。
