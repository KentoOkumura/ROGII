# 要件

## 依頼

`KAGGLE_DIRECTION.md` のアイデアバックログ最上位 `pseudo_tail_postprocess_cv_audit` を実験化する。

`exp024` raw pseudo-tail anchor に対して、distance bucket shrink / near-only shrink / far alpha が clean CV で改善するかを、same-OOF fit ではなく held-out 監査で確認する。

## 制約

- 親 anchor は `exp024_pseudo_tail_inference_postprocess` の raw pseudo-tail とする。
- `exp023` selected variant `pseudo_tail_3_cutoffs_distance_balanced` の fold-safe OOF 相当を再生成する。
- pseudo cutoff は train fold の well 内だけで作り、valid fold は元の `TVT_input` NaN tail だけを評価する。
- postprocess alpha の採用判断は original-fold 外、または well-hash holdout の評価で行う。
- same-OOF alpha fit は診断値として記録するが、提出判断には使わない。
- この実験では submission を生成しない。

## 受け入れ基準

- exp025 の steering docs と実験ディレクトリがある。
- train notebook が selected pseudo-tail OOF 再生成と postprocess audit を実行できる。
- raw、exp014 bucket shrink、near-only shrink、far alpha、bucket alpha fit の metrics / alphas / bucket summary が artifact として保存される。
- `metrics.json` に selected clean CV、selected method、postprocess 採用可否が保存される。
- validation / compile / lint が通る。
