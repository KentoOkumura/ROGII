# 要件

## 依頼

`exp212_heatmap_mdn_full_grid_path_generation_probe` backlog を実験化する。バックログ名に exp212 が含まれるが、リポジトリの最新番号を確認し、正しい実験番号で作る。

## 制約

- Route: `pf_beam`
- 親: `exp202_heatmap_mdn_candidate_generator_probe`
- dense source: `exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe`
- comparison: `exp099_pf_multi_observation_likelihood_probe`
- downstream at design time: `exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158` (later closed/rejected after exp215)
- CNN 再学習、LightGBM 学習、selector training、inference、submit は範囲外。
- 再現性: `docs/06_reproducibility.md` に従い、入力 SHA、gzip decompressed SHA、Kaggle kernel version を記録する。

## 受け入れ基準

- exp099 feature-cache row grid を対象に full-grid candidate path artifact を出力する。
- 必須列 `id,well,row_id,row_index,md_since,path_rank,tvt_pred,source_window_count,overlap_weight,assignment_gap_flag,local_rank_mix,path_step_abs,curvature_abs,candidate_score,coverage_flag,fallback_flag` を持つ。
- `row_coverage_rate_vs_cache=1.0` と、source-covered row 比率 / fallback row 比率を分けて記録する。
- duplicate key rows、null required values、row gaps、path continuity、curvature を監査する。
- stitched-only top1/top3/top5 RMSE、existing+new oracle、distance bucket、by-well improved/same/worse を train-side oracle readout として保存する。
- true TVT、oracle best、abs-error、within10、candidate true-error rank を path generation、fill、candidate_score に使わない。
