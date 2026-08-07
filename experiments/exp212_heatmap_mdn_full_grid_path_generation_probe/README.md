# exp212_heatmap_mdn_full_grid_path_generation_probe

## 目的

exp208 の dense heatmap MDN local paths を source にして、exp099 feature-cache row grid 全体を覆う topK continuous candidate path artifact を作る。exp210 は covered rows 限定の contract だったため、後続 selector が通常候補として読む前に、全 row coverage、fallback 比率、continuity、oracle headroom をここで検証する。

## 仮説

exp208 dense local paths は direct replacement には弱いが、全 row grid へ補間・外挿つきで contract 化すれば、selector が通常候補として扱える candidate schema と fallback 診断を得られる。既存 PF/Beam union との oracle headroom が fallback rows でも保たれる場合だけ、exp204 系の selectable candidate 追加に進む価値がある。

## 方針

- Route: `pf_beam`
- 親: `exp202_heatmap_mdn_candidate_generator_probe`
- dense source: `exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe`
- comparison: `exp099_pf_multi_observation_likelihood_probe`
- downstream: `exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158`
- 実行: exp208 cached dense paths を target-free stitch し、sparse stitched rows を exp099 row grid へ reindex する。
- active local topK: `10` のみ。full-grid output が大きいため、初回は primary artifact に絞る。
- fill: source-covered rows はそのまま使い、未カバー rows は row_index 線形補間または端点外挿で埋める。

## 出力 contract

primary artifact は `exp212_heatmap_mdn_full_grid_path_generation_probe_localtopk10_full_grid_candidate_paths.csv.gz`。

必須列:

- `id`
- `well`
- `row_id`
- `row_index`
- `md_since`
- `path_rank`
- `tvt_pred`
- `source_window_count`
- `overlap_weight`
- `assignment_gap_flag`
- `local_rank_mix`
- `path_step_abs`
- `curvature_abs`
- `candidate_score`
- `coverage_flag`
- `fallback_flag`

`coverage_flag=true` は exp208 local window が直接その row を出したことを表す。`fallback_flag=true` は補間または端点外挿で埋めた row を表す。true TVT、oracle best、abs error、within10、candidate true-error rank は path generation / fill / candidate_score に入れない。

## 検証方針

Kaggle CPU train-side diagnostic として実行する。full-grid contract、duplicate/null、row coverage、source coverage、fallback rate、fill method、path continuity、curvature、stitched-only top1/top3/top5 RMSE、existing+new oracle、distance bucket、by-well improved/same/worse、入力/出力 SHA を確認する。

## 所見

Kaggle train v1 は `COMPLETE`。full-grid contract は成立し、exp099 feature-cache row grid の `3,783,989` unique rows を rank `1..5` で全覆盖できた。duplicate key rows は `0`、null required values は `0`。

一方、source-covered unique row ids は `1,627,462`、source coverage は `0.4300916308160515` で、`2,156,527` unique rows (`0.5699083691839485`) が fallback。実 run では未カバー行はすべて `right_extrapolated` だった。

Oracle readout は existing union RMSE `7.434029841` に対して existing + stitched top5 RMSE `5.941479995` と headroom を示したが、stitched-only top5 RMSE は `50.085237573` と弱い。artifact は selector candidate follow-up の入力候補として残すが、direct replacement、softmax average、PF weight replacement、inference、submit には使わない。

Plot overlay では途中から直線 tail になる。これは exp208 source が `max_tail_rows=2048` までの dense windows に限定され、exp212 がその後ろを `np.interp` の endpoint hold で埋めているため。したがって exp212 は「最後まで heatmap path を生成した artifact」ではなく、source-supported prefix + fallback tail の診断 artifact として扱う。

## 状態

`kaggle_train_v1_complete_diagnostic_only`。CNN 0 models、LightGBM 0 configs / 0 boosters、parent/control retraining なし、GPU disabled。
