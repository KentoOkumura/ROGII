# 要件

## 依頼

backlog `heatmap_mdn_full_well_path_generation_probe` を実装する。exp202/exp208 heatmap MDN local paths から、後続 selector が入力として読める well ごとの full-row candidate path artifact を生成する。

## 制約

- Route: `pf_beam`
- 親実験: `exp202_heatmap_mdn_candidate_generator_probe`
- dense path source: `exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe`
- stitch score: exp207/208 と同じ target-free beam stitch を使う。
- 再現性: `docs/06_reproducibility.md` に従い、input/output SHA、decompressed gzip SHA、Kaggle kernel source を記録する。
- GPU 学習、LightGBM 学習、parent/control retraining は行わない。
- true TVT、oracle best、abs-error、within10、candidate true-error rank を path generation、stitch score、full-path contract table に入れない。
- direct TVT replacement、softmax average、PF weight replacement、postprocess blend、inference、submit は行わない。

## 受け入れ基準

- `experiments/exp210_heatmap_mdn_full_well_path_generation_probe/` が作成され、`config.yaml` の `experiment.route` が `pf_beam` である。
- train notebook が exp208 dense path artifact を読み、local topK 5/10 の full-well top5 candidates を生成する。
- primary output として `*_localtopk10_full_well_candidate_paths.csv.gz` を保存し、少なくとも `well`、`row_id`、`md_from_ps`、`path_rank`、`tvt_pred`、`source_window_count`、`overlap_weight`、`assignment_gap_flag`、`local_rank_mix`、`path_step_abs`、`curvature_abs`、`candidate_score` を含む。
- schema CSV、contract metrics CSV、candidate union metrics、distance bucket metrics、by-well metrics、summary JSON を保存する。
- top1/top3/top5 の full-well candidates について、stitched-only oracle と existing + full-path oracle を評価する。
- gzip 生成物は decompressed content SHA を summary / metrics に記録する。
- inference notebook は train-side diagnostic only として submit しないことを明示する。
