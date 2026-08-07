# exp210_heatmap_mdn_full_well_path_generation_probe

## 目的

exp202/exp208 の heatmap MDN local path を、後続 selector が読める `well,row_id,path_rank,tvt_pred` 形式の full-well candidate path artifact に整形する。exp208 では dense stitch の oracle 改善自体は弱かったが、selector 候補追加に進む前の exp099 candidate-cache intersection schema、coverage、physicality、SHA 記録をここで固める。

## 仮説

exp208 dense local paths は direct replacement には弱いが、target-free stitch 後の topK candidates を exp099 candidate-cache covered rows に保存すれば、後続 selector が heatmap MDN 候補を扱うための schema / join / physicality 契約を確認できる。

## 方針

- Route: `pf_beam`
- 親: `exp202_heatmap_mdn_candidate_generator_probe`
- dense path source: `exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe`
- stitch score: exp207/208 と同じ target-free beam stitch
- 入力: exp208 `dense_candidate_paths_top10.npz` と `dense_path_samples.csv.gz`
- 出力: local topK 5/10 それぞれの full-well top5 candidate path table
- 比較: exp099 candidate cache との row alignment、coverage、source overlap、physicality、existing + full-path oracle

## 出力 contract

primary artifact は `exp210_heatmap_mdn_full_well_path_generation_probe_localtopk10_full_well_candidate_paths.csv.gz`。行は downstream selector と join できるよう、exp099 candidate cache と一致する `id` に限定する。

必須列:

- `id`
- `well`
- `row_id`
- `row_index`
- `md_from_ps`
- `path_rank`
- `tvt_pred`
- `source_window_count`
- `overlap_weight`
- `assignment_gap_flag`
- `local_rank_mix`
- `path_step_abs`
- `curvature_abs`
- `candidate_score`

`candidate_score` は target-free な stitch cost の単調変換で、true TVT、oracle best、abs-error、within10、candidate true-error rank は path generation / stitch score / contract table に入れない。

## 範囲外

direct TVT replacement、softmax weighted average、PF weight replacement、postprocess blend、selector training、raw-test generation、inference、submit は行わない。この artifact は全区間 trajectory ではないため、selector の通常候補として使う前には `exp212_heatmap_mdn_full_grid_path_generation_probe` で full-row coverage / continuity を別途検証する。

## 検証方針

Kaggle CPU train-side diagnostic として実行する。exp208 dense path input、exp099 cache、full-well path output の SHA を記録し、required schema、duplicate key、null required values、coverage、source overlap、path step、curvature、assignment gap、top1/top3/top5 の stitched-only oracle と existing + full-path oracle を確認する。

## 所見

Kaggle train v1 COMPLETE。covered-row contract は required columns present、duplicate key rows 0、null required value count 0 で成立した。primary local topK10 の contract rows は 8,137,310、unique row ids は 1,627,462、coverage vs exp099 cache は 0.430091631。source overlap は 773 wells / 24,679 pairs、source gap pair count 0、stitched row gap count 0。

existing + stitched top5 oracle RMSE は 4.407737500 で、existing union 5.139413349 から -0.731675849 改善した。一方 stitched only top5 は RMSE 46.958946049 と弱く、さらに全 row grid を覆らないため、direct replacement、softmax average、PF weight replacement、inference、submit はしない。

## 状態

完了。Kaggle output は `kaggle/output/train_v1` に取得済み。予定どおり CNN 0 models、LightGBM 0 configs / 0 boosters、parent/control retraining なし。
