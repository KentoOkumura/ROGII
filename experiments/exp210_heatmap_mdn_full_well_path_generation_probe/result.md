# exp210_heatmap_mdn_full_well_path_generation_probe 結果

## 状態

Kaggle train v1 は COMPLETE。kernel は `kentookumura/exp210-hmdn-full-well-path-generation-train`、CPU、GPU disabled、internet disabled。inference / submit は実施していない。

output は `experiments/exp210_heatmap_mdn_full_well_path_generation_probe/kaggle/output/train_v1` に取得済み。

## 主要結果

- exp208 dense path input: `25,452` samples、`773` wells、topK `10`、path tensor `(25452, 10, 128)`。
- exp099 candidate cache: `3,783,989` rows / `773` wells。available existing candidates は `pf_ancc`、`beam_mean`、`likpf_mean`、`sc_ens`、`hyb`。
- primary local topK10 full-well contract は required columns present `True`、duplicate key rows `0`、null required value count `0`。
- full-well contract rows は `8,137,310`、unique row ids は `1,627,462`、wells は `773`、coverage vs exp099 cache は `0.430091631`。
- rows by rank は top1-top5 すべて `1,627,462` rows。path count は `3,865`。
- source overlap は `773` wells / `24,679` pairs、source gap pair count `0`。stitched row gap count も `0`。
- physicality は local topK10 で path step abs mean `0.248510989 ft`、curvature abs mean `0.487512452 ft`、overlap row rate `0.941656697`。

## Oracle readout

covered rows 上の existing union oracle RMSE は `5.139413349`、within10 は `0.947522584`。

local topK10:

- stitched only top1: RMSE `47.553115411`、within10 `0.274478913`。
- stitched only top3: RMSE `47.188322489`、within10 `0.285964895`。
- stitched only top5: RMSE `46.958946049`、within10 `0.292765668`。
- existing + stitched top1: RMSE `4.449415331`、delta `-0.689998018`、new-best rate `0.066178504`。
- existing + stitched top3: RMSE `4.420752853`、delta `-0.718660496`、new-best rate `0.071626864`。
- existing + stitched top5: RMSE `4.407737500`、delta `-0.731675849`、new-best rate `0.075083781`。

distance bucket では `1000_plus` が existing `6.352450934` から union `5.403899359` に改善し、delta は `-0.948551575`。by-well は `524 improved / 249 same / 0 worse`。best well は `1b1eba53` で RMSE `37.534730853 -> 21.331056019`。

## 判断

この backlog の成功条件だった selector-facing covered-row artifact contract、coverage、schema parity、duplicate/gap 監査、physicality、SHA 記録は成立した。したがって `heatmap_mdn_full_well_path_generation_probe` は完了扱いにする。

一方、この生成物は exp099 candidate-cache intersection の covered rows に限定されており、exp072/exp083 plot の全 `md_since` 区間を覆う full-grid trajectory ではない。stitched-only も RMSE `46.958946049` と粗いため、単独 direct replacement、softmax average、PF weight replacement、postprocess blend、inference、submit には進めない。selector の通常候補として使うには、別 backlog `exp212_heatmap_mdn_full_grid_path_generation_probe` で全 row grid coverage / continuity / oracle headroom を先に検証する。

## 主要生成物

- `artifacts/exp210_heatmap_mdn_full_well_path_generation_probe_localtopk10_full_well_candidate_paths.csv.gz`
- `artifacts/exp210_heatmap_mdn_full_well_path_generation_probe_localtopk10_full_well_contract_metrics.csv`
- `artifacts/exp210_heatmap_mdn_full_well_path_generation_probe_localtopk10_full_well_path_schema.csv`
- `artifacts/exp210_heatmap_mdn_full_well_path_generation_probe_localtopk10_candidate_union_metrics.csv`
- `artifacts/exp210_heatmap_mdn_full_well_path_generation_probe_localtopk10_candidate_union_distance_bucket_metrics.csv`
- `artifacts/exp210_heatmap_mdn_full_well_path_generation_probe_localtopk10_candidate_union_by_well.csv`
- `artifacts/exp210_heatmap_mdn_full_well_path_generation_probe_summary.json`

primary full-well candidate path decompressed SHA256 は `f22808f0c0af8cc8a2953680284db9d8564fcecfa401a8921c6130e29f8509f0`。

## 実装時 smoke

2 wells debug smoke では primary local topK `10` の full-well contract で required columns present `True`、duplicate key rows `0`、null required value count `0`、debug rows `21,110`、unique row ids `4,222`、path count `10` を確認した。これは Kaggle 評価結果ではない。
