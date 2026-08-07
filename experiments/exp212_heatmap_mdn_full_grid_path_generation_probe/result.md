# exp212_heatmap_mdn_full_grid_path_generation_probe 結果

## 状態

Kaggle train v1 完了。診断 artifact は生成済みだが、direct replacement、softmax average、PF weight replacement、inference、submit はしない。

- kernel: `kentookumura/exp212-hmdn-full-grid-path-generation-train`
- version: `1`
- runtime: `4635.838601125` sec
- output: `kaggle/output/train_v1`
- primary artifact: `artifacts/exp212_heatmap_mdn_full_grid_path_generation_probe_localtopk10_full_grid_candidate_paths.csv.gz`

## Contract

exp099 feature-cache row grid 全体に対する full-grid contract は通った。

- rows: `18,919,945`
- unique row ids: `3,783,989`
- wells: `773`
- row coverage vs cache: `1.0`
- duplicate key rows: `0`
- null required value count: `0`
- required columns present: `true`
- rows by rank: rank `1..5` が各 `3,783,989`
- path count: `3,865`

ただし source support は限定的だった。

- source-covered unique row ids: `1,627,462`
- source coverage vs grid: `0.4300916308160515`
- fallback unique row ids: `2,156,527`
- fallback unique row rate: `0.5699083691839485`
- fill method rows: `source_window=8,137,310`, `right_extrapolated=10,782,635`

## Oracle Readout

既存 PF/Beam union に対して heatmap path を追加した場合の oracle headroom はある。

| candidate set | RMSE | MAE | within10 | new-best rate |
| --- | ---: | ---: | ---: | ---: |
| existing union | 7.434029841 | 3.745227753 | 0.906525363 | - |
| stitched only top5 | 50.085237573 | 31.427729178 | 0.314262013 | - |
| existing + stitched top5 | 5.941479995 | 3.110541407 | 0.933460166 | 0.115735009 |

`existing + stitched top5` は existing union から RMSE `-1.492549846`、within10 `+0.026934803`。by-well は `567 improved / 206 same / 0 worse`、`1000_plus` は `8.161796577 -> 6.491812973`。

## 判定

Full-grid artifact と schema は成立した。一方で、行の `56.99%` が右端点外挿で、stitched-only top5 RMSE は `50.085237573` と弱い。したがって exp212 artifact は単独予測や平均化には使わず、後続 exp204 系で selector が guarded candidate として扱えるかだけ検証する。

プロットで途中から直線になる挙動は、exp212 の既知の失敗モードとみなす。親の exp208 dense source は各 well で約 33 windows / 2175 source rows に限定され、`model.training.max_tail_rows=2048` の範囲しか heatmap 推論していない。exp212 はその後ろを `np.interp` の endpoint hold で埋めているため、最後まで新しい path を生成しているわけではない。

後続へ渡す場合は `coverage_flag`、`fallback_flag`、`fill_method`、`candidate_score`、`source_window_count`、`overlap_weight`、`path_step_abs`、`curvature_abs` を必ず feature / guard として使い、fallback bucket の選択率と誤差を監査する。
