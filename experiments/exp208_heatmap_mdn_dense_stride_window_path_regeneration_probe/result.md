# exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe 結果

## 状態

Kaggle train v1 は COMPLETE。kernel は `kentookumura/exp208-hmdn-dense-stride-train`、CPU、GPU disabled、internet disabled。inference / submit は実施していない。

output は `experiments/exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe/kaggle/output/train_v1` に取得済み。

## 主要結果

- exp202 saved fold model から stride 64 dense local path artifact を生成できた。dense samples は `25,452`、対象 wells は `773`、topK は `10`。
- source overlap は exp207 の `3 wells / 39 pairs` から、exp208 では `773 wells / 24,679 pairs` に増えた。gap pair count は `0`。
- row coverage は exp207 の `0.352337441` から、exp208 は `0.430091631` に増えた。
- local topK10 の stitched only top3 oracle RMSE は `47.188322489`。exp207 の `50.798377042` より改善したが、単体候補としてはまだ弱い。
- local topK10 の existing + stitched top3 oracle RMSE は `4.420752853`、delta `-0.718660496`、within10 `0.960702615`、new-best rate `0.071626864`。
- exp207 の existing + stitched top3 oracle RMSE は `4.418699605` だったため、dense 化しても oracle top3 は微小に悪化した。
- `1000_plus` bucket は existing `6.352450934` から union `5.422416550` へ改善したが、exp207 の `5.414525` には届かない。
- by-well は local topK10 で `509 improved / 264 same / 0 worse`。best well は `1b1eba53` で `37.534730853 -> 21.331056019`。
- physicality は overlap rows `4,740,687`、overlap abs mean `7.196697638 ft`、stitched step abs mean `0.247286874 ft`、curvature abs mean `0.485335775 ft`、stitched row gap count `0`。

## exp207 との差分

exp208 は、設計どおり各 well の dense stride local paths を再生成している。全 row そのものではなく stride 64 の row-center windows だが、各 well に平均 `32.926` windows を作り、128-row local paths の overlap を持つため、exp207 の sparse artifact より full-well stitch の物理性を評価しやすい。

一方、追加 coverage と overlap は oracle top3 の更新にはつながらなかった。stitched only は exp207 より良くなったものの、RMSE `47.188` で既存 PF/Beam 候補群の代替にはならない。existing union に足すと covered rows 上で改善はするが、exp207 の sparse stitch と同程度で、密化による上積みは確認できない。

## 判断

この backlog の主目的である「dense path が overlap 付き full-well stitch として成立するか」は確認できた。ただし、exp207 からの実利改善はなく、direct replacement、softmax average、PF weight replacement、inference、submit には進めない。

後続は dense stitch 自体の再調整より、既存の exp204 selector candidate route で heatmap MDN 候補を target-free selection に渡す方向を優先する。stride 32 の再実行は、artifact size / CPU time に対する期待値が低いため現時点では保留する。
