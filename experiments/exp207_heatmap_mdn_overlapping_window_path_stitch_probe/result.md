# exp207_heatmap_mdn_overlapping_window_path_stitch_probe 結果

## 状態

Kaggle train v2 は COMPLETE。kernel は `kentookumura/exp207-hmdn-path-stitch-train`、CPU、GPU disabled。inference / submit は実施していない。

v1 は notebook metadata に `kernelspec.name` がなく、Kaggle papermill が `ValueError: No kernel name found in notebook and no override provided.` で失敗した。v2 で train / inference percent source に Python 3 kernelspec header を追加し、同じ kernel id に再 push して完走した。

output は `experiments/exp207_heatmap_mdn_overlapping_window_path_stitch_probe/kaggle/output/train_v2` に取得済み。

## 主結果

- source windows: 773 wells、14 windows / well、10,822 samples。
- 現行 exp202 v2 artifact の overlap は非常に薄い。source overlap は 3 wells / 39 center pairs のみで、gap center pairs は 10,010。
- stitched coverage は exp099 cache 3,783,989 rows のうち 1,333,241 rows、coverage rate `0.352337441`。
- covered rows 上の existing union oracle RMSE は `5.154353660`、within10 `0.947161091`。
- existing + stitched top1 は oracle RMSE `4.472998031`、delta `-0.681355629`、within10 `0.958441872`。
- existing + stitched top3 は oracle RMSE `4.418699605`、delta `-0.735654055`、within10 `0.959487445`、new-best candidate rate `0.069157039`。
- stitched only top3 は oracle RMSE `50.798377042`、within10 `0.275946359` で単体候補としては粗い。

## bucket / well readout

`1000_plus` bucket では existing oracle RMSE `6.376418` から union top3 `5.414525` に改善し、new-best candidate rate は `0.091269`。near `0_50` は `0.313605` から `0.310983` とほぼ同等。

well 単位では 773 wells 中 461 improved、312 same、0 worse。平均 RMSE delta は `-0.338524069`、中央値は `-0.003918543`。最大改善は `1b1eba53` の `37.761571 -> 17.418515`。

## 物理性 / coverage

best path rank distribution は rank1 `0.632693`、rank2 `0.216503`、rank3 `0.087692`。assignment overlap abs mean は `4.555301 ft`、gap boundary abs mean は `9.725981 ft`、p95 は `31.206543 ft`。

ただし source overlap が 3 wells だけなので、この結果は full-well overlapping-window stitch の証拠ではなく、sparse local path artifacts を stitched candidate として replay した診断に留まる。

## 判断

train-side oracle headroom はある。covered rows 上で existing + stitched top3 は existing union から `-0.735654` RMSE 改善し、worse well もない。

一方、stitched path 単体は弱く、現行 exp202 v2 artifact は sparse で overlap が足りない。したがって direct replacement、softmax average、PF weight replacement、inference port、submit には進めない。backlog `heatmap_mdn_overlapping_window_path_stitch_probe` は exp207 で完了として閉じ、次に進めるなら dense stride window path regeneration を別候補として扱う。

## 主 SHA

- exp202 path npz: `e615f0d01a08fd37685fd1ac46335b99306f0bb0c9c43d37c1e1f620040839a3`
- exp202 path samples decompressed: `cea6a29c716a1c5dedda1efec64a1b1f2371d1eadfd298084576f06170d0a7de`
- exp099 candidate cache decompressed: `1939d536b1e56f7c0ea3847cc386ef769b0d33759d16e816c9ce180f0532df9a`
- stitched path rows decompressed: `5177a3e11a06675b0b591d53d9c0ac3c0da29f5ee9b08daf686273ddadb1945b`
- stitched window assignments decompressed: `ce9b637961ca9f549d42ae6d613c8741da901b13803f5d408d11586ff5abc1be`
- candidate union metrics: `3e029251de93b86c576bd9319cf39accab9d545080f574e831c5abda711d33bd`

## 次アクション

続けるなら `heatmap_mdn_dense_stride_window_path_regeneration_probe` として、exp202 model artifact から stride 32/64 などの dense local path を再生成し、overlap pair 数と row coverage を増やしたうえで stitch を再評価する。
