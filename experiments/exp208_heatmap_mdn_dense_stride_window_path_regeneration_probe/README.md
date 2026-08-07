# exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe

## 目的

exp207 で確認した heatmap MDN local path stitch の弱点、つまり source overlap がほぼ存在しない問題を切り分ける。exp202 の saved fold model を再利用し、validation wells に対して dense row-center stride window の local path を再生成してから、同じ target-free stitch readout を行う。

## 仮説

exp207 の covered-row oracle headroom は positive だったが、入力 path artifact は 14 windows / well の sparse output だった。stride 64 の dense window にすると 128-row local path 間の overlap が増え、full-well stitch としての物理性、coverage、candidate union headroom をより正しく評価できる。

## 方針

- Route: `pf_beam`
- 親: `exp202_heatmap_mdn_candidate_generator_probe`
- 比較: `exp207_heatmap_mdn_overlapping_window_path_stitch_probe`
- cache: `exp099_pf_multi_observation_likelihood_probe`
- dense generation: validation wells、stride 64、topK 10
- stitch readout: local topK 5 / 10、beam width 6、output top3
- GPU 学習なし、LightGBM なし、inference / submit なし

## 範囲外

direct TVT replacement、softmax weighted average、PF weight replacement、postprocess blend、selector training、inference port、submit は行わない。positive の場合も、まず物理性と target-free selection 可能性を別実験で検証する。

## 検証方針

Kaggle CPU train-side diagnostic として実行する。exp202 saved fold model から dense stride local path artifact を生成し、local topK 5 / 10 の stitch readout を source overlap、row coverage、gap boundary abs、overlap disagreement、stitched only oracle、existing + stitched oracle、distance bucket、by-well delta、path step / curvature で比較する。

## 所見

Kaggle train v1 COMPLETE。dense stride local path は全 773 wells で生成でき、source overlap は exp207 の `3 wells / 39 pairs` から `773 wells / 24,679 pairs` へ増えた。row coverage も `0.352337441` から `0.430091631` へ改善した。

ただし、local topK10 の existing + stitched top3 oracle RMSE は `4.420752853` で、exp207 の `4.418699605` を更新しなかった。stitched only top3 は `50.798377042 -> 47.188322489` に改善したが、単体候補としてはまだ粗い。

dense path は物理的に stitch 可能であることは確認できたが、direct replacement、softmax average、PF weight replacement、inference、submit には進めない。後続は dense stitch の追加調整ではなく、exp204 selector candidate route で heatmap MDN 候補を target-free selection に渡す方向を優先する。

## 状態

診断完了。Kaggle output は `kaggle/output/train_v1` に取得済み。CNN training 0、LightGBM 0 configs / 0 boosters、parent/control retraining なし、inference / submit なし。

## 主要指標

| 指標 | exp207 | exp208 |
| --- | ---: | ---: |
| source overlap wells | 3 | 773 |
| source overlap pairs | 39 | 24,679 |
| row coverage vs exp099 cache | 0.352337441 | 0.430091631 |
| stitched only top3 oracle RMSE | 50.798377042 | 47.188322489 |
| existing + stitched top3 oracle RMSE | 4.418699605 | 4.420752853 |
| existing + stitched top3 within10 | 0.959487445 | 0.960702615 |
| `1000_plus` union oracle RMSE | 5.414525 | 5.422416550 |
