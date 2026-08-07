# exp202_heatmap_mdn_candidate_generator_probe 結果

## 状態

Kaggle train v1 / v2 は COMPLETE。kernel は `kentookumura/exp202-heatmap-mdn-candgen-train`、T4、5 folds、5 CNN models。inference / submit は実施していない。

v1 output は `experiments/exp202_heatmap_mdn_candidate_generator_probe/kaggle/output/train_v1` に取得済み。v2 output は local path plotting artifact の materialization 目的で再実行し、`experiments/exp202_heatmap_mdn_candidate_generator_probe/kaggle/output/train_v2` に取得済み。

## 主結果

- heatmap only top10: within10 `0.808907780`、oracle RMSE `13.352563025`。既存 candidate union より単独では粗い。
- existing union: 5 candidates、within10 `0.949639623`、oracle RMSE `5.068679053`。
- existing + heatmap top10: 15 candidates、within10 `0.986970985`、oracle RMSE `2.745528140`。
- top10 追加による改善: oracle RMSE `-2.323150913`、within10 `+0.037331362`、new-best candidate rate `0.252541120`。
- fold 平均 heatmap coverage: top3 within10 `0.496516249`、top10 within10 `0.808984978`、top10 oracle RMSE `13.293297981`。

## bucket / well readout

距離 bucket では、near `0_50` と `50_100` は変化なし。`1000_plus` は existing oracle RMSE `6.413572416` から `3.295946470` へ `-3.117625946` 改善し、new-best candidate rate は `0.317448544`。

well 単位では 773 wells 中 668 wells が改善、105 wells が同値、悪化 0 wells。平均 oracle RMSE delta は `-1.281989629`。残存 worst は `b0d42b0d` の heatmap-union oracle RMSE `17.351555`。

## 判断

heatmap 候補は「単独予測」ではなく、既存 PF/Beam 候補集合の補助 candidate として有望。特に longtail / far bucket の oracle headroom が大きい。

ただしこれは oracle union 診断であり、selector が target-free に正しい候補を選べることはまだ示していない。direct TVT replacement、softmax weighted average、PF weight replacement、postprocess blend、submit には進めない。

## v2 local path plotting output

2026-07-06 の v2 rerun で、validation sample ごとの deduplicate 済み center-row top10 candidate に対応する local 128-row path を保存した。これは full well trajectory ではなく、exp202 が学習・評価している local window path。

- `exp202_heatmap_mdn_candidate_generator_probe_heatmap_candidate_paths_top10.npz`
- `exp202_heatmap_mdn_candidate_generator_probe_heatmap_candidate_path_samples.csv.gz`
- `exp202_heatmap_mdn_candidate_generator_probe_heatmap_candidate_path_rank_index.csv.gz`

shape:

- samples: `10822`
- paths: `108220`
- topK: `10`
- horizon: `128`
- `pred_tvt_path`: `(10822, 10, 128)`
- `pred_bin_path`: `(10822, 10, 128)`
- `true_tvt_path` / `tvt_input_path` / `md_path` / `z_path` / `horizontal_row_index`: `(10822, 128)`

主 SHA:

- path npz: `e615f0d01a08fd37685fd1ac46335b99306f0bb0c9c43d37c1e1f620040839a3`
- sample index decompressed: `cea6a29c716a1c5dedda1efec64a1b1f2371d1eadfd298084576f06170d0a7de`
- rank index decompressed: `a21b7c9c056272c20d3c1b60f6b7224e78edeb156cb1af19b6a950bd1278f349`

## 次アクション

`heatmap_mdn_candidates_into_selector_or_ml_features` として、heatmap topK candidates / mode score / entropy / margin を exp158/exp176 系 continuity selector、または exp148 系 ML confidence feature に渡す小実験に切る。candidate 値の直接置換ではなく、candidate score / distance / uncertainty / selector confidence として扱う。
