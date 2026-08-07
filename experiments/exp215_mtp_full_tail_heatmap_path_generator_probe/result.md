# exp215_mtp_full_tail_heatmap_path_generator_probe 結果

## 状態

Kaggle train v1 完了。train-side diagnostic のため inference / submit は行わない。

- kernel: `kentookumura/exp215-mtp-full-tail-heatmap-path-generator-train`
- version: 1
- GPU: `NvidiaTeslaT4`
- status: `KernelWorkerStatus.COMPLETE`
- source: Kaggle logs

## 実装内容

- exp202 の 5ch heatmap input を継承。
- `path_pred [B,K,L]` と `path_logit [B,K]` を出す continuous MTP head を実装。
- closest-mode path regression + mode CE loss を実装。
- valid fold の dense full-tail windows から full-grid candidate path artifact を生成。
- exp099 candidate cache と join して existing union、learned MTP topK、weighted path、existing+learned topK の oracle readout を保存。

## 主要結果

full-grid contract は成立した。

- rows: 18,919,945
- unique row ids: 3,783,989
- wells: 773
- path ranks: 5
- row coverage vs exp099 cache: 1.000000000
- fallback unique row rate: 0.000000000
- duplicate key rows: 0
- null required values: 0

candidate union readout:

- existing union oracle RMSE: 7.434029932 / within10 0.906525363
- learned MTP top5 only oracle RMSE: 32.333142886 / within10 0.477650966
- learned MTP weighted oracle RMSE: 59.272141581 / within10 0.153523702
- existing + learned MTP top5 oracle RMSE: 5.113654814 / within10 0.945863743
- existing + learned MTP top5 delta vs existing: -2.320375117 RMSE

MTP window aggregate:

- folds completed: 5
- train samples: 86,576
- valid samples: 7,730
- dense samples: 60,266
- dense top10 oracle center RMSE: 15.741723880
- dense weighted center RMSE: 60.093226732
- dense rank1 center RMSE: 64.792147625

## 解釈

exp212 の主問題だった source coverage 0.430091631 / fallback unique row rate 0.569908369 は、exp215 で coverage 1.0 / fallback 0.0 まで解消した。endpoint hold の直線 tail を避ける full-grid artifact contract は成立した。

一方、learned MTP 単体の weighted path は RMSE 59.272142 と弱く、direct TVT replacement、softmax weighted TVT、PF weight replacement、inference / submit へ進める根拠はない。learned top5 oracle は exp212 stitched-only top5 RMSE 50.085238 より良いが、単体候補としてはまだ粗い。

既存 PF/Beam candidate union に learned MTP top5 を追加した oracle headroom は positive で、7.434030 -> 5.113655 と -2.320375 改善した。したがって次に使うなら、MTP path を直接平均・置換せず、selector の selectable candidate / confidence feature として guarded に扱う。

## SDF の扱い

SDF は入力 channel の `TVT_input` history representation としてのみ使った。hengck23 dataset の `run_train_sdf.py` のような SDF output branch / `sdf_loss` は使っていない。

## hengck23 CNN MTP example との比較

参照 notebook: `hengck23/cnn-mtp-example` (`scriptVersionId=320093395`)。Kaggle CLI で取得した notebook は、`GeoStirringNet(K=10, L=24)` が `path [B,K,L]` と `logit [B,K]` を出し、各 mode の trajectory MSE から `best_k=argmin(error)` を選んで best path regression loss + `cross_entropy(logit,best_k)` を足す最小 MTP 例だった。

exp215 が一致している点:

- topK continuous path head と learned mode logit を使う。
- `softmax(path_logit)` で path probability を作る。
- closest-mode path loss と mode classification loss を使う。
- SDF output head / SDF target / `sdf_loss` は使わない。

exp215 が意図的に変えた点:

- 参照 notebook は 2ch (`heatmap`, `history`) / crop 64x24 / `L=24` の checkpoint visualization 例。exp215 は exp202 系の 5ch heatmap input / `K=10,L=128` / 5-fold train-side diagnostic。
- 参照 notebook の `history` は prefix path を線で描いた 64x24 binary image。exp215 の history channel は observed `TVT_input` prefix から作る連続 SDF-like representation。
- 参照 notebook の plot は y 軸が typewell bin index 0..63、x 軸が compressed horizontal segment 0..23。exp215 の primary output は TVT feet の連続 path で、後段 artifact では nearest grid bin に変換している。したがって exp215 の出力をそのまま描いても参照 plot と同じ見た目にはならない。
- 参照 notebook は flatten CNN -> MLP -> `Linear(K*L)` で全 path 点を同時に出す。exp215 は 2D conv feature から `Conv2d(K,1x1)` 相当の path map を作り、typewell 軸を平均して `path_pred [B,K,128]` にするため、head 構造も完全一致ではない。
- exp215 は raw full training fold、dense full-tail generation、full-grid artifact contract、candidate union oracle readout まで追加している。

plot-parity 再確認:

- exp215 train notebook には `matplotlib`/`imshow`/`plot` による hengck23 型の可視化セルはない。Kaggle output から `__results__.html` / `__notebook__.ipynb` のみの取得も試したが、CLI output として取得できたのは log のみだった。
- 代わりに `window_path_samples` と `window_path_rank_index` artifact を取得して center path の順位を確認した。60,266 dense samples で rank1 center RMSE は 65.567 ft、top5 best center RMSE は 31.761 ft、top5 within10 は 0.472。candidate 集合には近い path が入るケースがある一方、rank1/logit は当たりを十分選べていない。
- 例: `000d7d20_1442` は true center TVT 11747.380 ft に対し、rank1 11844.528 ft (97.148 ft error)、rank2 11746.194 ft (1.186 ft error)。この挙動では、参照 notebook の赤い path 群や orange の probability weighted average のように真値へきれいに寄る図にはならない。

検証判断:

MTP head / path_logit / closest-mode loss という中核は参照 notebook に沿っている。一方、参照 notebook の checkpoint、crop、plot coordinate、head 構造をそのまま再現したものではない。exp215 は ROGII 実験目的に合わせた full-tail/full-grid 拡張として Kaggle 実行と artifact contract は正しく完了しているが、hengck23 notebook の plot と同じような結果になったとは判断しない。

## 判断

train-side supported。exp204 系の selector-candidate 実験は再開候補。ただし direct replacement / weighted path submit は不採用。
