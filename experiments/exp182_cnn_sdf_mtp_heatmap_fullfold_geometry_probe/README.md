# exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe

## 状態

- ルート: ml_model
- 状態: completed_train_side_gpu_probe
- CV: top3 within10 0.500000 (`base_real_w128_b64_fullfold`)
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-03
- 親実験: `exp179_cnn_sdf_mtp_heatmap_probe`

## 仮説

exp179 の 1 fold smoke では、5ch heatmap CNN/SDF/MTP の `real_gr` が shuffled-GR / no-GR control を明確に上回った。一方で full-fold、geometry channel、larger window、worst-well、distance bucket の安定性は未確認である。exp182 では同じ target-free window と closest-mode CNN/MTP objective を維持し、fold 横断で GR signal が残るかを診断する。

## 変更点

- exp179 の単一 fold loop を `active_run_specs` に分解する。
- `base_real` / `base_shuffled` / `base_no_gr` を 5 folds で比較する。
- `geometry_real` を 5 folds で追加し、`sin/cos dMD/dZ`、`sin/cos dX/dY`、prefix distance、row location prior を加える。
- `geometry_shuffled` と `geometry_real_w256_b96` は fold 0/1 の小さい追試として実行できる。
- run spec、fold、window、channel set 別に metrics、fold metrics、well metrics、distance bucket metrics、model SHA を保存する。
- inference port と submission は作らない。

## 検証方針

- Fold: GroupKFold 5 folds
- Group: well id
- Metric: top3 within10 center coverage
- Secondary: top1/top5/top10 coverage、oracle topK RMSE、path continuity、worst-well、distance bucket
- Leakage Check: valid true TVT は label / metric のみに使う。input heatmap、normalization、typewell window center、sample schedule、negative controls、geometry channel には使わない。

## 実行入口

- 学習 notebook: `exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe_train.ipynb`
- 推論 notebook: `exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe`
- notebook 実行: Kaggle T4 GPU run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## GPU コスト

初期 active specs は 24 CNN models である。

- base controls: 3 specs x 5 folds = 15 models
- geometry real: 1 spec x 5 folds = 5 models
- geometry shuffled fold subset: 1 spec x 2 folds = 2 models
- larger-window geometry fold subset: 1 spec x 2 folds = 2 models

合計 24 models。LightGBM booster は 0。既存 baseline/control の再学習はない。Kaggle train v1 で実行済み。

## 結果

Kaggle train v1 は `kentookumura/exp182-cnn-sdf-mtp-geometry-probe-train` で完了した。T4 GPU / PyTorch 2.10.0+cu128、24 CNN models、773 usable wells、train samples 61,840 / valid samples 10,822 の full-fold control を含む。

| run spec | top3 within10 | top10 within10 | top10 oracle RMSE | 判定 |
| --- | ---: | ---: | ---: | --- |
| `base_real_w128_b64_fullfold` | 0.500000 | 0.808908 | 13.296284 | GR signal supported |
| `base_shuffled_w128_b64_fullfold` | 0.218536 | 0.545001 | 17.637821 | negative control |
| `base_no_gr_w128_b64_fullfold` | 0.071429 | 0.071429 | 134.767278 | negative control |
| `geometry_real_w128_b64_fullfold` | 0.487710 | 0.809647 | 11.995428 | top3 は base より悪化 |
| `geometry_shuffled_w128_b64_fold01` | 0.206682 | 0.549539 | 17.013071 | negative control |
| `geometry_real_w256_b96_fold01` | 0.417512 | 0.716129 | 18.643568 | 拡大 window は悪化 |

`base_real` は shuffled-GR に top3 +0.281464、no-GR に +0.428571 と明確に勝った。GR signal は full-fold でも支持される。一方で geometry 追加は top3 -0.012290、larger window fold01 も 0.417512 に落ちたため、現設定では採用しない。worst-well top3 は `base_real` / `geometry_real` とも 0.0 の well が残るため、full-length inference / direct TVT replacement / submit はしない。

## 所見

### 良かった点

- exp179 の positive smoke を壊さず、fold / window / channel set の比較単位を run spec として明示した。
- geometry channel は `MD,X,Y,Z` と observed prefix だけから作るため、hidden test でも再生成可能な入力に限定している。
- full-fold `base_real` が shuffled-GR / no-GR を大きく上回り、discussion 699853 系 heatmap CNN が GR signal を拾うことを確認できた。

### 悪かった点

- geometry channel は top10 oracle RMSE を少し改善したが、primary の top3 coverage は `base_real` より悪化した。
- `geometry_real_w256_b96_fold01` は fold subset でも弱く、larger window を広げる根拠にならなかった。
- worst-well top3 0.0 が残り、submission への直接的な改善証拠はない。

### リスク / 注意

- full-fold real GR margin は positive だが、worst-well guard は通っていない。
- heatmap 出力は direct TVT replacement ではなく、path feature / candidate verifier / confidence feature の材料に限定する。

## リスク / 注意

- GPU training は bitwise deterministic anchor として扱わない。
- positive でも direct TVT replacement、softmax average、PF weight replacement、submission には進めない。
- full-length inference port は full-fold margin、worst-well guard、distance-bucket guard を通過してから別途検討する。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
