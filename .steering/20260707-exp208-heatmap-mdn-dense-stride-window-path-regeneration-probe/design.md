# 設計

## アプローチ

exp202 の train v2 output にある fold 別 PyTorch model (`candidate_real_w128_b64_fullfold_fold{0..4}_model.pt`) を読み、同じ 5ch heatmap input builder で validation wells の dense window を推論する。row center は `prefix_end + 1` から `max_tail_rows=2048` まで stride 64 で作り、tail stop も追加して末尾 coverage を確保する。

生成した dense local path artifact は exp202 v2 の path artifact と同じ conceptual contract に合わせ、`path_npz_sample_index`、`well`、`row_center`、center score、topK path を持つ。以後は exp207 の target-free stitch scoreを再利用し、local topK 5 / 10 で readout する。

## 実験範囲

- 対象実験: `exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe`
- Route: `pf_beam`
- 親実験: `exp202_heatmap_mdn_candidate_generator_probe`
- 比較対象: `exp207_heatmap_mdn_overlapping_window_path_stitch_probe`、`exp099_pf_multi_observation_likelihood_probe`
- 変更する変数: validation window の row_center 密度、local topK stitch readout。
- 固定する変数: exp202 model architecture / fold split / weights、exp207 stitch score weights、exp099 candidate cache、submit なし。

## 再現性設計

- seed policy: 乱数 sampling は使わない。fold split は GroupKFold + sorted well id、exp202 と同じ seed 42 を記録する。
- stochastic 処理の有無: exp208 内には stochastic sampling / training はない。upstream exp202 は GPU training artifact なので deterministic submission anchor ではない。
- PF/Beam / likelihood-PF / seed bagging の有無: exp208 では新規 PF/Beam 生成なし。exp099 saved candidate cache に依存する。
- 並列処理と乱数の関係: DataLoader `num_workers=0`、shuffle なし。parallel RNG は使わない。
- CPU/GPU runtime と deterministic flags: Kaggle CPU runtime、GPU disabled。PyTorch saved model inference は CPU で実行する。
- train cache / test feature regeneration の SHA 記録方針: raw train data は個別 file SHA までは記録せず、exp202 model artifact、model manifest、exp099 candidate cache、dense path artifact、stitch outputs の SHA を記録する。
- model manifest / prediction / submission SHA 記録方針: exp202 model file SHA と model manifest SHA、dense predictions/path artifact SHA、stitch readout SHA を記録する。submission は作らない。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks --strict` 後に metadata と bootstrap config を検証する。

## リスク

- リークリスク: true TVT は supervised target と post-hoc oracle readout にのみ使う。stitch score には予測 score、rank、predicted path smoothness、overlap disagreement、gap continuity だけを使う。
- CV/LB 不一致リスク: train-side oracle 診断であり、selector や hidden inference の性能は示さない。positive でも submit に進めない。
- ランタイム/メモリリスク: stride 64 でも dense path rows と stitch output が大きくなる。debug mode は fold ごとの well 数を絞れるようにする。
- 再現性リスク: exp202 upstream GPU weights に依存するため deterministic anchor ではない。入力 model/cache/output SHA を記録して追跡可能にする。
