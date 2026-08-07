# 設計

## アプローチ

exp202 v2 が保存した validation sample ごとの local 128-row top10 path を読み、well 内で row_center 順に並べる。各 window では topK rank の segment を候補にし、beam search で rank choice の系列を選ぶ。score は exp202 の mode/center score、rank penalty、local path smoothness、隣接 window の overlap disagreement、gap boundary continuity だけで計算する。

できた stitched path は row-level candidate として保存し、最後に exp099 PF/Beam candidate cache と id join して oracle readout を出す。これは selector ではなく、candidate path が物理的に破綻しないか、既存 union に追加 headroom があるか、現在の exp202 sparse artifact で full-well stitch と呼べる coverage があるかを見る診断である。

## 実験範囲

- 対象実験: `exp207_heatmap_mdn_overlapping_window_path_stitch_probe`
- Route: `pf_beam`
- 親実験: `exp202_heatmap_mdn_candidate_generator_probe`
- 比較対象: `exp099_pf_multi_observation_likelihood_probe`
- 変更する変数: local heatmap topK path の well-level stitch 方法、overlap / gap continuity score、stitched path candidate oracle readout。
- 固定する変数: exp202 CNN weights / local path artifact、exp099 PF/Beam candidate cache、既存 candidate set、validation target。
- 実行コスト: CPU diagnostic。GPU 0、LightGBM 0 configs / 0 boosters、parent/control retraining なし。

## 再現性設計

- seed policy: exp207 自体は乱数を使わず、artifact order と deterministic sort のみ。config seed は project default 42 を記録する。
- stochastic 処理の有無: exp207 stitching にはなし。upstream exp202 は PyTorch GPU 学習、upstream exp099 は PF/Beam candidate cache 生成を含む。
- PF/Beam / likelihood-PF / seed bagging の有無: exp207 は既存 PF/Beam cache を読むだけで再生成しない。
- 並列処理と乱数の関係: exp207 は single-process pandas/numpy 処理。global RNG / thread scheduling に依存しない。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、GPU disabled。
- train cache / test feature regeneration の SHA 記録方針: exp202 path npz SHA、path sample gzip raw/decompressed SHA、exp099 cache gzip raw/decompressed SHA、exp207 output gzip raw/decompressed SHA を summary / metrics に記録する。
- model manifest / prediction / submission SHA 記録方針: exp207 は model なし、submission なし。stitched row-level path SHA を prediction-like output として記録する。
- Kaggle package bootstrap 確認方針: push する場合は `prepare-kaggle-notebooks --strict` 後、kernel source が exp202 / exp099、GPU disabled、config の topK/beam が正しいことを確認する。

## リスク

- リークリスク: exp202 path npz には `true_tvt_path` が含まれるが、helper は stitch score 用に読み込まない。sample CSV も `true_center_tvt` を usecols から外す。
- CV/LB 不一致リスク: train-side oracle diagnostic のみ。raw-test dense window generation、schema parity、fallback behavior は未検証なので submit 判断に使わない。
- ランタイム/メモリリスク: row-level stitched path は `output_topn=3` で保存する。dense window artifact に拡張すると出力が大きくなるため、初回は exp202 sparse artifact の coverage を明示する。
- 再現性リスク: exp207 は deterministic だが upstream exp202 は GPU trained artifact なので deterministic submission anchor ではない。
