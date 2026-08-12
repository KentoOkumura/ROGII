# 設計

## アプローチ

exp210 の stitch / contract / oracle readout を親にし、covered rows 限定だった path table を full-grid 化する。exp208 dense local paths を target-free beam stitch した後、sparse stitched rows を exp099 feature-cache row grid に reindex する。source-covered rows は stitched prediction をそのまま使い、未カバー rows は row_index 線形補間または端点外挿で埋める。

## 実験範囲

- 対象実験: `exp212_heatmap_mdn_full_grid_path_generation_probe`
- Route: `pf_beam`
- 親実験: `exp202_heatmap_mdn_candidate_generator_probe`
- dense source: `exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe`
- 比較: `exp099_pf_multi_observation_likelihood_probe`
- 変更する変数: full-grid row expansion、coverage/fallback contract、full-grid oracle readout。
- 固定する変数: exp208 cached local paths、exp207/208/210 stitch score、topK values、existing candidate union。

## 再現性設計

- seed policy: exp212 内では乱数を使わない。入力 artifact order と deterministic pandas/numpy 処理に依存する。
- stochastic 処理の有無: なし。ただし upstream exp202/exp208/exp099 は stochastic components として記録する。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規 PF sampling なし。beam stitch は deterministic。
- 並列処理と乱数の関係: 並列乱数なし。
- CPU/GPU runtime: Kaggle CPU、GPU disabled、internet disabled。
- train cache / test feature regeneration の SHA 記録方針: exp208 path npz、exp208 samples gzip decompressed SHA、exp099 cache gzip decompressed SHA、full-grid output gzip decompressed SHA を記録する。
- model manifest / prediction / submission SHA 記録方針: 新規 model なし、submission なし。prediction artifact として full-grid candidate paths の decompressed SHA を主証拠にする。
- Kaggle package bootstrap 確認方針: `make prepare-kaggle-notebooks --strict` で train package metadata と kernel sources を確認する。

## リスク

- リークリスク: full-grid fill へ target-derived columns が混入すると selector 候補として使えない。入力 sample columns から true/abs-error 系を除外し、oracle は paths 固定後にのみ実施する。
- CV/LB 不一致リスク: train-side oracle headroom は selector の実性能や LB を保証しない。直接置換、平均、PF weight replacement、submit はしない。
- ランタイム/メモリリスク: full-grid top5 output は exp099 rows x 5 で大きい。Kaggle CPU で gzip 出力し、debug は wells を制限する。
- 再現性リスク: upstream GPU-trained exp202 weights と exp099 PF/Beam cache に依存するため deterministic submission anchor ではない。
