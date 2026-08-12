# 設計

## アプローチ

exp202 の fold-safe 5ch heatmap input builder を土台にし、model head を grid-cell classifier から continuous MTP に置き換える。入力は `typewell GR`、`horizontal GR`、GR 差分、observed `TVT_input` prefix history SDF、observed mask の 5ch を維持する。ただし SDF は入力表現だけで、SDF output head や `sdf_loss` は使わない。

モデルは `path_pred [B,K,L]` と `path_logit [B,K]` を返す。loss は各 mode の true TVT path regression error から `best_k` を選び、best path regression loss と `cross_entropy(path_logit, best_k)` を足す。valid fold では `softmax(path_logit)` から topK rank path と probability weighted path を作り、dense full-tail window を row grid に aggregation する。

exp212 は source coverage 0.430091631、fallback unique row rate 0.569908369 で、後半が endpoint hold 直線 tail になった。exp215 は `max_tail_rows: null`、`row_center_stride: 64`、`include_tail_stop: true` で valid wells の tail 全体に source windows を作り、source coverage と fallback rate を success gate にする。

## 実験範囲

- 対象実験: `exp215_mtp_full_tail_heatmap_path_generator_probe`
- Route: `pf_beam`
- 親実験: `exp202_heatmap_mdn_candidate_generator_probe`、`exp208_heatmap_mdn_dense_stride_window_path_regeneration_probe`、`exp212_heatmap_mdn_full_grid_path_generation_probe`
- 比較対象: `exp099_pf_multi_observation_likelihood_probe` candidate union、exp212 stitched-only top5 RMSE 50.085237573
- 変更する変数: heatmap MTP head を continuous `path_pred/path_logit` full-tail generator に変更し、full-grid learned path artifact を作る。
- 固定する変数: 5ch base heatmap input、GroupKFold by well 5 folds、K=10、L=128、no inference/submission、no LightGBM。

## 再現性設計

- seed policy: global seed 42。DataLoader shuffle は experiment / run spec / fold 由来の stable seed を使う。
- stochastic 処理の有無: PyTorch CUDA convolution、AdamW、DataLoader shuffle が stochastic。bitwise deterministic anchor とは扱わない。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規 PF/Beam 生成なし。exp099 saved candidate cache を比較 readout に読む。
- 並列処理と乱数の関係: DataLoader `num_workers=0`。thread scheduling による RNG 消費差を避ける。
- CPU/GPU runtime と deterministic flags: Kaggle T4 GPU 必須。`torch.use_deterministic_algorithms(True, warn_only=True)`、CuDNN benchmark false、deterministic true。
- train cache / test feature regeneration の SHA 記録方針: sample index、dense validation predictions、window path npz/index、full-grid candidate paths、candidate union metrics、feature schema、run spec manifest、exp099 cache の SHA を記録する。gzip は decompressed content SHA を主証拠にする。
- model manifest / prediction / submission SHA 記録方針: fold model SHA、model manifest SHA、prediction / full-grid artifact SHA を記録する。submission は作らない。
- Kaggle package bootstrap 確認方針: push 前に metadata の `enable_gpu=true`、`machine_shape=NvidiaTeslaT4`、internet false、kernel source、bootstrap 内 config を確認する。

## リスク

- リークリスク: true TVT 由来の oracle best、abs-error、within10、true-error rank を aggregation / candidate score / downstream feature に入れると過大評価になる。true TVT は loss と train-side readout に限定する。
- CV/LB 不一致リスク: train-side pseudo-tail diagnostic であり LB は測らない。positive でも raw-test parity、hidden-like stress、selector 適用前の guard が必要。
- ランタイム/メモリリスク: 5 CNN models と full-grid top5 artifact が大きい。full-grid output は exp212 と同規模の 18.9M rows 想定。
- 再現性リスク: GPU 学習は bitwise stable ではない。seed、model SHA、sample index SHA、prediction SHA、full-grid artifact SHA、Kaggle version を証拠として保存する。
