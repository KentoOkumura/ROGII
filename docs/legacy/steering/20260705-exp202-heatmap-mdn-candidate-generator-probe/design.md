# 設計

## アプローチ

exp182 の 5ch heatmap CNN/MTP 実装を土台に、出力を「予測値」ではなく `K=10` 個の候補 TVT/path と mode score として扱う。valid split の各 pseudo-tail sample について heatmap topK を保存し、exp099 の保存済み PF/Beam/likPF/sc/hyb 候補 cache と `id` で join する。

評価は selector を学習せず、まず oracle readout に限定する。既存 union の oracle RMSE / within10 と、heatmap top1/top3/top5/top10 を追加した union の oracle RMSE / within10 / heatmap new-best rate を比較する。positive であってもこの exp では hard replacement も submit もせず、後続を `heatmap_mdn_candidates_into_exp158_selector` または `heatmap_mdn_confidence_features_on_exp148` に分ける。

実装上は exp182 と同じ closest-mode loss を維持する。公開 discussion 699853 の `GeoStirringNet(K=10, L=24)` / path-logit head の発想を参照するが、初手では既存 exp182 の `K=10` compact CNN に寄せ、仕様差分を candidate-union 保存と readout に絞る。

2026-07-06 追記: plot 用に、各 validation sample の deduplicate 済み center-row top10 candidate に対応する local 128-row path を保存する。保存形式は `*_heatmap_candidate_paths_top10.npz` と sample/rank index CSV。ここで保存する path は exp202 の local window path であり、overlapping windows を stitch した full-well trajectory ではない。stitch 案は別 backlog `heatmap_mdn_overlapping_window_path_stitch_probe` に切り出す。

## 実験範囲

- 対象実験: `exp202_heatmap_mdn_candidate_generator_probe`
- Route: `pf_beam`
- 親実験: `exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe`、`exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158`、`exp099_pf_multi_observation_likelihood_probe`
- 変更する変数: heatmap topK TVT/path を candidate set として保存し、既存 PF/Beam candidate union へ追加した oracle headroom を評価する。加えて、plot 用に validation local 128-row path を npz + index CSV で保存する。
- 固定する変数: 5ch heatmap input、target-free flat-prior typewell TVT window center、GroupKFold by well、K=10 path head、closest-mode CE、T4 GPU、no inference/submission。
- GPU 学習予定: 1 active run spec x 5 folds = 5 CNN models。LightGBM 0 boosters。parent/control 再学習なし。

## 再現性設計

- seed policy: global seed 42 と run spec / fold / well keyed SHA256。sample order と DataLoader shuffle は immutable key 由来にする。
- stochastic 処理の有無: PyTorch CUDA convolution、AdamW、DataLoader shuffle が stochastic。deterministic submission anchor とは扱わない。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規 PF/Beam 生成はしない。exp099 の保存済み候補 cache を読む。
- 並列処理と乱数の関係: DataLoader `num_workers=0`。thread scheduling による RNG 消費差を避ける。
- CPU/GPU runtime と deterministic flags: Kaggle T4 GPU 必須。`torch.use_deterministic_algorithms(True, warn_only=True)`、CuDNN benchmark false、deterministic true。
- train cache / test feature regeneration の SHA 記録方針: sample index、validation predictions、heatmap candidate CSV、candidate local path npz/index CSV、candidate union metrics、feature schema、run spec manifest、exp099 candidate cache の SHA を保存する。gzip は decompressed content SHA を主証拠にする。
- model manifest / prediction / submission SHA 記録方針: run spec x fold ごとの model SHA と validation prediction SHA を記録する。submission は作らない。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks` 後、metadata の `enable_gpu=true`、`machine_shape=NvidiaTeslaT4`、kernel sources、bootstrap 内 config の active run spec と candidate cache path を確認する。

## リスク

- リークリスク: valid true TVT を heatmap input、window center、normalization、candidate grid、feature source、inference-time selection に混ぜると過大評価になる。true TVT は label / metric / oracle readout に限定する。
- CV/LB 不一致リスク: train-side pseudo-tail diagnostic であり LB は測らない。positive でも raw-test heatmap generation、sparse interpolation coverage、schema parity、fallback behavior が未確認。
- ランタイム/メモリリスク: CNN 5 models と exp099 3.8M row candidate cache join が入る。candidate cache は必要列だけを読む。
- 再現性リスク: GPU 学習は bitwise stable ではない。seed、model SHA、sample index SHA、prediction SHA、candidate union SHA、Kaggle version を証拠として保存する。
