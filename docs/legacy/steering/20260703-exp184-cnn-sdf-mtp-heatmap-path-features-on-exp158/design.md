# 設計

## アプローチ

exp183 の selector enrichment 実装パターンを使い、cluster prior feature の代わりに exp182 CNN/SDF/MTP heatmap path feature を差し込む。候補集合、exp099 multiobs features、exp072 dense enrichment、GroupKFold、LightGBM 3 config、exp158 Viterbi grid は固定する。

exp182 validation predictions は各 well の sparse sample 出力なので、`base_real_w128_b64_fullfold`、`base_shuffled_w128_b64_fullfold`、`base_no_gr_w128_b64_fullfold` の valid split だけを読み、well 内 `row_center` で exp158 row-level frame に線形補間する。補間は target-free な predicted TVT path center / score / path-step stats / prior center relation だけで行う。

## 実験範囲

- 対象実験: `exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158`
- Route: `pf_beam`
- 親実験: `exp158_segment_continuity_selector_on_exp157`
- 変更する変数: exp182 heatmap topK path center、score margin、entropy、spread、real-vs-shuffled/no-GR gap、PF/Beam/dense candidates との差分を selector feature に追加。
- 固定する変数: exp157 の 8候補、dense enrichment、exp099 multiobs surface、GroupKFold 5 folds、LightGBM 3 configs、exp158 Viterbi grid。
- 範囲外: heatmap direct TVT replacement、softmax blend、PF weight replacement、full-length inference、submission。

## 再現性設計

- seed policy: GroupKFold seed、LightGBM seed、candidate-long row subsample seed を固定する。
- stochastic 処理の有無: exp184 自体の feature generation は deterministic。upstream exp182 は PyTorch CUDA 学習済み artifact なので deterministic anchor とは扱わない。
- PF/Beam / likelihood-PF / seed bagging の有無: exp072 / exp099 の保存済み PF/Beam / likelihood-PF cache を読む。exp184 では再生成しない。
- 並列処理と乱数の関係: heatmap interpolation に乱数なし。LightGBM training と candidate-long subsample は固定 seed。
- CPU/GPU runtime と deterministic flags: exp184 train は Kaggle CPU。GPU は使わない。
- train cache / test feature regeneration の SHA 記録方針: exp099 / exp072 / exp182 gzip は decompressed content SHA を summary に記録する。exp182 sample index と summary JSON SHA も記録する。
- model manifest / prediction / submission SHA 記録方針: LightGBM model manifest、OOF prediction decompressed SHA、feature schema SHA、heatmap feature schema / summary SHA を記録する。submission は作らない。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks` 後に metadata と bootstrap support files が exp184 config / module / notebook source を含むことを確認する。

## リスク

- リークリスク: exp182 prediction file には true-error columns が含まれるため、読み込み usecols で `*_abs_error`、`within10`、true TVT、target_in_grid を除外する。
- CV/LB 不一致リスク: train-side sparse sample の interpolation feature であり hidden test feature parity は未確認。positive でも inference port / submit へ進めない。
- ランタイム/メモリリスク: heatmap row-level feature と candidate-long feature が増える。candidate-long training rows は exp183 と同じ 650k cap を維持する。
- 再現性リスク: exp182 は GPU diagnostic artifact なので deterministic submission anchor ではない。exp184 も train-side audit として扱う。
