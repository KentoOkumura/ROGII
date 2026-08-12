# 設計

## アプローチ

exp179 の 5ch heatmap CNN/SDF/MTP 実装を run spec 化する。各 run spec は `variant` (`real_gr` / `shuffled_gr` / `no_gr`)、fold list、horizontal window、typewell bins、history scale、channel set を持つ。初期 active specs は 5-fold base controls、5-fold geometry real、fold subset の larger-window real を含め、提出モデルではなく train-side diagnostic として比較する。

geometry channel は horizontal well の `MD,X,Y,Z` だけから作る。具体的には `sin/cos(dMD/dZ)`、`sin/cos(dX/dY)`、prefix anchor からの正規化距離、row location prior を heatmap に broadcast する。いずれも observed `TVT_input` prefix と raw geometry だけを使い、valid true TVT には依存しない。

## 実験範囲

- 対象実験: `exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe`
- Route: `ml_model`
- 親実験: `exp179_cnn_sdf_mtp_heatmap_probe`
- 変更する変数: fold coverage、horizontal window、typewell bins、channel set、GR control variant
- 固定する変数: target-free flat-prior window center、closest-mode CE、K=10 path head、seed 42、T4 GPU、no inference/submission

## 再現性設計

- seed policy: global seed 42 と run spec / fold / well keyed SHA256。sample order、shuffled-GR roll、DataLoader shuffle は immutable key から seed を作る。
- stochastic 処理の有無: PyTorch CUDA convolution、AdamW、DataLoader shuffle が stochastic。deterministic anchor とは扱わない。
- PF/Beam / likelihood-PF / seed bagging の有無: なし。PF/Beam candidate 生成や weight replacement は範囲外。
- 並列処理と乱数の関係: DataLoader `num_workers=0`。thread scheduling による RNG 消費差を避ける。
- CPU/GPU runtime と deterministic flags: Kaggle T4 GPU 必須。`torch.use_deterministic_algorithms(True, warn_only=True)`、CuDNN benchmark false、deterministic true。
- train cache / test feature regeneration の SHA 記録方針: sample index、validation predictions、metrics、training history、feature schema、run spec manifest の SHA を保存する。gzip は decompressed content SHA を主証拠にする。
- model manifest / prediction / submission SHA 記録方針: run spec x fold ごとの model SHA と validation prediction SHA を記録する。submission は作らない。
- Kaggle package bootstrap 確認方針: `prepare-kaggle-notebooks` 後、metadata の `enable_gpu=true`、`machine_shape=NvidiaTeslaT4`、bootstrap 内 config の active run specs を確認する。

## リスク

- リークリスク: valid true TVT を window center、normalization、candidate grid、input channel に混ぜると過大評価になる。実装では true TVT を label/metric に限定する。
- CV/LB 不一致リスク: train-side diagnostic であり LB は測らない。positive でも path feature / candidate verifier 材料に限定する。
- ランタイム/メモリリスク: active specs の CNN model 数が増える。push 前に model 数を数え、必要なら fold subset に絞る。
- 再現性リスク: GPU 学習は bitwise stable ではない。SHA は evidence として保存するが deterministic submission anchor にはしない。
