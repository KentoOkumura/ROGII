# 要件

## 依頼

`cnn_sdf_mtp_heatmap_probe` backlog を実装する。discussion 699853 の 5ch heatmap CNN/SDF/MTP アイデアを、まず 1 fold / small wells / fixed target-free window の GPU train-side probe として再現し、real GR が shuffled-GR / no-GR control を上回るか確認できる状態にする。

## 制約

- Route: `ml_model`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- Kaggle GPU を使用する。CUDA がない場合は train notebook を失敗させ、CPU fallback はしない。
- valid/test true TVT を heatmap 入力、normalization、typewell window center、候補生成、inference feature に使わない。
- window random split は不可。well GroupKFold で fold-safe に分割する。
- `mtp_heatmap_sdf_mdn_probe` umbrella 名では新規実験を切らず、具体 backlog 名 `cnn_sdf_mtp_heatmap_probe` として実装する。
- 提出、PF weight replacement、hard path replacement、softmax blend、inference port は範囲外。

## 受け入れ基準

- `experiments/exp179_cnn_sdf_mtp_heatmap_probe/` に config、Jupytext train source、train ipynb、README、SESSION_NOTES、result、metrics が揃う。
- train notebook は 5ch image (`t_gr`, `h_gr`, `t_gr-h_gr`, observed-TV T history SDF, mask) と K-path head を持つ PyTorch CNN を実装する。
- `real_gr`、`shuffled_gr`、`no_gr` の 3 variants を同じ sample schedule / fold / epochs で実行できる。
- typewell window center は target-free な `last_known_tvt - (Z - last_known_z)` prior で作る。
- metrics は target-in-grid rate、top1/top3/top5/top10 within10 center coverage、oracle topK RMSE、training history を保存する。
- GPU 学習コストとして active variant 数、fold 数、モデル数、control 再学習有無を `SESSION_NOTES.md` に記録する。
- deterministic anchor としては扱わない。GPU stochastic component と SHA 記録方針を `config.yaml` / `SESSION_NOTES.md` に明記する。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
