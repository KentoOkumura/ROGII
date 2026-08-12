# 要件

## 依頼

`cnn_sdf_mtp_heatmap_fullfold_geometry_probe` を実装する。exp179 で positive だった discussion 699853 の 5ch heatmap CNN/SDF/MTP 診断を、full-fold 確認、geometry channel ablation、larger-window probe まで拡張し、real GR が shuffled-GR / no-GR control を fold 横断でも上回るかを確認できる状態にする。

## 制約

- Route: `ml_model`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 親実験は `exp179_cnn_sdf_mtp_heatmap_probe`。direct TVT replacement、PF weight replacement、softmax weighted average、inference port、submission は行わない。
- valid/test true TVT を heatmap 入力、normalization、typewell window center、candidate grid、inference feature に使わない。true TVT は train pseudo-tail label と metric のみに使う。
- Kaggle GPU は T4 を明示する。P100 は Kaggle PyTorch 2.10 と非互換だったため避ける。
- Kaggle train push 前に active run specs、fold 数、CNN model 数、LightGBM booster 数を `SESSION_NOTES.md` に記録する。

## 受け入れ基準

- `experiments/exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe/` に config、Jupytext train source、train ipynb、inference guard、README、SESSION_NOTES、result、metrics が揃う。
- train notebook は run spec ごとに fold/window/channel/variant を明示し、active specs の CNN model 数を表示できる。
- train notebook は base 5ch channel と geometry-extended channel を切り替えられる。
- metrics は run spec、fold、variant、channel set、window、bins 別に top1/top3/top5/top10 coverage、oracle topK RMSE、distance bucket、worst-well、path continuity を保存する。
- 実行結果を deterministic anchor として扱わず、GPU train-side diagnostic として記録する。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
