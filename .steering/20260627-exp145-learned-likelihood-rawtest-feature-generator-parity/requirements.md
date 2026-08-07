# 要件

## 依頼

`learned_likelihood_rawtest_feature_generator_parity` を実装する。

exp144 で exp127 learned likelihood add-only features は hidden-like stress でも改善したが、exp112 feature cache は 155/773 wells subset に限られ、raw-test feature regeneration が未実装だった。exp127 feature family を submission 候補や segment verifier に進める前に、exp112 learned likelihood features を full train / raw test で target-free に再生成できる generator と schema parity audit を作る。

## 制約

- Route: `ml_model`
- 新規 LightGBM 学習はしない。exp111 の保存済み classifier / expected-error regressor を target-free transform として再利用する。
- true TVT、oracle candidate、absolute error label、true-error rank を feature source に入れない。
- raw test では exp072 系 raw replay の PF/Beam/likelihood-PF target-free features だけを使う。
- exp112 train OOF `ml_features` の schema を parity reference とする。
- この実験では `submission.csv` を作らない。
- 再現性: `docs/06_reproducibility.md` に従い、PF/Beam replay、保存済み model SHA、decompressed feature SHA、schema SHA を記録する。

## 受け入れ基準

- exp145 実験フォルダに generator module、train/inference notebook、config、記録がある。
- full-train cache から exp112 互換 `*_full_train_ml_features.csv.gz` を生成できる入口がある。
- raw test replay または提供済み rawtest cache から exp112 互換 `*_rawtest_ml_features.csv.gz` を生成できる入口がある。
- `*_feature_schema.csv` と exp112 schema の `*_schema_parity.csv` を保存する。
- summary に input SHA、model manifest SHA、model SHA、feature content SHA、decompressed SHA、schema parity pass/fail を記録する。
- gzip 生成物を比較する場合は raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
