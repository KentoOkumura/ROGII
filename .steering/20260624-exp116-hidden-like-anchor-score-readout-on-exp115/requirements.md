# 要件

## 依頼

`KAGGLE_DIRECTION.md` のバックログ `hidden_like_anchor_score_readout_on_exp115` を `exp116_hidden_like_anchor_score_readout_on_exp115` として実装する。`exp115_hidden_like_spatial_holdout_from_ppt` が Kaggle 上で生成した hidden-like holdout split を正とし、既存 anchor の OOF / train-side prediction を再学習なしで採点する。

## 制約

- Route: `ml_model`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 新規 ML 学習は行わない。既存 prediction / by-well metrics と exp115 split の deterministic merge のみを行う。
- 正の split は `experiments/exp115_hidden_like_spatial_holdout_from_ppt/kaggle/output/train_v1/artifacts/` の `fold_assignments.csv` / `holdout_wells.csv` / `well_metadata.csv` とする。
- 主比較対象は `exp092`、`exp073`、`exp098`。入力 prediction が欠ける場合は失敗扱いではなく source inventory に不足を記録し、利用可能な粒度で集計する。
- exp115 holdout は exact hidden split ではないため、LB 代替や提出判断の唯一根拠として扱わない。

## 受け入れ基準

- train notebook / script から、source inventory、overall metrics、bucket metrics、by-well metrics、worst-well delta、summary JSON が生成される。
- `verification_like_spatial` と `verification_like_typewell_purged` の両方で採点できる。
- row-level prediction がある source では distance bucket、spatial bin、eval length、GR coverage、typewell group size を出す。
- by-well metrics しかない source では overall / by-well を出し、bucket 不可の理由を inventory に残す。
- gzip prediction の SHA は decompressed content SHA を記録する。
- 新規モデル、submission、LB 記録がないことを `SESSION_NOTES.md` / `result.md` に明記する。
