# 要件

## 依頼

`ancc_surface_predictability_audit` を LightGBM なしの従来前提で実装する。
train-only の `ANCC` formation surface を hidden-test-compatible に推定できるかを、
validation fold の真 `ANCC` を使わずに監査する。

## 制約

- Route: `ml_model`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- LightGBM は使わない。
- 提出用 `submission.csv` は作らない。
- valid fold の真 `ANCC` は surface fitting、特徴量生成、normalization、anchor 推定に使わない。
- `ANCC`、`ASTNU`、`ASTNL`、`EGFDU`、`EGFDL`、`BUDA` は hidden test に存在しない前提で扱う。
- Kaggle Notebook 実行を正とし、ローカル notebook 実行は行わない。

## 受け入れ基準

- `global_median`、`row_knn_xy`、`well_plane_knn` の fold-safe OOF `ANCC_hat` が生成できる。
- method-level metrics に `RMSE`、`MAE`、`bias`、`p95/p99 abs error`、by-well worst、anchor-relative delta error が含まれる。
- distance bucket / near-prefix / longtail bucket の metrics が生成される。
- control `TVT - last_known_TVT`、absolute `TVT - ANCC_hat`、anchor-relative residual の分布 summary が生成される。
- 生成物の SHA256 が `metrics.json` に記録される。
- この実験は deterministic anchor や submit candidate として扱わないことが明記されている。
