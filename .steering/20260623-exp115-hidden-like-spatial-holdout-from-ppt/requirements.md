# 要件

## 依頼

`KAGGLE_DIRECTION.md` のバックログ `hidden_like_spatial_holdout_from_ppt` を `exp115_hidden_like_spatial_holdout_from_ppt` として実装する。公式 PPT slide10 の Verification map から赤い Verification well の空間分布を抽出し、train wells の中から隠しテストに似た固定 holdout を作る。Kaggle Notebook 上で動くことを前提にし、難しい場合もローカル smoke で最低限の生成物を確認できるようにする。

## 制約

- Route: `ml_model`。ML route anchor (`exp092` / `exp073` / `exp098`) を後続で読むための評価面を作る。
- モデル学習、PF/Beam 再生成、inference port、submission は行わない。
- PPT/PNG は説明資料であり、CSV と完全一致する source of truth ではない。exact hidden split 復元として扱わず、頑健性監査用 split として記録する。
- 公式データは `data/raw/`、Kaggle では competition input を使う。
- 再現性: `docs/06_reproducibility.md` に従い、PPTX SHA、slide image SHA、red component count、生成物 path を summary に記録する。

## 受け入れ基準

- Kaggle train notebook で `hidden_like_spatial_holdout_from_ppt.py` が実行できる。
- `holdout_wells.csv` に `verification_like_spatial` と `verification_like_typewell_purged` の valid wells が保存される。
- `fold_assignments.csv` に全 train wells の `train` / `valid` / `purged_train_excluded` role が保存される。
- `well_metadata.csv` に centroid、azimuth、eval length、prefix length、GR coverage、typewell exact group、PPT red distance が保存される。
- `distribution_report.csv` に all train と holdout の spatial / azimuth / eval length / prefix / GR / TVT bin 分布が保存される。
- `summary.json` と `metrics.json` に PPT 抽出状態、PPTX SHA、slide image SHA、red component count、holdout well count が記録される。
- deterministic submission anchor ではないことを `SESSION_NOTES.md` / `result.md` に明記する。
