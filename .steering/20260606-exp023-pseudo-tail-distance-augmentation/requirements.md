# 要件

## 依頼

`KAGGLE_DIRECTION.md` のバックログ先頭 `pseudo_tail_distance_augmentation` を実験として実装する。

## 制約

- train / valid は well 単位の `GroupKFold` を維持する。
- 仮想 `TVT_input` cutoff は training fold の well にだけ作る。
- validation score は本来の `TVT_input.isna()` 行だけで計算する。
- train-only formation columns / typewell geology は使わない。
- 初回は train-side CV 実験に限定し、改善が確認できるまで inference / submit は作らない。

## 受け入れ基準

- `exp023_pseudo_tail_distance_augmentation` が作成され、notebook / config / settings / notes が exp023 名で揃っている。
- control、1 cutoff/well、3 cutoffs/well、distance-balanced sampling、pseudo-tail + distance-balanced sampling を同一 GroupKFold で比較できる。
- variant 別 overall RMSE、距離 bucket 別 RMSE、source row summary、feature importance、`metrics.json` が保存される。
- `validate-exp`、py_compile、ruff、Kaggle train package generation が通る。
