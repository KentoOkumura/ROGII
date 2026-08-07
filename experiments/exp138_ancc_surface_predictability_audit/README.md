# exp138_ancc_surface_predictability_audit

## 状態

- ルート: ml_model
- 状態: completed
- CV: ANCC surface audit only
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-06-26
- 親実験: KAGGLE_DIRECTION ancc_surface_predictability_audit backlog

## 仮説

train-only の `ANCC` surface を、validation fold の真 `ANCC` を使わずに
`X,Y` と well-level 近傍から推定できるなら、後続の `ANCC_hat` residual target
または target-free confidence feature の前提になる。

## 変更点

- LightGBM は使わず、`global_median`、`row_knn_xy`、`well_plane_knn` を比較する。
- score rows の `ANCC_hat` と prefix anchor row の `ANCC_anchor_hat` を fold-safe に生成する。
- absolute error と anchor-relative delta error を別々に評価する。
- target ablation は行わず、target 分布 summary だけを出す。

## 検証方針

- Fold: project default の GroupKFold
- Group: well
- Stratification: なし
- Leakage Check: valid fold の真 `ANCC` は scoring のみに使い、surface fitting と anchor 推定には使わない。

## 実行入口

- 学習 notebook: `exp138_ancc_surface_predictability_audit_train.ipynb`
- 推論 notebook: `exp138_ancc_surface_predictability_audit_inference.ipynb`
- Kaggle 準備: `task prepare-kaggle-notebooks EXP=exp138_ancc_surface_predictability_audit`
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| best ANCC RMSE | `well_plane_knn` 24.388772 |
| best anchor-relative delta RMSE | `row_knn_xy` 23.559085 |
| control dTVT std | 15.510458 |
| best ANCC anchor-relative target std | `row_knn_xy_anchor_relative` 108.984070 |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- `well_plane_knn` は ANCC absolute RMSE 24.388772 まで下がり、train-only surface は fold-safe に一定程度推定できた。
- near-prefix の anchor-relative delta は安定しており、`well_plane_knn` の `000_050` delta RMSE は 0.649006。

### 悪かった点

- anchor-relative target 分布は `control_dTVT` より大幅に重い。全 rows std は control 15.510458 に対し、`row_knn_xy_anchor_relative` 108.984070。
- `1000_plus` longtail の delta RMSE は `row_knn_xy` 25.812688、`well_plane_knn` 28.249549 で、target conditioning には不安定。

### リスク / 注意

- row-level KNN は重いため、`max_rows_per_well` と `max_rows_total` で制限している。
- global OOF が良くても anchor-relative delta、near-prefix、worst-well が悪ければ target 化には進めない。

## 次

- `TVT - ANCC_hat` 系 target ablation には進まない。
- ANCC 系を残す場合は target-free confidence / diagnostic feature として再設計する。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
