# exp138_ancc_surface_predictability_audit 結果

## 仮説

LightGBM を使わない従来型の KNN / local plane surface imputer で、
hidden-test-compatible な `ANCC_hat` と anchor-relative `ANCC_delta_hat` が作れるかを監査する。

## 設定

- 親: `backlog/KAGGLE_DIRECTION.md` の `ancc_surface_predictability_audit` backlog
- 検証: GroupKFold by well。valid fold の真 `ANCC` は評価専用。
- メトリック: `ANCC_hat` RMSE / MAE / bias、anchor-relative delta RMSE、by-well worst、distance bucket、target distribution summary
- シード: 42
- Kaggle kernel: `kentookumura/exp138-ancc-surface-predictability-audit-train` v3

## 結果

| Method | ANCC RMSE | ANCC MAE | delta RMSE | well RMSE p95 | worst well RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `global_median` | 653.771513 | 533.094332 | 107.844818 | 1331.510289 | 1686.684360 |
| `row_knn_xy` | 27.000672 | 17.352936 | 23.559085 | 55.552483 | 183.309143 |
| `well_plane_knn` | 24.388772 | 16.232597 | 25.417857 | 42.309986 | 161.329787 |

| Target summary, all rows | std | abs p99 | by-well mean std |
| --- | ---: | ---: | ---: |
| `control_dTVT` | 15.510458 | 51.580000 | 12.325146 |
| `row_knn_xy_anchor_relative` | 108.984070 | 270.900474 | 88.616222 |
| `well_plane_knn_anchor_relative` | 111.241958 | 267.593222 | 90.584559 |

Near-prefix delta は良い。`well_plane_knn` の `000_050` delta RMSE は 0.649006、`050_100` は 1.455425。  
一方、`1000_plus` longtail では `row_knn_xy` delta RMSE 25.812688、`well_plane_knn` 28.249549 まで重くなる。

## 再現性

- deterministic anchor: false
- seed policy: fixed global seed + fold offset for row sampling
- kernel version: v3
- feature content SHA:
  - OOF predictions: `fbbccb2f8d924ef01529dd48deb9200b1e443f93338417052ef1df735e48e60f`
  - method metrics: `a9047b74cac2585ad1d7fdb5862c54f68950b33a1ee8aef1ca88d0debba601ed`
  - bucket metrics: `c6521feea07b6b07c2867a45a1913208e9a8b5a3aa1394fa996ab529e160763b`
  - target distribution summary: `e558516d3946704d335aa35a09afe965ed48835b92e6f9ed709ecf19c7345054`
- model SHA / manifest SHA: persistent model なし
- submission SHA: submission なし
- rerun result: 未実施

## 解釈

`ANCC` surface の絶対推定は fold-safe KNN / local plane でかなり改善した。`well_plane_knn` は absolute RMSE が最良で、by-well worst も `row_knn_xy` より軽い。一方、後続 target 化の条件である anchor-relative residual 分布は `control_dTVT` より大幅に重い。全 rows の std は `control_dTVT` 15.51 に対し、`row_knn_xy_anchor_relative` 108.98、`well_plane_knn_anchor_relative` 111.24。

したがって、`TVT - ANCC_hat` または `TVT - T0 - (ANCC_hat - ANCC_anchor_hat)` への target conditioning は現時点では不採用。使うとしても `ancc_hat` / `ancc_delta_from_anchor` / confidence diagnostic の add-only feature 候補に留める。

## 次

1. `ancc_surface_predictability_audit` は完了として backlog から外す。
2. `ancc_hat_residual_target_ablation_on_exp073` は現設計では実施しない。
3. ANCC 系を残すなら、target ではなく `ANCC_hat` の信頼度、near-prefix stability、longtail error risk の feature 化に限定する。
