# exp150_formation_physical_imputer_revisit 結果

## 状態

Kaggle train v1 完了。提出なし。

## 実装内容

Sunny 系の formation contact physical branch を hidden-safe に再構成した。

- train-fold wells の formation columns だけを teacher として使う。
- valid well では `X/Y/Z` と既知 prefix の `TVT_input` だけで surface と physical TVT 候補を作る。
- `contact_median`、`contact_prefix_weighted`、`contact_best_prefix` を評価する。
- `prefix_mae_best`、`formation_pred_spread`、`neighbor_dist` で confidence bucket を保存する。

## スコア

Kaggle full OOF:

| Candidate | Rows | Wells | RMSE | MAE | Bias | p95 abs | p99 abs | max well RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `well_plane_knn/contact_best_prefix` | 3,746,966 | 765 | 28.233897 | 18.281394 | -2.191246 | 58.205022 | 109.283497 | 136.526718 |
| `well_plane_knn/contact_prefix_weighted` | 3,746,966 | 765 | 28.300622 | 18.237811 | -2.131951 | 56.125894 | 110.678758 | 157.804503 |
| `well_plane_knn/contact_median` | 3,746,966 | 765 | 28.406306 | 18.262051 | -2.311171 | 56.212689 | 110.914867 | 160.888957 |
| `row_knn_xy/contact_best_prefix` | 3,746,966 | 765 | 28.736773 | 16.156194 | -2.073452 | 56.860609 | 131.336004 | 192.970636 |

best は `well_plane_knn/contact_best_prefix`。

Distance bucket for best:

| Bucket | Rows | RMSE | MAE | Bias |
| --- | ---: | ---: | ---: | ---: |
| `000_050` | 37,485 | 7.518282 | 5.063655 | -0.147885 |
| `050_100` | 38,250 | 8.003360 | 5.422620 | -0.123767 |
| `100_250` | 114,750 | 8.691162 | 6.097183 | -0.079748 |
| `250_500` | 191,158 | 10.032220 | 7.233987 | -0.360850 |
| `500_1000` | 381,912 | 12.690650 | 9.111396 | -0.669706 |
| `1000_plus` | 2,983,411 | 31.139509 | 20.962682 | -2.636696 |

Confidence diagnostics:

- `formation_pred_spread` は低い bucket ほど良い。`well_plane_knn/contact_best_prefix` で spread 最低 quartile RMSE 21.905116、高 spread quartile RMSE 36.390833。
- `neighbor_dist` も単調に悪化する。最低 quartile RMSE 18.912730、高 quartile RMSE 40.116758。
- `prefix_mae_best` は最高 quartile で RMSE 35.953199 まで悪化し、危険度 flag として使える。

Smoke:

| 条件 | best candidate | RMSE | MAE | max well RMSE |
| --- | --- | ---: | ---: | ---: |
| 8 wells local helper smoke | `well_plane_knn/contact_best_prefix` | 26.967070 | 18.333600 | 47.601074 |

Smoke は実装確認用であり、CV としては扱わない。

## 次

1. direct TVT candidate / inference port / submit は行わない。
2. `formation_pred_spread`、`neighbor_dist`、`prefix_mae_best` は confidence diagnostic として残す価値がある。
3. 後続に使う場合は exp092 系 add-only feature に限定し、direct replacement や target conditioning には戻さない。
