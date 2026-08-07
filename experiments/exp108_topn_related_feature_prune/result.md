# exp108_topn_related_feature_prune 結果

## 状態

2026-06-22 に Kaggle train v1 を完了。結論は rejected。OOF が exp098 full rank-slot と exp105 compact rank-slot の両方より悪いため、inference / submit は行わない。

## 実装内容

exp098 の full 260 feature surface を再構成し、静的 column set で pruning する。

- Active variant は `top3_related_pruned_260` のみ。
- `exp098_full_260`、`top1_related_pruned_260`、`top2_related_pruned_260`、`non_candidate_context_plus_topn_related` は config に残すが、GPU 節約のため disabled。
- top3 固定は exp098 の rank-slot distribution と feature importance から決めた。

## 評価

- Kaggle kernel: `kentookumura/exp108-topn-related-feature-prune-train`
- Kernel version: 1
- Runtime: 8775.76 sec
- Rows / wells: 3,783,989 / 773
- Active variant: `top3_related_pruned_260`
- Features: 195
  - `base_196_non_candidate_context`: 152
  - `base_196_topn_core_candidate_family`: 7
  - `rank_slot_top3_related`: 36

### Pooled OOF

| model | RMSE |
| --- | ---: |
| `lgb2` | 9.479370656 |
| `lgb1` | 9.491034034 |
| `lgb_mean` | 9.529005954 |
| `lgb0` | 9.798771537 |

### 比較

| 比較対象 | delta |
| --- | ---: |
| vs exp073 raw anchor 9.526374749 | -0.047004094 |
| vs exp077 policy 9.470514801 | +0.008855855 |
| vs exp098 best 9.358151052 | +0.121219603 |
| vs exp098 lgb_mean 9.427447987 | +0.101557966 |
| vs exp105 best 9.441103161 | +0.038267495 |
| vs exp092 best 9.322479896 | +0.156890760 |

best single は exp073 raw anchor よりは良いが、rank-slot 系の比較基準である exp098 full 260 に大きく負ける。top3 関連 static prune は不採用。

### Feature importance 上位

| feature | importance |
| --- | ---: |
| `slp_b_d_50` | 3857.666667 |
| `rank1_u_curvature` | 3825.666667 |
| `rank2_u_curvature` | 3805.533333 |
| `form_mean_d` | 3764.600000 |
| `rank3_u_curvature` | 3639.200000 |
| `spatial_knn_dist` | 3594.266667 |
| `frac` | 3573.466667 |
| `dx` | 3329.133333 |
| `rank2_u_slope` | 3320.466667 |
| `rank1_u_slope` | 3295.333333 |
| `rank3_u_slope` | 3263.866667 |
| `dense_std` | 3256.000000 |
| `dz` | 3244.600000 |
| `dense_dist` | 3200.600000 |
| `tvt_dense50_d` | 3172.066667 |

rank-slot U-shape は引き続き重要だが、full surface から broad context を削ると弱くなる。

### Worst wells (`lgb_mean`)

| well_id | RMSE | error_mean |
| --- | ---: | ---: |
| `86454a6f` | 54.305954 | -49.071064 |
| `fb03ae90` | 44.178391 | 42.328465 |
| `389ae58f` | 42.150928 | -39.926804 |
| `1b1eba53` | 41.643181 | -37.741670 |
| `91b301ce` | 35.722672 | 24.867075 |

### Bucket (`lgb_mean`)

| bucket | RMSE |
| --- | ---: |
| `000_050` | 1.073991 |
| `050_100` | 1.435376 |
| `100_250` | 2.350787 |
| `250_500` | 3.772918 |
| `500_1000` | 5.517871 |
| `1000_plus` | 10.441215 |

### SHA

- model manifest: `e773850dd0f9eac74a416d1936cdf9a9b2a511dda094083952919e2da253b88b`
- predictions gzip raw: `b16061aab916298e141a0482d2cfcee8cb06d5de1f6ab91592a5b9978cbc302b`
- predictions decompressed: `993fbf4e48a4e612e8b3a26d3d26fccc2d29d412e911a13098eeded4a11844ef`
- feature schema: `1633b2628df46a92a23597f6807992f64b3fe6bf5dd42baa33bbb80ba54821e7`
- source cache: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
- feature schema source: `700d38149f583c3ab6574ea7b163c3c8709c2514b675bea381d822f82f4809b8`
- summary source: `133f9be7a6bcf8606e18b7d41f4d24d84e1d8e0f128660717b21fea4fad46b7f`

## 判断

exp098 の 260 features から top3 selector 関連列だけを残す方針は、noise 削減より signal 欠落の影響が大きかった。exp098 full rank-slot を比較基準として維持し、今後は prune より exp092 への小さな add-only merge または candidate-generation / likelihood 側を優先する。
