# exp149_normalized_shape_addonly_features_on_exp092 結果

## 仮説

exp092 の U-projection surface に、well-local normalized U/MD shape を add-only で入れると、target 変更や PF/Beam hard selector なしに候補 path の相対形状を補助特徴として使える可能性がある。

## 実装

- 親: `exp092_u_projection_correction_disagreement_fullrun`
- cache 親: `exp072_exp063_full_replay_feature_cache`
- route: `ml_model`
- Kaggle kernel: `kentookumura/exp149-nshape-addonly-exp092-train` v1
- output: `experiments/exp149_normalized_shape_addonly_features_on_exp092/kaggle/output/train_v1`
- 実行: `normalized_shape_addonly` 1 variant、3 LightGBM configs、5 folds、15 boosters
- 追加特徴量: 94 normalized shape features

## 結果

Kaggle train v1 は完了。exp092 control は再学習せず、保存済み exp092 metrics と比較した。

| model | RMSE | exp092 同 model 差分 | exp139 同 model 差分 |
| --- | ---: | ---: | ---: |
| `lgb0` | 9.558981894 | +0.025855456 | - |
| `lgb1` | 9.315846067 | -0.006633828 | -0.009061573 |
| `lgb2` | 9.327179161 | -0.011013243 | -0.010399150 |
| `lgb_mean` | 9.341688371 | -0.001375695 | -0.028895854 |

`lgb1` と `lgb2` は exp092 の同 model を小幅に改善した。`lgb_mean` も exp092 `lgb_mean` から -0.001376 改善した。一方で `lgb0` は悪化した。

## Bucket

`lgb1` の distance bucket は次の通り。

| bucket | rows | RMSE |
| --- | ---: | ---: |
| `000_050` | 38,650 | 1.241339 |
| `050_100` | 38,650 | 1.536411 |
| `100_250` | 115,950 | 2.238764 |
| `250_500` | 193,157 | 3.485794 |
| `500_1000` | 385,911 | 5.166094 |
| `1000_plus` | 3,011,671 | 10.227322 |

worst wells は `86454a6f`、`fb03ae90`、`1b1eba53` などで、exp092 由来の hard well warning は残る。

## Feature Importance

normalized shape features は feature importance 上位に入った。上位は `nshp_likpf_mean_poly_curvature_norm`、`nshp_pf_z_poly_curvature_norm`、`nshp_pf_ancc_poly_curvature_norm`、各 candidate の polynomial slope 系で、normalized U-state shape には LightGBM が使う信号がある。

## SHA

- predictions decompressed SHA256: `af7191254d1b618aceb0fc9d43bf4061b22f240ad55429cb00d98d0cd00fe561`
- feature schema SHA256: `720c7b54038c24ba257ea337161633e1839e53e736d3e7e0c9f86f938d4be573`
- model manifest SHA256: `638dfe06ea40fa568b00a6fd522a075b667a03b5276e4b4b2d8dde6fd1500c5e`

## 判断

train-side OOF では支持。`normalized_shape_addonly` は exp092 に対して小幅だが一貫した改善を示し、exp139 small rank-slot merge も上回った。

ただし、exp092 は by-well warning が残る ML route submitted anchor であり、本実験は raw-test feature parity と exp115 hidden-like stress をまだ確認していない。現時点では direct inference port / submit はしない。

次は exp115 hidden-like stress readout と raw-test feature parity を確認し、near-row / worst-well regression が崩れない場合だけ inference port を検討する。
