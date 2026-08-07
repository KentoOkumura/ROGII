# exp104_pf_z_seedbag_scale_cache 結果

## 状態

Kaggle train v1 完了。提出なし。推論移植なし。

## 目的

exp100 `pf_z_xy_slope` を単発 PF 候補ではなく、exp072 `lik_pf` と同じ 128 seed likelihood-weighted seedbag として cache 化し、既存 PF/Beam 候補と比較した。

## 設定

- 親: `exp100_pf_z_unified_velocity_observation_prior`
- cache parent: `exp072_exp063_full_replay_feature_cache`
- Kernel: `kentookumura/exp104-pf-z-seedbag-scale-train` v1
- runtime: 41,132.91 sec
- rows: 3,783,989
- wells: 773
- seedbag: 128 seeds / 500 particles / scales 3, 5, 8, 12
- exp072 cache columns: `pf_z`, `likpf_mean_d`
- exp072 `likpf_scale_*` は cache に存在せず比較対象外

## 結果

| candidate | RMSE | MAE | within10 | bias |
| --- | ---: | ---: | ---: | ---: |
| `exp072_likpf_mean` | 11.594898 | 7.067633 | 0.772807 | -1.099423 |
| `pf_z_seedbag_scale_12` | 14.145856 | 8.776381 | 0.695260 | -0.953733 |
| `pf_z_seedbag_scale_8` | 14.171680 | 8.776667 | 0.694138 | -0.892882 |
| `pf_z_seedbag_scale_5` | 14.178127 | 8.768430 | 0.693555 | -0.819223 |
| `pf_z_seedbag_scale_3` | 14.215698 | 8.777034 | 0.692961 | -0.747068 |
| `pf_z_seedbag_mean` | 14.587060 | 9.664454 | 0.651736 | -1.047900 |
| `exp072_pf_z` | 17.788171 | 10.677487 | 0.647668 | -0.934560 |

Best seedbag は `pf_z_seedbag_scale_12`。exp072 plain `pf_z` より RMSE -3.642315 / within10 +0.047592 改善したが、exp072 `likpf_mean` より RMSE +2.550958 / within10 -0.077547 悪い。

距離 bucket では、`pf_z_seedbag_scale_12` は `1000_plus` で exp072 `pf_z` を改善したが、`md_since < 1000` の近・中距離 bucket は exp072 `pf_z` より悪化し、全 bucket で `exp072_likpf_mean` を下回った。

## 再現性

- deterministic anchor: false
- seed policy: `stable_sha256_seed_from_experiment_pf_z_seedbag_well`
- exp072 cache raw SHA: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
- exp072 cache decompressed SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- feature schema SHA: `700d38149f583c3ab6574ea7b163c3c8709c2514b675bea381d822f82f4809b8`
- candidate metrics SHA: `b3343b394a7acd356a547ba134e70cd958aacd38731c2f09f996f4a55efef19b`
- candidate wide raw SHA: `031e57c20e49faf681e194a2f8833f6d22a6780a49a5216161af9945ac8dafc7`
- candidate wide decompressed SHA: `3d27bb3c580f5c2df3542c9e6dcccb981c20b37f81a2a6a10cda17153e134fd2`
- candidate long raw SHA: `43a45040cdbee7e19fd7ee2f04f732b0e62372d14dfd8cb6fefbc3af2505d27d`
- candidate long decompressed SHA: `17e8a58595a4fc2fce62d7a30634cf3da48e41a64d7f11e93f10f131ae2851f6`
- summary SHA: `f5339a5fdb5855b3b15ad4e349f3b98695f384d762dffb8ec60483803fdb7fb3`

## 判定

`pf_z_seedbag_*` は plain `pf_z` の seed 1 本由来の弱さをかなり改善したが、既存の `likpf_mean` を置き換える根拠はない。直接推論移植、提出、exp073/exp092 系への add-only feature 化はしない。

PF 側を続ける場合は、`pf_z` seedbag ではなく、既存候補生成 / target-free likelihood の改善に戻す。
