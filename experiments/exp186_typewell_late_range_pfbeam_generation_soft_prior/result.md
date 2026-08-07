# exp186_typewell_late_range_pfbeam_generation_soft_prior 結果

## 結論

v3 で、raw train well/typewell から exp072-style full replay train feature cache を作り直した。
既存 full replay cache は generation input として使っていない。

これは train feature cache 生成実験であり、LightGBM 学習、inference、submit、Public LB は実施していない。
ただし exp072 full replay cache との direct PF/Beam RMSE TVT 比較では、主力の `likpf_mean` が大きく悪化したため、exp072 replacement としては不採用。

## 実行

- kernel: `kentookumura/exp186-typewell-late-soft-pfbeam-train` v3
- URL: https://www.kaggle.com/code/kentookumura/exp186-typewell-late-soft-pfbeam-train
- status: `KernelWorkerStatus.COMPLETE`
- runtime: summary `15783.764` sec、feature generation `14053.477` sec
- GPU / internet: false / false
- selected soft prior: `pct50_strong2_pct70_weak0p5`

## 生成物

| 項目 | 値 |
| --- | --- |
| rows | 3,783,989 |
| wells | 773 |
| columns | 199 |
| feature_count | 196 |
| schema lines | 197 |
| gzip size | 2,093,362,668 bytes |
| decompressed bytes | 7,430,756,999 |
| data rows by decompressed line count | 3,783,989 |

生成物:

- `artifacts/exp186_typewell_late_range_pfbeam_generation_soft_prior_full_replay_cache_pixiux_likpf_late_soft_prior_public_replay_train_features.csv.gz`
- `artifacts/exp186_typewell_late_range_pfbeam_generation_soft_prior_full_replay_cache_feature_schema.csv`
- `artifacts/exp186_typewell_late_range_pfbeam_generation_soft_prior_full_replay_cache_summary.json`

## SHA

- train features raw gzip: `4bb7a43278ec65143d61c3451353735093995d5258aad665b901237a6a469185`
- train features decompressed: `b4dd75312d91b21f55b8d1ad09a8590c6bb75857ddfbbbc84d7db175dbb75d15`
- feature schema: `8c875703e3c009c74cc28430c4a8451f239f11fd4dcd3e6e55c705a5adfb7830`
- summary: `8e85db2e6d48b93b2a436b160a6041e478426e8bf7ef62406b6be6e31f215c5f`

## exp072 との PF/Beam RMSE TVT 比較

比較条件:

- exp072 と exp186 の full train replay cache を row-wise に比較。
- rows: 3,783,989。
- id alignment: first `000d7d20_1442`、last `ffefef30_6420`。
- 真値 TVT は `last_known_tvt + target`。
- `pf_ancc` / `pf_z` は absolute TVT prediction として採点。
- Beam / likelihood-PF の `_d` 系は `last_known_tvt` を足して absolute TVT に戻して採点。

| candidate | exp072 RMSE | exp186 RMSE | delta | exp072 MAE | exp186 MAE | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pf_ancc` | 14.493061 | 14.220030 | -0.273031 | 8.921569 | 8.851547 | -0.070022 |
| `pf_z` | 17.788174 | 17.679589 | -0.108585 | 10.677493 | 10.697447 | +0.019954 |
| `beam_mean` | 15.774328 | 15.753703 | -0.020624 | 10.898586 | 10.888194 | -0.010392 |
| `beam_cons` | 16.023008 | 16.025383 | +0.002374 | 11.106713 | 11.112429 | +0.005715 |
| `beam_sm5` | 16.313542 | 16.309361 | -0.004181 | 11.300928 | 11.300965 | +0.000037 |
| `beam_med` | 15.987519 | 15.988469 | +0.000950 | 11.060277 | 11.067241 | +0.006964 |
| `likpf_mean` | 11.594898 | 12.942278 | +1.347381 | 7.067633 | 7.805225 | +0.737592 |

`pf_ancc` は RMSE -0.273、`pf_z` は -0.109、`beam_mean` は -0.021 と小改善した。
一方で exp072 の最強候補である `likpf_mean` が RMSE +1.347、MAE +0.738 と大きく悪化した。

## 解釈

今回の実験は「typewell late-range soft prior を PF/Beam/likelihood-PF generation 内に入れた full replay cache を raw から再生成する」条件を満たした。

v1/v2 の 192-row prefix-holdout audit は、意図した full replay cache 改善実験ではなかったため superseded とする。v2 の結果は履歴として残すが、exp186 の正式結果ではない。

この cache 自体は LightGBM CV/LB を持たない。ただし direct PF/Beam candidate の train-side RMSE TVT では、`likpf_mean` の悪化が大きく、exp072 cache の単純 replacement としては支持しない。
soft prior は PF_ANCC と Beam mean に小幅改善を出したが、likelihood-PF 側の particle/path weighting を崩しており、生成 cache 全体を採用する根拠にはならない。

## 次

この exp186 cache を exp072 の代替 input として downstream 学習へ進めない。
再利用する場合は、`pf_ancc` / Beam mean の小改善だけを候補生成や selector feature の材料として切り出す方向に限定する。
direct replacement、hard invalidation、clip、submit は実施しない。
