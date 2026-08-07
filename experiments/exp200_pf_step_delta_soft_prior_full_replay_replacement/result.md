# exp200_pf_step_delta_soft_prior_full_replay_replacement 結果

## 結論

完了。不採用。Kaggle train v1 は成功し、raw train horizontal/typewell から exp072-style full replay train feature cache を再生成できたが、exp072 direct 比較で主候補 `likpf_mean` が許容悪化 +0.02 RMSE を超えたため、direct replacement、inference port、submit には進めない。

- train kernel: `kentookumura/exp200-pf-step-delta-prior-train` v1
- comparison kernel: `kentookumura/exp200-pf-step-delta-prior-comparison` v5
- rows / wells / features: 3,783,989 / 773 / 196
- `id` mismatches vs exp072: 0
- selected prior: `delta_free010_cost0025_scale003`
- 判定: `completed_train_feature_cache_direct_pfbeam_rejected_no_submit`

## train 生成

Kaggle train summary は `train_feature_cache_completed`。出力 full gzip は Kaggle 上に存在し、comparison notebook の input として使用できた。ローカルへの full gzip download は途中で connection reset したため、local artifact には小さい summary/schema と comparison artifacts のみを保存した。

- elapsed seconds: 14,719.448
- feature elapsed seconds: 13,059.303
- raw gzip SHA256: `f2bc3026bcb1491716fcf8845a158badff8f229a7b4a124659cbbc7bde032233`
- likelihood-PF ESS mean: 372.644430
- likelihood-PF resampling rate mean: 0.021144

## exp072 direct 比較

| candidate | exp072 RMSE | exp200 RMSE | delta RMSE | delta MAE | delta within10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pf_ancc` | 14.493061 | 14.736794 | +0.243733 | +0.284850 | -0.009949 |
| `pf_z` | 17.788174 | 17.662427 | -0.125747 | +0.031351 | -0.000766 |
| `beam_mean` | 15.774328 | 15.774328 | +0.000000 | +0.000000 | +0.000000 |
| `likpf_mean` | 11.594898 | 11.618341 | +0.023444 | -0.083430 | +0.002485 |

`likpf_mean` は MAE と within10 は改善したが、RMSE が +0.023444 悪化した。事前 guard は最大 +0.02 なので、+0.003444 超過で fail。

## bucket 所見

`likpf_mean` は short/mid bucket では強く改善した。

| bucket | delta RMSE |
| --- | ---: |
| `000_050` | -0.848621 |
| `050_100` | -0.982801 |
| `100_250` | -0.694795 |
| `250_500` | -0.643344 |
| `500_1000` | -0.378803 |
| `1000_plus` | +0.073129 |

悪化は主に rows の多い `1000_plus` と worst wells が支配している。by-well では `likpf_mean` が 426 wells 改善、347 wells 悪化、最大悪化は well `70925e23` の +24.605219 RMSE。

## step-delta rate

| candidate | mean abs step delta | p95 | p99 | rate >0.10 |
| --- | ---: | ---: | ---: | ---: |
| `pf_ancc` | 0.040355 | 0.144000 | 0.405000 | 0.081180 |
| `pf_z` | 0.034975 | 0.121000 | 0.310000 | 0.067134 |
| `likpf_mean` | 0.028440 | 0.093750 | 0.194336 | 0.043508 |

step-delta soft prior は近距離の不自然な jump を抑える方向には効いたが、longtail/worst-well の RMSE を守るには不十分だった。

## 次の扱い

`pf_step_delta_soft_prior_full_replay_replacement` backlog は完了/不採用として閉じる。今の設定を full replay direct replacement として広げる価値は低い。使う場合も、生成 cache replacement ではなく short-distance confidence diagnostics の材料に限定する。
