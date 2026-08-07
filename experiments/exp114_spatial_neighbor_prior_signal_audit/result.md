# exp114_spatial_neighbor_prior_signal_audit 結果

## 仮説

X/Y、掘削方向、軌跡形状が似た train-fold wells から作る spatial neighbor TVT drift prior は、PF/Beam/likPF の base error と同じ方向に動く補助信号になり得る。

## 設定

- 親: `exp099_pf_multi_observation_likelihood_probe`
- 検証: well GroupKFold train-side spatial neighbor prior audit
- メトリック: RMSE / MAE / within10 / signal correlation / sign-match rate
- シード: 42

## 結果

| メトリック | 値 |
| --- | --- |
| CV / best RMSE | 11.151818387 |
| best candidate | `xy_plus_trajectory_shape_k8_likpf_mean_corr_a0p2_c40` |
| `likpf_mean` RMSE | 11.594897672 |
| delta vs `likpf_mean` RMSE | -0.443079285 |
| best MAE | 7.062013290 |
| best within10 | 0.779284506 |
| `likpf_mean` within10 | 0.772807479 |
| rows / wells | 3,783,989 / 773 |
| Public LB | - |
| Private LB | - |

## 主要結果

Kaggle train v1 は complete。best は `xy_plus_trajectory_shape_k8` の spatial prior を `likpf_mean` に `alpha=0.2`、clip 40 ft で弱く入れる候補だった。`likpf_mean` から RMSE -0.443079、within10 +0.006477 改善した。

| variant best | RMSE | MAE | within10 | 備考 |
| --- | ---: | ---: | ---: | --- |
| `xy_plus_trajectory_shape_k8_likpf_mean_corr_a0p2_c40` | 11.151818387 | 7.062013290 | 0.779284506 | best |
| `xy_only_k8_likpf_mean_corr_a0p2_c40` | 11.157375869 | 7.072735 | 0.779003 | X/Y control も強い |
| `xy_plus_direction_and_typewell_k8_likpf_mean_corr_a0p2_c40` | 11.200408 | 7.082558 | 0.778421 | same typewell 制約で少し弱い |
| `xy_plus_azimuth_k8_likpf_mean_corr_a0p2_c40` | 11.203437 | 7.106347 | 0.777353 | azimuth だけでは弱い |
| `likpf_mean` | 11.594897672 | 7.067632584 | 0.772807479 | baseline |

best candidate は全 distance bucket で `likpf_mean` より RMSE 改善した。特に `1000_plus` は 12.704015 -> 12.212863 で -0.491153。

| bucket | `likpf_mean` RMSE | best RMSE | delta |
| --- | ---: | ---: | ---: |
| 000_050 | 1.188878 | 0.988440 | -0.200437 |
| 050_100 | 1.925625 | 1.656305 | -0.269320 |
| 100_250 | 2.934160 | 2.672779 | -0.261381 |
| 250_500 | 4.701123 | 4.507915 | -0.193208 |
| 500_1000 | 6.676270 | 6.536488 | -0.139782 |
| 1000_plus | 12.704015 | 12.212863 | -0.491153 |

一方、well 単位では 416 wells 改善 / 357 wells 悪化 / 0 同値で、最大悪化は `f88ddb26` の +6.508121 RMSE。direct correction としては exp109/110 と同様に危険が残る。

signal metric では `xy_plus_trajectory_shape_k8` の prior-minus-base と true-minus-base の相関は `pf_ancc` で 0.399221、`likpf_mean` で 0.273578。符号一致率は `likpf_mean` で 0.582275。信号はあるが、hard postprocess ではなく feature / confidence 側へ回すべき強さ。

## 再現性

- deterministic anchor: false
- seed policy: deterministic_groupkfold_fixed_neighbor_rules_no_model_rng
- kernel version: `kentookumura/exp114-spatial-neighbor-prior-signal-audit-train` v1
- feature content SHA: OOF decompressed `9ffa9f9a026d43d3c0721a549fdff0aaf0acbd73d6c8209218ad9a45a314fe29`
- model SHA / manifest SHA: モデルなし
- prediction SHA: OOF raw gzip `7a328efd941b4acce476622d3e65e775c65bc9a385c600cdfed9efe3f0d75aa0`
- submission SHA: submission なし
- rerun result: 未実行

## 解釈

spatial / trajectory-shape neighbor prior は `likpf_mean` の誤差方向を説明する信号として有効。global RMSE は exp109 の native typewell prior best 11.143360 に近い 11.151818 まで改善し、distance bucket も全て改善した。

ただし、well-level regression は exp109/110 と同じ問題を持つ。最大悪化 +6.508121 は提出候補として許容できないため、このまま inference port / submit はしない。これは「prior は強いが補正すると悪化 well が出る」分岐なので、使う場合は exp092 系 confidence / gate の材料に限定する。ML に特徴量として入れる評価は別 backlog として扱う。

## 次

direct correction は閉じる。次は `spatial_neighbor_prior_confidence_gate_on_exp092` として、`prior_tvt`、`prior_delta`、`prior_std`、`neighbor_wells`、`distance_mean`、`azimuth_mismatch`、`prior_minus_likpf_mean`、clipped correction delta を使い、spatial prior を信用してよい row/well を判定する confidence / gate を評価する。ML に特徴量として入れる評価は `spatial_neighbor_prior_ml_features_on_exp092` として別 backlog に分ける。
