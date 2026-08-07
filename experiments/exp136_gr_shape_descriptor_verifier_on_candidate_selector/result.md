# exp136_gr_shape_descriptor_verifier_on_candidate_selector 結果

## 状態

Kaggle train v2 完了。提出なし。

## 仮説

GR shape descriptor は exp131 で within10 likelihood signal を示したが、direct top1 scorer としては崩壊した。今回は descriptor を候補 path の直接選択ではなく、exp101/102 系の低頻度 switch を承認・拒否する verifier として評価する。

## 設定

- Kernel: `kentookumura/exp136-gr-shape-verifier-train` v2
- output: `experiments/exp136_gr_shape_descriptor_verifier_on_candidate_selector/kaggle/output/train_v2`
- 親: `exp102_confidence_gated_likpf_fallback_on_exp101`
- score surface: exp101 saved booster + exp099 v2 cache
- rows / wells: 3,783,989 / 773
- runtime: 2,574.410 sec
- default: `likpf_mean`
- switch 候補: `pf_ancc`、`beam_mean`
- descriptor: `combo_descriptor_real`

v1 は descriptor 計算と exp101 score 復元後、4,000 超 variants の gate 評価中に `DeadKernelError` で失敗した。v2 は grid を 327 variants に縮小して完了した。

## 結果

best RMSE gate は `likpf_mean` から RMSE -0.009782 の小改善。ただし within10 はわずかに悪化した。

| variant | RMSE | MAE | within10 | switch rate |
| --- | ---: | ---: | ---: | ---: |
| `likpf_mean_single` | 11.594898 | 7.067633 | 0.772807 | 0.000000 |
| `exp101_error_ranker_rowwise` | 11.600097 | 7.006912 | 0.771452 | 0.452061 |
| `gate_descriptor_joint_margin_sr010_d035_std999999_df025_dm005_seg001` | 11.585115 | 7.057835 | 0.772744 | 0.010000 |
| `oracle` | 7.434030 | 3.745228 | 0.906525 | 0.614770 |

best gate の選択分布:

| candidate | rows | rate |
| --- | ---: | ---: |
| `likpf_mean` | 3,746,150 | 0.990000 |
| `pf_ancc` | 31,751 | 0.008391 |
| `beam_mean` | 6,088 | 0.001609 |

RMSE と within10 の両方を改善する variant もあったが、改善幅は小さい。代表例は `gate_descriptor_conservative_margin_sr005_d035_std000020_df035_dm000_seg001` で、RMSE 11.585504、within10 0.772849、`likpf_mean` から RMSE -0.009393、within10 +0.000041。

## Guardrail

best RMSE gate は 452 wells 改善、245 wells 悪化。最大 well regression は +3.542 RMSE、最大改善は -4.800 RMSE。

| bucket | delta RMSE vs `likpf_mean` | delta within10 |
| --- | ---: | ---: |
| distance `000_050` | -0.065584 | 0.000000 |
| distance `050_100` | -0.053551 | +0.000052 |
| distance `1000_plus` | -0.010945 | -0.000076 |
| tail rank `1000_plus` | -0.009783 | -0.000064 |
| distance `500_1000` | +0.000751 | -0.000029 |

near-row は壊していないが、worst-well regression が残り、within10 は best RMSE gate で微減した。

## 再現性

- exp099 train feature raw SHA: `4bd9df60f5c09f7a3029dac399afef73aa45b0158a7fd06a62a56f85fd0fde38`
- exp099 train feature decompressed SHA: `1939d536b1e56f7c0ea3847cc386ef769b0d33759d16e816c9ce180f0532df9a`
- exp099 schema SHA: `203e4f9a280fe901f5f21d39b85c3e0e2a7fe10c466081c15015c7fb014a0413`
- exp101 feature schema SHA: `ea2819375dd025448c3e294b56fd92179b8b261e22f5a7fb37fbf3e8ddfac9c6`
- exp101 model manifest SHA: `4f453761f1cc09042767baa934f8a1c5a89036bfb1c244a5f3fc5ab0cc843cc5`
- metrics SHA: `1ff417b9b15ddee8e92c02f27ddfa97a32d5d95d97e102051abdb22570aa174b`
- OOF predictions decompressed SHA: `ae45cfefe2c35fafd68529c3128b55b9bf36c17e1c90a9fa9a64f9c9b83bfdb0`
- best gate prediction SHA: `2bc210c26d445fe663a87ad90a92de03751c6555b9981dd37b34a3440940b952`
- descriptor well summary SHA: `3674dd83ffe66fbe1082db766105351dfb30f29ab6d36c19c02c889039a2e50c`

## 解釈

descriptor verifier は exp101 row-wise selector の崩壊を抑え、`likpf_mean` default から低 switch で小さく改善した。ただし改善幅は exp102 best gate より小さく、best RMSE gate は within10 をわずかに悪化させ、最大 well regression も残る。

したがって direct inference port / submit はしない。GR descriptor は hard selector ではなく、exp092 系 add-only confidence feature または他 confidence signal と併用する diagnostic 材料に限定する。

## 次

`gr_shape_descriptor_verifier_on_candidate_selector` は完了として backlog から外す。descriptor 系の次手は、既存の `gr_shape_descriptor_features_on_exp092` のように ML add-only confidence feature 側へ寄せる。単独 verifier / selector follow-up は優先しない。
