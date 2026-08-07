# exp089_pf_beam_disagreement_sample_weight 結果

## 状態

Kaggle train v1 完了。提出候補にはしない。

## 仮説

PF/Beam 直接置換は全体では弱いが、PF/Beam/likelihood-PF の disagreement は高誤差 row の信頼度として使える。exp089 では exp073 の target と base feature surface を固定し、予測置換ではなく confidence feature と sample weight だけを追加した。

## 設定

- Kernel: `kentookumura/exp089-pf-beam-disagreement-sample-weight-train`
- Version: 1
- Source cache: exp072 full replay train cache
- Rows: 3,783,989
- Wells: 773
- Base features: 196
- Mode: `gpu_repro_guard_dp_threads8`
- Models: 3 LightGBM configs x 5 folds x 4 variants

## 結果

| variant | lgb_mean RMSE | delta vs control | features | note |
| --- | ---: | ---: | ---: | --- |
| `sample_weight_unstable_downweight` | 9.521212047 | -0.005162526 | 196 | best, sample-weight only |
| `control_exp073_base196` | 9.526374573 | 0.000000000 | 196 | exp073 control reproduced |
| `confidence_features_plus_weight` | 9.562018858 | +0.035644285 | 199 | worsened |
| `confidence_features_core` | 9.564240270 | +0.037865697 | 199 | worsened |

sample-weight only は global RMSE では小改善したが、well-level guard は弱い。

| check | value |
| --- | ---: |
| improved wells | 374 |
| worsened wells | 399 |
| mean well delta | +0.010116 |
| median well delta | +0.010215 |
| max well worsen | +1.096752 |
| max well improve | -1.068382 |

distance bucket では near rows と 1000+ は改善したが、100-1000 の中距離 bucket は悪化した。

| bucket | control | sample-weight | delta |
| --- | ---: | ---: | ---: |
| 000_050 | 1.031219 | 0.995600 | -0.035619 |
| 050_100 | 1.366140 | 1.349185 | -0.016955 |
| 100_250 | 2.227290 | 2.242749 | +0.015459 |
| 250_500 | 3.698146 | 3.710147 | +0.012000 |
| 500_1000 | 5.438444 | 5.452646 | +0.014203 |
| 1000_plus | 10.446468 | 10.439277 | -0.007192 |

sample weight 分布:

- mean: 1.0
- std: 0.085432
- min: 0.769068
- p05: 0.860132
- p50: 0.999054
- p95: 1.144960
- max: 1.249405

## 再現性証拠

- source cache SHA: `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
- best lgb_mean prediction SHA: `7a25cbba2810a3f8e650d0274619d2ee5a86038750025dc55f80f8b9fc2944c1`
- OOF predictions decompressed SHA: `d6a94e064a24e594b822a4adeed9d5a5f3c631caff4f680385a2d7eec0d805b1`
- model count: 60
- output root: `/tmp/kaggle-output/exp089_pf_beam_disagreement_sample_weight/train_v1`
- copied small artifacts: `experiments/exp089_pf_beam_disagreement_sample_weight/artifacts/`

## 解釈

PF/Beam disagreement を feature としてそのまま足す方向は悪化したため不採用。sample-weight only は exp073 control から -0.005 RMSE の小改善だが、well 数では悪化が改善を上回り、中距離 bucket も悪化している。したがって submit candidate にはしない。

今後 PF/Beam signal を使うなら、単純な confidence feature 追加ではなく、`pf_candidate_ranker_or_nway_classifier` や observation likelihood 改善、または exp085 の U-projection correction+disagreement 完走に優先度を移す。
