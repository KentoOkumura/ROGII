# exp220_row_neighbor_input_context_features_on_exp148 結果

## 結論

Kaggle CPU split train v1 を完了した。3 split の OOF 予測をローカルで streaming aggregate した `lgb_mean` は RMSE 8.496282588。

exp148 GPU `lgb_mean` 8.501281182 からは -0.004998594 改善したが、exp193 8.456665439、exp198 8.457923653、現行 ML submitted anchor の exp218 8.475793752 には届かない。したがって exp220 は train-side completed / no submit とし、inference 化や提出には進めない。

## 実装内容

- route: `ml_model`
- 親: `exp148_learned_likelihood_fulltrain_addonly_on_exp092`
- variant: `row_neighbor_input_context_addonly`
- 追加 feature group: `row_neighbor_input_context`
- train split: `train_lgb0` / `train_lgb1` / `train_lgb2`
- runtime: CPU LightGBM deterministic flags

`TVT_input`、OOF prediction、前行 model prediction、valid/test true TVT、oracle best、true-error rank は feature source に使わない。

## 実行結果

Kaggle kernels:

- `kentookumura/exp220-row-neighbor-exp148-lgb0` v1: COMPLETE
- `kentookumura/exp220-row-neighbor-exp148-lgb1` v1: COMPLETE
- `kentookumura/exp220-row-neighbor-exp148-lgb2` v1: COMPLETE

CV:

| model | pooled RMSE TVT | prediction SHA |
| --- | ---: | --- |
| `lgb0` | 8.577046760 | `f96b7932dda27fc60542aadbcee40906e89209ac2e6dfede2264e2410091f29f` |
| `lgb1` | 8.532166021 | `ea39574d56476ae2559b19f0178ffe44349a91de9c55f42a7566d92b87dcca8e` |
| `lgb2` | 8.539115349 | `aced87ccb87a001cbd2ee824ae2dab0b0b0e4b5f1345d724a68b21f485d87912` |
| `lgb_mean` | 8.496282588 | `5be47377f9ffcf0ddd3023c3e93e57764316588a9a6c94693ea5ee6666bc4e21` |

`lgb_mean` は split 3本の OOF `pred_target` を streaming で平均して算出した。行順チェックは `lgb0/lgb1/lgb2` すべて pass。巨大な結合済み prediction CSV は保存していない。

主な distance bucket:

| bucket | rows | RMSE TVT |
| --- | ---: | ---: |
| `000_050` | 38,650 | 0.986487 |
| `050_100` | 38,650 | 1.334240 |
| `100_250` | 115,950 | 2.097604 |
| `250_500` | 193,157 | 3.309812 |
| `500_1000` | 385,911 | 4.826862 |
| `1000_plus` | 3,011,671 | 9.316960 |

`rnic_` features の上位 importance は `likpf_mean_d` の lead/lag 差と `uproj_source_u_std` の lead/lag 差に集中した。ただし global CV の改善幅は小さく、現行 anchor 更新には弱い。

成果物:

- `artifacts/exp220_row_neighbor_input_context_features_on_exp148_split_lgb_mean_summary.json`
- `artifacts/exp220_row_neighbor_input_context_features_on_exp148_split_lgb_mean_metrics.csv`
- `artifacts/exp220_row_neighbor_input_context_features_on_exp148_split_lgb_mean_bucket_metrics.csv`
- `artifacts/exp220_row_neighbor_input_context_features_on_exp148_split_lgb_mean_by_well.csv`
- `artifacts/exp220_row_neighbor_input_context_features_on_exp148_split_lgb_mean_feature_importance_mean.csv`
